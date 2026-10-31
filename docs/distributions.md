# Predictive distributions

Every head in [Ashlar](../README.md) answers with the same kind of object. `model.predict_dist(X)` returns an `ashlar.distributions.Distribution` for the rows of `X`, in the units of `y`, whatever the head:

```python
from ashlar import ASRRegressor

model = ASRRegressor(task="mixture").fit(X, y)
dist = model.predict_dist(X_new)

dist.batch_shape, dist.event_shape   # (rows,) and (): one number per row
dist.log_prob(y_new)                 # (rows,)
dist.sample(key, (1000,))            # (1000, rows)
dist.mean(), dist.variance()         # (rows,)
dist.cdf(threshold)                  # P(y <= threshold | x), (rows,)
dist.quantile(0.9)                   # (rows,)
dist[:10]                            # the first ten rows, still a distribution
```

`model.sample`, `model.log_prob`, `model.predict_quantiles`, `model.predict_interval`, and `model.calibrate` all read this object, so they agree with it and with each other.

## Shapes

`batch_shape` indexes rows, and `event_shape` is the shape of one outcome: `()` for one number, `(d,)` for a joint outcome over `d` columns. A value passed to `log_prob` or `cdf` broadcasts against `batch_shape + event_shape`, and `sample(key, sample_shape)` returns `sample_shape + batch_shape + event_shape`. A quantile level broadcasts against `batch_shape`, so levels in a column give one row of quantiles per level:

```python
import jax.numpy as jnp

dist.quantile(jnp.array([0.1, 0.5, 0.9])[:, None])   # (3, rows)
```

## What each head returns

| Head | Object | Quantile |
|---|---|---|
| `gaussian` | `Normal` | closed form |
| `mixture` | `Mixture` of five `Normal`s | bisection on the CDF |
| `flow` | `Flow`, the spline | closed form: the spline at the normal quantile |
| `histogram` | `Histogram` | closed form |
| `student_t` | `StudentT` | bisection on the CDF |
| `poisson`, `negative_binomial` | `Poisson`, `NegativeBinomial`, with the structural zero | the smallest whole number whose CDF reaches the level |
| `quantile` | `QuantileLevels` | only its three trained levels; no density, CDF, or draws |
| `hurdle` | `Hurdle`: a probability of exactly zero and a `LogNormal` or `Gamma` amount | 0 up to the zero share, the severity's quantile above it |
| `multivariate_gaussian` | `LowRankMultivariateNormal` over the target columns | per column, through `marginals()` |
| `multivariate_flow` | `AutoregressiveFlow` over the target columns | per column, from draws (`marginals()` is an `Empirical`) |
| `normal` | `Mixture` of `Normal`s; `policy="mean_parameters"` gives one `Normal` | closed form, or bisection for the mixture |
| `classification`, `ordinal`, tree | `Categorical` over `classes_` (the leaves, for a tree); values are class indices | none |

Two wrappers sit around the head. Every ensemble member is a component of an equal-weight `Mixture`, so a TabM model with 32 members is a mixture of 32 of the objects above. A head trained on a standardized target is wrapped in an `AffineTransformed` that puts it back in the units of `y`. Unwrap them when you need the members: `dist.base.components` is the per-member distribution with batch shape `(rows, members)`.

A bisection quantile is exact to float precision: it brackets the level with Cantelli's bound around the mean, halves the bracket 80 times, and then takes one Newton step that does not move the value but gives the implicit-function gradient. Before this object existed, the quantiles of every head but `quantile` were read from 1,000 draws; they are now exact, so they no longer depend on `n_samples` or `seed`, and calibration reads each held-out row's CDF instead of ranking it among draws.

The flow's mean and variance are integrals: closed form on its identity tails and a 512-point rule across the spline. Where the CDF jumps, at a count or at the hurdle's zero, the PIT is spread uniformly over the jump (`cdf_left(y)` to `cdf(y)`), so it is still uniform for a calibrated model.

## Several targets at once

`task="multivariate_gaussian"` takes `y` with one column per outcome, demand at eight sites or the duration and the cost of a job, and predicts their joint normal. Its covariance is `W Wᵀ + diag(D)`, with a factor `W` of rank 2 by default (`head_kwargs={"rank": r}`), and both change from row to row, so the model learns when outcomes move together as well as how much each one varies.

```python
import jax
from ashlar import ASRRegressor
from ashlar.distributions import Independent

model = ASRRegressor(task="multivariate_gaussian").fit(X, Y)   # Y is (rows, sites)
dist = model.predict_dist(X_new)          # batch (rows,), event (sites,)
dist.covariance()                         # (rows, sites, sites), in the units of Y
draws = dist.sample(jax.random.key(0), (4000,))         # (4000, rows, sites): coherent scenarios
p_over = (draws.sum(axis=-1) > capacity).mean(axis=0)   # P(total > capacity) per row

dist.marginals()                          # each site on its own: batch (rows, sites)
Independent(dist.marginals())             # the same sites with the dependence dropped
```

`model.predict` is the mean of every column, `model.predict_quantiles` gives each column's marginal quantiles as `(rows, columns, levels)`, and `model.log_prob(X, Y)` is the joint log density. `rank=0` trains independent columns. A question about several columns at once, such as a total, a difference, or whether two things both happen, needs the joint draws, because the marginals alone do not determine it. [The joint demand benchmark](benchmarks.md#joint-demand-across-locations) shows how much that matters on real loads.

## Dependence that is not a correlation

A joint normal can only say how strongly columns move together on a straight line. `task="multivariate_flow"` takes the same `y` and predicts a conditional masked autoregressive flow: `p(y | x) = p(y₁ | x) p(y₂ | y₁, x) ⋯`, where each column's conditional is a standard normal pushed through a monotone spline, shifted and scaled. A masked network reads the columns before each one and the row's context from the trunk, and sets that column's knots, shift, and scale. So the whole shape of a column, not only its mean, can change with the columns before it: one load that spikes when another is high, a cost that grows more uncertain with the duration, a dependence that is strong in the tails and absent in the middle.

```python
model = ASRRegressor(task="multivariate_flow").fit(X, Y)
dist = model.predict_dist(X_new)          # batch (rows,), event (columns,)
model.log_prob(X_new, Y_new)              # exact joint log density
draws = dist.sample(jax.random.key(0), (4000,))
```

`head_kwargs={"hidden": 64, "depth": 1}` are the defaults: the masked network's width and the number of autoregressive blocks, the column order reversed in every other block. One block already gives every column a flexible conditional; a second lets the first column depend on the later ones too, so the column order matters less. The column order is the order of `Y`'s columns, so put the ones that drive the others first when you know them.

The density is exact and costs one network pass. A draw takes one pass per column, so it is slower than the normal's. The mean, variance, and covariance have no closed form; `predict_dist(X).mean()` estimates them from 1,024 fixed draws, and `predict_quantiles` reads each column's quantiles from `n_samples` joint draws, so they depend on `n_samples` and `seed` here, unlike the other heads. On a curved dependence with no correlation at all, the second column the square of the first's noise, it beats `multivariate_gaussian` by 2.1 nats per held-out row in joint log density (−4.54 against −6.68, mean of three seeds with 2,500 training rows each; the data is in `tests/test_multivariate_flow.py`); on the electricity loads of [the joint demand benchmark](benchmarks.md#joint-demand-across-locations) the dependence is close to linear, and the gain is smaller.

## An exact zero and a positive amount

`task="hurdle"` is for a target that is often exactly zero and otherwise positive: a tip, an insurance claim, a day's rain, an order. One output is the probability of zero; the other two describe the amount when it is not zero, a `lognormal` by default or a `gamma` with `head_kwargs={"severity": "gamma"}`. The target stays in its own units.

```python
model = ASRRegressor(task="hurdle").fit(X, tips)
dist = model.predict_dist(X_new)
dist.cdf(0.0)             # P(tip = 0)
model.predict(X_new)      # the mean, zeros included
```

Its `log_prob` is a log probability at zero and a log density elsewhere, the usual likelihood of a mixed outcome, so compare it only with another model of the same kind; CRPS compares it with anything. Quantiles at or below the share of zeros are exactly 0, and an 80% interval that reaches down to 0 contains every zero, so it can hold more than 80% of rows even after calibration.

## JAX transforms

A distribution is an Equinox module, so it is a JAX pytree. It goes into `jax.jit` and `jax.vmap` as an argument, and `jax.grad` differentiates through `log_prob`, `cdf`, `quantile`, and, where the class sets `reparameterized = True`, through `sample`. The reparameterized ones are `Normal`, `StudentT`, `Flow`, `LogNormal`, `Gamma`, `LowRankMultivariateNormal`, and `AutoregressiveFlow`, and an `AffineTransformed` of any of them. A mixture, a histogram, a count, and a category draw a discrete choice, so their draws carry no gradient.

```python
import jax

@jax.jit
def exceedance(dist, threshold):
    return 1.0 - dist.cdf(threshold)

exceedance(model.predict_dist(X_new), 50.0)
```

## Your own head

A `task="custom"` model's `dist_fn(raw)` should return one of these objects, built from the head's raw output, so `sample`, `log_prob`, and the quantile methods work on it too. `ashlar.demos.linked` returns the `Mixture` of its members' normals this way.

## Forecasts

`ASRForecaster.predict_dist(history)` and `ASRHistoryForecaster.predict_dist(history)` return the same objects with batch shape `(windows, horizon)`, or `(horizon,)` for one history. Each step is its own marginal: the forecasters model every step of the horizon separately, and the object makes no claim about how the steps move together.
