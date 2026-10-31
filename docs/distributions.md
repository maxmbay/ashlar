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
| `normal` | `Mixture` of `Normal`s; `policy="mean_parameters"` gives one `Normal` | closed form, or bisection for the mixture |
| `classification`, `ordinal`, tree | `Categorical` over `classes_` (the leaves, for a tree); values are class indices | none |

Two wrappers sit around the head. Every ensemble member is a component of an equal-weight `Mixture`, so a TabM model with 32 members is a mixture of 32 of the objects above. A head trained on a standardized target is wrapped in an `AffineTransformed` that puts it back in the units of `y`. Unwrap them when you need the members: `dist.base.components` is the per-member distribution with batch shape `(rows, members)`.

A bisection quantile is exact to float precision: it brackets the level with Cantelli's bound around the mean, halves the bracket 80 times, and then takes one Newton step that does not move the value but gives the implicit-function gradient. Before this object existed, the quantiles of every head but `quantile` were read from 1,000 draws; they are now exact, so they no longer depend on `n_samples` or `seed`, and calibration reads each held-out row's CDF instead of ranking it among draws.

The flow's mean and variance are integrals: closed form on its identity tails and a 512-point rule across the spline. A count head's PIT is spread uniformly over the jump of its CDF at the observed count, so it is still uniform for a calibrated model.

## JAX transforms

A distribution is an Equinox module, so it is a JAX pytree. It goes into `jax.jit` and `jax.vmap` as an argument, and `jax.grad` differentiates through `log_prob`, `cdf`, `quantile`, and, where the class sets `reparameterized = True`, through `sample`. The reparameterized ones are `Normal`, `StudentT`, `Flow`, and `LowRankMultivariateNormal`, and an `AffineTransformed` of any of them. A mixture, a histogram, a count, and a category draw a discrete choice, so their draws carry no gradient.

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
