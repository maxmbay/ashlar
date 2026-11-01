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

A univariate distribution also gives the probabilities a coarse label needs, in log space so they stay finite far into the tails:

```python
dist.log_cdf(t), dist.log_survival(t)            # log P(y <= t), log P(y > t)
dist.log_cdf_left(t), dist.log_survival_left(t)  # log P(y < t), log P(y >= t); differ only at an atom
dist.log_interval(a, b)                          # log P(a <= y <= b), closed at both ends
dist.cdf_interval(a, b)                          # (P(y < a), P(y <= b)), where a coarse row's PIT lies
dist.convolve_normal(s)                          # y plus normal noise of sd s: normal, mixture, affine
```

They are what [`ashlar.observed`](observed.md) trains and scores censored, interval, truncated, and noisy labels with. A joint distribution gives the same for a row whose columns are each exact or in an interval:

```python
dist.log_box(lower, upper)       # log p(exact columns) + log P(the other columns in their intervals | them)
```

A column with `lower == upper` is exact. The joint normal and the mixture are closed form when a row has at most one coarse column, and the flow when its only coarse column is the last; otherwise `log_box` integrates at `box_points(256, columns)`, quasi-random points, with the GHK simulator. [Joint labels](observed.md#several-products-that-sell-out-together) has the detail.

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
| `flow` | `Flow`: the spline, then each row's shift and scale | closed form: the map at the normal quantile |
| `histogram` | `Histogram` | closed form |
| `student_t` | `StudentT` | bisection on the CDF |
| `poisson`, `negative_binomial` | `Poisson`, `NegativeBinomial`, with the structural zero | the smallest whole number whose CDF reaches the level |
| `quantile` | `QuantileLevels` | only its three trained levels; no density, CDF, or draws |
| `hurdle` | `Hurdle`: a probability of exactly zero and a `LogNormal` or `Gamma` amount | 0 up to the zero share, the severity's quantile above it |
| `multivariate_gaussian` | `LowRankMultivariateNormal` over the target columns | per column, through `marginals()` |
| `multivariate_mixture` | `Mixture` of five `LowRankMultivariateNormal`s over the target columns | per column, bisection on each marginal mixture's CDF |
| `multivariate_flow` | `AutoregressiveFlow` over the target columns | per column, from draws (`marginals()` is an `Empirical`) |
| `normal` | `Mixture` of `Normal`s; `policy="mean_parameters"` gives one `Normal` | closed form, or bisection for the mixture |
| `classification`, `ordinal`, tree | `Categorical` over `classes_` (the leaves, for a tree); values are class indices | none |

Two wrappers sit around the head. Every ensemble member is a component of an equal-weight `Mixture`, so a TabM model with 32 members is a mixture of 32 of the objects above. A head trained on a standardized target is wrapped in an `AffineTransformed` that puts it back in the units of `y`. Unwrap them when you need the members: `dist.base.components` is the per-member distribution with batch shape `(rows, members)`.

A bisection quantile is exact to float precision: it brackets the level with Cantelli's bound around the mean, halves the bracket 80 times, and then takes one Newton step that does not move the value but gives the implicit-function gradient. Before this object existed, the quantiles of every head but `quantile` were read from 1,000 draws; they are now exact, so they no longer depend on `n_samples` or `seed`, and calibration reads each held-out row's CDF instead of ranking it among draws.

The flow's mean and variance are integrals: closed form on the spline's linear tails and a 512-point rule across it, then moved by the row's shift and scale. Where the CDF jumps, at a count or at the hurdle's zero, the PIT is spread uniformly over the jump (`cdf_left(y)` to `cdf(y)`), so it is still uniform for a calibrated model.

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

`model.predict` is the mean of every column, `model.predict_quantiles` gives each column's marginal quantiles as `(rows, columns, levels)`, and `model.log_prob(X, Y)` is the joint log density. `rank=0` trains independent columns. A question about several columns at once, such as a total, a difference, or whether two things both happen, needs the joint distribution, because the marginals alone do not determine it. A weighted sum of the columns is exact under the joint normal and the joint mixture: `dist.project(weights)` is its distribution, a normal or a mixture of normals with batch `(rows,)`, so `dist.project(np.ones(sites)).cdf(capacity)` is P(total ≤ capacity) with no draws. Anything else is read from the joint draws. [The joint demand benchmark](benchmarks.md#joint-demand-across-locations) shows how much that matters on real loads.

### Calibrating the columns and the total

`calibrate` on a joint model calibrates each column on its own, the sum of the columns, and any other function of them you name, on rows the model was not fitted on:

```python
def peak(y):                     # a function of the last axis: (..., sites) -> (...)
    return y.max(axis=-1)

model.calibrate(X_holdout, Y_holdout, target=peak)        # each site, the sum, and peak
model.predict_interval(X_new, 0.8)                         # each site: (rows, sites, 2)
model.predict_interval(X_new, 0.8, target="sum")           # the total: (rows, 2)
model.predict_quantiles(X_new, [0.5, 0.95], target=peak)   # (rows, 2)
model.predict_exceedance(X_new, capacity, target="sum")    # P(total > capacity), (rows,)
```

Each column and each function keeps its own table of probability integral transforms, the same split-conformal step a one-column head takes ([calibrated intervals](heads.md#calibrated-intervals)), so the total's 80% interval holds 80% of new rows, and so does each site's. `target` is `"sum"`, one weight per column for that weighted sum (`[1, -1, 0, 0]` for a difference), a function defined with `def`, or a list of them; the sum is always included. A weighted sum is exact under the joint normal and the joint mixture; a function, and everything about the flow, is read from `n_samples` joint draws per row, 1,000 by default. A saved model keeps every table, a function's under its name, since its code is not saved: pass the function again after `load`. A target the calibration did not include is refused, and `calibrated=False` reads any target raw. The held-out rows must be exact: where a cell is censored or an interval, a function of the columns has no single value.

On the electricity loads of [the joint demand benchmark](benchmarks.md#calibrating-the-total), calibration puts the total's 80% interval at 79.5% to 80.5% of hours for every joint model tried, from 68.6% to 88.6% before, and makes P(total > capacity) more reliable where a model was off: where independent heads said 2% to 10%, the event happened 12.6% of the time before and 7.5% after. One table for every row cannot fix an error that changes from row to row, such as coverage that varies with the hour of the day; `calibrate(X, Y, groups=...)` keeps a table per group for that ([calibrated intervals](heads.md#calibrated-intervals)). With a table per hour, the flow's total lands inside its 80% interval in 77% to 83% of the hours at every hour of the day, from 67% to 91% with one table ([per hour of day](benchmarks.md#calibrating-per-hour-of-day)).

A box of per-column intervals is a poor region for outcomes that move together, and no box follows a curved dependence. `calibrate` also calibrates the joint highest-density region, the smallest set of outcomes with a given probability:

```python
threshold = model.predict_region(X_new, 0.8)          # a log density per row
inside = model.log_prob(X_new, Y_new) >= threshold     # True where the whole outcome is in the region
```

Uncalibrated, the region has probability 0.8 under the model. Calibration scores each held-out row by the model's probability of an outcome denser than the one seen, and moves the region's probability until it holds 80% of those rows, so it holds at least 80% of new rows exchangeable with them. The threshold is read from `n_samples` joint draws per row. A calibration by group (`groups=`, [calibrated intervals](heads.md#calibrated-intervals)) calibrates every column, function, and region within each group. On the eight electricity loads, the uncalibrated 80% region held the whole outcome in 61% of the test hours for the joint normal and 53% for the flow, and calibration took it to 79% and 78% ([the numbers](benchmarks.md#calibrating-per-hour-of-day)).

### Training on other proper scores

A joint head trains on its log density by default. `objective` trains it on other proper scores, alone or added to the log density:

```python
ASRRegressor(task="multivariate_flow", objective="energy")                    # the energy score alone
ASRRegressor(task="multivariate_flow", objective={"nll": 1.0, "energy": 1.0})  # added to the log density
ASRRegressor(task="multivariate_gaussian", objective={"nll": 1.0, "variogram": 1.0})
```

The log density scores each column given the ones before it. The energy score, `E‖X − y‖ − ½ E‖X − X′‖`, scores the whole outcome as one point, so it also rewards getting a total or another function of the columns right; with one column it is the CRPS. The variogram score of order 0.5 compares `|y_i − y_j|^0.5` for every pair of columns with what the model expects of it, so it is sensitive to how the columns move together. A shift of every column leaves it unchanged, so it cannot place the columns on its own, and it trains only in a blend with `"nll"` or `"energy"`.

Both are estimated without bias from 16 reparameterized draws per row, fresh at every step, on the standardized columns, so each column counts the same whatever its units. A mixture is drawn from every component, weighted by the component probabilities, so the gradient reaches those probabilities without a discrete draw. Validation reads the same draws every epoch, so epochs compare on equal terms. The flow takes one pass per column to draw, so on these scores it trains several times as slowly as on its log density.

On the electricity loads of [the joint demand benchmark](benchmarks.md#training-on-other-proper-scores), neither score did better than the log density. Trained on the energy score alone, the joint normal gave the total an 80% interval that held 70.8% of hours and the flow 64.7%, against 80.5% and 74.8% when trained on the log density, and each lost 1.7 to 3.5 nats of joint log density: the models grew sharper, not better at the total. Added to the log density, the energy or variogram score tied it on joint log density and did slightly worse on the total's CRPS. `calibrate` then put every one of them between 78.6% and 80.4%, so for the coverage of a total, calibrate the model. The objectives are here to compare training scores on your own data.

## Dependence that is not a correlation

A joint normal can only say how strongly columns move together on a straight line. `task="multivariate_mixture"` takes the same `y` and predicts a mixture of joint normals, five by default (`head_kwargs={"components": 5, "rank": 2}`), each with its own weight, location, and low-rank covariance for every row. It fits dependence that comes in regimes, such as a rush day when every column jumps together and an ordinary day when they barely move, which a single normal averages into one correlation. Its `log_prob` is exact, a draw picks a component and then draws from it, and its marginals are mixtures of normals with exact quantiles.

For dependence that is curved or skewed rather than a few regimes, `task="multivariate_flow"` takes the same `y` and predicts a conditional masked autoregressive flow: `p(y | x) = p(y₁ | x) p(y₂ | y₁, x) ⋯`, where each column's conditional is a standard normal pushed through a monotone spline, shifted and scaled. A masked network reads the columns before each one and the row's context from the trunk, and sets that column's knots, shift, and scale. So the whole shape of a column, not only its mean, can change with the columns before it: one load that spikes when another is high, a cost that grows more uncertain with the duration, a dependence that is strong in the tails and absent in the middle.

```python
model = ASRRegressor(task="multivariate_flow").fit(X, Y)
dist = model.predict_dist(X_new)          # batch (rows,), event (columns,)
model.log_prob(X_new, Y_new)              # exact joint log density
draws = dist.sample(jax.random.key(0), (4000,))
```

`head_kwargs={"hidden": 64, "depth": 1}` are the defaults: the masked network's width and the number of autoregressive blocks, the column order reversed in every other block. One block already gives every column a flexible conditional; a second lets the first column depend on the later ones too, so the column order matters less. The column order is the order of `Y`'s columns, so put the ones that drive the others first when you know them.

The density is exact and costs one network pass. A draw takes one pass per column, so it is slower than the normal's. The mean, variance, and covariance have no closed form; `predict_dist(X).mean()` estimates them from 1,024 fixed draws, and `predict_quantiles` reads each column's quantiles from `n_samples` joint draws, so they depend on `n_samples` and `seed` here, unlike the other heads. On a curved dependence with no correlation at all, the second column the square of the first's noise, it beats `multivariate_gaussian` by 2.1 nats per held-out row in joint log density (−4.54 against −6.68, mean of three seeds with 2,500 training rows each; the data is in `tests/test_multivariate_flow.py`); on the electricity loads of [the joint demand benchmark](benchmarks.md#joint-demand-across-locations) it has the best joint log density of every model tried, 0.83 nats per hour better than the rank-2 normal, but its interval for the total runs a little narrow (77% for a nominal 80%).

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

`predict_dist(X)` encodes the frame on the host, so `jax.grad` stops at the distribution and never reaches `X`. To differentiate a function of the distribution in the features, encode once with `model.encode(X)` and call `model.forward_dist` on the arrays. It runs imputation, the fitted scaling, the network, and the head inside JAX and returns the same `Distribution` as `predict_dist`:

```python
batch = model.encode(X_new)

def q90(numeric):
    dist = model.forward_dist(numeric, batch.categorical_onehot, batch.category_index,
                              batch.set_tokens, batch.set_mask)
    return dist.quantile(0.9).sum()

jax.grad(q90)(batch.numeric_raw)  # how each row's 90% quantile moves with each numeric feature, in its units
```

`numeric_raw` is in the units of `X`, missing cells as NaN; a missing cell is filled with the training median, so its gradient is 0. Categories are looked up on the host and have no gradient. `model.dist_from_raw(raw)` turns the network's raw output (`forward_raw`) into the distribution, for code that changes the network's output itself.

## Explaining the distribution

`model.explain(X, target)` attributes one number per row, read off the predictive distribution, to the features. The number can be a quantile, the spread, the width of an interval, or a tail probability, so it answers questions a mean cannot: which features widen this row's interval, and what drives its 95% quantile.

```python
model.explain(X_new, "quantile", level=0.95)               # the 95% quantile
model.explain(X_new, "interval_width", coverage=0.8)       # the 80% interval's width
model.explain(X_new, "std")                                # the predictive standard deviation
model.explain(X_new, "exceedance", threshold=500.0)        # P(y > 500)
model.explain(X_new, lambda d: d.quantile(0.9) - d.median())  # any fn(dist) -> one number per row
```

It returns an `Explanation`: `values` is `(rows, features)` in the target's units, `prediction` the target at each row, and `importance()` the mean absolute attribution per feature, largest first. `to_frame()` gives a pandas `DataFrame`. The function is also `ashlar.explain.explain(model, X, target)`.

- **Numeric features** get integrated gradients (Sundararajan et al., 2017) from a baseline row to the row through `forward_dist`, with 32 steps by default (`steps=`). The baseline is each column's training median, or the median of `background=` when given. The numeric attributions of a row sum to the target at the row minus the target with every numeric feature at its baseline (`baseline`, and `numeric_gap()` is what the integration misses). A quantile is differentiated by the implicit-function rule, so its gradient is exact. `method="gradient_x_input"` takes one gradient at the row instead, exact only for a target linear in the features. A missing cell sits at its baseline and gets 0.
- **Categorical features** have no input gradient, since the lookup happens on the host. Each gets an occlusion attribution: the target at the row minus its mean with that column set to each reference category in turn, the 32 most frequent values of `background`, or the training categories with equal weight. These do not enter the numeric sum.
- A joint head's target must be one number per row, for example `lambda d: d.project(weights).quantile(0.9)` for the 90% quantile of a weighted total.
- A cyclical column's sine and cosine attributions are summed back onto the column.

On synthetic data where one feature moves the location and another the spread, the median's attribution picks the first and the interval width's the second (`tests/test_explain.py`). On hourly bike rentals (`examples/explain_bike.py`, a mixture head, 1,000 test rows, a few seconds per target on a laptop CPU), the hour of day drives all three targets, but working days matter almost as much to the width of the 80% interval (28 rentals) as to the median (32), while the year and the temperature move the median far more than the width:

| feature | median | 80% width | P(> 500) |
|---|---:|---:|---:|
| hr | 111 | 49.9 | 0.052 |
| workingday | 31.9 | 28.0 | 0.004 |
| yr | 31.4 | 10.9 | 0.016 |
| weathersit | 19.8 | 13.3 | 0.011 |
| temp | 19.0 | 8.0 | 0.005 |
| season | 14.2 | 4.0 | 0.012 |

## Your own head

A `task="custom"` model's `dist_fn(raw)` should return one of these objects, built from the head's raw output, so `sample`, `log_prob`, and the quantile methods work on it too. `ashlar.demos.linked` returns the `Mixture` of its members' normals this way.

## Forecasts

`ASRForecaster.predict_dist(history)` and `ASRHistoryForecaster.predict_dist(history)` return the same objects with batch shape `(windows, horizon)`, or `(horizon,)` for one history. Each step is its own marginal: the forecasters model every step of the horizon separately, and the object makes no claim about how the steps move together.
