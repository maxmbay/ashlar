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

## Outcomes of different kinds, chained

The joint heads above model columns of one kind, continuous values. Many tables record several outcomes of different kinds about each row: how many claims a policy made and what they cost, whether a customer churned and when, which part failed and after how long. `task="chain"` predicts them together, with a head of its own kind for each column, each conditioned on the columns before it:

```
p(claims, cost, type | x) = p(claims | x) · p(cost | claims, x) · p(type | claims, cost, x)
```

```python
from ashlar import ASRRegressor

model = ASRRegressor(
    task="chain",
    head_kwargs={"columns": ["negative_binomial", "hurdle", {"kind": "categorical"}]},
).fit(X, Y)                                   # Y is (rows, 3): claim count, cost, claim type code

model.log_prob(X_new, Y_new)                  # exact joint log-likelihood per row
draws = model.sample(X_new, n=1000)           # (1000, rows, 3): coherent draws, one column at a time

def total_cost(y):
    return y[..., 1]

model.calibrate(X_holdout, Y_holdout, target=total_cost)
model.predict_interval(X_new, 0.9, target=total_cost)
```

A column's kind is any one-column head that has a likelihood: `poisson` and `negative_binomial` (with `"zero_inflated": True` in a dict), `hurdle` (with `"severity": "gamma"`), `categorical` (codes `0, 1, 2, ...`; the class count comes from the training values, or `"classes": n`), and the continuous `gaussian`, `mixture`, `flow`, `histogram`, and `student_t`, which are standardized column by column; counts, amounts, and classes stay in their own units. Each column's head reads the row's hidden state and the values of the columns before it, as observed in training and as drawn in a draw, through a small network of its own (`head_kwargs={"hidden": 32}`), so the cost of a claim can depend on how many claims there were, and its spread as well as its mean. The joint log-likelihood is the sum of the columns' conditional log-likelihoods, a mass for a count or a class and a density for an amount or a value, so it is exact. A question about several columns, such as the total cost of a policy, is read from the joint draws, as for the flow, and `calibrate` calibrates each column and any function of them the same way ([calibrating the columns and the total](#calibrating-the-columns-and-the-total)). Order the columns so that the ones that drive the others come first.

A chain takes a [joint label](observed.md#several-products-that-sell-out-together) too. A categorical column's cell can be a range of classes, `observed.interval(-inf, inf)` for a class that was not recorded, and the likelihood sums over the classes it allows; the last column's cell can be censored or an interval. Every other cell must be exact, because the columns after it condition on its value. That is enough for a censored time with an unknown cause, the competing-risks model in [docs/observed.md](observed.md).

On freMTPL2's 677,991 motor policies, a chain of a `negative_binomial` claim count and a `hurdle` cost has a joint NLL of 0.467 nats per policy, against 0.615 for the same two heads fitted apart, because it knows that a cost is zero exactly when there was no claim. It ties the actuary's frequency and severity model built by hand, 0.4672 against 0.4673, with the total's calibration the same, and gets there from the column kinds alone. [docs/benchmarks.md](benchmarks.md#chained-claim-counts-and-costs) has the table.

## Features that push one way

A tariff rule may say that a worse claims history can never lower a premium, a credit policy that more income can never raise a default rate. Boosted trees enforce that on the mean. `monotone=` enforces it on the whole distribution:

```python
model = ASRRegressor(task="quantile_spline", monotone={"BonusMalus": 1, "Price": -1}).fit(X, y)
```

The named numeric features are kept out of the trunk and added back through a term of their own: for each feature, a piecewise-linear shape over 16 bins at its training quantiles, rising in every bin, open-ended past the first and last. The term goes to the head outputs whose rise moves every quantile up and to nothing else, so `P(y > t | x)` cannot fall as a feature marked `+1` rises, for every threshold `t` and every row, and every quantile rises with it. That is first-order stochastic dominance, which a constraint on each parameter does not give: a rising mean with a falling spread still crosses.

| `task` | What a constrained feature moves |
|---|---|
| `regression` | the point |
| `gaussian`, `student_t` | the location; the scale and degrees of freedom read only the other features |
| `poisson`, `negative_binomial` | the log mean; the dispersion and a structural zero read only the other features |
| `quantile_spline` | the median, the slopes above it up, and the slopes below it down, so the spread can widen upward or narrow from below as the feature rises |

The `quantile_spline` head is a quantile function piecewise linear in the logistic score, a median and eight slopes per row, four on each side; its CDF, density, and quantiles are closed forms, and its tails past the last break are exponential. It is the head for a constrained feature that should move the spread as well as the location. The other heads take every quantile up by the same amount, a shift.

The price is that a constrained feature does not interact with the others: its effect is one shape, on the scale of the outputs it reaches, for every row. Mixtures, flows, and histograms are refused, since a feature moving their weights or splines can cross.

MONOTONE_RESULTS

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

## Missing and noisy inputs

A fitted model fills a missing numeric cell with the training median, and its predictive distribution is then as narrow as if the value were known. `with_uncertain_inputs` integrates over what is not known about each row instead:

```python
wide = model.with_uncertain_inputs(draws=32)
wide.predict_interval(X_new, 0.8)                     # wider where cells are missing
model.with_uncertain_inputs(32, noise={"income": 5000.0}).predict_interval(X_new, 0.8)  # a measurement error
model.predict_dist(X_new, input_draws=32)             # the same, for one call
model.input_log_density(X_new)                        # log p(x) of each row's numeric features
```

At `fit`, Ashlar fits a Gaussian copula to the numeric features of the training partition (`input_model_`, saved with the model): each column is mapped to a normal score through 129 of its empirical quantiles, and the scores share one correlation matrix. The view runs the network on `draws` completions of each row and returns the equal-weight mixture over the completions and the members, so `predict_interval`, `predict_quantiles`, `predict_exceedance`, `sample`, `log_prob`, `predict`, and `calibrate` all integrate over the inputs.

- A missing numeric cell is drawn from its distribution given the row's observed numeric cells, so a missing latitude is drawn near the latitudes that go with the row's longitude.
- `noise={"column": sd}` declares a measurement error: the cell is drawn as `value + sd * N(0, 1)`, a number or one standard deviation per row. Its measured value still conditions the missing cells.
- A row with no missing or noisy cell gets the same distribution as without the view.
- Categorical cells are not drawn; a missing category is still the unknown category. Training is unchanged.
- `input_log_density(X)` is the copula's `log p(x)` over each row's observed numeric cells, in the units of `X`, with missing cells integrated out: a score for rows unlike the training data. `fit_inputs(X)` refits the copula, for example on a model saved by an older version.

On California housing and hourly bike rentals (`examples/benchmark_inputs.py`, a `mixture` head, calibrated on complete rows, seeds 0 to 2), with numeric test cells masked at random, the median's 80% interval stops holding as cells go missing. The draws' interval holds within 80% to 85%:

| table | masked cells | median: coverage | median: NLL | 32 draws: coverage | 32 draws: NLL |
|---|---:|---:|---:|---:|---:|
| California | 10% | 65.5% | 0.84 | 80.8% | 0.26 |
| California | 30% | 48.3% | 1.87 | 82.4% | 0.53 |
| California | 50% | 38.3% | 2.49 | 82.9% | 0.79 |
| bike rentals | 10% | 72.8% | 4.88 | 81.8% | 4.63 |
| bike rentals | 30% | 59.3% | 5.67 | 83.3% | 4.90 |
| bike rentals | 50% | 48.3% | 6.33 | 83.5% | 5.15 |

Noise added to the most important feature at test time, with a standard deviation equal to the column's own, drops the 80% interval to 59.3% on California and 65.4% on bike rentals when it is ignored, and keeps it at 80.8% and 79.3% when it is declared. `input_log_density` tells held-out California rows from rows above the training range of income with an area under the ROC curve of 0.938. [benchmarks.md](benchmarks.md#missing-and-noisy-inputs) has every row.

## Deciding from the distribution

Most predictions end in a number someone acts on: an order, a capacity, a reserve. `ashlar.decide` reads it off any `Distribution`, and every function is traceable JAX, so it also runs inside `jax.jit` and `jax.grad`.

```python
from ashlar import decide

model.predict_order(X_new, under=9, over=1)        # newsvendor: a unit short costs 9, a unit left over 1
dist = model.predict_dist(X_new)
decide.newsvendor_order(dist, 9, 1)               # the same: the 9 / (9 + 1) = 0.9 quantile
decide.newsvendor_cost(y, order, 9, 1)            # the realised cost per row
decide.exceedance(dist, 500.0)                    # P(y > 500)
decide.cvar(dist, 0.95)                           # the mean outcome above the 95% quantile
decide.exceedance(joint_dist, 1200.0, weights=[1, 1, 1])  # P(total > capacity) on a joint head
decide.best_action(dist, candidates, cost, jax.random.key(0))  # any cost, by draws
```

`predict_order(X, under, over)` is `predict_quantiles` at the critical ratio `under / (under + over)`, so after `calibrate` it is the calibrated quantile: the share of new rows whose demand it covers holds at the critical ratio, a cost-optimal order with a service level that holds. `cvar` integrates the quantile function, exactly up to quadrature; `expected_utility` and `best_action` work from draws and take any utility or cost.

**Training on the decision's cost was tried and is not shipped.** A head trained on its likelihood plus the newsvendor cost of its own order (the critical-ratio quantile of its member mixture, differentiated through the quantile's implicit gradient) did not order better than the same head trained on its likelihood alone. On a skewed target with a 9:1 cost, the likelihood-trained mixture's plug-in order already costs 0.7% more than the true quantile's, and the blend cost 0.3% more than the plug-in, short of the 1% gain issue #32 set as the bar. On hourly bike rentals the blend's gain was under 1% at every cost ratio from 1:1 to 19:1. It helps only a misspecified head, and there `calibrate` gets most of the way: a normal head on the skewed target orders 59% above the oracle, 25% after `calibrate`, 1.4% when trained on the cost alone, at a heavy price in likelihood. The objective and the results are in `examples/benchmark_decide.py` and [benchmarks.md](benchmarks.md#decisions-from-the-distribution).

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
