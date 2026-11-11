# Dressing a model

[Ashlar](../README.md) can fit several networks and several label readers to one table and return them as one distribution. `ashlar.dressed()` is the recipe; the class is `ASRDressed`.

```python
import ashlar

model = ashlar.dressed().fit(X_train, y_train)

dist = model.predict_dist(X_test)        # one distribution per row
dist.log_prob(y_test)                    # exact log density
dist.cdf(y_test)                         # exact CDF
model.predict_quantiles(X_test, [0.1, 0.5, 0.9])
model.predict(X_test)                    # the mean
```

It is for one numeric target on a table of up to a few thousand rows, when the predicted distribution matters more than the time to fit. It is the slowest model in Ashlar: eleven fits in one.

## Why

A network with a density head fits a smooth map from the columns to the target. That is the wrong tool for some targets: one that is a linear function of the columns, one with a handful of values, one where a row's nearest neighbours already hold the answer.

A label reader (`ashlar.label_reader.LabelReader`) is the other tool. It has no weights from the columns to the target. For a row it fits a ridge regression on the training rows inside the forward pass, then spreads the row's probability over the training rows' leave-one-out residuals by attention. Its prediction is a set of points taken from the training labels.

Neither wins on every table, so the two are pooled: a weight `w` on the network and `1 - w` on the reader. But the reader predicts point masses. A pool with point masses has no density, so no log score, and a draw from it can only repeat a training label. Dressing gives that pool a density and then corrects it.

## What `fit` does

1. **Members.** The training rows are cut into five parts. Member `k` holds out part `k` and fits a network (`tabm_piecewise` with a `flow` head, stopped on the held-out CRPS, then annealed for three epochs) and a label reader on the other four parts.
2. **Pool weight.** One weight `w` from a grid of 21, by the CRPS of every training row under the member that held it out.
3. **Smoothing.** Each of a reader's points becomes a normal. Its width is Silverman's bandwidth on the row's points, times a factor from a grid of five. 1% of a wide normal at the training labels' mean and spread is added. One factor is picked by the held-out log score and a second by the held-out CRPS. A member's pool now has a density.
4. **Corrector.** One more network is fitted to where each training label falls in its own row's pool: the label's normal score under the pool of the member that held the row out. It trains on the held-out rows of the first four members and stops on the fifth's. A member's corrected CDF at `y` is the corrector's CDF at the normal score of `y` under that member's pool.
5. **Gate.** One weight from the same grid of 21, for the corrected pools against the pools smoothed at the CRPS factor, by the CRPS of the fifth member's held-out rows. If the corrector did not help on those rows, the gate is 0 and the prediction is the smoothed pools alone.

The prediction is the gate times the average of the five corrected pools, plus the rest times the average of the five smoothed pools. Every piece has a closed-form CDF and density, so the whole does too.

Bagging the members is standard. Correcting a prediction on its own probability scale is how probability calibration works, and composing models on that scale is not new either. What this model adds is the label reader, and using the held-out pool of the bagged members as the scale the corrector works on.

## What comes back

`predict_dist(X)` returns a `DressedDistribution`. A value array's last axis runs over the rows, as with every Ashlar distribution.

| Method | Returns |
|---|---|
| `log_prob(y)` | The exact log density. |
| `cdf(y)` | The exact CDF. |
| `quantile(levels)` | Quantiles, by solving the CDF. Accurate to about `3e-7` in probability, the precision of the networks' float32 CDFs. |
| `sample(n, seed)` | `n` draws per row, shape `(n, rows)`, by the quantile of uniform draws. |
| `mean()` | The mean, read from the quantiles at 203 levels. |
| `crps(y)` | The CRPS of the quantiles at 203 levels, which is how a quantile benchmark reads a density. |

On the estimator, `predict`, `predict_quantiles`, `log_prob`, `sample`, and `score` call these. A fitted model pickles.

## A target with few values

A target with at most 20 distinct values (`few_values=20`) is treated as a set of values. The members' networks are ordinal, the corrector is trained on the interval each value covers, and the prediction is a probability for each value. `dist.values` holds the values and `dist.probabilities()` each row's probability of each. `log_prob` is then the log probability, `-inf` off the values, and `quantile` returns the smallest value whose CDF reaches the level. `few_values=0` always fits a density.

## Settings

```python
ashlar.dressed(learning_rate=1e-3)             # any tabm_piecewise setting, for every network
ashlar.dressed(members=3, random_state=1)
ashlar.ASRDressed(network={"learning_rate": 1e-3}, reader={"max_steps": 2000})
```

| Argument | Default | Meaning |
|---|---|---|
| `members` | 5 | How many cross-fitted members. |
| `few_values` | 20 | The most distinct values a target may have and still be treated as a set of values. |
| `network` | `None` | Settings for `tabm_piecewise`, on top of `max_epochs=1000` and `cooldown_epochs=3`. The task, the seed, and the stopping metric are set by the model. |
| `reader` | `None` | Settings for each `LabelReader`. |
| `reader_rows` | 4096 | The most training rows a reader reads at once. A larger part is sampled down. |
| `floor_cost` | `None` | See [Limits](#limits). |
| `random_state` | 0 | Seeds the split, the members, and the corrector. |

After a fit, `pool_weight_`, `bandwidth_factor_`, `fallback_factor_`, and `gate_` are the four searched numbers, `members_` and `corrector_` the fitted parts, and `history_` the searched curves.

## Limits

- **One numeric target, exactly observed.** Censored and interval labels, joint targets, classes, and sample weights raise an error.
- **Small and mid-size tables.** Every test below is on at most 2,400 training rows. A reader's cost grows with the square of its rows, so a member's reader reads at most `reader_rows` of them; nothing above that size has been tested.
- **A target the columns determine almost exactly.** The 1% wide normal costs about 0.00006 of the labels' spread in CRPS. That is nothing on most tables. On a target whose CRPS is already below about a thousandth of its spread, it is a multiple of the whole score, and the corrector has to win it back. On one such table in the tests below it did on two folds and did not on the third. `floor_cost=0.01` lowers the wide normal's share until it costs at most 1% of the pool's held-out CRPS; it is off by default until it has passed a test of its own.
- **Time.** Five networks, five readers, and a corrector.

## How it was tested

The design was chosen on 23 regression tables: 12 from OpenML's CTR23 suite, 7 from PMLB, and 4 of those with the target binned into 5 to 10 ratings. Each is split five ways, with at most 2,400 training and 600 test rows. Fold 0 was used to design the model. Folds 1 and 2 were not looked at until the design was fixed, and the pass line was written down before they were read.

CRPS is lower-is-better. A ratio below 1 means the dressed model is better.

| Dressed model against | Folds 1 and 2, 46 fits: geometric mean | Median | Better by >1% | Within 1% | Worse by >1% |
|---|---|---|---|---|---|
| One member's network alone | 0.569 | 0.916 | 37 | 7 | 2 |
| The five networks, averaged | 0.568 | 0.916 | 37 | 7 | 2 |
| One network and one reader, pooled | 0.920 | 0.956 | 38 | 6 | 2 |
| The five members pooled, not dressed | 0.955 | 0.974 | 27 | 17 | 2 |

The last two rows have no density, so only the first two can be compared on log score. The dressed model's mean log score was above the five averaged networks' on 39 of the 46 fits, by 0.65 nats on average and 0.18 at the median.

The geometric means in the first two rows are pulled down by three tables where the network alone is far off (a linear target and two others); the medians are the better summary.

The pass line was on the last row: a geometric mean at most 0.5% above the undressed pool's, worse by more than 1% on at most 4 of the 46, and a log score above the networks' on at least 24 and on average. It passed. A second, looser line that also capped serious harm at one fit did not: two fits were worse than the undressed pool by more than 5%.

- `wave_energy`, fold 2: 5.5 times the undressed pool's CRPS. The target is a linear function of the columns, the case described under [Limits](#limits). On folds 0 and 1 of the same table the dressed model was 11% and 6% better. Against the network alone it is still about 200 times better on that fold.
- `557_analcatdata_apnea1_rated6`, fold 2: 10% worse. The gate was picked on 76 rows and chose the corrector where it did not help.

Without the first of these the geometric mean of the last row is 0.919.

On fold 0, which the design had seen, the same ratio was 0.913, better by more than 1% on 13 tables and worse on 1.

The numbers above come from the research code the design was worked out in. `ASRDressed` was then run on fold 0 of all 23 tables, with the same splits and seeds, to check that the library gives the same model:

| `ASRDressed` against the research code, fold 0, 23 fits | Median | Geometric mean | Largest gap |
|---|---|---|---|
| CRPS of the dressed model | 1.000 | 1.000 | 4.4% |
| CRPS of the five members pooled, not dressed | 1.000 | 1.000 | 2.1% |

On its own fits the library's dressed model was at 0.913 of its undressed pool, better by more than 1% on 13 tables and worse on 1, and its mean log score was above the five averaged networks' on 22 of the 23, by 0.69 nats on average.

Three earlier designs failed on fold 0 and were dropped: a network fitted on top of the reader's prediction alone (worse than the pool on 15 of 23 tables), smoothing with no corrector (2% worse than the undressed pool, worse on 7), and the corrector with no gate (7% better on average but worse on the four smallest tables, where it stops within five epochs).
