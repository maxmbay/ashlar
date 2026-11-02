# Additive models of the whole distribution

[Ashlar](../README.md) can fit a distribution whose every parameter is a sum of one shape per column. `backbone="additive"` gives each column of `X` a small network of its own, and each pair of columns you name one more. The networks meet only in the head, which is linear in them, so each of the head's outputs is an intercept plus one function per term:

```
output(x) = intercept + f_temp(temp) + f_hour(hour) + … + f_hour,workingday(hour, workingday)
```

Each output is on the scale the head's distribution reads it on: a location, a scale before its softplus, a log mean, a logit. This is a generalized additive model for location, scale, and shape (GAMLSS, [Rigby and Stasinopoulos, 2005](https://doi.org/10.1111/j.1467-9876.2005.00510.x)), with small networks in place of splines, trained with the rest of the model by gradient descent. It works on every built-in head except `multivariate_flow` and `chain`, which read the hidden layer through networks of their own. So the additive model also gets the count and hurdle heads, mixtures and flows, observed labels, monotone constraints, entity effects, and `calibrate`.

```python
from ashlar import ASRRegressor

model = ASRRegressor(
    task="negative_binomial",
    backbone="additive",
    backbone_kwargs={"pairs": [["hr", "workingday"]]},
).fit(X_train, y_train)

shapes = model.shape_functions(X_train)
shapes.names                      # ('season', 'yr', ..., 'hr:workingday')
shapes.outputs                    # ('log_mean', 'dispersion')
shapes.term("temp", "log_mean")   # temperature's part of each row's log mean
shapes.importance("dispersion")   # which columns move the spread, largest first
```

![Shape functions of an additive negative binomial on hourly bike rentals: the hour on working days and on weekends, temperature and felt temperature, and humidity on the log mean and the dispersion; and the calibrated 80% interval of rentals at 5 pm on a working day as the temperature rises.](images/additive-bike.png)

The figure is [`examples/benchmark_additive.py figure`](../examples/benchmark_additive.py): the model above on hourly bike rentals, trained on 64% of the hours and calibrated on 16%. Each curve is one term's part of each training row's log mean or dispersion, against its column.

- **The hour depends on the day.** On working days rentals peak at 8 am and again at 5 to 6 pm; on weekends and holidays they rise to one broad plateau from late morning to evening. Only the pair can draw two shapes; without it the hour has one, and its likelihood is 0.33 nats worse per hour.
- **Rentals rise with the temperature, then fall in the hottest hours.** The model splits the effect between the temperature and the felt temperature, which move together, so read the two together.
- **Humidity moves the spread more than the mean.** The mean barely changes until the air is humid, and then falls, while the dispersion rises on humid hours: rain makes rentals less predictable as well as fewer. The dispersion's shape is noisier than the mean's, as the spread is harder to learn than the level.
- **Quantiles come from the whole model.** The last panel holds a 5 pm working-day hour fixed and moves the temperature: the median rises from about 380 rentals on the coldest days to about 870, and falls on the hottest. That is "how the 90% quantile of demand depends on temperature" with a coverage guarantee: on the held-out hours, the calibrated 80% interval holds 81.5% of rentals.

## Reading the terms

`shape_functions(X)` returns a `ShapeFunctions` with one value per row, term, and output:

- `values` is `(rows, terms, outputs)`. On every row, `intercept` plus the terms is the head's output (`total()`), exactly. Each term is centered on the rows of `X`, its mean moved into the intercept, so pass the training rows, or a sample of them, to read every term against the data's average. `center=False` leaves each term as its network gives it.
- `names` are the terms: a column, `a:b` for a pair, and `column (random effect)` for a grouping column's offsets. A column that `cyclical` splits into a sine and a cosine is one term. `columns` gives each term's columns.
- `outputs` names the head's outputs, such as `loc` and `scale` for `gaussian`, `log_mean` and `dispersion` for `negative_binomial`, `zero_logit`, `severity_loc`, and `severity_scale` for `hurdle`, and `loc_1`, `scale_1`, `weight_1`, … for `mixture`. Location outputs are in the units of `y`. The others are on the head's raw scale: a raw scale or dispersion becomes the scale through a softplus, so a rising term means more spread, and a log mean, a weight, or a logit is read on its own scale.
- With members (`backbone_kwargs={"k": 4}`), each member is an additive model of its own. `values` averages them, `member_values` keeps them apart, and `spread` is their standard deviation, so a term the members disagree on, such as a shape far from the training rows, shows it.
- `importance(output)` is the mean absolute value of each term over the rows, largest first, and `to_frame(output)` is a pandas DataFrame of every term on one output.

To draw one column's shape, sort the rows by that column and plot `term(column, output)` against it. A pair's term is a function of both columns. Its part that depends on only one of them is not moved into that column's own term, so read a column's term together with its pairs: in the figure, the hour on a working day is the hour's term plus the pair's plus the working day's.

A monotone feature (`monotone={"price": -1}`) is a term of its own, the rising or falling piecewise-linear shape the constraint adds, and no network reads it, so it cannot be in a pair. A grouping column of `random_effects` adds its offsets on the location as the term `column (random effect)`; its spread is not a term, since it widens the distribution rather than moving an output.

## Options

| `backbone_kwargs` | Default | What it does |
|---|---|---|
| `pairs` | none | Pairs of columns, `[["hr", "workingday"], ...]`, each with a network over both. |
| `n_blocks` | 2 | Layers per term. |
| `d_block` | 64 | Width of each term's layers. |
| `d_term` | 8 | Width of each term's output. The head mixes these into its outputs, so this is how many independent shapes a term can give the head's outputs. |
| `k` | 1 | Members, each its own additive model. |
| `dropout` | 0 | Dropout inside each term. |
| `activation` | `"silu"` | `"relu"` or `"gelu"` instead. |

Numeric columns go through the same encoding as any backbone, piecewise-linear by default (`num_embeddings`), so each term reads its own column's encoding. `feature_dropout` drops a whole column, and so a whole term, on a training step.

The additive backbone refuses what would mix its terms: a `bottleneck`, `soft_stages`, `share_training_batches=False`, and the `multivariate_flow` and `chain` heads. `shape_functions` reads the one-column built-in heads; on `multivariate_gaussian` and `multivariate_mixture` the outputs are still additive, but it does not read them.

## Against R's gamlss and the residual network

On hourly bike rentals and abalone rings, three splits each, the additive Ashlar models were scored against R's `gamlss` with penalized splines and against Ashlar's default residual network with the same head. [docs/benchmarks.md](benchmarks.md#additive-models-against-rs-gamlss) has the tables.

## Limits

- An additive model learns only the interactions you name. That is the point: each shape is readable because it does not depend on the other columns. The residual network is the measure of what that costs; in the benchmark above, compare the additive rows with it.
- Weight decay and early stopping smooth the shapes, not a penalty chosen by a criterion as in `gamlss`. Where a column has few rows, its shape can wiggle, and members show it.
- The terms are on the head's raw outputs. A quantile, the mean, or an interval is a nonlinear function of several outputs, so it is not additive; read it from `predict_quantiles` or `predict_dist`, as the last panel of the figure does, or attribute it to the columns with `explain`.

The research models in [docs/models.md](models.md#research-models), `ASRAdditive` and `ASRGatedAdditive` among them, are experimental: they fit a point or class probabilities with their own `fit(X, y, task)`, and are not connected to the heads, `predict_dist`, observed labels, or `calibrate`. For an additive model of a distribution, use `backbone="additive"`.
