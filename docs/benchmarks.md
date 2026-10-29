# Benchmarks

[Ashlar](../README.md) beside XGBoost and LightGBM. Every number on the README comes from [`examples/benchmark_readme.py`](../examples/benchmark_readme.py), and this page has all of them, per table and per metric.

```bash
pip install -e ".[dataframes]" xgboost lightgbm pyarrow
JAX_PLATFORMS=cpu python examples/benchmark_readme.py --stage all
python examples/benchmark_readme.py --stage report
```

## How it was run

Each table is split 80/20 at random, three times, with seeds 0, 1, and 2. A cell is the mean over the three test sets ± the standard deviation across them. The temperature forecast is split on the clock instead, the last 20% of the series, so its three runs differ only in the seed.

**Ashlar** runs with its defaults unless a section says otherwise. It holds out 20% of the training rows to pick the epoch.

**XGBoost 3.4 and LightGBM 4.7** read the same columns, with strings as pandas categories so both use their native categorical splits. "Default" is the library's constructor defaults. "Early-stopped" is a learning rate of 0.05 and up to 4,000 trees, stopped after 100 rounds without improvement on a 20% validation slice of the training rows. A category the training rows never saw is missing to the trees, as it is index 0 to Ashlar.

**Distributional scores** are on the original scale of the target.

- *CRPS*, the continuous ranked probability score, is the expected absolute error of a draw minus half the expected distance between two draws. It scores the whole distribution, in the target's units, and stays finite on targets with a spike. Ashlar's is from 400 draws per row.
- *Pinball* is the quantile loss averaged over the 0.1, 0.5, and 0.9 levels. For a density head the levels are taken from the same 400 draws.
- *80% coverage* is the share of held-out values between the 0.1 and 0.9 levels, and *80% width* is the average distance between them.
- *NLL* is the negative log density per row. It is only comparable between continuous densities on a continuous target. On taxi tips and California values, which have a spike, a head can drive it down by making the spike narrow, and the histogram's is capped by its bin width. Read CRPS there.
- The trees have no density. "Mean + constant normal" is the early-stopped mean with a normal whose scale is the residual standard deviation on the validation slice. "Three quantile models" fits one early-stopped model per level with the quantile loss.

**The distributional heads on a table** go through `task="custom"` with the head in [`examples/table_heads.py`](../examples/table_heads.py). The target is standardized on the training rows first.

**The embedding table** also reports a longer schedule for Ashlar: learning rate `3e-3`, up to 100 epochs, patience 20. The defaults stop after a few epochs on that small, wide table. The schedule was picked by hand on split 0 and is reported on all three splits.

**Timing** is wall-clock on a 4-core cloud VM with several runs side by side, and the trees used one thread each. Read it as an order of magnitude.

**What is not here.** No hyperparameter search on either side. No TabM recipe: `tabm()` at its paper size is too slow for these CPU runs. No NGBoost or other distributional boosting library.

## Data

| Table | Source | Rows | Target | Notes |
|---|---|---:|---|---|
| California housing | [ageron/handson-ml2](https://github.com/ageron/handson-ml2) | 20,640 | median house value, $100k | Capped at 5.0. `ocean_proximity` is a string column. |
| Diamonds | [mwaskom/seaborn-data](https://github.com/mwaskom/seaborn-data) | 53,940 | log price; price in $1,000 for the distribution runs | The distribution runs use a fixed 12,000-row sample. |
| Adult income | [jbrownlee/Datasets](https://github.com/jbrownlee/Datasets) | 48,842 | income over $50k | Eight string columns. |
| Ames housing | [wblakecannon/ames](https://github.com/wblakecannon/ames) | 2,930 | log sale price | 79 columns, 43 of them strings. |
| Abalone | [jbrownlee/Datasets](https://github.com/jbrownlee/Datasets) | 4,177 | rings | |
| NYC taxi trips | [mwaskom/seaborn-data](https://github.com/mwaskom/seaborn-data) | 6,433 | tip in $, or log trip minutes | Tips leave out payment and total. Duration leaves out distance and fare. |
| NYC flights, 2013 | [byuidatascience/data4python4ds](https://github.com/byuidatascience/data4python4ds) (nycflights13) | 327,346 | arrival delay, minutes | Only what is known before departure: date, scheduled times, distance, carrier, origin, destination, and tail number (4,037 levels). |
| US flights, Jan–Jun 2001 | [vega/vega-datasets](https://github.com/vega/vega-datasets) `flights-3m` | 3,000,000 | arrival delay, minutes | Scheduled time, distance, origin, destination. Scored on a fixed 100,000-row sample of each test split. |
| Melbourne daily minimum temperature | [jbrownlee/Datasets](https://github.com/jbrownlee/Datasets) | 3,650 | °C, one day ahead | 28-day window. |

The large flight tables use a batch of 512 (327k rows) or 2,048 (3M rows) and a learning rate of `2e-3` for every Ashlar head, set before any test score was seen, so an epoch is a few hundred steps instead of tens of thousands. The 3-million-row table has one split, seed 0: two more ran out of memory on this VM beside the other runs, and were not repeated.

## Point accuracy

### Abalone, rings

4,177 rows, 3 splits.

| Model | RMSE | Seconds |
|---|---:|---:|
| Ashlar, default | 2.145 ± 0.087 | 16 |
| LightGBM, default | 2.236 ± 0.058 | 0.1 |
| LightGBM, early-stopped | 2.231 ± 0.072 | 0.1 |
| XGBoost, default | 2.302 ± 0.047 | 0.1 |
| XGBoost, early-stopped | 2.225 ± 0.060 | 0.3 |

### Adult income, over $50k

48,842 rows, 3 splits.

| Model | Accuracy | Log loss | Seconds |
|---|---:|---:|---:|
| Ashlar, default | 0.859 ± 0.002 | 0.305 ± 0.003 | 90 |
| LightGBM, default | 0.877 ± 0.001 | 0.271 ± 0.002 | 0.7 |
| LightGBM, early-stopped | 0.877 ± 0.002 | 0.271 ± 0.001 | 1.8 |
| XGBoost, default | 0.874 ± 0.001 | 0.277 ± 0.002 | 1.1 |
| XGBoost, early-stopped | 0.876 ± 0.002 | 0.271 ± 0.001 | 3.4 |

### Ames housing, log sale price

2,930 rows, 3 splits.

| Model | RMSE | Seconds |
|---|---:|---:|
| Ashlar, default | 0.119 ± 0.007 | 13 |
| LightGBM, default | 0.120 ± 0.002 | 0.4 |
| LightGBM, early-stopped | 0.120 ± 0.003 | 1.0 |
| XGBoost, default | 0.129 ± 0.007 | 0.9 |
| XGBoost, early-stopped | 0.122 ± 0.004 | 5.2 |

### California housing, median value in $100k

20,640 rows, 3 splits.

| Model | RMSE | Seconds |
|---|---:|---:|
| Ashlar, default | 0.563 ± 0.002 | 98 |
| LightGBM, default | 0.475 ± 0.006 | 0.3 |
| LightGBM, early-stopped | 0.457 ± 0.009 | 2.7 |
| XGBoost, default | 0.473 ± 0.008 | 0.7 |
| XGBoost, early-stopped | 0.464 ± 0.010 | 5.6 |

### Diamonds, log price

53,940 rows, 3 splits.

| Model | RMSE | Seconds |
|---|---:|---:|
| Ashlar, default | 0.108 ± 0.009 | 213 |
| LightGBM, default | 0.090 ± 0.001 | 0.5 |
| LightGBM, early-stopped | 0.085 ± 0.002 | 7.6 |
| XGBoost, default | 0.087 ± 0.001 | 0.6 |
| XGBoost, early-stopped | 0.085 ± 0.002 | 6.3 |

## Distributions on smaller tables

### Abalone, rings

4,177 rows, 3 splits.

| Model | CRPS | Pinball | 80% coverage | 80% width | NLL | MAE | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ashlar, `flow` | 1.071 ± 0.029 | 0.480 ± 0.012 | 77% | 4.218 ± 0.043 | 1.972 ± 0.077 | 1.481 ± 0.035 | 19 |
| Ashlar, `gaussian` | 1.109 ± 0.039 | 0.497 ± 0.019 | 83% | 5.078 ± 0.130 | 2.056 ± 0.042 | 1.535 ± 0.042 | 9.6 |
| Ashlar, `histogram` | 1.064 ± 0.033 | 0.476 ± 0.015 | 82% | 4.825 ± 0.158 | 1.921 ± 0.003 | 1.478 ± 0.037 | 17 |
| Ashlar, `mixture` | 1.055 ± 0.027 | 0.471 ± 0.011 | 77% | 4.458 ± 0.069 | 1.922 ± 0.025 | 1.472 ± 0.026 | 20 |
| Ashlar, squared error | — | — | — | — | — | 1.514 ± 0.034 | 17 |
| Ashlar, `quantile` | — | 0.477 ± 0.014 | 79% | 4.550 ± 0.140 | — | 1.487 ± 0.044 | 15 |
| LightGBM mean + constant normal | 1.161 ± 0.028 | 0.528 ± 0.012 | 83% | 5.075 ± 0.076 | 2.237 ± 0.038 | 1.546 ± 0.038 | — |
| LightGBM, three quantile models | — | 0.503 ± 0.015 | 72% | 4.070 ± 0.150 | — | 1.535 ± 0.035 | — |
| XGBoost mean + constant normal | 1.157 ± 0.021 | 0.526 ± 0.009 | 83% | 5.081 ± 0.098 | 2.234 ± 0.032 | 1.539 ± 0.024 | — |
| XGBoost, three quantile models | — | 0.512 ± 0.017 | 75% | 4.180 ± 0.075 | — | 1.559 ± 0.040 | — |

### California housing, median value in $100k

20,640 rows, 3 splits.

| Model | CRPS | Pinball | 80% coverage | 80% width | NLL | MAE | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ashlar, `flow` | 0.267 ± 0.009 | 0.120 ± 0.004 | 81% | 1.192 ± 0.084 | 0.408 ± 0.044 | 0.367 ± 0.013 | 80 |
| Ashlar, `gaussian` | 0.287 ± 0.006 | 0.129 ± 0.003 | 85% | 1.374 ± 0.031 | 0.669 ± 0.035 | 0.397 ± 0.008 | 68 |
| Ashlar, `histogram` | 0.271 ± 0.003 | 0.122 ± 0.001 | 87% | 1.375 ± 0.036 | 0.618 ± 0.015 | 0.374 ± 0.007 | 88 |
| Ashlar, `mixture` | 0.267 ± 0.013 | 0.119 ± 0.006 | 82% | 1.187 ± 0.052 | 0.394 ± 0.074 | 0.374 ± 0.017 | 70 |
| Ashlar, squared error | — | — | — | — | — | 0.388 ± 0.001 | 82 |
| Ashlar, `quantile` | — | 0.122 ± 0.000 | 79% | 1.187 ± 0.043 | — | 0.375 ± 0.002 | 76 |
| LightGBM mean + constant normal | 0.236 ± 0.003 | 0.108 ± 0.001 | 87% | 1.178 ± 0.020 | 0.638 ± 0.020 | 0.303 ± 0.004 | — |
| LightGBM, three quantile models | — | 0.098 ± 0.001 | 70% | 0.848 ± 0.016 | — | 0.288 ± 0.002 | — |
| XGBoost mean + constant normal | 0.238 ± 0.002 | 0.108 ± 0.001 | 87% | 1.182 ± 0.020 | 0.652 ± 0.022 | 0.305 ± 0.002 | — |
| XGBoost, three quantile models | — | 0.103 ± 0.001 | 71% | 0.873 ± 0.019 | — | 0.301 ± 0.002 | — |

### Diamonds, price in $1,000, 12,000-row sample

12,000 rows, 3 splits.

| Model | CRPS | Pinball | 80% coverage | 80% width | NLL | MAE | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ashlar, `flow` | 0.260 ± 0.007 | 0.118 ± 0.003 | 80% | 1.014 ± 0.025 | 0.231 ± 0.008 | 0.350 ± 0.010 | 53 |
| Ashlar, `gaussian` | 0.240 ± 0.001 | 0.107 ± 0.000 | 81% | 1.116 ± 0.050 | 0.133 ± 0.040 | 0.335 ± 0.002 | 45 |
| Ashlar, `histogram` | 0.354 ± 0.004 | 0.159 ± 0.002 | 95% | 2.251 ± 0.025 | 1.036 ± 0.011 | 0.437 ± 0.010 | 48 |
| Ashlar, `mixture` | 0.228 ± 0.005 | 0.102 ± 0.002 | 80% | 1.000 ± 0.039 | 0.085 ± 0.030 | 0.317 ± 0.006 | 46 |
| Ashlar, squared error | — | — | — | — | — | 0.336 ± 0.002 | 48 |
| Ashlar, `quantile` | — | 0.104 ± 0.002 | 81% | 1.066 ± 0.027 | — | 0.325 ± 0.006 | 51 |
| LightGBM mean + constant normal | 0.286 ± 0.012 | 0.130 ± 0.006 | 90% | 1.607 ± 0.026 | 0.914 ± 0.077 | 0.310 ± 0.013 | — |
| LightGBM, three quantile models | — | 0.100 ± 0.005 | 69% | 0.854 ± 0.045 | — | 0.292 ± 0.011 | — |
| XGBoost mean + constant normal | 0.274 ± 0.006 | 0.124 ± 0.003 | 91% | 1.551 ± 0.103 | 0.884 ± 0.033 | 0.295 ± 0.007 | — |
| XGBoost, three quantile models | — | 0.096 ± 0.003 | 74% | 0.836 ± 0.021 | — | 0.285 ± 0.010 | — |

### NYC taxi tips, $, payment type hidden

6,383 rows, 3 splits.

| Model | CRPS | Pinball | 80% coverage | 80% width | NLL | MAE | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ashlar, `flow` | 0.927 ± 0.007 | 0.402 ± 0.004 | 80% | 3.363 ± 0.103 | 0.325 ± 0.066 | 1.381 ± 0.019 | 28 |
| Ashlar, `gaussian` | 0.950 ± 0.020 | 0.409 ± 0.005 | 76% | 4.004 ± 0.166 | 1.763 ± 0.007 | 1.441 ± 0.054 | 12 |
| Ashlar, `histogram` | 0.910 ± 0.025 | 0.396 ± 0.013 | 80% | 3.734 ± 0.169 | 1.429 ± 0.004 | 1.359 ± 0.025 | 14 |
| Ashlar, `mixture` | 0.952 ± 0.001 | 0.413 ± 0.004 | 85% | 3.926 ± 0.026 | 0.455 ± 0.273 | 1.570 ± 0.014 | 13 |
| Ashlar, squared error | — | — | — | — | — | 1.401 ± 0.021 | 16 |
| Ashlar, `quantile` | — | 0.379 ± 0.006 | 77% | 3.792 ± 0.019 | — | 1.323 ± 0.028 | 17 |
| LightGBM mean + constant normal | 1.001 ± 0.034 | 0.460 ± 0.016 | 86% | 4.690 ± 0.066 | 2.076 ± 0.038 | 1.377 ± 0.044 | — |
| LightGBM, three quantile models | — | 0.371 ± 0.006 | 85% | 3.563 ± 0.067 | — | 1.323 ± 0.022 | — |
| XGBoost mean + constant normal | 1.046 ± 0.019 | 0.479 ± 0.009 | 88% | 4.998 ± 0.156 | 2.132 ± 0.008 | 1.438 ± 0.022 | — |
| XGBoost, three quantile models | — | 0.412 ± 0.010 | 79% | 3.429 ± 0.053 | — | 1.385 ± 0.031 | — |

## Distributions on large tables

### NYC flights 2013, arrival delay in minutes

327,346 rows, 3 splits.

| Model | CRPS | Pinball | 80% coverage | 80% width | NLL | MAE | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ashlar, `flow` | 18.821 ± 0.237 | 8.637 ± 0.116 | 78% | 65.639 ± 0.263 | 4.828 ± 0.019 | 24.736 ± 0.268 | 89 |
| Ashlar, `gaussian` | 20.657 ± 0.204 | 9.604 ± 0.102 | 91% | 107.180 ± 4.943 | 5.153 ± 0.020 | 27.118 ± 0.345 | 42 |
| Ashlar, `histogram` | 18.738 ± 0.196 | 8.600 ± 0.094 | 84% | 80.138 ± 0.876 | 4.708 ± 0.004 | 26.301 ± 0.395 | 102 |
| Ashlar, `mixture` | 18.658 ± 0.195 | 8.557 ± 0.096 | 79% | 73.668 ± 2.352 | 4.669 ± 0.007 | 26.273 ± 0.610 | 76 |
| Ashlar, squared error | — | — | — | — | — | 26.430 ± 0.182 | 62 |
| Ashlar, `quantile` | — | 8.574 ± 0.103 | 79% | 73.150 ± 1.956 | — | 24.754 ± 0.248 | 54 |
| LightGBM mean + constant normal | 19.012 ± 0.192 | 8.742 ± 0.092 | 90% | 99.580 ± 0.654 | 5.104 ± 0.018 | 24.053 ± 0.187 | — |
| LightGBM, three quantile models | — | 7.964 ± 0.108 | 77% | 65.663 ± 0.242 | — | 22.974 ± 0.246 | — |
| XGBoost mean + constant normal | 20.452 ± 0.166 | 9.419 ± 0.082 | 90% | 107.313 ± 0.618 | 5.172 ± 0.015 | 25.646 ± 0.188 | — |
| XGBoost, three quantile models | — | 8.491 ± 0.096 | 77% | 68.915 ± 0.191 | — | 24.630 ± 0.230 | — |

### US flights 2001, arrival delay in minutes, scored on 100,000 test rows

3,000,000 rows, 1 split.

| Model | CRPS | Pinball | 80% coverage | 80% width | NLL | MAE | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ashlar, `flow` | 12.056 | 5.536 | 76% | 41.363 | 4.367 | 15.822 | 603 |
| Ashlar, `gaussian` | 13.446 | 6.238 | 90% | 70.733 | 4.712 | 17.504 | 478 |
| Ashlar, `histogram` | 11.391 | 5.229 | 85% | 49.538 | 4.214 | 15.823 | 478 |
| Ashlar, `mixture` | 11.648 | 5.331 | 79% | 47.379 | 4.179 | 16.680 | 549 |
| Ashlar, squared error | — | — | — | — | — | 16.587 | 496 |
| Ashlar, `quantile` | — | 5.196 | 78% | 44.096 | — | 15.163 | 431 |
| LightGBM mean + constant normal | 11.454 | 5.290 | 92% | 64.804 | 4.633 | 14.082 | — |
| LightGBM, three quantile models | — | 4.659 | 78% | 39.695 | — | 13.797 | — |
| XGBoost mean + constant normal | 11.016 | 5.081 | 93% | 62.975 | 4.604 | 13.373 | — |
| XGBoost, three quantile models | — | 4.784 | 79% | 41.300 | — | 13.943 | — |

## Learned embeddings

### NYC taxi trips, log minutes, from zones without distance

6,343 rows, 3 splits.

| Model | RMSE | Seconds |
|---|---:|---:|
| Ashlar, 8-d embeddings, default | 0.687 ± 0.016 | 13 |
| Ashlar, 16-d embeddings, longer | 0.555 ± 0.017 | 34 |
| Ashlar, 8-d embeddings, longer | 0.588 ± 0.025 | 38 |
| Ashlar, one-hot, default | 0.688 ± 0.017 | 8.9 |
| Ashlar, one-hot, longer | 0.563 ± 0.014 | 30 |
| LightGBM, early-stopped | 0.520 ± 0.003 | 0.5 |
| XGBoost, early-stopped | 0.580 ± 0.002 | 1.2 |

## Forecast heads

### Melbourne daily minimum temperature, °C, one day ahead

3,650 rows, 3 splits.

| Model | CRPS | Pinball | 80% coverage | 80% width | NLL | MAE | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ashlar, `flow` | 1.248 ± 0.006 | 0.556 ± 0.002 | 84% | 6.029 ± 0.056 | 2.230 ± 0.005 | 1.754 ± 0.010 | 43 |
| Ashlar, `gaussian` | 1.264 ± 0.005 | 0.564 ± 0.002 | 85% | 6.348 ± 0.022 | 2.240 ± 0.005 | 1.756 ± 0.007 | 45 |
| Ashlar, `histogram` | 1.262 ± 0.004 | 0.562 ± 0.002 | 87% | 6.616 ± 0.034 | 2.284 ± 0.003 | 1.745 ± 0.006 | 45 |
| Ashlar, `mixture` | 1.243 ± 0.004 | 0.554 ± 0.001 | 83% | 6.052 ± 0.089 | 2.221 ± 0.002 | 1.726 ± 0.004 | 35 |
| Ashlar, squared error | — | — | — | — | — | 1.746 ± 0.003 | 43 |
| Ashlar, `quantile` | — | 0.566 ± 0.002 | 83% | 6.171 ± 0.085 | — | 1.797 ± 0.008 | 48 |
| LightGBM on the same lags | — | 0.573 ± 0.005 | 77% | 5.492 ± 0.058 | — | 1.771 ± 0.014 | — |
