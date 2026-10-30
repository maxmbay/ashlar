# Benchmarks

[Ashlar](../README.md) beside LightGBM, XGBoost, CatBoost, and other neural packages. The large-table numbers come from [`examples/benchmark_large.py`](../examples/benchmark_large.py), the small-table numbers from [`examples/benchmark_readme.py`](../examples/benchmark_readme.py), and the ablations from [`examples/ablation_gbm_parity.py`](../examples/ablation_gbm_parity.py).

```bash
pip install -e ".[dataframes]" xgboost lightgbm catboost openml pyarrow pytabkit
JAX_PLATFORMS=cpu python examples/benchmark_large.py --tables higgs covertype --models ashlar lightgbm --seeds 0
python examples/benchmark_large.py --report --seeds 0
JAX_PLATFORMS=cpu python examples/benchmark_readme.py --stage all
python examples/benchmark_readme.py --stage report
JAX_PLATFORMS=cpu python examples/benchmark_uncertainty.py --stage all
python examples/benchmark_uncertainty.py --stage report
```

- [How it was run](#how-it-was-run)
- [Large tables](#large-tables)
- [Beside other neural packages](#beside-other-neural-packages)
- [What each change buys](#what-each-change-buys)
- [Point accuracy on small tables](#point-accuracy)
- [Distributions on smaller tables](#distributions-on-smaller-tables)
- [Learned embeddings](#learned-embeddings)
- [Forecast heads](#forecast-heads)
- [Calibrated intervals](#calibrated-intervals)
- [Count heads](#count-heads)
- [Member uncertainty](#member-uncertainty)

## How it was run

**Versions.** LightGBM 4.7.0, XGBoost 3.4.1, CatBoost 1.2.10, PyTabKit 1.7.3, JAX 0.12.

**Hardware.** An Apple M3 Pro laptop with 12 cores and 18 GB, CPU only. Several runs shared the machine. Ashlar (JAX) used every core it could get; the trees ran on one thread each; PyTabKit models ran on four threads. Read the seconds as an order of magnitude.

**Large tables.** Eight tables from OpenML, each capped at 2,000,000 rows by a fixed random sample. One random 80/20 split per table (seed 0) so far; more seeds are running. Every model holds out the same share, 20% of the training rows, to stop early. The trees run at learning rate 0.05 with up to 4,000 trees and 100 rounds of patience, with native categorical handling. Ashlar is `ASRRegressor()`, `ASRClassifier()`, or `ASRRegressor(task="mixture")` with no other arguments. The distributional scores are on a fixed sample of 100,000 test rows; point and classification scores use every test row.

**Small tables.** Each table is split 80/20 at random, three times, with seeds 0, 1, and 2. A cell is the mean over the three test sets ± the standard deviation across them. The temperature forecast is split on the clock instead, the last 20% of the series, so its three runs differ only in the seed. XGBoost and LightGBM "default" is the library's constructor defaults; "early-stopped" is a learning rate of 0.05 and up to 4,000 trees, stopped after 100 rounds without improvement on a 20% validation slice of the training rows. A category the training rows never saw is missing to the trees, as it is index 0 to Ashlar.

**Distributional scores** are on the original scale of the target.

- *CRPS*, the continuous ranked probability score, is the expected absolute error of a draw minus half the expected distance between two draws. It scores the whole distribution, in the target's units, and stays finite on targets with a spike. Ashlar's is from 200 (large tables) or 400 (small tables) draws per row; a normal's is exact.
- *Pinball* is the quantile loss averaged over the 0.1, 0.5, and 0.9 levels. For a density head the levels are taken from the same draws.
- *80% coverage* is the share of held-out values between the 0.1 and 0.9 levels, and *80% width* is the average distance between them.
- *NLL* is the negative log density per row. On targets with a spike, such as taxi tips that are exactly zero or California values capped at 5, a head can drive it down by making the spike narrow. Read CRPS there.
- The trees have no density of their own. "Mean + constant normal" is the early-stopped mean with a normal whose scale is the residual standard deviation on the validation slice. "Three quantile models" fits one early-stopped model per level with the quantile loss. CatBoost's `RMSEWithUncertainty` fits a mean and a variance per row and is scored as that normal.

**What is not here.** No hyperparameter search on any side. Flight tables with a date column were dropped: under a random split a model can learn which days had bad weather and meet those same days again in the test set.

## Data

| Table | Source | Rows | Target | Notes |
|---|---|---:|---|---|
| Higgs | [OpenML 45664](https://www.openml.org/d/45664) | 2,000,000 of 5,100,000 | signal or background | 28 numeric columns. |
| Covertype | [OpenML 1596](https://www.openml.org/d/1596) | 581,012 | 7 forest cover types | 10 numeric columns and 44 binary indicators, read as categories. |
| Year prediction (MSD) | [OpenML 44027](https://www.openml.org/d/44027) | 515,345 | release year | 90 timbre features. |
| Zurich public transport | [OpenML 45034](https://www.openml.org/d/45034) | 2,000,000 of 5,465,575 | delay | Weather as observed (temperature, wind, rain, dew point, humidity) and time of day. A model of delay given the weather, not a forecast. |
| Airline delay | [OpenML 42728](https://www.openml.org/d/42728) | 2,000,000 of 10,000,000 | departure delay, minutes | Month, weekday, scheduled times, carrier, origin, destination, distance. Day of month dropped. Schedule only, no weather. |
| Click prediction | [OpenML 1218](https://www.openml.org/d/1218) | 1,997,410 | clicked | KDD Cup 2012 ad ids, read as numbers. |
| Airlines | [OpenML 1169](https://www.openml.org/d/1169) | 539,383 | delayed | Carrier, flight, airports, weekday, time, length. Schedule only. |
| NYC green taxi, Dec 2016 | [OpenML 42729](https://www.openml.org/d/42729) | 581,835 | tip, $ | Pickup and drop-off zones, rate code, total amount, and times. |
| California housing | [ageron/handson-ml2](https://github.com/ageron/handson-ml2) | 20,640 | median house value, $100k | Capped at 5.0. `ocean_proximity` is a string column. |
| Diamonds | [mwaskom/seaborn-data](https://github.com/mwaskom/seaborn-data) | 53,940 | log price; price in $1,000 for the distribution runs | The distribution runs use a fixed 12,000-row sample. |
| Adult income | [jbrownlee/Datasets](https://github.com/jbrownlee/Datasets) | 48,842 | income over $50k | Eight string columns. |
| Ames housing | [wblakecannon/ames](https://github.com/wblakecannon/ames) | 2,930 | log sale price | 79 columns, 43 of them strings. |
| Abalone | [jbrownlee/Datasets](https://github.com/jbrownlee/Datasets) | 4,177 | rings | |
| NYC taxi trips | [mwaskom/seaborn-data](https://github.com/mwaskom/seaborn-data) | 6,433 | tip in $, or log trip minutes | Tips leave out payment and total. Duration leaves out distance and fare. |
| Melbourne daily minimum temperature | [jbrownlee/Datasets](https://github.com/jbrownlee/Datasets) | 3,650 | °C, one day ahead | 28-day window. |

## Large tables

One split each (seed 0). Ashlar's seconds include every core of the laptop; each tree used one thread.

### Zurich public transport, delay, 2,000,000 of 5,465,575 rows

| Model | Splits | RMSE | MAE | CRPS | NLL | 80% coverage | 80% width | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Ashlar, default | 1 | 3.052 | 2.56 | — | — | — | — | 521 |
| Ashlar, `task="mixture"` | 1 | 3.05 | 2.558 | 1.621 | 2.005 | 79% | 8.03 | 536 |
| LightGBM | 1 | 3.047 | 2.554 | 1.747 | 2.534 | 79% | 7.803 | 35 |
| XGBoost | 1 | 3.047 | 2.553 | 1.747 | 2.534 | 79% | 7.804 | 59 |
| CatBoost | 1 | 3.047 | 2.553 | 1.747 | 2.534 | 79% | 7.804 | 351 |
| CatBoost, `RMSEWithUncertainty` | 1 | 3.047 | 2.554 | 1.742 | 2.531 | 80% | 7.781 | 552 |

### Airline delay, departure delay in minutes, 2,000,000 of 10,000,000 rows

| Model | Splits | RMSE | MAE | CRPS | NLL | 80% coverage | 80% width | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Ashlar, default | 1 | 29.04 | 13.89 | — | — | — | — | 751 |
| Ashlar, `task="mixture"` | 1 | 29.11 | 13.94 | 9.167 | 2.66 | 80% | 32.8 | 1438 |
| LightGBM | 1 | 29.06 | 13.91 | 12.2 | 4.813 | 94% | 74.87 | 12 |
| XGBoost | 1 | 29.11 | 13.92 | 12.21 | 4.813 | 94% | 74.93 | 25 |
| CatBoost | 1 | 29.05 | 13.9 | 12.17 | 4.812 | 94% | 74.91 | 880 |
| CatBoost, `RMSEWithUncertainty` | 1 | 29.06 | 13.92 | 11.46 | 4.722 | 94% | 66.82 | 626 |

### NYC green taxi, tip in $, 581,835 rows

| Model | Splits | RMSE | MAE | CRPS | NLL | 80% coverage | 80% width | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Ashlar, default | 1 | 1.977 | 0.872 | — | — | — | — | 190 |
| Ashlar, `tabm_piecewise()` | 1 | 1.979 | 0.8747 | — | — | — | — | 924 |
| Ashlar, `task="mixture"` | 1 | 1.977 | 0.8641 | 0.5935 | -1.973 | 82% | 2.814 | 520 |
| LightGBM | 1 | 1.752 | 0.6307 | 0.6583 | 1.978 | 95% | 4.464 | 52 |
| XGBoost | 1 | 1.869 | 0.6452 | 0.6877 | 2.055 | 95% | 4.726 | 177 |
| CatBoost | 1 | 1.758 | 0.6414 | 0.6633 | 1.995 | 95% | 4.528 | 1546 |
| CatBoost, `RMSEWithUncertainty` | 1 | 1.745 | 0.6455 | 0.4647 | 0.7008 | 83% | 1.952 | 2345 |
| RealMLP (pytabkit) | 1 | 1.934 | 0.892 | — | — | — | — | 4044 |
| TabM (pytabkit) | 1 | 1.798 | 0.7327 | — | — | — | — | 15169 |

### Year prediction (MSD), release year, 515,345 rows

| Model | Splits | RMSE | MAE | CRPS | NLL | 80% coverage | 80% width | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Ashlar, default | 1 | 8.698 | 5.903 | — | — | — | — | 323 |
| Ashlar, `task="mixture"` | 1 | 8.699 | 5.835 | 4.051 | 3.081 | 77% | 16.81 | 282 |
| LightGBM | 1 | 8.754 | 6.09 | 4.621 | 3.588 | 85% | 22.33 | 309 |
| XGBoost | 1 | 8.71 | 6.085 | 4.609 | 3.583 | 85% | 22.22 | 736 |
| CatBoost | 1 | 8.776 | 6.123 | 4.637 | 3.591 | 85% | 22.4 | 271 |
| CatBoost, `RMSEWithUncertainty` | 1 | 8.883 | 6.197 | 4.407 | 3.416 | 83% | 19.01 | 220 |

### Higgs, signal against background, 2,000,000 of 5,100,000 rows

| Model | Splits | Accuracy | AUC | Log loss | Seconds |
|---|---:|---:|---:|---:|---:|
| Ashlar, default | 1 | 0.7625 | 0.8464 | 0.4823 | 507 |
| LightGBM | 1 | 0.7493 | 0.8324 | 0.5002 | 261 |
| XGBoost | 1 | 0.7516 | 0.8347 | 0.497 | 391 |
| CatBoost | 1 | 0.7503 | 0.8328 | 0.5 | 855 |

### Covertype, 7 forest types, 581,012 rows

| Model | Splits | Accuracy | AUC | Log loss | Seconds |
|---|---:|---:|---:|---:|---:|
| Ashlar, default | 1 | 0.967 | 0.9988 | 0.09441 | 415 |
| LightGBM | 1 | 0.9628 | 0.9983 | 0.1165 | 361 |
| XGBoost | 1 | 0.9477 | 0.9973 | 0.1488 | 930 |
| CatBoost | 1 | 0.9102 | 0.9929 | 0.2474 | 692 |

### Airlines, delayed or not, 539,383 rows

| Model | Splits | Accuracy | AUC | Log loss | Seconds |
|---|---:|---:|---:|---:|---:|
| Ashlar, default | 1 | 0.6601 | 0.71 | 0.6143 | 267 |
| LightGBM | 1 | 0.6655 | 0.7185 | 0.6085 | 13 |
| XGBoost | 1 | 0.6642 | 0.7181 | 0.6089 | 14 |
| CatBoost | 1 | 0.6675 | 0.7215 | 0.6062 | 1011 |

### Click prediction, clicked or not, 1,997,410 rows

| Model | Splits | Accuracy | AUC | Log loss | Seconds |
|---|---:|---:|---:|---:|---:|
| Ashlar, default | 1 | 0.8421 | 0.7382 | 0.3943 | 365 |
| LightGBM | 1 | 0.8444 | 0.7544 | 0.3858 | 132 |
| XGBoost | 1 | 0.8442 | 0.7557 | 0.3852 | 188 |
| CatBoost | 1 | 0.8435 | 0.7502 | 0.3879 | 726 |

## Beside other neural packages

PyTabKit 1.7.3's tuned-default recipes (`RealMLP_TD`, `TabM_D`, `FTT_D`, `MLP_PLR_D`) on one split (seed 0), missing numbers filled with the training median because PyTabKit refuses them, four threads each. Same split as the ablations: adult is not stratified here. The taxi column is the large-table split above; `tabm_piecewise()` ran there before the automatic epoch budget grew, with 10 epochs.

| Model | California RMSE ↓ | Diamonds RMSE ↓ | Adult accuracy ↑ | NYC green taxi tips RMSE ↓ |
|---|---:|---:|---:|---:|
| Ashlar, default | 0.475 (28 s) | 0.0876 (2 min) | 0.864 (27 s) | 1.977 (3 min) |
| Ashlar, `tabm_piecewise()` | **0.440** (4 min) | 0.0842 (8 min) | **0.869** (4 min) | 1.979 (15 min) |
| TabM (PyTabKit) | 0.453 (11 min) | **0.0797** (49 min) | 0.863 (8 min) | **1.798** (4.2 h) |
| RealMLP (PyTabKit) | 0.472 (13 min) | 0.0882 (11 min) | 0.868 (4 min) | 1.934 (67 min) |
| FT-Transformer (PyTabKit) | 0.466 (11 min) | 0.0853 (38 min) | 0.864 (14 min) | — |
| MLP-PLR (PyTabKit) | 0.489 (2 min) | 0.0896 (5 min) | 0.862 (2 min) | — |
| LightGBM, early-stopped | 0.446 (2 s) | 0.0841 (6 s) | 0.875 (1 s) | 1.752 (1 min) |

PyTabKit's TabM is the strongest of the other packages and wins diamonds and the taxi tips; it is also the slowest by far on a CPU. `tabm_piecewise()` wins California and adult. The Ashlar default matches RealMLP and FT-Transformer in a tenth of the time or less.

## What each change buys

Three random splits of California, diamonds (log price), and adult, the same splits as the small-table comparison below except that adult is not stratified here. Each row adds one change to the row above. Mean ± standard deviation across the splits. Seconds are for one fit.

| Step | California RMSE ↓ | Diamonds RMSE ↓ | Adult accuracy ↑ | California seconds |
|---|---:|---:|---:|---:|
| Old default: 2 blocks of 64, batch 64, constant `1e-3`, 50 epochs | 0.562 ± 0.003 | 0.107 ± 0.008 | 0.856 ± 0.003 | 4 |
| Wider: 3 blocks of 256, dropout 0.1, batch 256, up to 100 epochs | 0.521 ± 0.004 | 0.102 ± 0.007 | 0.857 ± 0.004 | 55 |
| + cosine schedule at `2e-3` | 0.511 ± 0.005 | 0.094 ± 0.003 | 0.857 ± 0.004 | 41 |
| Wider + piecewise-linear embeddings, constant `1e-3` | 0.471 ± 0.007 | 0.089 ± 0.002 | 0.860 ± 0.005 | 38 |
| **New default**: wider + piecewise + cosine `2e-3` + auto budget | 0.472 ± 0.003 | 0.089 ± 0.003 | 0.861 ± 0.003 | 39 |
| `tabm_piecewise()`: 32-member TabM on piecewise embeddings of quantile-normalized columns | **0.447 ± 0.006** | **0.085 ± 0.000** | 0.865 ± 0.003 | 263 |

The piecewise-linear embedding of each numeric column is the largest single gain: it takes California from 0.52 to 0.47 and diamonds from 0.102 to 0.089. Width and the cosine schedule buy most of the rest. Once the embeddings are in, the schedule matters little on these tables; it is the default because it converges inside a fixed budget on large tables, where early stopping rarely fires. The TabM ensemble buys another 5% on California at about seven times the cost.

Variations that did not help, three splits each, on top of the new default:

| Variation | California | Diamonds | Adult |
|---|---:|---:|---:|
| Periodic embeddings instead of piecewise (constant rate) | 0.513 ± 0.004 | 0.096 ± 0.006 | 0.860 ± 0.004 |
| ReLU on the piecewise embedding | 0.470 ± 0.006 | 0.092 ± 0.005 | 0.860 ± 0.003 |
| Hidden layer per categorical column, 16-d categories | 0.465 ± 0.007 | 0.090 ± 0.005 | 0.860 ± 0.003 |

Single-split screens (seed 0) of batch size 1,024, width 512, dropout 0 or 0.2, embedding width 32, weight decay `1e-2`, and quantile scaling all landed within 0.006 of the default on California, inside the spread between splits. On one split the spread from initialization alone is about ±0.015 RMSE on California, so a single-split difference smaller than that is noise.

The large tables changed the epoch budget. With 10 epochs at a million rows, covertype reached 0.897; with 40 it reached 0.969. The automatic budget is now about 20 million row passes, at least 20 and at most 100 epochs.


## Point accuracy

Three random splits of five small tables. Ashlar is `ASRRegressor()` or `ASRClassifier()`.

### Abalone, rings

4,177 rows, 3 splits.

| Model | RMSE | Seconds |
|---|---:|---:|
| Ashlar, default | 2.166 ± 0.080 | 6.8 |
| LightGBM, default | 2.236 ± 0.058 | 0.1 |
| LightGBM, early-stopped | 2.231 ± 0.072 | 0.1 |
| XGBoost, default | 2.302 ± 0.047 | 0.1 |
| XGBoost, early-stopped | 2.225 ± 0.060 | 0.1 |

### Adult income, over $50k

48,842 rows, 3 splits.

| Model | Accuracy | Log loss | Seconds |
|---|---:|---:|---:|
| Ashlar, default | 0.864 ± 0.002 | 0.296 ± 0.003 | 62 |
| LightGBM, default | 0.877 ± 0.001 | 0.271 ± 0.002 | 0.6 |
| LightGBM, early-stopped | 0.877 ± 0.002 | 0.271 ± 0.001 | 1.4 |
| XGBoost, default | 0.874 ± 0.001 | 0.277 ± 0.002 | 0.6 |
| XGBoost, early-stopped | 0.876 ± 0.002 | 0.271 ± 0.001 | 1.9 |

### Ames housing, log sale price

2,930 rows, 3 splits.

| Model | RMSE | Seconds |
|---|---:|---:|
| Ashlar, default | 0.117 ± 0.011 | 8.7 |
| LightGBM, default | 0.120 ± 0.002 | 0.3 |
| LightGBM, early-stopped | 0.120 ± 0.003 | 0.6 |
| XGBoost, default | 0.129 ± 0.007 | 0.4 |
| XGBoost, early-stopped | 0.122 ± 0.004 | 3.0 |

### California housing, median value in $100k

20,640 rows, 3 splits.

| Model | RMSE | Seconds |
|---|---:|---:|
| Ashlar, default | 0.472 ± 0.003 | 38 |
| LightGBM, default | 0.475 ± 0.006 | 0.2 |
| LightGBM, early-stopped | 0.457 ± 0.009 | 2.1 |
| XGBoost, default | 0.473 ± 0.008 | 0.3 |
| XGBoost, early-stopped | 0.464 ± 0.010 | 3.8 |

### Diamonds, log price

53,940 rows, 3 splits.

| Model | RMSE | Seconds |
|---|---:|---:|
| Ashlar, default | 0.089 ± 0.002 | 138 |
| LightGBM, default | 0.090 ± 0.001 | 0.5 |
| LightGBM, early-stopped | 0.085 ± 0.002 | 6.1 |
| XGBoost, default | 0.087 ± 0.001 | 0.5 |
| XGBoost, early-stopped | 0.085 ± 0.002 | 4.4 |

## Distributions on smaller tables

### Abalone, rings

4,177 rows, 3 splits.

| Model | CRPS | Pinball | 80% coverage | 80% width | NLL | MAE | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ashlar, `flow` | 1.088 ± 0.046 | 0.488 ± 0.021 | 78% | 4.341 ± 0.202 | 2.020 ± 0.065 | 1.491 ± 0.057 | 8.2 |
| Ashlar, `gaussian` | 1.105 ± 0.025 | 0.494 ± 0.010 | 84% | 5.098 ± 0.251 | 2.059 ± 0.027 | 1.530 ± 0.038 | 6.7 |
| Ashlar, `histogram` | 1.089 ± 0.033 | 0.487 ± 0.015 | 81% | 4.847 ± 0.104 | 1.942 ± 0.008 | 1.520 ± 0.029 | 6.0 |
| Ashlar, `mixture` | 1.076 ± 0.039 | 0.481 ± 0.018 | 78% | 4.398 ± 0.063 | 1.941 ± 0.033 | 1.498 ± 0.036 | 7.6 |
| Ashlar, squared error | — | — | — | — | — | 1.528 ± 0.035 | 7.0 |
| Ashlar, `quantile` | — | 0.479 ± 0.015 | 79% | 4.584 ± 0.180 | — | 1.498 ± 0.055 | 8.6 |
| LightGBM mean + constant normal | 1.161 ± 0.028 | 0.528 ± 0.012 | 83% | 5.075 ± 0.076 | 2.237 ± 0.038 | 1.546 ± 0.038 | — |
| LightGBM, three quantile models | — | 0.503 ± 0.015 | 72% | 4.070 ± 0.150 | — | 1.535 ± 0.035 | — |
| XGBoost mean + constant normal | 1.157 ± 0.021 | 0.526 ± 0.009 | 83% | 5.081 ± 0.098 | 2.234 ± 0.032 | 1.539 ± 0.024 | — |
| XGBoost, three quantile models | — | 0.512 ± 0.017 | 75% | 4.180 ± 0.075 | — | 1.559 ± 0.040 | — |

### California housing, median value in $100k

20,640 rows, 3 splits.

| Model | CRPS | Pinball | 80% coverage | 80% width | NLL | MAE | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ashlar, `flow` | 0.208 ± 0.002 | 0.094 ± 0.001 | 74% | 0.786 ± 0.007 | 0.120 ± 0.009 | 0.285 ± 0.002 | 58 |
| Ashlar, `gaussian` | 0.225 ± 0.003 | 0.101 ± 0.001 | 83% | 1.014 ± 0.012 | 0.403 ± 0.018 | 0.311 ± 0.005 | 38 |
| Ashlar, `histogram` | 0.229 ± 0.003 | 0.103 ± 0.002 | 86% | 1.122 ± 0.035 | 0.479 ± 0.008 | 0.310 ± 0.002 | 32 |
| Ashlar, `mixture` | 0.205 ± 0.001 | 0.092 ± 0.001 | 76% | 0.812 ± 0.035 | 0.076 ± 0.037 | 0.287 ± 0.002 | 51 |
| Ashlar, squared error | — | — | — | — | — | 0.312 ± 0.006 | 40 |
| Ashlar, `quantile` | — | 0.094 ± 0.001 | 75% | 0.812 ± 0.007 | — | 0.288 ± 0.001 | 52 |
| LightGBM mean + constant normal | 0.236 ± 0.003 | 0.108 ± 0.001 | 87% | 1.178 ± 0.020 | 0.638 ± 0.020 | 0.303 ± 0.004 | — |
| LightGBM, three quantile models | — | 0.098 ± 0.001 | 70% | 0.848 ± 0.016 | — | 0.288 ± 0.002 | — |
| XGBoost mean + constant normal | 0.238 ± 0.002 | 0.108 ± 0.001 | 87% | 1.182 ± 0.020 | 0.652 ± 0.022 | 0.305 ± 0.002 | — |
| XGBoost, three quantile models | — | 0.103 ± 0.001 | 71% | 0.874 ± 0.019 | — | 0.301 ± 0.002 | — |

### Diamonds, price in $1,000, 12,000-row sample

12,000 rows, 3 splits.

| Model | CRPS | Pinball | 80% coverage | 80% width | NLL | MAE | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ashlar, `flow` | 0.232 ± 0.014 | 0.106 ± 0.006 | 78% | 0.861 ± 0.083 | 0.071 ± 0.117 | 0.307 ± 0.020 | 25 |
| Ashlar, `gaussian` | 0.216 ± 0.013 | 0.097 ± 0.006 | 84% | 1.126 ± 0.185 | -0.015 ± 0.094 | 0.294 ± 0.012 | 24 |
| Ashlar, `histogram` | 0.343 ± 0.003 | 0.154 ± 0.002 | 95% | 2.177 ± 0.007 | 1.007 ± 0.011 | 0.427 ± 0.005 | 17 |
| Ashlar, `mixture` | 0.202 ± 0.007 | 0.091 ± 0.003 | 80% | 0.907 ± 0.050 | -0.111 ± 0.056 | 0.283 ± 0.010 | 27 |
| Ashlar, squared error | — | — | — | — | — | 0.298 ± 0.009 | 23 |
| Ashlar, `quantile` | — | 0.090 ± 0.002 | 80% | 0.901 ± 0.038 | — | 0.274 ± 0.007 | 27 |
| LightGBM mean + constant normal | 0.286 ± 0.012 | 0.130 ± 0.006 | 90% | 1.607 ± 0.026 | 0.914 ± 0.077 | 0.310 ± 0.013 | — |
| LightGBM, three quantile models | — | 0.100 ± 0.005 | 69% | 0.854 ± 0.045 | — | 0.292 ± 0.011 | — |
| XGBoost mean + constant normal | 0.274 ± 0.006 | 0.124 ± 0.003 | 91% | 1.551 ± 0.103 | 0.884 ± 0.033 | 0.295 ± 0.007 | — |
| XGBoost, three quantile models | — | 0.096 ± 0.003 | 74% | 0.836 ± 0.021 | — | 0.284 ± 0.010 | — |

### NYC taxi tips, $, payment type hidden

6,383 rows, 3 splits.

| Model | CRPS | Pinball | 80% coverage | 80% width | NLL | MAE | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ashlar, `flow` | 0.958 ± 0.012 | 0.420 ± 0.006 | 74% | 3.033 ± 0.207 | -0.016 ± 0.404 | 1.403 ± 0.017 | 19 |
| Ashlar, `gaussian` | 0.950 ± 0.019 | 0.406 ± 0.007 | 79% | 4.038 ± 0.151 | 1.767 ± 0.018 | 1.443 ± 0.037 | 6.1 |
| Ashlar, `histogram` | 0.917 ± 0.017 | 0.398 ± 0.009 | 81% | 3.762 ± 0.129 | 1.416 ± 0.021 | 1.370 ± 0.033 | 7.2 |
| Ashlar, `mixture` | 0.872 ± 0.017 | 0.372 ± 0.005 | 84% | 3.689 ± 0.116 | 0.092 ± 0.386 | 1.420 ± 0.045 | 15 |
| Ashlar, squared error | — | — | — | — | — | 1.402 ± 0.025 | 8.2 |
| Ashlar, `quantile` | — | 0.384 ± 0.004 | 74% | 3.759 ± 0.154 | — | 1.337 ± 0.031 | 7.2 |
| LightGBM mean + constant normal | 1.001 ± 0.034 | 0.460 ± 0.016 | 86% | 4.690 ± 0.066 | 2.076 ± 0.038 | 1.377 ± 0.044 | — |
| LightGBM, three quantile models | — | 0.371 ± 0.006 | 85% | 3.563 ± 0.067 | — | 1.323 ± 0.022 | — |
| XGBoost mean + constant normal | 1.046 ± 0.019 | 0.479 ± 0.009 | 88% | 4.998 ± 0.156 | 2.132 ± 0.008 | 1.438 ± 0.022 | — |
| XGBoost, three quantile models | — | 0.412 ± 0.010 | 79% | 3.429 ± 0.053 | — | 1.385 ± 0.031 | — |

## Learned embeddings

### NYC taxi trips, log minutes, from zones without distance

6,343 rows, 3 splits.

| Model | RMSE | Seconds |
|---|---:|---:|
| Ashlar, 8-d embeddings, default | 0.681 ± 0.015 | 6.6 |
| Ashlar, 16-d embeddings, longer | 0.694 ± 0.008 | 8.8 |
| Ashlar, 8-d embeddings, longer | 0.682 ± 0.014 | 7.2 |
| Ashlar, one-hot, default | 0.651 ± 0.045 | 7.6 |
| Ashlar, one-hot, longer | 0.580 ± 0.017 | 14 |
| LightGBM, early-stopped | 0.520 ± 0.003 | 0.4 |
| XGBoost, early-stopped | 0.580 ± 0.002 | 1.0 |

## Forecast heads

### Melbourne daily minimum temperature, °C, one day ahead

3,650 rows, 3 splits.

| Model | CRPS | Pinball | 80% coverage | 80% width | NLL | MAE | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ashlar, `flow` | 1.248 ± 0.006 | 0.556 ± 0.002 | 84% | 6.029 ± 0.056 | 2.230 ± 0.005 | 1.755 ± 0.010 | 19 |
| Ashlar, `gaussian` | 1.264 ± 0.005 | 0.564 ± 0.002 | 85% | 6.348 ± 0.022 | 2.240 ± 0.005 | 1.756 ± 0.007 | 21 |
| Ashlar, `histogram` | 1.262 ± 0.004 | 0.562 ± 0.002 | 87% | 6.616 ± 0.034 | 2.284 ± 0.003 | 1.745 ± 0.006 | 18 |
| Ashlar, `mixture` | 1.243 ± 0.004 | 0.554 ± 0.001 | 83% | 6.052 ± 0.089 | 2.221 ± 0.002 | 1.726 ± 0.004 | 16 |
| Ashlar, squared error | — | — | — | — | — | 1.746 ± 0.003 | 19 |
| Ashlar, `quantile` | — | 0.564 ± 0.005 | 84% | 6.309 ± 0.058 | — | 1.779 ± 0.020 | 15 |
| LightGBM on the same lags | — | 0.573 ± 0.005 | 77% | 5.492 ± 0.058 | — | 1.771 ± 0.014 | — |

## Calibrated intervals

From [`examples/benchmark_uncertainty.py`](../examples/benchmark_uncertainty.py), stage `calibration`. This script and its three sections ran on a 4-core Linux VM with 15 GB, CPU only, LightGBM 4.7.0 on one thread, not on the laptop above. Each table is split 60% train, 20% calibration, 20% test, three times (seeds 0, 1, 2). Ashlar is `ASRRegressor(task="mixture")` or `ASRRegressor(task="quantile")` with no other arguments, then `calibrate(X_cal, y_cal)`. LightGBM is two early-stopped quantile models at 0.1 and 0.9 on the same training rows, then the same conformal step on the same calibration rows (conformalized quantile regression). Each cell is the test coverage of a nominal 80% interval, mean over three splits, before → after.

| Table | Rows | Ashlar `mixture` | Ashlar `quantile` | LightGBM quantiles |
|---|---:|---:|---:|---:|
| California housing | 20,640 | 75.9% → 80.7% | 77.3% → 80.4% | 68.9% → 79.4% |
| Diamonds, price in $1,000 | 53,940 | 79.4% → 79.4% | 80.4% → 79.5% | 73.8% → 79.1% |
| Bike rentals per hour | 17,379 | 76.8% → 79.7% | 77.5% → 79.0% | 65.3% → 79.3% |
| Abalone, rings | 4,177 | 80.2% → 80.3% | 78.5% → 78.8% | 76.0% → 79.8% |
| NYC taxi tips, $ | 6,383 | 88.6% → 79.9% | 80.8% → 80.1% | 84.4% → 84.4% |

Calibration moves every Ashlar interval to within a point of 80% and leaves an interval that was already right alone (diamonds, abalone). It also narrows one that was too wide: the taxi mixture covered 89% and calibration took it to 80%, from $3.98 to $3.72 wide on average. The 90% mixture interval behaves the same way, 86.9% → 89.9% on bike rentals and 87.3% → 90.6% on California.

Conformal calibration is not Ashlar's alone: the same step brings LightGBM's quantile models to 79–80% on four tables. The difference is width. After calibration the Ashlar mixture's 80% interval is narrower than LightGBM's on California (0.95 against 1.00, in $100k), diamonds (0.82 against 0.83), and bike rentals (74 against 87 rentals), and slightly wider on abalone (4.71 against 4.59 rings). On taxi tips CQR cannot move LightGBM: a third of tips are exactly $0, both quantile models put their 0.1 level there, so the conformal margin lands on a tie at zero and the trees keep their 84%.

A smaller table gives a noisier calibration: with 835 calibration rows on abalone the calibrated coverage of single splits ranges from 77.8% to 82.9%.

## Count heads

From the same script, stage `counts`. Four count targets, each split 80/20 three times. Every Ashlar model is `ASRRegressor(task=...)` with no other arguments; `gaussian` and `mixture` see the count as a number. LightGBM is early-stopped with the Poisson objective, and scored as a Poisson with that mean. CRPS and coverage are from 400 draws per row; NLL is the negative log mass of the count that happened, so it is only defined for the models that give a mass.

| Target | Rows | Mean | Variance | Zeros |
|---|---:|---:|---:|---:|
| Bike rentals per hour, all riders ([Capital Bikeshare via UCI](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/project-bikesharing/Bike-Sharing-Dataset)) | 17,379 | 189 | 32,900 | 0% |
| Bike rentals per hour, casual riders only | 17,379 | 36 | 2,431 | 9% |
| NYC taxi pickups per half hour, from the previous day and the same slot a week earlier ([NAB](https://github.com/numenta/NAB)) | 9,984 | 15,198 | 4.8 × 10⁷ | 0% |
| Abalone rings | 4,177 | 9.9 | 10.4 | 0% |

| Target | Model | CRPS | NLL | 80% coverage | 80% width |
|---|---|---:|---:|---:|---:|
| Bike rentals, all | Ashlar `negative_binomial` | 16.55 ± 0.51 | **4.50** | 80% | 71 |
| | Ashlar zero-inflated `negative_binomial` | 16.71 ± 0.30 | **4.50** | 81% | 72 |
| | Ashlar `poisson` | 18.44 ± 0.52 | 6.60 | 53% | 31 |
| | Ashlar `mixture` | **16.34 ± 0.44** | — | 76% | 64 |
| | Ashlar `gaussian` | 17.56 ± 0.40 | — | 79% | 76 |
| | LightGBM, Poisson objective | 18.43 ± 0.23 | 6.52 | 54% | 31 |
| Bike rentals, casual | Ashlar zero-inflated `negative_binomial` | **5.98 ± 0.09** | **3.28** | 82% | 25 |
| | Ashlar `negative_binomial` | 6.05 ± 0.14 | **3.28** | 82% | 26 |
| | Ashlar `poisson` | 6.38 ± 0.03 | 4.10 | 61% | 12 |
| | Ashlar `mixture` | 6.08 ± 0.09 | — | 78% | 25 |
| | Ashlar `gaussian` | 6.25 ± 0.10 | — | 82% | 28 |
| | LightGBM, Poisson objective | 6.26 ± 0.07 | 4.07 | 62% | 12 |
| Taxi pickups | Ashlar `negative_binomial` | 314.7 ± 5.5 | **7.83** | 76% | 1,231 |
| | Ashlar zero-inflated `negative_binomial` | 322.2 ± 13.3 | **7.79** | 79% | 1,372 |
| | Ashlar `poisson` | 378.1 ± 9.3 | 19.43 | 25% | 304 |
| | Ashlar `mixture` | **310.8 ± 2.1** | — | 75% | 1,176 |
| | Ashlar `gaussian` | 321.4 ± 7.0 | — | 79% | 1,341 |
| | LightGBM, Poisson objective | 397.8 ± 5.9 | 21.12 | 24% | 304 |
| Abalone rings | Ashlar `negative_binomial` | 1.189 ± 0.027 | **2.247** | 97% | 7.9 |
| | Ashlar `poisson` | 1.194 ± 0.026 | 2.249 | 96% | 7.9 |
| | Ashlar `mixture` | **1.075 ± 0.036** | — | 78% | 4.6 |
| | Ashlar `gaussian` | 1.098 ± 0.030 | — | 84% | 5.1 |
| | LightGBM, Poisson objective | 1.208 ± 0.022 | 2.258 | 95% | 7.9 |

Bold is the best in its column per target; NLL ties within 0.01.

- **Against a Poisson, the negative binomial wins every overdispersed target by a wide margin.** It has the best log mass on all four, 4.50 against LightGBM's 6.52 on bike rentals and 7.8 against 21.1 on taxi pickups. A Poisson's variance equals its mean, so its 80% interval holds only 24% to 62% of these counts; the negative binomial's holds 76% to 82%. The same holds against Ashlar's own `poisson` head, so the gain is the family, not the network.
- **Against the continuous heads it is close.** The `mixture` has the best CRPS on bike totals (16.34 against 16.55) and taxi pickups (310.8 against 314.7), within about one standard deviation across splits; the zero-inflated negative binomial has the best on casual riders (5.98 against 6.08). What the count heads add is a real probability for each count: whole-number draws and quantiles, a log mass you can compare across models, and the 9% of zero hours in the casual counts, which the negative binomial reproduces (9.2% of its draws are 0) and the mixture does not (0.0%).
- **On abalone the count heads lose.** Rings vary less than their mean once the shell measurements are known, and neither a Poisson nor a negative binomial can be narrower than a Poisson. Their 80% intervals hold 96% of rings and are 70% wider than the mixture's. Use `gaussian` or `mixture` for an underdispersed count.
- **Zero inflation helps only where there are zeros.** It ties the plain negative binomial on log mass everywhere and helps CRPS only on the casual counts.

## Member uncertainty

From the same script, stage `uncertainty`. The model is `tabm_piecewise(task="gaussian")`, a TabM backbone with 32 members, and `predict_uncertainty(X_test, n_samples=200)`. Two tests per table:

- **Ranking.** An 80/20 split. Each score ranks the test rows; the table gives its Spearman correlation with the absolute error of the point prediction, and the RMSE of the half of the rows the score is surest of. A random ranking keeps about the full RMSE.
- **Shift.** Train on rows at or below the 80th percentile of one column, then compare held-out rows inside that range with every row above it. A score that knows what the model has not seen should rise above the range.

These fits are slow on 4 CPU cores (about an hour per split), so this section has fewer splits than the others: two for California (seeds 0 and 1) and one for diamonds and bike rentals (seed 0). Treat the single-split numbers as one draw, not a mean. `python examples/benchmark_uncertainty.py --stage uncertainty` runs all three seeds and skips any already saved.

| Table | Rows | Test RMSE | Spearman with abs error: epistemic / aleatoric / total | RMSE of surest half: epistemic / aleatoric / random |
|---|---:|---:|---:|---:|
| California housing, $100k, 2 splits | 20,640 | 0.445 | 0.48 / 0.52 / 0.53 | 0.242 / 0.228 / 0.452 |
| Diamonds, price in $1,000, 20,000-row sample, 1 split | 20,000 | 0.572 | 0.70 / 0.72 / 0.72 | 0.111 / 0.103 / 0.574 |
| Bike rentals per hour, 1 split | 17,379 | 39.2 | 0.62 / 0.65 / 0.65 | 16.0 / 15.6 / 40.2 |

| Table | Shift column | RMSE inside → above | Mean epistemic inside → above | Mean aleatoric inside → above |
|---|---|---:|---:|---:|
| California housing | median income | 0.440 → 0.884 | 0.084 → 0.211 | 0.379 → 0.587 |
| Diamonds | carat | 0.290 → 6.01 | 0.055 → 0.223 | 0.220 → 0.462 |
| Bike rentals | temperature | 40.7 → 60.1 | 9.8 → 15.5 | 30.8 → 45.4 |

- **Every score ranks errors well.** Keeping the half of the rows the model is surest of cuts the RMSE by half on California, by 60% on bike rentals, and by more than 80% on diamonds, where error grows with price.
- **In range, the noise ranks errors slightly better than the member spread.** Aleatoric beats epistemic on all three tables, by 0.02 to 0.04 in Spearman. On rows like the training rows most of the error is noise, and the noise head already sizes it.
- **Out of range, the member spread rises most.** Above the training range the mean epistemic grows 2.5 times on California, 4 times on diamonds and 1.6 times on bike rentals, against 1.5, 2.1 and 1.5 times for the aleatoric part. That is the signal `epistemic` exists for: the members agree on what they saw and drift apart on what they did not.
- **It flags a shift; it does not size the error.** On diamonds above 80th-percentile carat the RMSE grows 21 times and the epistemic part only 4 times. Use the spread to find rows to distrust, not as an interval. Calibrated intervals (above) are for exchangeable rows, and these rows are not.
