# CPU benchmark

One seed on two public Grinsztajn tables, scored on a shared outer test partition. Ashlar uses its default recipe and one `search=True` run. XGBoost uses a constructor with no arguments. DeepTab and PyTabKit use their own architectures and training defaults, with the epoch ceiling set to 50 so the neural fits stay on the same inexpensive CPU budget as Ashlar's default `max_epochs`.

These numbers are a single-machine observation. They are not a leaderboard, and they are not evidence that a neural net beats trees or that trees beat neural nets.

## Hardware

Linux 6.12.94+, x86_64, 4 CPUs, Intel Xeon. `JAX_PLATFORMS=cpu`, JAX device `cpu:0`. PyTorch reports CUDA unavailable. No GPU was used.

## Versions

Ashlar environment (`/tmp/gnet-venv`):

| Package | Version |
|---|---|
| Python | 3.12.3 |
| ashlar-tabular | 0.1.0 |
| jax / jaxlib | 0.12.0.1 |
| equinox | 0.13.8 |
| optax | 0.2.8 |
| scikit-learn | 1.9.1 |
| xgboost | 3.1.2 |
| numpy | 2.5.4 |
| pandas | 3.1.0 |

Torch environment (`/tmp/deeptab-venv`):

| Package | Version |
|---|---|
| Python | 3.12.3 |
| deeptab | 2.1.0 |
| pytabkit | 1.7.3 |
| torch | 2.14.1+cu130 |
| pytorch-lightning | 2.6.6 |
| scikit-learn | 1.9.1 |
| numpy | 2.5.4 |
| pandas | 2.3.3 |

Nothing was registered or uploaded.

## Datasets and split provenance

Both files downloaded from the Hugging Face release of the Grinsztajn transformed matrices, `inria-soda/tabular-benchmark`. The download was not blocked. The transformed Higgs classification table (OpenML [44129](https://www.openml.org/d/44129), 940160 rows) was left undownloaded because it is outside this CPU budget. A `wine_quality` CSV was fetched with that download. It is scored later, with `cpu_act` and `sulfur`, under the unshared TabM recipe.

The split is the medium-size procedure in appendix A.3 of Grinsztajn, Kadra, Varoquaux, and Müller, as implemented by `data_to_train_test` in [LeoGrin/tabular-benchmark](https://github.com/LeoGrin/tabular-benchmark) `src/generate_dataset_pipeline.py`:

- Train fraction `min(10000 / n, 0.70)`.
- Of the remainder, 30% is the paper validation partition and 70% is the test partition (`val_test_prop=0.3`).
- Validation and test caps of 50,000 rows do not bind here.
- `numpy.random.RandomState(0)`, one fold. Appendix A.3 uses one fold when the test partition has more than 6,000 rows.
- The same `RandomState` object is passed to both `train_test_split` calls, which matches that function.

This is the paper's split procedure on the transformed CSV. It is not a stored OpenML task split, and it is not a fresh split of the raw California Housing table from scikit-learn.

The paper validation partition is scored below and was not used to train or to choose a model.

| Dataset | Task | Source | Rows in file | Train / paper-val / test | Features | Target |
|---|---|---|---|---|---|---|
| houses | regression | [HF `reg_num/houses.csv`](https://huggingface.co/datasets/inria-soda/tabular-benchmark/resolve/main/reg_num/houses.csv), transformed OpenML [44138](https://www.openml.org/d/44138), original OpenML [537](https://www.openml.org/d/537) | 20640 | 10000 / 3192 / 7448 | 8 | `medianhousevalue` |
| california | classification | [HF `clf_num/california.csv`](https://huggingface.co/datasets/inria-soda/tabular-benchmark/resolve/main/clf_num/california.csv), transformed OpenML [45028](https://www.openml.org/d/45028), original description at the StatLib California housing page | 20634 | 10000 / 3190 / 7444 | 8 | `price_above_median` |

`houses` is the numerical-regression form of California Housing after the paper's transform. The target is not in units of $100,000. `california` is the numerical-classification form: a binary label for price above the median, on a transformed feature matrix. The two files do not share a row order.

## Metric

Section 3.4 of the paper uses test **R²** for regression and test **accuracy** for classification. Those are the numbers in the test columns. Regression RMSE is also reported, on the transformed target, so the scale is visible. Accuracy is the unweighted fraction of exact label matches on the named partition.

## How each fit was configured

**Ashlar regularized default.** `tabular()` on houses and `classifier()` on california, with no extra arguments, as of the run that filled the tables above. That path used weight decay `1e-4`, feature dropout `0.1`, ReLU, and early stopping on a 20% slice of the train partition (`patience=10`, `max_epochs=50`). The backbone was the residual MLP with hidden dropout 0, median impute, standardization, batch size 64, learning rate `1e-3`, and `random_state=0`. Houses ran 50 epochs. California ran 30 epochs. An earlier fit of the same split with weight decay 0 and no feature dropout scored test R² 0.7428 (search 0.7657) and test accuracy 0.8704. A later houses sweep, recorded below, changed the no-argument path to feature dropout `0` and SiLU. The tables above were not refit after that change.

**Ashlar search.** The same constructors with `fit(..., search=True)`. `DEFAULT_SEARCH` is still two trials of two epochs, then a full train. The two trials compare weight decay `1e-4` and `1e-3`. Learning rate and feature dropout stay at the constructor values. On houses the short-trial validation losses were 1.985 and 1.842, so the final train used weight decay `0.001` and ran 50 epochs. On california the weighted accuracies were 0.834 and 0.832, so the tie rule kept weight decay `1e-4`, and the final train ran 30 epochs. The constructor weight decay stayed `1e-4`. Omitting `search` does not run the grid.

**XGBoost 3.1.2.** `XGBRegressor()` and `XGBClassifier()` with no arguments. Unset hyperparameters stay at the library defaults. `get_num_boosting_rounds` uses 100 trees when `n_estimators` is unset. `early_stopping_rounds` is unset, so there is no early stopping. The objective stored on the estimator is `reg:squarederror` or `binary:logistic`. `random_state` is unset. The fit sees all 10,000 train rows. The paper's own tree protocol (early stopping, a much larger estimator cap, and a random search) was not used.

**DeepTab 2.1.0.** `MLPRegressor` and `MLPClassifier` with the library `MLPConfig` (default `layer_sizes=[256, 128, 32]`; the houses regressor logged 50.1K parameters). `fit` received `max_epochs=50`, `random_state=0`, and a checkpoint directory. Every other trainer setting stayed at `TrainerConfig`: patience 15, batch size 128, learning rate `1e-4`, weight decay `1e-6`, `val_size=0.2`. The houses log says training stopped because `max_epochs=50` was reached. DeepTab's default of 100 epochs, and any longer published budget, was not run.

**PyTabKit 1.7.3.** `RealMLP_TD_Regressor` / `RealMLP_TD_Classifier` and `TabM_D_Regressor` / `TabM_D_Classifier`. Passed arguments: `device="cpu"`, `random_state=0`, `n_epochs=50`, `n_cv=1`, `n_refit=0`, `n_repeats=1`, `verbosity=0`. Everything else is the class default, including `val_fraction=0.2`. RealMLP's own default epoch count is 256. TabM's own default is an epoch cap of one billion with patience 16, batch size 256, learning rate `2e-3`, dropout 0.1, and `tabm_k=32`. Both were stopped by the shared cap of 50. Hyperparameter-search classes (`*_HPO_*`) were not run.

Ashlar, DeepTab, and PyTabKit each draw their own internal validation rows from the train partition. Those slices are not the same rows. XGBoost does not draw one.

## Test scores and wall time

Wall time is `fit` plus `predict` on the paper validation partition and the test partition. Ashlar calls `jax.block_until_ready` on the predictions, so the clock includes compilation. The first Ashlar predict on each dataset is the slower one (0.54 s and 0.64 s). Later predicts on an already compiled shape are about 0.01 s.

### houses, test R² (Grinsztajn regression metric)

| Model | Test R² | Test RMSE, transformed target | Paper-val R² | Wall seconds |
|---|---:|---:|---:|---:|
| Ashlar regularized default | 0.7322 | 0.2962 | 0.7508 | 22.06 |
| Ashlar `search=True` | 0.7299 | 0.2975 | 0.7503 | 22.70 |
| XGBoost defaults | 0.8268 | 0.2382 | 0.8296 | 0.09 |
| DeepTab MLP, 50-epoch cap | 0.1832 | 0.5173 | 0.2217 | 15.43 |
| PyTabKit RealMLP-TD, 50-epoch cap | 0.8247 | 0.2397 | 0.8322 | 8.31 |
| PyTabKit TabM-D, 50-epoch cap | 0.8366 | 0.2314 | 0.8406 | 247.73 |

### california, test accuracy (Grinsztajn classification metric)

| Model | Test accuracy | Paper-val accuracy | Wall seconds |
|---|---:|---:|---:|
| Ashlar regularized default | 0.8657 | 0.8646 | 13.54 |
| Ashlar `search=True` | 0.8657 | 0.8646 | 15.30 |
| XGBoost defaults | 0.8968 | 0.8978 | 0.09 |
| DeepTab MLP, 50-epoch cap | 0.8478 | 0.8555 | 14.91 |
| PyTabKit RealMLP-TD, 50-epoch cap | 0.8853 | 0.8843 | 7.92 |
| PyTabKit TabM-D, 50-epoch cap | 0.8916 | 0.8903 | 254.28 |

On california the two Ashlar fits produced the same test accuracy. The search kept weight decay `1e-4` because that short trial had the higher weighted accuracy.

XGBoost was fit again in the same command as the regularized Ashlar rows. Its test R² and test accuracy matched the earlier run exactly, so the XGBoost, DeepTab, RealMLP, and TabM rows are the same baselines as before.

## Quantile map on the regularized residual MLP

Same houses split, same metric, same residual MLP as the regularized-default row: learning rate `1e-3`, weight decay `1e-4`, feature dropout `0.1`, ReLU, hidden dropout 0, batch 64, `max_epochs=50`, patience 10, `random_state=0`. The only change is the numeric map. Mean and population standard deviation are not applied.

The map is scikit-learn `QuantileTransformer` (`n_quantiles=1000`, `output_distribution="normal"`, `subsample` equal to the internal training row count, `random_state=0`), fit on the median-filled numeric columns of Ashlar's internal training partition only. That partition is the 80% left after `validation_size=0.2`, not the paper validation or test rows. The forward is `QuantileTransformer._transform_col`: the average of an ascending and a reversed linear interpolation, then a bound halo of `max(1e-7, one float32 ulp)`, then the normal quantile function, clipped to the SciPy bounds `-5.199337582605575` and `5.19933758270342`. Knots are stored float32. This run used 1000 quantiles. It stopped at epoch 24. Best internal validation loss was 0.11655575782060623. The last internal validation loss was 0.16528528928756714.

| Model | Test R² | Test RMSE | Paper-val R² | Epochs | Wall seconds |
|---|---:|---:|---:|---:|---:|
| Regularized default, mean/std, ReLU, feature dropout 0.1 | 0.7322 | 0.2962 | 0.7508 | 50 | 22.06 |
| Same model, train-only normal quantile map | 0.6519 | 0.3377 | 0.6532 | 24 | 16.34 |

The exact houses test R² is 0.6518968125086434. The hypothesis failed. The score did not stay within 0.01 of 0.7322; it fell by 0.0803, away from the XGBoost R² of 0.8268. A monotone train-only quantile map does not account for the gap on this seed. California was not refit. XGBoost, DeepTab, RealMLP, and TabM were not refit.

## Feature dropout and activation

Same houses split and the same regularized residual MLP, with mean/std scaling left in place. Learning rate `1e-3`, weight decay `1e-4`, hidden dropout 0, batch 64, `max_epochs=50`, patience 10, and `random_state=0` stay fixed. The grid is feature dropout `{0, 0.1}` by activation `{relu, gelu, silu}`. ReLU is the activation the regularized-default row used. The feature-dropout `0.1`, ReLU cell reproduced that row's test R² exactly (0.7322478112213036).

| Feature dropout | Activation | Test R² | Paper-val R² | Epochs | Wall seconds |
|---|---|---:|---:|---:|---:|
| 0 | relu | 0.7340 | 0.7473 | 44 | 16.64 |
| 0 | gelu | 0.7602 | 0.7649 | 38 | 14.70 |
| 0 | silu | 0.7615 | 0.7672 | 50 | 19.95 |
| 0.1 | relu | 0.7322 | 0.7508 | 50 | 21.30 |
| 0.1 | gelu | 0.7188 | 0.7431 | 50 | 21.65 |
| 0.1 | silu | 0.6748 | 0.6909 | 44 | 18.41 |

Exact test R², in that row order: 0.7340272640197965, 0.7602485207169098, 0.7615065490594078, 0.7322478112213036, 0.7188468453307002, 0.6747571522553988.

Feature dropout 0 with ReLU is 0.0018 above 0.7322, inside the 0.01 band. Feature dropout 0.1 with GELU or SiLU is worse than the ReLU default. Feature dropout 0 with GELU is 0.0280 above 0.7322. Feature dropout 0 with SiLU is 0.0293 above 0.7322. Both clear the 0.01 bar. They differ from each other by 0.0013.

The no-argument path now uses feature dropout `0` and SiLU, the higher of those two cells. Weight decay stays `1e-4`. The other backbones stay ReLU. This houses R², 0.7615, is still short of XGBoost at 0.8268. The comparison tables above were not rerun with SiLU.

## Learning rate and numeric embeddings

Same houses split. The base is the no-argument path after the sweep: SiLU residual MLP, feature dropout 0, weight decay `1e-4`, mean/std, batch 64, `max_epochs=50`, patience 10, `random_state=0`, learning rate `1e-3`. The optimizer is AdamW on every row. Each run changes one setting. `linear` is a learning-rate decay inside AdamW, from the stated rate to 0 across the 50-epoch budget (`optax.linear_schedule`). It is not a different optimizer. The embedding run uses the existing `num_embeddings="linear_relu"` at the default width 32 and leaves the learning rate at `1e-3`.

| Change from the SiLU default | Test R² | Paper-val R² | Epochs | Wall seconds |
|---|---:|---:|---:|---:|
| Learning rate `3e-3`, constant | 0.7680 | 0.7830 | 42 | 17.76 |
| Learning rate `3e-4`, constant | 0.7321 | 0.7462 | 50 | 18.74 |
| Learning rate `1e-4`, constant | 0.7122 | 0.7188 | 50 | 18.90 |
| Learning rate `3e-3`, linear decay to 0 | 0.7685 | 0.7793 | 49 | 18.71 |
| `linear_relu` numeric embeddings, learning rate `1e-3` | 0.7452 | 0.7584 | 46 | 18.55 |

Exact test R²: 0.767972358117888, 0.7321469248459602, 0.7122151856337134, 0.7684640285343317, 0.7451792371862184.

The best of these, linear decay from `3e-3`, is 0.0070 above 0.7615. Constant `3e-3` is 0.0065 above. Neither clears 0.01. The two lower constant rates and the linear-ReLU embeddings are below 0.7615. The no-argument default is unchanged: SiLU, feature dropout 0, weight decay `1e-4`, mean/std, learning rate `1e-3`, constant AdamW. The baseline tables above were not refit.

## TabM borrowings

`tabm()` on the same houses split, not the residual-MLP default. The trunk stays the recipe: BatchEnsemble, `k=32`, `d_block=512`, three blocks, hidden dropout 0.1, ReLU, AdamW, learning rate `2e-3`, weight decay 0, constant schedule, batch 256, patience 16, `random_state=0`. `max_epochs` is 50 here, the cap on the scored TabM-D row. The recipe's own cap is 200. All three fits used the full 50 epochs, so patience did not stop them. The residual-MLP default was not changed. The baseline tables above were not refit.

`quantile_tabr` is PyTabKit's `TabrQuantileTransformer` with `random_state=0`: knot count `max(min(n_train // 30, 1000), 10)`, which was **266** on the internal training rows, normal output, subsample `1_000_000_000`, and Gaussian noise on the fit matrix only with per-column scale `1e-3 / max(column_std, 1e-3)`. Transform adds no noise. There is no median imputation and no following z-score. This is not the 1,000-knot map that scored 0.6519.

The label z-score uses the internal-training mean and population standard deviation. This run's mean was 12.088793975114822 and the standard deviation was 0.5593105375431461. Training divides by `std + 1e-30`. `predict` multiplies by `std` and adds the mean. Early stopping is still the member-mean loss, on the z-scored labels.

| `tabm()`, 50-epoch cap | Houses test R² | Versus XGBoost 0.8268 | Versus TabM-D 0.8366 |
|---|---:|---:|---:|
| `quantile_tabr` only | 0.6964 | -0.1304 | -0.1402 |
| Label z-score only | 0.8113 | -0.0155 | -0.0253 |
| Both, shared minibatches | 0.8251 | -0.0017 | -0.0115 |
| Both, distinct shuffle per member | 0.8321 | +0.0053 | -0.0045 |

Exact test R²: 0.6964168473958859, 0.8113209224615071, 0.8250537475579031, 0.8321330464556183. Paper-val R²: 0.7048, 0.8287, 0.8306, 0.8385. Wall seconds: 378, 387, 409, 402. The shared pair lands next to XGBoost and short of the capped TabM-D row. `quantile_tabr` alone is worse than the SiLU residual default.

The last row changes only member batching on the shared pair: `share_training_batches=False`, so each of the 32 members draws its own training permutation. Prediction still averages members on the same rows. AdamW, `quantile_tabr`, the label z-score, and the 50-epoch cap stay as in the 0.8251 fit. It ran all 50 epochs. Houses test R² is 0.8321330464556183, 0.0053 above XGBoost's 0.8268 and 0.0045 below TabM-D's 0.8366. The residual-MLP default was not changed.

## Five seeds of the unshared TabM recipe

The recipe is the last row above: `tabm()` with `quantile_tabr`, a train-only label z-score inverted on `predict`, `share_training_batches=False`, AdamW (learning rate `2e-3`, weight decay 0), and `max_epochs=50`. XGBoost is `XGBRegressor()` with no arguments, fit on the same training rows. The residual-MLP default was not changed.

The Grinsztajn medium protocol stores one fold when the test partition has more than 6000 rows, so there is no multi-fold file to iterate and the published test indices are not a frozen training exclusion for later seeds. This section repeats that same split procedure five times. Seed `s` passes `numpy.random.RandomState(s)` into `grinsztajn_medium_split` and sets the estimator `random_state` to `s`. Each seed therefore has its own train, paper-validation, and test partition. That seed's paper-validation and test rows are not used to train it and are not used to pick a checkpoint. The estimator seed also moves initialization and the internal 20% validation slice. The `quantile_tabr` fit-time noise generator stays at 0, as in the 0.8321 run. The helper's provenance string still says `RandomState(0)`; the seed column below is the generator that was actually passed. Every fit used 266 quantile knots and ran all 50 epochs. Standard deviations are the sample standard deviation (`ddof=1`).

California was not run in this section. The same recipe is not defined for classification: `standardize_target` raises because the label z-score is regression-only, and `share_training_batches=False` raises unless the task is regression. The classification scores later use `tabm(task="classification")`, which passes neither flag.

TabM-D uses the same redraws and the setup that scored 0.8366 on seed 0: `TabM_D_Regressor(device="cpu", random_state=seed, n_epochs=50, n_cv=1, n_refit=0, n_repeats=1, verbosity=0)`. Every other constructor argument stays at the class default, including `val_fraction=0.2`. Ashlar was not retrained. The splits were regenerated with `grinsztajn_medium_split` and `numpy.random.RandomState(seed)` in the same environment as the Ashlar fits. Seed 0's arrays match the cached houses split exactly, and that seed's TabM-D test R² matches the published single-split score 0.8366028121954286.

| Seed | Ashlar test R² | XGBoost test R² | TabM-D test R² | Ashlar − XGBoost | Ashlar − TabM-D |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.8321 | 0.8268 | 0.8366 | +0.0053 | -0.0045 |
| 1 | 0.8326 | 0.8303 | 0.8320 | +0.0023 | +0.0006 |
| 2 | 0.8382 | 0.8342 | 0.8404 | +0.0040 | -0.0022 |
| 3 | 0.8361 | 0.8344 | 0.8390 | +0.0017 | -0.0029 |
| 4 | 0.8315 | 0.8338 | 0.8310 | -0.0022 | +0.0005 |
| Mean | 0.8341 | 0.8319 | 0.8358 | +0.0022 | -0.0017 |
| Std | 0.0029 | 0.0033 | 0.0041 | | |

Exact Ashlar test R²: 0.8321330464556183, 0.8326013971560782, 0.8382055162497992, 0.8361043403882867, 0.8315462498260481. Mean 0.834118110015166, sample std 0.0028942771501172053.

Exact XGBoost test R²: 0.826815147573534, 0.830313556820696, 0.8341584449907107, 0.8343576632836112, 0.8337618466294554. Mean 0.8318813318596014, sample std 0.003277878091313743.

Exact TabM-D test R²: 0.8366028121954286, 0.8320261739965192, 0.8403966442456273, 0.8389681233284765, 0.8310484605114116. Mean 0.8358084428554926, sample std 0.004142136676087085.

Seed 0 matches the single-split row: Ashlar 0.8321330464556183, XGBoost 0.826815147573534, and TabM-D 0.8366028121954286. Ashlar's mean is 0.0022 above XGBoost's mean and 0.0017 below TabM-D's mean on these five splits. Ashlar is higher than TabM-D on seeds 1 and 4 and lower on seeds 0, 2, and 3. The published XGBoost point 0.8268 is the low end of these five XGBoost draws; the five-seed XGBoost mean is 0.8319. This is five redraws of one CPU recipe, not a claim that one library beats another in general.

## Three other numerical regression tables

Same recipe as the unshared houses fit, one seed: `tabm(numeric_scaling="quantile_tabr", standardize_target=True, share_training_batches=False, max_epochs=50, random_state=0)`. AdamW stays at the `tabm()` defaults, learning rate `2e-3` and weight decay 0. XGBoost is `XGBRegressor()` with no arguments. TabM-D is `TabM_D_Regressor(device="cpu", random_state=0, n_epochs=50, n_cv=1, n_refit=0, n_repeats=1, verbosity=0)`, with every other constructor argument at the class default, including `val_fraction=0.2`. The residual-MLP default was not changed. Classification was not run in this section, because the label z-score and `share_training_batches=False` raise unless the task is regression. The classification scores later use a different call.

The tables are three other kept numerical-regression files from the same Hugging Face release. Each split is one call to `grinsztajn_medium_split` with `numpy.random.RandomState(0)`. The paper uses more than one fold when the test partition has 6000 rows or fewer. All three test partitions here are smaller than that, so this is one seed of the split function, not the paper's multi-fold aggregate. The paper-validation rows were scored and were not used to train or to choose a checkpoint. The last column is the OpenML default target: `quality`, `usr`, and `y1`. On sulfur, `y2` is a non-target feature in OpenML 23515.

| Dataset | OpenML | Rows | Train / paper-val / test | Features | Ashlar test R² | XGBoost test R² | TabM-D test R² | Ashlar − XGBoost | Ashlar − TabM-D |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| wine_quality | [287](https://www.openml.org/d/287) | 6497 | 4547 / 585 / 1365 | 11 | 0.4049 | 0.4285 | 0.4118 | -0.0236 | -0.0068 |
| cpu_act | [197](https://www.openml.org/d/197) | 8192 | 5734 / 737 / 1721 | 21 | 0.9852 | 0.9744 | 0.9849 | +0.0107 | +0.0003 |
| sulfur | [23515](https://www.openml.org/d/23515) | 10081 | 7056 / 907 / 2118 | 6 | 0.8498 | 0.7751 | 0.8083 | +0.0747 | +0.0415 |

Exact Ashlar test R²: wine_quality 0.40493251825263477, cpu_act 0.9851878334347116, sulfur 0.8497774405322199. Exact XGBoost: 0.4284963911647759, 0.9744379358531527, 0.7750885241576986. Exact TabM-D: 0.4117637706326597, 0.9848813392378664, 0.8082551699106548.

Ashlar is below both XGBoost and TabM-D on wine_quality. It is above both on cpu_act and on sulfur. The cpu_act gap over TabM-D is 0.0003. Wine stopped at epoch 30 (patience 16). Cpu_act and sulfur ran all 50 epochs. Quantile knots were 121, 152, and 188. Sources: [wine_quality](https://huggingface.co/datasets/inria-soda/tabular-benchmark/resolve/main/reg_num/wine_quality.csv), [cpu_act](https://huggingface.co/datasets/inria-soda/tabular-benchmark/resolve/main/reg_num/cpu_act.csv), [sulfur](https://huggingface.co/datasets/inria-soda/tabular-benchmark/resolve/main/reg_num/sulfur.csv).

Wine quality was fit again on that same split as an ordinal classifier. The score in the table above is still the regression recipe. Here the metric is test R² of the expected label, `predict_score`. The most likely label is a worse R² readout, so it is listed separately and was not used as the comparison.

`ordinal()` is the residual-MLP ordinal recipe: mean/std, learning rate `1e-3`, weight decay `1e-4`, 50 epochs. The TabM row is `tabm(task="ordinal", prediction_policy="mean_probabilities", numeric_scaling="quantile_tabr", max_epochs=50, random_state=0)`. Label z-scoring and unshared member batches stay off, because both raise unless the task is regression. `XGBClassifier()` cannot take quality values 3–9 directly in this build, so those labels were mapped to `0..6` and the probabilities were mapped back before the expectation. That control is a nominal multiclass model plus an expected score, not an ordinal likelihood.

| Wine readout | Test R² |
|---|---:|
| Ordinal residual, expected label | 0.3592 |
| Ordinal residual, most likely label | 0.2265 |
| Ordinal TabM, expected label | 0.4139 |
| Ordinal TabM, most likely label | 0.3258 |
| Ordinal TabM, softmax head, expected label | 0.3957 |
| Ordinal TabM, softmax head, most likely label | 0.2382 |
| Ordinal TabM, weight decay 1, expected label | 0.3800 |
| Ordinal TabM, weight decay 1, most likely label | 0.2810 |
| Ordinal TabM, surface learning rate, expected label | 0.2807 |
| Ordinal TabM, surface learning rate, most likely label | 0.1526 |
| Ordinal TabM, empirical-Bayes prior and binned embeddings, expected label | 0.4029 |
| Ordinal TabM, empirical-Bayes prior and binned embeddings, most likely label | 0.2800 |
| Same prior and embeddings, patience 0, 100 epochs, expected label | 0.4029 |
| Same prior and embeddings, patience 0, 100 epochs, most likely label | 0.2800 |
| Ordinal TabM, late bottleneck width 32, expected label | 0.4186 |
| Ordinal TabM, late bottleneck width 32, most likely label | 0.3112 |
| Ordinal TabM, HL-Gaussian histogram, expected label | 0.4151 |
| Ordinal TabM, HL-Gaussian histogram, most likely label | 0.3053 |
| Ordinal TabM, soft stage histogram, expected label | 0.4178 |
| Ordinal TabM, soft stage histogram, most likely label | 0.3024 |
| `XGBClassifier` expected label | 0.4900 |
| `XGBClassifier` most likely label | 0.3540 |
| Regression TabM from the table above | 0.4049 |
| `XGBRegressor` from the table above | 0.4285 |
| TabM-D from the table above | 0.4118 |

Exact expected-label R²: ordinal residual 0.359192121230657, ordinal TabM 0.41394214246508565, `XGBClassifier` 0.48996919003701056. Exact mode R²: 0.2265180965577408, 0.32575728416920047, 0.35397234731363514. Both ordinal fits ran all 50 epochs. The ordinal TabM expectation is 0.0090 above the regression TabM and 0.0022 above TabM-D. It is 0.0146 below `XGBRegressor` and 0.0760 below the `XGBClassifier` expectation. The residual ordinal expectation is below all of those.

The softmax-head experiment in `wine-architecture.md` uses that same ordinal TabM call with `head="softmax"`. Expected-label test R² is 0.3956767364524363. Most-likely-label test R² is 0.23819329510026555. It ran 31 epochs, so patience 16 stopped it, and the best internal validation loss was 1.003374457359314. That is 0.0183 below the cumulative-logit TabM expectation of 0.41394214246508565, and it does not clear the pre-set bar of 0.01 above that score. The hypothesis fails. The residual-MLP regression default was not changed.

The weight-decay experiment uses that same cumulative ordinal TabM call with only `weight_decay=1.0`. Expected-label test R² is 0.3799832563545966. Most-likely-label test R² is 0.281002356422856. It ran all 50 epochs. The best internal validation loss was 1.4093810319900513. The gap to the `XGBClassifier` expectation of 0.48996919003701056 widened from 0.0760 to 0.1100. Stronger weight decay did not close that gap. The residual-MLP regression default was not changed.

The surface-rate experiment uses that same cumulative call, weight decay still 0, with `learning_rate_schedule="surface"`. Warmup was 15 steps at `1e-4`. The batch curvature was −0.00028266687877476215, so the steepness was the root-mean-square gradient 6.333742931019515e-05. With `n_train` 3637 the unclipped ratio is 5742.260207921903, and the clip sets the AdamW rate to 0.1. Expected-label test R² is 0.28070258803405446. Most-likely-label test R² is 0.15257517245508456. It ran 30 epochs, so patience 16 stopped it, and the best internal validation loss was 1.1453269720077515. That expected-label score is 0.1332 below 0.41394214246508565. It does not exceed that score by 0.01. The residual-MLP regression default was not changed.

The empirical-Bayes run in `wine-architecture.md` keeps that cumulative call and adds a frozen Jeffreys class prior plus learned embeddings of eight train-only quantile bins of `alcohol`, `volatile.acidity`, and `sulphates` (`alcohol_bin`, `volatile.acidity_bin`, `sulphates_bin`). Wine has no stored categorical column. Expected-label test R² is 0.4028560414884015. Most-likely-label test R² is 0.2800294232109789. It ran 30 epochs, so patience 16 stopped it, and the best internal validation loss was 1.044643759727478. That expected-label score is 0.0111 below 0.41394214246508565. It does not exceed that score by 0.01. The gap to the `XGBClassifier` expectation is 0.0871. The residual-MLP regression default was not changed.

That same prior-and-embedding call was fit again with `patience=0` and `max_epochs=50`, so early stopping did not cut the run and the validation split stayed in place. Validation loss at epoch 50 was 1.057012915611267, below 1.0585740804672241 at epoch 40, so training continued through epoch 100. The lowest validation loss remained epoch 14, at 1.044643759727478. The lowest loss after epoch 50 was 1.0482879877090454 at epoch 99. The fitted model is that epoch-14 checkpoint. Expected-label test R² is 0.4028560414884015, the same value as the patience-16 run. Most-likely-label test R² is 0.2800294232109789. Against the ordinal TabM score 0.41394214246508565, this is 0.0111 lower. Against the `XGBClassifier` expectation 0.48996919003701056, the gap is 0.0871. The first 50 validation losses of the 100-epoch fit match the 50-epoch fit. The residual-MLP regression default was not changed.

The late-bottleneck experiment in `wine-architecture.md` uses that same cumulative ordinal TabM call with `bottleneck=True`. That flag inserts a per-member linear map of width 32 and a ReLU after the TabM blocks and before the ordinal head. Expected-label test R² is 0.41855174134115225. Most-likely-label test R² is 0.3111632859910447. It ran all 50 epochs. The best internal validation loss was 1.1645258665084839. That expected-label score is 0.0046 above 0.41394214246508565. The bar for changing the residual default is more than 0.01 above that score, so the residual-MLP regression default was not changed. The gap to the `XGBClassifier` expectation 0.48996919003701056 is 0.0714.

The histogram experiment in `wine-strategy.md` uses that same ordinal TabM call with `head="softmax"` and `histogram_variance=0.5`. The target is a Gaussian of variance 0.5 truncated to the seven quality bins, and the loss is the cross-entropy to that target. Expected-label test R² is 0.4150779471645716. Most-likely-label test R² is 0.30532568671978233. It ran 47 epochs, so patience 16 stopped it, and the best internal validation loss was 1.4194178581237793. That loss is the histogram cross-entropy, not the one-hot negative log-likelihood of the earlier rows. The expected-label score is 0.0011 above 0.41394214246508565. The bar for changing the residual default is more than 0.01 above that score, so the residual-MLP regression default was not changed. The gap to the `XGBClassifier` expectation 0.48996919003701056 is 0.0749.

The soft-stage experiment in `gbm-borrow.md` uses that same cumulative ordinal TabM call with `soft_stages=8`. Eight soft stumps mix the quantile-scaled chemicals, pass the coordinate through a sigmoid threshold, and mix two class-logit leaves. The sum is scaled by shrinkage 0.3, added to the cumulative log-probabilities, and renormalized. Expected-label test R² is 0.417837586328735. Most-likely-label test R² is 0.3024068870841512. It ran all 50 epochs. The best internal validation loss was 1.0000371932983398. That loss is the ordinal negative log-likelihood of the combined probabilities, the same objective as the cumulative run at 1.163618564605713. The expected-label score is 0.0039 above 0.41394214246508565. The bar for changing the residual default is more than 0.01 above that score, so the residual-MLP regression default was not changed. The gap to the `XGBClassifier` expectation 0.48996919003701056 is 0.0721.

The expected-label experiment uses that same cumulative ordinal TabM call with `expectation_weight=1`. The training loss is the mean member negative log-likelihood plus one times the squared error of `probabilities @ class_values`. Those probabilities are the mean of the member probabilities, the same readout as `predict_score`. Both terms use the batch weight reduction. Expected-label test R² is 0.3922491062217387. Most-likely-label test R² is 0.2703000910922083. It ran all 50 epochs. The best internal validation loss was 1.6458075046539307. That loss is the mixed objective, so it is not the likelihood 1.163618564605713 of the cumulative run. The expected-label score is 0.0217 below 0.41394214246508565. The gap to the `XGBClassifier` expectation 0.48996919003701056 is 0.0977. The residual-MLP regression default was not changed.

## Where each model wins

Seven more public tables, one `numpy.random.RandomState(0)` call to `grinsztajn_medium_split` each. The same CPU and the same model protocol as the rows above. Regression uses test R². Classification uses test accuracy. A peaked integer target is also scored as an ordinal model when `ordinal()` can fit: the number in that row is test R² of the expected label, the same readout as the wine ordinal section. The paper-validation partition was scored and was not used to train or to choose a checkpoint. Defaults were not changed. Nothing was registered or uploaded.

The residual column is `tabular()` or `classifier()` or `ordinal()`, with no extra arguments. The TabM column on a regression row is `tabm(numeric_scaling="quantile_tabr", standardize_target=True, share_training_batches=False, max_epochs=50, random_state=0)`. On an ordinal row it is `tabm(task="ordinal", prediction_policy="mean_probabilities", numeric_scaling="quantile_tabr", max_epochs=50, random_state=0)`. That regression recipe raises on classification, so those cells are empty. A later section scores `tabm(task="classification")`, which does not pass either flag. XGBoost is `XGBRegressor()` or `XGBClassifier()` with no arguments. This XGBoost build still needs class ids `0..K-1`, so classifier labels were mapped and mapped back. TabM-D is `TabM_D_Regressor` or `TabM_D_Classifier` with `device="cpu"`, `random_state=0`, `n_epochs=50`, `n_cv=1`, `n_refit=0`, `n_repeats=1`, `verbosity=0`, and every other constructor argument at the class default. The winner is the highest test score on that row.

| Dataset | Task | Rows | Train / paper-val / test | Features | Residual MLP | Ashlar TabM | XGBoost | TabM-D | Winner |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| yprop_4_1 | regression | 8885 | 6219 / 799 / 1867 | 42 | −0.0155 | 0.1013 | −0.0192 | 0.1116 | TabM-D |
| Ailerons | regression | 13750 | 9625 / 1237 / 2888 | 33 | −1.2720 | 0.8426 | 0.8274 | 0.8439 | TabM-D |
| Bike_Sharing_Demand | regression | 17379 | 10000 / 2213 / 5166 | 6 | 0.6584 | 0.6975 | 0.6854 | 0.6893 | Ashlar TabM |
| electricity | classification | 38474 | 10000 / 8542 / 19932 | 7 | 0.7795 |  | 0.8544 | 0.8204 | XGBoost |
| eye_movements | classification | 7608 | 5325 / 684 / 1599 | 20 | 0.5510 |  | 0.6429 | 0.5641 | XGBoost |
| pol | regression | 15000 | 10000 / 1500 / 3500 | 26 | 0.9954 | 0.9905 | 0.9830 | 0.9905 | residual MLP |
| pol | ordinal, expected label | 15000 | 10000 / 1500 / 3500 | 26 | 0.9957 | 0.9889 | 0.9885 | 0.9898 | residual ordinal |
| abalone | regression | 4177 | 2923 / 376 / 878 | 8 | 0.5885 | 0.5709 | 0.4942 | 0.5745 | residual MLP |
| abalone | ordinal, expected label | 4177 | 2923 / 376 / 878 | 8 |  |  | 0.5314 | 0.5534 | TabM-D |

Electricity is the only new test partition with more than 6000 rows, so one fold matches appendix A.3 there. The other six test partitions are smaller, so this is one seed of the split function, as with wine_quality, cpu_act, and sulfur.

`yprop_4_1` is OpenML [416](https://www.openml.org/d/416), transformed [45032](https://www.openml.org/d/45032), target column `oz252`. [Ailerons](https://www.openml.org/d/296) / [44137](https://www.openml.org/d/44137), target `goal`. [Bike_Sharing_Demand](https://www.openml.org/d/42712) / [44142](https://www.openml.org/d/44142), target `count`. [electricity](https://www.openml.org/d/151) / [44120](https://www.openml.org/d/44120), target `class`, with the release balanced at 19237 `DOWN` and 19237 `UP`. [eye_movements](https://www.openml.org/d/1044) / [44130](https://www.openml.org/d/44130), target `label`, balanced 3804 / 3804. [pol](https://www.openml.org/d/201) / [44133](https://www.openml.org/d/44133), target `foo`. [abalone](https://www.openml.org/d/42726) / [45042](https://www.openml.org/d/45042), target `Classnumberofrings`. Abalone comes from the categorical regression file; `Sex` is an integer code in that file, so every model saw 8 numeric columns. Sources are the Hugging Face files named in `examples/benchmark_coverage.py`.

`pol` is the second peaked integer. Its 11 labels are 0, 10, …, 100. Of 15000 rows, 9342 are 0 and 3005 are 100. `ordinal()` fits. Its expected-label test R², 0.9957131362802631, is the highest score on pol, 0.0003 above the residual regressor. `abalone` rings are the unimodal integer: 28 levels from 1 to 29, mode 9 on 689 of 4177 rows. The outer training partition has a single row for rings 1, 24, 25, and 26, so the stratified holdout used for this table raises and both `ordinal()` and the ordinal TabM call are blank. A later section keeps those rows in training. The residual regressor still has the highest R² on abalone, 0.5884793173419238. Bike sharing `count` has 869 distinct values and a mode of 5 (260 rows), so it stayed a regression. Ailerons `goal` has 35 distinct values in the file, the most common being −0.0006 (1854 rows), and it stayed a regression.

Exact test scores. yprop_4_1 R²: residual −0.015524945119663291, Ashlar TabM 0.10130612952812834, XGBoost −0.01919088592989815, TabM-D 0.11157329967795337. Ailerons R²: −1.2719611397966966, 0.8426452992264438, 0.8274166850231689, 0.8438557053456597. Bike_Sharing_Demand R²: 0.6584418017747304, 0.6975069886793399, 0.6853970823098485, 0.6892814825374981. Electricity accuracy: 0.7795003010234798, 0.8543548063415614, 0.8203893237005819. Eye_movements accuracy: 0.5509693558474046, 0.642901813633521, 0.5641025641025641. Pol regression R²: 0.995414437772962, 0.9904846819079811, 0.9829587889198385, 0.9905399344639576. Pol expected-label R²: ordinal residual 0.9957131362802631, ordinal TabM 0.9888897863758372, `XGBClassifier` 0.9885216337211069, TabM-D classifier 0.989847939743679. Abalone regression R²: 0.5884793173419238, 0.5709209808555413, 0.49419283632827204, 0.5744665500860998. Abalone expected-label R²: `XGBClassifier` 0.5313882944755375, TabM-D classifier 0.5534104987332197.

Ashlar fits that stopped before 50 epochs: yprop TabM at 24, Ailerons TabM at 36, bike TabM at 37, eye_movements residual at 26, abalone residual at 48, abalone TabM at 41. The other Ashlar fits ran 50 epochs. Quantile knots on the regression TabM recipe were 165, 256, 266, 266, and 77 for yprop, Ailerons, bike, pol, and abalone. On pol the ordinal TabM used 266 knots as well.

On the tables already in this file, the highest recorded test score is TabM-D on houses (0.8366028121954286), XGBoost on california (0.8968 accuracy; that Ashlar cell is the earlier ReLU, feature-dropout 0.1 fit), `XGBClassifier` expected label on wine quality (0.48996919003701056), XGBoost on the wine regression row (0.4284963911647759), Ashlar TabM on cpu_act (0.9851878334347116), and Ashlar TabM on sulfur (0.8497774405322199). Those rows were not refit.

## Predictor default

`tabular()`, `classifier()`, and `ordinal()` now median-impute each numeric column, center it at the training median, divide by the training interquartile range, and apply `z / sqrt(1 + z²/9)`. That smooth clip saturates at ±3. A zero interquartile range uses half the training range. A constant column uses divisor 1. Object, bool, and other non-numeric columns get a learned embedding of width 8. Integer columns stay numeric unless they are named. `categorical_embeddings=[]` one-hots every categorical. `numeric_scaling="standard"` restores the mean and population standard deviation, with no clip. `num_embeddings` stays off. On the houses split, `linear_relu` numeric embeddings scored 0.7451792371862184, below the mean/std residual 0.7615065490594078, and a normal quantile map on that residual scored 0.6518968125086434. Robust scale is the numeric front end in RealMLP-TD. That model scored 0.8247 on houses beside XGBoost at 0.8268. The backbone, SiLU, weight decay `1e-4`, feature dropout 0, batch 64, 50 epochs, patience 10, and learning rate `1e-3` are the same no-argument call. `tabm()` does not set `numeric_scaling`, so a bare `tabm()` inherits robust scale. The TabM cells below still pass `numeric_scaling="quantile_tabr"`.

### Same splits

The seven tables above were scored again with `tabular()`, `classifier()`, or `ordinal()` and no other arguments, on the splits already stored under `/tmp/ashlar-bench/coverage-splits`. The new scores are the `predictor` and `ordinal_predictor` keys in `/tmp/ashlar-bench/coverage-results.json`. The previous `residual` and `ordinal_residual` keys are unchanged. Ashlar TabM, XGBoost, and TabM-D on these seven tables were left as recorded. Those Ashlar TabM calls pass `quantile_tabr`, and none of the seven frames has a string column, so the embedding default does not enter them. Abalone `Sex` is an integer code in that file, so it stayed numeric. The movement on these seven rows is the robust scale.

| Dataset | Task | Previous residual | New residual | Delta | Winner | Changed |
|---|---|---:|---:|---:|---|---|
| yprop_4_1 | regression | −0.0155 | 0.0257 | +0.0413 | TabM-D | no |
| Ailerons | regression | −1.2720 | 0.1469 | +1.4188 | TabM-D | no |
| Bike_Sharing_Demand | regression | 0.6584 | 0.6502 | −0.0083 | Ashlar TabM | no |
| electricity | classification | 0.7795 | 0.7790 | −0.0005 | XGBoost | no |
| eye_movements | classification | 0.5510 | 0.5491 | −0.0019 | XGBoost | no |
| pol | regression | 0.9954 | 0.9894 | −0.0060 | TabM-D | yes |
| pol | ordinal, expected label | 0.9957 | 0.9922 | −0.0036 | residual ordinal | no |
| abalone | regression | 0.5885 | 0.5802 | −0.0082 | residual MLP | no |
| abalone | ordinal, expected label |  |  |  | TabM-D | no |

The pol regression row is the winner change. The previous residual R², 0.995414437772962, led that row. The new residual R² is 0.9894350078756996. TabM-D at 0.9905399344639576 now leads Ashlar TabM at 0.9904846819079811 by 0.000055. At four decimals those two frozen scores both read 0.9905. The ordinal row on the same split still belongs to `ordinal()`. Its new expected-label R², 0.9921589473558322, is still the highest score on pol, above the TabM-D regression score and above the frozen ordinal TabM, `XGBClassifier`, and TabM-D classifier scores.

Exact new test scores. yprop_4_1 R² 0.025728009549908726, stopped at 26 epochs; the previous score was −0.015524945119663291 after 50 epochs. Ailerons R² 0.14688212752663465, from −1.2719611397966966. Bike_Sharing_Demand R² 0.6501886155269434, from 0.6584418017747304. Electricity accuracy 0.7789985952237608, from 0.7795003010234798. Eye_movements accuracy 0.5490931832395247 at 14 epochs, from 0.5509693558474046 at 26 epochs. Pol regression R² 0.9894350078756996, from 0.995414437772962. Pol expected-label R² 0.9921589473558322, from 0.9957131362802631. Abalone regression R² 0.5802327842522979 at 36 epochs, from 0.5884793173419238 at 48 epochs. That abalone score still leads TabM-D's frozen regression R² 0.5744665500860998. Abalone `ordinal()` again raises on the stratified holdout in this section: rings 1, 24, 25, and 26 have one row in the outer training partition. TabM-D's expected-label R² 0.5534104987332197 is the best expected-label score recorded here. A later section fits `ordinal()` by keeping those rings in training.

Ailerons is the large movement that leaves the winner in place. The residual R² rises from −1.2719611397966966 to 0.14688212752663465. TabM-D stays ahead at 0.8438557053456597. yprop_4_1 crosses from a negative R² to 0.025728009549908726 and still trails TabM-D at 0.11157329967795337.

### Diamonds

The seven files above have no string categoricals. The Hugging Face `reg_cat/diamonds.csv` stores cut, color, and clarity as integers, so Ashlar would keep them numeric. This table is the ggplot2 diamonds file shipped with seaborn, [diamonds.csv](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv). `cut`, `color`, and `clarity` are strings. `price` is the raw dollar price, from 326 to 18823, rather than the log target in the transformed benchmark file. 53940 rows, one `RandomState(0)` call to `grinsztajn_medium_split`, 10000 / 13182 / 30758. Nine features: six numeric and those three strings. The test partition has more than 6000 rows, so one fold matches appendix A.3.

`tabular()` embedded `cut`, `color`, and `clarity` and robust-scaled the six numeric columns. The Ashlar TabM cell is the same regression recipe as the rows above, `tabm(numeric_scaling="quantile_tabr", standardize_target=True, share_training_batches=False, max_epochs=50, random_state=0)`, and it embedded the same three columns because `categorical_embeddings` was left unset. XGBoost and TabM-D saw a one-hot fit on the training categories only. An unseen level is a zero in that encoding. The file has no missing categoricals.

| Dataset | Task | Rows | Train / paper-val / test | Features | Residual MLP | Ashlar TabM | XGBoost | TabM-D | Winner |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| diamonds | regression | 53940 | 10000 / 13182 / 30758 | 9 | 0.9739 | 0.9764 | 0.9755 | 0.9768 | TabM-D |

Exact test R²: residual 0.9738658601382127 (39 epochs), Ashlar TabM 0.9763608686086517 (45 epochs, 266 quantile knots), XGBoost 0.9754618505722888, TabM-D 0.9767674106491309. TabM-D leads Ashlar TabM by 0.000406. There is no earlier diamonds row to compare. Nothing was registered or uploaded.

## Label z-score

`tabular()` now z-scores regression labels with the mean and population standard deviation of the internal training partition and inverts that map in `predict`. A standard deviation below `1e-6` is replaced by 1. `standardize_target=False` leaves `y` on its original scale. `classifier()` and `ordinal()` keep the flag off, including when it is omitted. An explicit `True` on a non-regression task still raises. A checkpoint that omits the flag loads with it off. The coverage TabM calls already passed `standardize_target=True` and were not refit. XGBoost, TabM-D, classification, and the wine ordinal rows stay as recorded.

The regression rows use the same coverage splits as the predictor rescore, plus the cached houses split. The previous residual is robust scale with labels on their original scale. The new residual is `tabular()` with no extra arguments: robust scale and the label z-score. Scores are the `unit_target` key in `/tmp/ashlar-bench/horizon-results.json`, written by `examples/benchmark_horizon.py`.

| Dataset | Previous residual | New residual | Delta | Winner | Changed |
|---|---:|---:|---:|---|---|
| yprop_4_1 | 0.0257 | 0.0635 | +0.0378 | TabM-D | no |
| Ailerons | 0.1469 | 0.8417 | +0.6949 | TabM-D | no |
| Bike_Sharing_Demand | 0.6502 | 0.6379 | −0.0123 | Ashlar TabM | no |
| pol | 0.9894 | 0.9876 | −0.0019 | TabM-D | no |
| abalone | 0.5802 | 0.5801 | −0.0002 | residual MLP | no |
| diamonds | 0.9739 | 0.9736 | −0.0002 | TabM-D | no |
| houses | 0.7508 | 0.7551 | +0.0043 | TabM-D | no |

No winner on these rows changes. The previous residual for the six coverage tables is the predictor-rescore column above. Houses had no robust-scale score before this run, so its previous residual is a separate fit of `tabular(standardize_target=False)` on the same cached split, the `robust_only` key in the same JSON.

Ailerons is the movement that puts the residual next to the leaders. Test R² rises from 0.14688212752663465 to 0.8417441807864341. RMSE falls from 0.00037090777770926387 to 0.00015975014621874342. The fit stopped at 33 epochs. The best internal validation loss is 0.17182528972625732, against 1.3205198001742247e-07 on the unscaled labels. The training-label mean is −0.0008753376672658223 and the population standard deviation is 0.0004092321953405172. The new residual leads XGBoost at 0.8274166850231689 and trails Ashlar TabM at 0.8426452992264438 by 0.000901 and TabM-D at 0.8438557053456597 by 0.002112. TabM-D remains the winner.

yprop_4_1 rises from 0.025728009549908726 to 0.06352824036932392 and still trails TabM-D at 0.11157329967795337. Its training-label standard deviation is 0.0308967239808073, and the fit stopped at 24 epochs. Bike_Sharing_Demand falls from 0.6501886155269434 to 0.6379200630141588 and still trails Ashlar TabM at 0.6975069886793399. Pol falls from 0.9894350078756996 to 0.9875721051733851 and still trails TabM-D at 0.9905399344639576. Abalone moves from 0.5802327842522979 to 0.5800630957948113 and still leads TabM-D's frozen regression R² 0.5744665500860998. Diamonds moves from 0.9738658601382127 to 0.9736246903652803 and still trails TabM-D at 0.9767674106491309. The diamonds fit embedded `cut`, `color`, and `clarity` and stopped at 28 epochs.

Houses separates the two defaults. The earlier SiLU residual on mean and standard deviation, with labels left alone, is 0.7615065490594078. Robust scale with labels left alone scores 0.7508186061878339. Robust scale plus the label z-score scores 0.7551135965443034. The training-label standard deviation is 0.5593105375431461, so the z-score is a small shift, +0.004295 on this split. The drop from 0.7615065490594078 is the robust scale. XGBoost remains at 0.826815147573534 and TabM-D at 0.8366028121954286. Both houses fits ran 50 epochs.

Exact new test R²: yprop_4_1 0.06352824036932392, Ailerons 0.8417441807864341, Bike_Sharing_Demand 0.6379200630141588, pol 0.9875721051733851, abalone 0.5800630957948113, diamonds 0.9736246903652803, houses 0.7551135965443034. Houses robust scale only: 0.7508186061878339. Nothing was registered or uploaded.

## Classification TabM

`tabm(task="classification")` is legal. The trunk is the `tabm()` recipe: 32 members, width 512, three blocks, hidden dropout 0.1, AdamW at `2e-3` with weight decay 0, batch 256, patience 16. The prediction policy is `mean_probabilities`. Labels are not z-scored. Members share one minibatch. `numeric_scaling` stays `robust`; a quantile map is still an explicit argument. `share_training_batches=False` still raises. The scored call adds `max_epochs=50` and `random_state=0`. The recipe's own epoch cap remains 200. `classifier()` is still the residual MLP.

The empty Ashlar TabM cells in the coverage table are the regression recipe, which passes `standardize_target=True` and `share_training_batches=False`. This section does not pass those flags. XGBoost and TabM-D were not refit. Electricity and eye_movements use the coverage splits. California uses the cached split from `examples/benchmark.py`. Scores are in `/tmp/ashlar-bench/tabm-classification-results.json`.

| Dataset | Current `classifier()` | Ashlar TabM | XGBoost | TabM-D | Winner | Changed |
|---|---:|---:|---:|---:|---|---|
| electricity | 0.7790 | 0.8024 | 0.8544 | 0.8204 | XGBoost | no |
| eye_movements | 0.5491 | 0.5716 | 0.6429 | 0.5641 | XGBoost | no |
| california | 0.8508 | 0.8814 | 0.8968 | 0.8916 | XGBoost | no |

The ensemble is ahead of the current residual classifier on all three tables and behind XGBoost on all three. Eye_movements is the one row where it also leads TabM-D. `classifier()` stays the no-argument classification path.

Exact test accuracy. Electricity: TabM 0.8024282560706402 at 45 epochs, from the current `classifier()` 0.7789985952237608. Frozen XGBoost is 0.8543548063415614 and TabM-D is 0.8203893237005819. Paper-val accuracy is 0.793022711308827. Eye_movements: TabM 0.5716072545340838 at 21 epochs, from `classifier()` 0.5490931832395247. Frozen XGBoost is 0.642901813633521 and TabM-D is 0.5641025641025641. Paper-val accuracy is 0.6125730994152047. California: TabM 0.881380977968834 at 45 epochs. The current `classifier()` on that same split is 0.8507522837184309 at 29 epochs. The earlier California cell, 0.8656636217087588, is the ReLU fit with feature dropout 0.1, and it was not refit. Frozen XGBoost is 0.8968296614723267 and TabM-D is 0.8915905427189683. Paper-val accuracy is 0.8799373040752351 for TabM and 0.8501567398119122 for the current classifier. Nothing was registered or uploaded.

## Rare-level split

An iid classification or ordinal split keeps every class with fewer than two rows in the training partition and applies `validation_size` to the remaining rows. Regression does not stratify, so the residual regressor is unchanged. On the abalone coverage split, rings 1, 24, 25, and 26 each have one row in the outer training partition. `ordinal()` with no extra arguments now fits. The expected-label test R² is 0.5780754759977107. Most-likely-label test R² is 0.26226003844387125. It ran all 50 epochs. The best internal validation loss was 2.2186717987060547. Paper-val expected-label R² is 0.4951741524840977.

That expected-label score leads the frozen `XGBClassifier` expectation 0.5313882944755375 and the frozen TabM-D classifier expectation 0.5534104987332197. The expected-label row, which was blank for Ashlar, now belongs to `ordinal()`. It is 0.0020 below the current residual regressor on the same split, test R² 0.5800630957948113. The residual regression default was not changed. XGBoost, TabM-D, and the ordinal TabM cell were not refit. The score is in `/tmp/ashlar-bench/abalone-ordinal-results.json`.

## Five-score additive, gate, and neighborhood

The cuts are the frozen XGBoost numbers in `clear-benchmark.md`. A row clears when its test score reaches that number. An equal score would count. All three bodies ran, because every additive row was short and every gate row stayed short. The residual regression default was not changed. XGBoost was not refit. Nothing was registered or uploaded.

The additive body is a piecewise-linear main effect per numeric column, knots at the internal-training quantiles 10% through 80%, and a width-8 embedding per categorical column. These five tables are numeric, so the embedding tables stay idle. At most ten pairwise products are added, and only when the internal-validation metric strictly rises. Each product is the leading subspace of that pair’s basis that carries 99.9% of the column energy on the fitting rows. The ridge grid is `{1e-4, 1e-2, 1, 100}`. Logistic fits use Newton-Cholesky at `C = 1 / ridge`. The internal holdout is `validation_size=0.2`, `random_state=0`, stratified for classification and ordinal. Regression is not stratified. On the wine outer train the ordinal split is 3,637 fitting rows and 910 selection rows. After the pairs and the ridge are chosen, coefficients are refit on the full outer train. Knots and pair subspaces stay on the fitting rows. Paper-validation is reported and was not used to fit or to select. The gate then puts two specialists behind a main-effect gate, with the pair budget of 10 split across the specialists. The neighborhood is a diagonal Nadaraya–Watson mean, or a distance-weighted class vote, with one scale per column and one bandwidth. Scores are in `/tmp/ashlar-bench/clear-benchmark-results.json`.

| Row | Cut | Additive | Gate | Neighborhood | Highest | Cleared |
|---|---:|---:|---:|---:|---|---|
| Wine expected label | 0.48996919003701056 | 0.320255239872416 | 0.32244612293981967 | 0.3141675065891435 | gate | no |
| Wine regression | 0.4284963911647759 | 0.36953233064235214 | 0.34073973065522356 | 0.3007719996147282 | additive | no |
| Electricity accuracy | 0.8543548063415614 | 0.7982640979329721 | 0.8048866144892635 | 0.7528095524784266 | gate | no |
| Eye_movements accuracy | 0.642901813633521 | 0.5941213258286429 | 0.5828642901813633 | 0.5684803001876173 | additive | no |
| California accuracy | 0.8968296614723267 | 0.8804406233207953 | 0.8781569048898442 | 0.8069586243954863 | additive | no |

Wine expected label. Additive test R² 0.320255239872416, paper-val 0.3272722464925648, 6 pairs, ridge 1.0, internal selection 0.4467386530780343, 106.7 seconds. The gate is a label gate on scores 6, 7, 8, and 9, with 5 pairs on each side. Its test R² is 0.32244612293981967 and its paper-val R² is 0.32028984232682456, in 3.8 seconds. The neighborhood bandwidth is 3.777463902511902. Its test R² is 0.3141675065891435 and its paper-val R² is 0.2468797601439776, in 3.6 seconds.

Wine regression. Additive test R² 0.36953233064235214, paper-val 0.3200788517088127, 10 pairs, ridge 1.0, internal selection 0.34750706789538055, 4.7 seconds. The unstratified holdout on this file is also 3,637 / 910. The gate is a label gate on scores 6 and 7, with 5 pairs on each side. Its test R² is 0.34073973065522356 and its paper-val R² is 0.3059332836010261, in 3.6 seconds. The neighborhood bandwidth is 4.268935656368664. Its test R² is 0.3007719996147282 and its paper-val R² is 0.2326349521836064, in 3.6 seconds.

Electricity. The outer train holdout is 8,000 / 2,000. Additive test accuracy 0.7982640979329721, paper-val 0.7979395926012643, 8 pairs, ridge 0.01, internal selection 0.8115, 5.5 seconds. The gate is a feature gate on column 1 at threshold 0.702128, with 5 pairs on each side. Its test accuracy is 0.8048866144892635 and its paper-val accuracy is 0.8016857878716928, in 10.2 seconds. The neighborhood bandwidth is 2.5227353870896128. Its test accuracy is 0.7528095524784266 and its paper-val accuracy is 0.7577850620463592, in 12.8 seconds.

Eye_movements. The outer train holdout is 4,260 / 1,065. Additive test accuracy 0.5941213258286429, paper-val 0.5789473684210527, 10 pairs, ridge 1.0, internal selection 0.6150234741784038, 28.7 seconds. The gate is a feature gate on column 18 at threshold 4, with 5 pairs on each side. Its test accuracy is 0.5828642901813633 and its paper-val accuracy is 0.5906432748538012, in 30.1 seconds. The neighborhood bandwidth is 2.87023583721229. Its test accuracy is 0.5684803001876173 and its paper-val accuracy is 0.5818713450292398, in 8.4 seconds.

California. The outer train holdout is 8,000 / 2,000. Additive test accuracy 0.8804406233207953, paper-val 0.8821316614420063, 8 pairs, ridge 0.0001, internal selection 0.889, 7.8 seconds. The gate is a feature gate on column 1 at threshold 29, with 1 pair on the low side and 3 on the high side. Its test accuracy is 0.8781569048898442 and its paper-val accuracy is 0.8899686520376175, in 12.7 seconds. The neighborhood bandwidth is 1.0779197125307578. Its test accuracy is 0.8069586243954863 and its paper-val accuracy is 0.8106583072100313, in 12.5 seconds.

No row reached its cut. Pair index lists, scales, and the other fit fields are in the JSON file named above.

## Differentiable oblivious ensemble

The additive, gate, and neighborhood rows above are short of every cut. This section is a different body: one layer of 512 oblivious trees of depth 6, with 1.5-entmax feature choices and 1.5-entmax soft thresholds, trained end to end. The choice and the frozen constants are in `clear-benchmark.md`. Adam uses learning rate `0.1`, batch 512, at most 80 epochs, and patience 16. The epoch count is chosen on the internal 20% holdout (`random_state=0`) by the task metric. A fresh run then trains on the full outer train for that many epochs, and the scored weights are the average of the last five of those epochs. Ordinal and classification use cross-entropy. The ordinal number is the expected label. Regression labels are z-scored inside this model only. Paper-validation was not used to fit or to select. XGBoost was not refit. The residual regression default was not changed. Nothing was registered or uploaded. Scores are in `/tmp/ashlar-bench/node-results.json`.

| Row | Cut | Test score | Paper-val | Epochs | Cleared |
|---|---:|---:|---:|---:|---|
| Wine expected label | 0.48996919003701056 | 0.39021639759961135 | 0.3410888047447478 | 28 | no |
| Wine regression | 0.4284963911647759 | 0.3901891210144286 | 0.3590892476028087 | 43 | no |
| Electricity accuracy | 0.8543548063415614 | 0.7807043949428055 | 0.7774525872161087 | 29 | no |
| Eye_movements accuracy | 0.642901813633521 | 0.6222639149468417 | 0.6169590643274854 | 33 | no |
| California accuracy | 0.8968296614723267 | 0.8593498119290703 | 0.8633228840125392 | 9 | no |

Wine expected label used the stratified 3,637 / 910 holdout. Its internal selection R² was 0.43220253312335266, and the fit took 111.1 seconds. Wine regression used the unstratified holdout, also 3,637 / 910 on this file. Its internal selection R² was 0.3590513859855491, in 153.2 seconds. Electricity’s holdout was 8,000 / 2,000, internal accuracy 0.788, in 245.8 seconds. Eye_movements’ holdout was 4,260 / 1,065, internal accuracy 0.6009389671361502, in 157.3 seconds. California’s holdout was 8,000 / 2,000, internal accuracy 0.863, in 114.7 seconds.

No row reached its cut.

## Histogram trees, linear leaves, and a learned neighborhood

Three more bodies, each scored on all five cuts. A row clears when its test score reaches the frozen XGBoost number. An equal score would count. The iteration, stage, or epoch count is chosen on the internal 20% holdout (`random_state=0`) by the task metric, with a strict improvement. Classification and ordinal stratify that holdout. Regression does not. Wine expected label is the stratified 3,637 / 910 split. Wine regression is the unstratified holdout, also 3,637 / 910 on this file. Electricity and california use 8,000 / 2,000. Eye_movements uses 4,260 / 1,065. The chosen budget is then refit on the full outer train. Paper-validation is reported and was not used to fit or to select. Ordinal training is multiclass cross-entropy, and the ordinal number is the expected label. XGBoost was not refit. The residual regression default was not changed. Nothing was registered or uploaded. Scores are in `/tmp/ashlar-bench/rounds-results.json`.

The histogram body is scikit-learn `HistGradientBoostingRegressor` or `HistGradientBoostingClassifier`: leaf-wise trees, at most 31 leaves, learning rate `0.1`, 255 bins, `min_samples_leaf` 20, and no internal early stopping. The cap is 300 iterations and the patience is 20. The linear-leaf body grows one split at a time, up to 200 stages with patience 20 and shrinkage `0.1`. Each side of the split is a ridge line, penalty `1.0` on the mean squared residual and none on the intercept, with 16 quantile thresholds and at least 32 rows in a leaf when the table is large enough. The learned neighborhood embeds each row with a width-64, width-64, width-32 network, dropout `0.1`, and a positive temperature. Training is AdamW at `1e-3` with weight decay `1e-4`, batch 256, at most 80 epochs, and patience 16. The scored weights are the average of the last five checkpoints. A query is a softmax over the training rows in that embedding.

| Row | Cut | Histogram | Linear leaf | Learned neighborhood | Cleared |
|---|---:|---:|---:|---:|---|
| Wine expected label | 0.48996919003701056 | 0.4943760799796796 | 0.3346736275373333 | 0.35706569667098875 | histogram |
| Wine regression | 0.4284963911647759 | 0.46964234173056674 | 0.39247101302005805 | 0.3930972636243897 | histogram |
| Electricity accuracy | 0.8543548063415614 | 0.84457154324704 | 0.783614288581176 | 0.7700180614087899 | no |
| Eye_movements accuracy | 0.642901813633521 | 0.6197623514696685 | 0.5784865540963102 | 0.5584740462789243 | no |
| California accuracy | 0.8968296614723267 | 0.8933369156367544 | 0.8456475013433637 | 0.8547823750671681 | no |

Wine expected label. Histogram test R² 0.4943760799796796, paper-val 0.4276138283944608, 85 iterations, internal selection 0.5065305810706368, 2.9 seconds. Linear leaf test R² 0.3346736275373333, paper-val 0.2884140361723585, 200 stages, internal selection 0.36653372868294787, 23.3 seconds. Learned neighborhood test R² 0.35706569667098875, paper-val 0.2870437753226327, epoch 1, internal selection 0.38314016369949655, 3.9 seconds.

Wine regression. Histogram test R² 0.46964234173056674, paper-val 0.4308374337286627, 189 iterations, internal selection 0.3950626340758374, 0.6 seconds. Linear leaf test R² 0.39247101302005805, paper-val 0.3617000056407996, 195 stages, internal selection 0.33172623580123506, 17.0 seconds. Learned neighborhood test R² 0.3930972636243897, paper-val 0.3142874541627442, epoch 1, internal selection 0.3362703490176434, 2.6 seconds.

Electricity. Histogram test accuracy 0.84457154324704, paper-val 0.8475766799344416, 98 iterations, internal selection 0.845, 0.6 seconds. Linear leaf test accuracy 0.783614288581176, paper-val 0.7824865371107469, 163 stages, internal selection 0.793, 22.6 seconds. Learned neighborhood test accuracy 0.7700180614087899, paper-val 0.7700772652774526, epoch 1, internal selection 0.769, 5.7 seconds.

Eye_movements. Histogram test accuracy 0.6197623514696685, paper-val 0.6608187134502924, 50 iterations, internal selection 0.6150234741784038, 0.5 seconds. Linear leaf test accuracy 0.5784865540963102, paper-val 0.5906432748538012, 23 stages, internal selection 0.5568075117370892, 31.8 seconds. Learned neighborhood test accuracy 0.5584740462789243, paper-val 0.5994152046783626, epoch 6, internal selection 0.5727699530516432, 3.6 seconds.

California. Histogram test accuracy 0.8933369156367544, paper-val 0.8934169278996865, 55 iterations, internal selection 0.8895, 0.5 seconds. Linear leaf test accuracy 0.8456475013433637, paper-val 0.8510971786833855, 61 stages, internal selection 0.8605, 6.0 seconds. Learned neighborhood test accuracy 0.8547823750671681, paper-val 0.8504702194357366, epoch 58, internal selection 0.857, 18.7 seconds.

The histogram body clears wine expected label and wine regression. It is short on electricity, eye_movements, and california. Linear leaves and the learned neighborhood are short on every row.

## Priors on the three open cuts

Wine is left as the histogram cleared it. These three rows are the open cuts. Each model is differentiable. The BART-shaped penalty sits on 40 soft depth-2 trees: split rate `0.95 / (1 + depth)^2`, a sparse penalty on the feature simplex, a U-shaped penalty on the gate, and a Gaussian leaf of scale `3 / sqrt(40)`. The variational head is a mean-field logistic layer on four soft quantile bins per column and every pairwise product of those bins. The gradient penalty is an L1 charge on the input gradient of a width-128 network. The choices and coefficients are in `clear-benchmark.md`. The epoch count comes from the stratified internal 20% holdout, `random_state=0`, with patience 16 and a cap of 80. A fresh run then trains on the full outer train for that many epochs. The scored weights are the average of the last five. Paper-validation was not used to fit or to select. XGBoost was not refit. The residual regression default was not changed. Nothing was registered or uploaded. Scores are in `/tmp/ashlar-bench/prior-results.json`.

| Row | Cut | BART prior | Variational head | Gradient penalty | Cleared |
|---|---:|---:|---:|---:|---|
| Electricity accuracy | 0.8543548063415614 | 0.7491471001404776 | 0.7745835841862332 | 0.7824603652418222 | no |
| Eye_movements accuracy | 0.642901813633521 | 0.5572232645403377 | 0.5634771732332708 | 0.5697310819262039 | no |
| California accuracy | 0.8968296614723267 | 0.7808973670069855 | 0.8437667920472864 | 0.8753358409457281 | no |

Electricity’s holdout was 8,000 / 2,000. BART prior test accuracy 0.7491471001404776, paper-val 0.7415125263404355, epoch 11, internal selection 0.7495, feature peak 0.9486458897590637, gate margin 0.16125425696372986, root split 0.9543956518173218, child split 0.25061550736427307, 4.3 seconds. Variational head test accuracy 0.7745835841862332, paper-val 0.7715991571060642, epoch 39, internal selection 0.7735, 5.7 seconds. Gradient penalty test accuracy 0.7824603652418222, paper-val 0.7780379302271131, epoch 40, internal selection 0.7845, 6.7 seconds.

Eye_movements’ holdout was 4,260 / 1,065. BART prior test accuracy 0.5572232645403377, paper-val 0.5862573099415205, epoch 29, internal selection 0.5558685446009389, feature peak 0.9897899031639099, gate margin 0.09164713323116302, root split 0.9522069692611694, child split 0.24305811524391174, 3.7 seconds. Variational head test accuracy 0.5634771732332708, paper-val 0.5935672514619883, epoch 44, internal selection 0.5615023474178403, 11.4 seconds. Gradient penalty test accuracy 0.5697310819262039, paper-val 0.6198830409356725, epoch 26, internal selection 0.5737089201877934, 3.3 seconds.

California’s holdout was 8,000 / 2,000. BART prior test accuracy 0.7808973670069855, paper-val 0.7862068965517242, epoch 49, internal selection 0.822, feature peak 0.9777435064315796, gate margin 0.07740043103694916, root split 0.9521018266677856, child split 0.24710988998413086, 5.6 seconds. Variational head test accuracy 0.8437667920472864, paper-val 0.841692789968652, epoch 78, internal selection 0.8475, 8.7 seconds. Gradient penalty test accuracy 0.8753358409457281, paper-val 0.8692789968652037, epoch 73, internal selection 0.878, 8.8 seconds.

No open cut cleared. On every row the gradient penalty is the highest of these three, and it is still short.

## Gradient penalty on three more classification tables

The same gradient-penalty classifier, with the same coefficients, on three more numerical binary tables from the Grinsztajn `clf_num` release. bank-marketing has 10,578 rows and 7 features, the same feature count as electricity. MagicTelescope has 13,376 rows and 10 features. credit has 16,714 rows and 10 features, the nearest unused row count to california. Each file is balanced across two classes. The split is one call of `grinsztajn_medium_split` with `numpy.random.RandomState(0)`. The test partitions have 2,222, 2,810, and 4,700 rows. The network is `GradientSparseNet(random_state=0)`: width 128, SiLU, binary cross-entropy plus 0.01 times the mean absolute input gradient, Adam at 1e-3, batch 256, patience 16, cap 80. The epoch count comes from the stratified internal 20% holdout. A fresh run then trains on the full outer train for that many epochs. The scored weights are the average of the last five. Default `XGBClassifier()` with no arguments is fit on the same outer train. Labels are mapped to 0..K-1 and predictions are mapped back. Paper-validation was not used to fit or to select. The residual regression default was not changed. Nothing was registered or uploaded. Before these fits, a probe of the training loss returned finite parameter gradients and a finite input gradient, and the loss matched binary cross-entropy plus the gradient penalty. The model stayed differentiable. Scores are in `/tmp/ashlar-bench/gradient-more-results.json`.

| Dataset | Source | Rows in file | Train / paper-val / test | Features | Gradient penalty | Default XGBoost |
|---|---|---:|---|---:|---:|---:|
| bank-marketing | [HF `clf_num/bank-marketing.csv`](https://huggingface.co/datasets/inria-soda/tabular-benchmark/resolve/main/clf_num/bank-marketing.csv), transformed OpenML [44126](https://www.openml.org/d/44126), original OpenML [1461](https://www.openml.org/d/1461) | 10578 | 7404 / 952 / 2222 | 7 | 0.7889288928892889 | 0.7983798379837984 |
| MagicTelescope | [HF `clf_num/MagicTelescope.csv`](https://huggingface.co/datasets/inria-soda/tabular-benchmark/resolve/main/clf_num/MagicTelescope.csv), transformed OpenML [44125](https://www.openml.org/d/44125), original OpenML [1120](https://www.openml.org/d/1120) | 13376 | 9363 / 1203 / 2810 | 10 | 0.8487544483985765 | 0.8629893238434164 |
| credit | [HF `clf_num/credit.csv`](https://huggingface.co/datasets/inria-soda/tabular-benchmark/resolve/main/clf_num/credit.csv), transformed OpenML [44089](https://www.openml.org/d/44089), Give Me Some Credit | 16714 | 10000 / 2014 / 4700 | 10 | 0.7393617021276596 | 0.7551063829787235 |

bank-marketing’s holdout was 5,923 / 1,481. Gradient penalty test accuracy 0.7889288928892889, paper-val 0.7899159663865546, epoch 20, internal selection 0.787305874409183, 4.0 seconds. Default XGBoost test accuracy 0.7983798379837984, paper-val 0.7888655462184874, 0.05 seconds.

MagicTelescope’s holdout was 7,490 / 1,873. Gradient penalty test accuracy 0.8487544483985765, paper-val 0.8445552784704904, epoch 64, internal selection 0.8606513614522157, 8.4 seconds. Default XGBoost test accuracy 0.8629893238434164, paper-val 0.8495428096425602, 0.08 seconds.

credit’s holdout was 8,000 / 2,000. Gradient penalty test accuracy 0.7393617021276596, paper-val 0.7487586891757696, epoch 24, internal selection 0.7535, 4.4 seconds. Default XGBoost test accuracy 0.7551063829787235, paper-val 0.7631578947368421, 0.06 seconds.

Default XGBoost is ahead on each test partition. The gradient penalty stayed a width-128 SiLU network with an input-gradient penalty.

## Gradient penalty on fifteen classification tables

Fifteen further numerical binary tables. Electricity, eye_movements, california, bank-marketing, MagicTelescope, and credit were already scored with this classifier, so they stay in the earlier sections. Ten tables are Grinsztajn `clf_num` files. phoneme and `kdd_ipums_la_97-small` are from commit `f918b61` of that release. Five are OpenML tables whose features are numeric and complete. The split is one call of `grinsztajn_medium_split` with `numpy.random.RandomState(0)`. The network is the same `GradientSparseNet(random_state=0)`: width 128, SiLU, binary cross-entropy plus 0.01 times the mean absolute input gradient, Adam at 1e-3, batch 256, patience 16, cap 80. The epoch count comes from the stratified internal 20% holdout. A fresh run then trains on the full outer train for that many epochs. The scored weights are the average of the last five. Default `XGBClassifier()` with no arguments is fit on the same outer train. Labels are mapped to 0..K-1 and predictions are mapped back. The comparison below is test accuracy. Paper-validation was recorded and was not used to fit or to select. `pol` in this section is `clf_num/pol.csv`, transformed OpenML [44122](https://www.openml.org/d/44122). The regression `pol` row earlier in this file is `reg_num/pol.csv`, transformed OpenML [44133](https://www.openml.org/d/44133). The residual regression default was not changed. Nothing was registered or uploaded. A probe of the training loss returned finite parameter gradients and a finite input gradient. The model stayed differentiable. Scores are in `/tmp/ashlar-bench/gradient-suite-results.json`.

| Dataset | Rows | Train / paper-val / test | Features | Gradient penalty | Default XGBoost | Beats or matches |
|---|---:|---|---:|---:|---:|---|
| house_16H | 13488 | 9441 / 1214 / 2833 | 16 | 0.8828097423226262 | 0.8806918460995411 | yes |
| heloc | 10000 | 7000 / 900 / 2100 | 22 | 0.73 | 0.7014285714285714 | yes |
| default-of-credit-card-clients | 13272 | 9290 / 1194 / 2788 | 20 | 0.710545193687231 | 0.6944045911047346 | yes |
| Bioresponse | 3434 | 2403 / 309 / 722 | 419 | 0.7825484764542936 | 0.7423822714681441 | yes |
| pol | 10082 | 7057 / 907 / 2118 | 26 | 0.9844192634560907 | 0.9815864022662889 | yes |
| Diabetes130US | 71090 | 10000 / 18327 / 42763 | 7 | 0.6013609896405772 | 0.5826298435563454 | yes |
| MiniBooNE | 72998 | 10000 / 18899 / 44099 | 50 | 0.9051225651375314 | 0.9333091453320937 | no |
| jannis | 57580 | 10000 / 14274 / 33306 | 54 | 0.7392361736624031 | 0.757521167357233 | no |
| phoneme | 3172 | 2220 / 285 / 667 | 6 | 0.992503748125937 | 0.9985007496251874 | no |
| kdd_ipums_la_97-small | 5188 | 3631 / 467 / 1090 | 21 | 0.9926605504587156 | 1.0 | no |
| eeg-eye-state | 14980 | 10000 / 1494 / 3486 | 14 | 0.9216867469879518 | 0.9251290877796902 | no |
| mozilla4 | 15545 | 10000 / 1663 / 3882 | 5 | 0.8894899536321483 | 0.9448737764039155 | no |
| spambase | 4601 | 3220 / 414 / 967 | 57 | 0.9276111685625646 | 0.9482936918304034 | no |
| twonorm | 7400 | 5180 / 666 / 1554 | 20 | 0.9768339768339769 | 0.9716859716859717 | yes |
| qsar-biodeg | 1055 | 738 / 95 / 222 | 41 | 0.8513513513513513 | 0.8693693693693694 | no |

The gradient penalty beats or matches default XGBoost on test accuracy for house_16H, heloc, default-of-credit-card-clients, Bioresponse, pol, Diabetes130US, and twonorm.

house_16H. [HF `clf_num/house_16H.csv`](https://huggingface.co/datasets/inria-soda/tabular-benchmark/resolve/main/clf_num/house_16H.csv), original OpenML [821](https://www.openml.org/d/821), transformed OpenML [44123](https://www.openml.org/d/44123). Holdout 7,552 / 1,889. Gradient penalty test accuracy 0.8828097423226262, paper-val 0.8665568369028006, epoch 44, internal selection 0.8798305982001059, 6.5 seconds. Default XGBoost test accuracy 0.8806918460995411, paper-val 0.8797364085667215, 0.11 seconds.

heloc. [HF `clf_num/heloc.csv`](https://huggingface.co/datasets/inria-soda/tabular-benchmark/resolve/main/clf_num/heloc.csv), transformed OpenML [45026](https://www.openml.org/d/45026). Holdout 5,600 / 1,400. Gradient penalty test accuracy 0.73, paper-val 0.7233333333333334, epoch 9, internal selection 0.7064285714285714, 2.8 seconds. Default XGBoost test accuracy 0.7014285714285714, paper-val 0.7333333333333333, 0.06 seconds.

default-of-credit-card-clients. [HF `clf_num/default-of-credit-card-clients.csv`](https://huggingface.co/datasets/inria-soda/tabular-benchmark/resolve/main/clf_num/default-of-credit-card-clients.csv), original OpenML [42477](https://www.openml.org/d/42477), transformed OpenML [45020](https://www.openml.org/d/45020). Holdout 7,432 / 1,858. Gradient penalty test accuracy 0.710545193687231, paper-val 0.6943048576214406, epoch 17, internal selection 0.6894510226049516, 3.8 seconds. Default XGBoost test accuracy 0.6944045911047346, paper-val 0.6825795644891123, 0.10 seconds.

Bioresponse. [HF `clf_num/Bioresponse.csv`](https://huggingface.co/datasets/inria-soda/tabular-benchmark/resolve/main/clf_num/Bioresponse.csv), original OpenML [4134](https://www.openml.org/d/4134), transformed OpenML [45019](https://www.openml.org/d/45019). Holdout 1,922 / 481. Gradient penalty test accuracy 0.7825484764542936, paper-val 0.7637540453074434, epoch 20, internal selection 0.7817047817047817, 2.6 seconds. Default XGBoost test accuracy 0.7423822714681441, paper-val 0.7540453074433657, 0.41 seconds.

pol. [HF `clf_num/pol.csv`](https://huggingface.co/datasets/inria-soda/tabular-benchmark/resolve/main/clf_num/pol.csv), original OpenML [722](https://www.openml.org/d/722), transformed OpenML [44122](https://www.openml.org/d/44122). Holdout 5,645 / 1,412. Gradient penalty test accuracy 0.9844192634560907, paper-val 0.9856670341786108, epoch 61, internal selection 0.9837110481586402, 6.5 seconds. Default XGBoost test accuracy 0.9815864022662889, paper-val 0.980154355016538, 0.05 seconds.

Diabetes130US. [HF `clf_num/Diabetes130US.csv`](https://huggingface.co/datasets/inria-soda/tabular-benchmark/resolve/main/clf_num/Diabetes130US.csv), original OpenML [4541](https://www.openml.org/d/4541), transformed OpenML [45022](https://www.openml.org/d/45022). Holdout 8,000 / 2,000. Gradient penalty test accuracy 0.6013609896405772, paper-val 0.6012986304359688, epoch 10, internal selection 0.612, 3.4 seconds. Default XGBoost test accuracy 0.5826298435563454, paper-val 0.5804550662956294, 0.06 seconds.

MiniBooNE. [HF `clf_num/MiniBooNE.csv`](https://huggingface.co/datasets/inria-soda/tabular-benchmark/resolve/main/clf_num/MiniBooNE.csv), original OpenML [41150](https://www.openml.org/d/41150), transformed OpenML [44128](https://www.openml.org/d/44128). Holdout 8,000 / 2,000. Gradient penalty test accuracy 0.9051225651375314, paper-val 0.9084078522673157, epoch 77, internal selection 0.915, 9.8 seconds. Default XGBoost test accuracy 0.9333091453320937, paper-val 0.9337531086300862, 0.27 seconds.

jannis. [HF `clf_num/jannis.csv`](https://huggingface.co/datasets/inria-soda/tabular-benchmark/resolve/main/clf_num/jannis.csv), original OpenML [41168](https://www.openml.org/d/41168), transformed OpenML [45021](https://www.openml.org/d/45021). The released classification file is binary. Holdout 8,000 / 2,000. Gradient penalty test accuracy 0.7392361736624031, paper-val 0.7428191116715707, epoch 13, internal selection 0.7435, 3.7 seconds. Default XGBoost test accuracy 0.757521167357233, paper-val 0.7633459436738125, 0.31 seconds.

phoneme. [HF commit `f918b61` `clf_num/phoneme.csv`](https://huggingface.co/datasets/inria-soda/tabular-benchmark/resolve/f918b61498213f463fc183c6e070b8cf9fb7c146/clf_num/phoneme.csv). Holdout 1,776 / 444. Gradient penalty test accuracy 0.992503748125937, paper-val 0.9929824561403509, epoch 18, internal selection 0.9932432432432432, 2.2 seconds. Default XGBoost test accuracy 0.9985007496251874, paper-val 1.0, 0.02 seconds.

kdd_ipums_la_97-small. [HF commit `f918b61` `clf_num/kdd_ipums_la_97-small.csv`](https://huggingface.co/datasets/inria-soda/tabular-benchmark/resolve/f918b61498213f463fc183c6e070b8cf9fb7c146/clf_num/kdd_ipums_la_97-small.csv). Holdout 2,904 / 727. Gradient penalty test accuracy 0.9926605504587156, paper-val 0.9914346895074947, epoch 15, internal selection 0.9931224209078404, 2.4 seconds. Default XGBoost test accuracy 1.0, paper-val 1.0, 0.02 seconds.

eeg-eye-state. OpenML [1471](https://www.openml.org/d/1471). Holdout 8,000 / 2,000. Gradient penalty test accuracy 0.9216867469879518, paper-val 0.9330655957161981, epoch 80, internal selection 0.912, 10.2 seconds. Default XGBoost test accuracy 0.9251290877796902, paper-val 0.927710843373494, 0.08 seconds.

mozilla4. OpenML [1046](https://www.openml.org/d/1046). Holdout 8,000 / 2,000. Gradient penalty test accuracy 0.8894899536321483, paper-val 0.8983764281419122, epoch 37, internal selection 0.894, 5.8 seconds. Default XGBoost test accuracy 0.9448737764039155, paper-val 0.9530968129885748, 0.06 seconds.

spambase. OpenML [44](https://www.openml.org/d/44). Holdout 2,576 / 644. Gradient penalty test accuracy 0.9276111685625646, paper-val 0.9371980676328503, epoch 28, internal selection 0.937888198757764, 2.8 seconds. Default XGBoost test accuracy 0.9482936918304034, paper-val 0.9420289855072463, 0.09 seconds.

twonorm. OpenML [1507](https://www.openml.org/d/1507). Holdout 4,144 / 1,036. Gradient penalty test accuracy 0.9768339768339769, paper-val 0.987987987987988, epoch 8, internal selection 0.9662162162162162, 2.5 seconds. Default XGBoost test accuracy 0.9716859716859717, paper-val 0.984984984984985, 0.08 seconds.

qsar-biodeg. OpenML [1494](https://www.openml.org/d/1494). Holdout 590 / 148. Gradient penalty test accuracy 0.8513513513513513, paper-val 0.8526315789473684, epoch 25, internal selection 0.8648648648648649, 1.9 seconds. Default XGBoost test accuracy 0.8693693693693694, paper-val 0.8526315789473684, 0.05 seconds.

## Multiclass gradient penalty

The gradient-penalty classifier now accepts more than two classes. Two classes still use one logit, sigmoid cross-entropy, and the mean absolute input gradient. Three or more classes use one logit per class, softmax cross-entropy, and the same coefficient on the mean absolute input Jacobian of those logits. Width 128, SiLU, Adam at 1e-3, batch 256, patience 16, and the cap of 80 are unchanged. The epoch count comes from the stratified internal 20% holdout. A fresh run then trains on the full outer train for that many epochs. The scored weights are the average of the last five. These five OpenML tables are numerical and have no missing values. The split is one call of `grinsztajn_medium_split` with `numpy.random.RandomState(0)`. Default `XGBClassifier()` with no arguments is fit on the same outer train. Labels are mapped to 0..K-1 and predictions are mapped back. The comparison is test accuracy. Paper-validation was recorded and was not used to fit or to select. A probe of the multiclass loss returned finite parameter gradients and a finite input Jacobian, and the loss matched softmax cross-entropy plus the Jacobian penalty. The model stayed differentiable. The residual regression default was not changed. Nothing was registered or uploaded. Scores are in `/tmp/ashlar-bench/gradient-multiclass-results.json`.

| Dataset | Classes | Rows | Train / paper-val / test | Features | Gradient penalty | Default XGBoost | Beats or matches |
|---|---:|---:|---|---:|---:|---:|---|
| letter | 26 | 20000 | 10000 / 3000 / 7000 | 16 | 0.9557142857142857 | 0.946 | yes |
| pendigits | 10 | 10992 | 7694 / 989 / 2309 | 16 | 0.9917713295799048 | 0.9900389779125163 | yes |
| optdigits | 10 | 5620 | 3933 / 506 / 1181 | 64 | 0.9889923793395428 | 0.983065198983912 | yes |
| satimage | 6 | 6430 | 4501 / 578 / 1351 | 36 | 0.9030347890451518 | 0.9148778682457439 | no |
| waveform-5000 | 3 | 5000 | 3500 / 450 / 1050 | 40 | 0.8561904761904762 | 0.8504761904761905 | yes |

The gradient penalty beats or matches default XGBoost on test accuracy for letter, pendigits, optdigits, and waveform-5000.

letter. OpenML [6](https://www.openml.org/d/6). Holdout 8,000 / 2,000. Gradient penalty test accuracy 0.9557142857142857, paper-val 0.9543333333333334, epoch 80, internal selection 0.946, 78.4 seconds. Default XGBoost test accuracy 0.946, paper-val 0.9493333333333334, 1.01 seconds.

pendigits. OpenML [32](https://www.openml.org/d/32). Holdout 6,155 / 1,539. Gradient penalty test accuracy 0.9917713295799048, paper-val 0.9908998988877654, epoch 19, internal selection 0.9935022742040286, 7.1 seconds. Default XGBoost test accuracy 0.9900389779125163, paper-val 0.9929221435793731, 0.30 seconds.

optdigits. OpenML [28](https://www.openml.org/d/28). Holdout 3,146 / 787. Gradient penalty test accuracy 0.9889923793395428, paper-val 0.9920948616600791, epoch 72, internal selection 0.9796696315120712, 9.6 seconds. Default XGBoost test accuracy 0.983065198983912, paper-val 0.9762845849802372, 0.29 seconds.

satimage. OpenML [182](https://www.openml.org/d/182). Holdout 3,600 / 901. Gradient penalty test accuracy 0.9030347890451518, paper-val 0.9134948096885813, epoch 28, internal selection 0.8967813540510544, 5.3 seconds. Default XGBoost test accuracy 0.9148778682457439, paper-val 0.916955017301038, 0.26 seconds.

waveform-5000. OpenML [60](https://www.openml.org/d/60). Holdout 2,800 / 700. Gradient penalty test accuracy 0.8561904761904762, paper-val 0.8355555555555556, epoch 9, internal selection 0.8485714285714285, 3.0 seconds. Default XGBoost test accuracy 0.8504761904761905, paper-val 0.8511111111111112, 0.39 seconds.

## Ten more multiclass tables

The same multiclass gradient-penalty classifier, with the same coefficients, on ten more public numerical tables. letter, pendigits, optdigits, satimage, and waveform-5000 were already scored and were not fit again. Each file has at least three classes, numeric features, and no missing values. The split is one call of `grinsztajn_medium_split` with `numpy.random.RandomState(0)`. On jannis the 50,000-row test cap in that helper binds, so the test partition is 50,000 rows. `jannis` here is OpenML [41168](https://www.openml.org/d/41168), the 4-class table. The binary `jannis` row earlier in this file is the Grinsztajn `clf_num` file. Default `XGBClassifier()` with no arguments is fit on the same outer train. Labels are mapped to 0..K-1 and predictions are mapped back. The comparison is test accuracy. Paper-validation was recorded and was not used to fit or to select. The residual regression default was not changed. Nothing was registered or uploaded. Scores are in `/tmp/ashlar-bench/gradient-multiclass-more-results.json`.

| Dataset | Classes | Rows | Train / paper-val / test | Features | Gradient penalty | Default XGBoost | Beats or matches |
|---|---:|---:|---|---:|---:|---:|---|
| vehicle | 4 | 846 | 592 / 76 / 178 | 18 | 0.7471910112359551 | 0.7528089887640449 | no |
| segment | 7 | 2310 | 1617 / 207 / 486 | 19 | 0.9588477366255144 | 0.9711934156378601 | no |
| page-blocks | 5 | 5473 | 3831 / 492 / 1150 | 10 | 0.96 | 0.9695652173913043 | no |
| shuttle | 7 | 58000 | 10000 / 14400 / 33600 | 9 | 0.9958928571428571 | 0.9995535714285714 | no |
| texture | 11 | 5500 | 3849 / 495 / 1156 | 40 | 0.9878892733564014 | 0.9852941176470589 | yes |
| wall-robot-navigation | 4 | 5456 | 3819 / 491 / 1146 | 24 | 0.9240837696335078 | 0.9965095986038395 | no |
| cardiotocography | 10 | 2126 | 1488 / 191 / 447 | 35 | 1.0 | 1.0 | yes |
| steel-plates-fault | 7 | 1941 | 1358 / 174 / 409 | 27 | 0.78239608801956 | 0.8141809290953546 | no |
| gas-drift | 6 | 13910 | 9737 / 1251 / 2922 | 128 | 0.9911019849418207 | 0.9917864476386037 | no |
| jannis | 4 | 83733 | 10000 / 22119 / 50000 | 54 | 0.64834 | 0.68504 | no |

The gradient penalty beats or matches default XGBoost on test accuracy for texture and cardiotocography.

vehicle. OpenML [54](https://www.openml.org/d/54). Holdout 473 / 119. Gradient penalty test accuracy 0.7471910112359551, paper-val 0.7236842105263158, epoch 32, internal selection 0.7815126050420168, 2.6 seconds. Default XGBoost test accuracy 0.7528089887640449, paper-val 0.7105263157894737, 0.08 seconds.

segment. OpenML [36](https://www.openml.org/d/36). Holdout 1,293 / 324. Gradient penalty test accuracy 0.9588477366255144, paper-val 0.9420289855072463, epoch 64, internal selection 0.9691358024691358, 4.4 seconds. Default XGBoost test accuracy 0.9711934156378601, paper-val 0.9710144927536232, 0.12 seconds.

page-blocks. OpenML [30](https://www.openml.org/d/30). Holdout 3,064 / 767. Gradient penalty test accuracy 0.96, paper-val 0.9735772357723578, epoch 11, internal selection 0.9621903520208605, 3.4 seconds. Default XGBoost test accuracy 0.9695652173913043, paper-val 0.9796747967479674, 0.14 seconds.

shuttle. OpenML [40685](https://www.openml.org/d/40685). Holdout 8,000 / 2,000. Gradient penalty test accuracy 0.9958928571428571, paper-val 0.9948611111111111, epoch 20, internal selection 0.9965, 7.2 seconds. Default XGBoost test accuracy 0.9995535714285714, paper-val 0.9990972222222222, 0.14 seconds.

texture. OpenML [40499](https://www.openml.org/d/40499). Holdout 3,079 / 770. Gradient penalty test accuracy 0.9878892733564014, paper-val 0.98989898989899, epoch 26, internal selection 0.9935064935064936, 5.7 seconds. Default XGBoost test accuracy 0.9852941176470589, paper-val 0.9777777777777777, 0.41 seconds.

wall-robot-navigation. OpenML [1497](https://www.openml.org/d/1497). Holdout 3,055 / 764. Gradient penalty test accuracy 0.9240837696335078, paper-val 0.9368635437881874, epoch 67, internal selection 0.8952879581151832, 6.3 seconds. Default XGBoost test accuracy 0.9965095986038395, paper-val 1.0, 0.12 seconds.

cardiotocography. OpenML [1466](https://www.openml.org/d/1466). Holdout 1,190 / 298. Gradient penalty test accuracy 1.0, paper-val 1.0, epoch 8, internal selection 1.0, 2.8 seconds. Default XGBoost test accuracy 1.0, paper-val 1.0, 0.08 seconds.

steel-plates-fault. OpenML [40982](https://www.openml.org/d/40982). Holdout 1,086 / 272. Gradient penalty test accuracy 0.78239608801956, paper-val 0.7931034482758621, epoch 38, internal selection 0.7867647058823529, 3.1 seconds. Default XGBoost test accuracy 0.8141809290953546, paper-val 0.8045977011494253, 0.25 seconds.

gas-drift. OpenML [1476](https://www.openml.org/d/1476). Holdout 7,789 / 1,948. Gradient penalty test accuracy 0.9911019849418207, paper-val 0.991207034372502, epoch 39, internal selection 0.9953798767967146, 11.6 seconds. Default XGBoost test accuracy 0.9917864476386037, paper-val 0.9944044764188649, 1.38 seconds.

jannis. OpenML [41168](https://www.openml.org/d/41168). Holdout 8,000 / 2,000. Gradient penalty test accuracy 0.64834, paper-val 0.6456892264568923, epoch 18, internal selection 0.64, 6.0 seconds. Default XGBoost test accuracy 0.68504, paper-val 0.6793706767937068, 1.10 seconds.

## Categorical embeddings

The gradient-penalty classifier now learns a categorical embedding and stays differentiable. A numeric matrix still uses the previous one-logit or multiclass loss, with the same width, penalty, and schedule. A column is categorical when it is not numeric and not boolean. Integer columns stay numeric.

Each categorical column has its own table of shape `(1 + V, 8)`. `V` is the number of levels seen in that phase's fit rows, scanned in the frame's row order. Index 0 is the unknown row. Missing values and levels that were not seen in that phase land on index 0. Known levels occupy indices `1..V` in that first-seen order. The table is drawn from a normal distribution and multiplied by 0.02. Numeric columns are z-scored with the mean and standard deviation of the same fit rows. The network input is those z-scored numbers concatenated with the embedding vectors. The penalty is still `0.01` times the mean absolute Jacobian of the logits with respect to that concatenated vector. The category index is discrete, so the gradient does not flow through the index. It does flow into the embedding table, through the lookup and through the Jacobian penalty. Two classes keep one logit and sigmoid cross-entropy. Three or more classes keep one logit per class and softmax cross-entropy. The internal 20% stratified holdout chooses the epoch, then the network is reinitialized and trained on every outer-train row. That refit rebuilds the vocabulary and the numeric moments on all of those rows. Width 128, SiLU, Adam `1e-3`, batch 256, patience 16, and a cap of 80 epochs are unchanged.

Default `XGBClassifier()` with no arguments is fit on the same outer split. Its categorical columns are a train-only one-hot, and unseen or missing levels are filled with 0. Its numeric columns are not standardized. Labels are mapped to `0..K-1` and predictions are mapped back. The comparison is test accuracy. Paper-validation was recorded and was not used to fit or to select. The five tables were fixed before either model was fit: OpenML credit-g, cmc, car, mushroom, and adult, each with at least one non-numeric column. The split is one call of `grinsztajn_medium_split` with `numpy.random.RandomState(0)`. The residual regression default was not changed. Nothing was registered or uploaded. A probe checked that the embedding parameters and the input Jacobian have finite gradients, and that the loss equals cross-entropy plus the penalty. Scores are in `/tmp/ashlar-bench/gradient-categorical-results.json`.

| Dataset | Classes | Rows | Train / paper-val / test | Numeric | Categorical | Gradient penalty | Default XGBoost | Beats or matches |
|---|---:|---:|---|---:|---:|---:|---:|---|
| credit-g | 2 | 1000 | 700 / 90 / 210 | 7 | 13 | 0.7619047619047619 | 0.7571428571428571 | yes |
| cmc | 3 | 1473 | 1031 / 132 / 310 | 2 | 7 | 0.5419354838709678 | 0.5193548387096775 | yes |
| car | 4 | 1728 | 1209 / 155 / 364 | 0 | 6 | 0.7225274725274725 | 0.9835164835164835 | no |
| mushroom | 2 | 8124 | 5686 / 731 / 1707 | 0 | 22 | 0.9994141769185706 | 1.0 | no |
| adult | 2 | 48842 | 10000 / 11652 / 27190 | 6 | 8 | 0.8541375505700626 | 0.8626333210739242 | no |

The gradient penalty beats or matches default XGBoost on test accuracy for credit-g and cmc.

credit-g. OpenML [31](https://www.openml.org/d/31). Seven numeric columns and 13 categorical columns, none missing. Holdout 560 / 140. Network input width 111. Gradient penalty test accuracy 0.7619047619047619, paper-val 0.8111111111111111, epoch 29, internal selection 0.7857142857142857, 4.32 seconds. Default XGBoost test accuracy 0.7571428571428571, paper-val 0.8, 61 one-hot columns, 0.03 seconds.

cmc. OpenML [23](https://www.openml.org/d/23). Two numeric columns and seven categorical columns, none missing. Three classes. Holdout 824 / 207. Network input width 58. Gradient penalty test accuracy 0.5419354838709678, paper-val 0.5454545454545454, epoch 18, internal selection 0.6038647342995169, 3.06 seconds. Default XGBoost test accuracy 0.5193548387096775, paper-val 0.5378787878787878, 24 one-hot columns, 0.06 seconds.

car. OpenML [40975](https://www.openml.org/d/40975). Six categorical columns and no numeric columns. Four classes. Holdout 967 / 242. Network input width 48. The internal holdout accuracy stayed at 0.6983471074380165, which is the share of majority class `unacc` in that holdout, so the selected epoch is 1. Gradient penalty test accuracy 0.7225274725274725, the share of `unacc` in the test partition (263 of 364), paper-val 0.6451612903225806, the share of `unacc` there (100 of 155), 2.75 seconds. Default XGBoost test accuracy 0.9835164835164835, paper-val 0.9741935483870968, 21 one-hot columns, 0.07 seconds.

mushroom. OpenML [24](https://www.openml.org/d/24). Twenty-two categorical columns and no numeric columns. `stalk-root` has 2,480 missing values in the full table; those rows use the unknown embedding. Holdout 4,548 / 1,138. Network input width 176. Gradient penalty test accuracy 0.9994141769185706 (1,706 of 1,707), paper-val 1.0, epoch 11, internal selection 1.0, 6.64 seconds. Default XGBoost test accuracy 1.0, paper-val 1.0, 116 one-hot columns, 0.04 seconds.

adult. OpenML [1590](https://www.openml.org/d/1590). Six numeric columns and eight categorical columns. `workclass` has 2,799 missing values, `occupation` 2,809, and `native-country` 857; those values use the unknown embedding. The train cap binds, so the outer train has 10,000 rows. Holdout 8,000 / 2,000. Network input width 70. Gradient penalty test accuracy 0.8541375505700626, paper-val 0.8559903879162376, epoch 29, internal selection 0.858, 8.05 seconds. Default XGBoost test accuracy 0.8626333210739242, paper-val 0.867833848266392, 105 one-hot columns, 0.15 seconds.

## Forecasting

Two transformer forecasts share five univariate series, the same forecast origins, and one default lag-XGBoost baseline. The horizon is the next 12 steps in that series' own sampling interval. The lookback is 48 steps. The last 20% of each file is the test. An origin is the first index of the horizon. Its target is the next 12 values, and those 12 values stay out of the input. Test origins are every index in the test tail that still has a full horizon, so a later test input can contain earlier test observations. The comparison is test MAE over every test origin and every horizon step, with each point weighted equally. RMSE and R² use those same points. A transformer beats or matches XGBoost when its test MAE is lower or equal.

Both transformers minimize mean squared error on a z-scored horizon. The scale is the population mean and standard deviation. A standard deviation below `1e-6` is replaced by 1. During epoch selection the scale is fit on the series before the validation cut. The validation cut is the last 10% of the pre-test prefix. Selection keeps an epoch only when the validation MSE is strictly lower. Patience is 8 and the cap is 30 epochs. The network then starts again from `PRNGKey(0)` and trains for the chosen number of epochs on every pre-test origin, with the scale fit on the whole pre-test prefix. The last five checkpoints are averaged. Width is 32, with 4 heads, 2 pre-norm blocks, and a SiLU feed-forward layer of width 64. Adam uses learning rate `1e-3`, with no weight decay and no dropout. The input projection is drawn from a normal distribution and multiplied by 0.5. The learned recent positions are multiplied by 0.02. The horizon head is multiplied by `sqrt(1/32)`.

The window-locked transformer reads only `series[origin - 48:origin]`. Each of those 48 values is one token. The token is the value times the input projection, plus a bias, plus the learned position. The blocks are self-attention over that window. The last token emits the 12-step horizon.

The full-history transformer reads `series[:origin]`. The last 48 observations are one token each, with the same projection and learned positions, plus a sinusoidal age counted back from the origin. The newest of those tokens has age 1 and the oldest has age 48. Every older observation is in exactly one patch. Patches tile the older prefix from the start of the series in non-overlapping blocks of 48. A leftover tail is one patch, zero-filled on the empty positions, with a 0/1 mask of the real positions. The patch token is a linear map of the 48 values, initialized at `sqrt(1/48)`, plus a linear map of the mask, initialized at 0.02. Its sinusoidal age is the number of steps from the end of the patch back to the origin. A learned null token, initialized at 0.02, is always available so attention stays defined on a prefix that has no older patch. Each block lets the 48 recent tokens attend to each other, then cross-attend to the null token and the real patches. Padded patch slots are masked off. The last recent token emits the horizon. Changing the early part of a prefix changes this forecast when the last 48 values are held fixed. The patch weights and the learned recent positions receive finite gradients, and the loss equals the mean squared error.

XGBoost is `MultiOutputRegressor(XGBRegressor())` with no arguments and no early stopping. It is one model, shared by both tests. Its features are the same 48 raw lags, unscaled, and it trains on every pre-test origin.

The five series were fixed before either model was fit. ETTh1 is hourly oil temperature, column `OT`, from the [ETT repository](https://github.com/zhouhaoyi/ETDataset), 17,420 rows. Electricity is hourly load for the first client only, column 0 of the Lai et al. matrix in the [multivariate time-series repository](https://github.com/laiguokun/multivariate-time-series-data), 26,304 rows. Bike Sharing is the hourly rental count `cnt` from the [UCI bike file](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset), 17,379 rows in file order, with hours that are absent from `hour.csv` left absent. Daily minimum temperature is Melbourne, column `Temp`, from the [Brownlee daily series](https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv), 3,650 rows. Monthly sunspots are column `Sunspots` from the [Brownlee monthly series](https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv), 2,820 rows. The residual regression default was not changed. Nothing was registered or uploaded. Scores are in `/tmp/ashlar-bench/forecast-results.json`.

### Window locked to 48 lags

| Dataset | Sampling | Rows | Test origins | Window MAE | Window RMSE | Window R² | XGBoost MAE | XGBoost RMSE | XGBoost R² | Beats or matches |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ETTh1 OT | hour | 17420 | 3473 | 1.129487529149374 | 1.576520982866627 | 0.7907582105792549 | 1.1157094293566012 | 1.5593546849954465 | 0.7952901531113463 | no |
| Electricity client 0 | hour | 26304 | 5250 | 5.9920901567280245 | 11.18954429744584 | 0.7888377080095172 | 6.764110058158798 | 12.430031234195098 | 0.7394230434922053 | yes |
| Bike Sharing hourly | hour | 17379 | 3465 | 59.31380717651175 | 89.68871251796028 | 0.8345356951834966 | 55.89146993625497 | 87.94540623081888 | 0.840905539553876 | no |
| Daily min temperature | day | 3650 | 719 | 2.218903783662846 | 2.8742520769272777 | 0.5110962770107692 | 2.316358339535068 | 2.9629771303059296 | 0.480446547026502 | yes |
| Monthly sunspots | month | 2820 | 553 | 21.22998918727645 | 28.962496211079767 | 0.7255077158588397 | 20.940454126641292 | 29.987006255300724 | 0.705744640543083 | no |

The window-locked transformer beats or matches default XGBoost on test MAE for daily minimum temperature and electricity client 0.

### Full prefix

| Dataset | Sampling | Rows | Test origins | Full-history MAE | Full-history RMSE | Full-history R² | XGBoost MAE | XGBoost RMSE | XGBoost R² | Beats or matches |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ETTh1 OT | hour | 17420 | 3473 | 1.116001868955398 | 1.4976846756934177 | 0.8111618718590414 | 1.1157094293566012 | 1.5593546849954465 | 0.7952901531113463 | no |
| Electricity client 0 | hour | 26304 | 5250 | 6.4897393568690145 | 11.567201391331416 | 0.7743433371061236 | 6.764110058158798 | 12.430031234195098 | 0.7394230434922053 | yes |
| Bike Sharing hourly | hour | 17379 | 3465 | 75.03087097145124 | 108.730418101196 | 0.7568183193766078 | 55.89146993625497 | 87.94540623081888 | 0.840905539553876 | no |
| Daily min temperature | day | 3650 | 719 | 2.03880103754959 | 2.604255333831636 | 0.5986338326848941 | 2.316358339535068 | 2.9629771303059296 | 0.480446547026502 | yes |
| Monthly sunspots | month | 2820 | 553 | 27.22795967342131 | 36.81913113600846 | 0.556386011180538 | 20.940454126641292 | 29.987006255300724 | 0.705744640543083 | no |

The full-history transformer beats or matches default XGBoost on test MAE for daily minimum temperature and electricity client 0.

On daily minimum temperature the full-history MAE is lower than the window-locked MAE. On electricity client 0 the window-locked MAE is lower than the full-history MAE. Every forecast in both tables has a lower test MAE than the mean of the pre-test observations: ETTh1 7.040872897620491, electricity 21.026263916892777, bike 175.02485510219452, temperature 3.3991049847263763, sunspots 49.813707429174194.

ETTh1 oil temperature. Hourly, 17,420 rows. Pre-test prefix 13,936 rows, validation cut at 12,542. Selection uses 12,483 train origins and 1,383 validation origins. The refit and XGBoost use 13,877 origins. Test has 3,473 origins and 41,676 points. Window-locked test MAE 1.129487529149374, RMSE 1.576520982866627, R² 0.7907582105792549, epoch 21, selection MSE 0.03219291567802429, 123.19 seconds. Full-history test MAE 1.116001868955398, RMSE 1.4976846756934177, R² 0.8111618718590414, epoch 18, selection MSE 0.026299964636564255, 526.16 seconds. Default XGBoost test MAE 1.1157094293566012, RMSE 1.5593546849954465, R² 0.7952901531113463, 2.53 seconds.

Electricity client 0. Hourly, 26,304 rows. Pre-test prefix 21,043 rows, validation cut at 18,938. Selection uses 18,879 train origins and 2,094 validation origins. The refit and XGBoost use 20,984 origins. Test has 5,250 origins and 63,000 points. Window-locked test MAE 5.9920901567280245, RMSE 11.18954429744584, R² 0.7888377080095172, epoch 15, selection MSE 0.3798631429672241, 137.59 seconds. Full-history test MAE 6.4897393568690145, RMSE 11.567201391331416, R² 0.7743433371061236, epoch 9, selection MSE 0.38650989532470703, 662.63 seconds. Default XGBoost test MAE 6.764110058158798, RMSE 12.430031234195098, R² 0.7394230434922053, 1.77 seconds.

Bike Sharing hourly. 17,379 rows. Pre-test prefix 13,903 rows, validation cut at 12,512. Selection uses 12,453 train origins and 1,380 validation origins. The refit and XGBoost use 13,844 origins. Test has 3,465 origins and 41,580 points. Window-locked test MAE 59.31380717651175, RMSE 89.68871251796028, R² 0.8345356951834966, epoch 29, selection MSE 0.3275105059146881, 148.55 seconds. Full-history test MAE 75.03087097145124, RMSE 108.730418101196, R² 0.7568183193766078, epoch 29, selection MSE 0.29485610127449036, 705.86 seconds. Default XGBoost test MAE 55.89146993625497, RMSE 87.94540623081888, R² 0.840905539553876, 2.94 seconds.

Daily minimum temperature. Daily, 3,650 rows. Pre-test prefix 2,920 rows, validation cut at 2,628. Selection uses 2,569 train origins and 281 validation origins. The refit and XGBoost use 2,861 origins. Test has 719 origins and 8,628 points. Window-locked test MAE 2.218903783662846, RMSE 2.8742520769272777, R² 0.5110962770107692, epoch 11, selection MSE 0.514397144317627, 17.60 seconds. Full-history test MAE 2.03880103754959, RMSE 2.604255333831636, R² 0.5986338326848941, epoch 25, selection MSE 0.4192812144756317, 53.36 seconds. Default XGBoost test MAE 2.316358339535068, RMSE 2.9629771303059296, R² 0.480446547026502, 1.96 seconds.

Monthly sunspots. Monthly, 2,820 rows. Pre-test prefix 2,256 rows, validation cut at 2,030. Selection uses 1,971 train origins and 215 validation origins. The refit and XGBoost use 2,197 origins. Test has 553 origins and 6,636 points. Window-locked test MAE 21.22998918727645, RMSE 28.962496211079767, R² 0.7255077158588397, epoch 14, selection MSE 0.209747314453125, 17.49 seconds. Full-history test MAE 27.22795967342131, RMSE 36.81913113600846, R² 0.556386011180538, epoch 14, selection MSE 0.16860222816467285, 29.13 seconds. Default XGBoost test MAE 20.940454126641292, RMSE 29.987006255300724, R² 0.705744640543083, 2.32 seconds.

## Commands

From `/cursor/stores/self/ashlar`, after the CSVs are at `/tmp/ashlar-bench/reg_num-houses.csv` and `/tmp/ashlar-bench/clf_num-california.csv`:

```bash
JAX_PLATFORMS=cpu /tmp/gnet-venv/bin/python examples/benchmark.py
/tmp/deeptab-venv/bin/python examples/benchmark_torch.py
JAX_PLATFORMS=cpu /tmp/gnet-venv/bin/python examples/benchmark_scaling_sweep.py
JAX_PLATFORMS=cpu /tmp/gnet-venv/bin/python examples/benchmark_lr_embed.py
JAX_PLATFORMS=cpu /tmp/gnet-venv/bin/python examples/benchmark_tabm_borrow.py
JAX_PLATFORMS=cpu /tmp/gnet-venv/bin/python examples/benchmark_tabm_unshared.py
JAX_PLATFORMS=cpu /tmp/gnet-venv/bin/python examples/benchmark_tabm_seeds.py
JAX_PLATFORMS=cpu /tmp/gnet-venv/bin/python examples/benchmark_tabm_d_seeds.py --dump-splits
/tmp/deeptab-venv/bin/python examples/benchmark_tabm_d_seeds.py
JAX_PLATFORMS=cpu /tmp/gnet-venv/bin/python examples/benchmark_extra_regression.py --stage ashlar
/tmp/deeptab-venv/bin/python examples/benchmark_extra_regression.py --stage tabm
JAX_PLATFORMS=cpu /tmp/gnet-venv/bin/python examples/benchmark_wine_ordinal.py
JAX_PLATFORMS=cpu /tmp/gnet-venv/bin/python examples/benchmark_coverage.py --stage ashlar
/tmp/deeptab-venv/bin/python examples/benchmark_coverage.py --stage tabm
JAX_PLATFORMS=cpu /tmp/gnet-venv/bin/python examples/benchmark_horizon.py
JAX_PLATFORMS=cpu /tmp/gnet-venv/bin/python examples/benchmark_tabm_classification.py
JAX_PLATFORMS=cpu /tmp/gnet-venv/bin/python examples/benchmark_abalone_ordinal.py
JAX_PLATFORMS=cpu /tmp/gnet-venv/bin/python examples/benchmark_clear.py
JAX_PLATFORMS=cpu /tmp/gnet-venv/bin/python examples/benchmark_node.py
JAX_PLATFORMS=cpu /tmp/gnet-venv/bin/python examples/benchmark_rounds.py
JAX_PLATFORMS=cpu /tmp/gnet-venv/bin/python examples/benchmark_priors.py
JAX_PLATFORMS=cpu /tmp/gnet-venv/bin/python examples/benchmark_gradient_more.py
JAX_PLATFORMS=cpu /tmp/gnet-venv/bin/python examples/benchmark_gradient_suite.py
JAX_PLATFORMS=cpu /tmp/gnet-venv/bin/python examples/benchmark_gradient_multiclass.py
JAX_PLATFORMS=cpu /tmp/gnet-venv/bin/python examples/benchmark_gradient_multiclass_more.py
JAX_PLATFORMS=cpu /tmp/gnet-venv/bin/python examples/benchmark_gradient_categorical.py
JAX_PLATFORMS=cpu /tmp/gnet-venv/bin/python examples/benchmark_forecast.py
```

`examples/benchmark_scaling_sweep.py` writes `/tmp/ashlar-bench/scaling-sweep-results.json` and does not overwrite `jax-results.json`. `examples/benchmark_lr_embed.py` writes `/tmp/ashlar-bench/lr-embed-results.json`. `examples/benchmark_tabm_borrow.py` writes `/tmp/ashlar-bench/tabm-borrow-results.json`. `examples/benchmark_tabm_unshared.py` writes `/tmp/ashlar-bench/tabm-unshared-results.json`. `examples/benchmark_tabm_seeds.py` writes `/tmp/ashlar-bench/tabm-seeds-results.json`. `examples/benchmark_tabm_d_seeds.py` writes `/tmp/ashlar-bench/tabm-d-seeds-results.json` and does not refit Ashlar. `examples/benchmark_extra_regression.py` writes `/tmp/ashlar-bench/extra-regression-results.json`. `examples/benchmark_wine_ordinal.py` writes `/tmp/ashlar-bench/wine-ordinal-results.json` and does not refit the regression rows. `examples/benchmark_coverage.py` writes `/tmp/ashlar-bench/coverage-results.json` and does not overwrite the earlier result files. A later run adds `predictor`, `ordinal_predictor`, and the diamonds row, and leaves the earlier `residual` scores in place. `examples/benchmark_horizon.py` writes `/tmp/ashlar-bench/horizon-results.json` and does not refit classification, ordinal, XGBoost, or TabM. `examples/benchmark_tabm_classification.py` writes `/tmp/ashlar-bench/tabm-classification-results.json` and does not refit XGBoost or TabM-D. `examples/benchmark_abalone_ordinal.py` writes `/tmp/ashlar-bench/abalone-ordinal-results.json` and does not refit the residual regressor, XGBoost, or TabM-D. `examples/benchmark_clear.py` writes `/tmp/ashlar-bench/clear-benchmark-results.json` and does not refit XGBoost, the residual default, or the earlier result files. `examples/benchmark_node.py` writes `/tmp/ashlar-bench/node-results.json` and does not refit XGBoost, the residual default, or the earlier result files. `examples/benchmark_rounds.py` writes `/tmp/ashlar-bench/rounds-results.json` and does not refit XGBoost, the residual default, or the earlier result files. `examples/benchmark_priors.py` writes `/tmp/ashlar-bench/prior-results.json` and does not refit XGBoost, the residual default, wine, or the earlier result files. `examples/benchmark_gradient_more.py` writes `/tmp/ashlar-bench/gradient-more-results.json`. It fits the unchanged gradient-penalty classifier and a fresh default `XGBClassifier()` on bank-marketing, MagicTelescope, and credit, and does not refit the earlier tables, the residual default, or the earlier result files. `examples/benchmark_gradient_suite.py` writes `/tmp/ashlar-bench/gradient-suite-results.json`. It fits that same classifier and a fresh default `XGBClassifier()` on the fifteen tables in the section above, and does not refit the earlier tables, the residual default, or the earlier result files. `examples/benchmark_gradient_multiclass.py` writes `/tmp/ashlar-bench/gradient-multiclass-results.json`. It fits the multiclass gradient-penalty classifier and a fresh default `XGBClassifier()` on letter, pendigits, optdigits, satimage, and waveform-5000, and does not refit the earlier tables, the residual default, or the earlier result files. `examples/benchmark_gradient_multiclass_more.py` writes `/tmp/ashlar-bench/gradient-multiclass-more-results.json`. It fits that same classifier and a fresh default `XGBClassifier()` on the ten tables in the section above, and does not refit letter, pendigits, optdigits, satimage, waveform-5000, the residual default, or the earlier result files. `examples/benchmark_gradient_categorical.py` writes `/tmp/ashlar-bench/gradient-categorical-results.json`. It fits the gradient-penalty classifier with learned categorical embeddings and a fresh default `XGBClassifier()` on credit-g, cmc, car, mushroom, and adult, and does not refit the earlier tables, the residual default, or the earlier result files. `examples/benchmark_forecast.py` writes `/tmp/ashlar-bench/forecast-results.json`. It fits the window-locked transformer, the full-history transformer, and a fresh default lag `XGBRegressor` on ETTh1 oil temperature, electricity client 0, Bike Sharing hourly counts, daily minimum temperatures, and monthly sunspots, and does not refit the earlier tables, the residual default, or the earlier result files.

`examples/benchmark.py` writes the split caches and the Ashlar/XGBoost JSON. `examples/benchmark_torch.py` reads those caches. Both commands exited 0. No torch import or fit failed.

## What this does not show

- It does not show that neural nets beat trees, or the reverse. The comparison tables above the five-seed section are one `RandomState(0)` split. The five-seed section is five redraws of the unshared TabM recipe against default XGBoost, and only that comparison.
- It does not reproduce the published Grinsztajn curves. Those curves take the best model on the paper validation partition after about 400 random-search iterations, with neural nets trained on GPU for up to 300 epochs and gradient-boosted trees using early stopping and a large estimator cap, then aggregate over shuffles of that search. The headline comparison is one fold, `RandomState(0)`, and a 50-epoch ceiling. The five-seed section redraws only the unshared TabM recipe and default XGBoost.
- It does not match a named California Housing or Higgs number from TabR, TabM, or any other paper. RMSE here is on the transformed `medianhousevalue`. TabR's score is not a target.
- It does not use an official OpenML task split file. The procedure is the paper's, applied to the Hugging Face transformed CSV.
- The paper validation column was not a selection criterion. Using it to pick the row with the highest score would be a different experiment.
- Ashlar's two-trial search compares weight decay `1e-4` and `1e-3`. It does not change the no-search path into a search, and it is not a wide hyperparameter search. On california it kept the default weight decay.
- DeepTab and PyTabKit were not given their full default epoch budgets. A longer DeepTab run, or PyTabKit's HPO classes, could land elsewhere. The low DeepTab R² is the capped default MLP on this split, not a statement about the library.
- Wall times are not a JAX-versus-PyTorch speed result. They include compilation, library overhead, and different batch sizes and widths. TabM's minutes are a 32-member network on 4 CPUs. XGBoost's tenth of a second is 100 trees with no early stopping.
- The headline tables are one seed. The five-seed section reports sample standard deviations for the unshared Ashlar TabM recipe, default XGBoost, and PyTabKit TabM-D on houses. RealMLP and DeepTab were not repeated. Wine_quality, cpu_act, sulfur, and the coverage tables are one `RandomState(0)` split each, not the paper's multi-fold aggregate for test partitions of 6000 rows or fewer. Electricity and diamonds have test partitions above 6000 rows, so one fold matches appendix A.3 there. The predictor rescore and the label z-score rescore use those same splits. Houses in the label z-score section uses the cached split from `examples/benchmark.py`.
- Higgs, TabReD, and TabArena were not run.
