---
cursor:
  subagentId: "bc-7e0eb9a8-52e0-555d-a828-44369783e1d4"
---

# What Ashlar could borrow from TabM-D

The houses gap this note is about: Ashlar's no-argument residual MLP, feature dropout `0` and SiLU, test R² **0.7615** on the Grinsztajn medium houses split (`benchmark.md`). PyTabKit TabM-D, stopped at 50 epochs, scored **0.8366**. XGBoost defaults scored **0.8268**. The gap to TabM-D is about 0.075 R². Nothing here was refit.

Two recipes are easy to mix up. **TabM-D** is PyTabKit's name for one fixed configuration. The TabM paper does not define a model called TabM-D. The paper's published TabM runs tune depth, width, dropout, learning rate, and weight decay. The package notebook ends on piecewise-linear embeddings and a different optimizer. The 0.8366 row is the PyTabKit configuration, plus the benchmark's overrides.

## Sources

Pinned by reading the files, not by rerunning them.

| Source | Pin |
|---|---|
| PyTabKit 1.7.3 | tag `v1.7.3`, commit `c126ea51187c5080b91f28d352481dbd3b2194b0`. `default_params.py` (`TABM_D_REG`), `tabm_interface.py`, `nn_models/tabm.py`, `nn_models/models.py` (`quantile_tabr`), `utils.py` (`TabrQuantileTransformer`), `sklearn_interfaces.py` (`TabM_D_Regressor`) |
| TabM package and paper docs | `yandex-research/tabm` commit `28e47ae301c92ec37787dde1ce923a0793f405b4` (10 Nov 2025, `__version__ = 0.0.3`, current `main` at the time of this read). `tabm.py` `TabM.make`, `example.ipynb`, `README.md`, `paper/README.md`, `paper/bin/model.py`, `paper/lib/data.py`, `paper/exp/tabm/california/0-tuning.toml`, `paper/exp/tabm/california/0-evaluation/0.toml` |
| Ashlar | `recipes.py` `TABM_PAPER` and `tabm()`, `layers.py` `TabMBackbone`, `train.py`, `objectives.py`, `heads.py`, `preprocess.py`, `estimator.py`. Read only |

`benchmark.md` says the 0.8366 fit was `TabM_D_Regressor(device="cpu", random_state=0, n_epochs=50, n_cv=1, n_refit=0, n_repeats=1, verbosity=0)`. Every other constructor argument stayed at the class default, including `val_fraction=0.2`.

## What the 0.8366 fit set

`TABM_D_REG` is the same dict as `TABM_D_CLASS`. The comment in `default_params.py` says the dict comes from `example.ipynb`. Several notebook lines differ; those differences are listed after the table. `get_config` fills a constructor `None` from this dict and then drops keys that are still `None`. `share_training_batches` is not in the dict, so the interface default `False` is what runs.

| Piece | TabM-D value that produced 0.8366 |
|---|---|
| Architecture | `arch_type='tabm'`: BatchEnsemble MLP. Block is linear, ReLU, dropout. `k=32`, `d_block=512`, `dropout=0.1`. `n_blocks='auto'` resolves to **3** when there are no piecewise embeddings, **2** when there are. First-layer input scaling is random signs, one draw per feature chunk; every later scaling is ones. The `k` biases of a layer share one initialization draw. Output is an unshared linear map per member (`NLinear`). |
| Embeddings | `num_emb_type='none'`. No piecewise-linear, periodic, or linear-ReLU embeddings. `d_embedding=16` and `num_emb_n_bins=48` are stored and unused on this path. Categorical columns, if any, are one-hot inside the model. Houses has none. |
| Optimizer | AdamW, learning rate `2e-3`, weight decay `0`. PyTorch defaults for the moments (`0.9`, `0.999`). Parameter groups give biases weight decay 0, which does nothing while the global weight decay is already 0. No gradient clipping (`gradient_clipping_norm=None`). `allow_amp=False`, `compile_model=False`. |
| Schedule | None. The learning rate stays `2e-3` until the epoch cap or patience. |
| Regularization | Hidden dropout `0.1` after each ReLU. Weight decay 0. No feature dropout. Early stopping: patience **16**, checkpoint is the best validation score, not the last epoch. The benchmark cap is **50** epochs. The class cap is `1_000_000_000`; the benchmark replaced it. |
| Ensembling | One fitted network, `k=32` members. Training loss is the mean of the member losses (MSE for regression), not the loss of the averaged prediction. `share_training_batches=False`: each member draws its own permutation, and the step's labels follow that permutation. Prediction averages the 32 members. Classification averages probabilities; the houses fit is regression. `n_cv=1`, `n_refit=0`, `n_repeats=1`, so there is no outer seed ensemble. |
| Preprocessing | `tfms=['quantile_tabr']` only, fit on the internal training rows. See the next section. Regression labels are then standardized with the training mean and the population standard deviation (`std` with `correction=0`), and predictions are mapped back. That standardization is in `TabMSubSplitInterface`, not in `TABM_D_REG`. The dict does not set `transformed_target`, and the TabM constructor has no such argument. Constant numeric columns after the feature transform are dropped. Validation metric is RMSE of the averaged prediction in the original label units (the implemented score is the negative of that RMSE). The training loss stays MSE on the standardized labels. |

### `quantile_tabr` is a specific map

`TabrQuantileTransformer` in PyTabKit 1.7.3, `random_state=0`:

- Knot count `max(min(n_train // 30, 1000), 10)`. On 8,000 internal training rows that is **266**, not 1,000.
- `output_distribution="normal"`, `subsample=1_000_000_000`.
- Before `fit` only, add Gaussian noise with per-column scale `1e-3 / max(column_std, 1e-3)`. `transform` does not add noise.
- No median imputation and no second z-score.

The implementation comment points at TabR `lib/data.py`, not at the TabM paper's `noisy-quantile`.

### What the paper and the notebook set instead

Facts from the same TabM commit. These did not produce the 0.8366 row.

**Paper training script** (`paper/bin/model.py`, `paper/lib/data.py`). Fixed for the published TabM runs: batch size 256, patience 16, `n_epochs = -1` (no epoch cap), AdamW with no learning-rate schedule, gradient clipping norm **1.0**, AMP on in the California config, `k=32`, `arch_type='tabm'`, `share_training_batches=false` in `exp/tabm/california/0-evaluation/0.toml`. The code default for sharing is `True`; the published TabM config turns it off. The ♠ experiments are the shared-batch variant. Feature policy is `noisy-quantile`: scikit-learn `QuantileTransformer`, normal output, knot count `max(min(n_train // 30, 1000), 10)`, subsample `1e9`, plus **absolute** `N(0, 1e-5)` noise on the training matrix only. NaNs become 0 after the transform. Constant columns are removed. Regression labels are standardized with the training mean and `numpy.std` (population). Early stopping maximizes validation score. The default regression score in that loader is RMSE, stored with a minus sign. Depth, width, dropout, learning rate, and weight decay are tuned (California's evaluated run: 3 blocks, width 400, dropout about 0.208, learning rate about `8.7e-4`, weight decay about `0.038`). That is one tuned dataset, not a default, and it is not this Grinsztajn split. The paper's multi-seed ensemble (`bin/ensemble.py`) is a further average over training seeds. TabM-D's benchmark fit did not do that.

**`TabM.make` defaults** when no embeddings are passed: 3 blocks, width 512, dropout 0.1, ReLU, `k=32`, `arch_type='tabm'`, first scaling `random-signs`. With any numerical embedding module those defaults become **2** blocks and first scaling `normal`.

**README optimizer line:** AdamW, learning rate `0.002`, weight decay `0.0003`. That is the package recommendation. TabM-D uses weight decay 0.

**`example.ipynb`.** The preprocessing cell fits a normal quantile transform with the same knot formula and absolute `N(0, 1e-5)` noise, and the text says regression labels should be standardized. The model cell assigns `num_embeddings` four times; the last assignment is the one that executes: `PiecewiseLinearEmbeddings`, 48 bins, `d_embedding=16`, `activation=False`, `version='B'`. `TabM.make` then uses 2 blocks and normal first scaling. The optimizer cell is AdamW at `2e-3` with weight decay `3e-4` and gradient clipping `1.0`, on all parameters, not the bias-excluded groups. The training cell sets `share_training_batches = True` and says `True` is the efficiency default, with `False` sometimes better. Batch size 256 and patience 16 match TabM-D. AMP and `torch.compile` are off.

**PyTabKit's own HPO class** is a different sampler: `tabm-mini`, piecewise-linear embeddings, tuned learning rate, weight decay, depth, width, dropout, and bin count. The benchmark did not run it.

## What Ashlar already has

Two Ashlar paths matter. The scored path is `tabular()` / `DEFAULT_RECIPE`. `tabm()` / `TABM_PAPER` was not the model in the table.

**Already aligned with TabM-D, on `tabm()` only.** Batch-ensemble trunk: linear, ReLU, dropout, `k=32`, `d_block=512`, `dropout=0.1`, `n_blocks=3`, first `r` random signs with one draw per feature chunk, later `r` and every `s` ones, shared bias initialization. Unshared linear head per member. Training loss is the mean of member squared errors. `predict` averages members. AdamW, learning rate `2e-3`, weight decay 0, constant schedule, batch 256, patience 16, no clipping. `layers.py` records this as the equations of `tabm` 0.0.3 at commit `28e47ae`. `max_epochs` on that recipe is 200. The benchmark cap that produced 0.8366 was 50. The recipe does not set `numeric_scaling`, `feature_dropout`, or `learning_rate_schedule`, so the estimator defaults apply: z-score, feature dropout 0, constant rate.

**On the scored default, and still on `tabm()`.** Median impute, then mean and population standard deviation. No label standardization (`design.md`: predictions stay on the scale of `y`). One shared minibatch for every member (`ensemble_view` copies the same rows). Early stopping watches the member-mean validation loss, patience 10 on the default and 16 on `tabm()`. No piecewise-linear or periodic embeddings. `num_embeddings` is `None` unless the caller passes `linear_relu`, which is a per-feature linear map and a ReLU, not piecewise-linear. `numeric_scaling='quantile'` exists and is a different map: scikit-learn `QuantileTransformer`, `n_quantiles=min(1000, n_rows)`, no noise, fit after median imputation. The houses probe of that map on the old regularized residual MLP (ReLU, feature dropout 0.1, learning rate `1e-3`) scored test R² **0.6519** and stopped at epoch 24. That probe is not a measurement of `quantile_tabr`, of label standardization, or of `tabm()`.

The SiLU default that scored 0.7615 is the residual MLP: width 64, one member, learning rate `1e-3`, weight decay `1e-4`, batch 64, feature dropout 0, z-score, 50-epoch cap. It does not use the TabM trunk.

## Ranked borrowings

Ranked for movement from 0.7615 toward 0.8366 on this split. Cost is new code plus how many other knobs the change would confound. A change Ashlar already implements is not listed as a borrow.

1. **`quantile_tabr`, train rows only, in front of the existing `tabm()` trunk.** Fact: this is the feature map in the 0.8366 fit, and it is not the map that scored 0.6519. The knot count, the `1e-3 / std` noise, and the absence of a following z-score are the differences. Fact: the paper's `noisy-quantile` uses the same knot formula and a smaller, scale-free `1e-5` noise. Guess: on these eight raw count columns, this map is the largest single piece of the 0.075 gap. The 0.6519 result says a normal quantile map can also hurt a different network, so the probe has to be this transformer on `tabm()`, not another 1,000-knot map on the residual MLP.

2. **Standardize regression labels with the training mean and population standard deviation, and invert that map on predict.** Fact: the notebook calls this highly recommended, the paper script always does it, and TabM-D's fit loop always does it for regression even though `TABM_D_REG` never mentions a target transform. The learning rate `2e-3` in that recipe is the step size on the standardized labels. Ashlar applies that learning rate, when `tabm()` is used, to the raw target. Guess: the houses target is already near unit variance (test variance about 0.33 in `gap-analysis.md`), so a second standardization is a smaller effect than the feature map. It is still part of the recipe that the learning rate was written for. Early stopping on RMSE of the averaged members, in the original units, is the checkpoint rule that goes with it. Ashlar checkpoints the mean of the member losses. Those two scores can pick different epochs. Guess: the checkpoint rule is the smaller half of this item.

3. **Give each member its own training permutation (`share_training_batches=False`).** Fact: that is the 0.8366 setting and the published California TabM config. Ashlar always shares one minibatch across members, which matches the notebook's efficiency default and the paper's ♠ variant, not the main TabM run. The forward width stays `(batch, k, d_block)`; the rows in the member axis change. Guess: this is a real regularizer and a smaller gain than the feature map. It is a poor first probe on the width-64 residual MLP, which has one member.

Below the line, and not a way to reproduce 0.8366:

- **Piecewise-linear embeddings** (version B, 48 bins, embedding size 16, no activation, then 2 blocks and normal first scaling). Fact: they are the notebook's last `num_embeddings` assignment, the README's usual best-effort embedding, and the paper's TabM†. Fact: TabM-D leaves them off, and the 0.8366 row did too. Borrowing them changes the model away from the row that set the gap. `linear_relu` is already in Ashlar and is not this module.
- **Gradient clipping at 1.0 and AdamW weight decay `3e-4`.** Fact: both are in the notebook, clipping is in the paper configs, and the README's default weight decay is `3e-4`. Fact: TabM-D sets clipping to `None` and weight decay to 0, and a PyTabKit issue on clipping (the 1.7.x line) says 1.0 was not better. Matching 0.8366 means leaving both off. `tabm()` already has weight decay 0.
- **A longer epoch cap.** Fact: TabM-D's own cap is a billion epochs with patience 16. Fact: the 0.8366 number replaced that cap with 50, the same ceiling as Ashlar's default. A longer run would answer a different question than the measured gap.
- **Switching the scored default to `tabm()`'s width, member count, dropout, learning rate, and batch size.** Fact: that code is already in `recipes.py`, and it was not the 0.7615 fit. It is the trunk the three borrowings above attach to. Running it alone still uses a z-score and the raw target, which `TABM_PAPER` already says is short of the paper recipe.

## Facts and guesses, separated

**Facts.** The 0.8366 model is BatchEnsemble TabM, 3×512, dropout 0.1, ReLU, `k=32`, no numerical embeddings, AdamW at `2e-3` with weight decay 0, no schedule, no clipping, batch 256, patience 16, a 50-epoch cap, distinct member shuffles, mean of member losses, mean of member predictions, `quantile_tabr` on the features, and a train-only z-score of `y` that is inverted at prediction. Ashlar's `tabm()` already has the trunk, the loss, the average, the optimizer numbers, the batch, and the patience. It does not have `quantile_tabr`, the label z-score, or distinct member shuffles. The scored 0.7615 model is the SiLU residual MLP, not `tabm()`. Ashlar's existing quantile option, measured on an older residual MLP, scored 0.6519.

**Guesses.** That `quantile_tabr` on `tabm()` moves houses most of the way from 0.7615 to 0.8366. That label standardization matters less here because the target is already near unit scale, and that it still belongs in the same run because the learning rate assumes it. That distinct member shuffles help, by less than the feature map. That piecewise-linear embeddings, clipping, weight decay `3e-4`, or a longer cap are how this seed reaches 0.8366.
