# Wine quality: one head change

The current best Ashlar expected-label test R² on this split is **0.41394214246508565**, from ordinal TabM with the cumulative-logit head. `XGBRegressor` is 0.4284963911647759. `XGBClassifier`, scored the same way as an expected label, is 0.48996919003701056.

## The change

Replace the proportional-odds head with an unrestricted softmax over the seven quality values.

The cumulative head gives each member one latent score and a set of cutpoints that do not depend on the row. Every class probability is a slice of that one logistic curve. The softmax head gives each quality value its own linear score from the member's hidden state. The class probabilities are a log-softmax of those scores. `predict_score` is still the expected label, `probabilities @ classes_`.

On the wine run the call is the ordinal TabM fit that scored 0.4139, plus `head="softmax"`:

```python
tabm(
    task="ordinal",
    head="softmax",
    prediction_policy="mean_probabilities",
    numeric_scaling="quantile_tabr",
    max_epochs=50,
    random_state=0,
)
```

`ordinal()` with no head argument stays the cumulative logit. The residual-MLP regression default stays the regressor.

## Why this target

`quality` is an integer from 3 to 9. In the file, 5 and 6 are 2,138 and 2,836 of 6,497 rows. Scores 3 and 9 are 30 and 5 rows. On the test split, 76.6% of rows are 5 or 6. A linear model on all 11 features reaches test R² 0.292. Alcohol, volatile acidity, and sulphates already reach 0.272. The conditional mean mostly moves between the crowded middle scores, and the tails are thin.

A single latent cannot raise the chance of a 7 without also pushing 8 and 9, and it cannot use one chemical for the 5-versus-6 cut and another for the 6-versus-7 cut. Those cuts are tied to the same score. The within-XGBoost gap on this split is the evidence that the link is costly: the same default trees score 0.4285 as a regressor and 0.4900 once each quality value has its own probability and the label is the expectation. The cumulative TabM only gained 0.0090 over the regression TabM (0.4049 to 0.4139). The distribution was estimated inside a one-dimensional family, so the expectation could not move the way the tree probabilities did.

Quantile scaling is already on the 0.4139 run, so this proposal does not treat the numeric transform as the missing piece. The features are 11 chemical measurements. The binding constraint in the current head is the shared latent.

## What this does not change

- The residual-MLP regression default: SiLU, feature dropout 0, weight decay `1e-4`, learning rate `1e-3`, mean/std, constant AdamW.
- The default ordinal head. `ordinal()` and `tabm(task="ordinal")` still use the cumulative logit.
- The wine split: the saved `RandomState(0)` medium split, 4547 / 585 / 1365. The paper-validation rows are not used to train or to pick a checkpoint.
- The TabM trunk on this run: `k=32`, `d_block=512`, three blocks, hidden dropout 0.1, ReLU, AdamW, learning rate `2e-3`, weight decay 0, batch 256, patience 16, `max_epochs=50`, `numeric_scaling="quantile_tabr"`, `random_state=0`.
- Label z-scoring and `share_training_batches=False`. Both stay regression-only.
- The metric. The comparison is test R² of `predict_score`, not the most likely label, and not a new loss weight or learning rate.

## The falsifying experiment

One CPU fit, the call above, on that same split. The hypothesis is that the shared latent is what held the expected label down.

The hypothesis fails if the softmax-head expected-label test R² does not exceed 0.41394214246508565 by more than 0.01. A score of 0.4239 or lower means this head was not the constraint, and the result stays in `benchmark.md` either way.

## Result

The fit returned expected-label test R² **0.3956767364524363**. The most likely label scored 0.23819329510026555. Training stopped at epoch 31 under patience 16. The best internal validation loss was 1.003374457359314, against 1.163618564605713 for the cumulative head that ran all 50 epochs.

0.3957 is 0.0183 below 0.4139. It does not exceed that score by 0.01. The hypothesis fails: on this split, giving each quality value its own score did not raise the expected label. The validation negative log-likelihood improved while the expected-label R² fell, so a better class likelihood under this early-stopping rule was not a better conditional mean.

## Regularization

Default `XGBClassifier` in XGBoost 3.1.2, after fit, uses depth 6, shrinkage 0.3, 100 trees, subsample 1, column sample 1 at tree, level, and node, L1 0, L2 1, minimum child weight 1, gamma 0, and no early stopping. The 0.4139 ordinal TabM uses AdamW at learning rate `2e-3`, weight decay 0, hidden dropout 0.1, feature dropout 0, and patience 16. That fit ran all 50 epochs, so patience did not restore an earlier checkpoint.

The one change on that call was `weight_decay=1.0`. Optax multiplies the penalty by the learning rate, so each step shrinks weights by a factor of about `1 - 0.002`. Depth, dropout, the cumulative head, and the learning rate stayed as they were. Subsample and column sample are already 1 on the XGBoost side, so those were not the knob.

Expected-label test R² is **0.3799832563545966**. The most likely label scored 0.281002356422856. Training ran all 50 epochs. The best internal validation loss was 1.4093810319900513.

The gap to the `XGBClassifier` expectation 0.48996919003701056 widened from 0.0760 to 0.1100. Weight decay did not close it. The residual-MLP regression default stayed the regressor.

## Learning rate from the loss surface

The next run is the same cumulative ordinal TabM call, including weight decay 0, with `learning_rate_schedule="surface"`.

```python
tabm(
    task="ordinal",
    prediction_policy="mean_probabilities",
    numeric_scaling="quantile_tabr",
    max_epochs=50,
    random_state=0,
    learning_rate_schedule="surface",
)
```

AdamW stays the optimizer. The first epoch, at most 20 steps, trains at `1e-4`. On the last of those steps the training objective is differentiated with dropout off. The directional curvature is `κ = uᵀHu` along `u = g / ||g||`. When `κ` is positive, that curvature is the steepness. Otherwise the steepness is the root-mean-square gradient. The rate for the remaining epochs is `(n_train / 10000) / steepness`, clipped to `[1e-5, 1e-1]`. `n_train` is the internal training partition. The reference size 10000 is the houses training-cap scale and is not multiplied by `2e-3`. A smaller partition or a steeper surface lowers the rate. Warmup Adam moments are reset after the probe. The warmed parameters are kept. The `tabm()` constructor rate `2e-3` stays on the estimator and is left out of the formula.

The residual-MLP regression default stays constant AdamW at `1e-3`. The comparison is still test R² of `predict_score` on the same wine split. The score has to exceed 0.41394214246508565 by more than 0.01 to beat that run.

## Surface-rate result

Warmup ran 15 steps at `1e-4` on 3637 internal training rows. The directional curvature on that batch was −0.00028266687877476215, so the steepness was the root-mean-square gradient, 6.333742931019515e-05. `(3637 / 10000) / 6.333742931019515e-05` is 5742.260207921903. The clip sets the AdamW rate to **0.1**. Weight decay stayed 0. The head stayed the cumulative logit.

Expected-label test R² is **0.28070258803405446**. The most likely label scored 0.15257517245508456. Training stopped at epoch 30 under patience 16. The best internal validation loss was 1.1453269720077515.

0.2807 is 0.1332 below 0.4139. It does not exceed that score by 0.01. The rate chosen from this batch sat on the upper clip, and the expected label fell.

## Empirical-Bayes prior and binned embeddings

The gap this section is aimed at is the cumulative ordinal TabM expected label, 0.41394214246508565, against the `XGBClassifier` expectation, 0.48996919003701056. The softmax head, weight decay 1, and the surface learning rate are already recorded above. Each one lowered the expected label. The softmax run is the useful failure: its validation negative log-likelihood improved while the expected label fell. Extra probability on the true score can still move the expectation the wrong way when the leftover mass sits on a far score. Weight decay pulled on every parameter and did not fix that. This design puts the prior on the class distribution.

Wine has no categorical column. The eleven inputs are continuous chemical measurements. The embedding path is still part of the fit. Three of those measurements are copied into train-only quantile bins and learned as categories. The columns are `alcohol`, `volatile.acidity`, and `sulphates`, under the names `alcohol_bin`, `volatile.acidity_bin`, and `sulphates_bin`. Those three already carry most of the linear signal on this split. The original numeric columns stay on `quantile_tabr`.

### The change

One fit. Same cumulative ordinal TabM call as the 0.414 run: AdamW, learning rate `2e-3`, weight decay 0, batch 256, patience 16, `max_epochs=50`, `random_state=0`, `numeric_scaling="quantile_tabr"`. Two arguments are added.

The class prior is Jeffreys' Dirichlet prior, one half pseudo-count on each training class. With internal-training counts `n_k` and `K` classes,

`π_k = (n_k + 1/2) / (n + K/2)`.

The cutpoints are the logits of the cumulative sums of `π`, so a zero latent reproduces `π`. Those cuts are held fixed with `stop_gradient`. The latent bias starts at 0, so the intercept starts on that prior. The member latent can still move the conditional distribution. `head_kwargs={"class_prior": "empirical"}` turns this on. `ordinal()` and `tabm(task="ordinal")` without that argument keep the learned linspace cuts.

The three bin columns use the existing learned-embedding tables. Eight quantile bins, edges at `1/8` through `7/8`, fit on the internal training rows only. The paper-validation rows and the test rows do not set the edges. Embedding width stays at the constructor default, 8. A value outside the outer edges stays in the end bin.

```python
tabm(
    task="ordinal",
    prediction_policy="mean_probabilities",
    numeric_scaling="quantile_tabr",
    max_epochs=50,
    random_state=0,
    head_kwargs={"class_prior": "empirical"},
    categorical_embeddings=[
        "alcohol_bin",
        "volatile.acidity_bin",
        "sulphates_bin",
    ],
)
```

### What this does not change

- The residual-MLP regression default: SiLU, feature dropout 0, weight decay `1e-4`, learning rate `1e-3`, mean/std, constant AdamW.
- The default ordinal head. Cuts stay learnable unless `class_prior` is set.
- Weight decay, the learning rate, and the cumulative head on this run. The failed weight-decay, surface-rate, and softmax settings are not combined in.
- The wine split: the saved `RandomState(0)` medium split, 4547 / 585 / 1365. The paper-validation rows are not used to train or to pick a checkpoint.

### The falsifying check

The hypothesis fails if the expected-label test R² does not exceed 0.41394214246508565 by more than 0.01. The score is recorded either way. There is no second fit and no search over the pseudo-count, the bin count, or the embedding width.

## Prior-and-embedding result

The fit returned expected-label test R² **0.4028560414884015**. The most likely label scored 0.2800294232109789. Training stopped at epoch 30 under patience 16. The best internal validation loss was 1.044643759727478.

The embedded columns were `alcohol_bin`, `volatile.acidity_bin`, and `sulphates_bin`. Wine has no stored categorical field, so those are eight-bin copies of `alcohol`, `volatile.acidity`, and `sulphates`. The Jeffreys prior on classes 3 through 9 was 0.003982969372338965, 0.03364922400769125, 0.32866364510369456, 0.4374399120999863, 0.16494986952341711, 0.030627661035572035, 0.0006867188572998214.

0.4029 is 0.0111 below 0.4139. It does not exceed that score by 0.01. The hypothesis fails. The validation negative log-likelihood improved, from 1.1636 on the cumulative run to 1.0446, while the expected label fell. The gap to the `XGBClassifier` expectation 0.48996919003701056 is 0.0871, wider than the 0.0760 gap of the untouched ordinal TabM. The residual-MLP regression default stayed the regressor.

## Late bottleneck

The 0.4139 cumulative TabM reads the full block, width 512, in the ordinal head. This fit inserts one narrow projection after those three blocks and before that head. `bottleneck=True` selects the default width, 32. Each member has its own linear map from 512 to 32, then a ReLU. The cumulative head reads those 32 units. The projection has no dropout.

The rest of the call is the 0.4139 run: `tabm(task="ordinal", prediction_policy="mean_probabilities", numeric_scaling="quantile_tabr", max_epochs=50, random_state=0)` plus `bottleneck=True`. Weight decay stays 0, the learning rate stays `2e-3`, and the head stays the cumulative logit. No class prior, no binned embeddings, no softmax head, and no surface learning rate. The width was fixed at 32 before the test score. There is no sweep.

The residual-MLP regression default leaves `bottleneck` unset, so `tabular()` still has no late projection. This width is on for this wine fit only. That residual default changes only if the expected-label test R² exceeds 0.41394214246508565 by more than 0.01.

## Late-bottleneck result

The fit returned expected-label test R² **0.41855174134115225**. The most likely label scored 0.3111632859910447. Training used all 50 epochs. The best internal validation loss was 1.1645258665084839. The bottleneck width was 32.

0.4186 is 0.0046 above 0.41394214246508565. The bar for changing the residual default is more than 0.01 above that score, so the residual-MLP regression default stayed the regressor. The gap to the `XGBClassifier` expectation 0.48996919003701056 is 0.0714. The untouched ordinal TabM gap to that expectation is 0.0760.
