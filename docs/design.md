# Design

This document answers the brief's design questions for the prototype in `/cursor/stores/self/ashlar`. Facts describe the code that was installed and run. Judgments say why that contract was chosen over the brief's sketch. `gnet-tabular` was the working name; the import is now `ashlar`.

## API relative to the sketch

The brief sketched `ModelSpec` plus `DistributionalEstimator`. That sketch was not treated as a required API.

**Fact.** There is one sklearn-style class, `ashlar.TabularEstimator`. After `fit`, `core_` is an ordinary Equinox `PredictiveCore`. `prepare` returns the fitted numeric map and one-hot categories. Learned categorical indices are not in that pair. `forward_transformed` applies the fitted numeric transform and the core. `predict` returns a point. `predict_dist` returns a distribution object. `predict_raw` returns the head output with the member axis intact.

**Judgment.** A second estimator class and a spec object would duplicate `get_params` without changing the three workflows. Task-specific subclasses were also rejected: classification, regression, the normal head, and the custom linked head share one shell so a custom head does not require a new estimator type. That is the contrast with DeepTab's config-plus-`BaseModel`-plus-estimator path.

## Encoder, backbone, head, objective, and prediction

**Fact. Encoder.** `FeatureEncoder.fit` runs on the training partition only. `transform` returns an `EncodedBatch`: raw numerics (NaNs still present), one-hot columns, integer indices for learned categoricals, and padded set tokens with a mask. `NumericTransform`, an Equinox module, replaces numeric NaNs with the training median. The default map then centers on that median, divides by the training interquartile range, and applies a smooth clip that saturates at ±3. `numeric_scaling="standard"` is the mean and population standard deviation, with no clip. Category lookup stays on the host. The embedding tables themselves are inside the core, so their weights take gradients. Index values do not.

**Fact. Default recipe.** `TabularEstimator()` matches `DEFAULT_RECIPE` in `ashlar.recipes`: regression, `backbone="residual_mlp"`, `categorical_embeddings=None`, no `set_features`, iid split, `validation_size=0.2`, batch 64, 50 epochs, patience 10, learning rate `1e-3`, weight decay `1e-4`, feature dropout `0`, numeric scaling `robust`, `standardize_target=True`. The residual MLP activation is SiLU unless `backbone_kwargs` sets `activation` to `relu` or `gelu`. Hidden dropout stays 0 unless `backbone_kwargs` sets it. Integer columns stay numeric unless named in `categorical`, `categorical_embeddings`, or `set_features`. Object, bool, and other non-numeric columns are categorical. `categorical_embeddings=None` learns an embedding for every categorical column that is not a set feature. `[]` one-hots those columns. A non-empty list embeds the named columns and one-hots the rest. `num_embeddings` stays off.

**Fact. Predictor default.** The no-argument path now does two train-only predictor treatments and leaves the rest of the recipe alone. Categorical columns get a learned embedding of width 8 instead of a one-hot. Numeric columns use the RealMLP-TD front end: median center, interquartile scale, and `z / sqrt(1 + z²/9)`. On the houses split, RealMLP-TD, which uses that front end among other differences, scored 0.8247 beside XGBoost at 0.8268. The mean/std residual on the same split scored 0.7615065490594078. A normal quantile map on that residual scored 0.6518968125086434, and `linear_relu` numeric embeddings scored 0.7451792371862184, so both stay opt-in. `numeric_scaling="quantile_tabr"` stays the explicit TabM regression recipe. `tabm()` does not set `numeric_scaling` or `categorical_embeddings`, so a bare `tabm()` inherits robust scale and categorical embeddings. Pass `numeric_scaling="standard"` or `categorical_embeddings=[]` to restore the previous predictor handling. For robust scaling the schema field `std` stores the interquartile divisor, not the population standard deviation. The mean is still stored. A zero interquartile range uses half the training range, and a constant column uses divisor 1. Old checkpoints omit `numeric_scaling` and still load as `standard`. The schema records each column's encoding, so an old one-hot checkpoint still predicts as one-hot. No new Equinox array leaf was added. The same-split coverage rescore is in `benchmark.md`. Regression `tabular()` also z-scores the training labels and inverts that map in `predict`. `classifier()` and `ordinal()` set `standardize_target=False`. Omitting the flag does the same, because the label z-score is regression-only. The reasons, and the strategies left for later, are in `horizon.md`.

**Fact. Learning-rate schedule.** The default is `constant` AdamW at the constructor rate. `linear` decays that rate to zero across the epoch budget. `surface` is off unless requested. It keeps AdamW, warms up at `1e-4` for one epoch and at most 20 steps, and then sets the rate to `(n_train / 10000) / steepness`, clipped to `[1e-5, 1e-1]`. `n_train` is the internal training partition. Steepness is the directional curvature `uᵀHu` on one warmup batch when that curvature is positive, and the root-mean-square gradient otherwise. Dropout is off for that measurement. The constructor rate stays on the estimator and is left out of the formula. `tabular()` and `DEFAULT_RECIPE` keep `constant`.

**Fact. Default regularization.** Weight decay and early stopping are on for `TabularEstimator()` and `ashlar.classifier()`. Weight decay `1e-4` is the AdamW penalty; `weight_decay=0` turns it off. Feature dropout defaults to `0`. Setting it above 0 drops each concatenated input field on a training step and rescales the kept fields. `predict` passes `key=None`, so it never drops fields. Early stopping monitors the full held-out slice (`validation_size=0.2`) and restores the best checkpoint after `patience=10` epochs without improvement. `patience=0` disables that stop. `validation_size=0` trains on every row. The houses sweep in `benchmark.md` is why feature dropout is off and the residual activation is SiLU. That is not a claim that the resulting fit matches a gradient-boosted tree.

**Fact. Classification default.** `ashlar.classifier()` and `CLASSIFIER_RECIPE` match `DEFAULT_RECIPE` except `task="classification"` and `prediction_policy="mean_probabilities"`. `TabularEstimator(task="classification")` is the same task. `predict` returns the original labels. `predict_proba` returns class probabilities with one column per class in `classes_` order. Binary targets use two classes. Missing values and unseen categories use embedding index 0. They use the one-hot unknown bucket when the column is one-hot, which is `categorical_embeddings=[]` or a column left out of an explicit embedding list.

**Fact. Search.** `fit(..., search=None)` and `search=False` do not search. `search=True` uses `DEFAULT_SEARCH`: two trials, two epochs, patience 0, weight decay `1e-4` and `1e-3`. Learning rate and feature dropout stay at the constructor values unless the space lists them. A dict replaces `n_trials`, `max_epochs`, `patience`, and, when present, the whole `space`. The grid is the cartesian product in key order, truncated to `n_trials`. Allowed keys are `learning_rate`, `weight_decay`, `feature_dropout`, `batch_size`, and `backbone_kwargs`. Regression, normal, and ordinal trials are ranked by validation loss, lower better. For ordinal that loss is the cumulative-logit negative log-likelihood. Classification trials are ranked by weighted accuracy, higher better: `sum(w * 1[predicted = label]) / sum(w)` on the validation partition, using the same effective row weights as the weighted cross-entropy. Those weights are `sample_weight` times one class-weight multiplier. The accuracy score does not multiply class weights again. The final train after search still minimizes the task loss, which for classification is weighted cross-entropy, not squared error. Trial weights are discarded. The constructor attributes stay as the user set them. `resolved_` and `search_["chosen"]` record what was trained. Search raises if there is no validation partition. `resume=True` cannot be combined with search.

**Fact. Learned categoricals and sets.** `categorical_embeddings=None` sends every non-set categorical through `CategoricalEmbeddings` on the core. A list names the columns to embed, and the other categoricals stay one-hot. `[]` one-hots every categorical. `set_features` names columns whose cells are lists of labels. `SetEmbeddings` mean-pools those tokens with a mask. Padding and unknown share index 0; padding has mask 0. An empty set is a zero vector. `set_max_length` defaults to 8 and caps the training-partition maximum. A column cannot be both an embedding and a set feature. Fitted names are `embedded_columns_`. The constructor value stays what `get_params` returns.

**Fact. Field-set backbones.** After concatenation, each column is one chunk: a numeric column, an one-hot column, a learned embedding, or a pooled set. `backbone="transformer"` projects each chunk to `d_block`, runs pre-norm self-attention with no positional encoding, and mean-pools to `(batch, 1, d_block)`. `backbone="set"` uses the same projections and a mean pool plus a residual map, with no attention. Column identity is the per-field projection, not token order. **Judgment.** These two are the feature-set tools. They are not a catalogue entry beyond the residual MLP and TabM, and they are not the default.

**Fact. Backbone.** Built-in names are `residual_mlp`, `tabm`, `transformer`, and `set`. A user backbone is a class constructed as `Backbone(d_in, *, key, **backbone_kwargs)` and called as `backbone(x, *, key)` on the concatenated field vector `(batch, features)`. It must expose integer attributes `k` and `d_out`. `TabMBackbone` expands a rank-2 input with `ensemble_view`. `ResidualMLP`, `FeatureTransformer`, and `SetEncoder` keep an explicit member axis of length 1. A string outside those four names, or `backbone_kwargs` containing `scaling_chunks`, `feature_chunks`, or `arch_type`, raises `AshlarError` before any fitted attribute is set. `scaling_chunks` and `feature_chunks` are derived from the fitted schema. `arch_type` other than batch-ensemble TabM is not implemented, so the argument is refused rather than ignored. The set encoder raises if `n_blocks` is not 1, because it pools once.

**Fact. Head.** Built-in and user heads share one constructor:

```text
Head(d_in, k, *, n_events, n_classes, key, **head_kwargs)
```

The call is `head(hidden, *, key)` with `hidden` shaped `(batch, member, hidden)`. Built-in outputs:

| Head | Output |
|---|---|
| `PointHead` | `(batch, member, event)` |
| `LogitsHead` | `(batch, member, class)`, at least 2 classes |
| `NormalHead` | `(batch, member, event, 2)`, location and unconstrained scale |

A wrong rank or a member count that does not equal `k` raises `ShapeError` inside the head. The core also rejects encoded widths that do not match `n_numeric` and `n_categorical`.

**Fact. Supported task combinations.**

| `task` | Default head | Objective | `predict` | `predict_dist` policy |
|---|---|---|---|---|
| `regression` | `point` | member-mean squared error, then mean over events | mean of members; univariate event squeezed to `(batch,)` | not the default path |
| `classification` | `logits` | member-mean cross-entropy | `predict_proba`: `mean_probabilities` (default) or `mean_logits` | not used |
| `ordinal` | `ordinal` | member-mean cumulative-logit negative log-probability | most likely label; `predict_score` is the expected label | not used |
| `normal` | `normal` | member-mean negative log-density, summed over events | mean of member locations, univariate squeezed | `mixture` (default) or `mean_parameters` |
| `custom` | user class, required | user function `(raw, y, sample_weight) -> scalar` | user `point_fn(raw)` | user `dist_fn(raw)` when provided |

A string head that does not match the built-in task is rejected. `custom` requires a head class, an objective, and `point_fn`. `class_weight` is rejected unless the task is `classification` or `ordinal`.

**Fact. Ordinal default.** `ashlar.ordinal()` and `ORDINAL_RECIPE` match `DEFAULT_RECIPE` except `task="ordinal"` and `prediction_policy="mean_probabilities"`. Labels are real numbers. `classes_` is those values in sorted numeric order. The default head is a cumulative logit: one latent per member and strictly increasing cutpoints that do not depend on the row. `head="softmax"` replaces that latent with one linear score per class. `head_kwargs={"class_prior": "empirical"}` replaces the learned cuts with a frozen Jeffreys prior, `π_k = (n_k + 1/2) / (n + K/2)`, from the internal training counts, and starts the latent bias at 0. Omitting that argument leaves the cuts learnable. `predict` returns the most likely label. `predict_proba` averages member probabilities. `predict_score` returns `probabilities @ classes_`. The no-argument residual default stays regression.

**Fact. Objectives.** Training optimizes the mean of member losses, not the loss of the averaged prediction. Sample weights are applied after that member mean. The normal objective sums independent event coordinates, which is their joint density. Scale is `softplus(raw_scale) + 1e-4`.

**Fact. Prediction policies.** Regression accepts only `mean_members`. Classification accepts `mean_probabilities` or `mean_logits`. Ordinal accepts `mean_probabilities`. The normal task accepts `mixture` or `mean_parameters`. `predict_dist(..., policy=...)` overrides the stored policy for that call. `NormalMixture.variance` is `E[scale^2] + mean((loc - mean)^2)` on each event. `MeanParameterNormal.variance` is the square of the averaged scale. Members are not posterior draws, and neither policy is a calibrated split of aleatoric and epistemic uncertainty.

**Judgment.** Squeezing a univariate event axis makes `predict` match the usual one-target sklearn shape. Multivariate targets stay `(batch, event)`. Keeping the member axis inside `predict_raw` is what makes the mixture policy a real object instead of a comment.

## Axes and reductions

**Fact.**

| Axis | Where it lives |
|---|---|
| Batch | axis 0 of every array the core sees |
| Feature | last axis of the encoded input, `(batch, features)`, after numeric embeddings and one-hot categories are concatenated |
| Ensemble member | axis 1, always present. Residual MLP uses `k = 1`. Missing `k` is not inferred by broadcasting a rank-2 tensor into a linear map |
| Distribution parameter | last axis of `NormalHead`, size 2: location, then raw scale. It is not an event and not a member |
| Event | axis 2 of point and normal outputs. Independent events are reduced by a mean in squared error and by a sum in the normal log-density. They are never averaged together with members |

`ensemble_view` accepts `(batch, features)` and expands it to `k` identical views, or accepts `(batch, k, features)`. Any other member count or rank raises `ShapeError`. `LinearBatchEnsemble` rejects a rank-2 input. `weighted_mean` requires the per-example loss and the weight vector to have the same shape.

The weight reduction is named `batch_weight_sum`:

```text
sum_i (w_i * loss_i) / sum_i w_i
```

inside the minibatch that was drawn. Each batch is renormalized by its own weight mass. That equals the full-sample weighted risk only when one batch contains every training row. Padding rows use weight 0 and leave both sums. An all-zero training weight or a non-finite loss raises `WeightError`.

Class weights are a separate multiplier. For classification and ordinal classification the effective weight is `sample_weight * class_weight[label]`, applied once. There is no weighted sampler, so the two cannot double-count. `class_weight` may be `None`, `"balanced"` (`n_rows / (n_classes * count)` on the training partition), a dict keyed by the original labels, or an array in sorted class-index order. Explicit `sample_weight=None` means a vector of ones on every task, including regression and the normal head.

Zero-weight rows stay in the training partition and still affect the fitted schema. Drop the rows to exclude them.

**Judgment.** Refusing a silent broadcast is worth a stricter constructor. A wrong member axis that happens to multiply is a wrong ensemble, not a convenience.

## Custom modules, cloning, and persistence

**Fact.** Custom backbones, heads, objectives, `point_fn`, and `dist_fn` must be classes or functions with an importable `__module__` and `__qualname__`. `save` writes that pair as `module:qualname` in `metadata.json`. `load` imports it. Closures and objects defined in `__main__` raise `AshlarError` at save time. Weights are Equinox leaves (`core.eqx`, `numeric.eqx`). Nothing in the weight path is pickled. Optimizer moments are the array leaves of an Optax AdamW state in `optimizer.eqx`, restored into a state that was just initialized from the loaded core, the saved learning rate, and the saved weight decay. Learned categorical tables and set-token tables are leaves of `core_`.

`sklearn.base.clone` works through the constructor parameters. A custom class must therefore be the object passed as `head` or `backbone`, not a lambda created inside `fit`.

Load imports the recorded callables. Load only a directory you trust.

**Judgment.** `module:qualname` is less convenient than pickle and is the constraint that keeps a saved model readable. A closure would train and then fail at save; that failure is intentional.

## Fitted schema and target transforms

**Fact.** `metadata.json` stores the encoder schema: column order, role, numeric mean, the scale stored in `std`, median, per-column encoding, and the category vocabulary. For `numeric_scaling="standard"`, `std` is the population standard deviation (`ddof` 0). For `numeric_scaling="robust"`, `std` is the interquartile divisor. Unknown and missing categories are index 0. Known categories occupy indices `1..V` in the order observed on the training partition. The empty string is a real category if it was observed. Integer columns are numeric unless their names are listed in `categorical`, `categorical_embeddings`, or `set_features`. Object, bool, and other non-numeric columns are categorical. A requested categorical name that is absent from the frame is an error. A categorical column with no observed category on the training partition is an error.

Numeric standard deviation below `1e-6` is replaced by 1 on the standard path. A numeric column with no finite training value uses median 0. A robust divisor of 0 uses half the range, then 1.

Built-in regression and normal targets must be finite. Custom objectives may use NaNs as masks. The linked-head example does that for a binary outcome: a missing event does not drop the row or remove its weight from the normal term.

**Fact.** The regression default z-scores `y` with the internal-training mean and population standard deviation (`ddof` 0) and inverts that map in `predict`. A standard deviation below `1e-6` is replaced by 1. `standardize_target=False` skips it. Classification and ordinal do not z-score labels. A saved model that omits the flag loads with it off. Yeo-Johnson and other nonlinear target maps are not implemented. The same-split regression rescore is in `benchmark.md`.

**Judgment.** The linear z-score is the Ailerons case: a target whose standard deviation is about `4e-4` makes squared-error gradients smaller than the AdamW step. The inverse is stored next to the schema, so the prediction stays on the scale of `y`. Nonlinear target maps stay out until they have an explicit inverse of the same kind.

**Fact. Classification TabM.** `tabm(task="classification")` uses the TabM trunk, `prediction_policy="mean_probabilities"`, one shared minibatch, and no label z-score. `numeric_scaling` stays `robust` unless you pass another map. `share_training_batches=False` still raises, because unshared member permutations are regression-only. Passing `standardize_target=True` still raises. `classifier()` is a separate recipe and is not replaced by this call. The same-split scores are in `benchmark.md`.

**Fact. Expected-label penalty.** `expectation_weight` is off by default. On an ordinal task, a positive value adds that coefficient times the squared error of `probabilities @ class_values` to the class negative log-likelihood. The probabilities are the mean of the member probabilities, which is the `predict_score` readout. `histogram_variance` and a custom objective do not combine with it. The residual regression default does not set it.

## Validation and leakage

**Fact.** `split` is `iid`, `temporal`, or `group`.

- `iid` classification and ordinal classification use `sklearn.model_selection.train_test_split` with `stratify=y`. A class with fewer than two rows stays in the training partition, and the same fraction is applied to the remaining rows. A remainder that sklearn cannot stratify raises `AshlarError`. Regression does not stratify.
- `iid` regression draws a NumPy permutation from `random_state`.
- `temporal` holds out the last `round(validation_size * n_rows)` rows. At least one row stays in training and at least one in validation when `validation_size > 0`.
- `group` holds out whole groups. `groups` must align with rows, and at least two groups are required. The fraction is applied to the number of groups.

`validation_size=0` trains on every row and disables early stopping. An explicit `validation=(X_val, y_val)` or `(X_val, y_val, sample_weight)` uses all of `X` for training. Passing `groups` together with an explicit validation tuple is an error. `validation_sample_weight` is accepted only with an explicit validation tuple; internal splits slice the training `sample_weight`.

The encoder and the numeric moments are fit on the training partition only. Class-weight multipliers are estimated on the training partition and reused for validation labels. A validation label absent from the training classes is an error.

`patience=0` disables early stopping. Otherwise the loop restores the core and the matching optimizer state with the lowest full-partition validation loss. That validation loss uses `batch_weight_sum` on the whole validation partition in one pass, not a renormalized minibatch.

## PyTrees, metadata, and the host

**Fact.**

| Pure JAX PyTree | Static metadata | Host |
|---|---|---|
| `PredictiveCore` weights, numeric embeddings, categorical and set embedding tables, `NumericTransform` median/mean/std, Optax AdamW moments, `NormalMixture` and `MeanParameterNormal` location and scale | `k`, widths, dropout rate, `n_events`, `n_classes`, column roles, category vocabulary, task, policy, reduction name | frame dispatch (NumPy, pandas, Polars, dict), category lookup, one-hot, set padding, split indices, class encoding, JSON metadata, import of custom callables |

`forward_transformed(raw_numeric, categorical_onehot)` differentiates through imputation and the numeric map, including the robust center, scale, and smooth clip. The one-hot argument is a constant with respect to that gradient. Category indices are not a continuous input. Their embedding tables are.

## PRNG control

**Fact.** `random_state` is a non-negative integer. It is split into two streams that share the integer and do not share state:

- JAX: `PRNGKey(random_state)` splits into an init key and a train key. Epoch keys are `fold_in` of the train key. Dropout receives a key only when `dropout > 0`. `key=None` runs dropout in inference mode. The head is called with `key=None` because dropout lives in the backbone.
- NumPy: `np.random.default_rng(random_state)` permutes iid regression rows and group ids. Stratified classification uses sklearn's `random_state` argument.

No code path calls a global JAX or NumPy RNG. After `fit`, the stored train key is folded by the number of completed epochs and written into `metadata.json` as a `uint32` pair. `fit(..., resume=True)` continues from `core_`, `opt_state_`, and that key, so a resume does not replay the original minibatches. Resume still requires the same schema, `n_events`, and `n_classes`.

## Dependencies

**Fact.** Required: `jax`, `equinox`, `optax`, `numpy`, `scikit-learn`. Optional extra `dataframes`: `pandas`, `polars`. Optional extra `dev`: those plus `pytest`. The package does not depend on PyTorch, Lightning, Klax, Solstice, or FlowJAX.

**Judgment.** Dataframe adapters are optional because the core path is arrays and column dicts. Tracking is out of scope; history is a dict of train and validation losses on the estimator. A second distribution family should call FlowJAX later rather than grow a distribution library here. Klax and Solstice already wrap generic Optax loops and do not know this schema, this weight reduction, or member-mean losses, so the training loop stays local and short.

## Continuing in a custom JAX loop

**Fact.** After `fit`, the user holds `core_`, `numeric_transform_`, and `opt_state_`. `forward_transformed` is the differentiable numeric path used by `examples/jax_composition.py`. `save(..., save_optimizer=True)` writes `optimizer.eqx`. `load` restores those moments. `fit(..., resume=True)` keeps using them inside the package loop.

The composition example does not need the optimizer moments: it builds a new Optax SGD state on a raw numeric feature, clips that feature to `[-2, 2]`, and leaves the fitted core fixed. A user who wants to keep training the core with their own loop uses `core_` and `opt_state_` directly. The package does not wrap that second loop.

**Judgment.** Preserving both states on request is the contract. Always threading a user's optimizer through `fit` would turn the shell into a trainer framework, which this release is not.
