# Competitive evidence

Pinned on 26 October 2026. Evidence grades:

- **confirmed**: behavior executed in this environment
- **source-observed**: current docs, tag, or wheel inspected, not executed
- **hypothesis**: not checked far enough to treat as a finding

No GitHub issues were filed. A source comment or branch tip is not recorded here as a runtime bug unless a probe reproduced it.

The earlier DeepTab snapshot in the brief was `ca18a4e2c9fe208dbb73b6fd2fe139deae29ec7c`. This pass used the released wheel **deeptab 2.1.0**, tag `v2.1.0`, commit `11d350fa2519c193a317abc3799a7a3929165436`.

## Summary

| Capability | DeepTab 2.1.0 | PyTabKit 1.7.3 | PyTorch Tabular 1.2.0 | This package |
|---|---|---|---|---|
| Ordinary mixed-frame fit | Confirmed | Source-observed sklearn `fit` | Source-observed dataframe `fit` | Confirmed, including Polars |
| Classifier row sampling | Confirmed `sample_weight` | Not a parameter on `fit` (source-observed) | `train_sampler` argument (source-observed) | Not implemented. Weights are not sampling. |
| Direct loss weights on regression and normal/LSS | Confirmed rejection | No `sample_weight` parameter (source-observed) | Custom `loss` module can do it; not a built-in weight vector (source-observed) | Confirmed, one reduction |
| Class weights separate from row weights | Confirmed | Not inspected beyond the missing `sample_weight` argument | Not executed | Confirmed multiplier, applied once |
| Member distributions vs averaged parameters | Confirmed average-then-transform; members only via `task_model` | Source-observed CV average of probabilities or outputs | MDN head exists; ensemble policy not executed | Confirmed `mixture` and `mean_parameters` |
| Custom head without a new estimator class | Documented three-class path; that path was executed | Not the extension model | Heads and `loss=` exist (source-observed) | Confirmed Equinox class |
| Fitted core inside a JAX grad through the numeric transform | Confirmed the public path returns NumPy after external preprocessing | PyTorch | PyTorch | Confirmed |
| Faithful TabM equation | DeepTab's TabM config is a different recipe (source-observed) | Defaults track the paper more closely (source-observed) | Not a TabM package | Confirmed equation match of one official `LinearBatchEnsemble`; JAX CPU matmul within one float32 ulp |

## DeepTab 2.1.0

Primary links: [repository](https://github.com/OpenTabular/DeepTab), [docs](https://deeptab.readthedocs.io/en/stable/), PyPI `deeptab==2.1.0`.

Runtime: Python 3.12.3, torch 2.14.1, lightning 2.6.6, numpy 2.5.4, scikit-learn 1.9.1, pandas 2.3.3, polars 1.34.0, pretab 1.0.0. `torch.cuda` was false. Commands and numbers are in `probe-results.md`.

| Claim | Where | Grade |
|---|---|---|
| Classifier `sample_weight` is per-row sampling, not a loss multiplier. `class_weight` is separate. | `deeptab/models/classifier_base.py` `_resolve_loss_and_sampler`; docs page "Sampling Is Not Weighted Loss"; probe | **Confirmed.** Sampler type `WeightedRandomSampler` with weights `{1, 50}`. Loss stayed `BCEWithLogitsLoss` and its `weight` was `None`. `class_weight="balanced"` produced `WeightedBCEWithLogitsLoss` with `pos_weight` about `[1.1053]` and a `RandomSampler`. Explicit `sample_weight=None` is accepted and uses `RandomSampler`. |
| Regressor and LSS reject `sample_weight`, including `None`, before mutation. | `regressor_base.py`, `lss_base.py`, `tests/contracts/test_sample_weight_contract.py`; probe | **Confirmed.** `UnsupportedParamError`, also a `ValueError`. `vars(estimator)` unchanged. |
| LSS prediction averages raw member outputs, then applies the family transform. | `SklearnBaseLSS.predict` in `lss_base.py` (mean on dim 1, then `family(...)`) | **Confirmed** on `TabMLSS`, `ensemble_size=2`. Public raw shape `(2, 2)`. Member tensor from `task_model.estimator` had shape `(2, 2, 2)`. Mean of per-member transformed scales was about `[0.6515, 0.6096]`; transform of the averaged raw scale was about `[0.6467, 0.6070]`. |
| Deployment API returns distribution parameters as arrays. | `deeptab/core/inference.py` `predict_params` | **Confirmed.** Shape `(2, 2)` for the normal probe. The method has `raw` and no member or mixture policy. |
| Custom architecture is config + `BaseModel` + task estimator. | [Custom Models](https://deeptab.readthedocs.io/en/stable/core_concepts/custom_models.html) | **Confirmed** for the supported path: a tiny regressor with those three types fit for one epoch. A negative test of a bare `nn.Module` was not run. |
| Fitted module can be called, but preprocessing is outside it. | `task_model` on `deeptab/core/inspection.py`; `MambularDataModule.preprocess_new_data` | **Confirmed.** `task_model.estimator` was an `MLP`. Gradient of the summed output w.r.t. the preprocessed numeric tensor was nonzero. `predict` on a frame returned `ndarray` with `requires_grad` false. Preprocessor type was `Preprocessor`. |
| Save/reload. | `MLPRegressor.save` / `load` | **Confirmed.** Artifact 172,885 bytes. Reloaded predictions differed by 0. |
| Polars frame. | Probe `reg.predict(polars_frame)` | **Confirmed failure of a clean path.** `ColumnNameError`: missing columns `['x1', 'city']` even though those names were present. Pandas with an unseen category and a missing number predicted. |
| DeepTab TabM is not the official TabM recipe. | `deeptab/configs/models/tabm_config.py` versus `tabm` 0.0.3 `TabM.make` | **Source-observed.** DeepTab defaults include `model_type="mini"`, `scaling_init="ones"`, `dropout=0.5`, `layer_sizes=[256, 256, 128]`, `ensemble_size=32`. Official `TabM.make` defaults used below are different. |
| `average_ensembles=True` prints a shape in `TabM.forward`. | `deeptab/architectures/tabm.py` around the `print(x.shape)` call on tag `v2.1.0` | **Source-observed only.** The probe left `average_ensembles` at its default `False`, so the print was not executed. Not treated as a reproduced bug. |

Ordinary adoption on DeepTab is real: mixed columns, missing values, unseen categories, validation, save/reload, and predict worked for pandas. The gaps that matter for this project are direct weights on non-classification losses, an explicit predictive mixture, and a JAX core.

## PyTabKit 1.7.3

Primary: [dholzmueller/pytabkit](https://github.com/dholzmueller/pytabkit), PyPI `pytabkit==1.7.3` (uploaded 2026-01-06). Wheel inspected locally. Torch was not used to train a PyTabKit model in this pass.

| Claim | Where | Grade |
|---|---|---|
| RealMLP and TabM ship with tuned defaults and HPO. | Project readme | **Source-observed.** |
| `fit` has no `sample_weight`. | `pytabkit/models/sklearn/sklearn_base.py` `fit(X, y, X_val, y_val, val_idxs, cat_indicator, cat_col_names, time_to_fit_in_seconds)` | **Source-observed.** A repo-wide search of the wheel found no `sample_weight` string. |
| TabM defaults sit close to the paper stack. | `TabMConstructorMixin` docstring | **Source-observed.** `tabm_k` default 32, `dropout` 0.1, `lr` 2e-3, `weight_decay` 0, `patience` 16, `n_blocks` auto (3 without numeric embeddings, 2 with them), `num_emb_type` default `none` with optional `pwl`, `batch_size` 256. CV ensembling averages probabilities for classification and outputs for regression. |
| Not a JAX composition layer. | Package imports and model wrappers | **Source-observed.** The neural models are PyTorch. |

Judgment: PyTabKit is the stronger recipe reference for "what TabM training defaults look like." It is not the place to get an Equinox module or a weighted distributional objective.

## PyTorch Tabular 1.2.0

Primary: [pytorch-tabular docs](https://pytorch-tabular.readthedocs.io/en/latest/tabular_model/), PyPI `pytorch-tabular==1.2.0` (2026-01-26). Not installed here.

| Claim | Where | Grade |
|---|---|---|
| Embeddings, backbones, and heads are separated. `MixtureDensityHead` exists. | Docs and project model list | **Source-observed.** |
| `fit(..., loss=nn.Module, train_sampler=...)` accepts a custom loss and a sampler as different arguments. | Training API | **Source-observed.** |
| A distribution object with `log_prob` / `sample` and an explicit ensemble policy is the default prediction API. | Not found in the fetched training API | **Hypothesis.** Do not treat absence from one page as absence from the library. |

Judgment: custom losses and mixture-density heads are already routine in this library. The remaining gap is not "mixture heads exist." It is JAX composition plus a visible ensemble reduction.

## Equimo 2.7.0

Primary: [RhizomeResearch/Equimo](https://github.com/RhizomeResearch/Equimo), PyPI `equimo==2.7.0`. Not installed.

| Claim | Grade |
|---|---|
| Equinox implementations exist, including a TabPFN-3 classifier and regressor core. Inputs are unbatched `x`, `y`, and `n_train`. It is not the upstream sklearn preprocessing and ensembling API. | **Source-observed** from the project readme. |
| It is a suitable home for a weighted TabM training recipe. | **Judgment: no.** Tabular coverage there is an in-context foundation model inside a multimodal research library. The readme warns that implementations may not be exact replicas. |

## TabFM 1.0.1

Primary: [google-research/tabfm](https://github.com/google-research/tabfm), PyPI `tabfm==1.0.1`. Not installed.

| Claim | Grade |
|---|---|
| scikit-learn `TabFMClassifier` / `TabFMRegressor` with a JAX backend and zero-shot in-context prediction. | **Source-observed** from the repository readme and the Hugging Face model card for `google/tabfm-1.0.0-jax`. |
| Convenient JAX tabular prediction does not exist. | **False.** TabFM is a different learning regime (no per-dataset gradient training), but it removes that broad claim. |

## Klax 0.2.0, Solstice 0.0.1, FlowJAX 19.1.2

| Project | What was checked | Grade |
|---|---|---|
| [Klax](https://drenderer.github.io/klax/) 0.2.0 | Wheel `klax/_training.py`. `fit(model, data, ..., loss, optimizer, key)` trains an Equinox module. Docs say the API is early and aimed at computational mechanics. Validation logging uses a larger batch, not a full-partition weighted metric. | **Source-observed.** Not imported at runtime. |
| [Solstice](https://charl-ai.github.io/Solstice/) `solstice` 0.0.1 | Docs: alpha experiment container around `solstice.Experiment`. | **Source-observed** from the docs page. Not installed. |
| [FlowJAX](https://github.com/danielward27/flowjax) 19.1.2 | PyPI metadata: Equinox distributions, bijections, and flows. Requires Python `>=3.11`, `equinox>=0.11`, `jax>=0.7`. | **Source-observed.** Not imported. |

Judgment: reuse Klax only if the training problem is "optimize this pytree." The tabular contracts (schema, direct weights, member-mean loss, full validation partition, ensemble policy) are the part Klax does not provide. A short Optax loop is smaller than depending on an early mechanics trainer and re-teaching it those contracts. FlowJAX should be the extension path for a second distribution family. One diagonal normal does not justify the dependency.

## Official TabM

Primary: [yandex-research/tabm](https://github.com/yandex-research/tabm), PyPI `tabm==0.0.3`, git commit `28e47ae301c92ec37787dde1ce923a0793f405b4` (2025-11-10). Runtime import of the layer used torch 2.14.1 from the DeepTab environment and JAX from the Ashlar environment.

| Claim | Grade |
|---|---|
| `LinearBatchEnsemble` is `((x * r) @ W.T) * s + bias` on `(batch, member, in)`. | **Source-observed** in `tabm.py`, and **confirmed** by a weight-copy forward. NumPy float32 and torch agreed at max abs `0`. The JAX layer differed from torch by max abs `1.192e-7`. |
| TabM init: first `r` random, later scalings ones, shared bias draw across members, optional chunks. | **Source-observed**, and the chunk and shared-bias consequences were **confirmed** on the official module (`r` equal inside the chunk; the two bias rows matched). |
| Train the mean of member losses. At inference, average predictions; for classification, usually average probabilities. | **Source-observed** in the package README. |
| `TabM.make` defaults without embeddings: `n_blocks=3`, `d_block=512`, `dropout=0.1`, `k=32`, `arch_type='tabm'`, `start_scaling_init='random-signs'`. With embeddings: `n_blocks=2` and `start_scaling_init='normal'`. | **Source-observed** in `TabM.make`. |
| `LinearReLUEmbeddings` is `relu(bias + weight * x[..., None])`, uniform init on `± d_embedding**-0.5`. | **Source-observed** in `rtdl_num_embeddings.py` on `main`. Not bit-matched at runtime. |

## What is already routine, so it is not the pitch

Automatic preprocessing, custom losses somewhere in the ecosystem, numeric embeddings, distributional regression, mixture heads, dataclasses, sklearn `fit`, serialization, and parallel ensembles all exist in at least one of the packages above. The package pitch is the combination that was missing for a JAX user: direct weights on every task, a named ensemble policy with member distributions, and an Equinox core that still knows the fitted numeric transform.
