# Probe results

Recorded 26 October 2026. Hardware: Linux x86_64, 4 CPUs, no GPU. JAX saw `CpuDevice(id=0)` with `JAX_PLATFORMS=cpu`. `torch.cuda` was false. DeepTab and this package used separate virtualenvs so their dependency pins did not fight. The working name was `gnet-tabular` (import `gnet`). The tree is now `/cursor/stores/self/ashlar`, distribution `ashlar-tabular`, import `ashlar`. The DeepTab numbers below are from the earlier probe. The package numbers are the Ashlar re-run.

Evidence grades match `competitive-matrix.md`: confirmed means this environment ran it, source-observed means a file or doc was read and not executed, hypothesis means it was not checked.

## Environments

DeepTab venv `/tmp/deeptab-venv`, Python 3.12.3:

| Package | Version |
|---|---|
| deeptab | 2.1.0 |
| torch | 2.14.1+cu130 (CUDA not used) |
| lightning | 2.6.6 |
| numpy | 2.5.4 |
| scikit-learn | 1.9.1 |
| pandas | 2.3.3 |
| polars | 1.34.0 |
| pretab | 1.0.0 |
| tabm | 0.0.3 |

The wheel tag is `v2.1.0`, commit `11d350fa2519c193a317abc3799a7a3929165436`. The brief's older snapshot `ca18a4e` was not the code under test.

JAX venv `/tmp/gnet-venv`, Python 3.12.3, editable install of `/cursor/stores/self/ashlar`:

| Package | Version |
|---|---|
| ashlar-tabular (import `ashlar`) | 0.1.0 |
| jax / jaxlib | 0.12.0.1 |
| equinox | 0.13.8 |
| optax | 0.2.8 |
| numpy | 2.5.4 |
| scikit-learn | 1.9.1 |
| pandas | 3.1.0 |
| polars | 2.1.0 |
| pytest | 9.1.1 |

Install:

```bash
python3 -m venv /tmp/gnet-venv
/tmp/gnet-venv/bin/pip install -e "/cursor/stores/self/ashlar[dataframes,dev]"
```

`python3 -m venv` needed the `python3.12-venv` package because `ensurepip` was missing on the image. That is an environment fact, not a package defect.

## DeepTab 2.1.0

Command:

```bash
/tmp/deeptab-venv/bin/python -u /tmp/deeptab_probes.py
```

The script lives in `/tmp` and is not part of the package. It fits tiny CPU models for one epoch. Times below are that overhead, not a benchmark. The default MLP reported 38.1K parameters on two columns because the default width is wide.

### Sample weight is not one mechanism

| Call | Result |
|---|---|
| `MLPRegressor.fit(..., sample_weight=None)` | `UnsupportedParamError`: `sample_weight is not supported for weighted-loss fitting.` Also a `ValueError`. `vars(estimator)` unchanged. `is_fitted_` stayed as it was set before the call |
| `MLPRegressor.fit(..., sample_weight=ones)` | Same error, state unchanged |
| `MLPLSS.fit(..., sample_weight=None)` and an array | Same error, state unchanged |
| `MLPClassifier.fit(..., sample_weight=weights)` with values `{1, 50}` | Fit about `0.078` s. Loss `BCEWithLogitsLoss`, `loss.weight is None`. Sampler `WeightedRandomSampler`, unique weights `[1.0, 50.0]`, max `50`, sum `865` |
| `MLPClassifier.fit(..., sample_weight=None)` | Accepted. Sampler `RandomSampler` |
| `MLPClassifier.fit(..., class_weight="balanced")` | Loss `WeightedBCEWithLogitsLoss`. Inner `loss.weight` is `None`. `pos_weight` is `[1.1052632]`. Sampler `RandomSampler` |

The `pos_weight` line is from a follow-up that printed `loss.pos_weight` on the same seed and the same `class_weight="balanced"` call. The original script printed `loss.weight`, which is the wrong attribute for this loss, so it showed `None` even while `pos_weight` was set.

Classifier sampling, class weights, and direct loss weights are three different things. This probe confirms the first two on DeepTab and confirms that the third is rejected for regression and LSS.

### Frames, save, custom model, gradient

| Check | Result |
|---|---|
| Pandas regressor, one epoch, missing number and unseen category | Fit about `0.048` s. `predict` returned an ndarray |
| Polars predict on a frame whose columns are `x1` and `city` | `ColumnNameError`: `Feature column names do not match what was seen during fit. Missing : ['x1', 'city']` |
| Save from the probe script | `/tmp/deeptab-probe-out/mlp.deeptab`, 172885 bytes |
| Save/reload on a second one-epoch fit with `output_dim=7` set explicitly | Artifact 173431 bytes. Reloaded `predict` max abs diff `0.0` |
| `TabMLSS`, `ensemble_size=2`, `average_ensembles=False`, family normal | Fit about `0.037` s. Public raw shape `(2, 2)`. Member tensor from `task_model.estimator` shape `(2, 2, 2)`. `returns_ensemble` true |
| Mean of per-member transformed scales versus transform of the mean raw scale | `[0.65148604, 0.6095646]` versus `[0.64666396, 0.60703015]`. Public `predict` matched mean-then-transform |
| `InferenceModel.predict_params` | Shape `(2, 2)`. The method takes a `raw` flag and has no member or mixture policy |
| Custom regressor: `TinyConfig`, `TinyNet(BaseModel)`, `TinyRegressor(SklearnBaseRegressor)` | Fit about `0.034` s and predicted. A bare `nn.Module` without those types was not tried |
| Public `predict` on a one-row frame | `ndarray`, `requires_grad` false |
| Gradient of the module output w.r.t. the already preprocessed numeric tensor | Nonzero, about `[-0.00082923, -0.00154978]` on the two-row batch |
| Preprocessor type | `Preprocessor` |

`average_ensembles=True` contains a `print(x.shape)` in `deeptab/architectures/tabm.py` on tag `v2.1.0`. The probe left the flag at its default `False`, so that print did not run. Source-observed only. Not recorded as a bug.

## Official TabM layer

`tabm` 0.0.3 `LinearBatchEnsemble(5, 7, k=3, scaling_init="random-signs")`, input `(4, 3, 5)`, float32, CPU. Weights copied out of the torch module.

| Comparison | Max abs |
|---|---|
| NumPy evaluation of `((x * r) @ W.T) * s + bias` versus torch `forward` | `0.0` |
| `ashlar.layers.LinearBatchEnsemble` versus torch `forward` | `1.1920928955078125e-07` |

The JAX gap is one float32 ulp of 1 and comes from the matmul, not from a different equation. Bias rows of the official module were equal, which matches its shared initialization draw. This is not a full `TabM.make` match and not an embedding match. `LinearReLUEmbeddings` was read from `rtdl_num_embeddings.py` and not executed.

Command shape, using the JAX interpreter so JAX and Equinox import, with `tabm` and torch on `PYTHONPATH`. This measurement was taken on the layer before the rename; the equation in `LinearBatchEnsemble` was not rewritten:

```bash
PYTHONPATH=/tmp/deeptab-venv/lib/python3.12/site-packages \
  JAX_PLATFORMS=cpu /tmp/gnet-venv/bin/python
```

The script constructed the torch layer, copied `weight`, `r`, `s`, and `bias` into the JAX module, and took `max(abs(jax - torch))`.

## This package

From `/cursor/stores/self/ashlar`, after `pip install -e ".[dataframes,dev]"` into `/tmp/gnet-venv`:

```bash
JAX_PLATFORMS=cpu /tmp/gnet-venv/bin/pytest -q -p no:cacheprovider
JAX_PLATFORMS=cpu /tmp/gnet-venv/bin/python examples/ordinary.py
JAX_PLATFORMS=cpu /tmp/gnet-venv/bin/python examples/embeddings.py
JAX_PLATFORMS=cpu /tmp/gnet-venv/bin/python examples/search.py
JAX_PLATFORMS=cpu /tmp/gnet-venv/bin/python examples/transformer.py
JAX_PLATFORMS=cpu /tmp/gnet-venv/bin/python examples/advanced.py
JAX_PLATFORMS=cpu /tmp/gnet-venv/bin/python examples/jax_composition.py
```

`-p no:cacheprovider` is required in this store because pytest cannot write `.pytest_cache` here. The tests do not need it.

### Tests

`27 passed`. Coverage includes the earlier semantics (weight reduction, mixture variance `26` versus `1`, TabM equation, splits, save/reload, gradient through standardization) plus the default recipe matching `TabularEstimator().get_params()`, `search=True` picking a listed learning rate while a no-search fit leaves `search_` as `None`, a learned city embedding with a set-valued `tags` column on `backbone="set"`, and a one-epoch `backbone="transformer"` fit.

### Ordinary example

`examples/ordinary.py`. Default residual MLP (no backbone override), 80 rows, pandas frame with missing age and missing city, 8 epochs, `search` omitted.

```text
ashlar 0.1.0 ordinary
backbone residual_mlp
search None
fit_seconds 1.8655
first_predict_seconds 0.2704
val_loss_tail 0.070068
pandas [1.9122052192687988, 1.4870924949645996]
polars_matches True
numpy_matches True
reload_matches True
```

Polars was built with `pl.DataFrame({...})`. `pl.from_pandas` on the object column asked for pyarrow, which is not a dependency. Predictions agree across pandas, Polars, a column dict, and a reload.

### Classifier example

`examples/classifier.py`. `ashlar.classifier()` on a three-class frame with a missing number and an unseen category, then `search=True` on a binary problem, then a transformer with `categorical_embeddings=["city"]`.

```text
ashlar 0.1.0 classifier
fit_seconds 2.3409
classes ['high', 'low', 'mid']
labels ['high', 'mid']
probabilities [[0.5847724080085754, 0.011782865971326828, 0.40344470739364624], [0.1305408775806427, 0.014198479242622852, 0.8552606105804443]]
search_seconds 2.6292
search_metric weighted_accuracy
search_scores [0.8507195115089417, 1.0]
chosen_learning_rate 0.003
constructor_learning_rate 0.001
embedding_onehot_widths []
embedding_probabilities [[0.5707340240478516, 0.07814499735832214, 0.3511210083961487], [0.15195100009441376, 0.42151200771331787, 0.4265359938144684]]
```

Search scores are in `[0, 1]`. The constructor learning rate stayed `0.001`. The chosen rate was `0.003`, the trial with accuracy `1.0`.

Also:

```bash
JAX_PLATFORMS=cpu /tmp/gnet-venv/bin/python examples/classifier.py
```

### Embeddings, search, and transformer

`examples/embeddings.py` names `categorical_embeddings=["city"]` and leaves search off.

```text
ashlar 0.1.0 embeddings
fit_seconds 2.0966
onehot_widths []
embedding_table (4, 4)
prediction [0.2940068542957306, 0.1338597536087036]
search None
```

`examples/search.py` uses `fit(..., search=True)`. The constructor learning rate stays `0.001`. The chosen rate is `0.003`, the second default trial.

```text
ashlar 0.1.0 search
fit_seconds 2.2413
switch search=True
default_space {'learning_rate': [0.001, 0.003]}
chosen_learning_rate 0.003
trial_val_losses [0.8183388710021973, 0.22924621403217316]
constructor_learning_rate 0.001
prediction [-0.5013490915298462, 2.1102230548858643]
```

`examples/transformer.py` uses `backbone="transformer"` with a learned `city` column and a variable-length `tags` set. Member count stays 1.

```text
ashlar 0.1.0 transformer
fit_seconds 3.2793
member_count 1
set_length 3
prediction [0.47314101457595825, 0.584101676940918]
```

### Advanced example

`examples/advanced.py`. `TABM_PROBE` widths (`k=4`, `d_block=16`, dropout `0`), custom `LinkedHead` from `ashlar.demos.linked`, direct weights with three exact zeros, 10 missing binary labels kept in the frame.

```text
ashlar 0.1.0 advanced
fit_seconds 2.3976
member_normal_shape (1, 4, 1, 2)
train_loss_tail 1.971063
point [0.10997553914785385, 0.19012711942195892]
fitted_mixture_variance 0.563342
fitted_mean_parameter_variance 0.563342
reference_mixture_variance 26.0
reference_mean_parameter_variance 1.0
reload_matches True
```

On this one-row prediction the four members had not separated, so the fitted mixture variance equals the parameter-average variance. That is an observation about an 8-epoch toy, not a counterexample. The semantic claim is the reference pair `26` versus `1`, which the example asserts, and the unit test on a constructed tensor.

### Composition example

`examples/jax_composition.py`. TabM probe widths, then `jax.grad` of `forward_transformed` with respect to one raw numeric value. The city one-hot stays constant. Fifteen SGD steps with the value clipped to `[-2, 2]`.

```text
ashlar 0.1.0 composition
fit_seconds 2.2399
grad_seconds 0.4162
jax_grad 0.155923
finite_difference 0.155926
prediction_before 0.106476
prediction_after 0.054121
optimized_rooms -0.195571
category_width 3
```

The gradient and the central difference at step `1e-3` differ by about `3e-6`. The example fails if that gap exceeds `1e-2`.

Pytest wall clock on this re-run was 18.0 seconds. The six examples together were 29.3 seconds. Those clocks include compilation.

## Failures that were fixed before this record

These are package defects found while bringing the tests up, not DeepTab findings.

| Failure | Fix |
|---|---|
| `from ashlar import __version__` inside the estimator cycled with package import | Version lives in `ashlar._version` |
| `float(array)` on a length-1 NumPy 2 result | Tests read a Python float from a reshaped value. Univariate `predict` now returns `(batch,)` so a one-row point is a length-1 vector, and a scalar loss indexes `[0]` |
| Pickle of an Optax state that contained Equinox treedefs (`'_Missing' object is not subscriptable` on load) | `optimizer.eqx` stores array leaves and is restored into a freshly initialized AdamW state |
| `jax.grad` of `forward_transformed(...)[0]` when the point was `(1, 1)` produced a non-scalar | Univariate squeeze makes `[0]` a scalar. A gradient with respect to a length-1 feature vector is still shape `(1,)` and is checked that way |
| `pl.from_pandas` on a column of objects | The example builds the Polars frame directly |

No probe was left failing. The pytest run and the three examples exited 0.
