# Recommendation

**Build a standalone package, and keep the first release narrow.** The prototype is the distribution `ashlar-tabular` with import name `ashlar`. `gnet-tabular` was the working name. It is not a backend port of DeepTab and it is not a model catalogue.

## Decision

| Option | Verdict |
|---|---|
| Standalone package | **Yes, for the scoped surface below.** |
| Upstream contribution or thin companion | Not sufficient for the JAX workflow the brief asks for. |
| Only loose Equinox modules | Too small. The workflow gain is the contracts around those modules. |

This is a judgment. The facts it rests on are in `competitive-matrix.md` and `probe-results.md`.

## Who it is for

A quantitative user who already fits models in JAX and Equinox, wants an ordinary tabular `fit` / `predict` path, and also needs to do at least one of these without leaving that stack:

- put a direct per-row weight on a regression or distributional loss
- keep ensemble members available and choose mixture versus parameter averaging explicitly
- drop the fitted numerical core into a larger Equinox computation, with gradients through the continuous part of the transform

Users who want the widest PyTorch model zoo, Lightning training, or a zero-shot foundation model are already served by DeepTab, PyTorch Tabular, PyTabKit, Equimo's TabPFN core, or TabFM. This package does not try to replace those.

## What was demonstrated

On CPU, with pinned DeepTab 2.1.0 and a clean install of this package:

- DeepTab classifier `sample_weight` changes the `WeightedRandomSampler`. It does not multiply the loss. `class_weight="balanced"` changes the loss (`pos_weight`) and leaves a `RandomSampler`. Regression and LSS reject `sample_weight`, including an explicit `None`, before fitted state changes.
- DeepTab LSS `predict` averages raw member outputs and then applies the normal transform. Averaging transformed scales differed from transforming the averaged raw scale on the probe row. `InferenceModel.predict_params` returns those averaged arrays. Member tensors are reachable through the public `task_model.estimator`, not through a predictive object.
- A DeepTab custom regressor required a config dataclass, a `BaseModel` subclass, and an estimator subclass. The same three-piece path did fit. A plain module is not the documented contract.
- DeepTab's public `predict` returns NumPy. A nonzero gradient was obtained only for the tensor that enters the module, after `pretab` preprocessing.
- The package trains classification, regression, and a diagonal normal with the same estimator class. A custom linked head is an ordinary Equinox module. Sample weights use `sum(w * loss) / sum(w)` on every task. `predict_dist(..., policy="mixture")` and `policy="mean_parameters"` are different objects. A constructed mixture of N(-5, 1) and N(5, 1) has variance 26; the parameter average has variance 1.
- `TabularEstimator()` is the default recipe: residual MLP, inferred one-hot categories, no search. `fit(..., search=True)` is the only switch that runs the two-rate default grid. `categorical_embeddings=["city"]` selects learned tables; other categoricals stay one-hot. `backbone="transformer"` and `backbone="set"` share the field-token contract with the residual MLP and TabM.
- Copying one official `tabm` 0.0.3 `LinearBatchEnsemble` into the JAX layer matched the same equation. NumPy and PyTorch float32 forwards agreed with max absolute error 0. The JAX CPU matmul differed from that result by at most `1.192e-7` (one float32 unit in the last place of 1).
- After `fit`, `forward_transformed` is differentiable in the raw numeric value. On the composition example the JAX gradient and a central finite difference agreed to about `3e-6` (0.129365 versus 0.129368).

## What was not built, and why

A large abstraction is not justified where an existing tool already does the job. These were left out on purpose:

- a model zoo, RealMLP, TabM-mini, TabM-packed, retrieval models, and foundation-model checkpoints
- piecewise-linear and periodic embeddings
- a general distribution or flow library (FlowJAX already exists)
- another general trainer (Klax and Solstice already wrap Optax loops, and they do not know tabular schemas or member-wise losses)
- a claim that neural nets beat trees, or that this TabM recipe matches a published dataset score

Contributing the missing weight and mixture behavior to DeepTab would help PyTorch users and would not produce an Equinox core. Contributing a TabM module to Equimo would land it inside a vision and TabPFN research library that does not own this preprocessing, weighting, or prediction policy. A thin adapter over DeepTab would not be differentiable through the fitted numeric transform inside JAX.

## Name

`gnet-tabular` was the working name. The tree is now Ashlar. A public lookup on 26 Oct 2026 got HTTP 404 for `https://pypi.org/pypi/ashlar-tabular/json`, so the distribution string `ashlar-tabular` is unused in that check. This does not reserve or claim it. Nothing was registered or uploaded.

- distribution name: `ashlar-tabular`
- import package: `ashlar`
- directory: `/cursor/stores/self/ashlar`

PyPI project `ashlar` 1.20.0 is an unrelated microscopy stitcher (Sorger lab / labsyspharm). The import `ashlar` collides with that package if both are installed. The earlier PyPI project `gnet` / `gNet` 0.2.1.1 is a different collision and is no longer the import used here.

## Package

Source, tests, and examples: `/cursor/stores/self/ashlar`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dataframes,dev]"
export JAX_PLATFORMS=cpu
pytest
python examples/ordinary.py
python examples/classifier.py
python examples/embeddings.py
python examples/search.py
python examples/transformer.py
python examples/advanced.py
python examples/jax_composition.py
```
