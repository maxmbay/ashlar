# ashlar

A narrow tabular modeling package for people already working in JAX. The
numerical core is Equinox. Optimization is Optax. The mutable shell follows
the usual `fit` / `predict` shape.

The import name is `ashlar`. The distribution name is `ashlar-tabular`.
Nothing here is registered or uploaded. On 26 Oct 2026, PyPI returned 404 for
`ashlar-tabular`. PyPI project `ashlar` 1.20.0 is an unrelated microscopy
stitcher, so installing both into one environment would collide on the import
name. The working name before this rename was `gnet-tabular`.

This is not a model zoo and not a general training framework. The default
backbone is a residual MLP. TabM, a feature-set transformer, and a set encoder
are the other backbones. Tasks are regression, classification, ordinal
classification, and a diagonal normal, plus a custom head.

## Install

From this directory, in a clean environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dataframes,dev]"
export JAX_PLATFORMS=cpu
```

CPU is enough. The probes do not need a GPU.

## Default recipe

`TabularEstimator()` or `ashlar.tabular()` is the path with almost no arguments:

- task `regression`
- backbone `residual_mlp` (`n_blocks=2`, `d_block=64`, SiLU, hidden dropout `0` unless you override `backbone_kwargs`; `activation` may be `relu` or `gelu`)
- numeric columns: training-partition median impute, median center, interquartile scale, then a smooth clip at ±3 (`numeric_scaling="standard"` restores mean and population standard deviation)
- regression labels: training-partition mean and population standard deviation, inverted in `predict` (`standardize_target=False` leaves `y` alone; classification and ordinal stay off)
- other columns inferred as categorical and given a learned embedding, with index 0 for missing and unseen values (`categorical_embeddings=[]` restores one-hot)
- integer columns stay numeric unless named; no set columns unless you name them
- iid validation split, `validation_size=0.2`, batch 64, 50 epochs, patience 10, learning rate `1e-3`
- regularization, on unless you turn a piece off:
  - weight decay `1e-4` (`weight_decay=0` disables it)
  - feature dropout `0` (`feature_dropout=0.1` drops each input field independently on a training step and rescales the rest; `predict` does not drop fields)
  - early stopping on that held-out slice (`patience=0` disables the stop; `validation_size=0` trains on every row)
- `search` left off

The AdamW schedule defaults to `constant`. `linear` decays the constructor rate to zero over the epoch budget. `surface` warms up for at most 20 steps at `1e-4`, measures the training-batch curvature, and then uses `(n_train / 10000) / steepness` clipped to `[1e-5, 1e-1]`. A smaller training partition or a steeper surface lowers that rate. The default recipe does not turn `surface` on.

```bash
python examples/ordinary.py
```

That example only shortens the toy budget and raises the learning rate. It does not turn search on, and it does not name an embedding column.

## Classification

`ashlar.classifier()` is the same preprocessing with `task="classification"` and `prediction_policy="mean_probabilities"`. `TabularEstimator(task="classification")` is the same flag.

`predict` returns class labels. `predict_proba` returns one probability column per class, in `classes_` order. Binary problems are two classes. The columns sum to one.

`tabm(task="classification")` is the same task on the TabM trunk. Members share one minibatch. Labels are not z-scored. The policy is `mean_probabilities` unless you pass `mean_logits`. Quantile scaling stays off unless you pass `numeric_scaling`. `share_training_batches=False` still raises. `classifier()` is unchanged.

`fit(..., search=True)` on a classifier ranks trials by **weighted accuracy**: `sum(w * correct) / sum(w)` on the full validation partition. `w` is the effective row weight already used by the weighted cross-entropy (`sample_weight`, times a class-weight multiplier once). The score does not multiply class weights again, and it is not squared error. The final train still minimizes that weighted cross-entropy. A fit that omits `search` does not compute the accuracy score.

```bash
python examples/classifier.py
```

## Ordinal classification

`ashlar.ordinal()` is the same preprocessing with `task="ordinal"`. The labels must be real numbers. Their sorted numeric order is the class order. The default loss is a cumulative logit: each member has one latent score and strictly increasing cutpoints, and the probability of a class is the logistic mass between its cuts. `head="softmax"` gives each class its own score instead. The default head stays the cumulative logit.

`head_kwargs={"class_prior": "empirical"}` freezes those cuts at a Jeffreys prior from the training counts. The default leaves the cuts learnable. `predict` returns the most likely label. `predict_proba` returns one probability column per class, in `classes_` order. `predict_score` returns the expected label, `probabilities @ classes_`. A fit that omits `search` early-stops on the ordered negative log-likelihood. `search=True` ranks ordinal trials by that validation loss, not by accuracy.

The iid holdout keeps a class with fewer than two rows in the training partition and stratifies the other rows. Regression does not use that rule.

`expectation_weight=1` adds the squared error of `probabilities @ classes_` beside that likelihood. The probabilities are the mean of the member probabilities, the same readout as `predict_score`. The default weight is off.

Label z-scoring and `share_training_batches=False` stay regression-only.

```bash
python examples/ordinal.py
```

## Search

Search runs only when you pass it to `fit`. `search=None` and `search=False` keep the constructor settings.

```python
model.fit(X, y, search=True)
```

`search=True` uses two trials, two epochs each, and patience 0. The two trials compare weight decay `1e-4` and `1e-3`. Learning rate and feature dropout stay at the constructor values unless you put them in the space. Regression and normal trials are ranked by validation loss. Classification trials are ranked by weighted accuracy. The winner is then trained with the estimator's own `max_epochs` and `patience`, still on the task loss. Trial checkpoints are discarded. The constructor's `weight_decay` is not overwritten; the trained value is `search_["chosen"]` and `resolved_`. Omitting `search` does not run this grid.

Widen the grid by replacing the space. The cartesian product is kept in list order and truncated to `n_trials`:

```python
model.fit(X, y, search={
    "n_trials": 4,
    "max_epochs": 3,
    "space": {
        "learning_rate": [1e-3, 3e-3],
        "weight_decay": [0.0, 1e-4],
    },
})
```

Allowed space keys: `learning_rate`, `weight_decay`, `feature_dropout`, `batch_size`, `backbone_kwargs`. Search needs a validation partition (`validation_size > 0` or an explicit `validation` tuple).

```bash
python examples/search.py
```

## Categorical embeddings and set features

Learned embeddings are the default for categorical columns. Pass an empty list to one-hot them, or name a subset:

```python
TabularEstimator(categorical_embeddings=["city"], categorical_embedding_dim=8)
```

Those tables live on `core_.categorical_embeddings`. Missing and unseen categories use row 0.

A variable-length category list is a set feature. Each cell is a list of labels. The core mean-pools a learned token embedding. Lengths above `set_max_length` (default 8) are truncated to the longest list seen in the training partition, capped by that maximum. An empty list becomes a zero vector.

```python
TabularEstimator(set_features=["tags"], categorical_embeddings=["city"])
```

```bash
python examples/embeddings.py
python examples/transformer.py
```

`examples/transformer.py` uses `backbone="transformer"` over those field tokens. `backbone="set"` is the same tokens with a mean pool and no attention. Both keep the member axis explicit (`k=1`).

## Other checks

```bash
pytest
python examples/advanced.py
python examples/jax_composition.py
```

`pytest` covers the weight reduction, the mixture-versus-parameter-average variance check, the TabM batch-ensemble equation, preprocessing fitted only on the training partition, save/reload, a gradient through the numeric map, the default recipe, search off versus `search=True`, learned embeddings, the transformer, binary and multiclass `predict` / `predict_proba`, and classification search scored by weighted accuracy.

## Weights

`sample_weight` is a direct per-example loss weight for classification, regression, normal outputs, and custom objectives. The reduction is

```text
sum(w * loss) / sum(w)
```

inside each minibatch (`batch_weight_sum`). It is not row sampling. Class weights are a separate multiplier and are applied once. There is no weighted sampler.

DeepTab 2.1.0 does something different: classifier `sample_weight` drives a `WeightedRandomSampler`, while regressors and LSS models reject the keyword, including an explicit `None`.

## Prediction

Member losses are averaged during training. Member predictions are aggregated only at prediction time:

- regression `predict`: mean of member outputs
- classification `predict_proba`: `mean_probabilities` (default) or `mean_logits`
- `predict_dist`: `mixture` (default) or `mean_parameters`

A mixture of N(-5, 1) and N(5, 1) has variance 26. Averaging the parameters produces a unit normal.

## Benchmark

`examples/benchmark.py` fits the default recipe, one `search=True` run, and default XGBoost on two Grinsztajn medium splits. `examples/benchmark_torch.py` fits DeepTab and PyTabKit on those same splits with a 50-epoch cap. The numbers and the limits of the comparison are in `/cursor/stores/self/docs/benchmark.md`.

## What is deferred

Piecewise-linear and periodic numeric embeddings, TabM-mini, TabM-packed, RealMLP, full-covariance heads, and a general distribution library. FlowJAX is the intended place for a richer family. Pandas and Polars are optional frame adapters.
