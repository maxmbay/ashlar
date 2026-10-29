# Fitting a table

[Ashlar](../README.md) trains a small residual MLP on a table. `TabularEstimator()` and `ashlar.tabular()` are the same default path. The pictures on this page use shorter epoch budgets than these defaults.

![A row becomes a prediction.](images/tabular-path.png)

Numeric columns are scaled. Categories become vectors. A backbone reads both, and the head emits the prediction.

![The fitted line follows the rows.](images/tabular-fit.png)

One numeric column, a few epochs, then `predict` on a grid. The line is the fit. The dots are the rows. [`examples/figures.py`](../examples/figures.py) draws these pictures from synthetic data.

## Defaults

`fit(X, y)` with no other arguments does the following.

Numeric columns are filled with the training-row median, centered on that median, and divided by the interquartile range, then passed through a smooth clip at ±3. `numeric_scaling="standard"` uses the mean and the standard deviation instead. An integer column stays numeric unless you name it as a category.

Regression labels are z-scored on the training rows. `predict` maps them back. `standardize_target=False` leaves `y` alone. Classification and ordinal labels stay on their original scale.

Every other column is categorical. It gets a learned embedding of width 8. Pass `categorical_embeddings=[]` for one-hot, or a list of names to embed those columns and one-hot the rest. Index 0 is the row for a missing value or a category that was not in the training rows.

The fit holds out 20% of the rows at random, then trains for up to 50 epochs with batch size 64, learning rate `1e-3`, and patience 10. Weight decay is `1e-4`. Feature dropout stays at 0. `patience=0` trains the full epoch budget. `validation_size=0` trains on every row.

The backbone is a residual MLP with two blocks of width 64 and a SiLU activation. `backbone_kwargs` can set `n_blocks`, `d_block`, `dropout`, and `activation` (`silu`, `relu`, or `gelu`).

## Tasks

```python
from ashlar import classifier, ordinal

classifier().fit(X, y)
ordinal().fit(X, y)
```

`predict` on a classifier returns the class label. `predict_proba` returns one probability column per class, in `classes_` order, and the columns sum to one. A binary problem is two classes.

An ordinal label has to be a real number. The sorted values are the class order. The default head is a cumulative logit: one score, and cutpoints that stay in order. `head="softmax"` gives each class its own score. `predict` returns the most likely label. `predict_score` returns `probabilities @ classes_`, the expected label.

![The expected score stays with the integer label.](images/ordinal-score.png)

`predict_score` is the expected label: class probabilities times the integer class values. The mean for each label sits on that integer.

`task="normal"` fits a diagonal normal. `predict_dist` returns the distribution. The default regression `predict` is the mean of the member outputs.

The mixture, flow, histogram, and quantile heads of the forecasters are not built into `TabularEstimator`. [`examples/table_heads.py`](../examples/table_heads.py) attaches any of them to a table through `task="custom"`, and the [README](../README.md#pick-a-head-for-the-shape-of-the-target) compares them on public tables.

## Embeddings

```python
model.fit({"rooms": rooms, "city": city}, y)
table = model.core_.categorical_embeddings.tables[0]
names = model.encoder_.embedding_columns[0].categories
```

Row 0 of `table` is the unknown category. `names` matches the rows after it.

![Categories with the same effect sit together.](images/categorical-embeddings.png)

Each category here is a learned 2-d vector, from `categorical_embedding_dim=2` on a synthetic table. Color is that category's effect on the target. Categories that move the target the same way land next to each other. The unknown row is left off the plot.

A set column is a cell that holds a list of labels. Pass `set_features=["tags"]`. The model mean-pools a learned token embedding. Lists longer than `set_max_length` (default 8) are truncated to the longest list in the training rows, capped by that maximum. An empty list becomes a zero vector.

## Search

Pass `search` to `fit` to run a search. `search=True` tries weight decay `1e-4` and `1e-3` for two epochs each, then trains the winner with the estimator's own epoch budget. Learning rate and feature dropout stay at the constructor values unless you put them in the space.

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

Classification trials are ranked by weighted accuracy on the validation rows. Regression, normal, and ordinal trials are ranked by validation loss. The chosen settings are `search_["chosen"]`. Leave `search` unset and the constructor settings are the whole fit.
