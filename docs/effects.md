# Entity effects

[Ashlar](../README.md) can pool what it learns about a store, a patient, or a product across all of them. A table with many entities and few rows for most of them is where an ordinary categorical embedding struggles: a store with three rows gets an embedding fitted to three rows, and a store that opens tomorrow gets whatever the network makes of a category it never saw. `random_effects` gives the grouping column the treatment a mixed model would: each group's effect is drawn from a population, the population's spread is learned, and a group with little data is pulled towards the rest.

```python
from ashlar import ASRRegressor

model = ASRRegressor(
    task="gaussian",
    random_effects={"store": ["intercept", "embedding"]},
).fit(X_train, y_train)

model.predict_dist(X_new)          # seen stores with their own effect, new stores with the population's
model.random_effects_["store"]     # the learned spread and every store's posterior
```

## Offsets

`random_effects={"store": "intercept"}` takes `store` out of the network's inputs and gives every store an offset on the head's location, `b_store ~ N(0, tau²)`, with `tau` learned. Each store's offset has a posterior of its own, fitted with the network by variational inference: training maximizes each row's expected log likelihood under its store's posterior, minus how far every posterior moves from `N(0, tau²)`. A store with three rows stays near the average store; a store with three thousand rows goes where its rows put it.

Stochastic training moves a store's posterior only on the minibatches that hold its rows, so a store with three rows ends short of where they put it, and a store whose rows all fell in the validation partition is never fitted at all. So when training stops, `fit` sets every store's offsets to the posterior the variational objective converges to under the trained network: Laplace's method from `N(0, tau²)` on all the store's rows, both partitions, which a `gaussian` head makes exact. Several grouping columns take turns, three times over, each with the others at their means. With labels from `ashlar.observed`, the posteriors stay as training left them.

At prediction a store seen in training reads its posterior, so its uncertainty about its own effect widens its distribution, and a store not seen in training, or a missing store, reads `N(0, tau²)`: the population's answer, wider by the spread between stores. Passing the column as missing gives the population distribution for any row.

The location is the output that moves the whole distribution: the mean of `gaussian` and `student_t`, every component of `mixture` together, the shift of `flow`, the median of `quantile` and `quantile_spline`, the log mean of `poisson` and `negative_binomial`, and the severity's log mean of `hurdle`. Those nine heads and `regression` take `random_effects`. On `gaussian` and `mixture` the offset's integral is exact at prediction, since a normal offset only widens each normal; the other heads, and training, read it on 16 and 8 equal-weight normal nodes, moment-matched so that a normal's expected log likelihood is exact.

A numeric feature named among the terms gets a random slope as well: `{"store": ["intercept", "price"]}` adds `b_store,price × price` (on the scaled feature), with its own `tau`, for stores whose sales answer price more or less than most. Several grouping columns add up, crossed or nested.

## Embeddings

An offset moves a store's whole distribution up or down, the same at every price and on every weekday. `"embedding"` among the terms also feeds the network a learned vector per store, shrunk the same way: `e_store ~ N(0, diag(tau_e²))`, with `tau_e` learned per dimension and a posterior per store. The network reads it like any other input, so a store can interact with the other features, such as a store whose sales jump on promotion days more than most. A store with few rows keeps a vector near 0, the average store, where the network has learned what an average store does, and a store not seen in training reads 0.

The vector is drawn from its posterior on every training step and read at its mean at prediction. Pass `["intercept", "embedding"]` for both, or `"embedding"` alone for the interaction without the offset.

## Nesting

A broader class can be a grouping column of its own:

```python
random_effects={
    "store_type": ["intercept", "embedding"],
    "store": {"terms": ["intercept", "embedding"], "within": "store_type"},
}
```

The store's offset adds to its type's, and its vector to its type's vector, so a store with few rows is pulled towards stores of its type, and the type towards every store. A new store of a known type starts from its type rather than from the average store. A nested column with an embedding needs its class to have one too, since its vector adds to the class's, and nesting can chain (store within city within region).

## New groups from a few rows

`with_context(X, y)` returns a copy of the model whose stores in `X` have their offsets' posteriors updated from those rows, without refitting the network:

```python
updated = model.with_context(X_first_week, y_first_week)
updated.predict_dist(X_next_month)
```

A store the model has not seen starts from the prior, `N(0, tau²)`, added to its class's offset when it is nested, so five rows of a new store give it an offset of its own, pulled towards the average store exactly as the fitted ones were. A store seen in training starts from its fitted posterior, so pass rows it was not fitted on. Stores not in `X` keep their posteriors, and the model it was called on is unchanged. The update is Laplace's method on each store's log posterior, which a `gaussian` head makes exact in one step. An embedding is not updated: a new store's stays at the average store's (or its class's).

## Reading the effects

`model.random_effects_` maps each grouping column to:

- `terms`: the offset terms, `intercept` and the slopes.
- `tau`: the learned standard deviation of each term between groups, in the units of the location: `y` for the normal-family heads, the log mean for the count heads and for `hurdle`'s severity.
- `categories`, `mean`, and `sd`: every group seen in training and its posterior. A nested column's means are its deviations from its class.
- `within`: the class it is nested in, or `None`.
- With `"embedding"`: `embedding_tau`, `embedding`, and `embedding_sd`, the vectors' spread and posteriors.

With `backbone="additive"`, `shape_functions` gives the offsets as one more term, `store (random effect)` ([docs/additive.md](additive.md#reading-the-terms)).

## Limits

- A grouping column cannot be a set feature or a monotone feature, and a slope cannot be on a monotone feature, which a group's slope could turn around.
- `random_effects` needs a built-in head without a cure fraction and `share_training_batches=True`.
- The offsets move the location only. A store whose sales are more variable than others', not just higher, gets that from the embedding, which the network can read into the spread, not from an offset.
- The variational posteriors are independent normals per group and term, so the posterior correlation between a group's intercept and its slope is not kept.

## Against the alternatives

<!-- results of examples/benchmark_effects.py -->
