# Backlog

Order is priority. Scope is the technical change, not a calendar. Each item is unblocked by the prototype except where a dependency is named. None of these are part of the first release that was just built.

## 1. Full TabM forward parity

**Scope.** One frozen `tabm` 0.0.3 `TabM.make` without embeddings, weights copied into `PredictiveCore`, float32 CPU forward and a parameter gradient. Then the same with `LinearReLUEmbeddings` against `rtdl_num_embeddings`.

**Acceptance.** Max absolute difference reported for the forward and for one gradient, on a fixed input, in `probe-results.md`. A gap larger than a few float32 ulps is a defect to fix or an explicit, tested deviation. No dataset score required.

**Depends on.** The existing `LinearBatchEnsemble` equation. No new package.

**Risk.** Init layouts differ (chunked first `r`, shared bias draw, output `LinearEnsemble`). A shape mismatch should raise, not broadcast.

## 2. Paper preprocessing that the recipe still omits

**Scope.** Quantile or piecewise-linear numeric embeddings as an optional encoder, off by default. Store the fitted bin edges in the schema next to the mean and standard deviation. Keep the gradient contract: only the transform that is actually an Equinox module is differentiable.

**Acceptance.** A fixed-weight embedding matches the reference on one batch, or the README states the remaining formula difference. `TABM_PAPER` grows a flag rather than changing the current default silently.

**Depends on.** Item 1 for the embedding bit, and the schema JSON.

**Risk.** Quantile edges fitted on the full frame would leak. The fit path has to stay on the training partition only.

## 3. TabM-mini and packed ensembles

**Scope.** One additional backbone class, selected by an explicit `arch_type` that is currently rejected. Same head and objective contracts. Same member axis.

**Acceptance.** `arch_type` is no longer a hard error for the one implemented variant. A rank or chunk mismatch still raises `ShapeError`. Default `tabm()` stays the batch-ensemble recipe.

**Depends on.** Item 1, so the mini variant is not the first time the forward is compared.

**Risk.** DeepTab's `model_type="mini"` is a different code base. Copying DeepTab's defaults would make the name "TabM" ambiguous again.

## 4. One richer distribution, behind an optional extra

**Scope.** A second probabilistic head whose density and sampling come from FlowJAX, imported only when that extra is installed. The estimator still returns `predict` as a point and `predict_dist` as an object with `mean`, `variance`, `log_prob`, and `sample`. Document whether members are mixed or averaged.

**Acceptance.** The core install still excludes FlowJAX. One example trains the new head on CPU with direct `sample_weight`. The diagonal normal path is unchanged.

**Depends on.** FlowJAX 19.1.2 or a later pin recorded at implementation time. Python `>=3.11`.

**Risk.** A general distribution zoo. Stop after one family.

## 5. Real grouped and temporal fixtures

**Scope.** One public table with a group id and one with a time order, small enough for CPU. The example prints the training-partition category vocabulary and shows that a category present only in the held-out rows is encoded as unknown.

**Acceptance.** The script fails if a held-out category appears in the fitted vocabulary. No leaderboard claim.

**Depends on.** A dataset whose license allows redistribution, or a download step the user runs. Do not vendor a large archive into the store.

**Risk.** Treating a random split of a temporal table as the result. The example has to use `split="temporal"` or `split="group"`.

## 6. Publish name

**Scope.** No upload was made. The import is `ashlar` and the distribution string is `ashlar-tabular`. On 26 Oct 2026, `https://pypi.org/pypi/ashlar-tabular/json` returned 404. That check does not reserve the name. PyPI `ashlar` 1.20.0 is an unrelated microscopy package, so the import collides if both are installed. `gnet-tabular` was only the working name.

**Acceptance.** Before any upload, repeat the PyPI check. Either keep `ashlar-tabular` if it is still unused and accept the import collision in the install notes, or change the import and record the new string. Do not register the name as part of this backlog item.

**Depends on.** A fresh PyPI lookup at release time. The October 404 goes stale.

**Risk.** Uploading import `ashlar` beside the microscopy package, or finding that `ashlar-tabular` was taken after this check.

## 7. GPU timing protocol

**Scope.** A script that reports cold start, compile, warm train step, warm predict, and peak memory for one fixed `TABM_PROBE` and one `TABM_PAPER` width, on whatever device JAX sees. Block on `block_until_ready`. Do not compare with PyTorch unless the same batch and the same matmul precision are stated.

**Acceptance.** One markdown table appended to a future probe log. No claim that JAX is faster.

**Depends on.** A machine with a GPU. This item does not require buying one. CPU remains the default for tests.

**Risk.** Quoting a compile-included number as a step time.

## 8. sklearn estimator checks, where they apply

**Scope.** Run `sklearn.utils.estimator_checks.check_estimator` on a tiny regression configuration and record which checks cannot apply to a stochastic neural net with a validation split.

**Acceptance.** A list of passed checks and a list of checks that were skipped with a reason. `clone`, `get_params`, and `set_params` already go through the constructor; this item is the audit.

**Depends on.** scikit-learn 1.9 behavior for checks that assume `predict` is deterministic given `y` only. Early stopping and dropout make some checks the wrong tool.

**Risk.** Disabling checks until the suite is green. A skip needs a sentence.

## 9. RealMLP

**Scope.** Not in this backlog's first cut. PyTabKit already ships a tuned RealMLP. A JAX port is a separate fidelity project of the same kind as item 1, and it should not start before TabM forward parity.

**Acceptance.** None until item 1 is done and a user asks for the port.

**Risk.** Growing a model catalogue to fill a comparison table.

## Explicitly out of the backlog

A general trainer, an experiment dashboard, AutoML, a foundation-model checkpoint port, and a family of target transforms. Target transforms can be reconsidered only with an inverse stored in the schema and with fit restricted to the training partition. They are not a silent follow-up to item 2.
