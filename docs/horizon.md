# Horizon

Where the no-argument fit loses, and which of those losses a pass is allowed to touch. The failure table is the state before the label z-score and before a legal classification TabM. Results are at the bottom. Nothing here is registered or published.

## Where the default loses

The residual MLP is the no-argument model. On the coverage rescore it robust-scales numerics and embeds string categoricals. Before this pass it does not standardize `y`. Classification and ordinal copy that recipe and then change the task.

| Failure | Tables | Latest Ashlar | Leader | Gap |
|---|---|---:|---:|---:|
| Tiny regression target | Ailerons `goal` | 0.1469 | TabM-D 0.8439 | 0.70 |
| Hard numeric regression | yprop_4_1 | 0.0257 | TabM-D 0.1116 | 0.09 |
| Classification | electricity, eye_movements, california | 0.7790, 0.5491, 0.8657 | XGBoost 0.8544, 0.6429, 0.8968 | 0.07–0.09 accuracy |
| Wine-like ordered score | wine quality, expected label | ordinal TabM 0.4178 | `XGBClassifier` 0.4900 | 0.072 |
| Small sample, rare levels | abalone ordinal | fit raises | residual regressor 0.5802 | no ordinal score |

Ailerons is the sharp one. After robust scaling, the residual still explains almost none of the variance: test R² 0.14688212752663465, RMSE 0.0003709, internal validation loss 1.32e-7. The test labels sit near −0.00086 with standard deviation about 0.00040. Squared error on that scale produces gradients far below the AdamW step, so the fit stalls near the mean. The same recipe with a quantile map and a label z-score reaches 0.84 on this split. yprop_4_1 is the milder version of the same stall: R² 0.0257 against a leader at 0.11.

Classification is a different gap. electricity, eye_movements, and california are balanced, so class weights do not explain it. The residual classifier is one SiLU MLP on cross-entropy. Before the classification pass there is no legal TabM classification call: `standardize_target` and `share_training_batches=False` both raise unless the task is regression, which is why those coverage cells are empty. Trees lead by 0.07 to 0.09 accuracy. TabM-D, a classification ensemble with its own feature map, sits between Ashlar and XGBoost.

Wine-like targets are ordered integers scored by the expected label. The ordinal TabM expectation is 0.41394214246508565. Soft stages reached 0.417837586328735. `XGBClassifier`'s expectation is 0.48996919003701056. The cumulative head, the softmax head, the histogram target, and the soft stumps all train a likelihood. The metric that wins is squared error of the expected label. Pol is the exception that already wins: eleven labels, a huge mass at 0 and 100, ordinal expectation 0.9922.

Small samples show up twice. Abalone has 2923 outer training rows and four ring values with a single row, so the stratified holdout raises and both ordinal fits are blank. The residual regressor still leads that table. Eye_movements (5325 train) and wine (4547 train) are small, but their gaps match the classification and wine-like rows above more than they match the holdout size.

## Strategies

1. **Put regression labels on a unit scale.** Fit the mean and population standard deviation on the internal training rows only, train on that z-score, and invert it in `predict`. A standard deviation below `1e-6` is treated as 1 so a constant label stays finite. This is the Ailerons fix, and the same switch RealMLP-TD calls `normalize_output`. It is regression-only. Classification and ordinal keep raw labels. Houses is a poor witness for this idea: its target variance is already about 0.33, and the gap analysis ranked a second target transform last on that file. Ailerons is the witness.

2. **Train the wine readout.** Add a squared-error term on `probabilities @ class_values` to the ordinal objective, beside the cumulative likelihood. Every likelihood-only change on the wine split has stayed within 0.01 of 0.414, while the winning control is an expectation. Leave the residual regression default alone for that experiment. Score wine and pol expected-label R² on the existing splits.

3. **Make a classification ensemble legal.** A `tabm(task="classification")` path that does not turn on the label z-score and does not require unshared batches. Quantile features stay explicit. Score electricity, eye_movements, and california accuracy against the frozen XGBoost and TabM-D numbers. Do not replace `classifier()` until that ensemble beats it on more than one of those tables.

4. **Fit through a rare level.** When a class has fewer than two rows, keep those rows in the training partition and stratify the rest, instead of raising. That is what makes `ordinal()` return a score on abalone. It does not promise to beat the residual regressor there. A smaller validation fraction for a few thousand rows can wait until that score exists.

## Label-scale pass

Strategy 1 only.

`tabular()` and `TabularEstimator()` set `standardize_target=True`. `classifier()`, `ordinal()`, and `tabm(task="ordinal")` keep it off, including when the flag is omitted. An explicit `True` on a non-regression task still raises. An explicit `False` leaves regression labels on their original scale. A checkpoint that omits the flag still loads as off, so old predictions do not get a second inverse. Bare `tabm()` is regression and inherits the flag; the coverage TabM calls already pass `True` and are not refit.

Rescore `tabular()` on the same coverage splits for yprop_4_1, Ailerons, Bike_Sharing_Demand, pol, abalone, and diamonds, and on the houses split. Those are the regression rows the flag touches. Classification, wine ordinal, and the frozen XGBoost and TabM-D cells stay as recorded.

## Classification pass

Strategy 3.

`tabm(task="classification")` uses the TabM trunk with `prediction_policy="mean_probabilities"`. The label z-score stays off. Members share one minibatch; `share_training_batches=False` still raises. `numeric_scaling` stays `robust` unless the caller passes a quantile map. `classifier()` is not replaced by this call. The scored row caps `max_epochs` at 50, the same ceiling as the other neural rows. The recipe's own cap remains 200.

Rescore that call on the electricity and eye_movements coverage splits and on the cached California classification split. XGBoost and TabM-D stay as recorded. A current `classifier()` fit on the California split is the control for the old ReLU row.

## Wine readout pass

Strategy 2, on the wine split.

The ordinal TabM call that scored 0.41394214246508565 gains `expectation_weight=1`. The loss is the class negative log-likelihood plus that coefficient times `(probabilities @ class_values - y)^2`. Probabilities are the mean of the member probabilities. The flag stays off unless it is set. The residual regression default stays as it is.

## Rare-level pass

Strategy 4.

An iid classification or ordinal split keeps a class with fewer than two rows in the training partition and stratifies the remaining rows at `validation_size`. Regression does not stratify. The abalone score is `ordinal()` on the existing coverage split.

## Later

Piecewise-linear numeric embeddings, a longer epoch cap, and any change to the SiLU residual width. Robust scaling and categorical embeddings are already the predictor default and are not redone here.

## Result

Strategy 1 is in. `tabular()` z-scores regression labels. The exact scores are in `benchmark.md` under “Label z-score”, from `/tmp/ashlar-bench/horizon-results.json`.

Ailerons test R² moves from 0.14688212752663465 to 0.8417441807864341. The training-label standard deviation is 0.0004092321953405172, and the best internal validation loss is 0.17182528972625732. The residual now leads XGBoost at 0.8274166850231689 and trails TabM-D at 0.8438557053456597 by 0.002112. yprop_4_1 moves from 0.025728009549908726 to 0.06352824036932392 and still trails TabM-D at 0.11157329967795337. Bike, pol, and diamonds each fall by less than 0.013. Abalone moves by −0.000170. No winner on these rows changes.

Houses has training-label standard deviation 0.5593105375431461. Robust scale with the label left alone scores 0.7508186061878339. Robust scale plus the z-score scores 0.7551135965443034. The earlier mean/std residual is 0.7615065490594078, so robust scale is the drop from that score and the z-score itself is +0.004295. TabM-D at 0.8366028121954286 still leads houses.

## Classification result

Strategy 3 is in. `tabm(task="classification")` keeps shared batches, leaves the label z-score off, and uses `mean_probabilities`. Quantile scaling stays an explicit argument. Exact scores are in `benchmark.md` under “Classification TabM”, from `/tmp/ashlar-bench/tabm-classification-results.json`.

Electricity test accuracy is 0.8024282560706402, from `classifier()` 0.7789985952237608, behind TabM-D 0.8203893237005819 and XGBoost 0.8543548063415614. Eye_movements is 0.5716072545340838, from `classifier()` 0.5490931832395247, ahead of TabM-D 0.5641025641025641 and behind XGBoost 0.642901813633521. California is 0.881380977968834, from the current `classifier()` 0.8507522837184309, behind TabM-D 0.8915905427189683 and XGBoost 0.8968296614723267. The ensemble leads the residual classifier on all three tables. XGBoost still wins each row, so `classifier()` stays the no-argument path.

## Wine readout result

Strategy 2 is in, on the wine split. The mix is the ordinal negative log-likelihood plus `1.0 * (probabilities @ class_values - y)^2`, with probabilities equal to the mean of the member probabilities. Expected-label test R² is 0.3922491062217387. That is 0.0217 below the ordinal TabM likelihood score 0.41394214246508565, and the gap to the `XGBClassifier` expectation 0.48996919003701056 is 0.0977. The fit ran 50 epochs. The best mixed validation loss is 1.6458075046539307. The residual regression default was not changed. Pol was not refit. The exact row is in `benchmark.md`.

## Rare-level result

Strategy 4 is in. Rings 1, 24, 25, and 26 each have one row in the abalone outer training partition, and those rows stay in training. `ordinal()` expected-label test R² is 0.5780754759977107. That leads the TabM-D expectation 0.5534104987332197 and the `XGBClassifier` expectation 0.5313882944755375. It is 0.0020 below the current residual regressor, 0.5800630957948113. The residual regression default was not changed. The exact row is in `benchmark.md` under “Rare-level split”.
