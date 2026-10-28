# General methods aimed at the default tree

The bar is a general-purpose tabular model that beats or matches default XGBoost, with ordinal expected labels, categorical embeddings, and a differentiable parameter vector available on that same model. An equal test score counts as a match. Wine is the first test of that claim. It is one split among the benchmark rows where XGBoost currently holds the highest recorded score. This note does not fit anything.

## Where XGBoost currently wins

Scores below are the latest recorded test numbers in `benchmark.md`. Regression and ordinal rows are R². Classification rows are accuracy. XGBoost is `XGBRegressor()` or `XGBClassifier()` with no arguments. On an ordinal row the classifier is read out as `probabilities @ class_values`.

| Dataset | Task | XGBoost | Next recorded score |
|---|---|---:|---:|
| wine quality | expected label | 0.48996919003701056 | 0.41855174134115225, late bottleneck |
| wine quality | regression | 0.4284963911647759 | 0.4117637706326597, TabM-D |
| electricity | classification | 0.8543548063415614 | 0.8203893237005819, TabM-D |
| eye_movements | classification | 0.642901813633521 | 0.5716072545340838, Ashlar TabM |
| california | classification | 0.8968296614723267 | 0.8915905427189683, TabM-D |

The other rows in that file have a different leader. Houses, cpu_act, sulfur, yprop_4_1, Ailerons, Bike_Sharing_Demand, pol, abalone, and diamonds stay out of the cut. On the five houses redraws, the unshared TabM mean is 0.8341 and the XGBoost mean is 0.8319, with TabM-D at 0.8358.

The wine file is `/tmp/ashlar-bench/extra-regression-splits/wine_quality.npz` (4547 / 585 / 1365). Electricity and eye_movements are `/tmp/ashlar-bench/coverage-splits/electricity.npz` and `eye_movements.npz` (10000 / 8542 / 19932 and 5325 / 684 / 1599). California is `/tmp/ashlar-bench/split-california.csv` (10000 / 3190 / 7444).

Changes that stayed on the ordinal TabM trunk, and only changed its head, penalty, rate, prior, bins, width, target, or loss, are already measured on the wine split. The expected-label scores run from 0.28070258803405446 (surface learning rate) to 0.41855174134115225 (width-32 bottleneck). Softmax, weight decay 1.0, the Jeffreys prior with binned embeddings, 100 epochs of that pair, HL-Gaussian, eight soft stumps, and an expected-label squared-error term are inside that range. Exact rows are in `benchmark.md`. Those runs leave the mixed trunk in place. The three methods below replace the function class and use the same procedure on every table.

What every method learns from the training rows, rather than from a fixed label set: the columns and their kinds, the distinct label values, and, for an integer or ordinal target, the numeric value of each class. The prediction for that target is `probabilities @ class_values`. A regression target uses a single real output. A classification target uses one score per observed class and is judged by accuracy. Categorical columns use a learned vector of width 8, the library default, fit on levels seen in the fitting rows. A level absent from those rows uses one shared unknown vector. The four tables in the cut are numeric, so that path is idle there.

Declared in advance, and shared by every table: internal holdout `validation_size=0.2`, `random_state=0`; piecewise-linear numeric bases with knots at the internal-training quantiles 10% through 80%; ridge grid `{1e-4, 1e-2, 1, 100}`; at most 10 pairwise terms. Paper-validation is reported and is not used to fit or to select. The final fit, after selection, uses the full outer train, which is the row count default XGBoost already trains on. XGBoost is not refit.

## 1. Main effects, then a few pairs

This is the method to try first.

Each numeric column gets one smooth shape, estimated from every training row. A categorical column contributes its embedding. A small set of pairwise products is added on top. Lou, Caruana, Gehrke, and Hooker (KDD 2013) report this function class, GA2M, in line with gradient-boosted trees when the signal is low order. The shapes here are parameters, so the ordinal expectation, the embeddings, and the body train as one differentiable model.

A depth-6 tree stores a separate leaf value in each region. A smooth effect that is shared across the table is re-estimated once per leaf, which is where a peaked integer loses the tail probabilities that move an expectation, and where a binary boundary loses rows that would have agreed on a main effect. The pairs are the part of the model that can carry an interaction without opening a new leaf for every combination.

Learned on each table: the shape coefficients, the ridge value, and which pairs enter. A pair is accepted only when the internal-validation metric strictly rises. The metric is expected-label R² on an ordinal row, R² on a regression row, and accuracy on a classification row. The search stops at 10 accepted pairs or when no candidate improves the slice. The knot locations are the internal-training quantiles named above. A monotone re-expression of a numeric column may be fit on those same rows before the knots are placed.

The idea is false if any of the five test scores falls short of its frozen XGBoost number. An equal score meets the cut.

## 2. Two specialists behind a learned gate

The sample is split into a high-mass piece and a remainder. Each piece has its own predictor. A gate mixes them. The gate and both predictors are differentiable, and the pair budget of 10 is split across the two predictors. The gate itself is main effects only. The ordinal expectation, the class scores, and the embeddings are the same readouts as in the first method.

The high-mass piece is learned. When the training label has at least three distinct values and those values lie on an equally spaced grid, start at the modal value and add adjacent values from the observed support while the internal-validation metric rises. That set is the piece. The gate is the probability of membership, fit on the internal training rows. One specialist sees the rows inside the set. The other sees the rest. The prediction is the law of total expectation: gate times the inside prediction, plus the complementary weight times the outside prediction. For a classification label the two predictions are class distributions and the reported score is the accuracy of the gated distribution.

When the label has two values, or is not equally spaced, there is no label set to grow. The gate is then the single numeric feature and threshold that most raises the internal-validation metric. One specialist trains on each side of that threshold. The threshold and the feature are chosen only on the internal slice.

A single tree pays for every split with one loss, so the majority of the rows chooses the partition and the remainder is whatever is left in the leaves. A learned gate spends a specialist on that remainder. On a binary table the remainder is a region of the features. On an integer rating it is the label mass away from the mode.

The idea is false if any of those five scores falls short. An equal score meets the cut. A result that uses three full 100-tree boosters would raise capacity as well as split the sample. The budget here is the one additive body, split as above.

## 3. A neighborhood in a learned metric

Predict from the training labels near the query. The distance is diagonal: one positive scale per numeric column, plus one bandwidth. A categorical column uses the same embedding width as the other methods, and the distance includes those coordinates. Scales, embeddings, and bandwidth are chosen on the internal-validation metric only. Regression and ordinal targets use the distance-weighted mean of the training labels (Nadaraya–Watson). Classification uses the distance-weighted class vote. The scales are parameters.

An axis-aligned tree approximates an oblique level set with a staircase and spends leaves on that staircase. A neighborhood averages the training labels along the learned metric, including rows the staircase would have separated. The failure mode of a distance is a column that barely moves the label and still stretches the neighborhood. The learned scales are what decide which columns collapse.

The idea is false if any of those five scores falls short. An equal score meets the cut.

## The experiment

Run method 1 first. Wine is the first test. The same fitted procedure then runs on electricity, eye_movements, and california. The claim is false if any of the five scores falls short of its frozen XGBoost number. An equal score meets the cut. Falling short on the wine expected label is enough to stop; the other four scores are part of the same claim.

On every table:

1. Load the saved split named above. Do not redraw it.
2. Hold out the internal slice with `sklearn.model_selection.train_test_split`, `test_size=0.2`, `random_state=0`. Stratify when the task is classification or ordinal. A class with fewer than two outer-train rows stays in the fitting side, which is the library rule, and `validation_size` applies to the remaining rows. Regression does not stratify. On the wine outer train the stratified call yields 3,637 fitting rows and 910 selection rows. That slice is the only place pairs, the ridge, the gate, and the metric scales are chosen.
3. Build the numeric bases and the categorical embeddings from the fitting rows, as declared above.
4. Fit the main-effect model for the task. Choose the ridge from the grid by the internal-validation metric.
5. Forward-select pairs under the cap and the strict-improvement rule.
6. Retune the ridge once on the same grid with the accepted pairs fixed. Refit that formula on the full outer train. Selection does not run again.
7. Score the test partition. Leave the stored XGBoost numbers as they are.

Wine reports two numbers from that one split. The expected-label test R² has to reach 0.48996919003701056. The regression test R², from the squared-error form of the same procedure, has to reach 0.4284963911647759. Electricity accuracy has to reach 0.8543548063415614. Eye_movements accuracy has to reach 0.642901813633521. California accuracy has to reach 0.8968296614723267. Reaching a number includes equaling it.

A wine expected-label score above 0.41855174134115225 and below 0.48996919003701056 means the additive model beat the measured trunk and fell short of the leaf histogram on the first test. The general-purpose claim is still false in that case. A score equal to 0.48996919003701056 meets the cut.

Methods 2 and 3 use these splits, this holdout, and these five cuts. Only the body changes.

Nothing in Ashlar changes on the strength of this note. The residual regression default stays the regressor.
