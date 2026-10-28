# Clearing the five XGBoost scores

The bar is the one in `wine-ideas.md`. A test score meets its cut when it reaches the frozen XGBoost number. An equal score counts. Nothing here is fit, registered, or published. The residual regression default stays the regressor through this sequence.

| Score | Cut | Nearest recorded score | Shortfall |
|---|---:|---:|---:|
| Wine expected label | 0.48996919003701056 | 0.41855174134115225 | 0.0714 |
| Wine regression | 0.4284963911647759 | 0.4117637706326597 | 0.0167 |
| Electricity accuracy | 0.8543548063415614 | 0.8203893237005819 | 0.0340 |
| Eye_movements accuracy | 0.642901813633521 | 0.5716072545340838 | 0.0713 |
| California accuracy | 0.8968296614723267 | 0.8915905427189683 | 0.0052 |

The nearest wine expected-label score is the width-32 bottleneck. The other four nearest scores are TabM-D, except eye_movements, where Ashlar classification TabM is ahead of TabM-D. Trunk changes on wine already sit between 0.2807 and 0.4186. The shortfalls above are why the next build is a new function class.

Splits, holdout, bases, ridge grid, pair cap, and the refit-on-the-outer-train rule stay as written in `wine-ideas.md`. Paper-validation is reported and is not used to fit or to select. Stored XGBoost numbers are not refit.

## Order

**1. Build the additive fitter.** One differentiable model: a smooth main effect per numeric column, a width-8 embedding per categorical column, and at most 10 pairwise products. Knots, the ridge grid `{1e-4, 1e-2, 1, 100}`, and the pair cap are the declared constants. Pairs and the ridge are chosen only when the internal-validation metric strictly rises. The metric is expected-label R², regression R², or accuracy, matching the task. Class values and column kinds come from the fitting rows. The same body emits all three readouts.

Success for this step is a fit on the saved wine split that writes both wine test numbers under that protocol. Those numbers do not have to meet the cuts yet. A fit that required a new knot grid, a larger pair cap, or a look at the test rows is not this step.

**2. Score that fitter on all five cuts.** Wine expected label first, then wine regression on the same split, then electricity, eye_movements, and california. Record each test score against its cut before building anything else.

Success for this step is every score at or above its cut. The bar is cleared and steps 3 and 4 stay unbuilt. A wine expected-label score below 0.48996919003701056 ends the claim for this fitter. Finish the other four scores anyway, so the next build knows which rows are still short. Do not add pairs, change the ridge grid, or move a knot after a test number is in hand.

**3. Add the learned gate on the short rows only.** Two specialists share the same pair budget of 10. The gate is main effects only. On an equally spaced label with at least three values, the high-mass set grows from the mode while the internal-validation metric rises. On a binary label, the gate is the single feature and threshold that most raises that metric. Rescore only the rows that fell short in step 2. Leave a row that already met its cut on the step-2 number.

Success is those rescored rows at or above their cuts. Rows that already met the cut stay met.

**4. Fit the learned diagonal neighborhood on any row that is still short.** One scale per numeric column, the same embeddings, and one bandwidth, chosen on the internal-validation metric. Regression and ordinal rows predict a distance-weighted mean. Classification predicts a distance-weighted vote. Rescore only the remaining short rows.

Success is those rows at or above their cuts. A row that is still short after this step stays short. This plan has no fifth model.

## What success looks like

All five of these are true on the saved splits:

- Wine expected-label test R² ≥ 0.48996919003701056
- Wine regression test R² ≥ 0.4284963911647759
- Electricity test accuracy ≥ 0.8543548063415614
- Eye_movements test accuracy ≥ 0.642901813633521
- California test accuracy ≥ 0.8968296614723267

The model that posts each number is one of the three bodies above, with ordinal expected labels, the embedding path, and differentiable parameters still on that body. A passing internal-validation score is not one of the five.

## What to skip

- Another head, penalty, learning rate, prior, bin embedding, epoch cap, bottleneck, histogram target, soft stump, or expected-label loss weight on the ordinal TabM trunk. That range is already measured.
- A classification-only retune of `tabm(task="classification")`. It has already been scored at 0.8024282560706402, 0.5716072545340838, and 0.881380977968834, all short of the three accuracy cuts.
- A hard axis-aligned leaf histogram. That copies the frozen booster and drops the differentiable body.
- Three full 100-tree boosters in place of the gate. The pair budget stays 10.
- Houses, cpu_act, sulfur, yprop_4_1, Ailerons, Bike_Sharing_Demand, pol, abalone, and diamonds. XGBoost does not hold those rows.
- Refitting XGBoost, redrawing the five splits, or selecting on paper-validation.
- Changing the residual regression default inside this sequence.
- Raising the pair cap, editing the ridge grid, or moving the knots after seeing a test score.

## 5. Differentiable oblivious ensemble

Steps 1–4 are scored in `benchmark.md`. None of the five cuts cleared. The additive weights are a ridge or logistic fit on a fixed basis. The gate mixes two of those fits. The neighborhood is a kernel smoother on the raw columns. One extra ReLU after the three TabM blocks, the width-32 bottleneck, moved wine expected-label R² from 0.41394214246508565 to 0.41855174134115225.

The model for this step is one layer of differentiable oblivious trees, the default architecture in Popov, Morozov, and Babenko, “Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data” (arXiv:1909.06312). An oblivious tree of depth 6 uses one split per level and a response at each of the 64 leaves. Feature choice and the soft threshold are 1.5-entmax, so a split can put its mass on one column and the routing is a smooth step. The layer averages 512 of those trees. Every feature logit, threshold, scale, and leaf is trained from the task loss in one backward pass.

This is the same kind of function as the booster that holds the cuts: an ensemble of depth-6 trees, with a separate value on each side of a split. The booster’s trees are unrestricted and built greedily. These trees are oblivious, which is the weaker family, so the layer uses more of them and fits them together. It is not another residual block, and it is not a linear map of a fixed hinge basis. The paper’s own default used 2048 trees and Quasi-Hyperbolic Adam at learning rate `1e-3`. This environment’s Optax has no QHAdam, and a learning rate of `1e-3` did not move a synthetic two-feature conjunction off the majority class within the epoch budget below. Adam at learning rate `0.1` did, on that synthetic conjunction, with 512 trees of depth 6. Those two substitutions are fixed before any of the five test scores.

Frozen for every table:

- One layer, depth 6, 512 trees, 1.5-entmax
- Adam, learning rate `0.1`, batch 512, at most 80 epochs, patience 16
- The internal holdout is `validation_size=0.2`, `random_state=0`, stratified for classification and ordinal. Regression is not stratified. Early stopping watches the task metric on that holdout: expected-label R², regression R², or accuracy
- The epoch count is chosen there. A fresh run then trains on the full outer train for that many epochs. The scored weights are the average of the last five of those epochs, or of every epoch when the run is shorter than five
- Columns are z-scored from the rows used in that phase. Regression labels are z-scored inside this model only. Ordinal and classification use cross-entropy. The ordinal readout is the expected label, `probabilities @ class values`
- Initial feature logits are uniform on `(0, 1)`. Initial thresholds are values from the fitting rows. Initial threshold scales are 2. Leaf responses start at standard normal
- Paper-validation is reported and is not used to fit or to select. XGBoost is not refit. The residual regression default stays the regressor

Success is the same five inequalities as above. A passing internal-validation score is not one of them. These constants do not move after a test score.

## 6. Priors on the three open cuts

The histogram booster cleared wine expected label and wine regression. Those two rows are not fit again. The open cuts are electricity accuracy 0.8543548063415614, eye_movements accuracy 0.642901813633521, and california accuracy 0.8968296614723267. A match counts. The bodies below stay differentiable. None of them grows a hard tree.

Three penalties, each the kind of preference a tree gets without being asked. A BART-shaped penalty on soft depth-2 trees: split probability `0.95 / (1 + depth)^2`, a sparse Dirichlet penalty on the feature simplex, a U-shaped penalty on the gate, and a Gaussian leaf penalty of scale `3 / sqrt(40)`. A variational logistic head on four soft quantile bins per column and every pairwise product of those bins, with a unit diagonal prior after the bin design’s second moment. An L1 penalty on the input gradient of a two-layer network, which charges a path for using many columns. Frozen before these three test scores: 40 trees, Adam at `0.05`, feature coefficient `1e-3`, gate coefficient `1e-3`, depth coefficient `0.1`, variational Adam at `1e-2`, gradient penalty `1e-2` with Adam at `1e-3`. Each uses batch 256, at most 80 epochs, and patience 16. The epoch count is the internal 20% holdout, `random_state=0`, stratified. A fresh run then trains on the full outer train for that many epochs, and the scored weights are the average of the last five. Paper-validation is not used to fit or to select. The residual regression default stays the regressor.

Scored in `benchmark.md`. None of the three open cuts cleared. The highest test accuracy on each row is the gradient penalty: electricity 0.7824603652418222, eye_movements 0.5697310819262039, california 0.8753358409457281. The BART prior did bind: feature peaks were 0.9486458897590637, 0.9897899031639099, and 0.9777435064315796, and the child split rate stayed near a quarter. That penalty produced a shallow sparse soft tree, and the shallow tree was short of every open cut.
