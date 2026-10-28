# What wine quality can borrow from gradient-boosted trees

Two questions, on the saved medium split (4547 / 585 / 1365). Nothing here was refit. Nothing was registered or published.

The short answer: on this split the tree lead is mostly inside XGBoost. The default classifier, read out as an expected quality, scores 0.48996919003701056. The default regressor scores 0.4284963911647759. The ordinal TabM scores 0.41394214246508565. The same tree hyperparameters account for 0.0615 of the 0.0760 gap to the classifier, and the tree regressor accounts for the other 0.0146. Changes to the smooth TabM head, penalty, learning rate, prior, bins, width, and training length left Ashlar between 0.2807 and 0.4186. The piece to borrow is an axis-aligned partition whose leaf is a class histogram, grown with the multiclass gradient and hessian and shrunk across stages. The expectation readout is already in Ashlar.

## Facts

Scores are test R². Exact values are in `benchmark.md`. The ordinal rows use `predict_score`, the expected label. The most likely label is listed only where it separates the expectation from the mode.

| Readout on this split | Test R² |
|---|---:|
| `XGBClassifier` expectation | 0.48996919003701056 |
| `XGBClassifier` most likely label | 0.35397234731363514 |
| `XGBRegressor` | 0.4284963911647759 |
| Ordinal TabM expectation | 0.41394214246508565 |
| Ordinal TabM most likely label | 0.32575728416920047 |
| Regression TabM | 0.40493251825263477 |
| TabM-D regression | 0.4117637706326597 |
| Ordinal residual expectation | 0.359192121230657 |

Arithmetic on those scores:

- Classifier expectation minus ordinal TabM: 0.0760.
- Classifier expectation minus `XGBRegressor`: 0.0615.
- `XGBRegressor` minus ordinal TabM: 0.0146. Those two pieces add to the 0.0760.
- Ordinal TabM minus regression TabM: 0.0090.
- Classifier expectation minus its own mode: 0.1360. Ordinal TabM expectation minus its own mode: 0.0882. Mode gap alone: 0.0282.

`XGBClassifier()` in XGBoost 3.1.2, after fit, uses depth 6, shrinkage 0.3, 100 trees, subsample 1, column sample 1 at tree, level, and node, L1 0, L2 1, minimum child weight 1, gamma 0, and no early stopping (`wine-architecture.md`). Quality values 3–9 were mapped to `0..6` for the fit and mapped back before the expectation. That control is a nominal multiclass model plus an expected score, not an ordinal likelihood (`benchmark.md`). `XGBRegressor()` is the no-argument regressor. The two XGBoost rows share that recorded tree configuration. The 0.0615 between them is the objective and the readout.

The ordinal TabM call is `tabm(task="ordinal", prediction_policy="mean_probabilities", numeric_scaling="quantile_tabr", max_epochs=50, random_state=0)`. The trunk is `k=32`, `d_block=512`, three blocks, hidden dropout 0.1, ReLU, AdamW at `2e-3`, weight decay 0, batch 256, patience 16. That fit used all 50 epochs. The best internal validation loss was 1.163618564605713. Patience did not restore an earlier checkpoint.

The label is an integer from 3 to 9. In the file, 5 and 6 are 2,138 and 2,836 of 6,497 rows. Scores 3 and 9 are 30 and 5 rows. On the test split, 76.6% of rows are 5 or 6. A linear model on all 11 features reaches test R² 0.292. Alcohol, volatile acidity, and sulphates reach 0.272. Wine has no stored categorical column. All 11 features are numeric.

The same benchmark does not show a general tree lead for this architecture. On five houses redraws the TabM recipe's mean test R² is 0.8341 and default `XGBRegressor` is 0.8319. On cpu_act and sulfur, the regression TabM is above `XGBRegressor` (0.9852 vs 0.9744, and 0.8498 vs 0.7751). Wine regression is the table where `XGBRegressor` is ahead (0.4285 vs 0.4049).

### What was changed, and the expected label

Each row starts from the 0.4139 call and changes one thing, except the prior-and-embedding pair and the 100-epoch rerun of that pair.

| Change | Expected-label R² | Against 0.4139 | Gap to 0.490 |
|---|---:|---:|---:|
| One-hot softmax head | 0.3956767364524363 | −0.0183 | 0.0943 |
| Weight decay 1.0 | 0.3799832563545966 | −0.0340 | 0.1100 |
| Surface learning rate, chosen rate 0.1 | 0.28070258803405446 | −0.1332 | 0.2093 |
| Jeffreys prior plus three 8-bin embeddings | 0.4028560414884015 | −0.0111 | 0.0871 |
| That pair, patience 0, 100 epochs | 0.4028560414884015 | −0.0111 | 0.0871 |
| Late bottleneck, width 32 | 0.41855174134115225 | +0.0046 | 0.0714 |
| Softmax with HL-Gaussian variance 0.5 | 0.4150779471645716 | +0.0011 | 0.0749 |

The softmax run improved the validation negative log-likelihood, from 1.1636 to 1.0034, and lowered the expected label. It stopped at epoch 31. The weight-decay run used all 50 epochs. Optax multiplies the penalty by the learning rate, so weight decay 1.0 with learning rate `2e-3` shrinks weights by about `1 - 0.002` per step. The surface run warmed up for 15 steps at `1e-4`, measured a negative curvature, and clipped the rate to 0.1. It stopped at epoch 30. The prior run froze the cumulative cuts at a Jeffreys prior from the internal training counts and embedded eight train-only quantile bins of alcohol, volatile acidity, and sulphates. It stopped at epoch 30. The 100-epoch rerun of that same pair kept the epoch-14 checkpoint. The first 50 validation losses matched. The bottleneck and the histogram ran on the plain cumulative setup's trunk, the histogram with `head="softmax"`. The histogram stopped at epoch 47. Its validation loss is a different objective, so 1.4194 is not comparable to 1.1636. The highest expected label in the table is the width-32 bottleneck, 0.0046 above 0.4139 and 0.0714 below 0.490. None cleared 0.01 above 0.4139.

The 0.414 cumulative model itself was not refit past 50 epochs. The 100-epoch result belongs to the prior-and-embedding model.

## Judgments

The 0.0615 inside XGBoost is the part of the gap a head change would have had to match. It did not. The classifier and the regressor share the recorded depth, shrinkage, row sample, column sample, and penalties. They differ in the loss that grows the trees and in the value stored for each leaf: class probabilities whose mean is the prediction, versus a squared-error fit to the integer. That comparison does not include a dump of this fit's split gains or leaf histograms. The inference uses the scores plus the definition of the two fitted models.

The softmax result narrows the inference. A softmax over the TabM features can represent any distribution on the seven qualities, which is the support of the classifier's leaf vectors. Trained with one-hot cross-entropy it scored 0.3957, below both the cumulative head and the regression TabM. HL-Gaussian, the same histogram with a truncated normal of variance 0.5 as the target, returned the score to 0.4151. A flexible distribution on this trunk is available. The training signal and the features in front of it did not produce the classifier's mean.

The cumulative head is a further restriction inside that negative result, not the whole explanation. It gained 0.0090 over the regression TabM and then lost 0.0183 when the restriction was removed. Proportional odds is a real limit of the 0.414 model: one latent cannot move the 5-versus-6 cut separately from the 6-versus-7 cut. Removing the limit without changing the trunk made the expected label worse. The binding constraint on this split is the function of the eleven chemicals, not the last linear map.

The penalty and the step size are poor candidates for the same reason. XGBoost's fitted L2 is 1, its subsample and column sample are 1, and it does not early-stop. Putting weight decay 1.0 on the ordinal TabM widened the gap. A single learning rate taken from the batch curvature fell to 0.2807. Those runs changed the global optimizer. They did not build thresholds.

The prior and the binned embeddings are the closest attempt to a tree-like input, and they are not a tree. Eight quantile bins of three chemicals became embedding tables inside the same MLP. The expected label fell to 0.4029, and training to 100 epochs did not move the scored checkpoint off epoch 14. A smooth map on top of fixed bins is a different model from a split that chooses the threshold and stores a class histogram on each side.

The width-32 bottleneck is the only gain that moved the plain trunk, and it is 0.0046. A narrower vector in front of the cumulative head does not account for 0.0760.

This is one split. The houses, cpu_act, and sulfur results say the same TabM recipe is not behind default XGBoost on those regression tasks. The wine judgment does not travel with them.

## Pieces to borrow

Three pieces, in the order the scores support. They are not implemented.

1. **A leaf histogram on an axis-aligned partition, scored by its mean.** Depth 6, 100 additive stages, shrinkage 0.3, one probability per quality, prediction `probabilities @ quality`. That is the model that scored 0.490. Ashlar already multiplies class probabilities by the class values in `predict_score`. The missing object is the probability, not the multiply. The probabilities that produced 0.490 sit in leaves of threshold rules on the original chemicals. Quantile scaling is already on the 0.414 run. Embedding eight bins of three chemicals is the variant that was tried, and it scored 0.403.

2. **The multiclass gradient and hessian as the rule that places the thresholds.** The same tree geometry, grown for squared error, scores 0.4285. The 0.0615 is that choice of loss. A global AdamW rate, including the surface-rate rule that picked 0.1, is not a substitute: the surface run scored 0.2807. The borrow is per-split, from the sum of the multiclass gradients and hessians in a region. It is not one scalar for the whole network.

3. **Shrinkage across those stages, with L2 inside the leaf weight.** The fitted classifier uses shrinkage 0.3 and L2 1, with subsample 1, column sample 1, gamma 0, and minimum child weight 1. In XGBoost that L2 enters the leaf weight and the split gain. AdamW weight decay 1.0 on the dense TabM is a different penalty and scored 0.3800. Borrow the stage-wise shrinkage and the leaf penalty. Leave the residual MLP's weight decay at `1e-4`.

The ordinal cumulative head, the one-hot softmax, the HL-Gaussian target, the frozen Jeffreys cuts, the late bottleneck, and a longer run of the prior model are measured, and none of them is the piece that closes 0.0760. A longer run of the untouched 0.414 trunk was not measured.

## Differentiable stages

Hard axis-aligned leaves are not used. The analogue is `soft_stages=8` on the 0.4139 cumulative call. Eight stumps read the quantile-scaled numeric columns. Each stump mixes those columns with a softmax, compares the coordinate to a threshold through a sigmoid, and mixes two class-logit leaves. The eight residuals are summed with shrinkage 0.3, added to the cumulative log-probabilities, and renormalized. Leaf logits start at zero, so the module starts as the untouched head. Feature weights and thresholds start random so the stumps are not copies. Every array in the stump is a parameter. The ordinal class loss trains them jointly with the TabM trunk. There is no greedy grow step and no detached Newton update. The residual regression default leaves `soft_stages` unset.

On the same wine split that call scores expected-label test R² 0.417837586328735 and most-likely-label test R² 0.3024068870841512. It ran all 50 epochs. The best internal validation loss was 1.0000371932983398, the ordinal negative log-likelihood of the combined probabilities. That expectation is 0.0039 above 0.41394214246508565 and 0.0721 below the `XGBClassifier` expectation 0.48996919003701056. It does not clear the 0.01 bar, so the residual regression default was not changed.
