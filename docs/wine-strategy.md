# Wine quality: histogram target

The comparison on the saved medium split is still the expected label. Cumulative ordinal TabM scores 0.41394214246508565. The width-32 late bottleneck scores 0.41855174134115225. `XGBRegressor` scores 0.4284963911647759. `XGBClassifier`, read out as an expected label, scores 0.48996919003701056. The gap this note is aimed at is 0.4139 against 0.490.

## What the recent literature says

Grinsztajn, Oyallon, and Varoquaux (NeurIPS 2022, [arXiv:2207.08815](https://arxiv.org/abs/2207.08815)) still frame the medium-data gap. On tables of about ten thousand rows, trees beat neural nets after a large hyperparameter search. Their three inductive biases are irregular target functions, uninformative features, and rotation invariance that mixes features trees would have cut one at a time. Wine quality sits in that size range, and it is numerical only, so the categorical half of the story is not the binding one.

Gorishniy, Kotelnikov, and Babenko (ICLR 2025, [arXiv:2410.24210](https://arxiv.org/abs/2410.24210)) argue that a parameter-efficient ensemble of MLPs, TabM, is the practical neural baseline, and that attention and retrieval do not pay for their cost. Holzmüller, Grinsztajn, and Steinwart ([arXiv:2407.04491](https://arxiv.org/abs/2407.04491)) find RealMLP competitive with gradient-boosted trees once both are meta-tuned, with trees still the cheaper default. Erickson, Purucker, Tschalzev, Holzmüller, Desai, Salinas, and Hutter (NeurIPS 2025, [arXiv:2506.16791](https://arxiv.org/abs/2506.16791)) find the same split in TabArena: trees remain strong at a practical budget, and neural nets catch up when many tuned configurations are ensembled. None of those papers is a reason to replace the TabM trunk on one wine split. The trunk is already that ensemble. The runs that moved weight decay, the learning rate, embeddings, patience, and a late bottleneck left the target alone.

The ordinal literature is the closer match. McCullagh (Journal of the Royal Statistical Society B, 1980) is the proportional-odds model already in the cumulative head: one latent, cuts that do not depend on the row. Beckham and Pal (ICML 2017, [PMLR 70](https://proceedings.mlr.press/v70/beckham17a.html)) show why a one-hot cross-entropy on ordered classes is a bad training signal even when the log-loss of the true class is fixed. Any leftover probability can sit on a distant class and the loss does not move. Their Figure 1 is the softmax result already recorded in `wine-architecture.md`: validation negative log-likelihood improved, from 1.1636 to 1.0034, and the expected label fell from 0.4139 to 0.3957. They also name the readout this package already uses, the expectation of the class scores. Hou, Yu, and Samaras ([arXiv:1611.05916](https://arxiv.org/abs/1611.05916)) make a related point with squared earth mover's distance: moving mass to a far class should cost more than moving it to a neighbor. Okuno and Harada (Journal of Computational and Graphical Statistics, 2024, [arXiv:2303.17823](https://arxiv.org/abs/2303.17823)) relax proportional odds with a neural net. A fully non-proportional cumulative model can represent the same probability vectors as a softmax, so that relaxation repeats the parameterization the softmax run already had.

Imani and White (ICML 2018, [PMLR 80:2157–2166](https://proceedings.mlr.press/v80/imani18a.html)) train a histogram over bins by cross-entropy to a target distribution chosen in advance, then predict the histogram's mean. Their main loss, HL-Gaussian, uses a truncated Gaussian centered on the observed target. In the reported table the Gaussian variance is the radius of a bin. The one-bin special case, all of the target mass on the bin that contains the label, is ordinary one-hot cross-entropy. They find that one-bin histogram loss can lose to plain regression, while HL-Gaussian improves the mean prediction, and that the gain is in the gradient rather than in a tuned variance. The failed softmax run is that one-bin loss. The `XGBClassifier` expectation is the other half of the same idea: a flexible distribution over the integer scores, scored by its mean. On this split that mean is 0.490 and the most likely label is only 0.354, so the probability mass away from the mode is carrying the metric.

## What this run changes

Keep the 0.4139 TabM call and change only the training target. The head is the softmax histogram, one bin per quality value 3 through 9. Those seven bins have width 1 on the label, from 2.5 to 9.5. The target for a row is a Gaussian centered on the observed quality, truncated to that interval, with variance 0.5, the bin radius from Imani and White. Training minimizes the cross-entropy from that soft target to the member log-probabilities. Early stopping uses the same loss. `predict_score` is still the expected label.

```python
tabm(
    task="ordinal",
    head="softmax",
    prediction_policy="mean_probabilities",
    numeric_scaling="quantile_tabr",
    max_epochs=50,
    random_state=0,
    histogram_variance=0.5,
)
```

`histogram_variance=None` stays the package default, including `tabular()` and `ordinal()`. Setting it on a task other than ordinal, or on the cumulative head, is an error. There is no second fit and no search over the variance. Weight decay stays 0, the learning rate stays `2e-3`, and the bottleneck, the class prior, and the binned embeddings stay off.

The hypothesis fails if the expected-label test R² does not exceed 0.41394214246508565 by more than 0.01. The residual-MLP regression default changes only in that case. The comparison that matters either way is the gap to 0.48996919003701056.

## Result

The fit returned expected-label test R² **0.4150779471645716**. The most likely label scored 0.30532568671978233. Training stopped at epoch 47 under patience 16. The best internal validation loss was 1.4194178581237793, on the histogram cross-entropy rather than the one-hot negative log-likelihood.

0.4151 is 0.0011 above 0.41394214246508565. It does not exceed that score by 0.01. The soft Gaussian target did lift the softmax head from its one-hot score of 0.3956767364524363 back to the cumulative-logit score, and it left the gap to the `XGBClassifier` expectation 0.48996919003701056 at 0.0749. The residual-MLP regression default stayed the regressor.
