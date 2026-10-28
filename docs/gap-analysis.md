# Gap analysis

What would have to change for Ashlar's default fit to approach XGBoost, PyTabKit RealMLP-TD, and PyTabKit TabM-D on the two splits in `benchmark.md`. The ranking below was written before the houses probes. Those probes are summarized at the end. Nothing was registered or published.

The default at the time of this ranking was `DEFAULT_RECIPE`: residual MLP (`n_blocks=2`, `d_block=64`, hidden dropout 0, ReLU, one member), median impute then mean and population standard deviation, no numeric embeddings, batch 64, learning rate `1e-3`, constant AdamW, weight decay `1e-4`, feature dropout `0.1`, `validation_size=0.2`, patience 10, `max_epochs=50`. The houses sweep later set feature dropout to `0` and the residual activation to SiLU. `ashlar.classifier()` is that recipe with a classification head. `fit(..., search=True)` compares weight decay `1e-4` and `1e-3` for two epochs and does not run unless asked. `tabm()` is a separate recipe (`k=32`, `d_block=512`, learning rate `2e-3`, weight decay 0, batch 256, patience 16, `max_epochs=200`) and was not the model in the table. That recipe still omits quantile preprocessing and piecewise-linear embeddings.

## What the one seed shows

Both tables are one fold, `numpy.random.RandomState(0)`, medium Grinsztajn split (10,000 train rows, then 30/70 of the remainder). Eight numeric columns. Test metrics are R² and accuracy. The paper validation partition was not used to train or to pick a model.

| | Ashlar regularized default | Ashlar search | XGBoost defaults | RealMLP-TD, 50-epoch cap | TabM-D, 50-epoch cap |
|---|---:|---:|---:|---:|---:|
| houses test R² | 0.7322 | 0.7299 | 0.8268 | 0.8247 | 0.8366 |
| california test accuracy | 0.8657 | 0.8657 | 0.8968 | 0.8853 | 0.8916 |

Gaps from the regularized default: houses R² is 0.095 behind XGBoost, 0.093 behind RealMLP-TD, and 0.104 behind TabM-D. California accuracy is 0.031, 0.020, and 0.026 behind those three.

The same split was fit once before weight decay and feature dropout were turned on. That run scored houses R² 0.7428, and a search that selected learning rate `3e-3` scored 0.7657. California accuracy was 0.8704 for both. Relative to that earlier default, the regularized default lost about 0.011 R² and 0.005 accuracy. The search that then tried weight decay `1e-3`, ranked on a two-epoch validation loss, scored 0.7299. The two-epoch loss preferred the stronger penalty; the full-train test score did not.

Houses used all 50 epochs, so patience 10 did not stop that run. California stopped at epoch 30. XGBoost fit all 10,000 train rows with 100 trees and no early stopping, in about 0.1 s. RealMLP-TD reached 0.8247 R² in 8.3 s inside the same 50-epoch cap. TabM-D reached 0.8366 in 248 s, also capped at 50. DeepTab's wider MLP (`[256, 128, 32]`, learning rate `1e-4`, weight decay `1e-6`, 50 epochs) scored houses R² 0.1832 and california accuracy 0.8478.

From the houses RMSE and R², the test-target variance is about `0.2962² / (1 - 0.7322) ≈ 0.33`. The scored target is already near unit scale. The regression feature file still has the raw California Housing columns (income, age, room counts, population, households, lat/lon). The classification file uses the ratio features (`MedInc`, `AveRooms`, `AveOccup`, and so on). Those are different matrices.

PyTabKit 1.7.3 defaults, read from the installed `default_params.py`, not from this benchmark's logs: RealMLP-TD regression uses hidden size `[256, 256, 256]`, Mish, dropout 0.15, weight decay `0.02`, learning rate `0.2`, a cosine-log schedule, Adam, robust scaling plus smooth clip, PBLD numeric embeddings, and `normalize_output=True`. Its own epoch default is 256. TabM-D uses `k=32`, `d_block=512`, learning rate `2e-3`, weight decay 0, dropout 0.1, batch 256, patience 16, feature transform `quantile_tabr`, and `num_emb_type='none'`. `TABM_D_REG` does not set `transformed_target`. Both libraries were stopped at 50 epochs in the benchmark, before those library caps.

## What that does not show

One seed has no error bar. A 0.01 R² move is the size of the regularization change; it is not a stable ranking. The 0.09 R² gap is larger than any within-Ashlar change on this split.

RealMLP-TD matching XGBoost does not isolate which of its many differences mattered. TabM-D being the highest houses score does not show that Ashlar must become a 512-wide, 32-member network. DeepTab's poor R² does not show that a wider MLP cannot work; that fit used a different learning rate and no robust or quantile feature map.

The benchmark did not log the per-epoch validation curve, so "still descending at epoch 50" is not established. It only establishes that the stopping rule did not fire on houses and did fire on california.

## Hypotheses

**Regularization.** Fact: weight decay `1e-4` plus feature dropout `0.1` lowered both test scores on this split, and a two-epoch search then picked a stronger penalty that was worse after the full train. Hypothesis: feature dropout `0.1` is a large column drop when there are only eight columns (about 0.8 columns missing on a typical step), and the two-epoch ranking is too short to choose a penalty. Hypothesis, already in tension with the RealMLP result: regularization in general is the reason neural nets trail trees here. RealMLP-TD uses a larger dropout and a larger weight decay and still matches XGBoost, under a different parameterization and a learning rate of `0.2`. Removing Ashlar's two new penalties should recover the earlier 0.7428 / 0.8704. It should not, by itself, produce 0.83 R².

**Training budget and early stopping.** Fact: the houses run hit the epoch cap; the california run did not. RealMLP-TD and TabM-D also stopped at 50 epochs and still led. Hypothesis: the houses residual net was still improving at epoch 50, so a longer cap would close part of the gap. That can be true and still leave most of the 0.09 unexplained, because a different MLP already finished the job inside 50 epochs. Hypothesis: holding out 20% of the 10,000 rows (about 2,000 rows) is the gap. Unmeasured. Eight thousand rows remain for eight features, so this is a weak candidate for a 0.09 R² difference.

**Learning-rate schedule.** Fact: on the unregularized recipe, a two-epoch search chose learning rate `3e-3` over `1e-3` and the full train gained about 0.023 R² (0.7657 versus 0.7428). California accuracy did not change. Ashlar's optimizer is constant AdamW. There is no decay, warmup, or cosine. Hypothesis: a modestly higher constant rate, or a schedule, is worth another couple of points of R² and will not close the rest. RealMLP's learning rate of `0.2` is not a drop-in replacement; it sits on NTK-style parameterization, a front scale, and output normalization.

**Numerical embeddings and TabM fidelity.** Fact: the default path sets `num_embeddings=None`. The code can build `linear_relu` embeddings, which are not PBLD and not piecewise-linear. TabM-D's own default is `num_emb_type='none'` with `quantile_tabr` on the features. RealMLP-TD's default is PBLD embeddings on top of robust scaling. `tabm()` was not scored. Hypothesis: turning on Ashlar's existing linear-ReLU embedding is a cheap probe and a weak stand-in for PBLD. Hypothesis: copying TabM-D's width and member count without `quantile_tabr` will not reproduce 0.8366. The recipe comment already records that quantile preprocessing and piecewise-linear embeddings were left out.

**Target transforms.** Fact: every row in the table was scored on the same CSV target. That target's test variance is about 0.33, so a second standardization is a small rescaling, not a change of units from dollars. TabM-D does not enable `transformed_target`. RealMLP-TD does enable `normalize_output`. Hypothesis: another target transform is a minor optimization detail on this file. It is not the main houses gap. It would matter more on a raw dollar target, which this benchmark is not.

**Default backbone.** Fact: the scored Ashlar model is a width-64 residual MLP with one member. A different MLP recipe (RealMLP-TD) matched XGBoost. A third MLP recipe (DeepTab, learning rate `1e-4`) did not. Hypothesis: the default class is the wrong inductive bias for these two datasets. The RealMLP result makes that unlikely as the primary cause. Hypothesis that survives: the default is the wrong *representation* of the eight houses columns. Mean and standard deviation give unit variance while leaving a heavy tail in charge of the scale. Trees ignore that scaling. RealMLP-TD and TabM-D both replace it, with robust scaling or a quantile map. That is the best account of why a 50-epoch MLP can sit at 0.82 while this residual MLP sits at 0.73, and why the classification gap on the already-ratio features is smaller.

## Ranked by expected gain versus cost

Expected gain is movement toward the XGBoost / RealMLP-TD / TabM-D numbers on these two splits, not a new benchmark. Cost is code, CPU time, and how many other knobs the change would confound.

1. **Train-only robust scaling or a quantile map instead of mean/std, same residual MLP, same 50 epochs.** Highest expected gain on houses, because that is where the features are raw counts and where almost the whole gap to RealMLP-TD sits. California should move less if the hypothesis is right. Cost is a preprocessor and a decision about whether that map stays outside the gradient (quantile) or stays differentiable (the current z-score is inside an Equinox module). This is the change that can falsify the top hypothesis in one run.
2. **Drop the new penalties back to zero** (`weight_decay=0`, `feature_dropout=0`). Gain is the 0.011 R² and 0.005 accuracy already measured. Cost is a default change and a doc edit. Worth doing so the default is not the worse of two measured recipes. It does not close the gap.
3. **A higher constant learning rate, then a cosine schedule only if the constant rate still helps.** The `3e-3` run is the largest helpful change measured so far (+0.023 R²) and it was cheap. Further schedule work is easy to confound with parameterization. Do it after the feature map, on whichever map won, or the comparison answers the wrong question.
4. **`num_embeddings='linear_relu'` on the default numeric columns.** The flag exists. Gain is unknown and capped: it is not PBLD. A failure does not clear PBLD; a gain says a learned per-feature map helps on top of whatever scaling is in front of it.
5. **Train on all 10,000 rows, or refit on the internal validation rows after the best epoch.** Measures the holdout. Low expected gain. Early stopping goes away if `validation_size=0`, so this needs a stated substitute (fixed epoch count, or a refit).
6. **Raise `max_epochs` only for the houses setup, where the cap was binding.** California already stopped. RealMLP-TD's lead inside 50 epochs caps how much this can explain. Record the validation curve first; a flat curve makes the longer run unnecessary.
7. **A small TabM as the default backbone** (`k` well below 32 or `d_block` well below 512), still without claiming `tabm()` parity. CPU cost is the reason it is this far down: TabM-D at full width took about 11 times the regularized Ashlar fit. Expected gain is lower than the feature-map change because an MLP recipe already matched the tree.
8. **A second target transform.** Lowest expected gain on this file. The scored target is already near unit variance, and TabM-D does not transform it again.

Piecewise-linear or PBLD embeddings are the likely follow-up if rank 1 helps but leaves a gap to RealMLP-TD. They are new code, not a flag, so they wait on that result.

## The experiment that would falsify the top hypothesis

Top hypothesis: the houses gap is mostly the default numeric map. Z-scoring raw counts leaves the residual MLP at 0.73 R², while a monotone train-only robust or quantile map is what lets a plain network reach the low 0.80s on this split inside 50 epochs.

Run both datasets on the same split and the same regularized residual MLP (`max_epochs=50`, patience 10, learning rate `1e-3`, weight decay `1e-4`, feature dropout `0.1`, batch 64, `random_state=0`). Change only the numeric transform: fit it on the training partition, and replace mean/std with either a quantile map to a normal margin or a median/IQR scale plus a clip. Do not change the backbone, the epoch cap, the penalties, or the learning rate in this run.

The hypothesis is false on this seed if houses test R² stays within 0.01 of 0.7322. That is the size of the regularization effect already measured; a smaller move means this transform did not account for the gap to 0.8268. The hypothesis is supported if houses test R² moves a large part of the way from 0.7322 toward 0.8268, with the architecture and the cap held fixed. California accuracy is the check that the same map is harmless on the ratio features. It is not a second hypothesis.

Log the best and last internal validation scores in that run. If houses is still improving at epoch 50 after the new map, budget becomes the next hypothesis. If the curve is flat and the test R² is still near 0.73, both the feature-map hypothesis and the "just train longer" hypothesis are weak, and the next candidate is the learning rate on top of the old z-score.

## Result of that run

The numeric change was a train-only normal quantile map, not a median/IQR scale. scikit-learn `QuantileTransformer`, `n_quantiles=1000`, `output_distribution="normal"`, every row of Ashlar's internal training partition, `random_state=0`, fit after median imputation, replacing mean/std. Same regularized residual MLP as the 0.7322 row: ReLU, feature dropout `0.1`, weight decay `1e-4`, learning rate `1e-3`, 50-epoch cap, patience 10.

Houses test R² is 0.6518968125086434. The hypothesis failed. The score left the band around 0.7322, but downward by 0.080, not toward 0.8268. Training stopped at epoch 24. Best internal validation loss was 0.11655575782060623 and the last was 0.16528528928756714, so the cap was not what stopped it. A later sweep, in `benchmark.md`, held the z-score fixed and varied only feature dropout and activation. Dropout 0 with SiLU scored 0.7615065490594078 and dropout 0 with GELU scored 0.7602485207169098. Both beat 0.7322 by more than 0.01. The no-argument path is now feature dropout `0` and SiLU. That still leaves the houses gap to 0.8268.
