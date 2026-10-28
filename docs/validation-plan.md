# Validation plan

Parity here means "this implementation does what the reference implementation does on a named check." It does not mean a neural net should beat a tree, and it does not mean a California Housing score for one architecture is a target for another.

## Reference models

| Reference | Pin used on 26 Oct 2026 | Role |
|---|---|---|
| [yandex-research/tabm](https://github.com/yandex-research/tabm) `tabm` 0.0.3, commit `28e47ae301c92ec37787dde1ce923a0793f405b4` | Installed in the DeepTab environment and imported | Equation and initialization of `LinearBatchEnsemble`. `TabM.make` defaults recorded in `TABM_PAPER` |
| [rtdl-num-embeddings](https://github.com/yandex-research/rtdl-num-embeddings) `LinearReLUEmbeddings` on `main` | Source read, not executed | Formula for the one embedding this package implements: `relu(bias + weight * x[..., None])`, uniform on `± d_embedding**-0.5` |
| [PyTabKit](https://github.com/dholzmueller/pytabkit) 1.7.3 | Wheel inspected. A capped CPU fit is in `benchmark.md` | Training defaults copied into `TABM_PAPER`: learning rate `2e-3`, weight decay `0`, batch size `256`, patience `16`. The benchmark fit is not that budget |
| DeepTab 2.1.0 `TabMConfig` | Source of the installed wheel | A different recipe (`model_type="mini"`, `scaling_init="ones"`, dropout `0.5`, `layer_sizes=[256, 256, 128]`). Not a parity target |

`TABM_PAPER` records those architecture and training defaults. It does not claim a published dataset score. Quantile preprocessing, piecewise-linear embeddings, and gradient clipping from the paper experiments are omitted on purpose.

## What was checked

| Check | Result | Grade |
|---|---|---|
| Official `LinearBatchEnsemble` equation, weights copied into NumPy float32 | Max abs diff versus torch `0.0` on the probed `(4, 3, 5) -> (4, 3, 7)` input | Confirmed |
| Same weights copied into `ashlar.layers.LinearBatchEnsemble` | Max abs diff versus torch `1.192e-7` on CPU. That is one float32 unit in the last place of 1, from the JAX matmul, not from a different formula | Confirmed |
| Official module: bias rows share one initialization draw | The probed bias rows matched | Confirmed |
| Chunked first scaling | Covered by `tests/test_semantics.py` on the JAX layer: one draw is shared inside a chunk | Confirmed for this package. Not a weight-copy against `tabm` |
| Mean of member losses, not loss of the mean | Implemented in `objectives.py` and covered by the reduction tests | Confirmed as this package's contract. Not compared to a `tabm` training step |
| Mixture variance versus parameter-average variance | Constructed `N(-5, 1)` and `N(5, 1)`: mixture variance `26`, mean-parameter variance `1` | Confirmed |
| Direct weight reduction | `sum(w*loss)/sum(w)` unit test, including a zero weight | Confirmed |
| Preprocess fit on the training partition only | Temporal and group split tests | Confirmed on synthetic rows |
| Gradient through median impute and standardization | `tests/test_estimator.py` and `examples/jax_composition.py` (JAX grad `0.129365`, central difference `0.129368`) | Confirmed |

## What was not checked

- A full `TabM.make` forward, including the output `LinearEnsemble`, against `tabm` 0.0.3 with copied weights.
- Bit-match or gradient-match of `LinearReLUEmbeddings` against `rtdl_num_embeddings`.
- Score parity on Grinsztajn, TabReD, TabArena, California Housing, or Higgs. `benchmark.md` records one CPU seed on two Grinsztajn medium splits. That file is not a parity band.
- Several random seeds of a real dataset. Tolerance for a future score comparison has to come from the spread of the reference under its own seeds. No margin is invented here.
- Higgs. The transformed classification table is OpenML 44129 (940160 rows) and was not downloaded.
- GPU timing. Every probe in `probe-results.md` used a 4-CPU machine and `JAX_PLATFORMS=cpu`. JAX was not faster or slower than PyTorch in any way that should be quoted: the runs are one-epoch or eight-epoch toys, and the clocks include compilation.

## Later data, with the leakage constraint

Use published benchmarks to test the port, in this order:

1. **Fixed-weight forward and gradient** of one `TabM.make` module without embeddings, then with `LinearReLUEmbeddings`. Same weights, same input, CPU float32, report max abs diff. The layer check above is the pattern.
2. **Loss reduction and inference aggregation** on a frozen minibatch: member-mean loss, `mean_probabilities` versus `mean_logits`, mixture versus parameter average. Compare to the reference functions, not to a leaderboard.
3. **Grinsztajn tabular benchmark** only with that paper's transformed matrices, row budgets, splits, and metrics. A score on a re-split of the raw tables is a different experiment.
4. **TabReD** after the data release and the official splits are actually in hand. The [TabReD audit](https://arxiv.org/html/2406.19380v4) documents leakage in older benchmarks. A win on an affected split is not evidence of better generalization, and this plan does not treat it as one.
5. **California Housing and Higgs Small** as separate reproduction checks against the named model that published the result. TabR's score is not TabM's target. This package does not have to win both.
6. **TabArena** as an external evaluation route after the recipe is stable, not as a gate for the first release.

## Cost accounting for the probes already run

Hardware: Linux x86_64, 4 CPUs, JAX `CpuDevice(id=0)`, `torch.cuda` false. No GPU was assumed or used.

| Probe | What the clock includes | Observed |
|---|---|---|
| DeepTab classifier, regressor, LSS, custom regressor | Lightning fit of 1 epoch on 40 or 80 rows, plus process startup already paid | about `0.03` to `0.08` seconds of fit. Default MLP reported 38.1K parameters on two features because the default width is large |
| Ashlar ordinary, embeddings, search, transformer, advanced, and composition examples | `fit` of a few epochs on toy rows, including JAX compilation | fit about `1.9` to `3.3` seconds; the transformer example was the slow one, at `3.28` seconds |
| pytest, 19 tests | imports, compilation, synthetic fits | the whole file finished inside the same command as the three examples, 22.7 seconds wall clock for versions plus pytest plus three examples |

These numbers are cold-start costs of tiny shapes. They are not a training-speed comparison. A later GPU note should separate cold start, compilation, warm step time, host-to-device transfer, preprocessing, and the tuning budget, and it should block on JAX async dispatch before reading the clock. No accelerator command is required for the checks in this repository today.

## Tolerance policy

- Layer and embedding parity: report max absolute difference on a named input and dtype. The current JAX-versus-torch gap is `1.192e-7` and is acceptable for the equation check. It is not a license to ignore a larger gap in a full-network forward.
- Score parity, when a dataset run exists: set the band from the reference's own seed-to-seed variation on that split. Do not pick a percentage in advance.
- Weight and mixture semantics: exact on constructed tensors (`26` versus `1`, the weighted-mean fixture). Those do not need a tolerance.
