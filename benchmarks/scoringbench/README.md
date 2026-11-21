# Ashlar on ScoringBench

This folder reproduces the ScoringBench numbers in Ashlar's [README](../../README.md) and in the
[guide to the dressed model](../../docs/dressed.md), and the analysis behind them, from the rankings of 21 and of 64
models to the tests on the 23 development tables.

## What is here

- `analyze.py` runs the scripts in `analysis/` in order and prints Markdown. `RESULTS.md` is its output and starts
  with the numbers quoted in the README and the guide. Tables and numbers go to `out/` (`numbers.json` and one file
  per table), figures to `out/figures/`.
- `analysis/common.py` ranks by ScoringBench's rules, with the autorank package as its `autorank_leaderboard.py`
  uses it. Lines taken from ScoringBench (Apache-2.0) are marked with their source.
- `results/` holds our per-fold results, one file per Ashlar entry, with every column as ScoringBench wrote it except
  `model`, which holds the file's name.
- `results/model_families.json` is our grouping of the 63 published models, with the lines of ScoringBench's code
  that each call rests on.
- `published_files.json` has the SHA-256 sum of every ScoringBench file the scripts read; they stop if one differs.
- `ashlar_dressed_wrapper.py` runs `ashlar.dressed()` in ScoringBench's current code. It reads each row's quantiles
  at the same 203 levels as the run did.

## Reproduce

This folder does not copy ScoringBench's files. With Git LFS installed:

```bash
git clone https://github.com/jonaslandsgesell/ScoringBenchOutput.git
git -C ScoringBenchOutput checkout b590a95
git -C ScoringBenchOutput lfs pull
git clone https://github.com/jonaslandsgesell/ScoringBench.git
git -C ScoringBench checkout 3bf48c9
python analyze.py --published ScoringBenchOutput --code ScoringBench | diff - RESULTS.md
```

The scripts need numpy, pandas, pyarrow, scipy, autorank, mpmath and matplotlib; we used Python 3.12, pandas 3.1,
scipy 1.18 and autorank 1.3. `--code` is optional. Without it the model families have no line numbers or
organizations, and the loader check skips the part that runs ScoringBench's own loader, which also needs
ScoringBench's requirements (torch among them). We ran the scripts on copies of both repositories laid out as the
clones are, and have not run the clone commands.

## What was run

- `ashlar.dressed()` with its default settings, from Ashlar commit 95363c9, which is the model code released as
  0.0.7 and 0.1.0.
- ScoringBench's code at commit 3bf48c9, ranked against the published results of ScoringBenchOutput b590a95
  (7 November 2026).
- All 99 tables and 495 folds, on the CPU of a 12-core Apple M3 Pro laptop with six fits at a time: 448 hours of
  fitting and predicting, summed over the folds. The run was paused twice, and the folds in progress then count
  the pause in their times.

The rank comes from our own run. ScoringBench's maintainer has since run the model and listed it on the [leaderboard](https://scoringbench.com) on November 19, 2026.

Of the five Ashlar entries in the table of what each part adds, the second and third come from the same fits, and
so do the fourth and fifth. The first three ran from research code that predates the library's dressed model, so
only their results are here. The first run also scored a mixture head (`mixture_head_100_epochs.parquet`).

## Steps that are not here

These inputs come from steps that need every table downloaded or our research code. We ship what they wrote.

- `results/target_values.csv`: rows, columns and distinct target values per fold, counted with ScoringBench's loader
  and splits, without fitting. The 12 tables with at most 20 target values come from it.
- `results/loading_fix_tables.csv`: under pandas 3, twelve of ScoringBench's tables could not be read and went
  through a local fix that reads their categorical columns as pandas 2 did. The list comes from the times
  ScoringBench cached them. Eleven are among the 99.
- `results/five_pairs_pooled_rescored_7f36e0b.parquet` and `results/five_pairs_pooled_rescored_9406d60.parquet`: the
  undressed pool's saved predictions, scored again by ScoringBench's code of 13 November (commit 7f36e0b) and of
  14 November (commit 9406d60). The run did not save the dressed model's predictions.
- `results/development/`: the results on the 23 development tables, written by the research code, which is not in
  this repository. Field names are plain; values are unchanged.

## Since the run

ScoringBench has changed its scoring code since the run. With the code of 13 November the undressed pool's CRPS is the same on every
rescored fold. A change dated 14 November spreads the probability beyond a model's outermost quantiles out to the ends of the
training labels' range, and with it the pool's CRPS moves by less than 0.16% on a fold and 0.13% on a table. The other models'
published results come from the earlier code, so the ranks here hold for the results of 7 November and may shift when ScoringBench
regenerates them.

## Credit

The tables, folds, scoring and published results come from
[ScoringBench](https://github.com/jonaslandsgesell/ScoringBench) ([arXiv:2603.29928](https://arxiv.org/abs/2603.29928)).
