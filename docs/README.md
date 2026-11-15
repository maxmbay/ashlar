# Ashlar documentation

[Ashlar](../README.md) is a Python library for regression and classification on tabular data, built on JAX. The README has the overview, the dressed model, and the quick start. These pages have the detail.

## Install

```bash
pip install "ashlar-tabular[dataframes]"
```

The package is imported as `ashlar` and needs Python 3.11 or later. This installs the CPU version of JAX, which is all Ashlar needs; every Ashlar result in the README was computed on CPUs.

To train on an NVIDIA GPU on Linux, also install JAX's CUDA build. It includes the CUDA libraries, so only the NVIDIA driver needs to be installed:

```bash
pip install "ashlar-tabular[dataframes]" "jax[cuda12]"
python -c "import jax; print(jax.devices())"   # should list a CudaDevice
```

No code changes are needed, because JAX uses the GPU when one is available. For other CUDA versions and platforms, see [JAX's install guide](https://docs.jax.dev/en/latest/installation.html).

## Start here

- [estimator.md](estimator.md): what the estimators take, what they return and in what shape, which scikit-learn tools work, the calls that change for someone coming from another library, saving and loading.
- [heads.md](heads.md): which head suits which target, calibrated intervals, and uncertainty across ensemble members.
- [dressed.md](dressed.md): the dressed model for small tables, step by step, with its settings, its limits, and every test.

## Guides, by what your data looks like

- [observed.md](observed.md): labels that were not seen exactly: censored, survival, interval, rounded, truncated, noisy, and totals over several rows, and which heads take them.
- [distributions.md](distributions.md#several-targets-at-once): several targets with one joint distribution, and outcomes of different kinds chained.
- [effects.md](effects.md): random effects for a grouping column with few rows per group, and predictions for new groups.
- [additive.md](additive.md): additive models of every parameter of the distribution, with readable terms.
- [forecast.md](forecast.md): forecasting one series, or many series with covariates and joint paths (work in progress).

## Reference

- [models.md](models.md): the five choices that define a model, a table of what the general estimator covers with the setting and the guide for each, and the research models beyond the defaults.
- [tabular.md](tabular.md): defaults, recipes, tasks, embeddings, set-valued columns, and hyperparameter search.
- [distributions.md](distributions.md): the distribution object, JAX transforms, monotone features, missing inputs, decisions, and explanations.

## Evidence

- [showcase.md](showcase.md): the large-table benchmarks with their figures, and the examples from the README's gallery, with code, results, and the cases where another tool does better.
- [benchmarks.md](benchmarks.md): every benchmark result, per table and metric, with the baselines, how each was run, and the cases where the baselines do better, including [a comparison with other neural packages](benchmarks.md#beside-other-neural-packages).
- [`examples/`](../examples/): runnable scripts, including the benchmarks, and [`examples/readme/`](../examples/readme/), which redraws the figures in the README and these pages from saved results.
- [CHANGELOG.md](../CHANGELOG.md): what changed in each release.
