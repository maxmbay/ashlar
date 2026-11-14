# Ashlar documentation

[Ashlar](../README.md) is a Python library for regression and classification on tabular data, built on JAX. The README has the overview, the dressed model, and the quick start. These pages have the detail.

## Start here

- [estimator.md](estimator.md): what the estimators take, what they return and in what shape, which scikit-learn tools work, saving and loading.
- [heads.md](heads.md): which head suits which target, calibrated intervals, and uncertainty across ensemble members.
- [dressed.md](dressed.md): the dressed model for small tables, step by step, with its settings, its limits, and every test.

## Guides, by what your data looks like

- [observed.md](observed.md): labels that were not seen exactly: censored, survival, interval, rounded, truncated, noisy, and totals over several rows.
- [distributions.md](distributions.md#several-targets-at-once): several targets with one joint distribution, and outcomes of different kinds chained.
- [effects.md](effects.md): random effects for a grouping column with few rows per group, and predictions for new groups.
- [additive.md](additive.md): additive models of every parameter of the distribution, with readable terms.
- [forecast.md](forecast.md): forecasting one series, or many series with covariates and joint paths (work in progress).

## Reference

- [models.md](models.md): the five choices that define a model, and the research models beyond the defaults.
- [tabular.md](tabular.md): defaults, recipes, tasks, embeddings, set-valued columns, and hyperparameter search.
- [distributions.md](distributions.md): the distribution object, JAX transforms, monotone features, missing inputs, decisions, and explanations.

## Evidence

- [showcase.md](showcase.md): the examples from the README's gallery, with code, results, and the cases where another tool does better.
- [benchmarks.md](benchmarks.md): every benchmark result, per table and metric, with the baselines and how each was run.
- [CHANGELOG.md](../CHANGELOG.md): what changed in each release.
