"""Shared loading, aggregation and ranking for the scripts in this folder.

Inputs. Our own results are in ``results/``. ScoringBench's are read from the reader's clones, set by ``analyze.py``
(or by the environment variables below when a script is run on its own), and each file is checked against its
SHA-256 sum in ``published_files.json`` before it is used:

- ``SCORINGBENCH_OUTPUT``: a clone of ScoringBenchOutput at commit b590a95, with its Git LFS files (the published
  per-fold results and the published CRPS leaderboard);
- ``SCORINGBENCH_CODE``: a clone of ScoringBench's code at commit 3bf48c9 (its list of models with organizations and
  links, the model registry and wrappers that the family classification cites, and the code the loader check
  imports). Optional; without it the family table has no evidence lines or organizations, and the loader check
  skips its first part.

Outputs go to ``out/`` (or ``SCORINGBENCH_ANALYSIS_OUT``).

Ranking follows ScoringBench's ``autorank_leaderboard.py`` at commit 3bf48c9 (Apache-2.0): a model's value on a table
is the mean over the folds it has; density scores and the PIT test leave out tables flagged ``discrete_y``; a model
with a score on fewer than 90% of the tables is dropped, then every table a kept model lacks; coverage is scored as
|coverage - nominal| per fold; ``rank_with_autorank`` calls the autorank package. The few lines needed are copied
below, each marked with its source.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from autorank import autorank  # noqa: E402

FOLDER = Path(__file__).resolve().parents[1]
RESULTS = FOLDER / "results"
DEVELOPMENT = RESULTS / "development"
OUT = Path(os.environ.get("SCORINGBENCH_ANALYSIS_OUT", FOLDER / "out"))
FIGURES = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
MANIFEST = json.loads((FOLDER / "published_files.json").read_text())
OUTPUT_FOLDER = "output_univariate_scoringbench_d1_n3000"

DRESSED = "dressed"
POOL = "pool"

# The 20 published models that are untuned and trained from scratch (results/model_families.json).
UNTUNED = [
    "catboost_quantile", "cde_MDN", "cde_NF", "condensite_pitbst_v0_1_1", "crepes_catboost_difficulty",
    "crepes_catboost_difficulty_mondrian", "crepes_xgb_difficulty", "crepes_xgb_difficulty_mondrian",
    "flexcode_randomforest", "flexcode_xgboost", "forest_diffusion_flow", "nflows_rqs", "ngboost_gaussian",
    "pymc_bart", "pytabkit_realmlp_td", "surjectors_maf", "tabm_d", "xgb_vector", "xgb_vector_quantile",
    "xgblss_Gaussian",
]
TABM_TUNED = "tabm_hpo_cv_8_tabarena"
REALMLP_TUNED = "pytabkit_realmlp_hpo_cv_8_new"
LEAKY_TABLE = "Brazilian_houses"  # dropped by ScoringBench after the run (target leakage)
ALPHA = 0.05

NON_SCORES = ("dataset", "model", "discrete_y", "error", "error_type")
TIMES = ("fit_time", "predict_time", "train_time")
HIGHER_IS_BETTER = ("r2", "dispersion", "pit_ks_pvalue")  # autorank_leaderboard.py, main()

# scoringbench/univariate/metrics.py at 3bf48c9: DENSITY_BASED_KEYS and PIT_KEYS, the scores that ScoringBench
# leaves out on a discrete target.
DENSITY_BASED_KEYS = ("cde_loss", *[f"dpd_beta_{b}" for b in [0.01, 0.2, 0.5, 1.0]],
                      *[f"pseudospherical_alpha_{a}" for a in [1.5, 2.0, 3.0]])
PIT_KEYS = ("pit_ks_stat", "pit_ks_pvalue")
DISCRETE_NAN_KEYS = set(DENSITY_BASED_KEYS) | set(PIT_KEYS)
COVERAGE_LEVELS = (20, 40, 60, 80, 90, 95)


# ---------------------------------------------------------------- ScoringBench's files, checked


def _location(variable: str) -> Path | None:
    value = os.environ.get(variable)
    return Path(value) if value else None


def output_folder() -> Path:
    """The folder of published univariate results inside the reader's ScoringBenchOutput clone."""
    root = _location("SCORINGBENCH_OUTPUT")
    if root is None:
        raise SystemExit("Set SCORINGBENCH_OUTPUT (or run analyze.py --published PATH).")
    return root / OUTPUT_FOLDER if (root / OUTPUT_FOLDER).is_dir() else root


def code_folder() -> Path | None:
    return _location("SCORINGBENCH_CODE")


def checked(path: Path, expected: str, name: str) -> bytes:
    """A file's bytes, after checking its SHA-256 sum; stops with a plain message if it differs."""
    if not path.exists():
        raise SystemExit(f"{name} is missing ({path.name} not found).")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        hint = " It is a Git LFS pointer: run git lfs pull." if data.startswith(b"version https://git-lfs") else ""
        raise SystemExit(f"{name} differs from the file these results were computed from.{hint}")
    return data


def check_published() -> list[str]:
    """Check every published file of ScoringBenchOutput that the scripts read; return the problems found."""
    folder, problems = output_folder(), []
    for rel, digest in MANIFEST["ScoringBenchOutput"]["sha256"].items():
        try:
            checked(folder / rel.removeprefix(OUTPUT_FOLDER + "/"), digest, rel)
        except SystemExit as error:
            problems.append(str(error))
    return problems


def code_file(rel: str) -> str | None:
    """A file of ScoringBench's code at 3bf48c9 from the reader's clone, checked; None without a clone."""
    root = code_folder()
    if root is None:
        return None
    return checked(root / rel, MANIFEST["ScoringBench"]["sha256"][rel], f"ScoringBench's {rel}").decode("utf-8")


# ---------------------------------------------------------------- loading


def _numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Score columns as floats (a missing score reads as NaN, as ScoringBench's aggregation writes it)."""
    df = df.copy()
    for c in df.columns:
        if c in NON_SCORES:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    df["discrete_y"] = df["discrete_y"].astype(bool)
    df["fold"] = df["fold"].astype(int)
    df["dataset"] = df["dataset"].astype(str)
    return df


_PUBLISHED: dict[str, pd.DataFrame] | None = None


def published() -> dict[str, pd.DataFrame]:
    """The 63 published models, one frame each (one row per table and fold), each file checked."""
    global _PUBLISHED
    if _PUBLISHED is None:
        folder, frames = output_folder(), {}
        for rel, digest in sorted(MANIFEST["ScoringBenchOutput"]["sha256"].items()):
            if rel.endswith(".parquet"):
                path = folder / Path(rel).name
                checked(path, digest, rel)
                frames[path.stem] = _numeric(pd.read_parquet(path))
        _PUBLISHED = frames
    return dict(_PUBLISHED)


def ours(name: str, label: str | None = None) -> pd.DataFrame:
    """One of our result files in results/, with the model column set to ``label``."""
    df = _numeric(pd.read_parquet(RESULTS / f"{name}.parquet"))
    if label is not None:
        df["model"] = label
    return df.sort_values(["dataset", "fold"]).reset_index(drop=True)


def dressed() -> pd.DataFrame:
    return ours("dressed", DRESSED)


def pool() -> pd.DataFrame:
    return ours("five_pairs_pooled", POOL)


def target_values() -> pd.DataFrame:
    """Per fold: training and test rows, columns, and distinct target values in the training rows."""
    return pd.read_csv(RESULTS / "target_values.csv")


def leaderboard() -> dict:
    """ScoringBench's published CRPS leaderboard (figures/leaderboard/cd_data_crps.json), checked."""
    rel = f"{OUTPUT_FOLDER}/figures/leaderboard/cd_data_crps.json"
    data = checked(output_folder() / "figures" / "leaderboard" / "cd_data_crps.json",
                   MANIFEST["ScoringBenchOutput"]["sha256"][rel], rel)
    return json.loads(data)


def models_json() -> list[dict] | None:
    text = code_file("scoringbench/univariate/models.json")
    return None if text is None else json.loads(text)


def development(name: str) -> list[dict]:
    """One file of development-set results, results/development/<name>.jsonl."""
    return [json.loads(line) for line in (DEVELOPMENT / f"{name}.jsonl").read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------- ScoringBench's aggregation and ranking

# After autorank_leaderboard.py at 3bf48c9, lines 58 to 85 (its log warning left out).
def _with_amce(df):
    """Derive the per-fold mean absolute marginal coverage error (ScoringBench's ``amce``)."""
    columns = [f"coverage_{level}" for level in COVERAGE_LEVELS]
    if not any(column in df.columns for column in columns):
        return df
    values = df.reindex(columns=columns).to_numpy(dtype=float)
    if np.isinf(values).any() or ((values < 0) | (values > 1)).any():
        raise ValueError("coverage must be fractions in [0, 1] or NaN")
    return df.assign(amce=np.mean(np.abs(values - np.asarray(COVERAGE_LEVELS) / 100), axis=1))


# autorank_leaderboard.py at 3bf48c9, lines 105 to 115.
def _discrete_datasets(df):
    if "discrete_y" not in df.columns or "dataset" not in df.columns:
        return set()
    flagged = df["discrete_y"].fillna(False).astype(bool)
    return set(df.loc[flagged, "dataset"].tolist())


# autorank_leaderboard.py at 3bf48c9, lines 311 to 329.
def rank_with_autorank(pivot, metric, order, hib, alpha):
    result = autorank(pivot, alpha=alpha, order=order)
    if not hasattr(result, "rankdf") or result.rankdf is None:
        return None, None
    ranked_df = result.rankdf.copy()
    ranked_df.index.name = "model"
    ranked_df = ranked_df.reset_index()
    ranked_df = ranked_df.sort_values("meanrank", ascending=True).reset_index(drop=True)
    ranked_df.insert(0, "rank", ranked_df.index + 1)
    return ranked_df, result


def metric_matrix(frames: dict[str, pd.DataFrame], metric: str, *, coverage_target: float | None = None):
    """ScoringBench's ``load_metric_matrix`` (autorank_leaderboard.py at 3bf48c9, lines 140 to 258) on in-memory
    frames. Returns (pivot, models kept, models dropped, tables dropped)."""
    rows, discrete = [], set()
    for model, df in frames.items():
        if metric == "amce":
            df = _with_amce(df)
        if df.empty or metric not in df.columns:
            continue
        discrete |= _discrete_datasets(df)
        rows.append(pd.DataFrame({"dataset": df["dataset"].values, "fold": df["fold"].values, "model": model,
                                  "score": df[metric].astype(float).values}))
    if not rows:
        return None, [], [], []
    df_metric = pd.concat(rows, ignore_index=True)
    if metric in DISCRETE_NAN_KEYS and discrete:
        df_metric = df_metric[~df_metric["dataset"].isin(discrete)]
    if coverage_target is not None:
        df_metric = df_metric.assign(score=(df_metric["score"] - coverage_target).abs())
    df_agg = df_metric.groupby(["dataset", "model"])["score"].mean().reset_index()
    pivot = df_agg.pivot(index="dataset", columns="model", values="score")
    with_data = pivot.columns[pivot.notna().any()].tolist()
    total = len(pivot)
    kept = [m for m in with_data if total and pivot[m].notna().sum() / total >= 0.90]
    dropped = [m for m in with_data if m not in kept]
    subset = pivot[kept]
    filtered = subset.dropna(how="any")
    tables_dropped = sorted(set(subset.index) - set(filtered.index))
    return filtered, kept, dropped, tables_dropped


def source_metric_name(metric: str) -> str:
    """The name ScoringBench's leaderboard gives a score column (coverage becomes an absolute coverage error)."""
    if metric.startswith("coverage_"):
        return f"absolute_marginal_coverage_error_{metric.removeprefix('coverage_')}"
    if metric == "amce":
        return "mean_absolute_marginal_coverage_error"
    return metric


def coverage_target(metric: str) -> float | None:
    return int(metric.removeprefix("coverage_")) / 100.0 if metric.startswith("coverage_") else None


def higher_is_better(metric: str) -> bool:
    return source_metric_name(metric) in HIGHER_IS_BETTER


def rank(pivot: pd.DataFrame, metric: str = "crps"):
    """ScoringBench's ``rank_with_autorank`` with its direction rule and alpha 0.05. Returns (table, result)."""
    import contextlib
    import io
    name = source_metric_name(metric)
    hib = name in HIGHER_IS_BETTER
    with contextlib.redirect_stdout(io.StringIO()):
        table, result = rank_with_autorank(pivot, name, "descending" if hib else "ascending", hib, ALPHA)
    return table, result


def ranked(frames: dict[str, pd.DataFrame], metric: str = "crps", tables=None):
    """metric_matrix then rank, optionally on a subset of tables. Returns (table, result, pivot)."""
    pivot, *_ = metric_matrix(frames, metric, coverage_target=coverage_target(metric))
    if tables is not None:
        pivot = pivot.loc[[t for t in pivot.index if t in set(tables)]]
    table, result = rank(pivot, metric)
    return table, result, pivot


def nemenyi_cd(k: int, n: int, alpha: float = ALPHA) -> float:
    """The Nemenyi critical difference from scipy's studentized range (autorank uses statsmodels' approximation)."""
    from scipy.stats import studentized_range
    return float(studentized_range.ppf(1 - alpha, k, np.inf) * math.sqrt(k * (k + 1) / (12 * n)))


def position(table: pd.DataFrame, model: str) -> int:
    """1 + the number of models with a strictly lower mean rank (ties share a position)."""
    me = float(table.loc[table.model == model, "meanrank"].iloc[0])
    return int((table.meanrank < me - 1e-12).sum()) + 1


def meanrank(table: pd.DataFrame, model: str) -> float:
    return float(table.loc[table.model == model, "meanrank"].iloc[0])


# ---------------------------------------------------------------- folds and tables

def long(frames: dict[str, pd.DataFrame], metric: str = "crps") -> pd.DataFrame:
    return pd.concat([df[["dataset", "fold"]].assign(model=m, score=df[metric].values) for m, df in frames.items()],
                     ignore_index=True)


def fold_matrix(frames: dict[str, pd.DataFrame], metric: str = "crps") -> pd.DataFrame:
    return long(frames, metric).pivot_table(index=["dataset", "fold"], columns="model", values="score")


def common_folds(frames: dict[str, pd.DataFrame], metric: str = "crps") -> pd.DataFrame:
    return fold_matrix(frames, metric).dropna(how="any")


def geo(x) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.exp(np.mean(np.log(x))))


# ---------------------------------------------------------------- outputs

def record(section: str, values: dict) -> None:
    """Put a script's headline numbers under ``section`` in out/numbers.json (other sections are kept)."""
    path = OUT / "numbers.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data[section] = _plain(values)
    path.write_text(json.dumps(dict(sorted(data.items())), indent=2) + "\n")


def _plain(x):
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def write_json(name: str, values) -> Path:
    path = OUT / name
    path.write_text(json.dumps(_plain(values), indent=2) + "\n")
    return path


def write_csv(name: str, df: pd.DataFrame, **kwargs) -> Path:
    path = OUT / name
    df.to_csv(path, index=kwargs.pop("index", False), **kwargs)
    return path


def tex_name(model: str) -> str:
    if model == DRESSED:
        return "Dressed model (our run)"
    if model == POOL:
        return "Five members pooled, not dressed (our run)"
    return r"\texttt{" + model.replace("_", r"\_") + "}"


def tex_escape(text: str) -> str:
    return text.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def write_tex(name: str, columns: str, header: list[str] | str, rows: list[list[str]], *, groups: dict[int, str] | None = None,
              notes: str | None = None, long: bool = False) -> Path:
    """A booktabs table body: no float, caption or label. ``header`` is a list of cells or ready LaTeX header lines.
    With ``long`` it is a ``longtable`` whose header repeats on every page."""
    width = len(columns.replace("@{}", "").replace("|", ""))
    head = header if isinstance(header, str) else " & ".join(header) + r" \\"
    env = "longtable" if long else "tabular"
    lines = [rf"\begin{{{env}}}{{{columns}}}", r"\toprule", head, r"\midrule"]
    if long:
        lines += [r"\endfirsthead", r"\toprule", head, r"\midrule", r"\endhead", r"\bottomrule", r"\endlastfoot"]
    for i, row in enumerate(rows):
        if groups and i in groups:
            if i:
                lines.append(r"\midrule")
            lines.append(rf"\multicolumn{{{width}}}{{l}}{{\emph{{{groups[i]}}}}} \\")
        lines.append(" & ".join(row) + r" \\")
    if not long:
        lines.append(r"\bottomrule")
    lines.append(rf"\end{{{env}}}")
    if notes:
        lines.insert(0, "% " + notes)
    path = OUT / name
    path.write_text("\n".join(lines) + "\n")
    return path


def fmt(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}"
