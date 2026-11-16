"""Calibration. Empirical coverage of the central 20 to 95% intervals against nominal, and the PIT test.

Coverage of a table is the mean over its folds (each model's own folds); the average over tables weights each
table equally. Shown for the dressed model and the three strongest other untuned models by CRPS mean rank, plus the
untuned model with the best mean absolute coverage error (a conformal model), on all 99 tables and separately on the
56 tables with a continuous target and the 43 that ScoringBench flags as discrete (``discrete_y``), and split by
the dressed model's path: the 12 tables with at most 20 distinct training values in every fold (it predicts a
probability for each value there, so a central interval holds whole values and covers more than nominal by
construction) and the other 87 (it fits a density). The 12 come from results/target_values.csv, counted on the
benchmark's own splits.

The PIT Kolmogorov-Smirnov test is defined by ScoringBench only on the continuous tables (per fold, against the
uniform distribution).

Writes out/coverage.csv, out/pit.csv and the section ``calibration`` of out/numbers.json.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

LEVELS = (20, 40, 60, 80, 90, 95)
pub = c.published()
frames = {m: pub[m] for m in c.UNTUNED} | {c.DRESSED: c.dressed()}
ranks = pd.read_csv(c.OUT / "untuned_crps_ranks.csv")
strongest = ranks[ranks.model != c.DRESSED].model.head(3).tolist()
scores = pd.read_csv(c.OUT / "all_scores.csv").set_index("score")
amce_best = scores.loc["amce", "ahead_of_dressed"].split(";")[0].split(" ")[0]
shown = [c.DRESSED] + strongest + [amce_best]
discrete = set(c.dressed().query("discrete_y").dataset)
# The 12 tables where the dressed model takes its few-values path (at most 20 distinct training values in every fold),
# from results/target_values.csv; all 12 are among the 43 discrete ones.
few = set(c.target_values().groupby("table").at_most_20_target_values.min().pipe(lambda s: s[s == 1]).index)
assert few <= discrete

rows = []
for m in shown:
    df = frames[m]
    per_table = df.groupby("dataset")[[f"coverage_{l}" for l in LEVELS]].mean()
    subsets = [("all", per_table.index), ("continuous", [t for t in per_table.index if t not in discrete]),
               ("discrete", [t for t in per_table.index if t in discrete])]
    if few:
        subsets += [("at most 20 values", [t for t in per_table.index if t in few]),
                    ("more than 20 values", [t for t in per_table.index if t not in few]),
                    ("discrete, more than 20 values", [t for t in per_table.index if t in discrete and t not in few])]
    for subset, tables in subsets:
        sub = per_table.loc[tables]
        for l in LEVELS:
            cov = sub[f"coverage_{l}"]
            rows.append({"model": m, "tables": subset, "n_tables": len(sub), "nominal": l / 100, "mean_coverage": cov.mean(),
                         "median_coverage": cov.median(), "mean_abs_error": (cov - l / 100).abs().mean(),
                         "share_tables_below_nominal": (cov < l / 100).mean(),
                         "pooled_over_folds": df[df.dataset.isin(tables)][f"coverage_{l}"].mean()})
cov = pd.DataFrame(rows)
c.write_csv("coverage.csv", cov, float_format="%.6g")
pd.set_option("display.width", 220)
wide = cov.pivot_table(index=["tables", "model"], columns="nominal", values="mean_coverage").round(3)
print(wide.to_string())
err = cov.groupby(["tables", "model"]).mean_abs_error.mean().unstack(0).round(4)
print("\nmean absolute coverage error, averaged over the six levels:\n", err.to_string())

# PIT test on the continuous tables.
pit_rows = []
for m, df in frames.items():
    d = df[~df.dataset.isin(discrete)]
    d = d[d.pit_ks_stat.notna()]
    pit_rows.append({"model": m, "folds": len(d), "tables": d.dataset.nunique(),
                     "mean_ks_stat_over_tables": d.groupby("dataset").pit_ks_stat.mean().mean(),
                     "median_ks_stat_folds": d.pit_ks_stat.median(),
                     "share_folds_p_below_0_05": (d.pit_ks_pvalue < 0.05).mean(),
                     "share_folds_p_below_0_01": (d.pit_ks_pvalue < 0.01).mean()})
pit = pd.DataFrame(pit_rows).sort_values("mean_ks_stat_over_tables").reset_index(drop=True)
c.write_csv("pit.csv", pit, float_format="%.6g")
print("\nPIT KS test on continuous tables:\n", pit.round(4).to_string(index=False))

summary = {"models_shown": shown, "strongest_other_untuned": strongest, "best_coverage_error_untuned": amce_best,
           "n_discrete_tables": len(discrete), "n_continuous_tables": 99 - len(discrete)}
summary["n_few_value_tables"] = len(few)
for m in shown:
    for subset in ("all", "continuous", "discrete") + (("at most 20 values", "more than 20 values",
                                                         "discrete, more than 20 values") if few else ()):
        sub = cov[(cov.model == m) & (cov.tables == subset)]
        summary[f"{m}__{subset}__mean_coverage"] = dict(zip([f"{int(x * 100)}" for x in sub.nominal], sub.mean_coverage.round(4)))
        summary[f"{m}__{subset}__mean_abs_error_avg_levels"] = round(float(sub.mean_abs_error.mean()), 4)
d_pit = pit.set_index("model").loc[c.DRESSED]
summary["pit_dressed"] = {k: (float(v) if isinstance(v, (float, np.floating)) else int(v)) for k, v in d_pit.items()}
summary["pit_rank_by_mean_ks_stat"] = int(pit.index[pit.model == c.DRESSED][0]) + 1
summary["pit_share_folds_rejected_5pct"] = dict(zip(pit.model, pit.share_folds_p_below_0_05.round(4)))
c.record("calibration", summary)
