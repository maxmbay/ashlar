"""The dressed model table by table among the 21 untuned models.

Per-table CRPS is each model's mean over the folds it has (ScoringBench's aggregation); a second reading uses only
the folds every one of the 21 models has on that table (Job_Profitability then has none and drops out). Ranks are
1 + the number of models with a strictly lower mean (no exact ties occur in either reading).

Writes out/by_table.csv (one row per table), out/by_table_misses.csv, out/tab_misses.tex and the section ``by_table``
of out/numbers.json.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

pub = c.published()
frames = {m: pub[m] for m in c.UNTUNED} | {c.DRESSED: c.dressed()}
pivot, *_ = c.metric_matrix(frames, "crps")
assert pivot.shape == (99, 21)
fold = c.fold_matrix(frames, "crps")
common = fold.dropna(how="any").groupby(level=0).mean()

rows = []
for t in pivot.index:
    row = pivot.loc[t]
    others = row.drop(c.DRESSED)
    beaten_by = others[others < row[c.DRESSED]].sort_values()
    rank = len(beaten_by) + 1
    best_other = others.idxmin()
    r_common = np.nan
    if t in common.index:
        rc = common.loc[t]
        r_common = int((rc.drop(c.DRESSED) < rc[c.DRESSED]).sum()) + 1
    rows.append({
        "dataset": t, "rank_among_21": rank, "rank_among_21_common_folds": r_common,
        "folds_common_to_all_21": int(fold.loc[t].dropna(how="any").shape[0]),
        "dressed_crps": row[c.DRESSED], "best_other": best_other, "best_other_crps": others.min(),
        "ratio_to_best_other": row[c.DRESSED] / others.min(),
        "beaten_by": "; ".join(f"{m} ({v / row[c.DRESSED]:.3f})" for m, v in beaten_by.items()),
    })
bt = pd.DataFrame(rows).sort_values("ratio_to_best_other").reset_index(drop=True)
c.write_csv("by_table.csv", bt)

misses = bt[bt.rank_among_21 > 1].sort_values(["rank_among_21", "dataset"], key=lambda s: s if s.name == "rank_among_21" else s.str.lower())
c.write_csv("by_table_misses.csv", misses[["dataset", "rank_among_21", "rank_among_21_common_folds", "best_other",
                                           "ratio_to_best_other", "beaten_by"]])
ratio = bt.ratio_to_best_other
q = np.percentile(ratio, [0, 10, 25, 50, 75, 90, 100])
counts = bt.rank_among_21.value_counts().sort_index().to_dict()
counts_common = bt.rank_among_21_common_folds.dropna().astype(int).value_counts().sort_index().to_dict()
changed = bt[(bt.rank_among_21_common_folds.notna()) & (bt.rank_among_21_common_folds != bt.rank_among_21)]
print("rank counts (each model's folds):", counts)
print("rank counts (folds all 21 have):", counts_common, "| tables left:", int(bt.rank_among_21_common_folds.notna().sum()))
print("tables whose rank changes with common folds:", changed[["dataset", "rank_among_21", "rank_among_21_common_folds"]].values.tolist())
print("ratio to best other untuned: min %.3f p10 %.3f p25 %.3f median %.3f p75 %.3f p90 %.3f max %.3f" % tuple(q))
print(f"below 0.9: {(ratio < 0.9).sum()}, 0.9-0.99: {((ratio >= 0.9) & (ratio < 0.99)).sum()}, 0.99-1: {((ratio >= 0.99) & (ratio < 1)).sum()},"
      f" above 1: {(ratio > 1).sum()}, above 1.1: {(ratio > 1.1).sum()}")
print(misses[["dataset", "rank_among_21", "rank_among_21_common_folds", "ratio_to_best_other", "beaten_by"]].to_string(index=False))

tex_rows = [[c.tex_escape(r.dataset), str(r.rank_among_21), f"{r.ratio_to_best_other:.3f}", c.tex_name(r.best_other)]
            for r in misses.itertuples()]
c.write_tex("tab_misses.tex", "lrrl", ["Table", "Place", "Ratio", "Lowest untuned model"], tex_rows,
            notes="The tables where the dressed model is not first of the 21 untuned models; ratio = dressed CRPS / the"
                  " lowest other untuned model's; every model ahead is listed in out/by_table_misses.csv."
                  " Script: analysis/by_table.py")
c.record("by_table", {
    "tables_first": int(counts.get(1, 0)), "rank_counts": counts, "n_not_first": int(len(misses)),
    "rank_counts_common_folds": counts_common, "tables_with_common_folds": int(bt.rank_among_21_common_folds.notna().sum()),
    "ratio_to_best_other_quantiles": dict(zip(["min", "p10", "p25", "median", "p75", "p90", "max"], q)),
    "ratio_geo_mean": c.geo(ratio), "n_ratio_below_0_9": int((ratio < 0.9).sum()), "n_ratio_above_1": int((ratio > 1).sum()),
    "n_ratio_above_1_1": int((ratio > 1.1).sum()),
    "worst_table": bt.iloc[-1].dataset, "worst_ratio": float(bt.iloc[-1].ratio_to_best_other),
    "best_table": bt.iloc[0].dataset, "best_ratio": float(bt.iloc[0].ratio_to_best_other),
    "misses": {r.dataset: int(r.rank_among_21) for r in misses.itertuples()},
    "tables_where_model_is_lower_than_dressed": dict(pd.Series([x.split(" (")[0] for b in misses.beaten_by for x in b.split("; ")])
                                                     .value_counts().items()),
})
