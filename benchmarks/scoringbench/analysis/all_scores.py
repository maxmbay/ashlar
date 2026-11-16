"""Every score in the files, for the 21 untuned models, by ScoringBench's rules.

For each score column of the published files (and ScoringBench's derived mean absolute coverage error, ``amce``):
the fold mean per table; for density scores and the PIT test, the 43 tables with a discrete target are left out
first; models with the score on fewer than 90% of the tables are dropped, then tables where a kept model lacks it;
coverage is scored as |coverage - nominal| per fold; direction: r2, dispersion and pit_ks_pvalue higher is better,
everything else lower; then ``rank_with_autorank``. The timing columns are ranked too, as ScoringBench's leaderboard
does, but our times are from a laptop CPU and the published ones from other hardware.

Writes out/all_scores.csv, out/tab_all_scores.tex and the section ``all_scores`` of out/numbers.json.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402

import pandas as pd  # noqa: E402

pub = c.published()
frames = {m: pub[m] for m in c.UNTUNED} | {c.DRESSED: c.dressed()}

GROUPS = [
    ("CRPS and its weighted forms", ["crps", "wcrps_left", "wcrps_center", "wcrps_right", "crts_alpha_1.01",
                                     "crts_alpha_1.2", "crts_alpha_1.5", "crts_alpha_2.0"]),
    ("Interval scores", [f"interval_score_{l}" for l in (20, 40, 60, 80, 90, 95)]),
    ("Energy scores", [f"energy_score_beta_{b}" for b in ("0.1", "0.3", "0.5", "0.7", "0.9", "1.0", "1.1", "1.3", "1.5",
                                                           "1.7", "1.8", "1.9")]),
    ("Coverage (absolute error from nominal)", [f"coverage_{l}" for l in (20, 40, 60, 80, 90, 95)] + ["amce"]),
    ("PIT (continuous targets)", ["pit_ks_stat", "pit_ks_pvalue"]),
    ("Density-based scores (continuous targets)", ["cde_loss", "dpd_beta_0.01", "dpd_beta_0.2", "dpd_beta_0.5",
                                                   "dpd_beta_1.0", "pseudospherical_alpha_1.5",
                                                   "pseudospherical_alpha_2.0", "pseudospherical_alpha_3.0"]),
    ("Point error", ["mae", "rmse", "r2"]),
    ("Sharpness and dispersion", ["sharpness", "dispersion"]),
    ("Time (not like for like: other hardware)", ["fit_time", "predict_time", "train_time"]),
]
listed = {m for _, ms in GROUPS for m in ms}
in_files = {col for df in pub.values() for col in df.select_dtypes("number").columns if col not in ("fold",)} | {"amce"}
assert listed == in_files - set(c.NON_SCORES), (listed ^ (in_files - set(c.NON_SCORES)))

# Scores that are another score under a second name (checked on every fold of every file).
# crts_alpha_2.0 and energy_score_beta_1.0 are CRPS by definition (ScoringBench's metrics.py), dpd_beta_1.0 is
# cde_loss; check it fold by fold (relative difference above 1e-6 counts as different).
same, same_detail = {}, {}
for a, b in (("crts_alpha_2.0", "crps"), ("energy_score_beta_1.0", "crps"), ("dpd_beta_1.0", "cde_loss")):
    off = {}
    for m, df in frames.items():
        ok = df[b].notna()
        rel = ((df[a] - df[b]).abs() / df[b].abs().clip(lower=1e-300))[ok]
        if (rel > 1e-6).any():
            off[m] = int((rel > 1e-6).sum())
    same[a] = b
    same_detail[a] = {"same_as": b, "models_with_folds_that_differ": off}

rows = []
for group, metrics in GROUPS:
    for metric in metrics:
        pivot, kept, dropped, tables_dropped = c.metric_matrix(frames, metric, coverage_target=c.coverage_target(metric))
        table, result = c.rank(pivot, metric)
        me = c.meanrank(table, c.DRESSED)
        pos = c.position(table, c.DRESSED)
        ahead = table[table.meanrank < me - 1e-12]
        best_other = table[table.model != c.DRESSED].iloc[0]
        first_tables = int((pivot.rank(axis=1, ascending=not c.higher_is_better(metric))[c.DRESSED] == 1).sum())
        cd = result.cd
        rows.append({
            "group": group, "score": metric, "ranked_as": c.source_metric_name(metric),
            "direction": "higher" if c.higher_is_better(metric) else "lower",
            "tables": len(pivot), "models": len(kept), "models_dropped": " ".join(dropped),
            "dressed_mean_rank": me, "dressed_position": pos, "tables_dressed_first": first_tables,
            "ahead_of_dressed": "; ".join(f"{r.model} {r.meanrank:.2f}" for r in ahead.itertuples()),
            "best_other": best_other.model, "best_other_mean_rank": float(best_other.meanrank),
            "test": f"{result.omnibus}/{result.posthoc}", "cd": float(cd) if cd is not None else None,
            "p_value": float(result.pvalue),
            "dressed_beyond_cd_of_best_other": (bool(best_other.meanrank - me > cd) if cd is not None and pos == 1 else None),
            "same_as": same.get(metric, ""),
        })
res = pd.DataFrame(rows)
c.write_csv("all_scores.csv", res, float_format="%.6g")
pd.set_option("display.width", 250)
print(res[["score", "tables", "models", "dressed_mean_rank", "dressed_position", "best_other", "best_other_mean_rank", "test",
           "cd", "models_dropped"]].to_string(index=False))
not_first = res[res.dressed_position > 1]
print("\nnot first:", not_first[["score", "dressed_position", "ahead_of_dressed"]].values.tolist())

tex_rows, groups = [], {}
for group, metrics in GROUPS:
    n_tables = sorted({int(res[res.score == m].tables.iloc[0]) for m in metrics})
    groups[len(tex_rows)] = f"{group}, {' or '.join(map(str, n_tables))} tables"
    for metric in metrics:
        r = res[res.score == metric].iloc[0]
        tex_rows.append([r"\texttt{" + c.tex_escape(metric) + "}", f"{r.dressed_mean_rank:.2f}", str(int(r.dressed_position)),
                         c.tex_name(r.best_other) if r.dressed_position == 1 else c.tex_name(r.ahead_of_dressed.split(" ")[0]),
                         f"{r.best_other_mean_rank:.2f}"])
assert (res.models == 21).all()
c.write_tex("tab_all_scores.tex", "lrrlr", ["Score", "Mean rank", "Place", "Best other model", "Its rank"],
            tex_rows, groups=groups,
            notes="The dressed model among the 21 untuned models on every score in the files, by ScoringBench's rules;"
                  " coverage is ranked as |coverage - nominal|. 'Best other' is the best other model when the dressed"
                  " model is first, else the first model. Script: analysis/all_scores.py")
summary = {}
for r in res.itertuples():
    summary[r.score] = {"mean_rank": round(r.dressed_mean_rank, 4), "position": int(r.dressed_position),
                        "tables": int(r.tables), "models": int(r.models), "best_other": r.best_other,
                        "best_other_mean_rank": round(r.best_other_mean_rank, 4)}
summary["n_scores"] = len(res)
summary["n_first"] = int((res.dressed_position == 1).sum())
summary["not_first"] = {r.score: int(r.dressed_position) for r in not_first.itertuples()}
summary["identical_scores"] = same_detail
first = res[res.dressed_position == 1]
summary["n_first_beyond_cd_of_best_other"] = int((first.dressed_beyond_cd_of_best_other == True).sum())  # noqa: E712
summary["first_not_beyond_cd"] = [r.score for r in first.itertuples() if r.dressed_beyond_cd_of_best_other is not True]
c.record("all_scores", summary)
