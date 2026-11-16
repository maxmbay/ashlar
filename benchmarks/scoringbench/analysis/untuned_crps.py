"""The dressed model among the 21 untuned models trained from scratch, by mean CRPS per table.

ScoringBench's rules throughout: CRPS averaged over each model's folds on a table, the 90% completeness rule, then
``rank_with_autorank`` (Friedman test, Nemenyi post hoc test at alpha 0.05, ties get the average rank).

Writes out/untuned_crps_ranks.csv, out/untuned_crps_pivot.csv, out/tab_untuned.tex and the section
``untuned_crps`` of out/numbers.json.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

pub = c.published()
frames = {m: pub[m] for m in c.UNTUNED} | {c.DRESSED: c.dressed()}
table, result, pivot = c.ranked(frames, "crps")
k, n = pivot.shape[1], pivot.shape[0]
me = c.meanrank(table, c.DRESSED)
others = table[table.model != c.DRESSED].reset_index(drop=True)
cd = float(result.cd)
beyond = others[others.meanrank - me > cd]

# The same test by hand, as a check on autorank: Friedman chi-square and the Iman-Davenport F form.
chi2, p_chi2 = stats.friedmanchisquare(*pivot.T.values)
ranks = pivot.rank(axis=1, method="average")
f_id = (n - 1) * chi2 / (n * (k - 1) - chi2)
p_id = stats.f.sf(f_id, k - 1, (k - 1) * (n - 1))

# Ties in the per-table means, and the per-table rank of the dressed model.
ties = int(sum(pivot.loc[t].duplicated().any() for t in pivot.index))
dressed_rank = ranks[c.DRESSED]
first = int((dressed_rank == 1).sum())
min_ranks = pivot.rank(axis=1, method="min")[c.DRESSED]

# Wins against each model, table by table (per-table means).
wins = {m: int((pivot[c.DRESSED] < pivot[m]).sum()) for m in pivot.columns if m != c.DRESSED}
losses = {m: int((pivot[c.DRESSED] > pivot[m]).sum()) for m in pivot.columns if m != c.DRESSED}

rows = table[["rank", "model", "meanrank"]].copy()
rows["behind_dressed"] = rows.meanrank - me
rows["beyond_cd"] = (rows.model != c.DRESSED) & (rows.behind_dressed > cd)
rows["tables_dressed_lower"] = rows.model.map(wins)
rows["tables_dressed_higher"] = rows.model.map(losses)
c.write_csv("untuned_crps_ranks.csv", rows.round(6))
c.write_csv("untuned_crps_pivot.csv", pivot.reset_index())

tex_rows = [[c.tex_name(r.model), c.fmt(r.meanrank), "" if r.model == c.DRESSED else f"{r.behind_dressed:.2f}",
             "" if r.model == c.DRESSED else f"{wins[r.model]}"] for r in rows.itertuples()]
c.write_tex("tab_untuned.tex", "lrrr", ["Model", "Mean rank", "Behind", "Tables lower"], tex_rows,
            notes=f"21 untuned models, {n} tables, CRPS; ScoringBench's autorank; Nemenyi CD {cd:.2f} at 5%;"
                  " 'Behind' = mean rank minus the dressed model's; 'Tables lower' = tables where the dressed model's"
                  " mean CRPS is lower. Script: analysis/untuned_crps.py")

print(f"{n} tables, {k} models | test {result.omnibus}/{result.posthoc} | dressed {me:.4f} | next {others.model[0]}"
      f" {others.meanrank[0]:.4f} | gap {others.meanrank[0] - me:.4f} | CD {cd:.4f} | Friedman p {result.pvalue:.3e}")
print(f"beyond one CD: {len(beyond)} of {len(others)} | first on {first} tables | tables with any tie {ties}")
print(f"by hand: Friedman chi2 {chi2:.2f} (df {k - 1}), p {p_chi2:.3e}; Iman-Davenport F {f_id:.2f}, p {p_id:.3e}")
print(table[["rank", "model", "meanrank"]].to_string(index=False))

c.record("untuned_crps", {
    "n_tables": n, "n_models": k, "omnibus_test": result.omnibus, "posthoc_test": result.posthoc,
    "all_normal": bool(result.all_normal),
    "dressed_mean_rank": me, "next_model": others.model[0], "next_mean_rank": float(others.meanrank[0]),
    "gap_to_next": float(others.meanrank[0] - me), "nemenyi_cd": cd, "nemenyi_cd_formula_check": c.nemenyi_cd(k, n),
    "friedman_p_autorank": float(result.pvalue), "friedman_chi2": float(chi2), "friedman_df": k - 1,
    "iman_davenport_F": float(f_id), "iman_davenport_p": float(p_id),
    "n_beyond_one_cd": int(len(beyond)), "n_other_models": int(len(others)),
    "tables_first": first, "tables_first_min_rank_ties": int((min_ranks == 1).sum()),
    "tables_with_exact_ties": ties,
    "mean_ranks": dict(zip(table.model, table.meanrank.round(4))),
})
