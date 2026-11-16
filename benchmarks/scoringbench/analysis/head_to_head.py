"""The dressed model head to head, by table and by fold, with the geometric mean CRPS ratio.

Opponents: each of the 20 untuned models, tuned TabM, tuned RealMLP, and the best pretrained model (the
pretrained or fine-tuned model with the lowest mean rank in the 64-model field; run full_field.py first).

By table, a model's CRPS on a table is its mean over the folds it has (ScoringBench's aggregation); a second
reading uses only the folds both models have on that table. By fold, only folds both models have are compared. A
ratio is the dressed model's CRPS divided by the opponent's, so below 1 means the dressed model is lower.

The sign test (two-sided binomial on wins and losses, ties left out) and the Wilcoxon signed-rank test on the log
ratios are our additions; ScoringBench's own test is the Friedman-Nemenyi ranking in untuned_crps.py.

Writes out/head_to_head.csv, out/tab_head_to_head.tex and the section ``head_to_head`` of out/numbers.json.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

pub = c.published()
dr = c.dressed()
field = pd.read_csv(c.OUT / "full_field_ranks.csv")
best_pre = field[field.family.isin(["pretrained", "fine-tuned"])].iloc[0].model
untuned_order = pd.read_csv(c.OUT / "untuned_crps_ranks.csv").model.tolist()
opponents = [m for m in untuned_order if m != c.DRESSED] + [c.TABM_TUNED, c.REALMLP_TUNED, best_pre]
kind = {m: "untuned" for m in c.UNTUNED} | {c.TABM_TUNED: "tuned", c.REALMLP_TUNED: "tuned", best_pre: "best pretrained"}

d_fold = dr.set_index(["dataset", "fold"]).crps
d_table = dr.groupby("dataset").crps.mean()

# A narrower fold set: the folds of the dressed model that at least one of the 20 untuned published models also has.
untuned_folds = set()
for m in c.UNTUNED:
    untuned_folds |= set(zip(pub[m].dataset, pub[m].fold))


def tests(ratio):
    r = np.asarray(ratio, dtype=float)
    lower, higher = int((r < 1).sum()), int((r > 1).sum())
    sign_p = stats.binomtest(lower, lower + higher, 0.5).pvalue if lower + higher else float("nan")
    logs = np.log(r[r != 1])
    wil_p = stats.wilcoxon(logs).pvalue if len(logs) > 0 else float("nan")
    return lower, higher, int((r == 1).sum()), float(sign_p), float(wil_p)


rows = []
for m in opponents:
    o = pub[m]
    o_fold = o.set_index(["dataset", "fold"]).crps
    o_table = o.groupby("dataset").crps.mean()
    # by table, each model's own folds (ScoringBench)
    tables = d_table.index.intersection(o_table.index)
    rt = d_table[tables] / o_table[tables]
    lt, ht, tt, sp_t, wp_t = tests(rt)
    # by table, folds both have
    both = d_fold.index.intersection(o_fold.index)
    dc = d_fold[both].groupby(level=0).mean()
    oc = o_fold[both].groupby(level=0).mean()
    rtc = dc / oc
    ltc, htc, _, _, _ = tests(rtc)
    # by fold
    rf = d_fold[both] / o_fold[both]
    lf, hf, tf, sp_f, wp_f = tests(rf)
    sub = [k for k in both if k in untuned_folds]
    lf_s = int((d_fold[sub] < o_fold[sub]).sum())
    rows.append({
        "opponent": m, "kind": kind[m],
        "tables": len(tables), "tables_dressed_lower": lt, "tables_dressed_higher": ht, "tables_tied": tt,
        "geo_ratio_tables": c.geo(rt), "median_ratio_tables": float(np.median(rt)),
        "sign_p_tables": sp_t, "wilcoxon_p_tables": wp_t,
        "tables_common_folds": len(rtc), "tables_dressed_lower_common_folds": ltc, "geo_ratio_tables_common_folds": c.geo(rtc),
        "folds": len(both), "folds_dressed_lower": lf, "folds_dressed_higher": hf, "folds_tied": tf,
        "geo_ratio_folds": c.geo(rf), "median_ratio_folds": float(np.median(rf)),
        "sign_p_folds": sp_f, "wilcoxon_p_folds": wp_f,
        "folds_with_an_untuned_model": len(sub), "folds_dressed_lower_with_an_untuned_model": lf_s,
    })
h2h = pd.DataFrame(rows)
c.write_csv("head_to_head.csv", h2h, float_format="%.6g")
pd.set_option("display.width", 250)
print(h2h[["opponent", "tables_dressed_lower", "tables", "geo_ratio_tables", "folds_dressed_lower", "folds",
           "geo_ratio_folds", "folds_dressed_lower_with_an_untuned_model", "folds_with_an_untuned_model", "sign_p_tables",
           "wilcoxon_p_tables"]]
      .to_string(index=False))

tex_rows = []
for r in h2h.itertuples():
    tex_rows.append([c.tex_name(r.opponent), f"{r.tables_dressed_lower} of {r.tables}", f"{r.geo_ratio_tables:.3f}",
                     f"{r.folds_dressed_lower} of {r.folds}", f"{r.geo_ratio_folds:.3f}"])
groups = {0: "Untuned, trained from scratch", len(c.UNTUNED): "Tuned on each table",
          len(c.UNTUNED) + 2: "Best pretrained model in the full field"}
header = (r" & \multicolumn{2}{c}{By table} & \multicolumn{2}{c}{By fold} \\ \cmidrule(lr){2-3} \cmidrule(lr){4-5}"
          "\n" r"Opponent & Lower & Ratio & Lower & Ratio \\")
c.write_tex("tab_head_to_head.tex", "lrrrr", header, tex_rows, groups=groups,
            notes="Dressed model against each opponent; 'lower' = dressed CRPS lower; ratio = dressed/opponent (below 1 favours"
                  " the dressed model); tables use each model's own fold mean, folds use the folds both have."
                  " Script: analysis/head_to_head.py")

key = lambda m: h2h.set_index("opponent").loc[m]  # noqa: E731
summary = {}
for name, m in (("tabm_tuned", c.TABM_TUNED), ("realmlp_tuned", c.REALMLP_TUNED), ("best_pretrained", best_pre),
                ("condensite", "condensite_pitbst_v0_1_1")):
    r = key(m)
    summary[name] = {"model": m, "tables_lower": int(r.tables_dressed_lower), "tables": int(r.tables),
                     "geo_ratio_tables": float(r.geo_ratio_tables), "sign_p_tables": float(r.sign_p_tables),
                     "wilcoxon_p_tables": float(r.wilcoxon_p_tables),
                     "folds_lower": int(r.folds_dressed_lower), "folds": int(r.folds), "geo_ratio_folds": float(r.geo_ratio_folds),
                     "folds_lower_with_an_untuned_model": int(r.folds_dressed_lower_with_an_untuned_model),
                     "folds_with_an_untuned_model": int(r.folds_with_an_untuned_model)}
unt = h2h[h2h.kind == "untuned"]
summary["untuned_min_tables_lower"] = int(unt.tables_dressed_lower.min())
summary["untuned_min_tables_lower_model"] = unt.loc[unt.tables_dressed_lower.idxmin(), "opponent"]
summary["untuned_max_geo_ratio_tables"] = float(unt.geo_ratio_tables.max())
summary["untuned_max_geo_ratio_tables_model"] = unt.loc[unt.geo_ratio_tables.idxmax(), "opponent"]
c.record("head_to_head", summary)
