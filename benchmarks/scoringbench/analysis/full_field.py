"""The full field of 64 models (63 published plus the dressed model), by mean CRPS per table.

ScoringBench's loader rules and ``rank_with_autorank``. Families come from out/families.csv (run families.py
first). Writes out/full_field_ranks.csv, out/tab_full_field.tex and the section ``full_field`` of out/numbers.json.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402

import pandas as pd  # noqa: E402

pub = c.published()
families = pd.read_csv(c.OUT / "families.csv").set_index("model").family.to_dict()
families[c.DRESSED] = "untuned"
frames = pub | {c.DRESSED: c.dressed()}
table, result, pivot = c.ranked(frames, "crps")
n, k = pivot.shape
me = c.meanrank(table, c.DRESSED)
# autorank's Friedman p underflows to 0 here; the statistic and log10 p are reported instead.
from scipy import stats  # noqa: E402
chi2_stat = float(stats.friedmanchisquare(*pivot.T.values).statistic)
import mpmath  # noqa: E402  (scipy's chi2.logsf underflows at this size)
log10_p = float(mpmath.log10(mpmath.gammainc((k - 1) / 2, chi2_stat / 2, mpmath.inf, regularized=True)))
pos = c.position(table, c.DRESSED)
cd = float(result.cd)

table = table[["rank", "model", "meanrank"]].copy()
table["family"] = table.model.map(families)
table["side"] = ["self" if m == c.DRESSED else ("ahead" if r < me else "behind") for m, r in zip(table.model, table.meanrank)]
table["within_cd_of_dressed"] = (table.meanrank - me).abs() <= cd
c.write_csv("full_field_ranks.csv", table.round(6))

ahead = table[table.side == "ahead"]
behind = table[table.side == "behind"]
by_side = {side: sub.family.value_counts().to_dict() for side, sub in (("ahead", ahead), ("behind", behind))}
pre_behind = behind[behind.family.isin(["pretrained", "fine-tuned"])]
significantly_ahead = ahead[me - ahead.meanrank > cd]
significantly_behind = behind[behind.meanrank - me > cd]
within = table[(table.side != "self") & table.within_cd_of_dressed]

print(f"{n} tables, {k} models | dressed position {pos}, mean rank {me:.4f} | CD {cd:.4f} | Friedman p {result.pvalue:.3e}"
      f" | {result.omnibus}/{result.posthoc}")
print("ahead by family:", by_side["ahead"], "| behind by family:", by_side["behind"])
print("pretrained or fine-tuned models behind the dressed model:", list(zip(pre_behind.model, pre_behind.meanrank.round(2))))
print(f"more than one CD ahead: {len(significantly_ahead)}; more than one CD behind: {len(significantly_behind)};"
      f" within one CD: {len(within)}")
for m in (c.REALMLP_TUNED, c.TABM_TUNED):
    print(f"  {m}: mean rank {c.meanrank(table, m):.4f}, position {c.position(table, m)}")
print(table.iloc[max(0, pos - 6):pos + 5].to_string(index=False))

labels = {"pretrained": "pretrained", "fine-tuned": "fine-tuned", "tuned": "tuned", "untuned": "untuned"}
tex_rows = [[str(r.rank), c.tex_name(r.model), labels[r.family], c.fmt(r.meanrank)] for r in table.itertuples()]
c.write_tex("tab_full_field.tex", "rllr", ["", "Model", "Family", "Mean rank"], tex_rows, long=True,
            notes=f"All 64 models, {n} tables, CRPS; ScoringBench's autorank; Nemenyi CD {cd:.2f}. Script: analysis/full_field.py")

c.record("full_field", {
    "n_tables": n, "n_models": k, "dressed_position": pos, "dressed_mean_rank": me, "nemenyi_cd": cd,
    "friedman_p_autorank": float(result.pvalue), "omnibus_test": result.omnibus,
    "friedman_chi2": chi2_stat, "friedman_df": k - 1, "friedman_log10_p": log10_p,
    "n_ahead": len(ahead), "ahead_by_family": by_side["ahead"], "behind_by_family": by_side["behind"],
    "all_ahead_pretrained_or_finetuned": bool(ahead.family.isin(["pretrained", "fine-tuned"]).all()),
    "pretrained_or_finetuned_behind": dict(zip(pre_behind.model, pre_behind.meanrank.round(4))),
    "n_more_than_one_cd_ahead": len(significantly_ahead), "n_more_than_one_cd_behind": len(significantly_behind),
    "n_within_one_cd": len(within),
    "realmlp_tuned_mean_rank": c.meanrank(table, c.REALMLP_TUNED), "realmlp_tuned_position": c.position(table, c.REALMLP_TUNED),
    "tabm_tuned_mean_rank": c.meanrank(table, c.TABM_TUNED), "tabm_tuned_position": c.position(table, c.TABM_TUNED),
    "best_model": table.model.iloc[0], "best_mean_rank": float(table.meanrank.iloc[0]),
    "best_untuned_published": behind[behind.family == "untuned"].model.iloc[0],
    "best_untuned_published_mean_rank": float(behind[behind.family == "untuned"].meanrank.iloc[0]),
})
