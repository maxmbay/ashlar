"""The tests on the 23 development tables, as docs/dressed.md reports them under "How it was tested".

Sources (results/development/): dressed_and_parts.jsonl (the dressed model and simpler models from the same fits,
folds 0 to 2), library_check.jsonl (the library's dressed model on fold 0), network_on_reader.jsonl (the design that
fitted a network on top of the reader), five_pairs.jsonl (one pair and five cross-fitted pairs on fold 0). The guide's
own file, examples/readme/artifacts/dressed/devset.json in this repository, holds folds 1 and 2 rounded to six digits;
both give the same numbers, and the 46 fit times are read from it.

Fields of dressed_and_parts.jsonl: network = one member's network alone; five_networks = the five networks averaged;
one_pair_pooled = one network and one reader pooled; five_pairs_pooled = the five members pooled, not dressed;
smoothed_by_log_score and smoothed_by_crps = the pools smoothed at the log-score and CRPS bandwidth factors, no
corrector; corrected_without_gate = the corrector with no gate; dressed = the dressed model; log scores likewise.

Writes out/devset_tests.json and the section ``devset`` of out/numbers.json.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

GUIDE_FILE = c.FOLDER.parents[1] / "examples" / "readme" / "artifacts" / "dressed" / "devset.json"
guide = pd.DataFrame(json.loads(GUIDE_FILE.read_text())).rename(
    columns={"bagged": "five_pairs_pooled", "log_dressed": "log_score_dressed", "log_bagged_network": "log_score_five_networks"})
parts = pd.DataFrame(c.development("dressed_and_parts"))
library = pd.DataFrame(c.development("library_check"))
on_reader = pd.DataFrame(c.development("network_on_reader"))
pairs = pd.DataFrame(c.development("five_pairs"))
held = parts[parts.fold.isin([1, 2])].reset_index(drop=True)
fold0 = parts[parts.fold == 0].set_index("table")
assert len(guide) == len(held) == 46 and len(fold0) == 23 and len(library) == 23


def line(ratio):
    """Counts and summaries of a set of ratios. ``upper_median`` is the higher of the two middle values; ``median`` is
    the usual one, their average."""
    r = np.asarray(ratio, dtype=float)
    return {"n": len(r), "geo_mean": c.geo(r), "median": float(np.median(r)), "upper_median": float(np.sort(r)[len(r) // 2]),
            "better_gt_1pct": int((r < 0.99).sum()),
            "within_1pct": int(((r >= 0.99) & (r <= 1.01)).sum()), "worse_gt_1pct": int((r > 1.01).sum()),
            "worse_gt_5pct": int((r > 1.05).sum())}


out = {}
# The table of the guide: folds 1 and 2, 46 fits, from both files.
for name, df in (("from_development_results", held), ("from_guide_file", guide)):
    key = df.set_index(["table", "fold"])
    rows = {}
    arms = [("one_network", "network"), ("five_networks_averaged", "five_networks"),
            ("one_network_one_reader_pooled", "one_pair_pooled"), ("five_members_pooled_not_dressed", "five_pairs_pooled")]
    for label, arm in arms:
        if arm in key:
            rows[label] = line(key.dressed / key[arm])
    out[f"table1_{name}"] = rows
for label, v in out["table1_from_development_results"].items():
    print(f"{label:34s} geo {v['geo_mean']:.3f} median {v['median']:.3f} (upper {v['upper_median']:.3f}) better {v['better_gt_1pct']} within {v['within_1pct']}"
          f" worse {v['worse_gt_1pct']} (worse >5%: {v['worse_gt_5pct']})")
for label, v in out["table1_from_guide_file"].items():
    print(f"  guide's file {label:22s} geo {v['geo_mean']:.3f} median {v['median']:.3f} {v['better_gt_1pct']}/{v['within_1pct']}/{v['worse_gt_1pct']}")

# The log score against the five averaged networks, folds 1 and 2.
d = held.log_score_dressed - held.log_score_five_networks
out["log_score_folds12"] = {"above": int((d > 0).sum()), "n": len(d), "mean_nats": float(d.mean()), "median_nats": float(d.median()),
                             "upper_median_nats": float(np.sort(d)[len(d) // 2])}
print("log score:", out["log_score_folds12"])

# Which tables pull the geometric mean of the network rows down.
net = (held.dressed / held.network).groupby(held.table).apply(lambda r: c.geo(r)).sort_values()
out["tables_far_below_one_vs_network"] = {t: float(v) for t, v in net.head(5).items()}
print("most extreme dressed/network by table:", {t: round(v, 3) for t, v in net.head(5).items()})
geo_wo = {k: c.geo((held.dressed / held[arm])[~held.table.isin(net.head(3).index)])
          for k, arm in (("one_network", "network"), ("five_networks_averaged", "five_networks"))}
out["geo_without_three_most_extreme_tables"] = geo_wo
print("geo-mean without those three tables:", {k: round(v, 3) for k, v in geo_wo.items()})

# The pass lines on folds 1 and 2: the line that applied (line 5 of its script) and the looser one (5R).
x = held.dressed / held.five_pairs_pooled
gain = c.geo(held.dressed / held.smoothed_by_crps)
line5 = {"geo_vs_pool": c.geo(x), "geo_vs_pool_needs": "<= 1.005", "geo_vs_smoothed_crps_pool": gain,
         "geo_vs_smoothed_crps_pool_needs": "<= 0.99", "worse_gt_1pct": int((x > 1.01).sum()), "worse_needs": "<= 4",
         "log_above": int((d > 0).sum()), "log_above_needs": ">= 24", "log_mean": float(d.mean())}
line5["passes"] = bool(line5["geo_vs_pool"] <= 1.005 and gain <= 0.99 and line5["worse_gt_1pct"] <= 4 and line5["log_above"] >= 24
                       and d.mean() > 0)
line5r = {"geo_vs_pool": c.geo(x), "needs": "<= 0.97", "worse_gt_1pct": int((x > 1.01).sum()), "worse_needs": "<= 8",
          "worse_gt_5pct": int((x > 1.05).sum()), "worse_gt_5pct_needs": "<= 1"}
line5r["passes"] = bool(c.geo(x) <= 0.97 and gain <= 0.99 and line5r["worse_gt_1pct"] <= 8 and line5r["worse_gt_5pct"] <= 1
                        and line5["log_above"] >= 24 and d.mean() > 0)
out["pass_line_5"], out["pass_line_5R"] = line5, line5r
print("pass line:", line5)
print("looser line:", line5r)
bad = held.assign(ratio=x)[x > 1.05][["table", "fold", "ratio", "gate", "rows"]]
out["worse_gt_5pct_fits"] = bad.round(4).to_dict("records")
print("fits worse than the pool by more than 5%:", out["worse_gt_5pct_fits"])
we = parts[parts.table == "wave_energy"].set_index("fold")
out["wave_energy"] = {"fold2_dressed_over_pool": float(we.loc[2, "dressed"] / we.loc[2, "five_pairs_pooled"]),
                      "fold0_dressed_over_pool": float(we.loc[0, "dressed"] / we.loc[0, "five_pairs_pooled"]),
                      "fold1_dressed_over_pool": float(we.loc[1, "dressed"] / we.loc[1, "five_pairs_pooled"]),
                      "fold2_network_over_dressed": float(we.loc[2, "network"] / we.loc[2, "dressed"])}
print("wave_energy:", {k: round(v, 3) for k, v in out["wave_energy"].items()})
ap = parts[(parts.table == "557_analcatdata_apnea1_rated6") & (parts.fold == 2)].iloc[0]
out["apnea_rated6_fold2"] = {"dressed_over_pool": float(ap.dressed / ap.five_pairs_pooled), "gate": float(ap.gate),
                             "rows": int(ap.rows), "gate_rows_one_fifth": int(ap.rows) // 5, "stop_rows": int(ap.gate_rows)}
print("557_analcatdata_apnea1_rated6 fold 2:", out["apnea_rated6_fold2"])
keep = ~((held.table == "wave_energy") & (held.fold == 2))
out["geo_vs_pool_without_wave_energy_fold2"] = c.geo(x[keep])
print(f"geo-mean against the pool without wave_energy fold 2: {out['geo_vs_pool_without_wave_energy_fold2']:.3f}")

# Fold 0, which the design had seen.
out["fold0_dressed_vs_pool"] = line(fold0.dressed / fold0.five_pairs_pooled)
print("fold 0 dressed/pool:", out["fold0_dressed_vs_pool"])

# The library's dressed model against the research code, fold 0.
p = library.set_index("table")
for arm, name in (("dressed", "port_dressed"), ("five_pairs_pooled", "port_pool")):
    r = p[arm] / fold0.loc[p.index, arm]
    out[name] = {"median": float(r.median()), "geo_mean": c.geo(r), "largest_gap": float((r - 1).abs().max()),
                 "largest_gap_table": (r - 1).abs().idxmax()}
lib = p.dressed / p.five_pairs_pooled
dl = p.log_score_dressed - p.log_score_five_networks
out["port_library_dressed_vs_own_pool"] = line(lib)
out["port_library_log_score"] = {"above": int((dl > 0).sum()), "n": len(dl), "mean_nats": float(dl.mean())}
print("library against research code:", {k: out[k] for k in ("port_dressed", "port_pool")})
print("library on its own fits:", out["port_library_dressed_vs_own_pool"], out["port_library_log_score"])

# Three designs that failed on fold 0.
b0 = pairs.set_index("table")
nr = on_reader.set_index("table")
r_pair = nr.network_on_reader / b0.loc[nr.index, "one_pair_pooled"]
r_pool = nr.network_on_reader / b0.loc[nr.index, "five_pairs_pooled"]
out["failed_network_on_reader"] = {"vs_one_network_one_reader": line(r_pair), "vs_five_members_pooled": line(r_pool),
                                   "worse_at_all_vs_one_network_one_reader": int((r_pair > 1).sum()),
                                   "worse_at_all_vs_five_members_pooled": int((r_pool > 1).sum())}
print("network on the reader vs one network and one reader:", line(r_pair), "| vs five pooled:", line(r_pool))
for arm in ("smoothed_by_log_score", "smoothed_by_crps", "corrected_without_gate"):
    out[f"failed_{arm}_vs_pool_fold0"] = line(fold0[arm] / fold0.five_pairs_pooled)
    print(f"{arm} vs pool, fold 0:", out[f"failed_{arm}_vs_pool_fold0"])
small = fold0.sort_values("rows").head(4)
out["corrector_no_gate_smallest_four"] = [{"table": t, "rows": int(r.rows), "ratio_vs_pool": float(r.corrected_without_gate / r.five_pairs_pooled),
                                           "corrector_epoch": int(r.corrector_epoch)} for t, r in small.iterrows()]
worse_cor = fold0[fold0.corrected_without_gate > fold0.five_pairs_pooled * 1.01]
out["corrector_no_gate_worse_tables"] = {t: {"rows": int(r.rows), "ratio": float(r.corrected_without_gate / r.five_pairs_pooled),
                                             "corrector_epoch": int(r.corrector_epoch)} for t, r in worse_cor.iterrows()}
out["fold0_tables_by_rows"] = [[t, int(r)] for t, r in fold0.rows.sort_values().items()]
print("corrector without gate, smallest four:", out["corrector_no_gate_smallest_four"])
print("corrector without gate, tables worse by >1%:", out["corrector_no_gate_worse_tables"])

# Fit times of the 46 held-out fits (the guide's file, 'seconds').
s = guide.seconds / 60
out["seconds_folds12"] = {"median_min": float(s.median()), "min_min": float(s.min()), "max_hours": float(s.max() / 60),
                          "rows_of_longest": int(guide.loc[guide.seconds.idxmax(), "rows"]),
                          "longest_table": guide.loc[guide.seconds.idxmax(), "table"]}
print("times:", out["seconds_folds12"])

# The 23 tables.
tables = sorted(held.table.unique())
rated = [t for t in tables if "_rated" in t]
out["tables"] = {"n": len(tables), "rated": rated, "max_rows": int(parts.rows.max())}

c.write_json("devset_tests.json", out)
c.record("devset", {
    "table1": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in out["table1_from_development_results"].items()},
    "log_score_folds12": out["log_score_folds12"], "pass_line_5_passes": line5["passes"], "pass_line_5R_passes": line5r["passes"],
    "geo_vs_pool_without_wave_energy_fold2": out["geo_vs_pool_without_wave_energy_fold2"],
    "fold0_dressed_vs_pool": out["fold0_dressed_vs_pool"], "port_dressed": out["port_dressed"], "port_pool": out["port_pool"],
    "port_library_dressed_vs_own_pool": out["port_library_dressed_vs_own_pool"], "port_library_log_score": out["port_library_log_score"],
    "seconds_folds12": out["seconds_folds12"],
})
