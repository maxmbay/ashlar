"""The ScoringBench numbers quoted in Ashlar's README and in docs/dressed.md, "On ScoringBench", as Markdown.

Reads what the other scripts wrote (out/numbers.json, out/untuned_crps_ranks.csv, out/by_table.csv) and the run's
own file, results/dressed.parquet, so it runs last. It computes nothing that the other scripts do not.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402

import pandas as pd  # noqa: E402

num = json.loads((c.OUT / "numbers.json").read_text())
u, lad, rob = num["untuned_crps"], num["ladder"]["steps"], num["robustness"]["21_untuned__98_tables"]
ranks = pd.read_csv(c.OUT / "untuned_crps_ranks.csv")
by_table = pd.read_csv(c.OUT / "by_table.csv")
dressed = c.dressed()
targets = c.target_values()
NAME = lambda m: "Dressed model (our run)" if m == c.DRESSED else f"`{m}`"  # noqa: E731
FILES = ["one_network_100_epochs", "one_network", "one_network_one_reader", "five_pairs_pooled", "dressed"]

few = targets.groupby("table").at_most_20_target_values.agg(["min", "max"])
assert (few["min"] == few["max"]).all()  # each table is on one side of the 20-value line in every fold
few = sorted(few.index[few["max"] == 1], key=str.lower)
assert len(few) == num["calibration"]["n_few_value_tables"]
out = ["## The numbers in Ashlar's README and docs/dressed.md", "",
       "### The run", "",
       f"The dressed model has results on {len(dressed)} folds of {dressed.dataset.nunique()} tables, with at most "
       f"{targets.training_rows.max():,} training rows and {targets.test_rows.max()} test rows in a fold. On {len(few)} "
       f"tables the target has at most 20 distinct values in the training rows of every fold: {', '.join(few)}.", "",
       f"A fit took a median of {num['timing']['fit_minutes_median']:.0f} minutes per fold "
       f"({num['timing']['fit_minutes_median']:.1f}). Fitting and predicting took {dressed.train_time.sum() / 3600:.0f} "
       f"hours summed over the {len(dressed)} folds (ScoringBench's `train_time`).", "",
       "### Mean ranks of the 21 untuned models", "",
       f"The dressed model and the 20 published models trained from scratch and untuned, by CRPS on {u['n_tables']} "
       "tables. A mean rank is a model's rank on each table, averaged over the tables; 1 is best.", "",
       "| Model | Mean rank |", "|---|---|"]
out += [f"| {NAME(r.model)} | {r.meanrank:.2f} |" for r in ranks.itertuples()]
out += ["", f"Friedman test: chi-square {u['friedman_chi2']:.1f} with {u['friedman_df']} degrees of freedom, "
        f"p = {u['friedman_p_autorank']:.1e}. The Nemenyi critical difference at the 5% level, for {u['n_models']} models "
        f"on {u['n_tables']} tables, is {u['nemenyi_cd']:.2f}. The dressed model's lead over the next model, "
        f"`{u['next_model']}`, is {u['gap_to_next']:.2f}, and "
        + (f"all {u['n_other_models']} other models are" if u["n_beyond_one_cd"] == u["n_other_models"]
           else f"{u['n_beyond_one_cd']} of the other {u['n_other_models']} models are")
        + " more than one critical difference behind it.", ""]

place = by_table.set_index("dataset").rank_among_21
out += ["### The dressed model's place on each table", "",
        f"It has the lowest mean CRPS of the 21 on {int((place == 1).sum())} of the {len(place)} tables. Its place on the "
        f"other {int((place > 1).sum())}:", "", "| Place | Tables |", "|---|---|"]
for k in sorted(place[place > 1].unique()):
    out.append(f"| {k} | {', '.join(sorted(place.index[place == k], key=str.lower))} |")

out += ["", "### What each part adds", "",
        "Each row ranks one Ashlar model alone among the same 20 published models. Lead: the mean rank of the best of the "
        "20 minus the row's. CRPS ratio: the geometric mean over the tables of the row's mean CRPS divided by the row "
        "above's; the last two columns count the tables where it is lower or higher by more than 1%.", "",
        "| Results file | Model | Mean rank | Place | Best of the 20 | Its mean rank | Lead | Critical difference "
        "| CRPS ratio | Tables lower | Tables higher |", "|---|---|---|---|---|---|---|---|---|---|---|"]
for name, s in zip(FILES, lad, strict=True):
    cells = [f"`{name}`", s["step"], f"{s['mean_rank']:.2f}", str(s["position"]), f"`{s['best_other']}`",
             f"{s['best_other_mean_rank']:.2f}", f"{s['lead']:.2f}", f"{s['cd']:.2f}"]
    cells += ([f"{s['ratio_to_previous_geo']:.3f}", str(int(s["lower_by_more_than_1pct"])), str(int(s["higher_by_more_than_1pct"]))]
              if pd.notna(s.get("ratio_to_previous_geo")) else ["", "", ""])
    out.append("| " + " | ".join(cells) + " |")
out += ["", f"Pooled with one label reader, the network's CRPS is {100 * (1 - lad[2]['ratio_to_previous_geo']):.1f}% lower on "
        f"the geometric mean. The dressing lowers the pooled members' CRPS by a further "
        f"{100 * (1 - lad[4]['ratio_to_previous_geo']):.1f}%.", ""]

r9 = num["rescore_9406d60"]
out += ["### Since the run", "",
        f"Rescored with ScoringBench's code of 13 November (commit 7f36e0b), the undressed pool's CRPS is identical on "
        f"{num['rescore']['crps_identical']} of the {num['rescore']['folds_rescored']} folds rescored. With the code of "
        f"14 November (commit 9406d60) it is identical on {r9['crps_identical']} of the {r9['folds_rescored']}; the largest "
        f"change on a fold is {r9['crps_largest_increase']:.3%} higher ({r9['crps_largest_increase_at'][0]} fold "
        f"{r9['crps_largest_increase_at'][1]}) and the largest decrease {r9['crps_largest_decrease']:.3%}, and by table the "
        f"ratio of mean CRPS runs from {r9['crps_table_ratio_min']:.5f} to {r9['crps_table_ratio_max']:.5f} "
        f"({r9['crps_table_ratio_max_dataset']}).", "",
        f"Without {c.LEAKY_TABLE}, which ScoringBench has since dropped, the dressed model's mean rank on the other "
        f"{rob['tables']} tables is {rob['dressed_mean_rank']:.2f} and `{rob['next_model']}`'s {rob['next_mean_rank']:.2f}, "
        f"and the critical difference is {rob['cd']:.2f}."]
print("\n".join(out))
