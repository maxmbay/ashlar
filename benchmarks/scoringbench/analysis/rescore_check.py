"""Rescoring: the undressed pool's saved predictions, scored again by two newer versions of ScoringBench's code.

ScoringBench has changed its scoring code twice since the run. The run saved the five pairs' pooled predictions (the
undressed pool) and not the dressed model's. results/five_pairs_pooled.parquet holds the run's own scores
(ScoringBench's code at 3bf48c9). results/five_pairs_pooled_rescored_7f36e0b.parquet holds the saved predictions as
the code of 13 November scores them (commit 7f36e0b), and results/five_pairs_pooled_rescored_9406d60.parquet as the
code of 14 November scores them (commit 9406d60, which spreads the mass beyond the outermost quantile levels as a
uniform slab out to the edge of the training range, where it was an atom at the outermost quantile). This compares
the scores fold by fold. A table's ratio is the ratio of its mean CRPS over the folds.

Three tables were not rescored: Brazilian_houses, which the newer code drops, and two tables whose saved predictions
had been overwritten by those of another table with the same training labels (the saved files were named by a hash
of the training labels).

Writes the sections ``rescore`` (7f36e0b) and ``rescore_9406d60`` of out/numbers.json.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402

import numpy as np  # noqa: E402

SKIP = ("dataset", "fold", "model", "discrete_y", "error", "error_type")


def compare(old, new):
    """The rescored scores against the run's own, fold by fold."""
    m = old.merge(new, on=["dataset", "fold"], suffixes=("_old", "_new"))
    ratio = m.crps_new / m.crps_old
    same = np.isclose(m.crps_old, m.crps_new, rtol=1e-12, atol=0)
    means = m.groupby("dataset")[["crps_old", "crps_new"]].mean()
    unchanged, changed = [], {}
    for col in [k for k in old.columns if k in new.columns and k not in SKIP and not k.endswith("_time")]:
        a, b = m[f"{col}_old"], m[f"{col}_new"]
        differs = ~((a == b) | (a.isna() & b.isna()))
        if differs.any():
            changed[col] = {"folds": int(differs.sum()),
                            "largest_relative_change": float(((b - a) / a).abs()[differs & (a != 0)].max())}
        else:
            unchanged.append(col)
    return m, ratio, same, means.crps_new / means.crps_old, unchanged, changed


old = c.pool()
new = {commit: c.ours(f"five_pairs_pooled_rescored_{commit}") for commit in ("7f36e0b", "9406d60")}
results = {commit: compare(old, frame) for commit, frame in new.items()}
other_scores = {commit: {"unchanged_on_every_fold": r[4], "changed": r[5]} for commit, r in results.items()}

m, ratio, same, table, unchanged, changed = results["7f36e0b"]
left_out = sorted(set(old.dataset) - set(new["7f36e0b"].dataset))
print(f"7f36e0b: {len(new['7f36e0b'])} folds on {new['7f36e0b'].dataset.nunique()} tables; matched {len(m)}; CRPS identical on"
      f" {int(same.sum())}; largest relative difference {float((ratio - 1).abs().max()):.2e}")
print("tables not rescored:", left_out)
c.record("rescore", {"folds_rescored": len(new["7f36e0b"]), "tables_rescored": int(new["7f36e0b"].dataset.nunique()),
                     "folds_matched": len(m), "crps_identical": int(same.sum()), "tables_not_rescored": left_out,
                     "source": "results/five_pairs_pooled_rescored_7f36e0b.parquet (ScoringBench's code at 7f36e0b)"})

m, ratio, same, table, unchanged, changed = results["9406d60"]
up, down = ratio.idxmax(), ratio.idxmin()
section = {
    "commit": "9406d60", "folds_rescored": len(new["9406d60"]), "tables_rescored": int(new["9406d60"].dataset.nunique()),
    "folds_matched": len(m), "crps_identical": int(same.sum()),
    "crps_folds_higher": int(((ratio > 1) & ~same).sum()), "crps_folds_lower": int(((ratio < 1) & ~same).sum()),
    "crps_largest_increase": float(ratio.max() - 1), "crps_largest_increase_at": [m.dataset[up], int(m.fold[up])],
    "crps_largest_decrease": float(1 - ratio.min()), "crps_largest_decrease_at": [m.dataset[down], int(m.fold[down])],
    "crps_folds_moved_more_than_0_1pct": int(((ratio - 1).abs() > 1e-3).sum()),
    "crps_folds_moved_more_than_0_01pct": int(((ratio - 1).abs() > 1e-4).sum()),
    "crps_table_ratio_max": float(table.max()), "crps_table_ratio_max_dataset": table.idxmax(),
    "crps_table_ratio_min": float(table.min()), "crps_table_ratio_min_dataset": table.idxmin(),
    "crps_table_ratio_geo_mean": c.geo(table),
    "crps_tables_above_1_0005": sorted(table[table > 1.0005].index), "crps_tables_below_0_9995": sorted(table[table < 0.9995].index),
    "tables_not_rescored": sorted(set(old.dataset) - set(new["9406d60"].dataset)),
    "other_scores": other_scores}
print(f"9406d60: {len(m)} folds matched; CRPS identical on {section['crps_identical']}, higher on {section['crps_folds_higher']},"
      f" lower on {section['crps_folds_lower']}; largest increase {section['crps_largest_increase']:.4%}"
      f" ({section['crps_largest_increase_at'][0]} fold {section['crps_largest_increase_at'][1]}), largest decrease"
      f" {section['crps_largest_decrease']:.4%} ({section['crps_largest_decrease_at'][0]} fold {section['crps_largest_decrease_at'][1]});"
      f" moved by more than 0.1% on {section['crps_folds_moved_more_than_0_1pct']} folds and by more than 0.01% on"
      f" {section['crps_folds_moved_more_than_0_01pct']}")
print(f"9406d60 by table: ratio of mean CRPS from {table.min():.5f} ({table.idxmin()}) to {table.max():.5f} ({table.idxmax()}),"
      f" geometric mean {c.geo(table):.5f}; above 1.0005: {section['crps_tables_above_1_0005']}")
for commit, v in other_scores.items():
    print(f"{commit}, scores unchanged on every fold: {', '.join(v['unchanged_on_every_fold'])}")
    print(f"{commit}, scores changed (folds, largest relative change): "
          + "; ".join(f"{k} {d['folds']}, {d['largest_relative_change']:.3g}" for k, d in v["changed"].items()))
c.record("rescore_9406d60", section)
