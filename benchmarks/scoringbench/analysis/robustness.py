"""The CRPS rankings under other readings of the same data.

Readings, for the 21 untuned models and for the 64-model field:
  as published   each model's mean over the folds it has on a table (ScoringBench's rule);
  common folds   only the folds every model in the comparison has (per-table means over those folds; a table with
                 no such fold drops out);
  98 tables      without Brazilian_houses, which ScoringBench dropped after the run (target leakage);
  both           common folds on the 98 tables.
Every reading is ranked with ScoringBench's ``rank_with_autorank``. Also lists which published models lack which folds.

Writes out/robustness.csv, out/missing_folds.csv and the section ``robustness`` of out/numbers.json.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402

import pandas as pd  # noqa: E402

pub = c.published()
dr = c.dressed()
families = pd.read_csv(c.OUT / "families.csv").set_index("model").family.to_dict()

# Which folds are missing, model by model.
missing = []
for m, df in pub.items():
    have = set(zip(df.dataset, df.fold))
    for t in sorted(df.dataset.unique()):
        gone = [f for f in range(5) if (t, f) not in have]
        if gone:
            missing.append({"model": m, "family": families[m], "dataset": t, "missing_folds": " ".join(map(str, gone))})
missing = pd.DataFrame(missing)
c.write_csv("missing_folds.csv", missing)
pattern = missing.groupby("model").apply(lambda g: "; ".join(f"{r.dataset} {r.missing_folds}" for r in g.itertuples()),
                                         include_groups=False)
groups = pattern.reset_index().groupby(0)["model"].apply(list)
for p, models in groups.items():
    print(f"{len(models):2d} model(s) lack folds [{p}]: {', '.join(models) if len(models) < 6 else str(len(models)) + ' models'}")
print("models with all 495 folds:", sorted(set(pub) - set(missing.model)))


def common_frames(frames, tables=None):
    """Each frame restricted to the folds every frame has (and to the given tables)."""
    keys = None
    for df in frames.values():
        k = set(zip(df.dataset, df.fold))
        keys = k if keys is None else keys & k
    if tables is not None:
        keys = {k for k in keys if k[0] in set(tables)}
    out = {m: df[[k in keys for k in zip(df.dataset, df.fold)]] for m, df in frames.items()}
    return out, keys


rows = []
all_tables = sorted(dr.dataset.unique())
t98 = [t for t in all_tables if t != c.LEAKY_TABLE]
for field_name, frames in (("21 untuned", {m: pub[m] for m in c.UNTUNED} | {c.DRESSED: dr}),
                           ("64 field", pub | {c.DRESSED: dr})):
    for reading in ("as published", "common folds", "98 tables", "both"):
        tables = t98 if reading in ("98 tables", "both") else None
        if reading in ("common folds", "both"):
            use, keys = common_frames(frames, tables)
            n_folds = len(keys)
        else:
            use = frames
            n_folds = sum(len(df[df.dataset.isin(tables or all_tables)]) for df in frames.values())
        table, result, pivot = c.ranked(use, "crps", tables=tables)
        others = table[table.model != c.DRESSED]
        me = c.meanrank(table, c.DRESSED)
        row = {"field": field_name, "reading": reading, "tables": len(pivot), "models": pivot.shape[1],
               "folds": n_folds if reading in ("common folds", "both") else None,
               "dressed_mean_rank": me, "dressed_position": c.position(table, c.DRESSED),
               "next_model": others.model.iloc[0], "next_mean_rank": float(others.meanrank.iloc[0]), "cd": float(result.cd),
               "friedman_p": float(result.pvalue), "test": f"{result.omnibus}/{result.posthoc}"}
        if field_name == "21 untuned":
            row["gap_to_next"] = row["next_mean_rank"] - me
            row["n_beyond_one_cd"] = int((others.meanrank - me > result.cd).sum())
            row["tables_first"] = int((pivot.rank(axis=1)[c.DRESSED] == 1).sum())
        else:
            ahead = table[table.meanrank < me]
            row["all_ahead_pretrained_or_finetuned"] = bool(ahead.model.map(families).isin(["pretrained", "fine-tuned"]).all())
            row["realmlp_tuned_mean_rank"] = c.meanrank(table, c.REALMLP_TUNED)
            row["tabm_tuned_mean_rank"] = c.meanrank(table, c.TABM_TUNED)
        rows.append(row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})

# One more reading: the dressed model's folds cut to those any untuned published model has.
union = set()
for m in c.UNTUNED:
    union |= set(zip(pub[m].dataset, pub[m].fold))
cut = dr[[k in union for k in zip(dr.dataset, dr.fold)]]
table, result, pivot = c.ranked({m: pub[m] for m in c.UNTUNED} | {c.DRESSED: cut}, "crps")
others = table[table.model != c.DRESSED]
row = {"field": "21 untuned", "reading": "dressed cut to folds the untuned field has", "tables": len(pivot), "models": 21,
       "folds": len(cut), "dressed_mean_rank": c.meanrank(table, c.DRESSED), "dressed_position": c.position(table, c.DRESSED),
       "next_model": others.model.iloc[0], "next_mean_rank": float(others.meanrank.iloc[0]), "cd": float(result.cd),
       "friedman_p": float(result.pvalue), "test": f"{result.omnibus}/{result.posthoc}",
       "gap_to_next": float(others.meanrank.iloc[0]) - c.meanrank(table, c.DRESSED),
       "n_beyond_one_cd": int((others.meanrank - c.meanrank(table, c.DRESSED) > result.cd).sum()),
       "tables_first": int((pivot.rank(axis=1)[c.DRESSED] == 1).sum())}
rows.append(row)
print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})

rob = pd.DataFrame(rows)
c.write_csv("robustness.csv", rob, float_format="%.6g")
num = {}
for r in rob.itertuples():
    key = f"{r.field.replace(' ', '_')}__{r.reading.replace(' ', '_')}"
    num[key] = {k: v for k, v in r._asdict().items() if k not in ("Index", "field", "reading") and v == v and v is not None}
num["missing_fold_patterns"] = {p: models for p, models in groups.items()}
c.record("robustness", num)
