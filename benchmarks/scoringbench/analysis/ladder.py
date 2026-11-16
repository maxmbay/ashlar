"""What each part adds on ScoringBench: Ashlar's earlier entries and the dressed model, each ranked alone among the
same 20 untuned published models (21 models, 99 tables, CRPS, ScoringBench's autorank), and the CRPS ratio between
each step and the one before it.

The steps: one network trained for at most 100 epochs; one network with the longer training; that network pooled
with one label reader; the five cross-fitted members pooled, not dressed (the dressed run's undressed pool); the
dressed model. The first three come from research code and earlier runs; the last two from the reported run. All
five are in results/.

Every step is cut to the 99 tables of the reported run before it is ranked or compared. The first run's file also
holds a 100th table (3D_Estimation_using_RSSI_of_WLAN_dataset_complete_1_target, 5 folds) that no published model
has; ScoringBench's completeness rule would drop it anyway, so the ranking is the same either way. Counts of tables
lower and higher are given both by more than 1% (the table) and at all (the csv).

Writes out/ladder.csv, out/tab_ladder.tex and the section ``ladder`` of out/numbers.json.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

pub = c.published()
base = {m: pub[m] for m in c.UNTUNED}
early = lambda name: c._numeric(pd.read_parquet(c.RESULTS / f"{name}.parquet"))  # noqa: E731
TABLES = set(c.dressed().dataset)
steps = [("One network, at most 100 epochs", early("one_network_100_epochs")),
         ("One network", early("one_network")),
         ("One network pooled with one label reader", early("one_network_one_reader")),
         ("Five cross-fitted members, not dressed", c.pool()),
         ("Dressed model", c.dressed())]
dropped = {label: sorted(set(frame.dataset) - TABLES) for label, frame in steps}
steps = [(label, frame[frame.dataset.isin(TABLES)].reset_index(drop=True)) for label, frame in steps]
assert all(len(frame) == 495 and frame.dataset.nunique() == 99 for _, frame in steps)
print("tables outside the 99, dropped before ranking:", {k: v for k, v in dropped.items() if v})

rows, previous = [], None
for label, frame in steps:
    table, res, _ = c.ranked(base | {label: frame}, "crps")
    others = table[table.model != label]
    mean = frame.groupby("dataset").crps.mean()
    row = {"step": label, "tables": int(frame.dataset.nunique()), "folds": int(len(frame)),
           "mean_rank": c.meanrank(table, label), "position": c.position(table, label),
           "best_other": others.model.iloc[0], "best_other_mean_rank": float(others.meanrank.iloc[0]),
           "lead": float(others.meanrank.iloc[0] - c.meanrank(table, label)), "cd": float(res.cd)}
    if previous is not None:
        r = (mean / previous).dropna()
        row |= {"ratio_to_previous_geo": c.geo(r), "ratio_to_previous_median": float(r.median()),
                "lower_by_more_than_1pct": int((r < 0.99).sum()), "within_1pct": int(((r >= 0.99) & (r <= 1.01)).sum()),
                "higher_by_more_than_1pct": int((r > 1.01).sum()), "lower_at_all": int((r < 1).sum()),
                "higher_at_all": int((r > 1).sum()), "largest_ratio": float(r.max()), "tables_compared": int(len(r))}
    rows.append(row)
    previous = mean
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})

t = pd.DataFrame(rows)
c.write_csv("ladder.csv", t)
body = []
for r in t.itertuples():
    ratio = "" if pd.isna(getattr(r, "ratio_to_previous_geo", np.nan)) else c.fmt(r.ratio_to_previous_geo, 3)
    wins = "" if ratio == "" else f"{int(r.lower_by_more_than_1pct)} / {int(r.higher_by_more_than_1pct)}"
    lead = c.fmt(r.lead) if r.lead >= 0 else "$-$" + c.fmt(-r.lead)  # a true minus sign
    body.append([r.step, c.fmt(r.mean_rank), str(r.position), c.fmt(r.best_other_mean_rank), lead, ratio, wins])
c.write_tex("tab_ladder.tex", "lrrrrrr",
            ["Model", "Mean rank", "Place", "Next model", "Lead", "CRPS ratio", "Lower / higher"], body,
            notes="Each row ranked alone among the 20 untuned published models (99 tables, CRPS, ScoringBench's"
                  " autorank). Next model: the best other mean rank (condensite's in every row); Lead: that minus the"
                  " row's. CRPS ratio: geometric mean over the 99 tables of the row's mean CRPS over the row above's;"
                  " Lower / higher: tables lower / higher by more than 1%. Script: analysis/ladder.py")
c.record("ladder", {"steps": t.to_dict(orient="records"), "tables_dropped_before_ranking": {k: v for k, v in dropped.items() if v}})
