"""How long the dressed model took to fit, fold by fold, by table size.

Table sizes are not in the result files, so they are read off the coverage columns: a fold's coverage values are
fractions k / n_test, and n_test is the smallest whole number that makes every coverage value of every model on that
fold a whole fraction. A table larger than 3,000 rows is cut to 2,400 training and 600 test rows per fold
(ScoringBench's SAMPLE_SIZE 3,000 with 5 folds); a smaller one is split five ways, so its rows are the sum of its five
test folds and a fold trains on the rest.

The run shared a 12-core laptop among six workers, was paused twice, and its workers were killed once. Its times are
wall-clock seconds (ScoringBench's ``time.perf_counter`` around ``fit``) under that load. ``predict_time`` also
includes saving the undressed pool's predictions, which the pool's entry reads. The published models' times come from
other hardware.

Writes out/table_sizes.csv, out/timing_folds.csv, out/timing_by_size.csv, out/timing_published.csv and the section
``timing`` of out/numbers.json.
"""
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

LEVELS = (20, 40, 60, 80, 90, 95)
pub = c.published()
dr, pl = c.dressed(), c.pool()
families = pd.read_csv(c.OUT / "families.csv").set_index("model").family.to_dict()
cov = pd.concat([df[["dataset", "fold"] + [f"coverage_{l}" for l in LEVELS]] for df in list(pub.values()) + [dr, pl]])


def n_test(values) -> int:
    """Smallest n with every value a whole multiple of 1/n (to 1e-9)."""
    values = np.unique(np.round(np.asarray(values, dtype=float), 12))
    values = values[np.isfinite(values)]
    n = 1
    for v in values:
        n = np.lcm(n, Fraction(float(v)).limit_denominator(100_000).denominator)
    if not all(abs(v * n - round(v * n)) < 1e-6 for v in values):
        raise ValueError("no common denominator")
    return int(n)


rows = []
for (t, f), g in cov.groupby(["dataset", "fold"]):
    rows.append({"dataset": t, "fold": int(f), "n_test": n_test(g.drop(columns=["dataset", "fold"]).values.ravel())})
sizes = pd.DataFrame(rows)
per_table = sizes.groupby("dataset").n_test.agg(["min", "max", "sum", "size"])
capped = per_table["min"] == 600
assert ((per_table["max"] - per_table["min"]) <= 1).all() or capped.any()
assert (per_table.loc[capped, "max"] == 600).all() and (per_table.loc[~capped, "max"] - per_table.loc[~capped, "min"] <= 1).all()
sizes["rows_in_table"] = sizes.dataset.map(lambda t: "more than 3,000" if capped[t] else int(per_table.loc[t, "sum"]))
sizes["n_train"] = [2400 if capped[t] else int(per_table.loc[t, "sum"]) - n for t, n in zip(sizes.dataset, sizes.n_test)]
sizes["discrete_y"] = sizes.dataset.isin(set(dr.query("discrete_y").dataset))
# Cross-check against the counts made on the benchmark's own splits (results/target_values.csv), which also give
# each table's column count.
ft = c.target_values().rename(columns={"table": "dataset", "training_rows": "train_rows"})
j = sizes.merge(ft, on=["dataset", "fold"], how="outer", indicator=True)
check = {"folds_in_both": int((j._merge == "both").sum()),
         "n_test_agree": int((j.n_test == j.test_rows).sum()), "n_train_agree": int((j.n_train == j.train_rows).sum())}
sizes = sizes.merge(ft[["dataset", "fold", "columns"]], on=["dataset", "fold"], how="left")
print("against results/target_values.csv:", check)
c.write_csv("table_sizes.csv", sizes)
print(f"{capped.sum()} tables capped at 2,400 training rows; the rest have {per_table.loc[~capped, 'sum'].min()} to"
      f" {per_table.loc[~capped, 'sum'].max()} rows; training rows per fold {sizes.n_train.min()} to {sizes.n_train.max()}")

t = dr[["dataset", "fold", "fit_time", "predict_time", "train_time"]].merge(sizes, on=["dataset", "fold"])
slowest = t.groupby("dataset").fit_time.median().sort_values(ascending=False).head(3)
widths = {d: (int(t[t.dataset == d]["columns"].iloc[0]) if "columns" in t else None) for d in slowest.index}
print("slowest tables by median fold fit time:", {d: round(v / 60, 1) for d, v in slowest.items()}, "columns:", widths)
t["fit_minutes"] = t.fit_time / 60
# A fold whose fit took far longer than the table's other folds: candidates for the pause and the restart.
t["fit_over_table_median_of_others"] = [
    row.fit_time / np.median(t[(t.dataset == row.dataset) & (t.fold != row.fold)].fit_time) for row in t.itertuples()]
c.write_csv("timing_folds.csv", t.sort_values(["n_train", "dataset", "fold"]), float_format="%.6g")
odd = t[t.fit_over_table_median_of_others > 2].sort_values("fit_over_table_median_of_others", ascending=False)
print("folds over twice their table's other folds:", odd[["dataset", "fold", "fit_minutes", "fit_over_table_median_of_others"]]
      .round(2).values.tolist())

bins = [0, 499, 999, 1999, 2399, 2400]
labels = ["under 500", "500 to 999", "1,000 to 1,999", "2,000 to 2,399", "2,400 (cut)"]
t["size_band"] = pd.cut(t.n_train, bins=bins, labels=labels)
band = t.groupby("size_band", observed=True).agg(folds=("fit_time", "size"), tables=("dataset", "nunique"),
                                                  median_fit_minutes=("fit_minutes", "median"),
                                                  min_fit_minutes=("fit_minutes", "min"),
                                                  max_fit_minutes=("fit_minutes", "max")).reset_index()
c.write_csv("timing_by_size.csv", band, float_format="%.4g")
print(band.round(1).to_string(index=False))
slope, intercept = np.polyfit(np.log(t.n_train), np.log(t.fit_time), 1)
print(f"fit time per fold: median {t.fit_minutes.median():.1f} min, from {t.fit_minutes.min():.1f} to {t.fit_minutes.max():.1f} min;"
      f" total {t.fit_time.sum() / 3600:.0f} worker-hours; log-log slope against training rows {slope:.2f}")
print(f"predict time per fold (includes saving the undressed pool's predictions): median {t.predict_time.median():.0f} s, max {t.predict_time.max() / 60:.1f} min")

# Published models' times, next to ours, with the hardware caveat.
pubt = []
for m, df in list(pub.items()) + [(c.DRESSED, dr)]:
    pubt.append({"model": m, "family": families.get(m, "untuned"), "folds": len(df),
                 "median_fit_seconds": df.fit_time.median(), "median_train_seconds": df.train_time.median(),
                 "max_train_seconds": df.train_time.max(), "total_train_hours": df.train_time.sum() / 3600})
pubt = pd.DataFrame(pubt).sort_values("median_train_seconds").reset_index(drop=True)
c.write_csv("timing_published.csv", pubt, float_format="%.6g")
print(pubt.round(1).to_string(index=False))
unt = pubt[pubt.model.isin(c.UNTUNED)]

c.record("timing", {
    "tables_capped_at_2400": int(capped.sum()), "tables_uncapped": int((~capped).sum()),
    "uncapped_rows_min": int(per_table.loc[~capped, "sum"].min()), "uncapped_rows_max": int(per_table.loc[~capped, "sum"].max()),
    "n_train_min": int(sizes.n_train.min()), "n_train_max": int(sizes.n_train.max()),
    "fit_minutes_median": float(t.fit_minutes.median()), "fit_minutes_min": float(t.fit_minutes.min()),
    "fit_minutes_max": float(t.fit_minutes.max()), "fit_minutes_p90": float(t.fit_minutes.quantile(0.9)),
    "fit_worker_hours_total": float(t.fit_time.sum() / 3600),
    "train_minutes_median": float((t.train_time / 60).median()),
    "predict_seconds_median": float(t.predict_time.median()),
    "loglog_slope_fit_time_vs_train_rows": float(slope),
    "by_size_band": {r.size_band: {"folds": int(r.folds), "median_min": round(float(r.median_fit_minutes), 2),
                                   "min_min": round(float(r.min_fit_minutes), 2), "max_min": round(float(r.max_fit_minutes), 2)}
                     for r in band.itertuples()},
    "folds_over_twice_table_median": [[r.dataset, int(r.fold), round(float(r.fit_over_table_median_of_others), 2)]
                                      for r in odd.itertuples()],
    "dressed_median_train_seconds": float(pubt.set_index("model").loc[c.DRESSED, "median_train_seconds"]),
    "slowest_untuned_published_by_median_train": unt.model.iloc[-1],
    "slowest_untuned_published_median_train_seconds": float(unt.median_train_seconds.iloc[-1]),
    "tabm_tuned_median_train_seconds": float(pubt.set_index("model").loc[c.TABM_TUNED, "median_train_seconds"]),
    "realmlp_tuned_median_train_seconds": float(pubt.set_index("model").loc[c.REALMLP_TUNED, "median_train_seconds"]),
    "sizes_vs_fold_targets_99": check,
    "slowest_tables_median_fit_minutes": {d: round(v / 60, 1) for d, v in slowest.items()}, "slowest_tables_columns": widths,
})
