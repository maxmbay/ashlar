"""The smaller numbers of the development record: the run's folds, the first run's mixture head, the development
tables, the design tests on them before the dressed model, the two tables whose saved pool predictions were
overwritten, and the quantile levels the wrapper passes to ScoringBench.

Sources: results/ (the per-fold files and target_values.csv), results/development/ (tables.csv, training_length,
reader_prototype, one_pair_fold0, one_pair_checks, five_pairs, solved_last_layer, network_on_reader), and
ashlar_dressed_wrapper.py. Unless a count says otherwise, "better" and "worse" mean by more than 1%.

The design tests, in the order they were run, each with the pass line its script applied:
  training length     a network's best checkpoint within 100 epochs, against longer training (at most 1,000 epochs,
                      patience 16) with a three-epoch cooldown, folds 0 to 2. No line for adopting the longer training;
                      its line kept the cooldown unless it raised the geometric mean CRPS, and the cap of 1,000 epochs
                      unless more than 10% of fits ran to it.
  reader prototype    fold 0 of six tables. K1: the reader within 10% of the network on at least 4 of 6, and better on
                      the few-values table. K2: the reader fitted to synthetic targets only at least 10% better than
                      nearest neighbours on at least 4 of 6, and the reader fitted to synthetic targets and then the
                      real labels at least 3% better than the reader on at least 2 of the 3 small tables.
  solved last layer   fold 0 of ten tables: re-solving the network's last layer as a ridge regression at least 1%
                      better, and more than 0.5% worse on at most two tables.
  one pair, fold 0    one network pooled with one reader: at least 3% better than the network, worse on at most two.
  rewrite check       the library's rewrite of the reader against the prototype, fold 0: within 1% on every table.
  one pair, folds 1-2 at least 3% better than the network, worse on at most four of the 46 fits.
  five pairs          five cross-fitted pairs pooled against the mean of the five taken singly, fold 0: at least 2%
                      better, better on at least 12 tables, worse on at most two.
  network on reader   a network fitted on top of the reader's prediction, in two forms (one reader; the five
                      cross-fitted readers), fold 0: at least 1% below one network pooled with one reader, worse on
                      at most two.
The tests of the dressed model's own parts are in devset_tests.py.

Writes out/record.json and the section ``record`` of out/numbers.json.
"""
import ast
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

out = {}
frame = lambda name: pd.DataFrame(c.development(name))  # noqa: E731


def counts(ratio, margin=0.01):
    r = np.asarray(ratio, dtype=float)
    return {"n": len(r), "geo_mean": c.geo(r), "better": int((r < 1 - margin).sum()), "worse": int((r > 1 + margin).sum()),
            "lower_at_all": int((r < 1).sum()), "higher_at_all": int((r > 1).sum())}


# ---------------------------------------------------------------- the run
dr = c.dressed()
out["run"] = {"folds": len(dr), "distinct_table_folds": int(len(dr.drop_duplicates(["dataset", "fold"]))),
              "tables": int(dr.dataset.nunique()), "folds_per_table": sorted(set(dr.groupby("dataset").fold.nunique()))}
print("the run:", out["run"])

# ---------------------------------------------------------------- the first run's mixture head
TABLES = set(dr.dataset)
mix = c.ours("mixture_head_100_epochs")
mix = mix[mix.dataset.isin(TABLES)].reset_index(drop=True)
pub = c.published()
table, res, pivot = c.ranked({m: pub[m] for m in c.UNTUNED} | {"mixture_head_100_epochs": mix}, "crps")
others = table[table.model != "mixture_head_100_epochs"]
out["mixture_head"] = {"tables": len(pivot), "mean_rank": c.meanrank(table, "mixture_head_100_epochs"),
                       "position": c.position(table, "mixture_head_100_epochs"), "first": table.model.iloc[0],
                       "first_mean_rank": float(table.meanrank.iloc[0]), "best_other": others.model.iloc[0],
                       "best_other_mean_rank": float(others.meanrank.iloc[0]), "cd": float(res.cd)}
print("first run, mixture head, among the 20 untuned:", out["mixture_head"])

# ---------------------------------------------------------------- the development tables
tables = pd.read_csv(c.DEVELOPMENT / "tables.csv")
parts = frame("dressed_and_parts")
assert set(tables.table) == set(parts.table)
binned = tables[tables.source == "binned copy"]
kind = parts[parts.fold == 0].set_index("table").kind
files = ["dressed_and_parts", "library_check", "network_on_reader", "five_pairs", "training_length", "one_pair_fold0",
         "one_pair_checks", "reader_prototype", "solved_last_layer"]
folds_used = sorted({int(f) for name in files for f in frame(name).get("fold", pd.Series(dtype=float)).dropna()})
base = {re.sub(r"_rated\d+$", "", t) for t in tables.table}
out["development_tables"] = {
    "n": len(tables), "by_source": tables.source.value_counts().to_dict(), "distinct_tables": len(base),
    "ratings_min": int(binned.ratings.min()), "ratings_max": int(binned.ratings.max()),
    "at_most_20_values": int((kind == "ordinal").sum()), "at_most_20_values_binned": int(kind[binned.table].eq("ordinal").sum()),
    "results_in_training_length_test": len(frame("training_length")), "folds_in_any_file": folds_used,
    "among_the_99_by_name": sorted((set(tables.table) | base) & TABLES)}
print("development tables:", out["development_tables"])

# ---------------------------------------------------------------- training length
tl = frame("training_length")
fits = tl.crps_longer_training_with_cooldown / tl.crps_at_most_100_epochs
means = tl.groupby("table")[["crps_at_most_100_epochs", "crps_longer_training_with_cooldown"]].mean()
per_table = means.crps_longer_training_with_cooldown / means.crps_at_most_100_epochs
out["training_length"] = {
    "tables": counts(per_table), "fits": counts(fits),
    "cooldown_against_none_geo_over_fits": c.geo(tl.crps_longer_training_with_cooldown / tl.crps_longer_training),
    "best_epoch_past_100": int((tl.best_epoch > 100).sum()), "ran_to_1000_epochs": int((tl.epochs_run >= 1000).sum())}
print("training length, longer with cooldown / at most 100 epochs:", out["training_length"])

# ---------------------------------------------------------------- reader prototype (K1, K2)
rp = frame("reader_prototype").set_index("table")
SMALL = ["560_bodyfat", "557_analcatdata_apnea1", "cars"]  # the prototype script's three small tables
few = [t for t in rp.index if t in set(binned.table)]
k1 = rp.reader / rp.network
k2a = rp.reader_synthetic_only / rp.nearest_neighbours
k2b = (rp.reader_synthetic_then_real / rp.reader).loc[SMALL]
out["reader_prototype"] = {
    "tables": len(rp), "k1_within_10pct_of_network": int((k1 <= 1.10).sum()), "k1_few_values_table": few,
    "k1_few_values_ratio": float(k1.loc[few].iloc[0]), "k1_passes": bool((k1 <= 1.10).sum() >= 4 and (k1.loc[few] < 1).all()),
    "k2_synthetic_only_10pct_better_than_neighbours": int((k2a <= 0.90).sum()),
    "k2_synthetic_then_real_3pct_better_on_small": int((k2b <= 0.97).sum()), "k2_small_tables": SMALL,
    "k2_synthetic_then_real_over_reader_small": {t: float(v) for t, v in k2b.items()},
    "k2_passes": bool((k2a <= 0.90).sum() >= 4 and (k2b <= 0.97).sum() >= 2)}
print("reader prototype:", out["reader_prototype"])

# ---------------------------------------------------------------- solved last layer
sl = frame("solved_last_layer")
r = sl.network_with_solved_layer / sl.network
out["solved_last_layer"] = {"tables": len(sl), "geo_mean": c.geo(r), "worse_by_more_than_half_pct": int((r > 1.005).sum()),
                            "passes": bool(c.geo(r) <= 0.99 and (r > 1.005).sum() <= 2)}
print("solved last layer:", out["solved_last_layer"])

# ---------------------------------------------------------------- one network pooled with one reader
p0 = frame("one_pair_fold0")
out["one_pair_fold0"] = counts(p0.one_pair_pooled / p0.network)
checks = frame("one_pair_checks")
rw = checks[checks.check == "rewrite_against_prototype"]
off = ((rw.label_reader - 1).abs() > 0.01) | ((rw.one_pair_pooled - 1).abs() > 0.01)
close = ((rw.label_reader - 1).abs() <= 0.001) & ((rw.one_pair_pooled - 1).abs() <= 0.001)
out["rewrite_check"] = {"tables": len(rw), "off_by_more_than_1pct": int(off.sum()), "within_0_1pct": int(close.sum()),
                        "reader_off_by_more_than_1pct": int(((rw.label_reader - 1).abs() > 0.01).sum()),
                        "pool_off_by_more_than_1pct": int(((rw.one_pair_pooled - 1).abs() > 0.01).sum())}
cf = checks[checks.check == "folds_1_and_2"].reset_index(drop=True)
ratio = cf.one_pair_pooled / cf.network
worse = cf[ratio > 1.01]
out["one_pair_folds12"] = counts(ratio) | {
    "worse_fits": [{"table": r.table, "fold": int(r.fold), "rows": int(r.rows)} for r in worse.itertuples()],
    "worse_fits_max_rows": int(worse.rows.max())}
print("one pair, fold 0:", out["one_pair_fold0"])
print("rewrite against the prototype, fold 0:", out["rewrite_check"])
print("one pair, folds 1 and 2:", out["one_pair_folds12"])

# ---------------------------------------------------------------- five cross-fitted pairs
fp = frame("five_pairs")
out["five_pairs"] = {"against_taken_singly": counts(fp.five_pairs_pooled / fp.five_pairs_taken_singly),
                     "against_one_pair": counts(fp.five_pairs_pooled / fp.one_pair_pooled)}
print("five pairs pooled:", out["five_pairs"])

# ---------------------------------------------------------------- a network on top of the reader
nr = frame("network_on_reader").set_index("table")
ref = fp.set_index("table").loc[nr.index]
out["network_on_reader"] = {form: {"against_one_pair": counts(nr[form] / ref.one_pair_pooled),
                                   "against_five_pairs_pooled": counts(nr[form] / ref.five_pairs_pooled)}
                            for form in ("network_on_reader", "network_on_five_readers")}
print("network on the reader:", out["network_on_reader"])

# ---------------------------------------------------------------- the figure of docs/dressed.md, "How it was tested"
# One row per table, sorted by the mean log ratio of the dressed model's CRPS to one network's over folds 1 and 2; each
# fold is a dot (the dressed model) and a ring (the five members pooled, not dressed), both relative to the network.
held = parts[parts.fold.isin([1, 2])].assign(dressed_ratio=lambda d: d.dressed / d.network,
                                             pool_ratio=lambda d: d.five_pairs_pooled / d.network)
order = held.groupby("table").dressed_ratio.apply(lambda s: float(np.mean(np.log(s)))).sort_values().index
dots = {t: held[held.table == t].dressed_ratio for t in order}
rings = {t: held[held.table == t].pool_ratio for t in order}
last5 = list(order[-5:])
out["guide_figure"] = {
    "tables_with_a_dot_at_or_below_0_4": sorted(t for t in order if (dots[t] <= 0.4).any()),
    "tables_with_both_dots_from_0_5_to_0_95": int(sum(dots[t].between(0.5, 0.95).all() for t in order)),
    "tables_with_a_dot_from_0_5_to_0_95": int(sum(dots[t].between(0.5, 0.95).any() for t in order)),
    "last_five_tables": last5,
    "last_five_with_both_dots_within_2pct": [t for t in last5 if (dots[t] - 1).abs().le(0.02).all()],
    "fits_with_ring_right_of_dot": int((held.pool_ratio > held.dressed_ratio).sum()),
    "tables_with_ring_right_of_dot_on_both_folds": int(sum((rings[t] > dots[t]).all() for t in order)),
    "largest_ring": float(held.pool_ratio.max()),
    "largest_ring_at": [held.loc[held.pool_ratio.idxmax(), "table"], int(held.loc[held.pool_ratio.idxmax(), "fold"])]}
print("the guide's figure:", out["guide_figure"])

# ---------------------------------------------------------------- the two tables whose saved pool files were overwritten
# The run saved each fold's pool predictions in a file named by a hash of the fold's training labels and number of test
# rows. These pairs of tables have the same training labels on every fold, so each pair's files share names.
PAIRS = [("227_cpu_small", "cpu_act"), ("yprop_4_1", "topo_2_1")]
pl = c.pool().set_index(["dataset", "fold"]).crps
dd = dr.set_index(["dataset", "fold"]).crps
tv = c.target_values().set_index(["table", "fold"])
shared = {}
for a, b in PAIRS:
    same_shape = bool((tv.loc[a][["training_rows", "test_rows", "distinct_target_values_in_training_rows"]]
                       == tv.loc[b][["training_rows", "test_rows", "distinct_target_values_in_training_rows"]]).all().all())
    differ = ~np.isclose(pl.loc[a].values, pl.loc[b].values, rtol=1e-9, atol=0)
    own = pl.loc[a] / dd.loc[a]
    shared[a] = {"shares_labels_with": b, "same_rows_and_value_counts_every_fold": same_shape,
                 "folds": len(differ), "folds_where_pool_differs_from_partner": int(differ.sum()),
                 "pool_crps_range": [float(pl.loc[a].min()), float(pl.loc[a].max())],
                 "partner_pool_crps_range": [float(pl.loc[b].min()), float(pl.loc[b].max())],
                 "pool_over_own_dressed_range": [float(own.min()), float(own.max())]}
out["shared_label_tables"] = shared
print("tables whose saved pool files were overwritten:", shared)

# ---------------------------------------------------------------- the quantile levels the wrapper passes
tree = ast.parse((c.FOLDER / "ashlar_dressed_wrapper.py").read_text())
expr = next(n.value for n in tree.body if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "LEVELS" for t in n.targets))
levels = eval(compile(ast.Expression(expr), "ashlar_dressed_wrapper.py", "eval"), {"np": np})
out["quantile_levels"] = {"n": len(levels), "lowest": float(levels[0]), "highest": float(levels[-1]),
                          "increasing": bool(np.all(np.diff(levels) > 0))}
print("quantile levels:", out["quantile_levels"])

c.write_json("record.json", out)
c.record("record", out)
