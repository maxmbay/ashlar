"""The dressed model against its own undressed pool (the five members pooled, from the same fits).

By table: the ratio of the two models' mean CRPS over the table's five folds (both have all 495 folds), and as a
second reading the geometric mean of the five per-fold ratios. By fold: the 495 per-fold ratios. Below 1 means the
dressed model is lower. Also: where the pool would place among the untuned models in the dressed model's stead,
and with both in the field (ScoringBench's autorank).

Writes out/pool_by_table.csv, out/pool_by_fold.csv and the section ``pool`` of out/numbers.json.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

dr, pl = c.dressed(), c.pool()
m = dr[["dataset", "fold", "crps"]].merge(pl[["dataset", "fold", "crps"]], on=["dataset", "fold"], suffixes=("_dressed", "_pool"))
assert len(m) == 495
m["ratio"] = m.crps_dressed / m.crps_pool
c.write_csv("pool_by_fold.csv", m)

t = m.groupby("dataset").agg(dressed=("crps_dressed", "mean"), pool=("crps_pool", "mean"),
                             geo_fold_ratio=("ratio", lambda r: c.geo(r)), folds=("ratio", "size")).reset_index()
t["ratio"] = t.dressed / t.pool
t = t.sort_values("ratio").reset_index(drop=True)
c.write_csv("pool_by_table.csv", t)


def bands(r):
    r = np.asarray(r)
    return {"lower_by_more_than_1pct": int((r < 0.99).sum()), "within_1pct": int(((r >= 0.99) & (r <= 1.01)).sum()),
            "higher_by_more_than_1pct": int((r > 1.01).sum()), "higher_at_all": int((r > 1).sum()),
            "lower_at_all": int((r < 1).sum()), "equal": int((r == 1).sum())}


table_bands, fold_bands = bands(t.ratio), bands(m.ratio)
alt_bands = bands(t.geo_fold_ratio)
w_tab = stats.wilcoxon(np.log(t.ratio)).pvalue
w_fold = stats.wilcoxon(np.log(m.ratio[m.ratio != 1])).pvalue
worse = m[m.ratio > 1.01].sort_values("ratio", ascending=False)
print(f"by table: geo-mean of table ratios {c.geo(t.ratio):.4f}, median {t.ratio.median():.4f}; {table_bands}")
print(f"by table, geo-mean of fold ratios per table: overall {c.geo(t.geo_fold_ratio):.4f}; {alt_bands}")
print(f"by fold: geo-mean {c.geo(m.ratio):.4f}, median {m.ratio.median():.4f}; {fold_bands}; largest {m.ratio.max():.4f},"
      f" smallest {m.ratio.min():.4f}")
print("folds worse by more than 1%:", worse[["dataset", "fold", "ratio"]].round(4).values.tolist())
print(f"Wilcoxon signed-rank on log ratios: tables p {w_tab:.2e}, folds p {w_fold:.2e}")
print("tables most improved:", t.head(5)[["dataset", "ratio"]].round(3).values.tolist())
print("tables highest ratio:", t.tail(5)[["dataset", "ratio"]].round(4).values.tolist())

# Where the pool places among the untuned models.
pub = c.published()
base = {m_: pub[m_] for m_ in c.UNTUNED}
table_pool, res_pool, _ = c.ranked(base | {c.POOL: pl}, "crps")
table_both, res_both, _ = c.ranked(base | {c.POOL: pl, c.DRESSED: dr}, "crps")
others = table_pool[table_pool.model != c.POOL]
print(f"pool in the dressed model's stead (21 models): mean rank {c.meanrank(table_pool, c.POOL):.4f}, position"
      f" {c.position(table_pool, c.POOL)}, next {others.model.iloc[0]} {others.meanrank.iloc[0]:.4f}, CD {res_pool.cd:.4f}")
print(f"both (22 models): dressed {c.meanrank(table_both, c.DRESSED):.4f}, pool {c.meanrank(table_both, c.POOL):.4f}, CD {res_both.cd:.4f}")
pool_field, res_field, _ = c.ranked(pub | {c.POOL: pl}, "crps")
print(f"pool in the 64-model field: position {c.position(pool_field, c.POOL)}, mean rank {c.meanrank(pool_field, c.POOL):.4f}")

c.record("pool", {
    "geo_ratio_tables": c.geo(t.ratio), "median_ratio_tables": float(t.ratio.median()), "table_bands": table_bands,
    "geo_ratio_tables_from_fold_ratios": c.geo(t.geo_fold_ratio), "table_bands_from_fold_ratios": alt_bands,
    "geo_ratio_folds": c.geo(m.ratio), "median_ratio_folds": float(m.ratio.median()), "fold_bands": fold_bands,
    "max_fold_ratio": float(m.ratio.max()), "min_fold_ratio": float(m.ratio.min()),
    "max_table_ratio": float(t.ratio.max()), "max_table_ratio_dataset": t.dataset.iloc[-1],
    "min_table_ratio": float(t.ratio.min()), "min_table_ratio_dataset": t.dataset.iloc[0],
    "folds_worse_by_more_than_1pct": [[r.dataset, int(r.fold), round(float(r.ratio), 4)] for r in worse.itertuples()],
    "wilcoxon_p_tables": float(w_tab), "wilcoxon_p_folds": float(w_fold),
    "pool_mean_rank_among_21": c.meanrank(table_pool, c.POOL), "pool_position_among_21": c.position(table_pool, c.POOL),
    "pool_next_model": others.model.iloc[0], "pool_next_mean_rank": float(others.meanrank.iloc[0]),
    "pool_gap_to_next": float(others.meanrank.iloc[0] - c.meanrank(table_pool, c.POOL)), "cd_21": float(res_pool.cd),
    "with_both_dressed_mean_rank": c.meanrank(table_both, c.DRESSED), "with_both_pool_mean_rank": c.meanrank(table_both, c.POOL),
    "pool_position_in_64_field": c.position(pool_field, c.POOL), "pool_mean_rank_in_64_field": c.meanrank(pool_field, c.POOL),
})
