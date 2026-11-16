"""Check that our loader is ScoringBench's, and that the published files reproduce its published leaderboard.

1. With ScoringBench's code at 3bf48c9 (SCORINGBENCH_CODE, with its requirements installed, torch among them): for
   every score column of the 63 published files and the derived ``amce``, ScoringBench's own ``load_metric_matrix``
   run on the published folder and ``common.metric_matrix`` run on the same files give the same matrix. Without
   that code, this part is skipped.
2. Ranking the 63 published models by CRPS with ``rank_with_autorank`` reproduces ScoringBench's published
   leaderboard file figures/leaderboard/cd_data_crps.json (every mean rank, the order, the critical difference).
3. The discrete-target flag of every table agrees between the published files and our two runs.
4. Both of our runs have 5 folds on each of the 99 tables, no missing CRPS, and the tables of the published files.

Writes out/check_loader.json and the section ``check_loader`` of out/numbers.json.
"""
import contextlib
import importlib.util
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402

import numpy as np  # noqa: E402

pub = c.published()
dr, pl = c.dressed(), c.pool()
metrics = sorted({col for df in pub.values() for col in df.select_dtypes("number").columns
                  if col not in ("fold", "index") and col not in c.NON_SCORES} | {"amce"})


def scoringbench_leaderboard():
    """ScoringBench's autorank_leaderboard.py from the reader's clone, or None if it is not there or will not import."""
    root = c.code_folder()
    if root is None:
        print("1. skipped: no clone of ScoringBench's code (analyze.py --code)")
        return None
    for rel in c.MANIFEST["ScoringBench"]["sha256"]:  # every file the import below loads, checked first
        c.code_file(rel)
    sys.path.insert(0, str(root))
    try:
        spec = importlib.util.spec_from_file_location("scoringbench_leaderboard", root / "autorank_leaderboard.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as error:  # e.g. torch is not installed
        print(f"1. skipped: ScoringBench's code did not import ({type(error).__name__})")
        return None
    if not module._DISCRETE_NAN_KEYS:
        print("1. skipped: ScoringBench's DENSITY_BASED_KEYS / PIT_KEYS did not import")
        return None
    return module


# 1. the matrices
lb = scoringbench_leaderboard()
checks, bad = {}, None
if lb is not None:
    for metric in metrics:
        target = c.coverage_target(metric)
        with contextlib.redirect_stdout(io.StringIO()):
            theirs, their_models = lb.load_metric_matrix(str(c.output_folder()), metric, coverage_target=target)
        ours, our_models, dropped, _ = c.metric_matrix(pub, metric, coverage_target=target)
        if theirs is None:
            checks[metric] = {"same": ours is None, "tables": 0, "models": 0}
            continue
        theirs = theirs.sort_index().sort_index(axis=1)
        ours = ours.sort_index().sort_index(axis=1)
        same = (list(theirs.index) == list(ours.index) and list(theirs.columns) == list(ours.columns)
                and np.allclose(theirs.values, ours.values, rtol=1e-12, atol=0))
        checks[metric] = {"same": bool(same), "tables": len(ours), "models": len(our_models), "models_dropped": dropped}
    bad = [m for m, v in checks.items() if not v["same"]]
    print(f"1. {len(metrics)} scores: our matrix equals ScoringBench's on {len(metrics) - len(bad)}; differs on {bad or 'none'}")

# 2. the published leaderboard
published_board = c.leaderboard()["autorank"]
table, result, pivot = c.ranked(pub, "crps")
mine = dict(zip(table.model, table.meanrank))
gaps = [abs(mine[m["name"]] - m["meanrank"]) for m in published_board["models"]]
same_order = [m["name"] for m in published_board["models"]] == list(table.model)
print(f"2. 63-model CRPS ranking: {len(pivot)} tables; largest mean-rank gap to the published leaderboard {max(gaps):.2e};"
      f" same order {same_order}; CD {result.cd:.4f} against published {published_board['cd']:.4f}")

# 3. discrete flags
flag = {}
for name, df in list(pub.items()) + [(c.DRESSED, dr), (c.POOL, pl)]:
    for t, f in df.groupby("dataset").discrete_y.agg(lambda s: bool(s.any())).items():
        flag.setdefault(t, set()).add(f)
disagree = sorted(t for t, f in flag.items() if len(f) > 1)
discrete = sorted(t for t, f in flag.items() if f == {True})
print(f"3. discrete-target flag: {len(discrete)} discrete tables; tables where files disagree: {disagree or 'none'}")

# 4. our runs
runs = {}
for name, df in ((c.DRESSED, dr), (c.POOL, pl)):
    folds = df.groupby("dataset").fold.nunique()
    runs[name] = {"rows": len(df), "tables": int(df.dataset.nunique()), "tables_with_5_folds": int((folds == 5).sum()),
                  "missing_crps": int(df.crps.isna().sum()),
                  "same_tables_as_published": set(df.dataset) == set(pub["tabm_d"].dataset)}
print("4.", runs)

c.write_json("check_loader.json", {
    "matrices_equal_on_every_score": (not bad) if bad is not None else None, "scores_checked": len(metrics),
    "per_score": checks,
    "published_crps_leaderboard_reproduced": bool(max(gaps) < 1e-9 and same_order),
    "max_meanrank_gap": max(gaps), "cd_ours": result.cd, "cd_published": published_board["cd"],
    "friedman_p_ours": result.pvalue, "friedman_p_published": published_board["pvalue"],
    "discrete_tables": discrete, "discrete_flag_disagreements": disagree, "runs": runs,
})
c.record("check_loader", {
    "matrices_equal_on_every_score": (not bad) if bad is not None else None,
    "published_crps_leaderboard_reproduced": bool(max(gaps) < 1e-9 and same_order),
    "n_discrete_tables": len(discrete),
    "dressed_folds": runs[c.DRESSED]["rows"], "pool_folds": runs[c.POOL]["rows"],
})
