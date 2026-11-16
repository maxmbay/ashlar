"""The tables that needed a local loading fix under pandas 3, and how they overlap the 12 few-values tables.

Under pandas 3.1, scikit-learn's ARFF reader failed on twelve of ScoringBench's tables. A local script read their
categorical columns as strings, as pandas 2 did, and then called ScoringBench's own loader, which cached each table.
results/loading_fix_tables.csv lists the twelve, with the time each table's processed copy was written to
ScoringBench's cache on the machine that ran the benchmark (8 November 2026, 00:28 to 00:35; the result files do not
record which tables went through the fix). The few-values tables come from results/target_values.csv.

Writes out/loading_fix_tables.csv and the section ``loading_fix`` of out/numbers.json.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402

import pandas as pd  # noqa: E402

fixed = pd.read_csv(c.RESULTS / "loading_fix_tables.csv").sort_values("processed_copy_written").table.tolist()
scored = set(c.dressed().dataset)
few = set(c.target_values().groupby("table").at_most_20_target_values.min().pipe(lambda s: s[s == 1]).index)
out = pd.DataFrame({"dataset": fixed, "among_the_99": [t in scored for t in fixed], "few_values": [t in few for t in fixed]})
c.write_csv("loading_fix_tables.csv", out)
print(out.to_string(index=False))
print(f"{len(fixed)} tables with the fix; {int(out.among_the_99.sum())} among the 99; overlap with the 12 few-values"
      f" tables: {sorted(set(fixed) & few)}")
c.record("loading_fix", {"tables": fixed, "n_tables": len(fixed), "n_among_the_99": int(out.among_the_99.sum()),
                         "not_among_the_99": sorted(set(fixed) - scored), "overlap_with_few_values": sorted(set(fixed) & few),
                         "source": "results/loading_fix_tables.csv (file times of ScoringBench's processed-data cache)"})
