"""Reproduce the ScoringBench numbers for Ashlar's models from the files in results/ and ScoringBench's published
results.

    python analyze.py --published PATH [--code PATH] [--out PATH] > RESULTS.md

--published  a clone of ScoringBenchOutput at commit b590a95, with its Git LFS files fetched (or its folder
             output_univariate_scoringbench_d1_n3000).
--code       optional: a clone of ScoringBench's code at commit 3bf48c9. With it, the model families get the file
             and line of their evidence and the organizations from ScoringBench's models.json, and the loader check
             runs ScoringBench's own loader on the published files (that part also needs ScoringBench's
             requirements, torch among them).
--out        where the tables, numbers and figures go (default: out/ next to this file).

Every ScoringBench file used is first checked against its SHA-256 sum in published_files.json. The scripts in
analysis/ then run in order, in one process. Each writes its tables to the output folder and its headline numbers to
numbers.json there. The printed Markdown starts with the numbers quoted in Ashlar's README and docs/dressed.md, then
gives each script's own output. RESULTS.md is this output.
"""
import argparse
import contextlib
import io
import os
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE / "analysis"
STEPS = ["untuned_crps", "check_loader", "families", "full_field", "head_to_head", "by_table", "pool", "rescore_check",
         "robustness", "all_scores", "calibration", "timing", "devset_tests", "ladder", "loading_fix_tables", "record",
         "fig_untuned_ranks", "fig_full_field", "fig_table_ratios", "fig_coverage"]

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--published", required=True, type=Path, help="a ScoringBenchOutput clone at commit b590a95")
parser.add_argument("--code", type=Path, help="a ScoringBench clone at commit 3bf48c9 (optional)")
parser.add_argument("--out", type=Path, help="output folder (default: out/)")
args = parser.parse_args()
os.environ["SCORINGBENCH_OUTPUT"] = str(args.published.resolve())
if args.code:
    os.environ["SCORINGBENCH_CODE"] = str(args.code.resolve())
else:
    os.environ.pop("SCORINGBENCH_CODE", None)
if args.out:
    os.environ["SCORINGBENCH_ANALYSIS_OUT"] = str(args.out.resolve())
sys.path.insert(0, str(ANALYSIS))
import common as c  # noqa: E402

problems = c.check_published()
if args.code:
    for rel in c.MANIFEST["ScoringBench"]["sha256"]:
        try:
            c.code_file(rel)
        except SystemExit as error:
            problems.append(str(error))
if problems:
    sys.exit("These files do not match published_files.json:\n  " + "\n  ".join(problems))
(c.OUT / "numbers.json").unlink(missing_ok=True)


def run(name: str) -> str:
    """Run analysis/<name>.py and return what it printed."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        runpy.run_path(str(ANALYSIS / f"{name}.py"), run_name="__main__")
    return buffer.getvalue().rstrip()


printed = {name: run(name) for name in STEPS}
summary = run("summary")


def first_line(name: str) -> str:
    text = (ANALYSIS / f"{name}.py").read_text()
    doc = text.split('"""')[1] if text.startswith('"""') else ""
    return " ".join(doc.split("\n\n")[0].split())


code = "with" if args.code else "without"
print("# ScoringBench results for Ashlar's models", "",
      "Printed by `analyze.py` from the files in `results/` and ScoringBench's published results (ScoringBenchOutput at "
      f"commit b590a95), {code} ScoringBench's code at commit 3bf48c9. Every ScoringBench file used matches its SHA-256 "
      "sum in `published_files.json`. The tables and numbers are in the output folder: `numbers.json` and one file per "
      "table.", "", summary, "", "## What each script printed", "", sep="\n")
for name in STEPS:
    print(f"### analysis/{name}.py", "", first_line(name), "", "```", printed[name], "```", "", sep="\n")
