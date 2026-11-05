"""Run Ashlar on TabArena-Lite and print where it lands on the leaderboard.

Usage (from this folder, in a venv with tabarena[benchmark] and ashlar):

    python run_lite.py --smoke                 # 3 small datasets, defaults only
    python run_lite.py                         # all 51 Lite datasets, defaults only
    python run_lite.py --n-configs 25          # default + 25 random configs (tuned, tuned+ensembled)
    python run_lite.py --time-limit 21600      # 6 hours per config instead of TabArena's 1 hour

Results are cached under experiments/<expname>, so an interrupted run resumes and a rerun only re-scores.
Runs with a longer --time-limit than the official hour are not comparable with the leaderboard, so give them
their own --expname.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ashlar_tabarena import LOCAL_SWEEP_MODELS, ROUND4_MODELS, SWEEP_MODELS, AshlarModel, AshlarRipModel, AshlarTabMModel, AshlarTabMRipModel  # noqa: E402

SMOKE = ["blood-transfusion-service-center", "QSAR_fish_toxicity", "anneal"]
MODELS = {
    "default": AshlarModel,
    "tabm": AshlarTabMModel,
    "rip": AshlarRipModel,
    "tabm_rip": AshlarTabMRipModel,
    **SWEEP_MODELS,
    **ROUND4_MODELS,
    **LOCAL_SWEEP_MODELS,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--models", nargs="*", default=["default", "tabm"], choices=list(MODELS))
    parser.add_argument("--n-configs", type=int, default=0)
    parser.add_argument("--expname", default="ashlar_lite")
    parser.add_argument("--no-debug", action="store_true", help="Use the Ray backend (parallel folds).")
    parser.add_argument("--keep-going", action="store_true", help="Record a failed job (e.g. over the time limit) and continue.")
    parser.add_argument(
        "--time-limit", type=int, default=None, help="Seconds per config across all folds (default: TabArena's 3600)."
    )
    args = parser.parse_args()

    from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
    from tabarena.contexts import TabArenaContext

    datasets = SMOKE if args.smoke else args.datasets
    expname = args.expname + ("_smoke" if args.smoke else "")
    results_dir = str(HERE / "experiments" / expname)
    eval_dir = HERE / "eval" / expname

    experiments = TabArenaV0pt1ExperimentBundle(
        models=[(MODELS[name].config_generator(), args.n_configs) for name in args.models],
    ).build_experiments(**({"time_limit": args.time_limit} if args.time_limit else {}))

    context = TabArenaContext()
    build_kwargs = {"dataset_names": datasets} if datasets else {}
    context.build_and_run_jobs(
        experiments,
        expname=results_dir,
        subset="lite",
        build_kwargs=build_kwargs,
        new_result_prefix="[New] ",
        debug_mode=not args.no_debug,
        raise_on_failure=not args.keep_going,
    )

    leaderboard = context.compare(output_dir=eval_dir, subset="lite", datasets=datasets)
    website = context.leaderboard_to_website_format(leaderboard=leaderboard)
    text = website.to_markdown(index=False)
    (eval_dir / "leaderboard.md").write_text(text)
    print(text)
    print(f"\nLeaderboard and figures in {eval_dir}")


if __name__ == "__main__":
    main()
