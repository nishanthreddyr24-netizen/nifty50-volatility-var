"""Plan or explicitly execute the prepared Experiment 1 and 2 runs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.exp.calendar import build_return_frame, common_evaluation_dates, load_nse_prices  # noqa: E402
from src.exp.config import (  # noqa: E402
    EXPECTED_NSE_PATH,
    EXPERIMENT_1_VARIANTS,
    EXPERIMENT_2_VARIANTS,
)
from src.exp.runner import build_run_metadata, run_experiment, write_run_outputs  # noqa: E402

PREFLIGHT = ROOT / "reports" / "audits" / "exp1_exp2_calendar_preflight.json"
REFIT_METADATA = {
    "GBM": 1,
    "HMM": 63,
    "GARCH": 21,
    "HMM_GARCH": {"hmm": 63, "garch": 21},
    "MS_GARCH": 63,
    "Historical Simulation": 500,
}


def inputs(variants):
    prices = load_nse_prices(ROOT / EXPECTED_NSE_PATH)
    frames = {
        variant.variant_id: build_return_frame(prices, variant.nominal_start)
        for variant in variants
    }
    common, _ = common_evaluation_dates(frames)
    return frames, common


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Run and write all historical model results")
    args = parser.parse_args()
    git_commit = None
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        pass

    plans = []
    for experiment, variants in [("exp1", EXPERIMENT_1_VARIANTS), ("exp2", EXPERIMENT_2_VARIANTS)]:
        frames, common = inputs(variants)
        plans.append({
            "experiment": experiment,
            "variants": list(frames),
            "common_start": str(common[0].date()),
            "common_end": str(common[-1].date()),
            "common_count": len(common),
        })
        if args.execute:
            results = run_experiment(experiment, frames, common)
            for (variant, model), result in results.items():
                metadata = build_run_metadata(
                    result,
                    EXPECTED_NSE_PATH,
                    str(PREFLIGHT.relative_to(ROOT)),
                    common,
                    REFIT_METADATA[model],
                    git_commit,
                )
                write_run_outputs(result, metadata, ROOT / "results")
    print(json.dumps({"execute": args.execute, "plans": plans}, indent=2))


if __name__ == "__main__":
    main()
