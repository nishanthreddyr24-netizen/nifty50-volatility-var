"""Complete Experiment 1 and 2 for the priority models (GBM, GARCH, Historical Simulation).

Existing outputs are validated and kept (never overwritten). Experiment 2 GBM/HS
results are reused from Experiment 1 only when verified identical-by-construction.
Re-running is safe and resumable.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # runner hashes the input via a relative path

from src.exp.calendar import load_nse_prices  # noqa: E402
from src.exp.config import EXPECTED_NSE_PATH  # noqa: E402
from src.exp.priority import (  # noqa: E402
    MODEL_TOKENS,
    PRIMARY_TOKENS,
    git_commit,
    prepare,
    process_item,
    write_summary,
)
from src.exp.runner import sha256_file  # noqa: E402

BAD = {"FAILED", "FAILED_VALIDATION", "EXISTING_INVALID", "BLOCKED"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", nargs="+", choices=["exp1", "exp2"], default=["exp1", "exp2"])
    parser.add_argument("--models", nargs="+", choices=list(MODEL_TOKENS), default=list(PRIMARY_TOKENS))
    parser.add_argument("--variants", nargs="+", default=None, help="Restrict to these variant ids")
    parser.add_argument("--dry-run", action="store_true", help="Plan only: no model runs, nothing written")
    parser.add_argument("--no-summary", action="store_true")
    args = parser.parse_args()

    prices = load_nse_prices(EXPECTED_NSE_PATH)
    ctx: dict = {"input_sha": sha256_file(EXPECTED_NSE_PATH)}
    for experiment in ("exp1", "exp2"):  # exp2 reuse needs exp1 frames even if only exp2 is requested
        ctx[experiment] = prepare(experiment, prices)
        _, common, _ = ctx[experiment]
        print(f"{experiment}: common {common[0].date()} -> {common[-1].date()}, {len(common)} forecasts")
    commit = git_commit(ROOT)

    outcomes = []
    for experiment in args.experiments:
        for variant in ctx[experiment][0]:
            if args.variants and variant not in args.variants:
                continue
            for token in args.models:
                started = time.time()
                outcome = process_item(ROOT, experiment, variant, token, ctx, commit, args.dry_run)
                print(f"[{outcome['status']:>17}] {outcome['item']:<55} {outcome['detail']} "
                      f"({time.time() - started:.0f}s)", flush=True)
                outcomes.append(outcome)

    if not args.dry_run and not args.no_summary:
        for experiment in args.experiments:
            frames, common, details = ctx[experiment]
            print("summary:", write_summary(ROOT, experiment, frames, common, details, args.models, ctx["input_sha"]))

    bad = [o for o in outcomes if o["status"] in BAD]
    print(f"\n{len(outcomes)} items, {len(bad)} problems")
    for o in bad:
        print("  PROBLEM:", o["item"], o["status"], o["detail"])
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())