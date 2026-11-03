"""Validate all Experiment 1/2 outputs: tables, metadata, refit schedules, cross-variant alignment."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.exp.calendar import load_nse_prices  # noqa: E402
from src.exp.config import EXPECTED_NSE_PATH  # noqa: E402
from src.exp.priority import (  # noqa: E402
    MODEL_TOKENS,
    PRIMARY_TOKENS,
    check_metadata,
    cross_checks,
    output_paths,
    prepare,
    read_table,
    validate_table,
)
from src.exp.runner import sha256_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=list(MODEL_TOKENS), default=list(PRIMARY_TOKENS))
    parser.add_argument("--report", default=str(ROOT / "results" / "validation_exp1_exp2.json"))
    args = parser.parse_args()

    prices = load_nse_prices(EXPECTED_NSE_PATH)
    input_sha = sha256_file(EXPECTED_NSE_PATH)
    rows, tables, refit_ends = [], {}, {}
    for experiment in ("exp1", "exp2"):
        frames, common, _ = prepare(experiment, prices)  # asserts 4,163 / 3,802 and exact date ranges
        for variant, frame in frames.items():
            for token in args.models:
                model = MODEL_TOKENS[token]
                csv_path, json_path = output_paths(ROOT, experiment, variant, model)
                item = f"{experiment}/{variant}/{model}"
                if not (csv_path.exists() and json_path.exists()):
                    rows.append({"item": item, "status": "MISSING", "errors": [], "warnings": []})
                    continue
                table = read_table(csv_path)
                meta = json.loads(json_path.read_text(encoding="utf-8"))
                errors = validate_table(table, frame, common, model, variant, experiment)
                meta_errors, warns = check_metadata(meta, model, variant, experiment, frame, common, input_sha)
                errors += meta_errors
                tables[(experiment, variant, model)] = table
                if model == "GARCH":
                    refit_ends[(experiment, variant)] = [
                        x["training_end"] for x in meta.get("refit_logs", []) if x.get("component") == "GARCH"
                    ]
                rows.append({
                    "item": item, "status": "FAIL" if errors else "PASS",
                    "n_forecasts": len(table), "first": table["date"].iloc[0], "last": table["date"].iloc[-1],
                    "violations": int(table["hit"].sum()),
                    "source": (meta.get("launcher") or {}).get("status", "pre-existing"),
                    "refits": len(meta.get("refit_logs") or []),
                    "errors": errors, "warnings": warns,
                })

    findings = cross_checks(tables)
    for experiment in ("exp1", "exp2"):  # GARCH refit dates must coincide across variants
        ends = {v: e for (x, v), e in refit_ends.items() if x == experiment}
        if len(ends) > 1:
            same = len({tuple(e) for e in ends.values()}) == 1
            findings.append(("PASS" if same else "FAIL", f"{experiment}/GARCH: refit dates identical across variants: {same}"))

    for row in rows:
        extra = (f"{row['n_forecasts']} rows {row['first']}..{row['last']} viol={row['violations']} "
                 f"refits={row['refits']} [{row['source']}]") if row["status"] != "MISSING" else ""
        print(f"[{row['status']:>7}] {row['item']:<55} {extra}")
        for message in row["errors"]:
            print("           ERROR:", message)
        for message in row["warnings"]:
            print("           warn :", message)
    print()
    for level, message in findings:
        print(f"[{level:>7}] {message}")

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps({"items": rows, "cross_checks": findings}, indent=2, default=str) + "\n",
                                 encoding="utf-8")
    failed = [r for r in rows if r["status"] in ("FAIL", "MISSING")] + [f for f in findings if f[0] == "FAIL"]
    print(f"\nreport: {args.report}\n{len(rows)} items, {len(failed)} FAIL/MISSING")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())