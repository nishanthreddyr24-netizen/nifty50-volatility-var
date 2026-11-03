"""Prepare Experiment 1 and 2 data/calendar metadata without fitting models."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.exp.calendar import (  # noqa: E402
    build_price_variant,
    build_return_frame,
    common_evaluation_dates,
    first_eligible_forecast_date,
    frame_metadata,
    load_nse_prices,
    refit_schedule,
)
from src.exp.config import (  # noqa: E402
    EXPECTED_NSE_PATH,
    EXPERIMENT_1_VARIANTS,
    EXPERIMENT_2_VARIANTS,
    FINAL_DATE,
    MODEL_CONFIGURATION,
    MIN_TRAINING_RETURNS,
    REGIME_DESCRIPTIONS,
    VAR_CONFIDENCE,
    FORECAST_HORIZON,
)

OUTPUT = ROOT / "reports" / "audits" / "exp1_exp2_calendar_preflight.json"
INPUT = ROOT / EXPECTED_NSE_PATH


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def versions() -> dict[str, str]:
    names = ["numpy", "pandas", "scipy", "arch", "hmmlearn", "statsmodels"]
    return {name: importlib.metadata.version(name) for name in names}


def variant_report(prices: pd.Series, variants) -> tuple[dict[str, object], dict[str, pd.DataFrame]]:
    report = {}
    frames = {}
    for variant in variants:
        selected = build_price_variant(prices, variant.nominal_start)
        frame = build_return_frame(prices, variant.nominal_start)
        frames[variant.variant_id] = frame
        item = {
            "variant_id": variant.variant_id,
            "nominal_start": variant.nominal_start,
            "description": variant.description,
            "price_observation_count": len(selected),
            **frame_metadata(frame),
            "price_first_date": str(selected.index[0].date()),
            "price_last_date": str(selected.index[-1].date()),
            "first_eligible_forecast_date": str(first_eligible_forecast_date(frame, MIN_TRAINING_RETURNS).date()),
            "regime_descriptions": dict(REGIME_DESCRIPTIONS),
            "weekend_observation_dates": [str(d.date()) for d in selected.index[selected.index.dayofweek >= 5]],
        }
        report[variant.variant_id] = item
    return report, frames


def paired_report(frames: dict[str, pd.DataFrame], frequencies: dict[str, int]) -> dict[str, object]:
    common, details = common_evaluation_dates(frames, MIN_TRAINING_RETURNS)
    schedules = {
        model: [str(date.date()) for date in refit_schedule(common, frequency)]
        for model, frequency in frequencies.items()
    }
    return {
        "common_evaluation_start": str(common[0].date()) if len(common) else None,
        "common_evaluation_end": str(common[-1].date()) if len(common) else None,
        "common_evaluation_count": len(common),
        "calendar_details": details,
        "refit_schedules": schedules,
        "refit_schedule_lengths": {model: len(dates) for model, dates in schedules.items()},
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    prices = load_nse_prices(INPUT)
    baseline = {
        "effective_start": "2008-01-01",
        "label": "BASELINE_CONFIGURED_2008",
        "evidence_status": "unverified_reproduction",
        "reason": "No reliable original input data or result metadata establishes the actual original sample start.",
    }
    exp1, exp1_frames = variant_report(prices, EXPERIMENT_1_VARIANTS)
    exp2, exp2_frames = variant_report(prices, EXPERIMENT_2_VARIANTS)
    frequencies = {"GBM": 1, "HMM": 63, "GARCH": 21, "HMM_GARCH_HMM": 63, "HMM_GARCH_GARCH": 21, "MS_GARCH": 63}
    manifest = {
        "python": sys.version,
        "platform": platform.platform(),
        "input_path": EXPECTED_NSE_PATH,
        "input_sha256": sha256(INPUT),
        "input_price_count": len(prices),
        "input_first_date": str(prices.index[0].date()),
        "input_last_date": str(prices.index[-1].date()),
        "baseline_resolution": baseline,
        "experiment_1": {"variants": exp1, **paired_report(exp1_frames, frequencies)},
        "experiment_2": {"variants": exp2, **paired_report(exp2_frames, frequencies)},
        "historical_constants": {
            "final_date": FINAL_DATE,
            "min_training_returns": MIN_TRAINING_RETURNS,
            "var_confidence": VAR_CONFIDENCE,
            "forecast_horizon": FORECAST_HORIZON,
        },
        "model_configuration": {
            name: dict(settings) for name, settings in MODEL_CONFIGURATION.items()
        },
        "package_versions": versions(),
        "models_executed": False,
        "weekend_observations_preserved": True,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT),
        "input_sha256": manifest["input_sha256"],
        "experiment_1_common_count": manifest["experiment_1"]["common_evaluation_count"],
        "experiment_2_common_count": manifest["experiment_2"]["common_evaluation_count"],
        "models_executed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
