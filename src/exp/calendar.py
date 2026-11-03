"""Read-only data and evaluation-calendar construction for experiments."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Mapping

import pandas as pd

from src.returns import log_returns
from src.exp.config import (
    BASE_DATE,
    FINAL_DATE,
    FREE_FLOAT_START,
    HISTORICAL_START,
    OFFICIAL_LAUNCH_DATE,
    REGIME_DESCRIPTIONS,
)

EXPECTED_COLUMNS = ("Index Name", "Date", "Open", "High", "Low", "Close")
BOUNDARIES = {
    "base_date": pd.Timestamp(BASE_DATE),
    "launch_boundary": pd.Timestamp(OFFICIAL_LAUNCH_DATE),
    "free_float_boundary": pd.Timestamp(FREE_FLOAT_START),
}


def load_nse_prices(path: str | Path) -> pd.Series:
    """Load and strictly validate the consolidated official NSE close series."""
    source = Path(path)
    data = pd.read_csv(source)
    if list(data.columns) != list(EXPECTED_COLUMNS):
        raise ValueError(f"Unexpected NSE columns: {list(data.columns)}")
    dates = pd.to_datetime(data["Date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("NSE Date contains invalid values")
    if dates.duplicated().any():
        raise ValueError("NSE Date contains duplicates")
    close = pd.to_numeric(data["Close"], errors="coerce")
    if close.isna().any():
        raise ValueError("NSE Close must be complete and numeric")
    if (close <= 0).any():
        raise ValueError("NSE Close must be strictly positive")
    prices = pd.Series(close.to_numpy(), index=dates, name="Close").sort_index()
    if prices.index[0] != pd.Timestamp(HISTORICAL_START):
        raise ValueError(f"Unexpected first NSE date: {prices.index[0].date()}")
    if prices.index[-1] != pd.Timestamp(FINAL_DATE):
        raise ValueError(f"Unexpected final NSE date: {prices.index[-1].date()}")
    return prices


def resolve_baseline(
    verified_start: str | None = None,
    evidence: str | None = None,
) -> dict[str, str | bool]:
    """Resolve a verified baseline or the documented configured fallback."""
    if verified_start is not None and evidence:
        return {
            "effective_start": str(pd.Timestamp(verified_start).date()),
            "label": "BASELINE_VERIFIED",
            "evidence_status": "verified",
            "reason": evidence,
        }
    return {
        "effective_start": "2008-01-01",
        "label": "BASELINE_CONFIGURED_2008",
        "evidence_status": "unverified_reproduction",
        "reason": "No reliable original input data or result metadata establishes the actual original sample start.",
    }


def build_price_variant(prices: pd.Series, nominal_start: str) -> pd.Series:
    """Select all official NSE prices from the first available date onward."""
    start = pd.Timestamp(nominal_start)
    variant = prices.loc[prices.index >= start].copy()
    if variant.empty:
        raise ValueError(f"No NSE prices available on or after {nominal_start}")
    return variant


def regime_id(date: pd.Timestamp) -> int:
    date = pd.Timestamp(date)
    if date < BOUNDARIES["base_date"]:
        return 0
    if date < BOUNDARIES["launch_boundary"]:
        return 1
    if date < BOUNDARIES["free_float_boundary"]:
        return 2
    return 3


def build_return_frame(prices: pd.Series, nominal_start: str) -> pd.DataFrame:
    """Build end-date-indexed returns and boundary/calendar metadata."""
    selected = build_price_variant(prices, nominal_start)
    returns = log_returns(selected)
    frame = pd.DataFrame({"r": returns})
    frame["prev_date"] = selected.index[:-1].to_numpy()
    frame["gap_days"] = (
        pd.Series(frame.index, index=frame.index) - frame["prev_date"]
    ).dt.days
    frame["regime_id"] = [regime_id(date) for date in frame.index]
    frame["prev_regime_id"] = [regime_id(date) for date in frame["prev_date"]]
    for name, boundary in BOUNDARIES.items():
        frame[f"{name}_cross"] = (
            (frame["prev_date"] < boundary) & (frame.index >= boundary)
        )
    frame["boundary_cross"] = frame[
        ["base_date_cross", "launch_boundary_cross", "free_float_boundary_cross"]
    ].any(axis=1)
    return frame


def first_eligible_forecast_date(
    returns: pd.DataFrame,
    warmup: int = 500,
) -> pd.Timestamp:
    if len(returns) <= warmup:
        raise ValueError(f"Need more than {warmup} returns, found {len(returns)}")
    return pd.Timestamp(returns.index[warmup])


def common_evaluation_dates(
    variant_frames: Mapping[str, pd.DataFrame],
    warmup: int = 500,
) -> tuple[pd.DatetimeIndex, dict[str, object]]:
    """Return dates common after eligibility and with identical previous dates."""
    eligible = {
        name: first_eligible_forecast_date(frame, warmup)
        for name, frame in variant_frames.items()
    }
    common_start = max(eligible.values())
    common = None
    for frame in variant_frames.values():
        dates = pd.DatetimeIndex(frame.index[frame.index >= common_start])
        common = dates if common is None else common.intersection(dates)
    assert common is not None
    common = common.sort_values()
    reference_name = next(iter(variant_frames))
    reference = variant_frames[reference_name].loc[common, "prev_date"]
    matching = pd.Series(True, index=common)
    for name, frame in variant_frames.items():
        matching &= frame.loc[common, "prev_date"].eq(reference)
    aligned = common[matching.to_numpy()]
    excluded = common[~matching.to_numpy()]
    return aligned, {
        "eligible_first_dates": {name: str(date.date()) for name, date in eligible.items()},
        "common_start_before_prev_date_check": str(common[0].date()) if len(common) else None,
        "excluded_previous_date_mismatch": [str(date.date()) for date in excluded],
    }


def refit_schedule(evaluation_dates: pd.DatetimeIndex, every: int) -> pd.DatetimeIndex:
    if every <= 0:
        raise ValueError("Refit frequency must be positive")
    return evaluation_dates[::every]


def regime_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {str(key): int(value) for key, value in frame["regime_id"].value_counts().sort_index().items()}


def boundary_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {
        column: int(frame[column].sum())
        for column in ["base_date_cross", "launch_boundary_cross", "free_float_boundary_cross", "boundary_cross"]
    }


def frame_metadata(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "price_first_date": str(frame.index[0].date()),
        "price_last_date": str(frame.index[-1].date()),
        "return_count": len(frame),
        "return_first_date": str(frame.index[0].date()) if len(frame) else None,
        "return_last_date": str(frame.index[-1].date()) if len(frame) else None,
        "weekend_return_count": int((frame.index.dayofweek >= 5).sum()),
        "regime_counts": regime_counts(frame),
        "boundary_counts": boundary_counts(frame),
    }
