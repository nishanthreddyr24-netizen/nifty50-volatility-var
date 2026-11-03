from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.exp.calendar import (
    build_return_frame,
    build_price_variant,
    common_evaluation_dates,
    first_eligible_forecast_date,
    load_nse_prices,
    regime_id,
    refit_schedule,
    resolve_baseline,
)
from src.returns import log_returns


def prices(n: int = 700) -> pd.Series:
    dates = pd.date_range("1990-01-01", periods=n, freq="B")
    return pd.Series(np.arange(100.0, 100.0 + n), index=dates, name="Close")


def write_nse(path: Path, data: pd.DataFrame) -> None:
    data.to_csv(path, index=False)


def valid_data(n: int = 3) -> pd.DataFrame:
    dates = pd.to_datetime(["1990-07-03"] + ["2026-10-30"] * (n - 1))
    return pd.DataFrame({
        "Index Name": "NIFTY 50",
        "Date": dates.strftime("%Y-%m-%d"),
        "Open": 1.0,
        "High": 1.0,
        "Low": 1.0,
        "Close": np.arange(100.0, 100.0 + n),
    })


def test_loader_parses_and_validates_close(tmp_path: Path) -> None:
    path = tmp_path / "nse.csv"
    data = valid_data(2)
    write_nse(path, data)
    loaded = load_nse_prices(path)
    assert loaded.index[0] == pd.Timestamp("1990-07-03")
    assert loaded.iloc[0] == 100


def test_loader_rejects_duplicate_missing_non_numeric_and_non_positive(tmp_path: Path) -> None:
    for mutation in ["duplicate", "missing", "non_numeric", "non_positive"]:
        data = valid_data(2)
        if mutation == "duplicate":
            data.loc[1, "Date"] = data.loc[0, "Date"]
        elif mutation == "missing":
            data.loc[1, "Close"] = pd.NA
        elif mutation == "non_numeric":
            data["Close"] = data["Close"].astype(object)
            data.loc[1, "Close"] = "bad"
        else:
            data.loc[1, "Close"] = 0
        path = tmp_path / f"{mutation}.csv"
        write_nse(path, data)
        with pytest.raises(ValueError):
            load_nse_prices(path)


def test_variant_and_returns_alignment() -> None:
    series = prices()
    variant = build_price_variant(series, "1990-01-10")
    frame = build_return_frame(series, "1990-01-10")
    expected = log_returns(variant)
    pd.testing.assert_series_equal(frame["r"], expected, check_names=False)
    assert frame.index.equals(variant.index[1:])
    assert frame["prev_date"].iloc[0] == variant.index[0]
    assert frame["gap_days"].iloc[0] == 1


def test_regimes_and_independent_boundary_flags() -> None:
    assert [regime_id(pd.Timestamp(d)) for d in ["1995-11-02", "1995-11-03", "1996-04-22", "2009-06-26"]] == [0, 1, 2, 3]
    series = prices(8000)
    frame = build_return_frame(series, "1990-01-01")
    assert set(["base_date_cross", "launch_boundary_cross", "free_float_boundary_cross"]).issubset(frame.columns)

    boundary_dates = pd.to_datetime([
        "1995-11-02", "1995-11-03", "1996-04-19",
        "1996-04-22", "2009-06-25", "2009-06-26",
    ])
    boundary_frame = build_return_frame(
        pd.Series(np.arange(100.0, 106.0), index=boundary_dates),
        "1995-11-02",
    )
    assert int(boundary_frame["base_date_cross"].sum()) == 1
    assert int(boundary_frame["launch_boundary_cross"].sum()) == 1
    assert int(boundary_frame["free_float_boundary_cross"].sum()) == 1


def test_first_forecast_and_insufficient_history() -> None:
    frame = build_return_frame(prices(), "1990-01-01")
    assert first_eligible_forecast_date(frame) == frame.index[500]
    with pytest.raises(ValueError):
        first_eligible_forecast_date(frame.iloc[:500])


def test_common_dates_previous_date_alignment_and_schedules() -> None:
    base = build_return_frame(prices(), "1990-01-01")
    other = base.copy()
    common, details = common_evaluation_dates({"base": base, "other": other})
    assert len(common) > 0
    assert details["excluded_previous_date_mismatch"] == []
    assert refit_schedule(common, 63).equals(refit_schedule(common, 63))

    other.loc[common[0], "prev_date"] = other.loc[common[0], "prev_date"] - pd.Timedelta(days=1)
    aligned, details = common_evaluation_dates({"base": base, "other": other})
    assert common[0] not in aligned
    assert str(common[0].date()) in details["excluded_previous_date_mismatch"]


def test_experiment_common_starts_are_independent() -> None:
    series = prices(9000)
    exp1, _ = common_evaluation_dates({
        "baseline": build_return_frame(series, "2008-01-01"),
        "extended": build_return_frame(series, "1990-01-01"),
    })
    exp2, _ = common_evaluation_dates({
        "prelaunch": build_return_frame(series, "1990-01-01"),
        "launched": build_return_frame(series, "1996-04-22"),
        "free_float": build_return_frame(series, "2009-06-26"),
    })
    assert exp1[0] != exp2[0]


def test_baseline_fallback_is_explicit() -> None:
    result = resolve_baseline()
    assert result["label"] == "BASELINE_CONFIGURED_2008"
    assert result["evidence_status"] == "unverified_reproduction"


def test_weekends_are_preserved() -> None:
    dates = pd.to_datetime(["1990-07-03", "1990-07-06", "1990-07-07", "1990-07-09"])
    series = pd.Series([100.0, 101.0, 102.0, 103.0], index=dates)
    frame = build_return_frame(series, "1990-07-03")
    assert pd.Timestamp("1990-07-07") in frame.index
