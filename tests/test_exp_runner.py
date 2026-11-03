from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.exp import runner
from src.exp.calendar import refit_schedule
from src.var_backtest import BacktestResult


def synthetic_returns(n: int = 700) -> pd.Series:
    index = pd.date_range("2010-01-01", periods=n, freq="B")
    values = np.linspace(-0.01, 0.01, n)
    return pd.Series(values, index=index, name="r")


def fake_gbm_fit(history: pd.Series) -> SimpleNamespace:
    return SimpleNamespace(mu_daily=float(history.mean()), sigma_daily=0.01)


def patch_gbm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "fit_gbm", fake_gbm_fit)


def test_no_lookahead_and_forecast_table_integrity(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_gbm(monkeypatch)
    returns = synthetic_returns()
    common = returns.index[500:505]
    first = runner.run_variant_model(returns, common, "GBM", "V1", "exp1")
    changed = returns.copy()
    changed.iloc[500:] += 1.0
    second = runner.run_variant_model(changed, common, "GBM", "V1", "exp1")

    assert first.table.loc[0, "var"] == second.table.loc[0, "var"]
    assert first.table.loc[1, "var"] != second.table.loc[1, "var"]
    assert (pd.to_datetime(first.table["training_end"]) < pd.to_datetime(first.table["date"])).all()
    assert (first.table["n_train"] >= 500).all()
    assert set(first.table[["date", "return", "var", "es", "hit", "n_train", "training_end", "scored", "model", "variant", "experiment"]]) == {
        "date", "return", "var", "es", "hit", "n_train", "training_end", "scored", "model", "variant", "experiment"
    }


def test_first_forecast_alignment() -> None:
    returns = synthetic_returns()
    common = returns.index[500:510]
    result = runner.run_variant_model(returns, common, "Historical Simulation", "V1", "exp1")
    assert result.table.loc[0, "date"] == str(common[0].date())
    assert result.table.loc[0, "n_train"] == 500


def test_fresh_factory_closures() -> None:
    assert runner.build_forecast_factory("HMM") is not runner.build_forecast_factory("HMM")
    assert runner.build_forecast_factory("GARCH") is not runner.build_forecast_factory("GARCH")
    assert runner.build_forecast_factory("MS_GARCH") is not runner.build_forecast_factory("MS_GARCH")


def test_refit_schedule_is_anchored_and_repeatable() -> None:
    dates = pd.date_range("2010-01-01", periods=130)
    assert refit_schedule(dates, 21).equals(refit_schedule(dates, 21))
    assert list(refit_schedule(dates, 63)) == [dates[0], dates[63], dates[126]]


def test_historical_simulation_window_and_identical_variants() -> None:
    returns = synthetic_returns()
    common = returns.index[500:550]
    first = runner.run_variant_model(returns, common, "Historical Simulation", "A", "exp1")
    second = runner.run_variant_model(returns.copy(), common, "Historical Simulation", "B", "exp1")
    pd.testing.assert_frame_equal(first.table.drop(columns=["variant"]), second.table.drop(columns=["variant"]))


def test_harness_hit_flags_are_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_gbm(monkeypatch)
    returns = synthetic_returns()
    result = runner.run_variant_model(returns, returns.index[500:510], "GBM", "V1", "exp1")
    assert set(result.table["hit"].unique()).issubset({0, 1})


def test_output_safety(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    patch_gbm(monkeypatch)
    returns = synthetic_returns()
    result = runner.run_variant_model(returns, returns.index[500:505], "GBM", "V1", "exp1")
    metadata = {"test": True}
    csv_path, json_path = runner.write_run_outputs(result, metadata, tmp_path)
    assert csv_path.exists() and json_path.exists()
    with pytest.raises(FileExistsError):
        runner.write_run_outputs(result, metadata, tmp_path)
