"""Causal orchestration for the prepared historical experiments.

This module reuses the existing model factories and backtest harness. It does
not implement model mathematics or execute experiments at import time.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import tempfile
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from unittest.mock import patch

from src.garch import fit_garch, make_garch_forecast_fn
from src.gbm import fit_gbm
from src.hmm import fit_gaussian_hmm, make_hmm_forecast_fn
from src.hmm_garch import make_hmm_garch_forecast_fn
from src.msgarch import fit_msgarch, make_msgarch_forecast_fn
from src.var_backtest import (
    BacktestResult,
    historical_sim_var_backtest,
    rolling_var_backtest,
)

MODEL_NAMES = (
    "GBM",
    "HMM",
    "GARCH",
    "HMM_GARCH",
    "MS_GARCH",
    "Historical Simulation",
)


@dataclass
class RunResult:
    table: pd.DataFrame
    backtest: BacktestResult
    refit_log: list[dict[str, object]]
    model: str
    variant: str
    experiment: str


def _training_info(returns: pd.Series) -> dict[str, object]:
    return {
        "n_train": len(returns),
        "training_end": str(returns.index[-1].date()) if len(returns) else None,
    }


def _record_fit(log: list[dict[str, object]], kind: str, returns: pd.Series, fit: object) -> None:
    item = {"component": kind, **_training_info(returns)}
    if hasattr(fit, "log_likelihood"):
        item["log_likelihood"] = float(fit.log_likelihood)
    if hasattr(fit, "persistence"):
        item["persistence"] = np.asarray(fit.persistence).tolist()
    if hasattr(fit, "stds"):
        item["state_stds"] = np.asarray(fit.stds).tolist()
    if hasattr(fit, "transmat"):
        item["transition_matrix"] = np.asarray(fit.transmat).tolist()
    if hasattr(fit, "success"):
        item["optimizer_success"] = bool(fit.success)
    log.append(item)


@contextmanager
def _fit_logging(model: str, log: list[dict[str, object]]) -> Iterator[None]:
    """Patch confirmed fit references only for the lifetime of one run."""
    from src import garch as garch_module
    from src import hmm as hmm_module
    from src import hmm_garch as hmm_garch_module
    from src import msgarch as msgarch_module

    patches = []

    def wrap(module: object, name: str, kind: str) -> None:
        original = getattr(module, name)

        def logged(*args: object, **kwargs: object) -> object:
            bound = __import__("inspect").signature(original).bind_partial(*args, **kwargs)
            history = bound.arguments.get("returns")
            fit = original(*args, **kwargs)
            if isinstance(history, pd.Series):
                _record_fit(log, kind, history, fit)
            return fit

        patches.append(patch.object(module, name, logged))

    if model == "HMM":
        wrap(hmm_module, "fit_gaussian_hmm", "HMM")
    elif model == "GARCH":
        wrap(garch_module, "fit_garch", "GARCH")
    elif model == "HMM_GARCH":
        wrap(hmm_garch_module, "fit_gaussian_hmm", "HMM")
        wrap(garch_module, "fit_garch", "GARCH")
    elif model == "MS_GARCH":
        wrap(msgarch_module, "fit_msgarch", "MS_GARCH")

    with ExitStack() as stack:
        for active_patch in patches:
            stack.enter_context(active_patch)
        yield


def build_forecast_factory(model: str) -> Callable[[pd.Series], tuple[float, float]]:
    """Build one fresh causal forecast closure for one model run."""
    if model == "GBM":
        def gbm_forecast(history: pd.Series) -> tuple[float, float]:
            fit = fit_gbm(history)
            return fit.mu_daily, fit.sigma_daily
        return gbm_forecast
    if model == "HMM":
        return make_hmm_forecast_fn(n_states=2, refit_every=63, n_iter=100, random_state=42)
    if model == "GARCH":
        return make_garch_forecast_fn(refit_every=21)
    if model == "HMM_GARCH":
        return make_hmm_garch_forecast_fn(hmm_refit_every=63, garch_refit_every=21)
    if model == "MS_GARCH":
        return make_msgarch_forecast_fn(refit_every=63, n_starts=1)
    raise ValueError(f"No forecast factory for {model!r}")


def _as_returns(returns: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(returns, pd.DataFrame):
        if "r" not in returns.columns:
            raise ValueError("Return frame must contain an 'r' column")
        return returns["r"]
    return returns


def _augment_table(
    backtest: BacktestResult,
    returns: pd.Series,
    common_dates: pd.DatetimeIndex,
    model: str,
    variant: str,
    experiment: str,
) -> pd.DataFrame:
    table = backtest.table.loc[backtest.table.index.intersection(common_dates)].copy()
    table.index.name = "date"
    rows = table.reset_index()
    positions = returns.index.get_indexer(rows["date"])
    if (positions < 1).any():
        raise ValueError("Every forecast target must have a prior training observation")
    rows["n_train"] = positions
    rows["training_end"] = [str(returns.index[position - 1].date()) for position in positions]
    rows["scored"] = True
    rows["model"] = model
    rows["variant"] = variant
    rows["experiment"] = experiment
    rows["date"] = pd.to_datetime(rows["date"]).dt.strftime("%Y-%m-%d")
    if not all(pd.to_datetime(rows["training_end"]) < pd.to_datetime(rows["date"])):
        raise AssertionError("training_end must precede date")
    return rows


def run_variant_model(
    returns: pd.Series | pd.DataFrame,
    common_dates: pd.DatetimeIndex,
    model: str,
    variant: str,
    experiment: str,
) -> RunResult:
    """Run one model on one complete variant, scoring only common dates."""
    series = _as_returns(returns).sort_index()
    common_dates = pd.DatetimeIndex(common_dates).sort_values()
    if len(common_dates) == 0:
        raise ValueError("Common evaluation dates cannot be empty")
    first_position = series.index.get_indexer([common_dates[0]])[0]
    if first_position < 500:
        raise ValueError("First common forecast must have at least 500 prior returns")

    refit_log: list[dict[str, object]] = []
    with _fit_logging(model, refit_log):
        if model == "Historical Simulation":
            backtest = historical_sim_var_backtest(series, window=500, alpha=0.99, step=1)
        else:
            forecast = build_forecast_factory(model)
            backtest = rolling_var_backtest(
                series,
                forecast,
                alpha=0.99,
                min_window=first_position,
                step=1,
            )
    table = _augment_table(backtest, series, common_dates, model, variant, experiment)
    return RunResult(table, backtest, refit_log, model, variant, experiment)


def run_experiment(
    experiment: str,
    variants: dict[str, pd.DataFrame],
    common_dates: pd.DatetimeIndex,
) -> dict[tuple[str, str], RunResult]:
    """Run all six models for every supplied variant."""
    results = {}
    for variant, frame in variants.items():
        for model in MODEL_NAMES:
            results[(variant, model)] = run_variant_model(
                frame, common_dates, model, variant, experiment
            )
    return results


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_versions() -> dict[str, str]:
    names = ("numpy", "pandas", "scipy", "arch", "hmmlearn", "statsmodels")
    return {name: importlib.metadata.version(name) for name in names}


def build_run_metadata(
    result: RunResult,
    input_path: str,
    preflight_report: str,
    common_dates: pd.DatetimeIndex,
    refit_frequency: int | dict[str, int] | None,
    git_commit: str | None = None,
) -> dict[str, object]:
    return {
        "experiment": result.experiment,
        "variant": result.variant,
        "model": result.model,
        "input_dataset_path": input_path,
        "input_sha256": sha256_file(input_path),
        "preflight_report": preflight_report,
        "first_forecast_date": str(common_dates[0].date()),
        "last_forecast_date": str(common_dates[-1].date()),
        "common_evaluation_period": [str(common_dates[0].date()), str(common_dates[-1].date())],
        "initial_training_count": int(result.table["n_train"].iloc[0]),
        "var_confidence": 0.99,
        "forecast_horizon": 1,
        "refit_frequency": refit_frequency,
        "python": sys.version,
        "platform": platform.platform(),
        "package_versions": _package_versions(),
        "git_commit": git_commit,
        "refit_logs": result.refit_log,
    }


def write_run_outputs(
    result: RunResult,
    metadata: dict[str, object],
    output_root: str | Path = "results",
) -> tuple[Path, Path]:
    """Write paired CSV/JSON outputs atomically, refusing existing targets."""
    directory = Path(output_root) / result.experiment / result.variant
    safe_model = result.model.lower().replace(" ", "_").replace("-", "_")
    csv_path = directory / f"{safe_model}.csv"
    json_path = directory / f"{safe_model}.json"
    if csv_path.exists() or json_path.exists():
        raise FileExistsError(f"Output conflict: {csv_path} or {json_path}")
    directory.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="exp-run-", dir=directory))
    temp_csv = temp_dir / csv_path.name
    temp_json = temp_dir / json_path.name
    try:
        result.table.to_csv(temp_csv, index=False)
        temp_json.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        if csv_path.exists() or json_path.exists():
            raise FileExistsError(f"Output conflict appeared during write: {directory}")
        temp_csv.replace(csv_path)
        temp_json.replace(json_path)
    finally:
        if temp_dir.exists():
            for path in temp_dir.iterdir():
                path.unlink()
            temp_dir.rmdir()
    return csv_path, json_path
