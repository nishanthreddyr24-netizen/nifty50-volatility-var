"""Priority Experiment 1/2 helpers: safe reuse, validation, summaries.

No model mathematics is implemented or changed here. All forecasts come from
src.exp.runner.run_variant_model (existing factories + existing harness).
The only independent formulas are the closed-form GBM and Historical
Simulation recomputations used to VERIFY stored forecasts.
"""

from __future__ import annotations

import functools
import json
import subprocess
import time
import warnings
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy import stats as sstats

from src.exp.calendar import build_return_frame, common_evaluation_dates
from src.exp.config import (
    EXPECTED_NSE_PATH,
    EXPERIMENT_1_VARIANTS,
    EXPERIMENT_2_VARIANTS,
    HISTORICAL_SIMULATION_WINDOW,
    MIN_TRAINING_RETURNS,
)
from src.exp.runner import (
    RunResult,
    build_run_metadata,
    run_variant_model,
    sha256_file,
    write_run_outputs,
)
from src.gbm import parametric_es, parametric_var
from src.var_backtest import BacktestResult, christoffersen_test, kupiec_test

PREFLIGHT_REPORT = "reports/audits/exp1_exp2_calendar_preflight.json"
VARIANTS = {"exp1": EXPERIMENT_1_VARIANTS, "exp2": EXPERIMENT_2_VARIANTS}
# (common start, common end, forecast count) stated in the task; verified, not assumed.
EXPECTED = {
    "exp1": ("2010-01-20", "2026-10-30", 4163),
    "exp2": ("2011-06-28", "2026-10-30", 3802),
}
MODEL_TOKENS = {
    "GBM": "GBM",
    "GARCH": "GARCH",
    "HS": "Historical Simulation",
    "HMM": "HMM",
    "HMM_GARCH": "HMM_GARCH",
    "MS_GARCH": "MS_GARCH",
}
PRIMARY_TOKENS = ("GBM", "GARCH", "HS")
REFIT_METADATA = {
    "GBM": 1,
    "HMM": 63,
    "GARCH": 21,
    "HMM_GARCH": {"hmm": 63, "garch": 21},
    "MS_GARCH": 63,
    "Historical Simulation": 500,
}
COMPONENTS = {  # component -> refit cadence expected in refit_logs
    "GARCH": {"GARCH": 21},
    "HMM": {"HMM": 63},
    "HMM_GARCH": {"HMM": 63, "GARCH": 21},
    "MS_GARCH": {"MS_GARCH": 63},
}
TABLE_COLUMNS = [
    "date", "return", "var", "es", "hit", "n_train",
    "training_end", "scored", "model", "variant", "experiment",
]
FLOAT_TOL = 1e-12
RETURN_TOL = 1e-15
LABELS = {
    "exp1": {
        "baseline_label": "BASELINE_CONFIGURED_2008",
        "baseline_evidence_status": "unverified_reproduction",
        "interpretation": (
            "BASELINE_CONFIGURED_2008 is a configured-start run on official NSE data. "
            "It is NOT claimed to reproduce the original experiment."
        ),
    },
    "exp2": {
        "interpretation": (
            "Sensitivity of forecasts to historical sample composition (start date). "
            "Not a pure causal test of index-methodology changes."
        ),
    },
}


# ----------------------------------------------------------------- basics
def model_stem(model: str) -> str:
    return model.lower().replace(" ", "_").replace("-", "_")  # mirrors runner.write_run_outputs


def output_paths(root: str | Path, experiment: str, variant: str, model: str) -> tuple[Path, Path]:
    directory = Path(root) / "results" / experiment / variant
    stem = model_stem(model)
    return directory / f"{stem}.csv", directory / f"{stem}.json"


def git_commit(root: str | Path) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def prepare(experiment: str, prices: pd.Series):
    """Build variant frames and common dates; verify them against the task's stated values."""
    frames = {v.variant_id: build_return_frame(prices, v.nominal_start) for v in VARIANTS[experiment]}
    common, details = common_evaluation_dates(frames, MIN_TRAINING_RETURNS)
    got = (str(common[0].date()), str(common[-1].date()), len(common))
    if got != EXPECTED[experiment]:
        raise AssertionError(f"{experiment}: common evaluation {got} != expected {EXPECTED[experiment]}")
    return frames, common, details


def read_table(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip")  # exact float round trip


def _close(a, b, tol: float) -> bool:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or not (np.isfinite(a).all() and np.isfinite(b).all()):
        return False
    return bool(np.max(np.abs(a - b), initial=0.0) <= tol)


# ------------------------------------------------- independent recomputation
def expected_forecast(model: str, returns: pd.Series, position: int) -> tuple[float, float]:
    """Closed-form VaR/ES from the same formulas the harness uses (verification only)."""
    if model == "GBM":
        history = returns.iloc[:position]
        mu, sigma = float(history.mean()), float(history.std(ddof=1))
        return float(parametric_var(mu, sigma, 0.99)), float(parametric_es(mu, sigma, 0.99))
    if model == "Historical Simulation":
        window = returns.iloc[position - HISTORICAL_SIMULATION_WINDOW: position]
        q = float(np.quantile(window.to_numpy(), 1 - 0.99))
        tail = window[window <= q]
        return -q, (float(-tail.mean()) if len(tail) else -q)
    raise ValueError(f"No independent recomputation for {model}")


# ------------------------------------------------------------ validation
def validate_table(table, frame, common, model, variant, experiment, spot_checks: int = 7) -> list[str]:
    """Return a list of error strings (empty list = valid)."""
    missing = [c for c in TABLE_COLUMNS if c not in table.columns]
    if missing:
        return [f"missing columns {missing}"]
    expected_dates = [str(d.date()) for d in common]
    if table["date"].tolist() != expected_dates:
        return [f"dates differ from common evaluation dates ({len(table)} rows vs {len(expected_dates)})"]
    positions = frame.index.get_indexer(common)
    if (positions < 0).any():
        return ["some common dates are absent from the variant return frame"]
    errors: list[str] = []
    returns = frame["r"]
    if not _close(table["return"], returns.to_numpy()[positions], RETURN_TOL):
        errors.append("realised returns differ from the variant return frame")
    if not np.array_equal(table["n_train"].to_numpy(), positions):
        errors.append("n_train differs from forecast-date position")
    if (positions < MIN_TRAINING_RETURNS).any():
        errors.append("a forecast has fewer than 500 prior returns")
    expected_end = [str(frame.index[p - 1].date()) for p in positions]
    if table["training_end"].tolist() != expected_end:
        errors.append("training_end is not the previous return date")
    if not (pd.to_datetime(table["training_end"]).to_numpy() < pd.to_datetime(table["date"]).to_numpy()).all():
        errors.append("training_end is not strictly before date")
    var = table["var"].to_numpy(dtype=float)
    es = table["es"].to_numpy(dtype=float)
    finite = bool(np.isfinite(var).all() and np.isfinite(es).all())
    if not finite:
        errors.append("non-finite VaR or ES")
    else:
        if (var <= 0).any() or (es <= 0).any():
            errors.append("VaR/ES must be positive")
        if (es < var - FLOAT_TOL).any():
            errors.append("ES below VaR")
    expected_hit = (-table["return"].to_numpy(dtype=float) > var).astype(int)
    if not np.array_equal(expected_hit, table["hit"].to_numpy()):
        errors.append("hit != (-return > VaR)")
    if not bool(table["scored"].astype(bool).all()):
        errors.append("scored column not all True")
    for column, wanted in (("model", model), ("variant", variant), ("experiment", experiment)):
        if not bool((table[column] == wanted).all()):
            errors.append(f"{column} column is not constantly {wanted!r}")
    if finite and model in ("GBM", "Historical Simulation"):
        picks = np.unique(np.linspace(0, len(table) - 1, spot_checks).astype(int))
        for k in picks:
            want_var, want_es = expected_forecast(model, returns, int(positions[k]))
            if abs(want_var - var[k]) > FLOAT_TOL or abs(want_es - es[k]) > FLOAT_TOL:
                errors.append(f"independent {model} recomputation differs on {expected_dates[k]}")
    return errors


def check_metadata(meta, model, variant, experiment, frame, common, input_sha):
    """Return (errors, warnings) for a run's JSON metadata."""
    errors: list[str] = []
    warns: list[str] = []
    i0 = int(frame.index.get_loc(common[0]))
    expected = {
        "experiment": experiment, "variant": variant, "model": model,
        "var_confidence": 0.99, "forecast_horizon": 1,
        "first_forecast_date": str(common[0].date()),
        "last_forecast_date": str(common[-1].date()),
        "initial_training_count": i0,
    }
    for key, wanted in expected.items():
        if meta.get(key) != wanted:
            errors.append(f"metadata {key}={meta.get(key)!r}, expected {wanted!r}")
    if meta.get("input_sha256") != input_sha:
        errors.append("metadata input_sha256 differs from the current NSE file")
    if meta.get("refit_frequency") != REFIT_METADATA[model]:
        warns.append(f"refit_frequency metadata {meta.get('refit_frequency')!r} != {REFIT_METADATA[model]!r}")
    if not meta.get("git_commit"):
        warns.append("git_commit missing")
    if not meta.get("package_versions"):
        warns.append("package_versions missing")
    if model in COMPONENTS:
        logs = meta.get("refit_logs") or []
        for component, every in COMPONENTS[model].items():
            got = [x["n_train"] for x in logs if x.get("component") == component]
            want = list(range(i0, len(frame), every))
            if got != want:
                errors.append(
                    f"{component} refit schedule: {len(got)} fits (first {got[:3]}), "
                    f"expected {len(want)} (first {want[:3]})"
                )
            got_end = [x["training_end"] for x in logs if x.get("component") == component]
            want_end = [str(frame.index[w - 1].date()) for w in want]
            if got_end != want_end:
                errors.append(f"{component} refit training_end dates differ from the anchored schedule")
    launcher = meta.get("launcher") or {}
    counts = launcher.get("warning_counts") or {}
    if counts:
        warns.append(f"{sum(counts.values())} runtime warnings: {list(counts)[:3]}")
    params = launcher.get("garch_fit_parameters") or []
    high = [p for p in params if p["persistence"] >= 0.999]
    if high:
        warns.append(f"{len(high)} GARCH fits with alpha+beta >= 0.999 (max {max(p['persistence'] for p in params):.5f})")
    return errors, warns


# ---------------------------------------------------------------- reuse
def series_identical(a: pd.Series, b: pd.Series) -> bool:
    return a.index.equals(b.index) and np.array_equal(a.to_numpy(), b.to_numpy())


def hs_windows_identical(frame_a, frame_b, first_date, window: int = HISTORICAL_SIMULATION_WINDOW) -> bool:
    """True iff every trailing-`window` return set for dates >= first_date is identical."""
    tails = []
    for frame in (frame_a, frame_b):
        start = int(frame.index.get_loc(first_date)) - window
        if start < 0:
            return False
        tails.append(frame["r"].iloc[start:])
    return series_identical(tails[0], tails[1])


def build_reused_table(source, frame, common, model, variant, experiment) -> pd.DataFrame:
    dates = [str(d.date()) for d in common]
    indexed = source.set_index("date")
    absent = [d for d in dates if d not in indexed.index]
    if absent:
        raise ValueError(f"source lacks {len(absent)} required dates (first {absent[0]})")
    selected = indexed.loc[dates]
    positions = frame.index.get_indexer(common)
    training_end = [str(frame.index[p - 1].date()) for p in positions]
    if selected["training_end"].tolist() != training_end:
        raise ValueError("training_end differs between source and target; reuse invalid")
    return pd.DataFrame({
        "date": dates,
        "return": selected["return"].to_numpy(),
        "var": selected["var"].to_numpy(),
        "es": selected["es"].to_numpy(),
        "hit": selected["hit"].to_numpy().astype(int),
        "n_train": positions,
        "training_end": training_end,
        "scored": True,
        "model": model,
        "variant": variant,
        "experiment": experiment,
    })


def make_run_result(table: pd.DataFrame, model: str, variant: str, experiment: str) -> RunResult:
    index = pd.DatetimeIndex(pd.to_datetime(table["date"]), name="date")
    bt_table = table[["return", "var", "es", "hit"]].copy()
    bt_table.index = index
    return RunResult(table, BacktestResult(bt_table, 0.99), [], model, variant, experiment)


def try_reuse(root, variant: str, model: str, ctx):
    """Return (table, info). table is None when reuse is not valid (info['reason'] says why)."""
    source_variant = "EXTENDED_1990"
    frames1, common1, _ = ctx["exp1"]
    frames2, common2, _ = ctx["exp2"]
    if model == "GBM":
        if variant != "PRELAUNCH_1990":
            return None, {"reason": "GBM expands from the variant start; no equivalent exp1 variant"}
        if not series_identical(frames1[source_variant]["r"], frames2[variant]["r"]):
            return None, {"reason": "return series differ from exp1 EXTENDED_1990"}
    elif model == "Historical Simulation":
        if not hs_windows_identical(frames1[source_variant], frames2[variant], common2[0]):
            return None, {"reason": "trailing 500-return windows differ"}
    else:
        return None, {"reason": f"{model} is never reused"}
    source_csv, source_json = output_paths(root, "exp1", source_variant, model)
    if not (source_csv.exists() and source_json.exists()):
        return None, {"reason": "exp1 source results missing"}
    source = read_table(source_csv)
    errors = validate_table(source, frames1[source_variant], common1, model, source_variant, "exp1")
    if errors:
        return None, {"reason": "exp1 source failed validation: " + "; ".join(errors[:3])}
    try:
        table = build_reused_table(source, frames2[variant], common2, model, variant, "exp2")
    except ValueError as exc:
        return None, {"reason": str(exc)}
    errors = validate_table(table, frames2[variant], common2, model, variant, "exp2")
    if errors:
        return None, {"reason": "reused table failed validation: " + "; ".join(errors[:3])}
    return table, {
        "source_experiment": "exp1",
        "source_variant": source_variant,
        "source_csv": str(source_csv.relative_to(Path(root))),
        "source_csv_sha256": sha256_file(source_csv),
        "checks": [
            "target forecast dates/returns/training_end/n_train recomputed from target variant frame",
            "source table passed full validation incl. independent recomputation",
            "series identical (GBM)" if model == "GBM" else "all 500-return windows identical (HS)",
        ],
    }


# ---------------------------------------------------------------- compute
@contextmanager
def record_garch_params(store: list) -> Iterator[None]:
    """Log fitted GARCH parameters. functools.wraps keeps the signature the runner's logger binds against."""
    from src import garch as garch_module

    original = garch_module.fit_garch

    @functools.wraps(original)
    def recorded(returns, *args, **kwargs):
        fit = original(returns, *args, **kwargs)
        store.append({
            "n_train": len(returns), "mu": float(fit.mu), "omega": float(fit.omega),
            "alpha": float(fit.alpha), "beta": float(fit.beta),
            "persistence": float(fit.alpha + fit.beta),
        })
        return fit

    with patch.object(garch_module, "fit_garch", recorded):
        yield


def compute_run(model, variant, experiment, frame, common, commit):
    params: list = []
    started = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if model in ("GARCH", "HMM_GARCH"):
            with record_garch_params(params):
                result = run_variant_model(frame, common, model, variant, experiment)
        else:
            result = run_variant_model(frame, common, model, variant, experiment)
    counts = Counter(f"{w.category.__name__}: {str(w.message)[:160]}" for w in caught)
    meta = build_run_metadata(result, EXPECTED_NSE_PATH, PREFLIGHT_REPORT, common, REFIT_METADATA[model], commit)
    meta["launcher"] = {
        "status": "computed",
        "seconds": round(time.time() - started, 1),
        "warning_counts": dict(counts),
        "garch_fit_parameters": params or None,
    }
    return result, meta


def process_item(root, experiment, variant, token, ctx, commit, dry_run=False) -> dict:
    model = MODEL_TOKENS[token]
    frames, common, _ = ctx[experiment]
    frame = frames[variant]
    label = f"{experiment}/{variant}/{model}"
    csv_path, json_path = output_paths(root, experiment, variant, model)

    if csv_path.exists() or json_path.exists():
        if not (csv_path.exists() and json_path.exists()):
            return {"item": label, "status": "BLOCKED",
                    "detail": "only one of CSV/JSON exists; inspect manually (nothing overwritten)"}
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        problems = validate_table(read_table(csv_path), frame, common, model, variant, experiment)
        problems += check_metadata(meta, model, variant, experiment, frame, common, ctx["input_sha"])[0]
        return {"item": label, "status": "EXISTING_INVALID" if problems else "EXISTING_VALID",
                "detail": "; ".join(problems[:4])}

    try:
        table, info = (None, {})
        if experiment == "exp2" and model in ("GBM", "Historical Simulation"):
            table, info = try_reuse(root, variant, model, ctx)
        if table is not None:
            if dry_run:
                return {"item": label, "status": "WOULD_REUSE", "detail": info["source_variant"]}
            result = make_run_result(table, model, variant, experiment)
            meta = build_run_metadata(result, EXPECTED_NSE_PATH, PREFLIGHT_REPORT, common,
                                        REFIT_METADATA[model], commit)
            meta["launcher"] = {"status": "reused", "reuse": info}
            status, detail = "REUSED", f"from exp1/{info['source_variant']}"
        else:
            if dry_run:
                return {"item": label, "status": "WOULD_COMPUTE", "detail": info.get("reason", "")}
            result, meta = compute_run(model, variant, experiment, frame, common, commit)
            problems = validate_table(result.table, frame, common, model, variant, experiment)
            problems += check_metadata(meta, model, variant, experiment, frame, common, ctx["input_sha"])[0]
            if problems:
                return {"item": label, "status": "FAILED_VALIDATION",
                        "detail": "; ".join(problems[:4]) + " (NOT written)"}
            status = "COMPUTED"
            detail = (f"{meta['launcher']['seconds']}s, {len(result.refit_log)} refits, "
                        f"{sum(meta['launcher']['warning_counts'].values())} warnings")
        write_run_outputs(result, meta, Path(root) / "results")
        return {"item": label, "status": status, "detail": detail}
    except Exception as exc:  # report and continue with the other items
        return {"item": label, "status": "FAILED", "detail": f"{type(exc).__name__}: {exc}"}


# -------------------------------------------------------------- cross checks
def cross_checks(tables: dict) -> list[tuple[str, str]]:
    """tables: {(experiment, variant, model): DataFrame}. Returns (level, message) findings."""
    findings: list[tuple[str, str]] = []
    groups = defaultdict(list)
    for (experiment, variant, model), table in tables.items():
        groups[(experiment, model)].append((variant, table))
    for (experiment, model), items in groups.items():
        base_variant, base = items[0]
        for variant, table in items[1:]:
            tag = f"{experiment}/{model}: {base_variant} vs {variant}"
            if table["date"].tolist() != base["date"].tolist():
                findings.append(("FAIL", f"{tag}: forecast dates differ"))
                continue
            ok = True
            if not _close(table["return"], base["return"], RETURN_TOL):
                findings.append(("FAIL", f"{tag}: realised returns differ")); ok = False
            if table["training_end"].tolist() != base["training_end"].tolist():
                findings.append(("FAIL", f"{tag}: training_end differs")); ok = False
            same_var = _close(table["var"], base["var"], 1e-15)
            same_es = _close(table["es"], base["es"], 1e-15)
            if model == "Historical Simulation" and not (same_var and same_es):
                findings.append(("FAIL", f"{tag}: HS must be identical across variants")); ok = False
            if model in ("GBM", "GARCH") and same_var and same_es:
                findings.append(("FAIL", f"{tag}: forecasts identical (variant start not applied?)")); ok = False
            if ok:
                findings.append(("PASS", f"{tag}: dates, returns, training_end aligned"
                                 + ("; HS identical" if model == "Historical Simulation" else "")))
    for (experiment, variant, model), table in tables.items():
        if experiment != "exp2" or model not in ("GBM", "Historical Simulation"):
            continue
        if model == "GBM" and variant != "PRELAUNCH_1990":
            continue
        source_key = ("exp1", "EXTENDED_1990", model)
        if source_key not in tables:
            continue
        try:
            source = tables[source_key].set_index("date").loc[table["date"].tolist()]
        except KeyError:
            findings.append(("FAIL", f"exp2/{variant}/{model}: dates missing from exp1 source")); continue
        same = _close(source["var"], table["var"], 1e-15) and _close(source["es"], table["es"], 1e-15)
        findings.append(("PASS" if same else "FAIL",
                         f"exp2/{variant}/{model} equals exp1/EXTENDED_1990 on all exp2 dates: {same}"))
    return findings


# ---------------------------------------------------------------- summaries
def write_summary(root, experiment, frames, common, details, tokens, input_sha) -> Path:
    root = Path(root)
    runs: dict = {}
    tables: dict = {}
    for variant in frames:
        for token in tokens:
            model = MODEL_TOKENS[token]
            csv_path, json_path = output_paths(root, experiment, variant, model)
            key = f"{variant}|{model}"
            if not (csv_path.exists() and json_path.exists()):
                runs[key] = {"status": "MISSING"}
                continue
            table = read_table(csv_path)
            meta = json.loads(json_path.read_text(encoding="utf-8"))
            tables[(variant, model)] = table
            backtest = BacktestResult(table.set_index("date")[["return", "var", "es", "hit"]], 0.99)
            kupiec, christoffersen = kupiec_test(backtest), christoffersen_test(backtest)
            launcher = meta.get("launcher") or {}
            entry = {
                "status": launcher.get("status", "pre-existing"),
                "csv": str(csv_path.relative_to(root)),
                "csv_sha256": sha256_file(csv_path),
                "n_forecasts": int(len(table)),
                "first_date": table["date"].iloc[0],
                "last_date": table["date"].iloc[-1],
                "n_violations": int(table["hit"].sum()),
                "violation_rate": float(table["hit"].mean()),
                "expected_rate": 0.01,
                "kupiec_p": float(kupiec["p_value"]),
                "kupiec_reject_5pct": bool(kupiec["reject_at_5pct"]),
                "christoffersen_p": float(christoffersen["p_value"]),
                "christoffersen_reject_5pct": bool(christoffersen["reject_at_5pct"]),
                "mean_var": float(table["var"].mean()),
                "mean_es": float(table["es"].mean()),
                "max_var": float(table["var"].max()),
                "refit_count": len(meta.get("refit_logs") or []),
                "runtime_warning_total": int(sum((launcher.get("warning_counts") or {}).values())),
            }
            params = launcher.get("garch_fit_parameters") or []
            if params:
                entry["garch_max_persistence"] = max(p["persistence"] for p in params)
                entry["garch_fits_persistence_ge_0_999"] = sum(p["persistence"] >= 0.999 for p in params)
            runs[key] = entry
    paired: dict = {}
    for model in sorted({m for (_, m) in tables}):
        present = [v for v in frames if (v, model) in tables]
        for i, a in enumerate(present):
            for b in present[i + 1:]:
                ta, tb = tables[(a, model)], tables[(b, model)]
                name = f"{model}|{a}-vs-{b}"
                if ta["date"].tolist() != tb["date"].tolist():
                    paired[name] = {"error": "forecast dates differ"}
                    continue
                ha, hb = ta["hit"].to_numpy(), tb["hit"].to_numpy()
                a_only = int(((ha == 1) & (hb == 0)).sum())
                b_only = int(((ha == 0) & (hb == 1)).sum())
                diff = ta["var"].to_numpy() - tb["var"].to_numpy()
                paired[name] = {
                    "violations_a": int(ha.sum()), "violations_b": int(hb.sum()),
                    "both": int(((ha == 1) & (hb == 1)).sum()), "a_only": a_only, "b_only": b_only,
                    "mcnemar_exact_p": (float(sstats.binomtest(a_only, a_only + b_only, 0.5).pvalue)
                                        if a_only + b_only else None),
                    "mean_var_diff_a_minus_b": float(diff.mean()),
                    "share_var_a_greater": float((diff > 0).mean()),
                    "max_abs_return_difference": float(np.max(np.abs(ta["return"].to_numpy() - tb["return"].to_numpy()))),
                }
    summary = {
        "experiment": experiment,
        **LABELS[experiment],
        "input_path": EXPECTED_NSE_PATH,
        "input_sha256": input_sha,
        "var_confidence": 0.99,
        "common_evaluation": {
            "start": str(common[0].date()), "end": str(common[-1].date()), "count": len(common),
            "previous_date_mismatch_excluded": len(details["excluded_previous_date_mismatch"]),
        },
        "weekend_observations_preserved": True,
        "variants": [{"variant_id": v.variant_id, "nominal_start": v.nominal_start,
                        "return_count": len(frames[v.variant_id]),
                        "first_eligible_forecast_date": details["eligible_first_dates"][v.variant_id]}
                        for v in VARIANTS[experiment]],
        "models_requested": [MODEL_TOKENS[t] for t in tokens],
        "runs": runs,
        "paired_comparisons": paired,
    }
    path = root / "results" / experiment / "summary_priority.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    return path