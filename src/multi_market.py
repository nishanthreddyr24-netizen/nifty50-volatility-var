"""Cross-market robustness study: the same six VaR models and the same regime
analysis as the NIFTY 50 study, run on long histories of other major indices.

Why: the NIFTY sample starts in 2008, so the 2008 GFC sits inside every model's
first training window and only one major crisis (COVID) is backtested
out-of-sample. Indices with data from 1990 put the 1997 Asian crisis, the
dot-com bust, the GFC and COVID all in the out-of-sample period, and let us
check whether the regime findings hold outside India.

Pre-2008 markets behaved differently (lower volatility in the mid-1990s and
2003-07, the dot-com bear market in between), so every backtest summary is
also split into pre-2008 / GFC / post-GFC sub-periods rather than reported
only as one long-run average.

Usage:
    python -m src.multi_market --download           # fetch and cache all series
    python -m src.multi_market --market sp500       # run one market (slow: MS-GARCH)
    python -m src.multi_market --aggregate          # build the final tables

Outputs go to reports/multi_market/. Long histories are cached under new names
(e.g. data/raw/sp500_1990.csv) so the NIFTY study's own cache is untouched.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.garch import make_garch_forecast_fn
from src.gbm import fit_gbm
from src.hmm import filtered_state_probs, fit_gaussian_hmm, make_hmm_forecast_fn
from src.hmm_garch import make_hmm_garch_forecast_fn
from src.loaders import PROJECT_ROOT, download_prices
from src import msgarch as msg
from src.returns import log_returns
from src.var_backtest import (
    BacktestResult,
    basel_traffic_light,
    christoffersen_test,
    historical_sim_var_backtest,
    kupiec_test,
    rolling_var_backtest,
)

ALPHA = 0.99
MIN_WINDOW = 500
CRASH_THRESHOLD = -0.03
OUT_DIR = PROJECT_ROOT / "reports" / "multi_market"

# name -> (ticker, start date, cache name). NIFTY uses the main study's cache and
# 2008 start so its numbers match model_comparison.py exactly.
MARKETS = {
    "sp500": ("^GSPC", "1990-01-01", "sp500_1990"),
    "ftse100": ("^FTSE", "1990-01-01", "ftse100_1990"),
    "dax": ("^GDAXI", "1990-01-01", "dax_1990"),
    "nikkei225": ("^N225", "1990-01-01", "nikkei225_1990"),
    "hangseng": ("^HSI", "1990-01-01", "hangseng_1990"),
    "sensex": ("^BSESN", "1997-07-01", "sensex_1997"),
    "nifty50": ("^NSEI", "2008-01-01", "nifty50"),
}

# Widely documented global stress episodes. Windows are approximate and the
# same for every market, so they are a common yardstick, not market-specific
# crisis dates.
CRISIS_WINDOWS = {
    "1997 Asian crisis": ("1997-07-01", "1998-01-31"),
    "1998 Russia/LTCM": ("1998-08-01", "1998-10-31"),
    "2000-02 dot-com bust": ("2000-03-10", "2002-10-09"),
    "2008 GFC": ("2008-09-01", "2009-03-31"),
    "2011 euro/US downgrade": ("2011-08-01", "2011-10-31"),
    "2015-16 China slowdown": ("2015-08-01", "2016-02-29"),
    "2020 COVID": ("2020-02-15", "2020-06-30"),
    "2022 rate-hike onset": ("2022-01-01", "2022-06-30"),
}

SUB_PERIODS = {
    "pre-2008": (None, "2007-12-31"),
    "GFC": ("2008-09-01", "2009-03-31"),
    "post-GFC": ("2009-04-01", None),
    "COVID": ("2020-02-15", "2020-06-30"),
}

MODEL_ORDER = [
    "GBM", "GARCH(1,1)", "HMM 2-state", "MS-GARCH 2-regime", "HMM-gated GARCH", "Historical simulation",
]


def load_returns(market: str) -> pd.Series:
    ticker, start, cache = MARKETS[market]
    df = download_prices(cache, ticker, start=start)
    close = df["Close"].dropna()
    close = close[close > 0]
    close = close[~close.index.duplicated(keep="first")].sort_index()
    if market == "nifty50":
        close = close.loc["2008-01-01":]
    r = log_returns(close)
    r.name = market
    return r


def data_checks(r: pd.Series) -> dict:
    """Flags for the write-up; nothing is removed automatically."""
    return {
        "n_days": len(r),
        "start": r.index[0].date().isoformat(),
        "end": r.index[-1].date().isoformat(),
        "zero_return_days": int((r == 0).sum()),
        "abs_return_gt_15pct": int((r.abs() > 0.15).sum()),
        "largest_drop": float(r.min()),
        "largest_rise": float(r.max()),
    }


def _gbm_forecast(history: pd.Series) -> tuple[float, float]:
    p = fit_gbm(history)
    return p.mu_daily, p.sigma_daily


def _model_cache(market: str, name: str) -> Path:
    safe = "".join(ch if ch.isalnum() else "_" for ch in name)
    return OUT_DIR / "cache" / f"{market}__{safe}.csv"


def run_backtests(r: pd.Series, log, market: str | None = None) -> dict[str, BacktestResult]:
    runs = [
        ("GBM", lambda: rolling_var_backtest(r, _gbm_forecast, alpha=ALPHA, min_window=MIN_WINDOW)),
        ("GARCH(1,1)", lambda: rolling_var_backtest(r, make_garch_forecast_fn(refit_every=21), alpha=ALPHA, min_window=MIN_WINDOW)),
        ("HMM 2-state", lambda: rolling_var_backtest(r, make_hmm_forecast_fn(n_states=2, refit_every=63), alpha=ALPHA, min_window=MIN_WINDOW)),
        ("HMM-gated GARCH", lambda: rolling_var_backtest(r, make_hmm_garch_forecast_fn(), alpha=ALPHA, min_window=MIN_WINDOW)),
        ("Historical simulation", lambda: historical_sim_var_backtest(r, window=MIN_WINDOW, alpha=ALPHA)),
        ("MS-GARCH 2-regime", lambda: rolling_var_backtest(r, msg.make_msgarch_forecast_fn(refit_every=63), alpha=ALPHA, min_window=MIN_WINDOW)),
    ]
    results = {}
    for name, fn in runs:
        cache = _model_cache(market, name) if market else None
        if cache is not None and cache.exists():
            table = pd.read_csv(cache, index_col=0, parse_dates=True)
            results[name] = BacktestResult(table=table, alpha=ALPHA)
            log(f"  {name} loaded from cache")
            continue
        t0 = time.time()
        results[name] = fn()
        elapsed = time.time() - t0
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            results[name].table.to_csv(cache)
            # Wall-clock time of this model's full backtest, kept with its results so
            # a cached rerun still reports the time of the run that produced them.
            pd.DataFrame([{
                "market": market, "model": name, "seconds": elapsed,
                "n_forecasts": results[name].n_obs, "n_history_days": len(r),
                "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }]).to_csv(cache.with_suffix(".time.csv"), index=False)
        log(f"  {name} done in {elapsed:.0f}s")
    return results


def _slice(result: BacktestResult, start, end) -> BacktestResult:
    return BacktestResult(table=result.table.loc[start:end], alpha=result.alpha)


def summarize_backtest(market: str, name: str, result: BacktestResult) -> dict:
    k = kupiec_test(result)
    c = christoffersen_test(result)
    zones = basel_traffic_light(result)["zone"].value_counts().to_dict()
    row = {
        "market": market,
        "model": name,
        "start": result.table.index[0].date().isoformat(),
        "n_obs": result.n_obs,
        "violations": result.n_violations,
        "expected": round(result.n_obs * (1 - ALPHA), 1),
        "rate": result.violation_rate,
        "kupiec_p": k["p_value"],
        "christoffersen_p": c["p_value"],
        "passes_both": (k["p_value"] >= 0.05) and (c["p_value"] >= 0.05),
        "basel_green": zones.get("green", 0),
        "basel_yellow": zones.get("yellow", 0),
        "basel_red": zones.get("red", 0),
    }
    for label, (s, e) in SUB_PERIODS.items():
        sub = _slice(result, s, e)
        if sub.n_obs == 0:
            row[f"{label}_hits"] = np.nan
            row[f"{label}_days"] = 0
            row[f"{label}_rate"] = np.nan
            row[f"{label}_kupiec_p"] = np.nan
            continue
        row[f"{label}_hits"] = sub.n_violations
        row[f"{label}_days"] = sub.n_obs
        row[f"{label}_rate"] = sub.violation_rate
        row[f"{label}_kupiec_p"] = kupiec_test(sub)["p_value"] if sub.n_obs >= 100 else np.nan
    return row


def regime_study(market: str, r: pd.Series) -> tuple[dict, list[dict], pd.DataFrame]:
    """Full-sample (descriptive, in-sample) regime labels: crisis = filtered P(crisis) > 0.5."""
    idx = r.index
    crash = (r <= CRASH_THRESHOLD).to_numpy()

    hmm_fit = fit_gaussian_hmm(r, n_states=2, n_iter=200, random_state=42)
    probs = filtered_state_probs(hmm_fit, r)
    ci = int(np.argmax(hmm_fit.stds))
    hmm_crisis = probs[:, ci] > 0.5
    hmm_p_stay = float(hmm_fit.transmat[ci, ci])

    ms_fit = msg.fit_msgarch(r, n_starts=4)
    path = msg.filter_path(ms_fit, r)
    pcol = [c for c in path.columns if "crisis" in c.lower() or "prob" in c.lower()][0]
    ms_crisis = path[pcol].to_numpy() > 0.5

    def spell_lengths(flag):
        lengths, run = [], 0
        for f in flag:
            if f:
                run += 1
            elif run:
                lengths.append(run); run = 0
        if run:
            lengths.append(run)
        return lengths

    hmm_spells = spell_lengths(hmm_crisis)
    ms_spells = spell_lengths(ms_crisis)
    rv = r.to_numpy()

    summary = {
        "market": market,
        "start": idx[0].date().isoformat(),
        "end": idx[-1].date().isoformat(),
        "n_days": len(r),
        "crash_days": int(crash.sum()),
        "crash_pct": crash.mean(),
        "hmm_crisis_pct": hmm_crisis.mean(),
        "hmm_calm_std": rv[~hmm_crisis].std(),
        "hmm_crisis_std": rv[hmm_crisis].std(),
        "hmm_crisis_mean": rv[hmm_crisis].mean(),
        "hmm_p_stay_crisis": hmm_p_stay,
        "hmm_n_spells": len(hmm_spells),
        "hmm_median_spell_days": float(np.median(hmm_spells)) if hmm_spells else 0.0,
        "hmm_longest_spell_days": max(hmm_spells) if hmm_spells else 0,
        "ms_crisis_pct": ms_crisis.mean(),
        "ms_calm_std": rv[~ms_crisis].std(),
        "ms_crisis_std": rv[ms_crisis].std() if ms_crisis.any() else np.nan,
        "ms_crisis_mean": rv[ms_crisis].mean() if ms_crisis.any() else np.nan,
        "ms_median_spell_days": float(np.median(ms_spells)) if ms_spells else 0.0,
        "crash_days_in_hmm_crisis": (crash & hmm_crisis).sum() / max(crash.sum(), 1),
        "crash_days_in_ms_crisis": (crash & ms_crisis).sum() / max(crash.sum(), 1),
        "hmm_ms_agree": (hmm_crisis == ms_crisis).mean(),
    }

    windows = []
    for ev, (s, e) in CRISIS_WINDOWS.items():
        mask = ((idx >= pd.Timestamp(s)) & (idx <= pd.Timestamp(e)))
        n = int(mask.sum())
        if n == 0:
            continue
        windows.append({
            "market": market,
            "window": ev,
            "days": n,
            "crash_days": int((crash & mask).sum()),
            "hmm_flagged_pct": (hmm_crisis & mask).sum() / n,
            "ms_flagged_pct": (ms_crisis & mask).sum() / n,
            "window_return": float(rv[mask].sum()),
        })

    flags = pd.DataFrame({"return": rv, "crash": crash, "hmm_crisis": hmm_crisis,
                          "hmm_p_crisis": probs[:, ci], "ms_crisis": ms_crisis}, index=idx)
    return summary, windows, flags


def run_market(market: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUT_DIR / f"{market}_log.txt"

    def log(msg_: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg_}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    r = load_returns(market)
    checks = data_checks(r)
    log(f"{market}: {checks}")

    if (OUT_DIR / f"{market}_regime_flags.csv").exists():
        log("regime study already saved, skipping")
    else:
        log("regime study (full-sample fits)...")
        summary, windows, flags = regime_study(market, r)
        pd.DataFrame([{**checks, **summary}]).to_csv(OUT_DIR / f"{market}_regimes.csv", index=False)
        pd.DataFrame(windows).to_csv(OUT_DIR / f"{market}_crisis_windows.csv", index=False)
        flags.to_csv(OUT_DIR / f"{market}_regime_flags.csv")
        log(f"regime study done: HMM crisis {summary['hmm_crisis_pct']:.1%}, MS-GARCH crisis {summary['ms_crisis_pct']:.1%}, crash days {summary['crash_days']}")

    log("backtests...")
    results = run_backtests(r, log, market=market)

    paths = pd.DataFrame({"return": next(iter(results.values())).table["return"]})
    for name, res in results.items():
        paths[f"var|{name}"] = res.table["var"].reindex(paths.index)
        paths[f"hit|{name}"] = res.table["hit"].reindex(paths.index)
    paths.to_csv(OUT_DIR / f"{market}_var_paths.csv")

    rows = [summarize_backtest(market, n, results[n]) for n in MODEL_ORDER]
    pd.DataFrame(rows).to_csv(OUT_DIR / f"{market}_backtest.csv", index=False)
    log("done")


def _tick_loss(r: np.ndarray, var: np.ndarray, tau: float = 1 - ALPHA) -> np.ndarray:
    """Quantile (tick) loss of the VaR forecast: q = -VaR is the forecast tau-quantile.
    Strictly consistent for the quantile, so it ranks forecasts by accuracy rather
    than only checking coverage. Lower is better."""
    q = -var
    return (tau - (r < q).astype(float)) * (r - q)


def _diebold_mariano(d: np.ndarray, lag: int = 5) -> tuple[float, float]:
    """DM test on the loss differential d (model minus benchmark), Newey-West
    variance with `lag` lags. Returns (statistic, two-sided p-value)."""
    from scipy import stats as sstats

    d = d[np.isfinite(d)]
    n = len(d)
    dc = d - d.mean()
    gamma0 = dc @ dc / n
    lr_var = gamma0
    for k in range(1, lag + 1):
        gk = dc[k:] @ dc[:-k] / n
        lr_var += 2 * (1 - k / (lag + 1)) * gk
    stat = d.mean() / np.sqrt(lr_var / n)
    return float(stat), float(2 * (1 - sstats.norm.cdf(abs(stat))))


def forecast_skill() -> pd.DataFrame:
    rows = []
    for p in sorted(OUT_DIR.glob("*_var_paths.csv")):
        market = p.name.replace("_var_paths.csv", "")
        paths = pd.read_csv(p, index_col=0, parse_dates=True)
        r = paths["return"].to_numpy()
        losses = {m: _tick_loss(r, paths[f"var|{m}"].to_numpy()) for m in MODEL_ORDER}
        gbm_mean = np.nanmean(losses["GBM"])
        for m in MODEL_ORDER:
            stat, pval = _diebold_mariano(losses[m] - losses["HMM 2-state"]) if m != "HMM 2-state" else (np.nan, np.nan)
            rows.append({
                "market": market, "model": m,
                "mean_tick_loss_x1e4": np.nanmean(losses[m]) * 1e4,
                "skill_vs_GBM_pct": (1 - np.nanmean(losses[m]) / gbm_mean) * 100,
                "DM_vs_HMM_stat": stat, "DM_vs_HMM_p": pval,
            })
    df = pd.DataFrame(rows)
    df["rank_in_market"] = df.groupby("market")["mean_tick_loss_x1e4"].rank()
    return df


def aggregate() -> None:
    bt = pd.concat([pd.read_csv(p) for p in sorted(OUT_DIR.glob("*_backtest.csv"))], ignore_index=True)
    rg = pd.concat([pd.read_csv(p) for p in sorted(OUT_DIR.glob("*_regimes.csv"))], ignore_index=True)
    cw = pd.concat([pd.read_csv(p) for p in sorted(OUT_DIR.glob("*_crisis_windows.csv"))], ignore_index=True)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)

    bt.to_csv(OUT_DIR / "ALL_backtest.csv", index=False)
    rg.to_csv(OUT_DIR / "ALL_regimes.csv", index=False)
    cw.to_csv(OUT_DIR / "ALL_crisis_windows.csv", index=False)

    print("\n=== Violation rate by market (expected 1.00%) ===")
    print((bt.pivot(index="market", columns="model", values="rate")[MODEL_ORDER] * 100).round(2).to_string())

    print("\n=== Passes both Kupiec and Christoffersen at 5% ===")
    print(bt.pivot(index="market", columns="model", values="passes_both")[MODEL_ORDER].to_string())
    print("\nModels passing both tests, count of markets:")
    print(bt.groupby("model")["passes_both"].sum().reindex(MODEL_ORDER).to_string())

    for label in ["pre-2008", "GFC", "post-GFC", "COVID"]:
        print(f"\n=== {label}: violation rate % (hits/days) ===")
        t = bt.assign(cell=lambda d: (d[f"{label}_rate"] * 100).round(2).astype(str)
                      + " (" + d[f"{label}_hits"].fillna(-1).astype(int).astype(str)
                      + "/" + d[f"{label}_days"].astype(int).astype(str) + ")")
        t.loc[t[f"{label}_days"] == 0, "cell"] = "n/a"
        print(t.pivot(index="market", columns="model", values="cell")[MODEL_ORDER].to_string())

    print("\n=== Regimes and crash days per market (full sample, descriptive) ===")
    cols = ["market", "start", "n_days", "crash_days", "crash_pct", "hmm_crisis_pct", "hmm_calm_std",
            "hmm_crisis_std", "hmm_median_spell_days", "ms_crisis_pct", "ms_crisis_std",
            "crash_days_in_hmm_crisis", "crash_days_in_ms_crisis", "hmm_ms_agree"]
    print(rg[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n=== Share of each crisis window flagged by the HMM crisis regime ===")
    print((cw.pivot(index="window", columns="market", values="hmm_flagged_pct") * 100).round(0)
          .reindex(list(CRISIS_WINDOWS)).to_string())
    print("\n=== Share of each crisis window flagged by the MS-GARCH crisis regime ===")
    print((cw.pivot(index="window", columns="market", values="ms_flagged_pct") * 100).round(0)
          .reindex(list(CRISIS_WINDOWS)).to_string())

    timing_files = sorted((OUT_DIR / "cache").glob("*.time.csv"))
    if timing_files:
        tm = pd.concat([pd.read_csv(p) for p in timing_files], ignore_index=True)
        tm["ms_per_forecast"] = tm["seconds"] / tm["n_forecasts"] * 1000
        tm.to_csv(OUT_DIR / "ALL_timings.csv", index=False)
        print("\n=== Wall-clock time per model's full backtest (minutes) ===")
        print((tm.pivot(index="market", columns="model", values="seconds")
               .reindex(columns=MODEL_ORDER) / 60).round(1).to_string())
        print("\n=== Milliseconds per daily forecast ===")
        print(tm.pivot(index="market", columns="model", values="ms_per_forecast")
              .reindex(columns=MODEL_ORDER).round(1).to_string())
        print("\nTotal minutes per model across all markets:")
        print((tm.groupby("model")["seconds"].sum().reindex(MODEL_ORDER) / 60).round(1).to_string())
        print("(Markets run in parallel on one machine, so times include some CPU contention.)")

    sk = forecast_skill()
    sk.to_csv(OUT_DIR / "ALL_forecast_skill.csv", index=False)
    print("\n=== Forecast skill: % reduction in 1% quantile (tick) loss vs GBM (higher is better) ===")
    print(sk.pivot(index="market", columns="model", values="skill_vs_GBM_pct")[MODEL_ORDER].round(1).to_string())
    print("\n=== Rank by tick loss within each market (1 = best) ===")
    ranks = sk.pivot(index="market", columns="model", values="rank_in_market")[MODEL_ORDER]
    print(ranks.to_string())
    print("\nMean rank across markets:")
    print(ranks.mean().sort_values().round(2).to_string())
    print("\n=== Diebold-Mariano vs HMM: statistic (negative = model beats HMM), p-value ===")
    dm = sk[sk.model != "HMM 2-state"].assign(cell=lambda d: d.DM_vs_HMM_stat.round(2).astype(str)
                                               + " (p=" + d.DM_vs_HMM_p.round(3).astype(str) + ")")
    print(dm.pivot(index="market", columns="model", values="cell")[[m for m in MODEL_ORDER if m != "HMM 2-state"]].to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--market", choices=list(MARKETS))
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    if a.download:
        for m in MARKETS:
            print(m, data_checks(load_returns(m)))
    if a.market:
        run_market(a.market)
    if a.aggregate:
        aggregate()


if __name__ == "__main__":
    main()
