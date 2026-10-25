"""Side-by-side out-of-sample VaR backtest comparison: GBM, HMM, GARCH family,
and historical simulation, all on the same NIFTY 50 sample and the same
min_window/alpha, using the shared harness in var_backtest.py.

Note on the 2008 stress window: the price history starts 2008-01-01, so there
is no pre-2008 data to train any model on -- the Sep 2008-Mar 2009 crash
itself cannot be backtested out-of-sample here without look-ahead (it falls
inside min_window, i.e. it's used only for the first fit, never forecasted
against). This is a genuine data limitation, not a bug: the report should say
so rather than silently backtest it anyway. The Mar 2020 COVID crash, with 12
years of prior training data, is the real stress test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.garch import make_garch_forecast_fn, student_t_es, student_t_var, fit_garch
from src.gbm import fit_gbm
from src.hmm import make_hmm_forecast_fn
from src.loaders import load_close
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


def window_violations(result: BacktestResult, start: str, end: str) -> tuple[int, int]:
    sub = result.table.loc[start:end]
    return int(sub["hit"].sum()), len(sub)


def summarize(name: str, result: BacktestResult) -> dict:
    kupiec = kupiec_test(result)
    christ = christoffersen_test(result)
    traffic = basel_traffic_light(result)
    zone_counts = traffic["zone"].value_counts().to_dict() if len(traffic) else {}

    covid_hits, covid_n = window_violations(result, "2020-02-15", "2020-06-30")

    return {
        "model": name,
        "n_obs": result.n_obs,
        "n_violations": result.n_violations,
        "violation_rate": result.violation_rate,
        "kupiec_p": kupiec["p_value"],
        "kupiec_reject": kupiec["reject_at_5pct"],
        "christoffersen_p": christ["p_value"],
        "christoffersen_reject": christ["reject_at_5pct"],
        "traffic_green": zone_counts.get("green", 0),
        "traffic_yellow": zone_counts.get("yellow", 0),
        "traffic_red": zone_counts.get("red", 0),
        "covid_violations": f"{covid_hits}/{covid_n}",
    }


def main() -> pd.DataFrame:
    prices = load_close("nifty50")
    returns = log_returns(prices)

    def gbm_forecast(history: pd.Series) -> tuple[float, float]:
        params = fit_gbm(history)
        return params.mu_daily, params.sigma_daily

    print("Running backtests (this refits each model hundreds of times; a minute or two)...")

    results: dict[str, BacktestResult] = {}

    results["GBM (expanding window)"] = rolling_var_backtest(
        returns, gbm_forecast, alpha=ALPHA, min_window=MIN_WINDOW
    )
    print("  GBM done")

    results["HMM 2-state"] = rolling_var_backtest(
        returns, make_hmm_forecast_fn(n_states=2, refit_every=63), alpha=ALPHA, min_window=MIN_WINDOW
    )
    print("  HMM done")

    results["GARCH(1,1) normal"] = rolling_var_backtest(
        returns, make_garch_forecast_fn(o=0, dist="normal", refit_every=21), alpha=ALPHA, min_window=MIN_WINDOW
    )
    print("  GARCH(1,1) done")

    results["GJR-GARCH normal"] = rolling_var_backtest(
        returns, make_garch_forecast_fn(o=1, dist="normal", refit_every=21), alpha=ALPHA, min_window=MIN_WINDOW
    )
    print("  GJR-GARCH normal done")

    nu_fixed = fit_garch(returns, o=1, dist="t").nu
    results["GJR-GARCH Student-t"] = rolling_var_backtest(
        returns,
        make_garch_forecast_fn(o=1, dist="t", refit_every=21),
        alpha=ALPHA,
        min_window=MIN_WINDOW,
        var_fn=lambda mu, sigma, a: student_t_var(mu, sigma, a, nu_fixed),
        es_fn=lambda mu, sigma, a: student_t_es(mu, sigma, a, nu_fixed),
    )
    print("  GJR-GARCH Student-t done")

    results["Historical simulation"] = historical_sim_var_backtest(returns, window=MIN_WINDOW, alpha=ALPHA)
    print("  Historical simulation done")

    summary = pd.DataFrame([summarize(name, r) for name, r in results.items()])
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    print("\n" + "=" * 100)
    print(f"Out-of-sample 1-day {ALPHA:.0%} VaR backtest comparison, NIFTY 50, min_window={MIN_WINDOW}")
    print("Expected violation rate under correct calibration: {:.2%}".format(1 - ALPHA))
    print("=" * 100)
    print(summary.to_string(index=False))

    print(
        "\nNote: the 2008 GFC (Sep 2008-Mar 2009) falls inside every model's "
        f"first {MIN_WINDOW}-day training window (the sample starts 2008-01-01, "
        "so there is no earlier data to train on) and is therefore not "
        "backtestable out-of-sample here without look-ahead. The Mar-Jun 2020 "
        "COVID crash column above is the real stress test."
    )

    return summary


if __name__ == "__main__":
    main()
