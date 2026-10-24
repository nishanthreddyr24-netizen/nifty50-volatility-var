"""Sanity tests: GBM parameter recovery and basic VaR backtest harness checks."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.gbm import fit_gbm, parametric_es, parametric_var, simulate_gbm_paths, GBMParams
from src.var_backtest import basel_traffic_light, christoffersen_test, kupiec_test, rolling_var_backtest


def test_gbm_parameter_recovery():
    """Simulate from known (mu, sigma), refit, and check we recover them."""
    true_params = GBMParams(mu=0.08, sigma=0.20, mu_daily=0.0, sigma_daily=0.0)
    # Fill in the daily figures consistent with mu/sigma (dt = 1/252).
    dt = 1 / 252
    true_params.sigma_daily = true_params.sigma * np.sqrt(dt)
    true_params.mu_daily = (true_params.mu - 0.5 * true_params.sigma**2) * dt

    paths = simulate_gbm_paths(S0=100.0, params=true_params, n_steps=20_000, n_paths=1, seed=123)
    log_rets = np.diff(np.log(paths[0]))
    returns = pd.Series(log_rets)

    fitted = fit_gbm(returns, dt=dt)

    assert abs(fitted.mu - true_params.mu) < 0.05, f"mu off: {fitted.mu} vs {true_params.mu}"
    assert abs(fitted.sigma - true_params.sigma) < 0.02, f"sigma off: {fitted.sigma} vs {true_params.sigma}"


def test_parametric_var_increases_with_confidence():
    var95 = parametric_var(mu_daily=0.0, sigma_daily=0.01, alpha=0.95)
    var99 = parametric_var(mu_daily=0.0, sigma_daily=0.01, alpha=0.99)
    assert var99 > var95 > 0


def test_es_at_least_var():
    mu_daily, sigma_daily = 0.0, 0.01
    var99 = parametric_var(mu_daily, sigma_daily, alpha=0.99)
    es99 = parametric_es(mu_daily, sigma_daily, alpha=0.99)
    assert es99 >= var99


def test_var_zero_mean_symmetric_distribution_matches_closed_form():
    # For N(0, sigma^2), VaR_alpha = sigma * |z_{1-alpha}| exactly.
    from scipy import stats as sstats

    sigma_daily = 0.015
    alpha = 0.99
    expected = sigma_daily * abs(sstats.norm.ppf(1 - alpha))
    got = parametric_var(mu_daily=0.0, sigma_daily=sigma_daily, alpha=alpha)
    assert abs(got - expected) < 1e-10


def test_rolling_backtest_harness_no_lookahead_and_runs():
    """Construct returns where volatility jumps partway through, and check the
    harness only uses history strictly before each forecast date (no look-ahead),
    and that Kupiec/Christoffersen/traffic-light all run without error."""
    rng = np.random.default_rng(0)
    n = 1200
    sigma_low, sigma_high = 0.005, 0.03
    vol = np.where(np.arange(n) < n // 2, sigma_low, sigma_high)
    returns = pd.Series(rng.normal(0, vol), index=pd.date_range("2015-01-01", periods=n, freq="B"))

    seen_future = {"flag": False}

    def forecast_fn(history: pd.Series) -> tuple[float, float]:
        # The forecast for day t must never see sigma_high before index n//2.
        if len(history) < n // 2 and history.std() > 0.02:
            seen_future["flag"] = True
        return history.mean(), history.std()

    result = rolling_var_backtest(returns, forecast_fn, alpha=0.99, min_window=200)
    assert not seen_future["flag"], "forecast_fn saw future high-volatility data"
    assert result.n_obs == n - 200

    k = kupiec_test(result)
    c = christoffersen_test(result)
    tl = basel_traffic_light(result, window=250)

    assert "p_value" in k and "p_value" in c
    assert set(tl["zone"]).issubset({"green", "yellow", "red"})


if __name__ == "__main__":
    test_gbm_parameter_recovery()
    test_parametric_var_increases_with_confidence()
    test_es_at_least_var()
    test_var_zero_mean_symmetric_distribution_matches_closed_form()
    test_rolling_backtest_harness_no_lookahead_and_runs()
    print("All tests passed.")
