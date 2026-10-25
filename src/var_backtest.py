"""Model-agnostic rolling VaR backtesting harness.

Any model plugs in by providing a `forecast_fn(history: pd.Series) -> (mu_daily, sigma_daily)`
that, given only returns strictly before date t, forecasts the 1-day-ahead
mean and volatility used for t's VaR. This enforces no look-ahead bias by
construction: the rolling loop only ever passes `returns.loc[:t_minus_1]` in.

Includes:
  - rolling_var_backtest: walks forward, computes VaR/ES, actual return, and
    a hit indicator (1 if the loss exceeded VaR).
  - kupiec_test: unconditional coverage (POF) test.
  - christoffersen_test: conditional coverage (independence of hits) test.
  - basel_traffic_light: zone classification over a 250-day window.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats as sstats


@dataclass
class BacktestResult:
    table: pd.DataFrame  # index=date, columns=[return, var, es, hit]
    alpha: float

    @property
    def n_obs(self) -> int:
        return len(self.table)

    @property
    def n_violations(self) -> int:
        return int(self.table["hit"].sum())

    @property
    def violation_rate(self) -> float:
        return self.n_violations / self.n_obs if self.n_obs else np.nan


def rolling_var_backtest(
    returns: pd.Series,
    forecast_fn: Callable[[pd.Series], tuple[float, float]],
    alpha: float = 0.99,
    min_window: int = 500,
    step: int = 1,
    var_fn: Callable[[float, float, float], float] | None = None,
    es_fn: Callable[[float, float, float], float] | None = None,
) -> BacktestResult:
    """Walk forward through `returns`, refitting/forecasting at each step.

    Parameters
    ----------
    returns : full log-return series, chronologically sorted.
    forecast_fn : given returns[:t] (strictly before t), returns (mu_daily, sigma_daily)
        for the forecast of day t. This is where look-ahead bias is prevented:
        the caller never sees returns.loc[t] or later.
    alpha : VaR confidence level (e.g. 0.99 for 99% VaR).
    min_window : minimum number of observations before the first forecast.
    step : stride between forecast dates, 1 = every day (slow but thorough).
    var_fn, es_fn : map (mu_daily, sigma_daily, alpha) -> VaR / ES (positive loss
        fractions). Defaults to the parametric normal formulas from gbm.py.
    """
    if var_fn is None or es_fn is None:
        from src.gbm import parametric_es, parametric_var

        var_fn = var_fn or (lambda mu, sigma, a: parametric_var(mu, sigma, a))
        es_fn = es_fn or (lambda mu, sigma, a: parametric_es(mu, sigma, a))

    dates, actual, var_list, es_list = [], [], [], []
    n = len(returns)
    for i in range(min_window, n, step):
        history = returns.iloc[:i]  # strictly before returns.index[i]
        t = returns.index[i]
        r_t = returns.iloc[i]

        mu_hat, sigma_hat = forecast_fn(history)
        var_t = var_fn(mu_hat, sigma_hat, alpha)
        es_t = es_fn(mu_hat, sigma_hat, alpha)

        dates.append(t)
        actual.append(r_t)
        var_list.append(var_t)
        es_list.append(es_t)

    table = pd.DataFrame(
        {"return": actual, "var": var_list, "es": es_list}, index=pd.Index(dates, name="date")
    )
    # Hit = 1 if the realized loss (-return) exceeded the predicted VaR.
    table["hit"] = (-table["return"] > table["var"]).astype(int)
    return BacktestResult(table=table, alpha=alpha)


def kupiec_test(result: BacktestResult) -> dict:
    """Kupiec (1995) proportion-of-failures (unconditional coverage) test.

    H0: true violation probability equals (1 - alpha).
    LR_POF = -2 * ln[ (1-p)^(n-x) p^x / (1-pi_hat)^(n-x) pi_hat^x ]  ~ chi2(1)
    where p = 1 - alpha (expected violation rate), x = violations, n = obs,
    pi_hat = x / n (observed violation rate).
    """
    n = result.n_obs
    x = result.n_violations
    p = 1 - result.alpha
    pi_hat = x / n if n else np.nan

    if x == 0:
        log_l_null = n * np.log(1 - p)
        log_l_alt = n * np.log(1 - pi_hat) if pi_hat < 1 else 0.0
    elif x == n:
        log_l_null = n * np.log(p)
        log_l_alt = n * np.log(pi_hat)
    else:
        log_l_null = (n - x) * np.log(1 - p) + x * np.log(p)
        log_l_alt = (n - x) * np.log(1 - pi_hat) + x * np.log(pi_hat)

    lr_stat = -2 * (log_l_null - log_l_alt)
    p_value = 1 - sstats.chi2.cdf(lr_stat, df=1)

    return {
        "test": "Kupiec POF",
        "n_obs": n,
        "n_violations": x,
        "expected_violations": n * p,
        "violation_rate": pi_hat,
        "expected_rate": p,
        "LR_stat": lr_stat,
        "p_value": p_value,
        "reject_at_5pct": p_value < 0.05,
    }


def christoffersen_test(result: BacktestResult) -> dict:
    """Christoffersen (1998) test of independence of VaR violations (conditional coverage).

    Models the hit sequence as a 2-state Markov chain and tests H0: transition
    probability out of "no violation" equals transition probability out of
    "violation" (i.e. violations don't cluster). LR_ind ~ chi2(1).
    """
    hits = result.table["hit"].to_numpy()
    n00 = n01 = n10 = n11 = 0
    for prev, cur in zip(hits[:-1], hits[1:]):
        if prev == 0 and cur == 0:
            n00 += 1
        elif prev == 0 and cur == 1:
            n01 += 1
        elif prev == 1 and cur == 0:
            n10 += 1
        else:
            n11 += 1

    n0 = n00 + n01
    n1 = n10 + n11
    pi01 = n01 / n0 if n0 else 0.0
    pi11 = n11 / n1 if n1 else 0.0
    pi = (n01 + n11) / (n0 + n1) if (n0 + n1) else 0.0

    def _ll(p, k, m):
        # log-likelihood contribution of k successes out of m trials under prob p
        if m == 0:
            return 0.0
        if p <= 0 or p >= 1:
            return 0.0 if k in (0, m) else -np.inf
        return k * np.log(p) + (m - k) * np.log(1 - p)

    log_l_null = _ll(pi, n01, n0) + _ll(pi, n11, n1)
    log_l_alt = _ll(pi01, n01, n0) + _ll(pi11, n11, n1)

    lr_stat = -2 * (log_l_null - log_l_alt)
    lr_stat = max(lr_stat, 0.0)
    p_value = 1 - sstats.chi2.cdf(lr_stat, df=1)

    return {
        "test": "Christoffersen independence",
        "n00": n00, "n01": n01, "n10": n10, "n11": n11,
        "pi01": pi01, "pi11": pi11,
        "LR_stat": lr_stat,
        "p_value": p_value,
        "reject_at_5pct": p_value < 0.05,
    }


def historical_sim_var_backtest(
    returns: pd.Series, window: int = 250, alpha: float = 0.99, step: int = 1
) -> BacktestResult:
    """Historical simulation VaR: a non-parametric benchmark. VaR_t is minus the
    empirical (1-alpha) quantile of the trailing `window` returns strictly
    before t (no distributional assumption at all, unlike GBM/HMM/GARCH's
    parametric normal or Student-t VaR). ES_t is the mean of the returns
    beyond that quantile.

    This doesn't fit the `forecast_fn(history) -> (mu, sigma)` plug-in shape
    used by `rolling_var_backtest` (there's no mu/sigma, just a trailing
    empirical quantile), so it builds a BacktestResult directly; the result
    is still fully compatible with kupiec_test / christoffersen_test /
    basel_traffic_light below.
    """
    dates, actual, var_list, es_list = [], [], [], []
    n = len(returns)
    for i in range(window, n, step):
        history = returns.iloc[i - window : i]  # strictly before returns.index[i]
        t = returns.index[i]
        r_t = returns.iloc[i]

        q = np.quantile(history, 1 - alpha)
        var_t = -q
        tail = history[history <= q]
        es_t = -tail.mean() if len(tail) else var_t

        dates.append(t)
        actual.append(r_t)
        var_list.append(var_t)
        es_list.append(es_t)

    table = pd.DataFrame(
        {"return": actual, "var": var_list, "es": es_list}, index=pd.Index(dates, name="date")
    )
    table["hit"] = (-table["return"] > table["var"]).astype(int)
    return BacktestResult(table=table, alpha=alpha)


def basel_traffic_light(result: BacktestResult, window: int = 250) -> pd.DataFrame:
    """Basel traffic-light classification of violations within each trailing
    `window`-day block (non-overlapping), at the 99% level convention:
    0-4 green, 5-9 yellow, 10+ red. Returns a DataFrame with one row per block.
    """
    hits = result.table["hit"]
    n = len(hits)
    rows = []
    for start in range(0, n, window):
        block = hits.iloc[start : start + window]
        if len(block) < window:
            break  # partial trailing block, skip
        x = int(block.sum())
        if x <= 4:
            zone = "green"
        elif x <= 9:
            zone = "yellow"
        else:
            zone = "red"
        rows.append(
            {
                "block_start": block.index[0],
                "block_end": block.index[-1],
                "n_violations": x,
                "zone": zone,
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.gbm import fit_gbm
    from src.loaders import load_close
    from src.returns import log_returns

    prices = load_close("nifty50")
    r = log_returns(prices)

    def gbm_forecast(history: pd.Series) -> tuple[float, float]:
        params = fit_gbm(history)
        return params.mu_daily, params.sigma_daily

    result = rolling_var_backtest(r, gbm_forecast, alpha=0.99, min_window=500)
    print(f"Backtest: {result.n_obs} obs, {result.n_violations} violations, rate={result.violation_rate:.4%}")

    kupiec = kupiec_test(result)
    print("Kupiec:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in kupiec.items()})

    christoffersen = christoffersen_test(result)
    print("Christoffersen:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in christoffersen.items()})

    traffic = basel_traffic_light(result)
    print(traffic["zone"].value_counts())
