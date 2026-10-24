"""Log returns and basic risk/annualization helpers.

Convention: r_t = ln(P_t / P_{t-1}), using daily close prices.
Annualization uses TRADING_DAYS = 252 (vol scales as sigma * sqrt(252),
mean scales as mu * 252).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def log_returns(prices: pd.Series) -> pd.Series:
    """Daily log returns from a price series. Drops the first (NaN) observation."""
    r = np.log(prices / prices.shift(1))
    r = r.dropna()
    r.name = "log_return"
    return r


def annualize_mean(daily_mean: float) -> float:
    return daily_mean * TRADING_DAYS


def annualize_vol(daily_vol: float) -> float:
    return daily_vol * np.sqrt(TRADING_DAYS)


def rolling_vol(returns: pd.Series, window: int = 21, annualize: bool = True) -> pd.Series:
    """Rolling realized volatility (std of log returns), optionally annualized."""
    vol = returns.rolling(window).std()
    if annualize:
        vol = vol * np.sqrt(TRADING_DAYS)
    vol.name = f"rolling_vol_{window}d"
    return vol


def summary_stats(returns: pd.Series) -> pd.Series:
    """Headline descriptive stats used throughout the stylized-facts analysis."""
    from scipy import stats as sstats

    n = len(returns)
    mean_d = returns.mean()
    std_d = returns.std(ddof=1)
    out = {
        "n_obs": n,
        "mean_daily": mean_d,
        "std_daily": std_d,
        "mean_annualized": annualize_mean(mean_d),
        "vol_annualized": annualize_vol(std_d),
        "skew": sstats.skew(returns),
        "excess_kurtosis": sstats.kurtosis(returns, fisher=True),
        "min": returns.min(),
        "max": returns.max(),
    }
    jb_stat, jb_p = sstats.jarque_bera(returns)
    out["jarque_bera_stat"] = jb_stat
    out["jarque_bera_pvalue"] = jb_p
    return pd.Series(out)
