"""Geometric Brownian Motion baseline.

Model: dS_t = mu * S_t dt + sigma * S_t dW_t, so log-returns
r_t = ln(S_t / S_{t-1}) ~ iid Normal(mu - sigma^2/2, sigma^2) under the exact
discretization with dt = 1 day. We estimate (mu, sigma) by MLE on daily log
returns (closed form: sample mean and sample std), simulate paths with the
exact (not Euler) discretization, and compute parametric normal VaR/ES.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sstats


@dataclass
class GBMParams:
    mu: float          # annualized drift of the underlying GBM (not of log-returns)
    sigma: float       # annualized volatility
    mu_daily: float     # daily log-return mean, i.e. (mu - 0.5*sigma^2) * dt
    sigma_daily: float  # daily log-return std


def fit_gbm(returns: pd.Series, dt: float = 1 / 252) -> GBMParams:
    """MLE fit of GBM to a series of daily log returns.

    For iid Normal log returns, MLE of (mean, std) is just the sample mean
    and sample std (ddof=0 for MLE; we report ddof=1 sample std, the standard
    practical choice, and note this is a minor MLE-vs-unbiased distinction).
    """
    mu_daily = returns.mean()
    sigma_daily = returns.std(ddof=1)

    sigma_annual = sigma_daily * np.sqrt(1 / dt)
    mu_annual = mu_daily / dt + 0.5 * sigma_annual**2  # invert r = (mu - 0.5*sigma^2)*dt

    return GBMParams(
        mu=mu_annual,
        sigma=sigma_annual,
        mu_daily=mu_daily,
        sigma_daily=sigma_daily,
    )


def simulate_gbm_paths(
    S0: float,
    params: GBMParams,
    n_steps: int,
    n_paths: int,
    dt: float = 1 / 252,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate GBM price paths using the exact discretization:
    S_{t+1} = S_t * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z),  Z ~ N(0,1).

    Returns an array of shape (n_paths, n_steps + 1), column 0 = S0.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_paths, n_steps))
    drift = (params.mu - 0.5 * params.sigma**2) * dt
    diffusion = params.sigma * np.sqrt(dt) * z
    log_increments = drift + diffusion

    log_paths = np.cumsum(log_increments, axis=1)
    log_paths = np.hstack([np.zeros((n_paths, 1)), log_paths])
    paths = S0 * np.exp(log_paths)
    return paths


def parametric_var(mu_daily: float, sigma_daily: float, alpha: float = 0.99, horizon: int = 1) -> float:
    """Parametric (normal) 1-step-ahead VaR as a positive loss fraction.

    VaR_alpha solves P(r < -VaR_alpha) = 1 - alpha for r ~ N(mu_daily, sigma_daily^2)
    scaled to `horizon` days under the sqrt-time rule (mean scales linearly,
    vol scales with sqrt(horizon)).
    """
    mu_h = mu_daily * horizon
    sigma_h = sigma_daily * np.sqrt(horizon)
    z = sstats.norm.ppf(1 - alpha)  # negative number
    var = -(mu_h + sigma_h * z)
    return var


def parametric_es(mu_daily: float, sigma_daily: float, alpha: float = 0.99, horizon: int = 1) -> float:
    """Parametric (normal) Expected Shortfall (average loss beyond VaR), positive number."""
    mu_h = mu_daily * horizon
    sigma_h = sigma_daily * np.sqrt(horizon)
    z = sstats.norm.ppf(1 - alpha)
    es = -(mu_h - sigma_h * sstats.norm.pdf(z) / (1 - alpha))
    return es


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.loaders import load_close
    from src.returns import log_returns

    prices = load_close("nifty50")
    r = log_returns(prices)

    params = fit_gbm(r)
    print(f"GBM fit: mu_annual={params.mu:.4f}, sigma_annual={params.sigma:.4f}")
    print(f"         mu_daily={params.mu_daily:.6f}, sigma_daily={params.sigma_daily:.6f}")

    var99 = parametric_var(params.mu_daily, params.sigma_daily, alpha=0.99)
    es99 = parametric_es(params.mu_daily, params.sigma_daily, alpha=0.99)
    print(f"1-day 99% VaR: {var99:.4%}, ES: {es99:.4%}")

    paths = simulate_gbm_paths(float(prices.iloc[-1]), params, n_steps=252, n_paths=5, seed=42)
    print(f"Simulated 5 paths of 252 steps, final values: {paths[:, -1]}")
