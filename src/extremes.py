"""Extreme-event (crash-day) analysis: plain Poisson baseline, an overdispersion
test, and a self-exciting Hawkes process as the alternative when extremes cluster
in time (as the stylized facts / volatility-clustering analysis predicts they will).

A "crash day" is defined by a fixed return threshold (not a regime label, and
not estimated from the same data used to test it -- this is a separate,
simple, descriptive analysis, so there is no circularity concern the way
there would be if "regime" were defined by a volatility threshold).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sstats
from scipy.optimize import minimize


def crash_days(returns: pd.Series, threshold: float = -0.03) -> pd.Series:
    """Boolean series: True on days where the log return fell below `threshold`."""
    return returns < threshold


def annual_counts(crash: pd.Series) -> pd.Series:
    """Number of crash days per calendar year."""
    return crash.groupby(crash.index.year).sum()


def poisson_fit_and_dispersion_test(counts: pd.Series) -> dict:
    """Fit a homogeneous Poisson rate (MLE = sample mean) to per-period counts,
    and test equidispersion (variance == mean, the Poisson assumption) via the
    classical dispersion statistic D = sum((x_i - lambda_hat)^2 / lambda_hat),
    which is chi2(n-1) distributed under H0. D >> n-1 indicates overdispersion
    (events cluster in time rather than occurring as an homogeneous Poisson
    process), motivating a self-exciting (Hawkes) alternative.
    """
    n = len(counts)
    lambda_hat = counts.mean()
    var_hat = counts.var(ddof=1)

    D = float(((counts - lambda_hat) ** 2 / lambda_hat).sum())
    df = n - 1
    p_value = 1 - sstats.chi2.cdf(D, df)

    return {
        "n_periods": n,
        "lambda_hat": lambda_hat,
        "variance": var_hat,
        "dispersion_index": var_hat / lambda_hat,
        "D_stat": D,
        "df": df,
        "p_value": p_value,
        "overdispersed_at_5pct": p_value < 0.05,
    }


@dataclass
class HawkesFit:
    mu: float      # background (exogenous) intensity, events/day
    alpha: float   # jump in intensity per event
    beta: float    # exponential decay rate of excitation
    branching_ratio: float  # alpha/beta: expected number of "child" events triggered per event
    log_likelihood: float


def _hawkes_neg_log_likelihood(params: np.ndarray, t: np.ndarray, T: float) -> float:
    mu, alpha, beta = params
    if mu <= 0 or alpha < 0 or beta <= alpha:  # beta > alpha required for stability/integrability here
        return 1e10

    n = len(t)
    # Recursive evaluation of the compensator sum (standard O(n) trick): for each
    # event i, R_i = sum_{j<i} exp(-beta*(t_i - t_j)) = exp(-beta*(t_i - t_{i-1})) * (1 + R_{i-1}).
    R = np.zeros(n)
    for i in range(1, n):
        R[i] = np.exp(-beta * (t[i] - t[i - 1])) * (1 + R[i - 1])

    intensities = mu + alpha * R
    if np.any(intensities <= 0):
        return 1e10
    log_sum = np.sum(np.log(intensities))

    integral = mu * T + (alpha / beta) * np.sum(1 - np.exp(-beta * (T - t)))

    return -(log_sum - integral)


def fit_hawkes(event_days: np.ndarray, T: float) -> HawkesFit:
    """MLE fit of a univariate Hawkes process with exponential kernel
    lambda(t) = mu + alpha * sum_{t_i < t} exp(-beta*(t - t_i))
    to event times `event_days` (as a float day-index into [0, T])."""
    t = np.sort(np.asarray(event_days, dtype=float))
    n = len(t)

    # Reasonable starting values: mu ~ unconditional rate, mild self-excitation.
    mu0 = n / T
    x0 = np.array([mu0 * 0.7, mu0 * 0.5, 1.0])
    bounds = [(1e-6, None), (0.0, None), (1e-3, None)]

    res = minimize(
        _hawkes_neg_log_likelihood, x0, args=(t, T), method="L-BFGS-B", bounds=bounds
    )
    mu, alpha, beta = res.x
    branching_ratio = alpha / beta if beta > 0 else np.nan

    return HawkesFit(mu=mu, alpha=alpha, beta=beta, branching_ratio=branching_ratio, log_likelihood=-res.fun)


def poisson_log_likelihood(event_days: np.ndarray, T: float) -> float:
    """Log-likelihood of a homogeneous Poisson process with MLE rate n/T,
    for a fair AIC/BIC comparison against the Hawkes fit."""
    n = len(event_days)
    rate = n / T
    return n * np.log(rate) - rate * T


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.loaders import load_close
    from src.returns import log_returns

    prices = load_close("nifty50")
    returns = log_returns(prices)

    threshold = -0.03
    crash = crash_days(returns, threshold=threshold)
    print(f"Crash-day threshold: daily log return < {threshold:.1%}")
    print(f"Total crash days: {crash.sum()} out of {len(returns)} ({crash.mean():.2%})")

    counts = annual_counts(crash)
    print("\nCrash days per calendar year:")
    print(counts)

    print("\n=== Poisson fit + equidispersion test ===")
    test = poisson_fit_and_dispersion_test(counts)
    for k, v in test.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n=== Hawkes process (self-exciting) fit ===")
    t0 = returns.index[0]
    event_days = np.array([(d - t0).days for d in returns.index[crash.values]], dtype=float)
    T = float((returns.index[-1] - t0).days)

    hawkes = fit_hawkes(event_days, T)
    print(f"  mu (background rate) = {hawkes.mu:.5f} events/day  (~{hawkes.mu * 252:.1f}/year)")
    print(f"  alpha (excitation jump) = {hawkes.alpha:.5f}")
    print(f"  beta (decay rate) = {hawkes.beta:.5f}  (half-life ~ {np.log(2)/hawkes.beta:.1f} days)")
    print(f"  branching ratio alpha/beta = {hawkes.branching_ratio:.3f}  (expected 'children' per event)")
    print(f"  log-likelihood = {hawkes.log_likelihood:.2f}")

    poisson_ll = poisson_log_likelihood(event_days, T)
    print(f"\n  Poisson log-likelihood (for comparison) = {poisson_ll:.2f}")
    lr_stat = 2 * (hawkes.log_likelihood - poisson_ll)
    lr_p = 1 - sstats.chi2.cdf(lr_stat, df=2)  # Hawkes has 2 extra params (alpha, beta) vs Poisson's 1
    print(f"  LR test (Hawkes vs Poisson): stat={lr_stat:.2f}, p-value={lr_p:.2e}")
    print(f"  -> Hawkes preferred at 5%: {lr_p < 0.05}")
