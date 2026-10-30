"""Two-regime Markov-switching GARCH(1,1) (Haas, Mittnik & Paolella, 2004),
normal innovations -- the joint GARCH + HMM model.

Each regime k in {0 = calm, 1 = crisis} carries its own GARCH(1,1) variance,
and all regime variances are updated in parallel every day from the
regime's own residual:

    sigma2_{k,t+1} = omega_k + alpha_k * (r_t - mu_k)^2 + beta_k * sigma2_{k,t}

A hidden 2-state Markov chain (transition matrix P) picks which regime
generates each return: r_t | s_t = k ~ N(mu_k, sigma2_{k,t}). The parallel
recursion (Haas et al.) is what keeps the likelihood tractable -- in the
original Gray (1996) formulation each variance depends on the whole regime
path. The likelihood is evaluated with the Hamilton filter, exactly as in
hmm.py, except that the per-regime variances now move every day.

Both existing models are nested special cases: alpha_k = beta_k = 0 gives
the Gaussian HMM in hmm.py (constant variance per regime), and identical
regimes give the single-regime GARCH(1,1) in garch.py.

As in garch.py, parameters are estimated on returns*100 (percent scale) for
numerical stability and converted back immediately, so everything outside
this module works in plain log-return units. Regimes are sorted by
long-run volatility after fitting (0 = calm), a post-hoc relabelling that
uses no future information.

The forecast function used by the rolling VaR backtest refits periodically
(warm-started from the previous fit) and between refits advances the filter
one observation at a time with fixed parameters, so nothing it uses is ever
drawn from data at or after the forecast date.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

SCALE = 100.0  # fit on returns * SCALE, as in garch.py
MAX_PERSISTENCE = 0.999  # alpha_k + beta_k is kept strictly below this
_LOG_2PI = math.log(2 * math.pi)

# theta layout (unconstrained, percent scale):
#   [mu0, mu1, log_omega0, log_omega1, persist0, persist1, share0, share1, logit_p00, logit_p11]
# alpha_k + beta_k = MAX_PERSISTENCE * expit(persist_k), alpha_k = (alpha_k + beta_k) * expit(share_k)
_REGIME_SWAP = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8]


@dataclass
class MSGARCHFit:
    mu: np.ndarray        # regime means, daily, original return scale
    omega: np.ndarray     # regime GARCH intercepts, original return scale
    alpha: np.ndarray
    beta: np.ndarray
    transmat: np.ndarray  # transmat[i, j] = P(s_{t+1} = j | s_t = i)
    log_likelihood: float  # at the fitted parameters, percent scale
    theta: np.ndarray     # unconstrained parameter vector, kept for warm starts
    # Filter state after the last observation, used for one-step forecasts:
    sigma2_next: np.ndarray  # each regime's variance for the next day
    prob_crisis: float       # filtered P(s_T = 1 | returns through T)

    @property
    def persistence(self) -> np.ndarray:
        return self.alpha + self.beta

    @property
    def long_run_vol(self) -> np.ndarray:
        """Daily long-run std of each regime's own GARCH, sqrt(omega / (1 - alpha - beta))."""
        return np.sqrt(self.omega / (1 - self.persistence))


def _unpack(theta: np.ndarray) -> tuple:
    mu0, mu1 = theta[0], theta[1]
    w0, w1 = math.exp(theta[2]), math.exp(theta[3])
    pers0 = MAX_PERSISTENCE * expit(theta[4])
    pers1 = MAX_PERSISTENCE * expit(theta[5])
    a0, a1 = pers0 * expit(theta[6]), pers1 * expit(theta[7])
    b0, b1 = pers0 - a0, pers1 - a1
    p00, p11 = expit(theta[8]), expit(theta[9])
    return mu0, mu1, w0, w1, a0, a1, b0, b1, p00, p11


def _run_filter(params: tuple, x, s0: float, s1: float, f1: float, store: bool = False):
    """Hamilton filter with parallel per-regime GARCH variances.

    Pure-Python scalar loop: with only two regimes this is several times
    faster than per-step numpy calls, which matters because the likelihood
    is evaluated thousands of times per fit.

    s0, s1 : each regime's variance for the first observation in x.
    f1     : filtered P(crisis) as of the observation before x.
    Returns (log_likelihood, s0, s1, f1, path) where s0/s1 are the variances
    for the day after x, f1 is the filtered P(crisis) at the end of x, and
    path (if store) holds per-day [P(crisis | data through t), sigma2_0,t, sigma2_1,t].
    """
    mu0, mu1, w0, w1, a0, a1, b0, b1, p00, p11 = params
    p01, p11_ = 1.0 - p00, p11
    ll = 0.0
    path = [] if store else None
    for xt in x:
        q1 = (1.0 - f1) * p01 + f1 * p11_  # predicted P(crisis today)
        e0 = xt - mu0
        e1 = xt - mu1
        d0 = math.exp(-0.5 * (e0 * e0 / s0 + _LOG_2PI + math.log(s0)))
        d1 = math.exp(-0.5 * (e1 * e1 / s1 + _LOG_2PI + math.log(s1)))
        j0 = (1.0 - q1) * d0
        j1 = q1 * d1
        lik = j0 + j1
        if lik < 1e-300:
            lik = 1e-300
        ll += math.log(lik)
        f1 = j1 / lik
        if store:
            path.append((f1, s0, s1))
        s0 = w0 + a0 * e0 * e0 + b0 * s0
        s1 = w1 + a1 * e1 * e1 + b1 * s1
    return ll, s0, s1, f1, path


def _stationary_crisis_prob(p00: float, p11: float) -> float:
    return (1.0 - p00) / ((1.0 - p00) + (1.0 - p11))


def _neg_loglik(theta: np.ndarray, x: list, s2_init: float) -> float:
    params = _unpack(theta)
    f1 = _stationary_crisis_prob(params[8], params[9])
    ll = _run_filter(params, x, s2_init, s2_init, f1)[0]
    return -ll if np.isfinite(ll) else 1e10


def _default_starts(x: np.ndarray, n_starts: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Calm regime: low intercept, moderate reaction; crisis regime: higher
    intercept and reaction. Extra starts jitter this baseline."""
    var = float(np.var(x))
    base = np.array([
        float(np.mean(x)), -abs(float(np.mean(x))),
        math.log(0.02 * var), math.log(0.10 * var),
        logit(0.95 / MAX_PERSISTENCE), logit(0.90 / MAX_PERSISTENCE),
        logit(0.05), logit(0.12),
        logit(0.98), logit(0.95),
    ])
    starts = [base]
    for _ in range(n_starts - 1):
        starts.append(base + rng.normal(0, 0.5, size=base.size))
    return starts


def fit_msgarch(
    returns: pd.Series,
    n_starts: int = 4,
    theta0: np.ndarray | None = None,
    random_state: int = 42,
) -> MSGARCHFit:
    """Fit by maximum likelihood (L-BFGS-B on the unconstrained theta).

    MS-GARCH likelihoods have local optima, so several starting points are
    tried and the best is kept; `theta0` (e.g. the previous fit in a rolling
    backtest) is tried first when given.
    """
    x = returns.to_numpy() * SCALE
    x_list = x.tolist()
    s2_init = float(np.var(x))

    rng = np.random.default_rng(random_state)
    starts = ([np.asarray(theta0, dtype=float)] if theta0 is not None else []) + _default_starts(x, n_starts, rng)

    best = None
    for start in starts:
        res = minimize(_neg_loglik, start, args=(x_list, s2_init), method="L-BFGS-B", options={"maxiter": 500})
        if best is None or res.fun < best.fun:
            best = res

    theta = best.x.copy()
    mu0, mu1, w0, w1, a0, a1, b0, b1, _, _ = _unpack(theta)
    if w0 / (1 - a0 - b0) > w1 / (1 - a1 - b1):  # relabel so regime 0 is the calm one
        theta = theta[_REGIME_SWAP]

    params = _unpack(theta)
    mu0, mu1, w0, w1, a0, a1, b0, b1, p00, p11 = params
    f1_init = _stationary_crisis_prob(p00, p11)
    ll, s0, s1, f1, _ = _run_filter(params, x_list, s2_init, s2_init, f1_init)

    return MSGARCHFit(
        mu=np.array([mu0, mu1]) / SCALE,
        omega=np.array([w0, w1]) / SCALE**2,
        alpha=np.array([a0, a1]),
        beta=np.array([b0, b1]),
        transmat=np.array([[p00, 1 - p00], [1 - p11, p11]]),
        log_likelihood=ll,
        theta=theta,
        sigma2_next=np.array([s0, s1]) / SCALE**2,
        prob_crisis=f1,
    )


def _params_original_scale(fit: MSGARCHFit) -> tuple:
    return (
        fit.mu[0], fit.mu[1], fit.omega[0], fit.omega[1],
        fit.alpha[0], fit.alpha[1], fit.beta[0], fit.beta[1],
        fit.transmat[0, 0], fit.transmat[1, 1],
    )


def filter_path(fit: MSGARCHFit, returns: pd.Series) -> pd.DataFrame:
    """Causal (forward-only) filter over `returns` with the fitted parameters:
    per day, P(crisis | data through t) and each regime's conditional std for
    that day (which uses data through t-1 only)."""
    x = returns.to_numpy()
    s2_init = float(np.var(x))
    p00, p11 = fit.transmat[0, 0], fit.transmat[1, 1]
    _, _, _, _, path = _run_filter(
        _params_original_scale(fit), x.tolist(), s2_init, s2_init, _stationary_crisis_prob(p00, p11), store=True
    )
    arr = np.array(path)
    return pd.DataFrame(
        {"prob_crisis": arr[:, 0], "sigma_calm": np.sqrt(arr[:, 1]), "sigma_crisis": np.sqrt(arr[:, 2])},
        index=returns.index,
    )


def next_day_mixture(fit: MSGARCHFit) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(weights, means, stds) of the one-step-ahead Gaussian mixture."""
    filtered = np.array([1 - fit.prob_crisis, fit.prob_crisis])
    weights = filtered @ fit.transmat
    return weights, fit.mu, np.sqrt(fit.sigma2_next)


def one_step_forecast(fit: MSGARCHFit) -> tuple[float, float]:
    """(mu_daily, sigma_daily) for the day after the fit's last observation:
    first two moments of the predictive mixture (moment-matched, as in hmm.py)."""
    weights, means, stds = next_day_mixture(fit)
    mu = float(np.dot(weights, means))
    second_moment = float(np.dot(weights, stds**2 + means**2))
    return mu, float(np.sqrt(max(second_moment - mu**2, 1e-12)))


def _advance_state(fit: MSGARCHFit, new_return: float) -> MSGARCHFit:
    """Roll the filter forward by one observed return without refitting."""
    _, s0, s1, f1, _ = _run_filter(
        _params_original_scale(fit), [new_return], fit.sigma2_next[0], fit.sigma2_next[1], fit.prob_crisis
    )
    fit.sigma2_next = np.array([s0, s1])
    fit.prob_crisis = f1
    return fit


def _refilter(fit: MSGARCHFit, history: pd.Series) -> MSGARCHFit:
    """Recompute the filter state over the full history with fixed parameters."""
    path = filter_path(fit, history)
    last = path.iloc[-1]
    x_last = history.iloc[-1]
    fit.prob_crisis = float(last["prob_crisis"])
    fit.sigma2_next = np.array([
        fit.omega[k] + fit.alpha[k] * (x_last - fit.mu[k]) ** 2 + fit.beta[k] * last[col] ** 2
        for k, col in enumerate(["sigma_calm", "sigma_crisis"])
    ])
    return fit


def simulate_msgarch(fit: MSGARCHFit, n: int, seed: int | None = None) -> np.ndarray:
    """Simulate n daily returns from fitted (or hand-set) MS-GARCH parameters."""
    rng = np.random.default_rng(seed)
    p00, p11 = fit.transmat[0, 0], fit.transmat[1, 1]
    state = int(rng.random() < _stationary_crisis_prob(p00, p11))
    sigma2 = fit.omega / (1 - fit.persistence)
    out = np.empty(n)
    for t in range(n):
        out[t] = fit.mu[state] + math.sqrt(sigma2[state]) * rng.standard_normal()
        sigma2 = fit.omega + fit.alpha * (out[t] - fit.mu) ** 2 + fit.beta * sigma2
        state = int(rng.random() < fit.transmat[state, 1])
    return out


def make_msgarch_forecast_fn(refit_every: int = 63, n_starts: int = 1):
    """Build a `forecast_fn(history) -> (mu_daily, sigma_daily)` for the rolling
    VaR backtest harness in var_backtest.py. Refits by MLE every `refit_every`
    trading days (~quarterly, matching the HMM), warm-starting from the
    previous fit's parameters (tried alongside `n_starts` default starts;
    each fit takes several seconds, so the full backtest takes minutes)."""
    state: dict = {"fit": None, "fit_at_len": -10**9, "last_n": None}

    def forecast_fn(history: pd.Series) -> tuple[float, float]:
        n = len(history)
        need_refit = state["fit"] is None or (n - state["fit_at_len"]) >= refit_every

        if need_refit:
            theta0 = state["fit"].theta if state["fit"] is not None else None
            state["fit"] = fit_msgarch(history, n_starts=n_starts, theta0=theta0)
            state["fit_at_len"] = n
        elif state["last_n"] == n - 1:
            state["fit"] = _advance_state(state["fit"], history.iloc[-1])
        else:
            state["fit"] = _refilter(state["fit"], history)
        state["last_n"] = n

        return one_step_forecast(state["fit"])

    return forecast_fn


if __name__ == "__main__":
    import sys
    import time
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.garch import fit_garch
    from src.hmm import mixture_var
    from src.gbm import parametric_var
    from src.loaders import load_close
    from src.returns import TRADING_DAYS, log_returns
    from src.var_backtest import basel_traffic_light, christoffersen_test, kupiec_test, rolling_var_backtest

    prices = load_close("nifty50")
    returns = log_returns(prices)
    vix = load_close("india_vix")

    print("=== Full-sample 2-regime MS-GARCH(1,1) fit ===")
    t0 = time.time()
    fit = fit_msgarch(returns, n_starts=6)
    print(f"(fit took {time.time() - t0:.1f}s, logL={fit.log_likelihood:.1f} on percent scale)")
    for k, name in enumerate(["calm", "crisis"]):
        print(
            f"  regime {k} ({name}): mu_annual={fit.mu[k] * TRADING_DAYS:+.2%}  "
            f"omega={fit.omega[k]:.3e}  alpha={fit.alpha[k]:.4f}  beta={fit.beta[k]:.4f}  "
            f"persistence={fit.persistence[k]:.4f}  long-run vol={fit.long_run_vol[k] * np.sqrt(TRADING_DAYS):.2%}"
        )
    print("  transition matrix (rows=from, cols=to):")
    print(np.round(fit.transmat, 4))
    print("  expected durations (days):", np.round(1 / (1 - np.diag(fit.transmat)), 1))

    single = fit_garch(returns)
    print(f"  single-regime GARCH(1,1) persistence for comparison: {single.alpha + single.beta:.4f}")

    path = filter_path(fit, returns)
    crisis = path["prob_crisis"] > 0.5
    print(f"\n  share of days with filtered P(crisis) > 0.5: {crisis.mean():.1%}")

    print("\n=== Validation against India VIX ===")
    common = crisis.index.intersection(vix.index)
    print("  mean India VIX by regime:\n", vix.loc[common].groupby(crisis.loc[common].astype(int)).mean())

    print("\n=== Validation against known crisis windows ===")
    for label, start, end in [
        ("2008 GFC", "2008-09-01", "2009-03-31"),
        ("2020 COVID", "2020-02-15", "2020-06-30"),
    ]:
        print(f"  {label}: {crisis.loc[start:end].mean():.1%} of days with filtered P(crisis) > 0.5")

    print("\n=== One-day-ahead sigma around key dates (full-sample parameters) ===")
    for date in ["2020-02-27", "2020-02-28", "2020-03-12", "2020-03-23", "2020-06-30", "2024-06-04"]:
        hist = returns.loc[:date]
        _refilter(fit, hist)
        mu_f, sigma_f = one_step_forecast(fit)
        w, m, s = next_day_mixture(fit)
        print(
            f"  after {date} (r={hist.iloc[-1]:+.2%}): P(crisis)={fit.prob_crisis:.1%}  "
            f"sigma_next={sigma_f:.2%}  [calm {s[0]:.2%}, crisis {s[1]:.2%}]  "
            f"normal VaR={parametric_var(mu_f, sigma_f):.2%}  exact mixture VaR={mixture_var(w, m, s):.2%}"
        )

    print("\n=== Rolling out-of-sample VaR backtest (MS-GARCH, 2-regime, refit every 63 days) ===")
    t0 = time.time()
    forecast_fn = make_msgarch_forecast_fn(refit_every=63)
    result = rolling_var_backtest(returns, forecast_fn, alpha=0.99, min_window=500)
    print(f"(backtest took {time.time() - t0:.0f}s)")
    print(f"Backtest: {result.n_obs} obs, {result.n_violations} violations, rate={result.violation_rate:.4%}")

    kupiec = kupiec_test(result)
    print("Kupiec:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in kupiec.items()})

    christoffersen = christoffersen_test(result)
    print("Christoffersen:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in christoffersen.items()})

    traffic = basel_traffic_light(result)
    print(traffic["zone"].value_counts())

    covid = result.table.loc["2020-02-15":"2020-06-30"]
    print(f"COVID window: {int(covid['hit'].sum())}/{len(covid)} violations")
