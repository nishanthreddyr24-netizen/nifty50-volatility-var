"""Regime-switching (Gaussian Hidden Markov Model) for NIFTY 50 returns.

Regimes are estimated from the data by MLE (Baum-Welch/EM via hmmlearn's
GaussianHMM) -- never hand-labelled by a volatility threshold, which would be
circular. After fitting, raw hidden-state indices (arbitrary order from EM)
are relabelled 0..k-1 by ascending volatility purely for readability (e.g.
state 0 = "calm", state k-1 = "crisis"); this relabelling is a post-hoc sort
of already-fitted parameters and uses no future information.

Two usages:
  1. `fit_gaussian_hmm` on the *full* sample, for in-sample regime discovery
     and validation against India VIX / known crisis dates.
  2. `make_hmm_forecast_fn` for the rolling, causal, out-of-sample VaR
     backtest: refits periodically (not every day -- EM refits are
     expensive and daily refitting is not how this would be run in
     practice), and between refits advances a causal (forward-only) filter
     using only the fixed, already-fitted parameters and data up to and
     including the current day. Nothing it uses is ever drawn from data at
     or after the forecast date.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy import stats as sstats
from scipy.special import logsumexp

from src.returns import TRADING_DAYS

# hmmlearn logs "Model is not converging" via the stdlib logger (not a
# warnings.warn) whenever log-likelihood dips by a tiny amount between EM
# iterations near a local optimum -- harmless but very noisy across many
# rolling refits, so it's quieted here rather than suppressed per-call.
logging.getLogger("hmmlearn").setLevel(logging.ERROR)


@dataclass
class HMMFit:
    model: GaussianHMM
    order: np.ndarray       # order[k] = raw hmmlearn state index of regime k (0=calm,...)
    means: np.ndarray       # regime means, sorted by ascending vol, daily
    stds: np.ndarray        # regime stds, sorted by ascending vol, daily
    transmat: np.ndarray    # regime transition matrix, reordered to match means/stds
    startprob: np.ndarray   # reordered start probabilities


def fit_gaussian_hmm(returns: pd.Series, n_states: int = 2, n_iter: int = 200, random_state: int = 42) -> HMMFit:
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=n_iter,
        random_state=random_state,
        tol=1e-6,
    )
    X = returns.to_numpy().reshape(-1, 1)
    model.fit(X)

    raw_stds = np.sqrt(model.covars_.reshape(n_states))
    order = np.argsort(raw_stds)  # ascending volatility

    means = model.means_.reshape(n_states)[order]
    stds = raw_stds[order]
    transmat = model.transmat_[np.ix_(order, order)]
    startprob = model.startprob_[order]

    return HMMFit(model=model, order=order, means=means, stds=stds, transmat=transmat, startprob=startprob)


def n_free_params(n_states: int) -> int:
    """Free parameters of a diagonal-covariance Gaussian HMM: start probs
    (n-1 free), transition matrix (n rows, each n-1 free), means (n), variances (n)."""
    return (n_states - 1) + n_states * (n_states - 1) + n_states + n_states


def aic_bic(fit: HMMFit, returns: pd.Series) -> dict:
    log_l = fit.model.score(returns.to_numpy().reshape(-1, 1))
    k = n_free_params(len(fit.means))
    n = len(returns)
    aic = -2 * log_l + 2 * k
    bic = -2 * log_l + k * np.log(n)
    return {"log_likelihood": log_l, "n_params": k, "AIC": aic, "BIC": bic}


def viterbi_regimes(fit: HMMFit, returns: pd.Series) -> pd.Series:
    """Most likely regime sequence (smoothed, uses the whole sample) --
    for in-sample description/validation only, never for the causal backtest."""
    X = returns.to_numpy().reshape(-1, 1)
    raw_states = fit.model.predict(X)
    inverse_order = np.argsort(fit.order)  # maps raw hmmlearn index -> vol-sorted regime index
    regimes = inverse_order[raw_states]
    return pd.Series(regimes, index=returns.index, name="regime")


def filtered_state_probs(fit: HMMFit, returns: pd.Series) -> np.ndarray:
    """Causal (forward-only) filtered P(state_t = k | returns[0..t]) for every t,
    using the fixed, already-fitted model parameters. Shape (T, n_states),
    columns in vol-sorted regime order (0 = calm, ..., k-1 = crisis).
    """
    n_states = len(fit.means)
    X = returns.to_numpy()
    T = len(X)

    log_emission = np.column_stack(
        [sstats.norm.logpdf(X, loc=fit.means[k], scale=fit.stds[k]) for k in range(n_states)]
    )
    log_transmat = np.log(fit.transmat + 1e-300)
    log_alpha = np.empty((T, n_states))

    log_alpha[0] = np.log(fit.startprob + 1e-300) + log_emission[0]
    log_alpha[0] -= logsumexp(log_alpha[0])
    for t in range(1, T):
        log_alpha[t] = logsumexp(log_alpha[t - 1][:, None] + log_transmat, axis=0) + log_emission[t]
        log_alpha[t] -= logsumexp(log_alpha[t])

    return np.exp(log_alpha)


def make_hmm_forecast_fn(n_states: int = 2, refit_every: int = 63, n_iter: int = 100, random_state: int = 42):
    """Build a `forecast_fn(history) -> (mu_daily, sigma_daily)` for the rolling
    VaR backtest harness in var_backtest.py.

    Refits the HMM by EM every `refit_every` trading days (~quarterly by
    default) -- not every day, since EM refitting is expensive and daily
    refitting is unrealistic practice. Between refits, the causal forward
    filter is advanced incrementally by one step per new observation using
    the fixed fitted parameters, so the per-day forecast uses only data up
    to and including the current filtering point (strictly before the day
    being forecast, since the harness always calls this with history that
    excludes the forecast date).

    The next-day forecast is the first two moments (mean, std) of the
    Gaussian-mixture predictive distribution obtained by one-step-ahead
    transitioning the filtered regime probabilities. Downstream VaR/ES then
    use the normal formulas (gbm.parametric_var/es) as a moment-matched
    approximation to that mixture -- see hmm.py's __main__ block for a
    direct check of how close this approximation is to the exact mixture
    quantile.
    """
    state: dict = {"fit": None, "fit_at_len": -10**9, "log_alpha": None, "last_n": None}

    def _recompute_full(fit: HMMFit, history: pd.Series) -> np.ndarray:
        probs = filtered_state_probs(fit, history)
        return np.log(probs[-1] + 1e-300)

    def forecast_fn(history: pd.Series) -> tuple[float, float]:
        n = len(history)
        need_refit = state["fit"] is None or (n - state["fit_at_len"]) >= refit_every

        if need_refit:
            state["fit"] = fit_gaussian_hmm(history, n_states=n_states, n_iter=n_iter, random_state=random_state)
            state["fit_at_len"] = n
            state["log_alpha"] = _recompute_full(state["fit"], history)
            state["last_n"] = n
        elif state["last_n"] == n - 1:
            fit = state["fit"]
            log_transmat = np.log(fit.transmat + 1e-300)
            x_new = history.iloc[-1]
            log_emission = sstats.norm.logpdf(x_new, loc=fit.means, scale=fit.stds)
            log_alpha_new = logsumexp(state["log_alpha"][:, None] + log_transmat, axis=0) + log_emission
            log_alpha_new -= logsumexp(log_alpha_new)
            state["log_alpha"] = log_alpha_new
            state["last_n"] = n
        else:
            state["log_alpha"] = _recompute_full(state["fit"], history)
            state["last_n"] = n

        fit = state["fit"]
        filtered_t = np.exp(state["log_alpha"])
        next_probs = filtered_t @ fit.transmat

        mu = float(np.dot(next_probs, fit.means))
        second_moment = float(np.dot(next_probs, fit.stds**2 + fit.means**2))
        sigma = float(np.sqrt(max(second_moment - mu**2, 1e-12)))
        return mu, sigma

    return forecast_fn


def mixture_var(weights: np.ndarray, means: np.ndarray, stds: np.ndarray, alpha: float = 0.99) -> float:
    """Exact 1-day VaR of a Gaussian mixture (not moment-matched), found by
    root-finding on the mixture CDF: VaR solves P(r < -VaR) = 1 - alpha."""
    from scipy.optimize import brentq

    target = 1 - alpha

    def mixture_cdf(x: float) -> float:
        return float(np.dot(weights, sstats.norm.cdf(x, loc=means, scale=stds)))

    lo, hi = means.min() - 10 * stds.max(), means.max() + 10 * stds.max()
    root = brentq(lambda x: mixture_cdf(x) - target, lo, hi)
    return -root


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.loaders import load_close
    from src.returns import log_returns
    from src.var_backtest import basel_traffic_light, christoffersen_test, kupiec_test, rolling_var_backtest

    prices = load_close("nifty50")
    returns = log_returns(prices)
    vix = load_close("india_vix")

    print("=== Model selection: 2 vs 3 states (full sample) ===")
    for k in (2, 3):
        fit_k = fit_gaussian_hmm(returns, n_states=k)
        ic = aic_bic(fit_k, returns)
        print(f"n_states={k}: logL={ic['log_likelihood']:.1f}  AIC={ic['AIC']:.1f}  BIC={ic['BIC']:.1f}")

    n_states = 3
    fit = fit_gaussian_hmm(returns, n_states=n_states)
    print(f"\n=== Fitted {n_states}-state HMM (full sample) ===")
    for k in range(n_states):
        ann_mu = fit.means[k] * TRADING_DAYS
        ann_sigma = fit.stds[k] * np.sqrt(TRADING_DAYS)
        print(f"  regime {k}: mu_annual={ann_mu:+.2%}  sigma_annual={ann_sigma:.2%}")
    print("  transition matrix (rows=from, cols=to, regime order 0..{}):".format(n_states - 1))
    print(np.round(fit.transmat, 3))

    regimes = viterbi_regimes(fit, returns)
    print("\n  regime frequencies:\n", regimes.value_counts(normalize=True).sort_index())

    print("\n=== Validation against India VIX ===")
    common = regimes.index.intersection(vix.index)
    vix_by_regime = vix.loc[common].groupby(regimes.loc[common]).mean()
    print("  mean India VIX by regime:\n", vix_by_regime)

    print("\n=== Validation against known crisis windows ===")
    for label, start, end in [
        ("2008 GFC", "2008-09-01", "2009-03-31"),
        ("2020 COVID", "2020-02-15", "2020-06-30"),
    ]:
        window = regimes.loc[start:end]
        if len(window) == 0:
            continue
        frac_high = (window == n_states - 1).mean()
        print(f"  {label}: {frac_high:.1%} of days classified in the highest-vol regime")

    print("\n=== Moment-matched normal VaR vs exact mixture VaR (sanity check) ===")
    filtered = filtered_state_probs(fit, returns)
    sample_idx = [len(returns) - 1000, len(returns) - 250, len(returns) - 1]
    for idx in sample_idx:
        next_probs = filtered[idx] @ fit.transmat
        mu_mm = float(np.dot(next_probs, fit.means))
        var_mm = float(np.dot(next_probs, fit.stds**2 + fit.means**2) - mu_mm**2)
        sigma_mm = np.sqrt(max(var_mm, 1e-12))

        from src.gbm import parametric_var

        var_normal_approx = parametric_var(mu_mm, sigma_mm, alpha=0.99)
        var_exact_mixture = mixture_var(next_probs, fit.means, fit.stds, alpha=0.99)
        date = returns.index[idx]
        print(
            f"  {date.date()}: moment-matched normal VaR={var_normal_approx:.4%}  "
            f"exact mixture VaR={var_exact_mixture:.4%}"
        )

    print("\n=== Rolling out-of-sample VaR backtest (HMM, 2-state, refit every 63 days) ===")
    forecast_fn = make_hmm_forecast_fn(n_states=2, refit_every=63)
    result = rolling_var_backtest(returns, forecast_fn, alpha=0.99, min_window=500)
    print(f"Backtest: {result.n_obs} obs, {result.n_violations} violations, rate={result.violation_rate:.4%}")

    kupiec = kupiec_test(result)
    print("Kupiec:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in kupiec.items()})

    christoffersen = christoffersen_test(result)
    print("Christoffersen:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in christoffersen.items()})

    traffic = basel_traffic_light(result)
    print(traffic["zone"].value_counts())
