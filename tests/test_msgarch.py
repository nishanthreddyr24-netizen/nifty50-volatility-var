"""Sanity tests for the 2-regime MS-GARCH model: the HMM is a nested special
case, incremental filtering matches a full recompute, and simulated
parameters are recovered."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.hmm import fit_gaussian_hmm, filtered_state_probs
from src.msgarch import MSGARCHFit, _advance_state, _refilter, filter_path, fit_msgarch, simulate_msgarch


def _hand_fit(mu, omega, alpha, beta, p00, p11) -> MSGARCHFit:
    return MSGARCHFit(
        mu=np.array(mu), omega=np.array(omega), alpha=np.array(alpha), beta=np.array(beta),
        transmat=np.array([[p00, 1 - p00], [1 - p11, p11]]),
        log_likelihood=np.nan, theta=np.zeros(10),
        sigma2_next=np.array(omega) / (1 - np.array(alpha) - np.array(beta)), prob_crisis=0.0,
    )


TRUE = dict(
    mu=[0.0005, -0.001], omega=[2e-6, 2e-5], alpha=[0.04, 0.12], beta=[0.90, 0.80], p00=0.99, p11=0.95,
)


def test_hmm_is_special_case():
    """With alpha = beta = 0 each regime's variance is constant, so the
    MS-GARCH filter must reproduce the Gaussian HMM's filtered probabilities
    (after a short burn-in, since the two filters start differently)."""
    rng = np.random.default_rng(1)
    states = np.cumsum(rng.random(3000) < 0.02) % 2
    returns = pd.Series(rng.normal(0, np.where(states == 1, 0.025, 0.008)))

    hmm = fit_gaussian_hmm(returns, n_states=2)
    as_msgarch = _hand_fit(
        hmm.means, hmm.stds**2, [0.0, 0.0], [0.0, 0.0], hmm.transmat[0, 0], hmm.transmat[1, 1]
    )

    ms_probs = filter_path(as_msgarch, returns)["prob_crisis"].to_numpy()
    hmm_probs = filtered_state_probs(hmm, returns)[:, 1]
    assert np.allclose(ms_probs[50:], hmm_probs[50:], atol=1e-6)


def test_incremental_update_matches_full_refilter():
    """Advancing the filter one day must give the same state as re-running
    it over the whole history -- the backtest relies on this between refits."""
    fit = _hand_fit(**TRUE)
    returns = pd.Series(simulate_msgarch(fit, 800, seed=2))

    incremental = _advance_state(_refilter(fit, returns.iloc[:-1]), returns.iloc[-1])
    inc_state = (incremental.prob_crisis, incremental.sigma2_next.copy())

    full = _refilter(_hand_fit(**TRUE), returns)
    assert abs(inc_state[0] - full.prob_crisis) < 1e-10
    assert np.allclose(inc_state[1], full.sigma2_next, rtol=1e-10)


def test_parameter_recovery():
    """Simulate from known parameters, refit, and check the key quantities."""
    true_fit = _hand_fit(**TRUE)
    returns = pd.Series(simulate_msgarch(true_fit, 6000, seed=3))

    fit = fit_msgarch(returns, n_starts=2)

    assert fit.long_run_vol[1] > fit.long_run_vol[0], "regimes not ordered calm -> crisis"
    assert np.allclose(fit.persistence, true_fit.persistence, atol=0.05), fit.persistence
    assert np.allclose(np.diag(fit.transmat), [TRUE["p00"], TRUE["p11"]], atol=0.03), fit.transmat
    assert np.allclose(fit.long_run_vol, true_fit.long_run_vol, rtol=0.25), fit.long_run_vol


if __name__ == "__main__":
    test_hmm_is_special_case()
    test_incremental_update_matches_full_refilter()
    test_parameter_recovery()
    print("All tests passed.")
