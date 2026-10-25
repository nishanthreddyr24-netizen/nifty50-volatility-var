"""Simulation study: simulate synthetic data from a known (fitted) 2-state HMM,
refit the model to each synthetic sample, and check that EM recovers the true
parameters. This is the standard sanity check that the estimator (not just the
model) is trustworthy before relying on it for inference on real data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.hmm import HMMFit, fit_gaussian_hmm
from src.loaders import load_close
from src.returns import TRADING_DAYS, log_returns


def simulate_from_hmm(true_fit: HMMFit, n_samples: int, seed: int) -> np.ndarray:
    """Draw a synthetic return path from the vol-sorted (means, stds, transmat,
    startprob) parameters stored in `true_fit`, using hmmlearn's own sampler
    so the generative process exactly matches what GaussianHMM.fit assumes."""
    from hmmlearn.hmm import GaussianHMM

    gen = GaussianHMM(n_components=len(true_fit.means), covariance_type="diag")
    gen.startprob_ = true_fit.startprob
    gen.transmat_ = true_fit.transmat
    gen.means_ = true_fit.means.reshape(-1, 1)
    gen.covars_ = (true_fit.stds**2).reshape(-1, 1)  # covariance_type="diag": shape (n_states, n_features)
    X, _ = gen.sample(n_samples, random_state=seed)
    return X.ravel()


def run_simulation_study(true_fit: HMMFit, n_samples: int, n_reps: int = 50, base_seed: int = 1000) -> pd.DataFrame:
    n_states = len(true_fit.means)
    rows = []
    for rep in range(n_reps):
        x = simulate_from_hmm(true_fit, n_samples, seed=base_seed + rep)
        synthetic_returns = pd.Series(x)
        try:
            fitted = fit_gaussian_hmm(synthetic_returns, n_states=n_states, random_state=base_seed + rep)
        except Exception:
            continue  # occasional EM failure on a degenerate synthetic draw; skip

        row = {"rep": rep}
        for k in range(n_states):
            row[f"mu_{k}_true"] = true_fit.means[k]
            row[f"mu_{k}_hat"] = fitted.means[k]
            row[f"sigma_{k}_true"] = true_fit.stds[k]
            row[f"sigma_{k}_hat"] = fitted.stds[k]
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    prices = load_close("nifty50")
    returns = log_returns(prices)

    n_states = 2
    true_fit = fit_gaussian_hmm(returns, n_states=n_states)
    print(f"=== 'True' parameters: {n_states}-state HMM fitted to the full NIFTY 50 sample ===")
    for k in range(n_states):
        print(
            f"  regime {k}: mu_daily={true_fit.means[k]:+.6f} "
            f"(annual {true_fit.means[k]*TRADING_DAYS:+.2%})  "
            f"sigma_daily={true_fit.stds[k]:.6f} (annual {true_fit.stds[k]*np.sqrt(TRADING_DAYS):.2%})"
        )

    n_samples = len(returns)  # match the real sample size
    n_reps = 50
    print(f"\nSimulating {n_reps} synthetic samples of n={n_samples} and refitting EM to each...")
    study = run_simulation_study(true_fit, n_samples=n_samples, n_reps=n_reps)
    print(f"Converged fits: {len(study)}/{n_reps}")

    print("\n=== Parameter recovery (mean +/- std across simulations) ===")
    for k in range(n_states):
        mu_true = true_fit.means[k]
        sigma_true = true_fit.stds[k]
        mu_hat_mean = study[f"mu_{k}_hat"].mean()
        mu_hat_std = study[f"mu_{k}_hat"].std()
        sigma_hat_mean = study[f"sigma_{k}_hat"].mean()
        sigma_hat_std = study[f"sigma_{k}_hat"].std()
        print(
            f"  regime {k}: mu true={mu_true:+.6f}  hat={mu_hat_mean:+.6f} +/- {mu_hat_std:.6f}  "
            f"(bias={mu_hat_mean - mu_true:+.6f})"
        )
        print(
            f"             sigma true={sigma_true:.6f}  hat={sigma_hat_mean:.6f} +/- {sigma_hat_std:.6f}  "
            f"(bias={sigma_hat_mean - sigma_true:+.6f})"
        )

    print(
        "\nInterpretation: small bias relative to the across-simulation std means EM "
        "recovers the generating parameters on data of this length -- the estimator "
        "itself is trustworthy, so the full-sample regime fit reported in hmm.py's "
        "__main__ output can be read at face value rather than as an artifact of a "
        "biased or unstable optimizer."
    )
