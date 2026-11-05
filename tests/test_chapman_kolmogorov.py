"""Sanity tests for the Chapman-Kolmogorov check."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chapman_kolmogorov import ck_hmm_crash, ck_markov, empirical_nstep, one_step_matrix

P = np.array([[0.97, 0.03], [0.10, 0.90]])


def _simulate_chain(P, n, seed=0):
    rng = np.random.default_rng(seed)
    s = np.zeros(n, dtype=int)
    u = rng.random(n)
    for t in range(1, n):
        s[t] = int(u[t] < P[s[t - 1], 1])
    return s


def test_true_markov_chain_passes_ck():
    """Data simulated from a Markov chain should match P^n at (almost) every horizon."""
    s = _simulate_chain(P, 60_000)
    P_hat = one_step_matrix(s)
    assert np.allclose(P_hat, P, atol=0.01)
    horizons = np.arange(1, 21)
    emp, lo, hi, _ = empirical_nstep(s, 1, horizons=horizons, n_boot=300)
    model = ck_markov(P_hat, 1, horizons=horizons)
    assert ((lo <= model) & (model <= hi)).mean() >= 0.9


def test_ck_matrix_power_identity():
    """P^(m+n) = P^m P^n: the 5-step value equals the product of 2- and 3-step matrices."""
    two, three = np.linalg.matrix_power(P, 2), np.linalg.matrix_power(P, 3)
    assert abs(ck_markov(P, 1, horizons=[5])[0] - (two @ three)[1, 1]) < 1e-12


def test_hmm_crash_prob_decays_to_base_rate():
    """Far ahead, P(crash | crash today) under the HMM tends to the unconditional crash rate."""
    means, stds = np.array([0.0005, -0.001]), np.array([0.008, 0.025])
    q = ck_hmm_crash(P, means, stds, -0.03, horizons=[1, 2000])
    from scipy.stats import norm
    pi = np.array([P[1, 0], P[0, 1]]) / (P[0, 1] + P[1, 0])
    base = pi @ norm.cdf((-0.03 - means) / stds)
    assert q[0] > 3 * base          # strong clustering one day ahead
    assert abs(q[1] - base) < 1e-9  # memory gone at long horizons
