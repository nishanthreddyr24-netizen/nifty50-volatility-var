"""Sanity tests for the HMM-gated GARCH combination rule."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.hmm_garch import gated_forecast

MEANS = np.array([0.0005, -0.001])
STDS = np.array([0.008, 0.025])


def _hmm_sigma(next_probs):
    mu = np.dot(next_probs, MEANS)
    return np.sqrt(np.dot(next_probs, STDS**2 + MEANS**2) - mu**2)


def test_equals_hmm_when_garch_is_lower():
    """If GARCH is below the HMM's crisis std, the gate changes nothing."""
    probs = np.array([0.3, 0.7])
    _, sigma = gated_forecast(MEANS, STDS, probs, sigma_garch=0.015)
    assert abs(sigma - _hmm_sigma(probs)) < 1e-12


def test_lifts_crisis_ceiling():
    """A GARCH sigma above the HMM crisis std raises the forecast past the HMM ceiling."""
    probs = np.array([0.0, 1.0])
    _, sigma = gated_forecast(MEANS, STDS, probs, sigma_garch=0.06)
    assert abs(sigma - 0.06) < 1e-12
    assert sigma > _hmm_sigma(probs)


def test_calm_days_ignore_garch():
    """With the HMM sure of calm, GARCH cannot move the forecast."""
    probs = np.array([1.0, 0.0])
    _, sigma = gated_forecast(MEANS, STDS, probs, sigma_garch=0.06)
    assert abs(sigma - STDS[0]) < 1e-12


if __name__ == "__main__":
    test_equals_hmm_when_garch_is_lower()
    test_lifts_crisis_ceiling()
    test_calm_days_ignore_garch()
    print("All tests passed.")
