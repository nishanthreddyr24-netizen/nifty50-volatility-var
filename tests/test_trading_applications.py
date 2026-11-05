"""Sanity tests for the trading test's building blocks."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.trading_applications import (block_boot_sharpe_p, max_drawdown_additive,
                                      max_drawdown_compound, sharpe)


def test_variance_swap_sign():
    """Selling variance at K = 20 vol pts gains when realised vol is 15, loses when it is 40."""
    K = 20.0
    calm, crash = (K**2 - 15.0**2) / (2 * K), (K**2 - 40.0**2) / (2 * K)
    assert calm > 0 and crash < 0
    assert abs(crash) > abs(calm)  # the short-vol payoff is convex against the seller


def test_drawdowns():
    assert abs(max_drawdown_compound(np.array([0.1, -0.5, 0.2])) - 0.5) < 1e-12
    assert abs(max_drawdown_additive(np.array([5.0, -3.0, -4.0, 10.0])) - 7.0) < 1e-12


def test_bootstrap_detects_a_clear_sharpe_gain():
    rng = np.random.default_rng(0)
    base = rng.normal(0.0, 1.0, 3000)
    better = base * 0.5 + 0.1           # same shocks, higher mean per unit risk
    assert sharpe(better, 252) > sharpe(base, 252)
    assert block_boot_sharpe_p(better, base, 252, 250, 500) < 0.05
    assert block_boot_sharpe_p(base, better, 252, 250, 500) > 0.95
