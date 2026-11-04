"""Regime transition matrices for every market.

For each market, refits the full-sample 2-state Gaussian HMM (same settings as
the regime study: n_iter=200, random_state=42) and the 2-regime MS-GARCH, and
reports the fitted transition matrix P[i, j] = P(regime j tomorrow | regime i today),
with regime 0 = calm, 1 = crisis. Expected time spent in a regime before
leaving it is 1 / (1 - P[i, i]) trading days.

Also reports the empirical transition matrix of the daily crisis flags
(filtered P(crisis) > 0.5) saved by multi_market.py, and the number of regime
switches per year.

Usage:  python -m src.regime_transitions
Output: reports/multi_market/ALL_transition_matrices.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.hmm import fit_gaussian_hmm
from src import msgarch as msg
from src.multi_market import MARKETS, OUT_DIR, load_returns


def empirical(flag: np.ndarray) -> np.ndarray:
    a, b = flag[:-1].astype(int), flag[1:].astype(int)
    m = np.zeros((2, 2))
    for i in (0, 1):
        n = (a == i).sum()
        for j in (0, 1):
            m[i, j] = ((a == i) & (b == j)).sum() / n if n else np.nan
    return m


def row(market, model, source, P, n_years, switches=np.nan):
    return {
        "market": market, "model": model, "source": source,
        "p_calm_to_calm": P[0, 0], "p_calm_to_crisis": P[0, 1],
        "p_crisis_to_calm": P[1, 0], "p_crisis_to_crisis": P[1, 1],
        "expected_calm_days": 1 / (1 - P[0, 0]),
        "expected_crisis_days": 1 / (1 - P[1, 1]),
        "stationary_crisis_share": P[0, 1] / (P[0, 1] + P[1, 0]),
        "switches_per_year": switches / n_years if switches == switches else np.nan,
    }


def main() -> None:
    rows = []
    for market in MARKETS:
        r = load_returns(market)
        n_years = len(r) / 252
        hmm = fit_gaussian_hmm(r, n_states=2, n_iter=200, random_state=42)
        P = hmm.transmat
        if hmm.stds[0] > hmm.stds[1]:  # put calm first
            P = P[::-1, ::-1]
        rows.append(row(market, "HMM", "fitted", P, n_years))

        ms = msg.fit_msgarch(r, n_starts=4)
        rows.append(row(market, "MS-GARCH", "fitted", ms.transmat, n_years))

        flags = pd.read_csv(OUT_DIR / f"{market}_regime_flags.csv", index_col=0)
        for model, col in [("HMM", "hmm_crisis"), ("MS-GARCH", "ms_crisis")]:
            f = flags[col].astype(bool).to_numpy()
            rows.append(row(market, model, "empirical flags", empirical(f), n_years,
                            switches=int((f[1:] != f[:-1]).sum())))
        print(f"{market} done", flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(OUT_DIR / "ALL_transition_matrices.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print(d.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
