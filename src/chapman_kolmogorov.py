"""Chapman-Kolmogorov check: do one-step models reproduce n-step behaviour?

For a Markov chain with one-step matrix P, the Chapman-Kolmogorov equations give
the n-step matrix as P^(n) = P^n. If the data really follow the chain, the
empirical n-step transition frequencies should match P^n at every horizon n.
A chain whose memory is too short fails this at longer horizons.

Two checks per market, horizons n = 1..40 trading days:

A. Crash days (log return <= -3%). Quantity: q(n) = P(crash at t+n | crash at t).
   Three models, each fitted on the full sample:
   - Poisson / i.i.d.: q(n) = p, the unconditional crash rate (no memory).
   - Observed 2-state Markov chain on the 0/1 crash flag: q(n) = (P^n)[1, 1].
   - 2-state Gaussian HMM on returns: the hidden regime is the Markov chain, so
         q(n) = sum_i w_i sum_j (A^n)[i, j] c_j,
     c_j = P(r <= -3% | regime j) = Phi((-0.03 - mu_j) / sigma_j),
     w_i = P(regime i | crash today) proportional to pi_i c_i (pi = stationary dist.).

B. HMM regime flags (filtered P(crisis) > 0.5, from multi_market.py).
   Quantity: P(flag at t+n = flag at t | flag at t), for calm and crisis.
   Models: the flags' own empirical one-step matrix raised to the n-th power, and
   the HMM's fitted matrix A^n.

Pass/fail: a model passes at horizon n if its value lies inside the 95%
moving-block-bootstrap interval of the empirical frequency (blocks of 250
trading days, 1,000 replications). The block bootstrap keeps the dependence
between nearby days, which a plain binomial interval would ignore.

All fits are full-sample, so this is an in-sample model check (like a
goodness-of-fit test), not a forecast evaluation.

Usage:  python -m src.chapman_kolmogorov
Output: reports/multi_market/CK_crash_days.csv, CK_regime_flags.csv,
        CK_summary.csv, figures/fig11_chapman_kolmogorov.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.hmm import fit_gaussian_hmm
from src.multi_market import CRASH_THRESHOLD, MARKETS, OUT_DIR, load_returns

HORIZONS = np.arange(1, 41)
BLOCK = 250
N_BOOT = 1000
SEED = 7
NAMES = {"sp500": "S&P 500", "ftse100": "FTSE 100", "dax": "DAX", "nikkei225": "Nikkei 225",
         "hangseng": "Hang Seng", "sensex": "Sensex", "nifty50": "NIFTY 50"}


# ---------------------------------------------------------------- core maths

def one_step_matrix(flag: np.ndarray) -> np.ndarray:
    """Empirical one-step transition matrix of a 0/1 series (row = today)."""
    a, b = flag[:-1].astype(int), flag[1:].astype(int)
    P = np.zeros((2, 2))
    for i in (0, 1):
        n = (a == i).sum()
        for j in (0, 1):
            P[i, j] = ((a == i) & (b == j)).sum() / n if n else np.nan
    return P


def stationary(P: np.ndarray) -> np.ndarray:
    """Stationary distribution of a 2x2 transition matrix."""
    p01, p10 = P[0, 1], P[1, 0]
    return np.array([p10, p01]) / (p01 + p10)


def ck_markov(P: np.ndarray, state: int, horizons=HORIZONS) -> np.ndarray:
    """P(state at t+n | state at t) = (P^n)[state, state] for each horizon n."""
    return np.array([np.linalg.matrix_power(P, n)[state, state] for n in horizons])


def ck_hmm_crash(A: np.ndarray, means: np.ndarray, stds: np.ndarray,
                 threshold: float, horizons=HORIZONS) -> np.ndarray:
    """HMM-implied P(crash at t+n | crash at t) via Chapman-Kolmogorov on the hidden chain."""
    c = norm.cdf((threshold - means) / stds)          # crash prob. in each regime
    w = stationary(A) * c
    w = w / w.sum()                                   # regime given a crash today
    return np.array([w @ np.linalg.matrix_power(A, n) @ c for n in horizons])


def empirical_nstep(flag: np.ndarray, state: int, horizons=HORIZONS,
                    n_boot: int = N_BOOT, block: int = BLOCK, seed: int = SEED):
    """Empirical P(flag_{t+n} = state | flag_t = state) with a 95% moving-block bootstrap
    interval. Returns arrays (estimate, low, high, n_conditioning_days)."""
    rng = np.random.default_rng(seed)
    f = flag.astype(int) == state
    est, lo, hi, cnt = [], [], [], []
    for n in horizons:
        a = f[:-n].astype(float)                      # in `state` today
        b = (f[:-n] & f[n:]).astype(float)            # ... and again n days later
        m = len(a)
        L = min(block, m)
        k = int(np.ceil(m / L))
        ca, cb = np.concatenate([[0], np.cumsum(a)]), np.concatenate([[0], np.cumsum(b)])
        starts = rng.integers(0, m - L + 1, size=(n_boot, k))
        sa = (ca[starts + L] - ca[starts]).sum(axis=1)
        sb = (cb[starts + L] - cb[starts]).sum(axis=1)
        boot = sb[sa > 0] / sa[sa > 0]
        est.append(b.sum() / a.sum())
        lo.append(np.quantile(boot, 0.025))
        hi.append(np.quantile(boot, 0.975))
        cnt.append(int(a.sum()))
    return np.array(est), np.array(lo), np.array(hi), np.array(cnt)


# ---------------------------------------------------------------- per market

def run_market(market: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    r = load_returns(market)
    fit = fit_gaussian_hmm(r, n_states=2, n_iter=200, random_state=42)
    A, mu, sd = fit.transmat, fit.means, fit.stds

    # A. crash days
    crash = (r <= CRASH_THRESHOLD).to_numpy()
    emp, lo, hi, cnt = empirical_nstep(crash, 1)
    models = {
        "Poisson (i.i.d.)": np.full(len(HORIZONS), crash.mean()),
        "Observed Markov chain": ck_markov(one_step_matrix(crash), 1),
        "HMM 2-state": ck_hmm_crash(A, mu, sd, CRASH_THRESHOLD),
    }
    rows_a = []
    for name, q in models.items():
        for n, e, l, h, c, v in zip(HORIZONS, emp, lo, hi, cnt, q):
            rows_a.append({"market": market, "model": name, "horizon": n, "empirical": e,
                           "ci_low": l, "ci_high": h, "n_crash_days": c, "model_value": v,
                           "pass": bool(l <= v <= h)})

    # B. regime flags
    flags = pd.read_csv(OUT_DIR / f"{market}_regime_flags.csv", index_col=0)
    f = flags["hmm_crisis"].astype(bool).to_numpy()
    Pf = one_step_matrix(f)
    rows_b = []
    for state, label in [(0, "calm"), (1, "crisis")]:
        emp, lo, hi, cnt = empirical_nstep(f, state)
        for name, P in [("Flags' one-step matrix", Pf), ("HMM fitted matrix", A)]:
            q = ck_markov(P, state)
            for n, e, l, h, c, v in zip(HORIZONS, emp, lo, hi, cnt, q):
                rows_b.append({"market": market, "state": label, "model": name, "horizon": n,
                               "empirical": e, "ci_low": l, "ci_high": h, "n_days": c,
                               "model_value": v, "pass": bool(l <= v <= h)})
    return pd.DataFrame(rows_a), pd.DataFrame(rows_b)


def summarise(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    sa = (a.groupby(["market", "model"])
            .agg(horizons_passed=("pass", "sum"), horizons=("pass", "size"),
                 mean_abs_error=("model_value", lambda s: 0.0))
            .reset_index())
    err = a.assign(e=(a.model_value - a.empirical).abs()).groupby(["market", "model"])["e"].mean()
    sa["mean_abs_error"] = err.to_numpy()
    sa.insert(1, "check", "crash days")
    b2 = b.assign(model=b.model + " (" + b.state + ")")
    sb = (b2.groupby(["market", "model"])
            .agg(horizons_passed=("pass", "sum"), horizons=("pass", "size")).reset_index())
    sb["mean_abs_error"] = (b2.assign(e=(b2.model_value - b2.empirical).abs())
                              .groupby(["market", "model"])["e"].mean().to_numpy())
    sb.insert(1, "check", "regime flags")
    return pd.concat([sa, sb], ignore_index=True)


def make_figure(a: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    markets = list(MARKETS)
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True)
    colors = {"Poisson (i.i.d.)": "#9a9a9a", "Observed Markov chain": "#d95f02",
              "HMM 2-state": "#1b6ca8"}
    for ax, m in zip(axes.flat, markets):
        d = a[a.market == m]
        e = d[d.model == "HMM 2-state"]
        ax.fill_between(e.horizon, e.ci_low, e.ci_high, color="#000000", alpha=0.12,
                        label="Empirical, 95% block-bootstrap band")
        ax.plot(e.horizon, e.empirical, color="#000000", lw=1.2, label="Empirical")
        for name, col in colors.items():
            g = d[d.model == name]
            ax.plot(g.horizon, g.model_value, color=col, lw=1.8, label=name)
        ax.set_title(NAMES.get(m, m))
        ax.set_ylim(bottom=0)
    axes.flat[-1].axis("off")
    h, l = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(h, l, loc="center", frameon=False)
    fig.supxlabel("Horizon n (trading days)")
    fig.supylabel("P(crash at t+n | crash at t)")
    fig.suptitle("Chapman-Kolmogorov check on crash days (log return <= -3%)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parts_a, parts_b = [], []
    for market in MARKETS:
        a, b = run_market(market)
        parts_a.append(a)
        parts_b.append(b)
        print(f"{market} done", flush=True)
    a, b = pd.concat(parts_a, ignore_index=True), pd.concat(parts_b, ignore_index=True)
    a.to_csv(OUT_DIR / "CK_crash_days.csv", index=False)
    b.to_csv(OUT_DIR / "CK_regime_flags.csv", index=False)
    s = summarise(a, b)
    s.to_csv(OUT_DIR / "CK_summary.csv", index=False)
    make_figure(a, OUT_DIR / "figures" / "fig11_chapman_kolmogorov.png")
    pd.set_option("display.width", 200)
    print(s.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
