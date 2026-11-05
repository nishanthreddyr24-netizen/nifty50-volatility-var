"""Report figures and theory-vs-data numbers for the course report (300 dpi).

Covers the syllabus items not drawn elsewhere: pipeline diagram, state-transition
diagram, transition heatmap and n-step probabilities, sojourn times, Poisson
checks (count process, inter-arrival times, count distribution), Poisson vs
Hawkes simulation, simulated sample paths (GBM, GARCH, HMM), theory vs simulation
vs data for the stylized facts, parameter recovery with our own EM, EM
convergence, and the NIFTY COVID VaR paths.

All on NIFTY 50 (2008-2026), the project's main market.

Usage:  python -m src.report_figures
Output: reports/report_figures/RF*.png, report_numbers.json, simulation_stats.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator

from src.em_hmm import fit_em
from src.extremes import fit_hawkes
from src.garch import fit_garch
from src.hmm import filtered_state_probs, fit_gaussian_hmm, viterbi_regimes
from src.loaders import load_close
from src.returns import log_returns

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "report_figures"
OUT.mkdir(parents=True, exist_ok=True)
DPI = 300
CALM, CRISIS, GREY, ORANGE, PURPLE, INK = "#1b6ca8", "#c0392b", "#8c8c8c", "#d95f02", "#7570b3", "#222222"
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100,
                     "savefig.bbox": "tight", "font.family": "DejaVu Sans"})
RNG = np.random.default_rng(2026)
N_SIM = 1000
NUM: dict = {}


def year_axis(ax, step=4):
    ax.xaxis.set_major_locator(mdates.YearLocator(step))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def save(fig, name):
    fig.savefig(OUT / name, dpi=DPI)
    plt.close(fig)
    print("wrote", name, flush=True)


# ---------------------------------------------------------------- theory helpers

def nstep_closed_form(a: float, b: float, n: np.ndarray) -> np.ndarray:
    """2-state chain, a = P(calm->crisis), b = P(crisis->calm). Returns P^n as (len(n), 2, 2):
    P^n = 1/(a+b) [[b, a], [b, a]] + (1-a-b)^n/(a+b) [[a, -a], [-b, b]]."""
    lam = (1 - a - b) ** n
    base = np.array([[b, a], [b, a]]) / (a + b)
    dev = np.array([[a, -a], [-b, b]]) / (a + b)
    return base[None] + lam[:, None, None] * dev[None]


def mixture_moments(pi, mu, sd):
    m = np.sum(pi * mu)
    var = np.sum(pi * (sd**2 + mu**2)) - m**2
    m4 = np.sum(pi * ((mu - m) ** 4 + 6 * (mu - m) ** 2 * sd**2 + 3 * sd**4))
    m3 = np.sum(pi * ((mu - m) ** 3 + 3 * (mu - m) * sd**2))
    return m, var, m3 / var**1.5, m4 / var**2 - 3


def hmm_acf_sq_theory(pi, mu, sd, lam, lags):
    """Exact ACF of r_t^2 for a stationary 2-state Gaussian HMM:
    Corr(r_t^2, r_{t+k}^2) = Var_pi(g(S)) lam^k / Var(r^2), g(i) = mu_i^2 + sd_i^2."""
    g = mu**2 + sd**2
    e4 = mu**4 + 6 * mu**2 * sd**2 + 3 * sd**4
    var_g = np.sum(pi * g**2) - np.sum(pi * g) ** 2
    var_r2 = np.sum(pi * e4) - np.sum(pi * g) ** 2
    return var_g * lam ** np.asarray(lags) / var_r2


def acf(x, lags):
    x = np.asarray(x) - np.mean(x)
    d = np.dot(x, x)
    return np.array([np.dot(x[:-k], x[k:]) / d for k in lags])


def spells(flag: np.ndarray, value: int) -> np.ndarray:
    out, run = [], 0
    for f in flag:
        if f == value:
            run += 1
        elif run:
            out.append(run)
            run = 0
    if run:
        out.append(run)
    return np.array(out)


def simulate_chain(P, n, s0, rng):
    s = np.empty(n, dtype=int)
    s[0] = s0
    u = rng.random(n)
    for t in range(1, n):
        s[t] = int(u[t] < P[s[t - 1], 1])
    return s


def simulate_hmm(P, mu, sd, pi, n, rng):
    s = simulate_chain(P, n, int(rng.random() < pi[1]), rng)
    return rng.normal(mu[s], sd[s]), s


def simulate_garch(g, n, rng):
    r = np.empty(n)
    s2 = g.omega / (1 - g.alpha - g.beta)
    eps_prev = 0.0
    for t in range(n):
        s2 = g.omega + g.alpha * eps_prev**2 + g.beta * s2
        eps_prev = np.sqrt(s2) * rng.standard_normal()
        r[t] = g.mu + eps_prev
    return r


def simulate_hawkes(mu, alpha, beta, T, rng):
    """Ogata thinning for an exponential-kernel Hawkes process on [0, T]."""
    t, events, excite = 0.0, [], 0.0   # excite = sum alpha*exp(-beta(t - t_i)) at time t
    while True:
        lam_bar = mu + excite
        w = rng.exponential(1 / lam_bar)
        t += w
        if t > T:
            break
        excite *= np.exp(-beta * w)
        if rng.random() * lam_bar <= mu + excite:
            events.append(t)
            excite += alpha
    return np.array(events)


def dispersion(counts):
    return counts.var(ddof=1) / counts.mean() if counts.mean() > 0 else np.nan


# ---------------------------------------------------------------- figures

def fig_pipeline():
    steps = [
        ("1  Real-world problem", "How large can tomorrow's\nloss be?\nDo crashes cluster?"),
        ("2  Data", "7 equity indices,\n1990-2026; VIX and\nIndia VIX (public)"),
        ("3  Probability model", "Normal mixture for\nreturns; Poisson vs\nHawkes for crashes"),
        ("4  Stochastic process", "Hidden Markov chain S_t;\nGBM; GARCH(1,1)"),
        ("5  Analysis", "P^n, stationary dist.,\nsojourn times,\nkurtosis, ACF"),
        ("6  Simulation", "Sample paths,\nparameter recovery,\nPoisson/Hawkes"),
        ("7  Validation", "Kupiec, Christoffersen,\nBasel, Chapman-\nKolmogorov, bootstrap"),
        ("8  Interpretation", "Calm/crisis regimes,\nVaR, risk control\nin trading"),
    ]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 2)
    ax.axis("off")
    pos = [(i + 0.5, 1.5) for i in range(4)] + [(3.5 - i, 0.5) for i in range(4)]
    for k, ((title, body), (x, y)) in enumerate(zip(steps, pos)):
        col = CALM if k < 4 else CRISIS if k == 6 else INK
        ax.add_patch(FancyBboxPatch((x - 0.42, y - 0.38), 0.84, 0.76, boxstyle="round,pad=0.02,rounding_size=0.06",
                                    fc="white", ec=col, lw=1.6))
        ax.text(x, y + 0.24, title, ha="center", va="center", fontsize=10.5, weight="bold", color=col)
        ax.text(x, y - 0.1, body, ha="center", va="center", fontsize=9, color=INK, linespacing=1.35)
    for k in range(7):
        (x0, y0), (x1, y1) = pos[k], pos[k + 1]
        if y0 == y1:
            sgn = 1 if x1 > x0 else -1
            a = FancyArrowPatch((x0 + sgn * 0.44, y0), (x1 - sgn * 0.44, y1), arrowstyle="-|>", mutation_scale=14, color=GREY, lw=1.4)
        else:
            a = FancyArrowPatch((x0, y0 - 0.4), (x1, y1 + 0.4), arrowstyle="-|>", mutation_scale=14, color=GREY, lw=1.4)
        ax.add_patch(a)
    save(fig, "RF01_pipeline.png")


def fig_state_diagram(P, mu, sd, pi):
    a, b = P[0, 1], P[1, 0]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_aspect("equal")
    R = 1.45
    for (x, col, name, i) in [(2.5, CALM, "Calm", 0), (7.5, CRISIS, "Crisis", 1)]:
        ax.add_patch(Circle((x, 2.4), R, fc="white", ec=col, lw=2.4))
        ax.text(x, 3.05, name, ha="center", fontsize=15, weight="bold", color=col)
        ax.text(x, 2.5, f"volatility {sd[i]*np.sqrt(252)*100:.1f}% a year", ha="center", fontsize=10)
        ax.text(x, 2.1, f"mean stay {1/(1-P[i,i]):.0f} trading days", ha="center", fontsize=10)
        ax.text(x, 1.7, f"long-run share {pi[i]*100:.1f}%", ha="center", fontsize=10)
        ax.add_patch(FancyArrowPatch((x - 0.45, 2.4 + R - 0.05), (x + 0.45, 2.4 + R - 0.05), connectionstyle="arc3,rad=-1.7",
                                     arrowstyle="-|>", mutation_scale=16, color=col, lw=1.8))
        ax.text(x, 4.75, f"stay {P[i,i]:.4f}", ha="center", fontsize=12, color=col)
    ax.add_patch(FancyArrowPatch((2.5 + R + 0.05, 2.85), (7.5 - R - 0.05, 2.85), connectionstyle="arc3,rad=-0.2",
                                 arrowstyle="-|>", mutation_scale=16, color=INK, lw=1.5))
    ax.text(5, 3.55, f"{a:.4f}", ha="center", fontsize=12)
    ax.add_patch(FancyArrowPatch((7.5 - R - 0.05, 1.95), (2.5 + R + 0.05, 1.95), connectionstyle="arc3,rad=-0.2",
                                 arrowstyle="-|>", mutation_scale=16, color=INK, lw=1.5))
    ax.text(5, 1.05, f"{b:.4f}", ha="center", fontsize=12)
    ax.text(5, 0.25, "NIFTY 50 2-state HMM: daily transition probabilities, full sample 2008-2026",
            ha="center", fontsize=9.5, color=GREY)
    save(fig, "RF02_state_transition_diagram.png")


def fig_transition_nstep(P, pi, rng):
    a, b = P[0, 1], P[1, 0]
    n = np.arange(0, 251)
    Pn = nstep_closed_form(a, b, n)
    # simulation: 10,000 chains started in each state
    reps = 10000
    sim = {}
    for s0 in (0, 1):
        s = np.full(reps, s0)
        frac = [s.mean()]
        for _ in range(n[-1]):
            u = rng.random(reps)
            s = (u < P[s, 1]).astype(int)
            frac.append(s.mean())
        sim[s0] = np.array(frac)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.7]})
    ax = axes[0]
    im = ax.imshow(P, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{P[i,j]:.4f}", ha="center", va="center", fontsize=13,
                    color="white" if P[i, j] > 0.5 else INK)
    ax.set_xticks([0, 1], ["Calm", "Crisis"])
    ax.set_yticks([0, 1], ["Calm", "Crisis"])
    ax.set_xlabel("State tomorrow")
    ax.set_ylabel("State today")
    ax.set_title("(a) Transition matrix P")
    ax.spines[:].set_visible(False)
    ax = axes[1]
    ax.plot(n, Pn[:, 1, 1], color=CRISIS, lw=2, label="Theory: P(crisis at n | crisis at 0)")
    ax.plot(n, Pn[:, 0, 1], color=CALM, lw=2, label="Theory: P(crisis at n | calm at 0)")
    ax.plot(n[::10], sim[1][::10], "o", color=CRISIS, ms=4, mfc="white", label="Simulation, 10,000 chains")
    ax.plot(n[::10], sim[0][::10], "o", color=CALM, ms=4, mfc="white")
    ax.axhline(pi[1], color=GREY, ls="--", lw=1.2, label=f"Stationary share {pi[1]:.3f}")
    ax.set_xlabel("Days ahead, n")
    ax.set_ylabel("Probability of being in crisis")
    ax.set_title("(b) n-step probabilities converge to the stationary distribution")
    ax.legend(frameon=False, loc="center right")
    save(fig, "RF03_transition_matrix_and_nstep.png")
    NUM["nstep_theory"] = {f"P_crisis_crisis_n{k}": float(Pn[k, 1, 1]) for k in (1, 5, 10, 20, 50, 100, 250)}
    NUM["nstep_theory"].update({f"P_calm_crisis_n{k}": float(Pn[k, 0, 1]) for k in (1, 5, 10, 20, 50, 100, 250)})
    NUM["nstep_sim_max_abs_error"] = float(max(np.abs(sim[1] - Pn[:, 1, 1]).max(), np.abs(sim[0] - Pn[:, 0, 1]).max()))


def fig_sojourn(P, viterbi, rng):
    sim_states = simulate_chain(P, 2_000_000, 0, rng)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, i, col, name in [(axes[0], 0, CALM, "Calm"), (axes[1], 1, CRISIS, "Crisis")]:
        p = P[i, i]
        k = np.arange(1, 400)
        theory = p ** (k - 1) * (1 - p)
        sim = spells(sim_states, i)
        obs = spells(viterbi, i)
        bins = np.unique(np.concatenate([[1], np.geomspace(1, 400, 25).astype(int)]))
        width = np.diff(bins)
        hs, _ = np.histogram(sim, bins=bins)
        ho, _ = np.histogram(obs, bins=bins)
        mid = np.sqrt(bins[:-1] * bins[1:])
        ax.plot(k, theory, color=INK, lw=1.8, label=f"Theory: geometric, mean {1/(1-p):.0f} days")
        ax.plot(mid, hs / hs.sum() / width, "o", color=col, mfc="white", ms=5, label="Simulated hidden chain")
        ax.plot(mid, np.where(ho > 0, ho / max(ho.sum(), 1) / width, np.nan), "s", color=col, ms=5,
                label=f"Observed (Viterbi), {len(obs)} spells, mean {obs.mean():.0f} days")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Length of stay (trading days)")
        ax.set_ylabel("Probability")
        ax.set_title(f"{'(a)' if i == 0 else '(b)'} {name} regime: sojourn-time distribution")
        ax.legend(frameon=False, fontsize=8.5, loc="lower left")
        ax.set_ylim(bottom=max(theory[theory > 0].min(), 1e-5 if i == 1 else 1e-4) / 2)
        NUM[f"sojourn_{name.lower()}"] = {"theory_mean": float(1 / (1 - p)), "simulated_mean": float(sim.mean()),
                                          "observed_viterbi_mean": float(obs.mean()), "observed_spells": int(len(obs)),
                                          "observed_viterbi_median": float(np.median(obs))}
    save(fig, "RF04_sojourn_times.png")


def fig_regime_timeline(prices, returns, p_crisis, vix):
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True, gridspec_kw={"height_ratios": [2, 1.3, 1.3]})
    idx = returns.index
    crisis = p_crisis > 0.5
    for ax in axes:
        start = None
        for t, c in zip(idx, crisis):
            if c and start is None:
                start = t
            if not c and start is not None:
                ax.axvspan(start, t, color=CRISIS, alpha=0.12, lw=0)
                start = None
    axes[0].plot(prices.index, prices, color=INK, lw=0.9)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("NIFTY 50 (log scale)")
    axes[0].set_title("NIFTY 50 with the HMM crisis regime shaded (filtered P(crisis) > 0.5)")
    axes[1].plot(idx, p_crisis, color=CRISIS, lw=0.7)
    axes[1].set_ylabel("Filtered P(crisis)")
    axes[1].set_ylim(-0.02, 1.02)
    v = vix.reindex(idx).ffill()
    axes[2].plot(idx, v, color=PURPLE, lw=0.8)
    axes[2].set_ylabel("India VIX")
    save(fig, "RF05_regime_probability_nifty.png")


def fig_poisson(returns, rng):
    crash = (returns <= -0.03).to_numpy()
    n_days = len(crash)
    k = int(crash.sum())
    lam = k / n_days   # per trading day
    days = np.flatnonzero(crash)
    gaps = np.diff(days)
    ks = stats.kstest(gaps, "expon", args=(0, 1 / lam))
    t0 = returns.index[0]
    cal = np.array([(d - t0).days for d in returns.index[crash]], dtype=float)
    T = float((returns.index[-1] - t0).days)
    hk = fit_hawkes(cal, T)
    years = returns.index.year
    counts = pd.Series(crash).groupby(years).sum()
    edges = [0.0] + [float((pd.Timestamp(f"{y}-01-01") - t0).days) for y in range(years.min() + 1, years.max() + 1)] + [T + 1]

    def yearly(ev):
        return np.histogram(ev, bins=edges)[0]

    pois_paths, hawk_paths = [], []
    for _ in range(N_SIM):
        pois_paths.append(np.sort(rng.uniform(0, T, rng.poisson(k / T * T))))
        hawk_paths.append(simulate_hawkes(hk.mu, hk.alpha, hk.beta, T, rng))
    grid = np.linspace(0, T, 400)
    cum = lambda ev: np.searchsorted(ev, grid)
    P_band = np.percentile([cum(e) for e in pois_paths], [2.5, 97.5], axis=0)
    H_band = np.percentile([cum(e) for e in hawk_paths], [2.5, 97.5], axis=0)
    disp_p = np.array([dispersion(yearly(e)) for e in pois_paths])
    disp_h = np.array([dispersion(yearly(e)) for e in hawk_paths])
    n_h = np.array([len(e) for e in hawk_paths])
    obs_disp = dispersion(counts.to_numpy())

    dates = t0 + pd.to_timedelta(grid, unit="D")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), gridspec_kw={"wspace": 0.28})
    ax = axes[0]
    ax.fill_between(dates, *P_band, color=GREY, alpha=0.3, lw=0, label="Poisson, 95% of 1,000 simulations")
    ax.fill_between(dates, *H_band, color=ORANGE, alpha=0.2, lw=0, label="Hawkes, 95% of 1,000 simulations")
    ax.plot(dates, k / T * grid, color=GREY, lw=1.5, ls="--", label="Poisson mean E[N(t)] = lambda t")
    ax.step(returns.index[crash], np.arange(1, k + 1), where="post", color=CRISIS, lw=2, label="Observed N(t)")
    ax.set_ylabel("Cumulative crash days N(t)")
    ax.set_title("(a) Counting process N(t)")
    year_axis(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax = axes[1]
    g = np.sort(gaps)
    surv = 1 - np.arange(1, len(g) + 1) / len(g)
    ax.step(g, np.maximum(surv, 1e-3), where="post", color=CRISIS, lw=2, label=f"Observed gaps (n = {len(gaps)})")
    x = np.linspace(0, g.max(), 300)
    ax.plot(x, np.exp(-lam * x), color=INK, lw=1.8, ls="--", label=f"Theory: Exponential, mean {1/lam:.0f} days")
    ax.set_yscale("log")
    ax.set_ylim(1e-2, 1.05)
    ax.set_xlabel("Trading days between crash days, x")
    ax.set_ylabel("P(gap > x), log scale")
    ax.set_title(f"(b) Inter-arrival times (KS p = {ks.pvalue:.0e})")
    ax.text(0.97, 0.55, f"{(gaps <= 5).mean()*100:.0f}% of gaps are 5 days or less\n(exponential: {(1-np.exp(-lam*5))*100:.0f}%)",
            transform=ax.transAxes, ha="right", fontsize=8.5)
    ax.legend(frameon=False, fontsize=8)
    ax = axes[2]
    kk = np.arange(0, counts.max() + 1)
    hist = np.bincount(counts.to_numpy(), minlength=len(kk)) / len(counts)
    ax.bar(kk, hist, color=CRISIS, alpha=0.55, label=f"Observed, {len(counts)} years")
    ax.plot(kk, stats.poisson.pmf(kk, counts.mean()), "o-", color=INK, ms=3, lw=1.2,
            label=f"Theory: Poisson({counts.mean():.2f})")
    ax.set_xlabel("Crash days in a calendar year")
    ax.set_ylabel("Probability")
    ax.set_title(f"(c) Yearly counts: var/mean = {obs_disp:.1f}")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "RF06_poisson_checks.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    b = np.linspace(0, max(np.nanpercentile(disp_h, 99), obs_disp) * 1.05, 50)
    ax.hist(disp_p, bins=b, color=GREY, alpha=0.6, density=True, label="Simulated Poisson")
    ax.hist(disp_h, bins=b, color=ORANGE, alpha=0.5, density=True, label="Simulated Hawkes (fitted)")
    ax.axvline(obs_disp, color=CRISIS, lw=2.2, label=f"Observed NIFTY: {obs_disp:.1f}")
    ax.set_xlabel("Dispersion index of yearly counts (variance / mean)")
    ax.set_ylabel("Density")
    ax.set_title("(a) Which model can produce the observed clustering?")
    ax.legend(frameon=False)
    ax = axes[1]
    tt = np.arange(0, T + 1)
    inten = np.full(len(tt), hk.mu)
    for e in cal:
        m = tt >= e
        inten[m] += hk.alpha * np.exp(-hk.beta * (tt[m] - e))
    ax.plot(t0 + pd.to_timedelta(tt, unit="D"), inten * 365.25, color=ORANGE, lw=0.9, label="Fitted Hawkes intensity")
    ax.plot(returns.index[crash], np.zeros(k), "|", color=CRISIS, ms=12, label="Crash days")
    ax.axhline(hk.mu * 365.25, color=GREY, ls="--", lw=1, label=f"Background rate {hk.mu*365.25:.2f}/year")
    ax.set_ylabel("Crash intensity (events per year)")
    ax.set_title("(b) Self-exciting intensity: each crash raises the rate")
    year_axis(ax)
    ax.legend(frameon=False, loc="upper right")
    save(fig, "RF07_poisson_vs_hawkes_simulation.png")

    NUM["poisson"] = {"crash_days": k, "trading_days": n_days, "lambda_per_trading_day": lam,
                      "theory_mean_gap": 1 / lam, "observed_mean_gap": float(gaps.mean()),
                      "observed_median_gap": float(np.median(gaps)), "observed_gap_cv": float(gaps.std() / gaps.mean()),
                      "theory_gap_cv": 1.0, "share_gaps_le_5_days": float((gaps <= 5).mean()),
                      "theory_share_gaps_le_5_days": float(1 - np.exp(-lam * 5)),
                      "ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue),
                      "years": int(len(counts)), "yearly_mean": float(counts.mean()),
                      "yearly_variance": float(counts.var(ddof=1)), "dispersion_index": float(obs_disp),
                      "years_with_zero_observed": int((counts == 0).sum()),
                      "years_with_zero_poisson_expected": float(len(counts) * np.exp(-counts.mean())),
                      "max_year_observed": int(counts.max())}
    NUM["poisson"]["prob_one_year_ge_35_under_poisson"] = float(stats.poisson.sf(34, counts.mean()))
    NUM["hawkes"] = {"mu_per_calendar_day": hk.mu, "alpha": hk.alpha, "beta": hk.beta, "branching_ratio": hk.branching_ratio,
                     "background_per_year": hk.mu * 365.25, "stationary_rate_per_year": hk.mu / (1 - hk.branching_ratio) * 365.25,
                     "half_life_calendar_days": float(np.log(2) / hk.beta),
                     "expected_events_theory": float(hk.mu * T / (1 - hk.branching_ratio)), "observed_events": k,
                     "simulated_events_median": float(np.median(n_h)),
                     "simulated_events_95": [float(x) for x in np.percentile(n_h, [2.5, 97.5])],
                     "dispersion_poisson_sim_95": [float(x) for x in np.nanpercentile(disp_p, [2.5, 97.5])],
                     "dispersion_hawkes_sim_95": [float(x) for x in np.nanpercentile(disp_h, [2.5, 97.5])],
                     "share_poisson_sims_ge_observed": float(np.mean(disp_p >= obs_disp)),
                     "share_hawkes_sims_ge_observed": float(np.nanmean(disp_h >= obs_disp))}


def fig_sample_paths_and_theory(returns, prices, fit, garch, rng):
    r = returns.to_numpy()
    n = len(r)
    mu_g, sd_g = r.mean(), r.std()
    P, mu, sd = fit.transmat, fit.means, fit.stds
    a, b = P[0, 1], P[1, 0]
    pi = np.array([b, a]) / (a + b)
    lam = 1 - a - b
    sims = {"GBM": rng.normal(mu_g, sd_g, n), "GARCH(1,1)": simulate_garch(garch, n, rng)}
    hmm_r, hmm_s = simulate_hmm(P, mu, sd, pi, n, rng)
    sims["HMM 2-state"] = hmm_r
    # (RF08) returns: actual vs one simulated path of each model
    fig, axes = plt.subplots(4, 1, figsize=(11, 8.5), sharex=True, sharey=True)
    series = [("NIFTY 50, actual 2008-2026", r, INK)] + [(f"{k}: one simulated path", v, c)
                                                          for (k, v), c in zip(sims.items(), [GREY, ORANGE, CALM])]
    for ax, (name, x, c) in zip(axes, series):
        ax.plot(np.arange(n), x * 100, color=c, lw=0.5)
        ax.set_title(name, loc="left", fontsize=10)
        ax.set_ylabel("Return (%)")
    axes[-1].set_xlabel("Trading day")
    axes[0].set_ylim(-15, 15)
    save(fig, "RF08_simulated_sample_paths.png")

    # (RF09) price fan: 500 paths each, GBM vs HMM, vs actual
    reps = 500
    gbm_paths = np.exp(np.cumsum(rng.normal(mu_g, sd_g, (reps, n)), axis=1)) * prices.iloc[0]
    hmm_paths = np.empty((reps, n))
    for i in range(reps):
        x, _ = simulate_hmm(P, mu, sd, pi, n, rng)
        hmm_paths[i] = np.exp(np.cumsum(x)) * prices.iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
    dd = returns.index
    for ax, paths, c, name in [(axes[0], gbm_paths, GREY, "GBM (random walk)"), (axes[1], hmm_paths, CALM, "2-state HMM")]:
        q = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)
        ax.fill_between(dd, q[0], q[4], color=c, alpha=0.18, lw=0, label="5-95% of 500 simulated paths")
        ax.fill_between(dd, q[1], q[3], color=c, alpha=0.3, lw=0, label="25-75%")
        for i in range(5):
            ax.plot(dd, paths[i], color=c, lw=0.5, alpha=0.8)
        ax.plot(dd, prices.iloc[1:].to_numpy(), color=INK, lw=1.4, label="Actual NIFTY 50")
        ax.set_yscale("log")
        ax.set_title(name)
        year_axis(ax)
        ax.legend(frameon=False, loc="upper left", fontsize=8)
    axes[0].set_ylabel("Index level (log scale)")
    save(fig, "RF09_simulated_price_paths.png")

    # simulation statistics: 500 paths per model
    lags = np.arange(1, 101)
    stats_rows = []
    acf_sims = {}
    for name in ["GBM", "GARCH(1,1)", "HMM 2-state"]:
        k_list, a1, a20, acfs = [], [], [], []
        for i in range(300):
            if name == "GBM":
                x = rng.normal(mu_g, sd_g, n)
            elif name == "GARCH(1,1)":
                x = simulate_garch(garch, n, rng)
            else:
                x, _ = simulate_hmm(P, mu, sd, pi, n, rng)
            k_list.append(stats.kurtosis(x))
            ac = acf(x**2, lags)
            acfs.append(ac)
            a1.append(ac[0])
            a20.append(ac[19])
        acf_sims[name] = np.mean(acfs, axis=0)
        stats_rows.append({"model": name, "excess_kurtosis_mean": np.mean(k_list),
                           "excess_kurtosis_95_low": np.percentile(k_list, 2.5), "excess_kurtosis_95_high": np.percentile(k_list, 97.5),
                           "acf_sq_lag1_mean": np.mean(a1), "acf_sq_lag1_95": [np.percentile(a1, 2.5), np.percentile(a1, 97.5)],
                           "acf_sq_lag20_mean": np.mean(a20), "acf_sq_lag20_95": [np.percentile(a20, 2.5), np.percentile(a20, 97.5)]})
    emp_acf = acf(r**2, lags)
    th_acf = hmm_acf_sq_theory(pi, mu, sd, lam, lags)
    m, var, skew, exk = mixture_moments(pi, mu, sd)
    stats_rows.insert(0, {"model": "NIFTY 50 actual", "excess_kurtosis_mean": float(stats.kurtosis(r)),
                          "acf_sq_lag1_mean": float(emp_acf[0]), "acf_sq_lag20_mean": float(emp_acf[19])})
    stats_rows.append({"model": "HMM theory (formula)", "excess_kurtosis_mean": float(exk),
                       "acf_sq_lag1_mean": float(th_acf[0]), "acf_sq_lag20_mean": float(th_acf[19])})
    pd.DataFrame(stats_rows).to_csv(OUT / "simulation_stats.csv", index=False)
    NUM["simulation_stats"] = stats_rows
    NUM["hmm_theory"] = {"stationary": pi.tolist(), "second_eigenvalue": float(lam),
                         "acf_decay_half_life_days": float(np.log(0.5) / np.log(lam)),
                         "mixture_mean_daily": float(m), "mixture_vol_annual": float(np.sqrt(var * 252)),
                         "mixture_skew": float(skew), "mixture_excess_kurtosis": float(exk),
                         "empirical_vol_annual": float(r.std() * np.sqrt(252)), "empirical_skew": float(stats.skew(r)),
                         "acf_sq_theory": {f"lag{k}": float(th_acf[k - 1]) for k in (1, 5, 10, 20, 50, 100)},
                         "acf_sq_empirical": {f"lag{k}": float(emp_acf[k - 1]) for k in (1, 5, 10, 20, 50, 100)}}

    # (RF10) theory vs simulation vs data
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    ax = axes[0]
    ci = 1.96 / np.sqrt(n)
    ax.bar(lags, emp_acf, color=INK, alpha=0.35, width=0.9, label="NIFTY 50 actual")
    ax.plot(lags, th_acf, color=CALM, lw=2.2, label=r"HMM theory: $c\,\lambda^k$, $\lambda$ = %.4f" % lam)
    ax.plot(lags, acf_sims["HMM 2-state"], color=CALM, ls=":", lw=1.8, label="HMM, mean of 300 simulations")
    ax.plot(lags, acf_sims["GARCH(1,1)"], color=ORANGE, lw=1.6, label="GARCH(1,1), mean of 300 simulations")
    ax.plot(lags, acf_sims["GBM"], color=GREY, lw=1.6, label="GBM (theory 0)")
    ax.axhspan(-ci, ci, color=GREY, alpha=0.15, lw=0)
    ax.set_xlabel("Lag k (trading days)")
    ax.set_ylabel(r"Autocorrelation of $r_t^2$")
    ax.set_title("(a) Volatility clustering: ACF of squared returns")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    xg = np.linspace(-0.12, 0.12, 600)
    dens_mix = sum(pi[i] * stats.norm.pdf(xg, mu[i], sd[i]) for i in range(2))
    ax.hist(r, bins=160, density=True, color=INK, alpha=0.3, label="NIFTY 50 actual")
    ax.plot(xg, stats.norm.pdf(xg, mu_g, sd_g), color=GREY, lw=1.8, label="Normal (GBM)")
    ax.plot(xg, dens_mix, color=CALM, lw=2, label="HMM stationary mixture (theory)")
    ax.set_yscale("log")
    ax.set_ylim(1e-2, 80)
    ax.set_xlabel("Daily log return")
    ax.set_ylabel("Density (log scale)")
    ax.set_title(f"(b) Fat tails: excess kurtosis {stats.kurtosis(r):.1f} actual, {exk:.1f} HMM theory, 0 GBM")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "RF10_theory_vs_simulation_vs_data.png")


def fig_parameter_recovery(returns, fit, rng):
    r = returns.to_numpy()
    own = fit_em(r, tol=0, max_iter=2000)   # our MLE on the real data = "truth" for the study
    P, mu, sd = own.transmat, own.means, own.stds
    pi = np.array([P[1, 0], P[0, 1]]) / (P[0, 1] + P[1, 0])
    n, reps = len(r), 100
    est = []
    for i in range(reps):
        x, _ = simulate_hmm(P, mu, sd, pi, n, rng)
        e = fit_em(x)
        est.append([e.means[0], e.means[1], e.stds[0], e.stds[1], e.transmat[0, 1], e.transmat[1, 0]])
    est = np.array(est)
    truth = [mu[0], mu[1], sd[0], sd[1], P[0, 1], P[1, 0]]
    names = [r"Calm mean $\mu_0$", r"Crisis mean $\mu_1$", r"Calm vol $\sigma_0$", r"Crisis vol $\sigma_1$",
             r"P(calm$\to$crisis)", r"P(crisis$\to$calm)"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2))
    rows = []
    for ax, j, name in zip(axes.flat, range(6), names):
        ax.hist(est[:, j], bins=20, color=CALM if j in (0, 2, 4) else CRISIS, alpha=0.55)
        ax.axvline(truth[j], color=INK, lw=2, label="True value")
        ax.axvline(est[:, j].mean(), color=ORANGE, lw=1.6, ls="--", label="Mean estimate")
        ax.set_title(name)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))
        bias = est[:, j].mean() - truth[j]
        se = est[:, j].std(ddof=1)
        rows.append({"parameter": name.replace("$", "").replace("\\", ""), "true": truth[j], "mean_estimate": est[:, j].mean(),
                     "std": se, "bias": bias, "bias_over_std": bias / se,
                     "bias_t_stat": bias / (se / np.sqrt(reps)), "relative_bias_pct": bias / abs(truth[j]) * 100})
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Parameter recovery: {reps} simulated NIFTY-length samples refitted with our own EM", y=1.0)
    fig.tight_layout()
    save(fig, "RF11_parameter_recovery.png")
    pd.DataFrame(rows).to_csv(OUT / "parameter_recovery_own_em.csv", index=False)
    NUM["parameter_recovery_own_em"] = rows

    # (RF12) EM convergence + comparison with hmmlearn
    path = np.array(own.log_likelihood_path)
    from hmmlearn.hmm import GaussianHMM
    m = GaussianHMM(2, covariance_type="diag", n_iter=2000, tol=1e-10, random_state=42,
                    covars_prior=0.0, covars_weight=0.0, min_covar=1e-12)
    m.fit(r.reshape(-1, 1))
    ll_lib = m.score(r.reshape(-1, 1))
    ll_prior = fit.model.score(r.reshape(-1, 1))
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(np.arange(1, len(path) + 1), path, "o-", color=CALM, ms=3, lw=1.4, label="Our Baum-Welch EM")
    ax.axhline(ll_lib, color=INK, ls="--", lw=1.2, label=f"hmmlearn, no prior: {ll_lib:.3f}")
    ax.axhline(ll_prior, color=ORANGE, ls=":", lw=1.4, label=f"hmmlearn, default prior: {ll_prior:.3f}")
    ax.set_xlabel("EM iteration")
    ax.set_ylabel("Log-likelihood")
    ax.set_ylim(path[2] - 2, max(path) + 3)
    ax.set_title(f"EM converges monotonically to {own.log_likelihood:.3f} in {len(path)} iterations")
    ax.legend(frameon=False)
    save(fig, "RF12_em_convergence.png")
    NUM["own_em"] = {"log_likelihood": own.log_likelihood, "iterations": len(path), "means": own.means.tolist(),
                     "stds": own.stds.tolist(), "transmat": own.transmat.tolist(),
                     "hmmlearn_no_prior_log_likelihood": float(ll_lib),
                     "hmmlearn_default_prior_log_likelihood": float(ll_prior),
                     "hmmlearn_default_stds": fit.stds.tolist(), "hmmlearn_default_means": fit.means.tolist(),
                     "hmmlearn_default_transmat": fit.transmat.tolist()}


def fig_var_covid():
    p = pd.read_csv(ROOT / "reports" / "multi_market" / "nifty50_var_paths.csv", index_col=0, parse_dates=True)
    w = p.loc["2020-01-01":"2020-07-31"]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.bar(w.index, w["return"] * 100, color=INK, alpha=0.45, width=1.0, label="Daily return")
    for m, c, ls in [("GBM", GREY, "--"), ("GARCH(1,1)", ORANGE, "-"), ("HMM-gated GARCH", CALM, "-")]:
        ax.plot(w.index, -w[f"var|{m}"] * 100, color=c, ls=ls, lw=1.8, label=f"-VaR, {m}")
        h = w[w[f"hit|{m}"] == 1]
        ax.plot(h.index, h["return"] * 100, "v", color=c, ms=7, mec="white", mew=0.6)
    ax.set_ylabel("Daily log return / -VaR (%)")
    ax.set_title("NIFTY 50, COVID crash 2020: one-day 99% VaR forecasts (out of sample); triangles = violations")
    ax.legend(frameon=False, loc="lower left", ncol=2)
    save(fig, "RF13_nifty_covid_var.png")


def main():
    prices = load_close("nifty50")
    returns = log_returns(prices)
    vix = load_close("india_vix")
    fit = fit_gaussian_hmm(returns, n_states=2, n_iter=200, random_state=42)
    P, mu, sd = fit.transmat, fit.means, fit.stds
    pi = np.array([P[1, 0], P[0, 1]]) / (P[0, 1] + P[1, 0])
    NUM["hmm_fit"] = {"means": mu.tolist(), "stds": sd.tolist(), "transmat": P.tolist(), "stationary": pi.tolist(),
                      "expected_stay_days": (1 / (1 - np.diag(P))).tolist(),
                      "annual_mean": (mu * 252).tolist(), "annual_vol": (sd * np.sqrt(252)).tolist()}
    garch = fit_garch(returns)
    NUM["garch_fit"] = {"mu": garch.mu, "omega": garch.omega, "alpha": garch.alpha, "beta": garch.beta,
                        "persistence": garch.alpha + garch.beta,
                        "half_life_days": float(np.log(0.5) / np.log(garch.alpha + garch.beta)),
                        "unconditional_vol_annual": float(np.sqrt(garch.omega / (1 - garch.alpha - garch.beta) * 252))}
    r = returns.to_numpy()
    NUM["gbm_fit"] = {"mu_daily_log": float(r.mean()), "sigma_daily": float(r.std(ddof=1)),
                      "sigma_annual": float(r.std(ddof=1) * np.sqrt(252)),
                      "mu_annual_gbm_drift": float(r.mean() * 252 + 0.5 * r.var(ddof=1) * 252)}
    p_crisis = filtered_state_probs(fit, returns)[:, 1]
    vit = viterbi_regimes(fit, returns).to_numpy()

    fig_pipeline()
    fig_state_diagram(P, mu, sd, pi)
    fig_transition_nstep(P, pi, RNG)
    fig_sojourn(P, vit, RNG)
    fig_regime_timeline(prices, returns, p_crisis, vix)
    fig_poisson(returns, RNG)
    fig_sample_paths_and_theory(returns, prices, fit, garch, RNG)
    fig_parameter_recovery(returns, fit, RNG)
    fig_var_covid()
    with open(OUT / "report_numbers.json", "w") as f:
        json.dump(NUM, f, indent=2, default=float)
    print("wrote report_numbers.json")


if __name__ == "__main__":
    main()
