"""Simulators and theory formulas used throughout the project.

Everything that generates random paths, or computes a theoretical quantity that a
simulation is compared against, lives here, so "where is the simulation code?" has
one answer. Used by src/report_figures.py (figures and simulated data) and the
tests.

Theory (2-state chain, a = P(calm->crisis), b = P(crisis->calm)):
  nstep_closed_form   P^n = 1/(a+b)[[b,a],[b,a]] + (1-a-b)^n/(a+b)[[a,-a],[-b,b]]
  mixture_moments     mean, variance, skew, excess kurtosis of the stationary HMM mixture
  hmm_acf_sq_theory   Corr(r_t^2, r_{t+k}^2) = Var_pi(g(S)) lam^k / Var(r^2), lam = 1-a-b
Simulators:
  simulate_chain      Markov chain path
  simulate_hmm        Gaussian HMM return path (and its hidden states)
  simulate_garch      GARCH(1,1) return path
  simulate_hawkes     Hawkes event times by Ogata thinning
Helpers:
  acf, spells, dispersion
"""

from __future__ import annotations

import numpy as np


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
