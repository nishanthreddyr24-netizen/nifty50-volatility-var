"""Our own Baum-Welch (EM) estimator for a Gaussian hidden Markov model.

The project's main fits use hmmlearn. This module implements the same estimator
from scratch, so the estimation step is our own code, and checks it against
hmmlearn on the same data (see report_figures.py and tests/test_em_hmm.py).

Model: hidden chain S_t on {0, ..., K-1} with transition matrix A and initial
distribution pi; observation r_t | S_t = k ~ Normal(mu_k, sigma_k^2).

E-step (forward-backward, scaled): with c_t the scaling constants,
    alpha_t(k) = P(S_t = k | r_1..r_t)
    beta_t(k)  proportional to P(r_{t+1}..r_T | S_t = k)
    gamma_t(k) = P(S_t = k | r_1..r_T) = alpha_t(k) beta_t(k)
    xi_t(i, j) = P(S_t = i, S_{t+1} = j | r_1..r_T)
               = alpha_t(i) A_ij b_j(r_{t+1}) beta_{t+1}(j) / c_{t+1}
    log L = sum_t log c_t
M-step (closed form):
    pi_k   = gamma_1(k)
    A_ij   = sum_t xi_t(i, j) / sum_t gamma_t(i)          (t = 1..T-1)
    mu_k   = sum_t gamma_t(k) r_t / sum_t gamma_t(k)
    sigma_k^2 = sum_t gamma_t(k) (r_t - mu_k)^2 / sum_t gamma_t(k)
Each iteration cannot decrease log L (Dempster, Laird and Rubin, 1977).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass
class EMResult:
    means: np.ndarray
    stds: np.ndarray
    transmat: np.ndarray
    startprob: np.ndarray
    log_likelihood: float
    n_iter: int
    log_likelihood_path: list


def forward_backward(x: np.ndarray, means, stds, A, pi):
    """Scaled forward-backward. Returns gamma (T, K), xi summed over t (K, K), log L."""
    T, K = len(x), len(means)
    b = norm.pdf(x[:, None], loc=means[None, :], scale=stds[None, :])  # (T, K) emission densities
    alpha = np.empty((T, K))
    c = np.empty(T)
    a = pi * b[0]
    c[0] = a.sum()
    alpha[0] = a / c[0]
    for t in range(1, T):
        a = (alpha[t - 1] @ A) * b[t]
        c[t] = a.sum()
        alpha[t] = a / c[t]
    beta = np.empty((T, K))
    beta[-1] = 1.0
    xi_sum = np.zeros((K, K))
    for t in range(T - 2, -1, -1):
        w = b[t + 1] * beta[t + 1]
        beta[t] = (A @ w) / c[t + 1]
        xi_sum += (alpha[t][:, None] * A * w[None, :]) / c[t + 1]
    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)
    return gamma, xi_sum, float(np.log(c).sum())


def fit_em(x: np.ndarray, n_states: int = 2, max_iter: int = 500, tol: float = 1e-8) -> EMResult:
    """Fit a Gaussian HMM by Baum-Welch. States are returned sorted by volatility
    (0 = calm). Initial values: equal-mass quantile split of the data by |x|."""
    x = np.asarray(x, dtype=float)
    K = n_states
    order = np.argsort(np.abs(x - x.mean()))
    groups = np.array_split(order, K)
    means = np.array([x[g].mean() for g in groups])
    stds = np.array([x[g].std() for g in groups])
    A = np.full((K, K), 0.05 / (K - 1)) + np.eye(K) * (0.95 - 0.05 / (K - 1))
    pi = np.full(K, 1.0 / K)

    path, prev = [], -np.inf
    for it in range(1, max_iter + 1):
        gamma, xi_sum, ll = forward_backward(x, means, stds, A, pi)
        path.append(ll)
        pi = gamma[0]
        A = xi_sum / xi_sum.sum(axis=1, keepdims=True)
        w = gamma.sum(axis=0)
        means = (gamma * x[:, None]).sum(axis=0) / w
        stds = np.sqrt((gamma * (x[:, None] - means[None, :]) ** 2).sum(axis=0) / w)
        if ll - prev < tol * abs(ll):
            break
        prev = ll
    _, _, ll = forward_backward(x, means, stds, A, pi)
    s = np.argsort(stds)
    return EMResult(means=means[s], stds=stds[s], transmat=A[np.ix_(s, s)], startprob=pi[s],
                    log_likelihood=ll, n_iter=it, log_likelihood_path=path)
