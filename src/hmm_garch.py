"""HMM-gated GARCH(1,1): a two-step hybrid of the Gaussian HMM (hmm.py) and
GARCH(1,1) (garch.py), fitted separately and combined at forecast time.

The HMM decides *when* the market is in its crisis regime; GARCH decides
*how large* crisis volatility is. Each day:

  1. the 2-state HMM gives next-day regime probabilities (p_calm, p_crisis)
     and its per-regime means/stds (refit every 63 days, causal filter in
     between -- the same scheme as hmm.make_hmm_forecast_fn);
  2. GARCH(1,1) gives its own next-day sigma_garch (refit every 21 days, as
     in garch.make_garch_forecast_fn);
  3. the forecast is the HMM's predictive mixture, except that the crisis
     component's std is max(sigma_crisis_HMM, sigma_garch). This keeps the
     HMM's early alarm (its crisis std is never lowered) and removes its
     volatility ceiling (GARCH can lift it); calm days are untouched.

Unlike msgarch.py this is not a single likelihood model -- the gate is a
rule on top of two separately estimated models -- so it can be judged by
backtests but not by AIC/BIC. Neither underlying module is modified; this
file only calls their public fitting/filtering functions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sstats
from scipy.special import logsumexp

from src.garch import make_garch_forecast_fn
from src.hmm import HMMFit, filtered_state_probs, fit_gaussian_hmm


def gated_forecast(
    means: np.ndarray, stds: np.ndarray, next_probs: np.ndarray, sigma_garch: float
) -> tuple[float, float]:
    """(mu_daily, sigma_daily): first two moments of the HMM mixture with the
    crisis std raised to sigma_garch when GARCH is higher. `means`/`stds`/
    `next_probs` are in vol-sorted regime order (0 = calm, 1 = crisis)."""
    gated_stds = np.array([stds[0], max(stds[1], sigma_garch)])
    mu = float(np.dot(next_probs, means))
    second_moment = float(np.dot(next_probs, gated_stds**2 + means**2))
    return mu, float(np.sqrt(max(second_moment - mu**2, 1e-12)))


def make_hmm_garch_forecast_fn(hmm_refit_every: int = 63, garch_refit_every: int = 21):
    """Build a `forecast_fn(history) -> (mu_daily, sigma_daily)` for the rolling
    VaR backtest harness in var_backtest.py. Uses only data in `history`."""
    hmm_state: dict = {"fit": None, "fit_at_len": -10**9, "log_alpha": None, "last_n": None}
    garch_fn = make_garch_forecast_fn(refit_every=garch_refit_every)

    def _hmm_next_probs(history: pd.Series) -> tuple[HMMFit, np.ndarray]:
        n = len(history)
        need_refit = hmm_state["fit"] is None or (n - hmm_state["fit_at_len"]) >= hmm_refit_every

        if need_refit:
            hmm_state["fit"] = fit_gaussian_hmm(history, n_states=2, n_iter=100)
            hmm_state["fit_at_len"] = n
            hmm_state["log_alpha"] = np.log(filtered_state_probs(hmm_state["fit"], history)[-1] + 1e-300)
        elif hmm_state["last_n"] == n - 1:
            fit = hmm_state["fit"]
            log_emission = sstats.norm.logpdf(history.iloc[-1], loc=fit.means, scale=fit.stds)
            log_alpha = logsumexp(hmm_state["log_alpha"][:, None] + np.log(fit.transmat + 1e-300), axis=0) + log_emission
            hmm_state["log_alpha"] = log_alpha - logsumexp(log_alpha)
        else:
            hmm_state["log_alpha"] = np.log(filtered_state_probs(hmm_state["fit"], history)[-1] + 1e-300)
        hmm_state["last_n"] = n

        fit = hmm_state["fit"]
        return fit, np.exp(hmm_state["log_alpha"]) @ fit.transmat

    def forecast_fn(history: pd.Series) -> tuple[float, float]:
        fit, next_probs = _hmm_next_probs(history)
        _, sigma_garch = garch_fn(history)
        return gated_forecast(fit.means, fit.stds, next_probs, sigma_garch)

    return forecast_fn


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.loaders import load_close
    from src.returns import log_returns
    from src.var_backtest import basel_traffic_light, christoffersen_test, kupiec_test, rolling_var_backtest

    prices = load_close("nifty50")
    returns = log_returns(prices)

    print("=== Rolling out-of-sample VaR backtest: HMM-gated GARCH (HMM refit 63d, GARCH refit 21d) ===")
    forecast_fn = make_hmm_garch_forecast_fn()
    result = rolling_var_backtest(returns, forecast_fn, alpha=0.99, min_window=500)
    print(f"Backtest: {result.n_obs} obs, {result.n_violations} violations, rate={result.violation_rate:.4%}")

    kupiec = kupiec_test(result)
    print("Kupiec:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in kupiec.items()})

    christoffersen = christoffersen_test(result)
    print("Christoffersen:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in christoffersen.items()})

    traffic = basel_traffic_light(result)
    print(traffic["zone"].value_counts())

    covid = result.table.loc["2020-02-15":"2020-06-30"]
    print(f"COVID window: {int(covid['hit'].sum())}/{len(covid)} violations")
