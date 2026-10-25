"""GARCH(1,1) via the `arch` package -- the industry-standard competitor to
both GBM and the HMM regime model.

Conditional variance recursion:
    sigma2_t = omega + alpha * resid_{t-1}^2 + beta * sigma2_{t-1}

`arch_model` is fit on returns*100 (percent scale) because its optimizer is
tuned for that scale and converges poorly on raw daily log returns (~1e-2
magnitude); all fitted quantities are converted back to the original return
scale immediately after fitting, by the dataclass below, so everything
downstream works in plain log-return units.

Like hmm.py, the forecast function used by the rolling VaR backtest refits
(by QMLE) only periodically -- MLE refitting daily is unrealistic practice --
and advances the conditional-variance recursion by hand between refits using
only the fixed, already-fitted parameters and new observations as they
arrive. Nothing it uses is ever drawn from data at or after the forecast date.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from arch import arch_model

SCALE = 100.0  # arch_model is fit on returns * SCALE for numerical stability


@dataclass
class GARCHFit:
    mu: float       # daily mean, original return scale
    omega: float    # daily variance scale, original return scale
    alpha: float
    beta: float
    last_resid: float    # resid_T = r_T - mu, original scale, at the end of the fit sample
    last_sigma2: float   # sigma2_T (conditional variance AT time T, i.e. using info through T-1)


def fit_garch(returns: pd.Series) -> GARCHFit:
    """Fit GARCH(1,1) by QMLE (normal quasi-likelihood)."""
    scaled = returns.to_numpy() * SCALE
    am = arch_model(scaled, mean="Constant", vol="GARCH", p=1, q=1, dist="normal")
    res = am.fit(disp="off")

    params = res.params
    mu = params["mu"] / SCALE
    omega = params["omega"] / SCALE**2
    alpha = params["alpha[1]"]
    beta = params["beta[1]"]

    last_resid = returns.iloc[-1] - mu
    last_sigma2 = float(res.conditional_volatility[-1] ** 2) / SCALE**2

    return GARCHFit(mu=mu, omega=omega, alpha=alpha, beta=beta, last_resid=last_resid, last_sigma2=last_sigma2)


def _next_sigma2(fit: GARCHFit) -> float:
    """sigma2_{T+1} from the recursion, given resid_T and sigma2_T stored in `fit`."""
    return fit.omega + fit.alpha * fit.last_resid**2 + fit.beta * fit.last_sigma2


def one_step_forecast(fit: GARCHFit) -> tuple[float, float]:
    """(mu_daily, sigma_daily) forecast for the day after the fit's last observation."""
    sigma2_next = _next_sigma2(fit)
    return fit.mu, float(np.sqrt(sigma2_next))


def _advance_state(fit: GARCHFit, new_return: float) -> GARCHFit:
    """Roll the recursion forward by one observed return, without refitting
    parameters: the new day's conditional variance becomes last_sigma2, and
    its residual becomes last_resid, ready for the next one_step_forecast."""
    sigma2_new = _next_sigma2(fit)
    resid_new = new_return - fit.mu
    fit.last_sigma2 = sigma2_new
    fit.last_resid = resid_new
    return fit


def make_garch_forecast_fn(refit_every: int = 21):
    """Build a `forecast_fn(history) -> (mu_daily, sigma_daily)` for the rolling
    VaR backtest harness in var_backtest.py. Refits by QMLE every `refit_every`
    trading days (~monthly by default; `arch` fits are cheap, so this can be
    set much lower than the HMM's refit cadence if desired)."""
    state: dict = {"fit": None, "fit_at_len": -10**9, "last_n": None}

    def forecast_fn(history: pd.Series) -> tuple[float, float]:
        n = len(history)
        need_refit = state["fit"] is None or (n - state["fit_at_len"]) >= refit_every

        if need_refit:
            state["fit"] = fit_garch(history)
            state["fit_at_len"] = n
            state["last_n"] = n
        elif state["last_n"] == n - 1:
            state["fit"] = _advance_state(state["fit"], history.iloc[-1])
            state["last_n"] = n
        else:
            state["fit"] = fit_garch(history)
            state["fit_at_len"] = n
            state["last_n"] = n

        return one_step_forecast(state["fit"])

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

    print("=== Full-sample GARCH(1,1) fit ===")
    fit = fit_garch(returns)
    persistence = fit.alpha + fit.beta
    print(
        f"mu={fit.mu:.6f} omega={fit.omega:.3e} alpha={fit.alpha:.4f} "
        f"beta={fit.beta:.4f} persistence={persistence:.4f}"
    )

    print("\n=== Rolling out-of-sample VaR backtest: GARCH(1,1), normal (refit every 21d) ===")
    forecast_fn = make_garch_forecast_fn(refit_every=21)
    result = rolling_var_backtest(returns, forecast_fn, alpha=0.99, min_window=500)
    print(f"Backtest: {result.n_obs} obs, {result.n_violations} violations, rate={result.violation_rate:.4%}")

    kupiec = kupiec_test(result)
    print("Kupiec:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in kupiec.items()})

    christoffersen = christoffersen_test(result)
    print("Christoffersen:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in christoffersen.items()})

    traffic = basel_traffic_light(result)
    print(traffic["zone"].value_counts())
