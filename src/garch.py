"""GARCH(1,1) and GJR-GARCH(1,1) via the `arch` package -- the industry-standard
competitor to both GBM and the HMM regime model.

Conditional variance recursion (GJR-GARCH; o=0 collapses it to plain GARCH):
    sigma2_t = omega + alpha * resid_{t-1}^2 + gamma * resid_{t-1}^2 * 1[resid_{t-1} < 0] + beta * sigma2_{t-1}

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
from scipy import stats as sstats

SCALE = 100.0  # arch_model is fit on returns * SCALE for numerical stability


@dataclass
class GARCHFit:
    mu: float       # daily mean, original return scale
    omega: float    # daily variance scale, original return scale
    alpha: float
    beta: float
    gamma: float    # 0 for plain GARCH, >0 for GJR-GARCH (asymmetric/leverage term)
    dist: str       # "normal" or "t"
    nu: float | None  # Student-t degrees of freedom, if dist == "t"
    last_resid: float    # resid_T = r_T - mu, original scale, at the end of the fit sample
    last_sigma2: float   # sigma2_T (conditional variance AT time T, i.e. using info through T-1)


def fit_garch(returns: pd.Series, o: int = 0, dist: str = "normal") -> GARCHFit:
    """Fit GARCH(1,1) (o=0) or GJR-GARCH(1,1) (o=1) by QMLE.

    o: order of the asymmetric term in arch's GARCH model (o=1 -> GJR-GARCH).
    dist: "normal" or "t" (Student-t innovations, for fatter conditional tails).
    """
    scaled = returns.to_numpy() * SCALE
    am = arch_model(scaled, mean="Constant", vol="GARCH", p=1, o=o, q=1, dist=dist)
    res = am.fit(disp="off")

    params = res.params
    mu = params["mu"] / SCALE
    omega = params["omega"] / SCALE**2
    alpha = params["alpha[1]"]
    beta = params["beta[1]"]
    gamma = params["gamma[1]"] if o > 0 else 0.0
    nu = params["nu"] if dist == "t" else None

    last_resid = (returns.iloc[-1] - mu)
    last_sigma2 = float(res.conditional_volatility[-1] ** 2) / SCALE**2

    return GARCHFit(
        mu=mu, omega=omega, alpha=alpha, beta=beta, gamma=gamma,
        dist=dist, nu=nu, last_resid=last_resid, last_sigma2=last_sigma2,
    )


def _next_sigma2(fit: GARCHFit) -> float:
    """sigma2_{T+1} from the recursion, given resid_T and sigma2_T stored in `fit`."""
    asym = fit.gamma * fit.last_resid**2 if fit.last_resid < 0 else 0.0
    return fit.omega + fit.alpha * fit.last_resid**2 + asym + fit.beta * fit.last_sigma2


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


def make_garch_forecast_fn(o: int = 0, dist: str = "normal", refit_every: int = 21):
    """Build a `forecast_fn(history) -> (mu_daily, sigma_daily)` for the rolling
    VaR backtest harness in var_backtest.py. Refits by QMLE every `refit_every`
    trading days (~monthly by default; `arch` fits are cheap, so this can be
    set much lower than the HMM's refit cadence if desired)."""
    state: dict = {"fit": None, "fit_at_len": -10**9, "last_n": None}

    def forecast_fn(history: pd.Series) -> tuple[float, float]:
        n = len(history)
        need_refit = state["fit"] is None or (n - state["fit_at_len"]) >= refit_every

        if need_refit:
            state["fit"] = fit_garch(history, o=o, dist=dist)
            state["fit_at_len"] = n
            state["last_n"] = n
        elif state["last_n"] == n - 1:
            state["fit"] = _advance_state(state["fit"], history.iloc[-1])
            state["last_n"] = n
        else:
            state["fit"] = fit_garch(history, o=o, dist=dist)
            state["fit_at_len"] = n
            state["last_n"] = n

        return one_step_forecast(state["fit"])

    return forecast_fn


def student_t_var(mu_daily: float, sigma_daily: float, alpha: float, nu: float) -> float:
    """Parametric VaR under standardized (unit-variance) Student-t innovations,
    matching arch's internal parameterization: sigma_daily is the conditional
    std regardless of nu, so the raw t quantile is rescaled by sqrt((nu-2)/nu)."""
    z = sstats.t.ppf(1 - alpha, nu) * np.sqrt((nu - 2) / nu)
    return -(mu_daily + sigma_daily * z)


def student_t_es(mu_daily: float, sigma_daily: float, alpha: float, nu: float) -> float:
    q = sstats.t.ppf(1 - alpha, nu)
    scale_corr = np.sqrt((nu - 2) / nu)
    es_std = -sstats.t.pdf(q, nu) / (1 - alpha) * (nu + q**2) / (nu - 1)
    return -(mu_daily + sigma_daily * scale_corr * es_std)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.loaders import load_close
    from src.returns import log_returns
    from src.var_backtest import basel_traffic_light, christoffersen_test, kupiec_test, rolling_var_backtest

    prices = load_close("nifty50")
    returns = log_returns(prices)

    print("=== Full-sample fits: GARCH(1,1) vs GJR-GARCH(1,1) ===")
    for label, o in [("GARCH(1,1)", 0), ("GJR-GARCH(1,1)", 1)]:
        fit = fit_garch(returns, o=o, dist="normal")
        persistence = fit.alpha + fit.beta + (fit.gamma / 2 if o else 0)
        print(
            f"{label}: mu={fit.mu:.6f} omega={fit.omega:.3e} alpha={fit.alpha:.4f} "
            f"beta={fit.beta:.4f} gamma={fit.gamma:.4f} persistence={persistence:.4f}"
        )

    print("\n=== Full-sample fit with Student-t innovations (GJR-GARCH) ===")
    fit_t = fit_garch(returns, o=1, dist="t")
    print(f"nu (degrees of freedom) = {fit_t.nu:.2f}  (lower = fatter tails; nu<~6 is quite fat)")

    for label, o, dist, refit_every in [
        ("GARCH(1,1), normal", 0, "normal", 21),
        ("GJR-GARCH(1,1), normal", 1, "normal", 21),
        ("GJR-GARCH(1,1), Student-t", 1, "t", 21),
    ]:
        print(f"\n=== Rolling out-of-sample VaR backtest: {label} (refit every {refit_every}d) ===")
        forecast_fn = make_garch_forecast_fn(o=o, dist=dist, refit_every=refit_every)

        if dist == "t":
            # nu drifts slightly across refits; use the full-sample nu as a fixed
            # approximation for the tail-quantile scaling (documented simplification).
            nu_fixed = fit_t.nu
            var_fn = lambda mu, sigma, a: student_t_var(mu, sigma, a, nu_fixed)
            es_fn = lambda mu, sigma, a: student_t_es(mu, sigma, a, nu_fixed)
        else:
            var_fn = es_fn = None

        result = rolling_var_backtest(
            returns, forecast_fn, alpha=0.99, min_window=500, var_fn=var_fn, es_fn=es_fn
        )
        print(f"Backtest: {result.n_obs} obs, {result.n_violations} violations, rate={result.violation_rate:.4%}")

        kupiec = kupiec_test(result)
        print("Kupiec:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in kupiec.items()})

        christoffersen = christoffersen_test(result)
        print(
            "Christoffersen:",
            {k: (round(v, 4) if isinstance(v, float) else v) for k, v in christoffersen.items()},
        )

        traffic = basel_traffic_light(result)
        print(traffic["zone"].value_counts())
