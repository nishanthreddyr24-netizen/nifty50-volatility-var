"""Trading test: do our volatility / regime forecasts make money in real quant trades?

Every forecast used here is the out-of-sample one saved by multi_market.py: the
VaR for day t was computed only from returns before day t. The models' daily
volatility forecast is recovered as sigma_t = VaR_t / z_0.99 (z_0.99 = 2.326).
As a no-model practitioner baseline we add "Trailing 21-day vol" (realised
volatility of the previous 21 days).

APPLICATION A - options: the volatility risk premium (VRP) trade
-----------------------------------------------------------------
Index options are usually priced above the volatility that follows, so desks
sell one-month variance (a variance swap, or a delta-hedged straddle) at the
implied level. It pays most months and loses heavily in crashes. VIX (S&P 500)
and India VIX (NIFTY 50) are, by their CBOE-style construction, the 30-day
variance-swap rate, so they are the strike.

  * Every 21 trading days, at the close before day t: strike K = VIX (vol points),
    forecast = model's annualised vol for day t. Hold for days t..t+20.
  * Short variance P&L per $1 of vega notional, in vol points:
        (K_eff^2 - RV) / (2 K)       RV = (252/21) * sum r^2 * 100^2
    K_eff = K - 0.5 when selling, K + 0.5 when buying (0.5 vol-point bid-ask).
  * Strategies: "Always short" (the industry carry trade); per model
    "Filter" = short only if forecast < K, otherwise flat; and
    "Spread" = short if forecast < K, long if forecast > K.

APPLICATION B - quant equity: volatility targeting
--------------------------------------------------
A fund holds the index with a daily 99% VaR budget of 2% of NAV:
exposure w_t = min(1, 0.02 / VaR_t) (no leverage). Cash earns 0 (conservative),
trading cost 5 bp per unit of turnover. Benchmarks: buy-and-hold, and a
constant exposure equal to the strategy's own average exposure (same average
risk; chosen with hindsight, which favours the benchmark).

PASS CRITERIA (fixed before the run)
-----------------------------------
A, per model (Filter): 1) max drawdown smaller than Always short, 2) worst month
smaller loss than Always short, 3) Sharpe >= Always short. Pass = all three.
Significance of the Sharpe gain: one-sided moving-block bootstrap of months
(6-month blocks, 5,000 replications).
B, per model and market: 1) days losing more than the budget not significantly
above 1% (one-sided binomial p >= 0.05), 2) max drawdown smaller than the
constant-exposure benchmark, 3) Sharpe >= buy-and-hold. Pass = all three.
Sharpe gain vs buy-and-hold tested by a daily moving-block bootstrap
(250-day blocks, 2,000 replications).

Usage:  python -m src.trading_applications
Output: reports/multi_market/TRADE_vrp_*.csv, TRADE_voltarget_*.csv,
        figures/fig12_vrp_trade.png, figures/fig13_vol_targeting.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom, norm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.loaders import download_prices
from src.multi_market import MARKETS, OUT_DIR

Z99 = norm.ppf(0.99)
MODELS = ["GBM", "GARCH(1,1)", "HMM 2-state", "MS-GARCH 2-regime", "HMM-gated GARCH",
          "Historical simulation", "Trailing 21-day vol"]
NAMES = {"sp500": "S&P 500", "ftse100": "FTSE 100", "dax": "DAX", "nikkei225": "Nikkei 225",
         "hangseng": "Hang Seng", "sensex": "Sensex", "nifty50": "NIFTY 50"}
HOLD = 21
SPREAD_VOLPTS = 0.5
BUDGET = 0.02
COST = 0.0005
WINDOWS = {"GFC": ("2008-09-01", "2009-03-31"), "Feb 2018 vol spike": ("2018-01-26", "2018-03-31"),
           "COVID": ("2020-02-15", "2020-06-30")}
VIX_SOURCES = {"sp500": ("vix_1990", "^VIX", "1990-01-01"), "nifty50": ("india_vix", "^INDIAVIX", "2008-01-01")}


# ------------------------------------------------------------------ shared

def load_forecasts(market: str) -> pd.DataFrame:
    """Daily return and each model's out-of-sample daily vol forecast for that day."""
    p = pd.read_csv(OUT_DIR / f"{market}_var_paths.csv", index_col=0, parse_dates=True)
    out = pd.DataFrame({"return": p["return"]})
    for m in MODELS[:-1]:
        out[m] = p[f"var|{m}"] / Z99
    # trailing realised vol from the returns before each day (needs the full history)
    full = pd.read_csv(OUT_DIR / f"{market}_regime_flags.csv", index_col=0, parse_dates=True)["return"]
    trail = full.rolling(21).std().shift(1)
    out["Trailing 21-day vol"] = trail.reindex(out.index)
    return out.dropna()


def max_drawdown_additive(pnl: np.ndarray) -> float:
    c = np.cumsum(pnl)
    return float((np.maximum.accumulate(np.concatenate([[0], c]))[1:] - c).max())


def max_drawdown_compound(ret: np.ndarray) -> float:
    w = np.cumprod(1 + ret)
    return float((1 - w / np.maximum.accumulate(w)).max())


def sharpe(x: np.ndarray, periods: int) -> float:
    s = x.std(ddof=1)
    return float(x.mean() / s * np.sqrt(periods)) if s > 0 else np.nan


def block_boot_sharpe_p(a: np.ndarray, b: np.ndarray, periods: int, block: int,
                        n_boot: int, seed: int = 11) -> float:
    """One-sided p-value for Sharpe(a) > Sharpe(b), resampling aligned blocks of (a, b)."""
    rng = np.random.default_rng(seed)
    n = len(a)
    L = min(block, n)
    k = int(np.ceil(n / L))
    starts = rng.integers(0, n - L + 1, size=(n_boot, k))
    idx = (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(n_boot, -1)[:, :n]
    A, B = a[idx], b[idx]

    def sr(X):
        s = X.std(axis=1, ddof=1)
        return np.where(s > 0, X.mean(axis=1) / np.where(s > 0, s, 1), 0) * np.sqrt(periods)

    d = sr(A) - sr(B)
    return float((d <= 0).mean())


def window_mask(index: pd.DatetimeIndex, start: str, end: str) -> np.ndarray:
    return (index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))


# ------------------------------------------------------------ application A

def vrp_trades(market: str) -> pd.DataFrame:
    f = load_forecasts(market)
    name, ticker, start = VIX_SOURCES[market]
    vix = download_prices(name, ticker, start=start)["Close"].dropna()
    vix = vix[vix > 0]
    prev_vix = vix.reindex(f.index.union(vix.index)).ffill().shift(1).reindex(f.index)  # close before day t
    f = f.assign(K=prev_vix).dropna()
    r = f["return"].to_numpy()
    rows = []
    for s in range(0, len(f) - HOLD + 1, HOLD):
        t = f.index[s]
        K = f["K"].iloc[s]
        rv = (252 / HOLD) * np.sum(r[s:s + HOLD] ** 2) * 100**2
        row = {"entry": t, "exit": f.index[s + HOLD - 1], "K": K, "realised_vol": np.sqrt(rv),
               "short_pnl": ((K - SPREAD_VOLPTS) ** 2 - rv) / (2 * K),
               "long_pnl": (rv - (K + SPREAD_VOLPTS) ** 2) / (2 * K)}
        for m in MODELS:
            row[f"fc|{m}"] = f[m].iloc[s] * np.sqrt(252) * 100
        rows.append(row)
    d = pd.DataFrame(rows)
    d.insert(0, "market", market)
    d["Always short"] = d["short_pnl"]
    for m in MODELS:
        cheap = d[f"fc|{m}"] < d["K"]
        d[f"Filter|{m}"] = np.where(cheap, d["short_pnl"], 0.0)
        d[f"Spread|{m}"] = np.where(cheap, d["short_pnl"], d["long_pnl"])
    return d


def vrp_summary(d: pd.DataFrame) -> pd.DataFrame:
    market = d["market"].iloc[0]
    base = d["Always short"].to_numpy()
    strategies = ["Always short"] + [f"Filter|{m}" for m in MODELS] + [f"Spread|{m}" for m in MODELS]
    rows = []
    for s in strategies:
        x = d[s].to_numpy()
        traded = (x != 0).mean()
        row = {"market": market, "strategy": s.split("|")[0], "model": s.split("|")[1] if "|" in s else "",
               "months": len(x), "months_traded_pct": traded * 100,
               "mean_pnl_volpts": x.mean(), "sharpe_ann": sharpe(x, 12),
               "worst_month": x.min(), "max_drawdown_volpts": max_drawdown_additive(x),
               "win_rate_pct": (x[x != 0] > 0).mean() * 100 if traded else np.nan}
        for w, (a, b) in WINDOWS.items():
            mask = (d["exit"] >= pd.Timestamp(a)) & (d["entry"] <= pd.Timestamp(b))
            row[f"pnl_{w}"] = x[mask.to_numpy()].sum()
        if s != "Always short":
            row["sharpe_gain_p"] = block_boot_sharpe_p(x, base, 12, 6, 5000)
        rows.append(row)
    out = pd.DataFrame(rows)
    b = out.iloc[0]
    out["pass_drawdown"] = out["max_drawdown_volpts"] < b["max_drawdown_volpts"]
    out["pass_worst_month"] = out["worst_month"] > b["worst_month"]
    out["pass_sharpe"] = out["sharpe_ann"] >= b["sharpe_ann"]
    out["pass_all"] = out[["pass_drawdown", "pass_worst_month", "pass_sharpe"]].all(axis=1)
    cols = ["pass_drawdown", "pass_worst_month", "pass_sharpe", "pass_all"]
    out[cols] = out[cols].astype(object)
    out.loc[0, cols] = None  # the baseline is not tested against itself
    return out


# ------------------------------------------------------------ application B

def vol_target(market: str) -> tuple[pd.DataFrame, dict]:
    f = load_forecasts(market)
    R = np.expm1(f["return"].to_numpy())
    n_years = len(R) / 252
    rows, curves = [], {"Buy and hold": np.cumprod(1 + R)}
    bh = {"sharpe": sharpe(R, 252), "mdd": max_drawdown_compound(R),
          "cagr": np.prod(1 + R) ** (1 / n_years) - 1, "vol": R.std() * np.sqrt(252)}
    rows.append({"market": market, "model": "Buy and hold", "avg_exposure": 1.0, "cagr_pct": bh["cagr"] * 100,
                 "vol_pct": bh["vol"] * 100, "sharpe": bh["sharpe"], "max_drawdown_pct": bh["mdd"] * 100})
    for m in MODELS:
        var = Z99 * f[m].to_numpy()
        w = np.minimum(1.0, BUDGET / var)
        turnover = np.abs(np.diff(np.concatenate([[w[0]], w])))
        ret = w * R - COST * turnover
        const = w.mean() * R
        loss_breach = (-ret > BUDGET)
        k, n = int(loss_breach.sum()), len(ret)
        row = {"market": market, "model": m, "avg_exposure": w.mean(),
               "cagr_pct": (np.prod(1 + ret) ** (1 / n_years) - 1) * 100,
               "vol_pct": ret.std() * np.sqrt(252) * 100, "sharpe": sharpe(ret, 252),
               "max_drawdown_pct": max_drawdown_compound(ret) * 100,
               "const_exposure_mdd_pct": max_drawdown_compound(const) * 100,
               "turnover_per_year": turnover.sum() / n_years,
               "budget_breach_pct": k / n * 100, "budget_breach_p": float(binom.sf(k - 1, n, 0.01)),
               "sharpe_gain_p": block_boot_sharpe_p(ret, R, 252, 250, 2000)}
        for wname, (a, b) in WINDOWS.items():
            mask = window_mask(f.index, a, b)
            if mask.any():
                row[f"ret_{wname}_pct"] = (np.prod(1 + ret[mask]) - 1) * 100
                row[f"bh_{wname}_pct"] = (np.prod(1 + R[mask]) - 1) * 100
        rows.append(row)
        curves[m] = np.cumprod(1 + ret)
    out = pd.DataFrame(rows)
    out["pass_budget"] = out["budget_breach_p"] >= 0.05
    out["pass_drawdown"] = out["max_drawdown_pct"] < out["const_exposure_mdd_pct"]
    out["pass_sharpe"] = out["sharpe"] >= bh["sharpe"]
    out["pass_all"] = out[["pass_budget", "pass_drawdown", "pass_sharpe"]].all(axis=1)
    cols = ["pass_budget", "pass_drawdown", "pass_sharpe", "pass_all"]
    out[cols] = out[cols].astype(object)
    out.loc[0, cols] = None  # buy-and-hold is the benchmark
    return out, {"index": f.index, **curves}


# ------------------------------------------------------------ figures

def figures(trades: dict[str, pd.DataFrame], curves: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    show = [("Always short", "#9a9a9a"), ("Filter|Trailing 21-day vol", "#7570b3"),
            ("Filter|GARCH(1,1)", "#d95f02"), ("Filter|HMM-gated GARCH", "#1b6ca8")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (mkt, d) in zip(axes, trades.items()):
        for col, c in show:
            label = col.replace("Filter|", "Sell only if forecast < implied: ")
            ax.plot(d["exit"], d[col].cumsum(), color=c, lw=1.8 if "gated" in col else 1.3, label=label)
        ax.axhline(0, color="#000000", lw=0.6)
        vix_name = "India VIX" if mkt == "nifty50" else "VIX"
        ax.set_title(f"{NAMES[mkt]}: sell one-month variance at the {vix_name} level")
        ax.set_ylabel("Cumulative P&L, vol points per $1 vega")
    axes[0].legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figures" / "fig12_vrp_trade.png", dpi=150)
    plt.close(fig)

    c = curves["sp500"]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    for m, col in [("Buy and hold", "#9a9a9a"), ("Trailing 21-day vol", "#7570b3"),
                   ("GARCH(1,1)", "#d95f02"), ("HMM-gated GARCH", "#1b6ca8")]:
        lw = 1.8 if "gated" in m else 1.2
        ax.plot(c["index"], c[m], color=col, lw=lw, label=m)
        ax2.plot(c["index"], (c[m] / np.maximum.accumulate(c[m]) - 1) * 100, color=col, lw=lw)
    ax.set_yscale("log")
    ax.set_ylabel("Growth of 1 (log scale)")
    ax.set_title("S&P 500: 2% daily VaR budget, exposure = min(1, 2% / VaR), 5 bp costs")
    ax.legend(frameon=False)
    ax2.set_ylabel("Drawdown from peak (%)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figures" / "fig13_vol_targeting.png", dpi=150)
    plt.close(fig)


def main() -> None:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)

    trades, vrp = {}, []
    for mkt in VIX_SOURCES:
        d = vrp_trades(mkt)
        trades[mkt] = d
        vrp.append(vrp_summary(d))
    pd.concat(trades.values()).to_csv(OUT_DIR / "TRADE_vrp_trades.csv", index=False)
    vrp = pd.concat(vrp, ignore_index=True)
    vrp.to_csv(OUT_DIR / "TRADE_vrp_summary.csv", index=False)
    print(vrp.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    vt, curves = [], {}
    for mkt in MARKETS:
        out, c = vol_target(mkt)
        vt.append(out)
        curves[mkt] = c
    vt = pd.concat(vt, ignore_index=True)
    vt.to_csv(OUT_DIR / "TRADE_voltarget_summary.csv", index=False)
    print(vt.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    figures(trades, curves)


if __name__ == "__main__":
    main()
