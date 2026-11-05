"""Figures for the seven-market regime-switching VaR study.

Reads the CSVs in reports/multi_market/ (written by multi_market.py,
crisis_duration.py and regime_transitions.py) and writes PNGs to
reports/multi_market/figures/.

Usage:  python -m src.make_multi_market_figures
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.multi_market import CRISIS_WINDOWS, OUT_DIR

FIG_DIR = OUT_DIR / "figures"

# Palette (validated categorical slots 1-3; everything else neutral ink)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f4f8fd", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])

GATED, HMM, MSG = "HMM-gated GARCH", "HMM 2-state", "MS-GARCH 2-regime"
MODEL_STYLE = {  # color, marker, size, zorder
    GATED: (BLUE, "o", 80, 6),
    HMM: (ORANGE, "o", 60, 5),
    MSG: (AQUA, "D", 45, 4),
    "GARCH(1,1)": (MUTED, "s", 34, 3),
    "Historical simulation": (MUTED, "^", 38, 3),
    "GBM": (MUTED, "x", 38, 3),
}
MODEL_ORDER = [GATED, HMM, MSG, "GARCH(1,1)", "Historical simulation", "GBM"]
SHORT = {GATED: "HMM-gated\nGARCH", HMM: "HMM", MSG: "MS-GARCH", "GARCH(1,1)": "GARCH",
         "Historical simulation": "Hist. sim", "GBM": "GBM"}
MARKET_NAME = {"sp500": "S&P 500", "ftse100": "FTSE 100", "dax": "DAX", "nikkei225": "Nikkei 225",
               "hangseng": "Hang Seng", "sensex": "Sensex", "nifty50": "NIFTY 50"}
MARKET_ORDER = ["sp500", "ftse100", "dax", "nikkei225", "hangseng", "sensex", "nifty50"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS,
    "xtick.color": MUTED, "ytick.color": INK2, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def _model_legend(ax, models=MODEL_ORDER, **kw):
    handles = [Line2D([], [], color=MODEL_STYLE[m][0], marker=MODEL_STYLE[m][1], linestyle="",
                      markersize=8 if m != GATED else 9, label=m) for m in models]
    ax.legend(handles=handles, **kw)


def fig_violation_rates(bt):
    fig, ax = plt.subplots(figsize=(9, 5))
    ys = np.arange(len(MARKET_ORDER))[::-1]
    for m in MODEL_ORDER[::-1]:
        c, mk, s, z = MODEL_STYLE[m]
        sub = bt[bt.model == m].set_index("market").reindex(MARKET_ORDER)
        ax.scatter(sub["rate"] * 100, ys, color=c, marker=mk, s=s, zorder=z,
                   edgecolor=SURFACE if mk not in "x" else None, linewidth=1.5)
    ax.axvline(1.0, color=INK, lw=1.2, ls="--", zorder=2)
    ax.text(1.0, len(ys) - 0.35, "target 1%", ha="center", va="bottom", color=INK, fontsize=9)
    ax.set_yticks(ys, [MARKET_NAME[m] for m in MARKET_ORDER])
    ax.set_xlabel("Share of days the loss exceeded the 99% VaR (%)")
    ax.set_ylim(-0.6, len(ys) - 0.1)
    ax.grid(axis="y", visible=False)
    ax.set_title("HMM-gated GARCH is closest to the 1% target in 6 of 7 markets")
    _model_legend(ax, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    _save(fig, "fig01_violation_rate_by_market.png")


def fig_dumbbell(bt):
    periods = [("rate", "Whole backtest"), ("GFC_rate", "GFC (Sep 2008 - Mar 2009)"),
               ("COVID_rate", "COVID (Feb - Jun 2020)")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True)
    ys = np.arange(len(MARKET_ORDER))[::-1]
    h = bt[bt.model == HMM].set_index("market").reindex(MARKET_ORDER)
    g = bt[bt.model == GATED].set_index("market").reindex(MARKET_ORDER)
    for ax, (col, title) in zip(axes, periods):
        hv, gv = h[col].to_numpy() * 100, g[col].to_numpy() * 100
        for y, a, b in zip(ys, hv, gv):
            if np.isfinite(a) and np.isfinite(b):
                ax.plot([a, b], [y, y], color=AXIS, lw=2, zorder=1, solid_capstyle="round")
        ax.scatter(hv, ys, color=ORANGE, s=60, zorder=3, edgecolor=SURFACE, linewidth=1.5)
        ax.scatter(gv, ys, color=BLUE, s=70, zorder=4, edgecolor=SURFACE, linewidth=1.5)
        same = np.isclose(hv, gv) & np.isfinite(hv)
        ax.scatter(gv[same], ys[same], s=150, facecolor="none", edgecolor=ORANGE, linewidth=1.8, zorder=5)
        for y, x in zip(ys[same], gv[same]):
            ax.text(x, y + 0.28, "same", ha="center", fontsize=8, color=INK2)
        ax.axvline(1.0, color=INK, lw=1, ls="--", zorder=0)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Violation rate (%)")
        ax.set_xlim(left=0)
    axes[0].set_yticks(ys, [MARKET_NAME[m] for m in MARKET_ORDER])
    axes[1].text(0.98, 0.02, "NIFTY 50 backtest starts 2010:\nno GFC days", transform=axes[1].transAxes,
                 ha="right", va="bottom", color=MUTED, fontsize=8.5)
    _model_legend(axes[2], models=[HMM, GATED], loc="lower right")
    fig.suptitle("Adding the GARCH floor to the HMM cuts violations, most of all at crisis onset (dashed = 1% target)",
                 x=0.06, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, "fig02_hmm_vs_gated_by_period.png")


def fig_sp500_gfc():
    p = pd.read_csv(OUT_DIR / "sp500_var_paths.csv", index_col=0, parse_dates=True)
    p = p.loc["2007-07-01":"2009-12-31"]
    loss = -p["return"] * 100
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.vlines(p.index, 0, loss.clip(lower=0), color=AXIS, lw=0.8, zorder=1)
    ax.plot(p.index, p[f"var|{HMM}"] * 100, color=ORANGE, lw=1.6, zorder=3)
    ax.plot(p.index, p[f"var|{GATED}"] * 100, color=BLUE, lw=2, zorder=4)
    hh, hg = p[f"hit|{HMM}"].astype(bool), p[f"hit|{GATED}"].astype(bool)
    only_h = hh & ~hg
    ax.scatter(p.index[only_h], loss[only_h] + 0.35, marker="v", color=ORANGE, s=34, zorder=5)
    ax.scatter(p.index[hg], loss[hg] + 0.35, marker="v", color=BLUE, s=40, zorder=6)
    gfc = (p.index >= "2008-09-01") & (p.index <= "2009-03-31")
    ax.axvspan(pd.Timestamp("2008-09-01"), pd.Timestamp("2009-03-31"), color=GRID, alpha=0.6, zorder=0, lw=0)
    top = max(loss.max(), (p[f"var|{GATED}"] * 100).max()) + 0.8
    ax.text(pd.Timestamp("2008-09-05"), top - 0.3, "GFC window", color=INK2, fontsize=9, va="top", ha="left")
    ax.text(pd.Timestamp("2009-12-31"), top - 0.3, "where only blue shows, the two VaR lines coincide (gate off)",
            color=MUTED, fontsize=8.5, va="top", ha="right")
    ax.set_ylabel("Daily loss / 99% VaR (%)")
    ax.set_ylim(0, top)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    handles = [
        Line2D([], [], color=AXIS, lw=3, label="daily loss"),
        Line2D([], [], color=ORANGE, lw=1.6, label=f"HMM VaR ({int(hh[gfc].sum())} violations in GFC window)"),
        Line2D([], [], color=BLUE, lw=2, label=f"HMM-gated GARCH VaR ({int(hg[gfc].sum())} violations)"),
        Line2D([], [], color=ORANGE, marker="v", ls="", label="violation, HMM only"),
        Line2D([], [], color=BLUE, marker="v", ls="", label="violation, gated model"),
    ]
    ax.legend(handles=handles, loc="upper left")
    ax.set_title("S&P 500 through the GFC: the HMM learned its crisis level in a calm market, the GARCH floor fixes it")
    _save(fig, "fig03_sp500_gfc_var_paths.png")


def _heatmap(ax, mat, rows, cols, fmt, vmin, vmax, cmap=SEQ):
    ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                continue
            frac = (v - vmin) / (vmax - vmin) if vmax > vmin else 0
            ax.text(j, i, fmt(v), ha="center", va="center", fontsize=9.5,
                    color="#ffffff" if frac > 0.55 else INK)
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)


def fig_basel(bt):
    piv = bt.pivot(index="market", columns="model", values="basel_red").reindex(index=MARKET_ORDER, columns=MODEL_ORDER)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    mat = piv.to_numpy(dtype=float)
    _heatmap(ax, mat, [MARKET_NAME[m] for m in MARKET_ORDER], [SHORT[m] for m in MODEL_ORDER],
             lambda v: f"{int(v)}", 0, max(10, np.nanmax(mat)))
    tot = piv.sum()
    for j, m in enumerate(MODEL_ORDER):
        ax.text(j, len(MARKET_ORDER) - 0.35, f"total {int(tot[m])}", ha="center", va="top",
                fontsize=9, color=INK, fontweight="bold" if m == GATED else "normal")
    ax.set_ylim(len(MARKET_ORDER) + 0.1, -0.5)
    ax.set_title("Basel red-zone years (a 250-day block with 10+ violations): gated model has 2, plain HMM 11")
    _save(fig, "fig04_basel_red_years.png")


def fig_skill(sk):
    piv = sk.pivot(index="market", columns="model", values="skill_vs_GBM_pct").reindex(index=MARKET_ORDER, columns=MODEL_ORDER)
    rank = sk.groupby("model")["rank_in_market"].mean().reindex(MODEL_ORDER)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.6, 1]})
    _heatmap(a1, piv.to_numpy(dtype=float), [MARKET_NAME[m] for m in MARKET_ORDER],
             [SHORT[m] for m in MODEL_ORDER], lambda v: f"{v:.0f}%", 0, 32)
    a1.set_title("Forecast skill: % lower quantile (tick) loss than GBM")
    order = rank.sort_values(ascending=False)
    colors = [MODEL_STYLE[m][0] if m in (GATED, HMM, MSG) else AXIS for m in order.index]
    a2.barh([SHORT[m].replace("\n", " ") for m in order.index], order.to_numpy(), color=colors, height=0.6)
    for y, v in enumerate(order.to_numpy()):
        a2.text(v + 0.08, y, f"{v:.2f}", va="center", fontsize=9.5, color=INK)
    a2.set_xlim(0, 6.8)
    a2.set_xlabel("Mean rank across 7 markets (1 = best)")
    a2.grid(axis="y", visible=False)
    a2.set_title("Mean rank by tick loss (lower is better)")
    fig.tight_layout()
    _save(fig, "fig05_forecast_skill.png")


def fig_transitions(tm):
    fitted = tm[tm.source == "fitted"]
    fig, axes = plt.subplots(2, 7, figsize=(17, 5.6))
    for r, model in enumerate(["HMM", "MS-GARCH"]):
        for c, market in enumerate(MARKET_ORDER):
            ax = axes[r, c]
            row = fitted[(fitted.model == model) & (fitted.market == market)].iloc[0]
            P = np.array([[row.p_calm_to_calm, row.p_calm_to_crisis], [row.p_crisis_to_calm, row.p_crisis_to_crisis]])
            _heatmap(ax, P, ["calm", "crisis"] if c == 0 else ["", ""], ["calm", "crisis"],
                     lambda v: f"{v:.3f}", 0, 1)
            ax.set_xlabel(f"stays: calm {row.expected_calm_days:.0f}d,\ncrisis {row.expected_crisis_days:.0f}d",
                          fontsize=8.5, color=INK2)
            if r == 0:
                ax.set_title(MARKET_NAME[market], fontsize=10.5)
        axes[r, 0].set_ylabel(f"{model}\ntoday", fontsize=10, color=INK)
    fig.suptitle("Regime transition matrices, P(tomorrow's regime | today's regime); expected stay = 1 / (1 - P(stay))",
                 x=0.02, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, "fig06_transition_matrices.png")


def _runs(idx, flag):
    out, start = [], None
    for d, f in zip(idx, flag):
        if f and start is None:
            start = d
        elif not f and start is not None:
            out.append((start, d)); start = None
    if start is not None:
        out.append((start, idx[-1]))
    return out


def fig_timeline():
    fig, ax = plt.subplots(figsize=(14, 4.8))
    for name, (s, e) in CRISIS_WINDOWS.items():
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), color=GRID, alpha=0.7, lw=0, zorder=0)
    ys = np.arange(len(MARKET_ORDER))[::-1]
    for y, market in zip(ys, MARKET_ORDER):
        f = pd.read_csv(OUT_DIR / f"{market}_regime_flags.csv", index_col=0, parse_dates=True)
        for s, e in _runs(f.index, f["hmm_crisis"].astype(bool).to_numpy()):
            ax.barh(y, mdates.date2num(e) - mdates.date2num(s) + 1, left=mdates.date2num(s),
                    height=0.62, color=BLUE, lw=0, zorder=2)
        pct = f["hmm_crisis"].mean() * 100
        ax.text(1.005, y, f"{pct:.0f}% of days", transform=ax.get_yaxis_transform(), va="center",
                fontsize=9, color=INK2)
    labels = {"1998 Russia/LTCM": "1998", "2000-02 dot-com bust": "dot-com", "2008 GFC": "GFC",
              "2011 euro/US downgrade": "2011", "2015-16 China slowdown": "China 15", "2020 COVID": "COVID",
              "2022 rate-hike onset": "2022", "1997 Asian crisis": "Asia 97"}
    for name, (s, e) in CRISIS_WINDOWS.items():
        mid = pd.Timestamp(s) + (pd.Timestamp(e) - pd.Timestamp(s)) / 2
        ha = {"1997 Asian crisis": "right", "1998 Russia/LTCM": "left"}.get(name, "center")
        ax.text(mid, len(ys) - 0.25, labels.get(name, name), ha=ha, va="bottom", fontsize=8, color=INK2)
    ax.set_yticks(ys, [MARKET_NAME[m] for m in MARKET_ORDER])
    ax.set_ylim(-0.6, len(ys) + 0.2)
    ax.grid(axis="y", visible=False)
    ax.xaxis_date()
    ax.set_xlim(pd.Timestamp("1990-01-01"), pd.Timestamp("2027-01-01"))
    ax.set_title("HMM crisis regime (blue) across seven markets, 1990-2026; grey bands = documented crisis windows")
    _save(fig, "fig07_regime_timeline.png")


def fig_crisis_share(reg):
    reg = reg.set_index("market").reindex(MARKET_ORDER)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ys = np.arange(len(MARKET_ORDER))[::-1]
    h = 0.26
    series = [("hmm_crisis_pct", "HMM crisis regime", BLUE), ("ms_crisis_pct", "MS-GARCH crisis regime", AQUA),
              ("crash_pct", "crash days (return <= -3%)", ORANGE)]
    for k, (col, label, c) in enumerate(series):
        v = reg[col].to_numpy() * 100
        ax.barh(ys + (1 - k) * h, v, height=h - 0.03, color=c, label=label)
        for y, x in zip(ys + (1 - k) * h, v):
            ax.text(x + 0.3, y, f"{x:.1f}", va="center", fontsize=8, color=INK2)
    for y, v in zip(ys, reg["crash_days_in_hmm_crisis"].to_numpy() * 100):
        ax.text(1.005, y, f"{v:.0f}% of crash days\ninside HMM crisis", transform=ax.get_yaxis_transform(),
                va="center", fontsize=8, color=INK2)
    ax.set_yticks(ys, [MARKET_NAME[m] for m in MARKET_ORDER])
    ax.set_xlabel("% of trading days")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    ax.set_title("How often each market is in crisis: 15-27% of days by the HMM, and it contains 90-100% of crashes")
    _save(fig, "fig08_crisis_days_by_market.png")


def fig_duration(dur):
    b = dur[(dur.kind == "bear") & (dur.model == "HMM") & (dur.gap == 5) & (dur.model_days > 0)]
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    lim = (15, 900)
    xs = np.array(lim)
    ax.fill_between(xs, xs * 0.5, xs * 1.5, color=GRID, alpha=0.7, lw=0, zorder=0)
    ax.plot(xs, xs, color=INK, lw=1, ls="--", zorder=1)
    ax.scatter(b.actual_days, b.model_days, s=36, color=BLUE, edgecolor=SURFACE, lw=1, zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ticks = [20, 50, 100, 200, 500]
    ax.set_xticks(ticks, [str(t) for t in ticks]); ax.set_yticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    notes = {("sp500", "2020-02-19"): "S&P COVID", ("sp500", "2007-10-09"): "S&P GFC",
             ("nikkei225", "1994-06-13"): "Nikkei 1994", ("hangseng", "2010-11-08"): "Hang Seng 2010",
             ("sp500", "2000-03-24"): "S&P dot-com", ("nifty50", "2008-08-11"): "NIFTY 2008"}
    for _, r in b.iterrows():
        key = (r.market, r.crisis[:10])
        if key in notes:
            ax.annotate(notes[key], (r.actual_days, r.model_days), xytext=(6, -3), textcoords="offset points",
                        fontsize=8.5, color=INK2)
    ax.text(380, 380 * 1.06, "equal length", rotation=45, fontsize=8.5, color=INK, ha="right", va="bottom")
    ax.text(18, 18 * 1.5 * 1.02, "within +/-50%", fontsize=8.5, color=INK2, va="bottom")
    ax.set_xlabel("Actual bear-market length, peak to trough (trading days)")
    ax.set_ylabel("HMM crisis-episode length (trading days)")
    ax.set_title("Crisis length: 83 bear markets, median HMM/actual = 1.17")
    _save(fig, "fig09_crisis_duration.png")


def fig_runtime(tim):
    ms = tim.groupby("model")["ms_per_forecast"].median().reindex(MODEL_ORDER).sort_values()
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    colors = [MODEL_STYLE[m][0] if m in (GATED, HMM, MSG) else AXIS for m in ms.index]
    ax.barh([SHORT[m].replace("\n", " ") for m in ms.index], ms.to_numpy(), color=colors, height=0.6)
    ax.set_xscale("log")
    for y, v in enumerate(ms.to_numpy()):
        ax.text(v * 1.12, y, f"{v:.1f} ms", va="center", fontsize=9.5, color=INK)
    ax.set_xlim(0.1, 1500)
    ax.set_xlabel("Milliseconds per daily forecast (median over 7 markets, log scale)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Cost: the gated model matches MS-GARCH's forecast rank at about 1/8 of the run time")
    ax.text(0.99, 0.02, "Gated vs HMM gap is CPU noise (7 markets ran in parallel)", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color=MUTED)
    _save(fig, "fig10_runtime.png")


def main() -> None:
    bt = pd.read_csv(OUT_DIR / "ALL_backtest.csv")
    sk = pd.read_csv(OUT_DIR / "ALL_forecast_skill.csv")
    tim = pd.read_csv(OUT_DIR / "ALL_timings.csv")
    reg = pd.read_csv(OUT_DIR / "ALL_regimes.csv")
    dur = pd.read_csv(OUT_DIR / "VALIDATION_crisis_duration.csv")
    fig_violation_rates(bt)
    fig_dumbbell(bt)
    fig_sp500_gfc()
    fig_basel(bt)
    fig_skill(sk)
    tm_path = OUT_DIR / "ALL_transition_matrices.csv"
    if tm_path.exists():
        fig_transitions(pd.read_csv(tm_path))
    fig_timeline()
    fig_crisis_share(reg)
    fig_duration(dur)
    fig_runtime(tim)


if __name__ == "__main__":
    main()
