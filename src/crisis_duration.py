"""How well do the regime models get the LENGTH of a crisis?

Reference crisis = bear market by the 20%-down / 20%-up rule (peak -> trough),
computed from each market's own prices (see crisis_validation.bear_markets),
plus the NBER US recessions for the S&P 500.

Model crisis episode. Daily crisis flags (filtered P(crisis) > 0.5) flicker, so
spells separated by at most GAP non-crisis trading days are merged into one
episode (GAP = 5, one trading week; GAP = 21, one month, as a sensitivity check).
For a bear market, the model's episode runs from the first crisis flag inside
[peak, trough] to the end of the merged episode containing the last flag inside
[peak, trough], so it may end after the trough.

Reported per bear: actual length, model length, start lag (first flag - peak),
end gap (model end - trough; > 0 = still in crisis after the bottom), and the
share of bear days flagged.

Labels come from full-sample fits (descriptive, not real-time forecasts).

Usage:  python -m src.crisis_duration
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.crisis_validation import NBER_RECESSIONS, _close, bear_markets
from src.multi_market import MARKETS, OUT_DIR

GAPS = (5, 21)
MIN_BEAR_DAYS = 20
MODELS = {"HMM": "hmm_crisis", "MS-GARCH": "ms_crisis"}


def merged_episodes(flag: np.ndarray, gap: int) -> list[tuple[int, int]]:
    """(start, end) index pairs, inclusive, of crisis runs merged across gaps <= gap."""
    idx = np.flatnonzero(flag)
    if len(idx) == 0:
        return []
    eps, s, e = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - e - 1 <= gap:
            e = i
        else:
            eps.append((s, e)); s = e = i
    eps.append((s, e))
    return eps


def episode_for_window(flag: np.ndarray, eps, lo: int, hi: int):
    """Model episode for a reference window [lo, hi] (positions). None if never flagged."""
    inside = np.flatnonzero(flag[lo:hi + 1])
    if len(inside) == 0:
        return None
    start = lo + inside[0]
    last = lo + inside[-1]
    end = next(e for s, e in eps if s <= last <= e)
    return start, end


def main() -> None:
    rows = []
    for market in MARKETS:
        f = OUT_DIR / f"{market}_regime_flags.csv"
        if not f.exists():
            continue
        flags = pd.read_csv(f, index_col=0, parse_dates=True)
        idx = flags.index
        close = _close(market, idx)

        refs = []
        for peak, trough, depth in bear_markets(close):
            lo, hi = int(idx.searchsorted(peak)), int(idx.searchsorted(trough))
            if hi - lo + 1 >= MIN_BEAR_DAYS:
                refs.append(("bear", f"{peak.date()} -> {trough.date()}", lo, hi, depth))
        if market == "sp500":
            for name, (s, e) in NBER_RECESSIONS.items():
                lo, hi = int(idx.searchsorted(pd.Timestamp(s))), int(idx.searchsorted(pd.Timestamp(e), "right")) - 1
                refs.append(("NBER", name, lo, hi, np.nan))

        for model, col in MODELS.items():
            flag = flags[col].astype(bool).to_numpy()
            for gap in GAPS:
                eps = merged_episodes(flag, gap)
                for kind, label, lo, hi, depth in refs:
                    ep = episode_for_window(flag, eps, lo, hi)
                    actual = hi - lo + 1
                    row = {"market": market, "model": model, "gap": gap, "kind": kind, "crisis": label,
                           "depth": depth, "actual_days": actual,
                           "flagged_share": flag[lo:hi + 1].mean()}
                    if ep is None:
                        row.update(model_days=0, start_lag=np.nan, end_gap=np.nan,
                                   model_start=pd.NaT, model_end=pd.NaT)
                    else:
                        s, e = ep
                        row.update(model_days=e - s + 1, start_lag=s - lo, end_gap=e - hi,
                                   model_start=idx[s].date(), model_end=idx[e].date())
                    rows.append(row)

    d = pd.DataFrame(rows)
    d["ratio"] = d["model_days"] / d["actual_days"]
    d.to_csv(OUT_DIR / "VALIDATION_crisis_duration.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    bears = d[d["kind"] == "bear"]

    def summary(g):
        found = g[g["model_days"] > 0]
        rho = stats.spearmanr(g["actual_days"], g["model_days"]).statistic if len(g) > 2 else np.nan
        return pd.Series({
            "bears": len(g),
            "detected": len(found),
            "median_actual": g["actual_days"].median(),
            "median_model": found["model_days"].median(),
            "median_ratio": found["ratio"].median(),
            "within_50pct": ((found["ratio"] >= 0.5) & (found["ratio"] <= 1.5)).sum() / len(g),
            "median_start_lag": found["start_lag"].median(),
            "median_end_gap": found["end_gap"].median(),
            "spearman_len": rho,
        })

    for gap in GAPS:
        b = bears[bears["gap"] == gap]
        print(f"\n=== Crisis length vs bear markets, spells merged across gaps <= {gap} trading days ===")
        print("By model, all markets pooled:")
        print(b.groupby("model").apply(summary).to_string(float_format=lambda v: f"{v:.2f}"))
        print("HMM by market:")
        print(b[b["model"] == "HMM"].groupby("market").apply(summary).to_string(float_format=lambda v: f"{v:.2f}"))

    b5 = bears[(bears["gap"] == 5) & (bears["model"] == "HMM")]
    cols = ["market", "crisis", "depth", "actual_days", "model_start", "model_end", "model_days",
            "start_lag", "end_gap", "flagged_share"]
    big = b5[b5["depth"] <= -0.30].sort_values(["crisis"])
    print("\n=== HMM episode vs every bear market deeper than 30% (gap 5) ===")
    print(big[cols].to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    nber = d[d["kind"] == "NBER"]
    print("\n=== S&P 500 vs NBER recessions ===")
    print(nber[["model", "gap", "crisis", "actual_days", "model_start", "model_end", "model_days",
                "start_lag", "end_gap", "flagged_share"]].to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    print("\n=== Length of all merged HMM crisis episodes (gap 5), per market ===")
    out = []
    for market in MARKETS:
        f = OUT_DIR / f"{market}_regime_flags.csv"
        if not f.exists():
            continue
        flag = pd.read_csv(f, index_col=0)["hmm_crisis"].astype(bool).to_numpy()
        L = np.array([e - s + 1 for s, e in merged_episodes(flag, 5)])
        out.append({"market": market, "episodes": len(L), "median_days": np.median(L),
                    "mean_days": L.mean(), "longest": L.max(), "episodes_ge_60d": int((L >= 60).sum())})
    print(pd.DataFrame(out).to_string(index=False, float_format=lambda v: f"{v:.1f}"))


if __name__ == "__main__":
    main()
