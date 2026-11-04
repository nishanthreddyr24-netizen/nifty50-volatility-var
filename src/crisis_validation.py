"""Do the regime models' crisis labels line up with real crisis periods?

Two independent yardsticks, both fixed before looking at the regime output:

1. Bear markets computed from each market's own closing prices with the
   standard 20% rule: a bear market runs from a peak to the subsequent trough
   once the index has fallen 20% from that peak, and ends when the index has
   risen 20% from the trough (the conventional threshold used in S&P 500
   bear-market tables such as Yardeni Research's). This is objective and
   market-specific, so no hand-picked windows.

2. Documented event dates from primary or official sources:
     - Thai baht devaluation, 2 Jul 1997 (Federal Reserve History, "Asian Financial Crisis")
     - Russian default/devaluation, 17 Aug 1998 (Chicago Fed Economic Perspectives 2001 Q1)
     - LTCM recapitalisation convened by the NY Fed, 23 Sep 1998 (FRBNY 1998 Annual Report)
     - WHO characterises COVID-19 as a pandemic, 11 Mar 2020
     - NBER US recessions: Mar-Nov 2001, Dec 2007-Jun 2009, Feb-Apr 2020
       (peak month to trough month; recession runs from the month after the peak)

Regime labels come from reports/multi_market/<market>_regime_flags.csv
(full-sample fits, filtered P(crisis) > 0.5), written by multi_market.py.
They are descriptive, in-sample labels, not forecasts.

Usage:  python -m src.crisis_validation
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.loaders import download_prices
from src.multi_market import MARKETS, OUT_DIR

BEAR_DOWN = 0.20
BEAR_UP = 0.20

EVENTS = {
    "Thai baht devaluation (2 Jul 1997)": "1997-07-02",
    "Russian default (17 Aug 1998)": "1998-08-17",
    "LTCM rescue (23 Sep 1998)": "1998-09-23",
    "WHO pandemic (11 Mar 2020)": "2020-03-11",
}
NBER_RECESSIONS = {  # month after peak .. end of trough month
    "NBER 2001 recession": ("2001-04-01", "2001-11-30"),
    "NBER 2007-09 recession": ("2008-01-01", "2009-06-30"),
    "NBER 2020 recession": ("2020-03-01", "2020-04-30"),
}
EVENT_WINDOW = 60  # trading days either side of an event date


def bear_markets(close: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp, float]]:
    """20%-down / 20%-up turning-point rule. Returns (peak_date, trough_date, depth)."""
    bears = []
    state = "bull"
    peak_v, peak_d = close.iloc[0], close.index[0]
    trough_v, trough_d = peak_v, peak_d
    for d, v in close.items():
        if state == "bull":
            if v > peak_v:
                peak_v, peak_d = v, d
            elif v <= peak_v * (1 - BEAR_DOWN):
                state = "bear"
                trough_v, trough_d = v, d
        else:
            if v < trough_v:
                trough_v, trough_d = v, d
            elif v >= trough_v * (1 + BEAR_UP):
                bears.append((peak_d, trough_d, trough_v / peak_v - 1))
                state = "bull"
                peak_v, peak_d = v, d
    if state == "bear":  # still in a bear at the end of the sample
        bears.append((peak_d, trough_d, trough_v / peak_v - 1))
    return bears


def _close(market: str, index: pd.DatetimeIndex) -> pd.Series:
    ticker, start, cache = MARKETS[market]
    c = download_prices(cache, ticker, start=start)["Close"].dropna()
    c = c[~c.index.duplicated(keep="first")].sort_index()
    return c.reindex(index).ffill()


def first_flag_lag(flag: pd.Series, anchor: pd.Timestamp, window: int = EVENT_WINDOW) -> float:
    """Onset of the crisis spell relative to `anchor`, in trading days.

    If the regime is already 'crisis' on the anchor date, returns minus the number
    of days since that crisis spell began (<= 0). Otherwise returns the number of
    days until the regime next switches to crisis (> 0), up to `window` days
    ahead. NaN = no crisis flag within `window` days after the anchor."""
    f = flag.to_numpy()
    pos = int(flag.index.searchsorted(anchor))
    if pos >= len(f):
        return np.nan
    if f[pos]:
        start = pos
        while start > 0 and f[start - 1]:
            start -= 1
        return float(start - pos)
    ahead = np.flatnonzero(f[pos:min(pos + window + 1, len(f))])
    return float(ahead[0]) if len(ahead) else np.nan


def main() -> None:
    bear_rows, overlap_rows, event_rows = [], [], []
    for market in MARKETS:
        f = OUT_DIR / f"{market}_regime_flags.csv"
        if not f.exists():
            print(f"skipping {market}: no regime flags yet")
            continue
        flags = pd.read_csv(f, index_col=0, parse_dates=True)
        close = _close(market, flags.index)
        in_bear = pd.Series(False, index=flags.index)

        for peak, trough, depth in bear_markets(close):
            mask = (flags.index >= peak) & (flags.index <= trough)
            n = int(mask.sum())
            if n < 20:
                continue
            in_bear |= mask
            row = {"market": market, "peak": peak.date(), "trough": trough.date(), "depth": depth, "days": n}
            for model, col in [("HMM", "hmm_crisis"), ("MS-GARCH", "ms_crisis")]:
                flag = flags[col].astype(bool)
                row[f"{model}_flagged_pct"] = flag[mask].mean()
                row[f"{model}_onset_lag"] = first_flag_lag(flag, peak, window=n)
            bear_rows.append(row)

        for model, col in [("HMM", "hmm_crisis"), ("MS-GARCH", "ms_crisis")]:
            flag = flags[col].astype(bool)
            tp = int((flag & in_bear).sum())
            overlap_rows.append({
                "market": market, "model": model,
                "bear_day_share": in_bear.mean(),
                "crisis_day_share": flag.mean(),
                "recall": tp / max(int(in_bear.sum()), 1),       # bear days flagged crisis
                "precision": tp / max(int(flag.sum()), 1),       # crisis days inside a bear
                "corr_pcrisis_bear": float(np.corrcoef(flags["hmm_p_crisis"], in_bear.astype(float))[0, 1])
                if model == "HMM" else np.nan,
            })

        for name, date in EVENTS.items():
            d = pd.Timestamp(date)
            if d < flags.index[0] or d > flags.index[-1]:
                continue
            event_rows.append({
                "market": market, "event": name,
                "HMM_lag_days": first_flag_lag(flags["hmm_crisis"].astype(bool), d),
                "MS-GARCH_lag_days": first_flag_lag(flags["ms_crisis"].astype(bool), d),
            })
        if market == "sp500":
            for name, (s, e) in NBER_RECESSIONS.items():
                mask = (flags.index >= pd.Timestamp(s)) & (flags.index <= pd.Timestamp(e))
                event_rows.append({
                    "market": market, "event": name,
                    "HMM_flagged_pct": flags["hmm_crisis"][mask].mean(),
                    "MS-GARCH_flagged_pct": flags["ms_crisis"][mask].mean(),
                })

    bears = pd.DataFrame(bear_rows)
    overlap = pd.DataFrame(overlap_rows)
    events = pd.DataFrame(event_rows)
    bears.to_csv(OUT_DIR / "VALIDATION_bear_markets.csv", index=False)
    overlap.to_csv(OUT_DIR / "VALIDATION_overlap.csv", index=False)
    events.to_csv(OUT_DIR / "VALIDATION_events.csv", index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    fmt = lambda v: f"{v:.2f}"
    print("=== Bear markets (20% rule) and how much each regime model flagged ===")
    print(bears.to_string(index=False, float_format=fmt))
    print("\n=== Overall overlap of crisis regime with bear-market days ===")
    print(overlap.to_string(index=False, float_format=fmt))
    print("\n=== Documented events: trading days from event to first crisis flag (+/-60d), NBER: share flagged ===")
    print(events.to_string(index=False, float_format=fmt))


if __name__ == "__main__":
    main()
