# Volatility Regimes and VaR Backtesting on NIFTY 50

Applied Stochastic Modelling course project. Story: returns are unpredictable,
but risk is predictable, so we model risk — using regime-switching (HMM) and
GARCH-family models against a GBM baseline, scored on out-of-sample VaR
backtests (Kupiec, Christoffersen, Basel traffic light).

## Setup

```
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

(Python 3.11 is used deliberately — `arch`/`hmmlearn` wheel support lags on
newer Python versions.)

## Layout

- `data/raw/` — cached CSV downloads (yfinance), `data/processed/` — derived data
- `src/loaders.py` — download/cache NIFTY 50, S&P 500, India VIX
- `src/returns.py` — log returns, annualization, rolling vol, summary stats
- `src/gbm.py` — GBM fit (MLE), exact-discretization simulation, parametric normal VaR/ES
- `src/var_backtest.py` — model-agnostic rolling VaR backtest harness: Kupiec POF test,
  Christoffersen independence test, Basel traffic light. Any model plugs in via a
  `forecast_fn(history) -> (mu_daily, sigma_daily)` callback that only ever sees
  data strictly before the forecast date (no look-ahead by construction).
- `notebooks/01_stylized_facts.py` — fat tails, vol clustering (ACF of r vs r^2),
  leverage effect, QQ plot, India VIX cross-check, 2008/2020 crisis windows.
  Jupytext "light" script format — plain, diffable, runnable as-is, or convert with
  `jupytext --to notebook 01_stylized_facts.py`.
- `reports/figures/` — output figures
- `tests/` — parameter-recovery and backtest-harness sanity tests

## Status

Done: data loading, log returns, stylized-facts analysis, GBM baseline
(fit/simulate/parametric VaR), rolling VaR backtest harness with Kupiec/
Christoffersen/traffic-light, sanity tests.

Already-visible finding: GBM with an expanding-window volatility estimate is
*too conservative* out-of-sample (0.44% observed 1-day-99% VaR violations vs
1% expected on NIFTY 50, 2008–2026) — the expanding window blends calm and
crisis volatility and lags the current regime. This is the setup for the
regime-switching (HMM) and GARCH models to show improvement.

Next: `src/hmm.py` (regime-switching, EM/MLE fit, validate against India VIX
and 2008/2020), `src/garch.py` (GARCH(1,1) and GJR-GARCH via `arch`),
`src/extremes.py` (crash-day Poisson/Hawkes), historical-simulation VaR
benchmark, simulation study for parameter recovery under the regime model.

## Reproducing

```
python src/loaders.py          # download/cache data
python notebooks/01_stylized_facts.py   # stylized facts -> reports/figures/
python src/gbm.py               # GBM fit + simulate + VaR
python src/var_backtest.py      # GBM rolling VaR backtest + tests
pytest tests/ -q                # sanity tests
```
