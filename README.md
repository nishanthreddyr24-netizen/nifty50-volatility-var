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

All computational pieces are done and runnable. Scope for the actual
write-up/presentation (deliberately kept tighter than everything that's been
built — see note below):

**Core story:** stylized facts → GBM baseline → HMM regime model →
GARCH(1,1) (plain, normal innovations — the syllabus version) → VaR backtest
comparison → extreme-event (Poisson/Hawkes) analysis.

**Supplementary / appendix only, not part of the main narrative, and not
presented:** historical simulation VaR benchmark, HMM parameter-recovery
simulation study. (GJR-GARCH and Student-t innovations were explored earlier
but removed from the codebase entirely — beyond what the course covers, and
not worth keeping around once they weren't going to be presented.)

### Headline results (NIFTY 50, out-of-sample 1-day 99% VaR, 2008–2026)

| Model                   | Violation rate | Kupiec p | Christoffersen p | COVID window (hits/days) |
|--------------------------|:---:|:---:|:---:|:---:|
| GBM (expanding window)   | 0.44% | 0.00 (fails, too conservative) | 0.69 | 10/89 |
| GARCH(1,1), normal       | 1.58% | 0.00 (fails, too aggressive) | 0.98 | 6/89 |
| **HMM 2-state**          | **0.92%** | **0.62 (passes)** | 0.40 | **4/89** |

Expected rate under correct calibration is 1.00%. GBM is too conservative
(its expanding-window volatility estimate lags the current regime); plain
GARCH(1,1) is too aggressive; the HMM is the only one of the three that
passes both the Kupiec and Christoffersen tests, and it also handles the
COVID stress window best. Reproduce this table with `python -m
src.model_comparison` (the full output also includes a historical-simulation
row kept for reference — just not part of the story).

Other findings: fat tails (excess kurtosis ≈ 16, Jarque-Bera rejects
normality), strong volatility clustering (ACF of squared returns, Ljung-Box
p≈0), 21-day realized vol correlates 0.87 with India VIX, the HMM's
highest-vol regime captures 100% of the 2008 GFC and 68.5% of the 2020 COVID
window, crash-day counts are badly overdispersed vs. Poisson (variance/mean
= 17.6) and a Hawkes process is strongly preferred (LR p < 1e-16, branching
ratio 0.76), and HMM parameter recovery in a 50-replication simulation study
shows bias well within one standard error on every parameter.

Caveat worth keeping in the write-up: a 3-state HMM has two regimes (0, 1)
that almost deterministically alternate rather than persist — that looks
like it's fitting short-term mean-reversion rather than a true slow-moving
volatility regime, so the 2-state model is the one actually used and trusted
here. Also, the 2008 GFC itself can't be backtested out-of-sample (it falls
inside every model's first training window, since the sample starts exactly
2008-01-01 with no earlier data to train on) — this is a genuine data
limitation, not an oversight.

Not yet done: report write-up itself (`reports/`); everything the course
project needs computationally is in place.

## Reproducing

```
python src/loaders.py                   # download/cache data
python notebooks/01_stylized_facts.py   # stylized facts -> reports/figures/
python -m src.gbm                       # GBM fit + simulate + VaR
python -m src.hmm                       # HMM fit + validation + backtest
python -m src.garch                     # GARCH(1,1) fit + backtest
python -m src.extremes                  # Poisson/Hawkes crash-day analysis
python -m src.simulation_study          # HMM parameter-recovery check
python -m src.model_comparison          # full side-by-side comparison table
pytest tests/ -q                        # sanity tests
```
