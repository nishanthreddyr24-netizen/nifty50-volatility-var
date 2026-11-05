# Final results: every number from the project, no rerun needed

Applied Stochastic Modelling course project: **volatility regimes and Value-at-Risk (VaR)
forecasting**, first on NIFTY 50, then on seven equity indices over 1990-2026, then used in two
real trading strategies.

**One-line answer.** Daily returns cannot be predicted, but their risk can: every market switches
between a persistent calm regime and a crisis regime 2.4-3.2x as volatile. A model that uses
those regimes (HMM-gated GARCH) gives the most robust 99% VaR across seven markets. No model with
normal errors gets exactly 1% coverage over 30+ years. Used for trading, the forecasts cut
crisis losses by 54-70%, but they do not significantly raise risk-adjusted returns.

Everything in this folder was copied from the scripts' own output files by
`python -m src.collect_results`. Nothing here needs to be rerun. `MANIFEST.csv` lists every
file with its size, original path and a one-line description.

### Use the data straight away

All 88 CSVs (11 input price series and 77 result tables) can be loaded straight from GitHub, with no
clone and no rerun. **[`CSV_INDEX.md`](CSV_INDEX.md)** gives a direct-download link and a row count
for every one.

```python
import pandas as pd
BASE = "https://raw.githubusercontent.com/nishanthreddyr24-netizen/nifty50-volatility-var/master/final_results/"

backtest = pd.read_csv(BASE + "05_seven_market_var_backtest/ALL_backtest.csv")            # every market x model
var_paths = pd.read_csv(BASE + "05_seven_market_var_backtest/sp500_var_paths.csv",
                        index_col=0, parse_dates=True)                                   # daily VaR and hits
regimes = pd.read_csv(BASE + "07_regimes_and_transitions/sp500_regime_flags.csv",
                      index_col=0, parse_dates=True)                                     # daily P(crisis)
prices = pd.read_csv(BASE + "00_input_data/sp500_1990.csv", index_col=0, parse_dates=True)  # raw input
```

Most-used files:

| What | File |
|---|---|
| Every backtest result (violations, Kupiec, Christoffersen, Basel, sub-periods) | [`05_seven_market_var_backtest/ALL_backtest.csv`](05_seven_market_var_backtest/ALL_backtest.csv) |
| Daily VaR forecast and hit for every model | `05_seven_market_var_backtest/<market>_var_paths.csv` |
| Forecast skill and Diebold-Mariano | [`06_forecast_skill_and_runtime/ALL_forecast_skill.csv`](06_forecast_skill_and_runtime/ALL_forecast_skill.csv) |
| Daily crisis probability and regime flags | `07_regimes_and_transitions/<market>_regime_flags.csv` |
| Transition matrices | [`07_regimes_and_transitions/ALL_transition_matrices.csv`](07_regimes_and_transitions/ALL_transition_matrices.csv) |
| Chapman-Kolmogorov summary | [`08_chapman_kolmogorov/CK_summary.csv`](08_chapman_kolmogorov/CK_summary.csv) |
| Every options trade | [`10_trading_applications/TRADE_vrp_trades.csv`](10_trading_applications/TRADE_vrp_trades.csv) |
| Volatility-targeting results | [`10_trading_applications/TRADE_voltarget_summary.csv`](10_trading_applications/TRADE_voltarget_summary.csv) |

`<market>` is one of `sp500`, `ftse100`, `dax`, `nikkei225`, `hangseng`, `sensex`, `nifty50`.

---

## Contents

| Folder | Study | Key files |
|---|---|---|
| [`00_input_data/`](00_input_data/) | Daily prices of all 7 indices, VIX, India VIX, and NSE NIFTY 1990-2026 | 11 CSVs |
| [`01_stylized_facts/`](01_stylized_facts/) | Fat tails, volatility clustering, India VIX check | `stylized_facts_output.txt`, 5 figures |
| [`02_nifty_model_comparison/`](02_nifty_model_comparison/) | Six VaR models on NIFTY 50 | `model_comparison_output.txt`, `nifty50_var_paths.csv` |
| [`03_extreme_events_poisson_hawkes/`](03_extreme_events_poisson_hawkes/) | Are crash days Poisson? Hawkes fit | `extremes_output.txt` |
| [`04_hmm_simulation_study/`](04_hmm_simulation_study/) | Does EM recover the HMM's parameters? | `simulation_study_output.txt` |
| [`05_seven_market_var_backtest/`](05_seven_market_var_backtest/) | Six models x seven markets, about 54,000 forecasts each | `ALL_backtest.csv`, `<market>_var_paths.csv` |
| [`06_forecast_skill_and_runtime/`](06_forecast_skill_and_runtime/) | Tick loss, Diebold-Mariano, run time | `ALL_forecast_skill.csv`, `ALL_timings.csv` |
| [`07_regimes_and_transitions/`](07_regimes_and_transitions/) | Calm/crisis regimes, transition matrices | `ALL_regimes.csv`, `ALL_transition_matrices.csv` |
| [`08_chapman_kolmogorov/`](08_chapman_kolmogorov/) | n-step Markov check, 40 horizons x 7 markets | `CK_summary.csv` |
| [`09_crisis_timing_and_length/`](09_crisis_timing_and_length/) | Crisis labels vs bear markets, NBER, dated events | `VALIDATION_*.csv` |
| [`10_trading_applications/`](10_trading_applications/) | Options (selling volatility) and volatility targeting | `TRADE_*.csv` |
| [`11_nse_data_experiments/`](11_nse_data_experiments/) | Official NSE NIFTY data back to 1990; data audits | `exp1/`, `exp2/`, `audits/` |
| [`12_run_logs/`](12_run_logs/) | Console output and logs of every run, unit tests | `*_log.txt`, `pytest_output.txt` |

The [master scorecard](#master-scorecard) at the end lists all 45 studies with a pass/fail verdict.

---

## Data and conventions

| Market | Ticker | Data from | Backtest from (after 500-day warm-up) | Forecast days |
|---|---|---|---|---|
| S&P 500 | ^GSPC | 1990 | 1991-12 | 8,777 |
| FTSE 100 | ^FTSE | 1990 | 1991-12 | 8,806 |
| DAX | ^GDAXI | 1990 | 1992-01 | 8,822 |
| Nikkei 225 | ^N225 | 1990 | 1992-01 | 8,537 |
| Hang Seng | ^HSI | 1990 | 1992-01 | 8,591 |
| Sensex | ^BSESN | 1997-07 | 1999-07 | 6,728 |
| NIFTY 50 | ^NSEI | 2008 | 2010-01 | 4,113 |

- Source: Yahoo Finance daily OHLCV, saved in `00_input_data/` exactly as downloaded (Close is
  used). NSE NIFTY series: `00_input_data/nifty50_nse_1990_2026.csv`.
  Implied volatility: CBOE VIX (^VIX, from 1990) and India VIX (^INDIAVIX, from 2008).
- Returns are daily log returns. **VaR** is the one-day 99% VaR, a positive loss fraction:
  the loss that should be exceeded on 1% of days. A **violation** (hit) is a day with return < -VaR.
- **Out of sample:** every forecast for day t uses only returns before day t. Refit frequency:
  GARCH every 21 days, HMM and MS-GARCH every 63 days, GBM and historical simulation daily.
- A test **passes** at p >= 0.05 unless stated otherwise.
- **Crash day:** daily log return <= -3%.
- **Crisis windows** used for sub-periods: GFC = 2008-09-01 to 2009-03-31, COVID = 2020-02-15 to
  2020-06-30, pre-2008 = before 2008-09-01, post-GFC = from 2009-04-01.

### The six models

| Model | What it is | Code |
|---|---|---|
| GBM | Normal returns, constant volatility estimated on an expanding window | `src/gbm.py` |
| GARCH(1,1) | Normal errors; volatility reacts to yesterday's shock and yesterday's volatility | `src/garch.py` |
| HMM 2-state | Gaussian hidden Markov model; calm and crisis regimes; forward filter gives P(crisis) | `src/hmm.py` |
| MS-GARCH 2-regime | Haas et al. regime-switching GARCH, Hamilton filter | `src/msgarch.py` |
| HMM-gated GARCH | In the crisis regime, volatility = max(HMM crisis volatility, GARCH volatility) | `src/hmm_garch.py` |
| Historical simulation | Empirical 1% quantile of the last 500 days | `src/var_backtest.py` |

### The tests

| Test | Question it answers |
|---|---|
| Kupiec (POF) | Is the violation rate 1%? Likelihood-ratio test on the count. |
| Christoffersen | Are violations independent? It fits a 2-state Markov chain to the hit sequence and LR-tests it against independent hits. |
| Basel traffic light | Per 250-day year: green 0-4 violations, yellow 5-9, red 10 or more (red raises a bank's capital multiplier from 3 to 4). |
| Tick (quantile) loss | Proper scoring rule for a quantile forecast; lower is better. |
| Diebold-Mariano | Is one model's tick loss significantly lower than another's? |
| Chapman-Kolmogorov | Does the one-step transition matrix raised to the n-th power reproduce the observed n-step behaviour? |

---

## 01. Stylized facts (NIFTY 50, 2008-2026)

Source: `01_stylized_facts/stylized_facts_output.txt` (from `notebooks/01_stylized_facts.py`).

| Statistic | Value |
|---|---|
| Observations | 4,613 daily log returns, 2008-01-01 to 2026-10-23 |
| Mean daily / annualised | 0.0282% / 7.11% |
| Volatility daily / annualised | 1.279% / 20.30% |
| Skewness | -0.317 |
| Excess kurtosis | 15.92 (normal = 0) |
| Worst / best day | -13.90% / +16.33% |
| Jarque-Bera | 48,800, p = 0 |
| Ljung-Box, returns, lags 10 / 20 | p = 2.3e-5 / 1.0e-7 |
| Ljung-Box, squared returns, lags 10 / 20 | p = 0 / 0 |
| Correlation, 21-day realised vol vs India VIX | 0.872 |
| Correlation, return today vs volatility change tomorrow | -0.014 |

| Check | Verdict |
|---|---|
| Returns are normal | **Rejected** (kurtosis 15.9, JB p = 0) |
| Volatility clusters | **Confirmed** (squared returns strongly autocorrelated) |
| Realised volatility tracks implied volatility (India VIX) | **Pass** (0.87) |
| Leverage effect (falls raise volatility) | **Not found** in this measure (-0.014, essentially zero) |
| Returns are serially uncorrelated | **Rejected**, but the autocorrelation is small; it is dwarfed by the clustering of squared returns |

Conclusion: a constant-volatility random walk (GBM) is the wrong model for risk. Volatility has to
be modelled as a time-varying (stochastic) quantity.

---

## 02. NIFTY 50 single-market comparison (2010-2026, 4,113 forecasts)

Source: `02_nifty_model_comparison/model_comparison_output.txt` (from `src.model_comparison`) and
`nifty50_backtest.csv`. Expected violation rate 1.00%.

| Model | Violations | Rate | Kupiec p | Christoffersen p | Basel red years | Verdict |
|---|---|---|---|---|---|---|
| GBM (expanding window) | 18 | 0.44% | 0.00005 | 0.69 | 0 | Fail: too conservative |
| GARCH(1,1), normal | 65 | 1.58% | 0.0006 | 0.98 | 0 | Fail: too narrow |
| HMM 2-state | 38 | 0.92% | 0.62 | 0.40 | 0 | **Pass both** |
| MS-GARCH 2-regime | 42 | 1.02% | 0.89 | 0.35 | 0 | **Pass both** |
| HMM-gated GARCH | 38 | 0.92% | 0.62 | 0.40 | 0 | **Pass both** |
| Historical simulation | 43 | 1.05% | 0.77 | 0.0009 | 1 | Fail: violations cluster |

Basel zones and the COVID crash (2020-02-15 to 2020-06-30, 89 trading days), from
`model_comparison_output.txt`:

| Model | Green / yellow / red years | COVID violations (expected about 1) |
|---|---|---|
| GBM (expanding window) | 15 / 1 / 0 | 10 of 89 |
| GARCH(1,1), normal | 10 / 6 / 0 | 6 of 89 |
| HMM 2-state | 15 / 1 / 0 | **4 of 89** |
| MS-GARCH 2-regime | 12 / 4 / 0 | 6 of 89 |
| HMM-gated GARCH | 15 / 1 / 0 | **4 of 89** |
| Historical simulation | 11 / 4 / 1 | 11 of 89 |

Full-sample HMM on NIFTY (`hmm_regime_output.txt`, from `python -m src.hmm`):

| Check | Result | Verdict |
|---|---|---|
| Regimes (2-state) | Calm: +14.1% a year, 13.4% vol; crisis: -26.9% a year, 40.0% vol | Two clear regimes |
| 2 vs 3 states, chosen by BIC (lower is better) | BIC -28,989 with 2 states (7 parameters) vs -28,953 with 3 (14 parameters); log-likelihood 14,524.0 vs 14,535.4 | **BIC prefers 2 states** |
| 3-state transition matrix | P(0->1) = 0.635, P(1->0) = 0.996: states 0 and 1 swap almost every day; mean India VIX in both is 16.7 | 3rd state is not a real regime; **2 states kept** |
| Crisis regime vs India VIX | Mean India VIX 34.0 in the crisis state vs 16.7 in the others | Pass |
| Crisis regime covers 2008 GFC | 100% of days | Pass |
| Crisis regime covers 2020 COVID | 68.5% of days | Partial |
| Moment-matched normal VaR vs exact mixture quantile | 4 Oct 2022: 2.85% vs 3.99%; 17 Oct 2025: 2.12% vs 2.22% | The normal approximation used in the backtest understates VaR when the regime is uncertain |

- The 2008 GFC cannot be backtested on NIFTY: Yahoo data starts in 2008, so it sits inside every
  model's first 500-day training window (see 11 for the NSE series back to 1990).
- HMM and HMM-gated GARCH are identical on NIFTY (38 = 38 violations): the gate only fired during COVID.

---

## 03. Extreme events: are crash days Poisson? (NIFTY 50)

Source: `03_extreme_events_poisson_hawkes/extremes_output.txt` (from `src.extremes`).
This study was moved to the appendix of the write-up on 25 Oct 2026 to keep the main story on VaR.

Crash days (log return <= -3%): **78 of 4,613 days (1.69%)**.

| Year | 2008 | 2009 | 2010 | 2011 | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Crash days | 35 | 15 | 1 | 2 | 0 | 3 | 0 | 2 | 1 | 0 | 0 | 0 | 12 | 2 | 2 | 0 | 1 | 1 | 1 |

| Test | Result | Verdict |
|---|---|---|
| Poisson equidispersion (yearly counts) | Mean 4.11, variance 72.43, dispersion index 17.64; D = 317.6 on 18 df, p = 0 | **Poisson rejected** |
| Hawkes self-exciting process (exponential kernel) | Background rate mu = 0.00271/day (0.7 per year); jump alpha = 0.0288; decay beta = 0.0379 (half-life 18.3 days); branching ratio 0.761 | Strong clustering |
| Hawkes vs Poisson, likelihood ratio | Log-likelihood -347.14 vs -427.29; LR = 160.3, p = 0 | **Hawkes preferred** |

Interpretation: each crash triggers on average 0.76 further crashes, and the extra risk halves
every 18 trading days.

---

## 04. HMM parameter-recovery simulation study

Source: `04_hmm_simulation_study/simulation_study_output.txt` (from `src.simulation_study`).
The 2-state HMM fitted to NIFTY 50 is used as the "true" model: 50 synthetic samples of 4,613
days are simulated from it and refitted by EM.

| Parameter | True | Mean estimate +/- std across 50 fits | Bias | Bias / std |
|---|---|---|---|---|
| Calm mean (daily) | +0.000559 (+14.1% a year) | +0.000577 +/- 0.000149 | +0.000018 | 0.12 |
| Calm volatility (daily) | 0.008441 (13.4% a year) | 0.008610 +/- 0.000104 | +0.000168 (+2.0%) | 1.62 |
| Crisis mean (daily) | -0.001065 (-26.9% a year) | -0.000920 +/- 0.000864 | +0.000145 | 0.17 |
| Crisis volatility (daily) | 0.025218 (40.0% a year) | 0.025598 +/- 0.000684 | +0.000380 (+1.5%) | 0.56 |

- 50 of 50 fits converged.
- Verdict: **pass, with one caveat**. Three of four parameters are recovered within one
  across-simulation standard deviation. Calm-regime volatility is overestimated by 2.0%
  (1.6 standard deviations): a small, systematic bias that does not change any conclusion.

---

## 05. Seven-market VaR backtest

Source: `05_seven_market_var_backtest/ALL_backtest.csv`; daily forecasts in `<market>_var_paths.csv`.

### Violation rate (%), expected 1.00

| Market | GBM | GARCH | HMM | MS-GARCH | HMM-gated GARCH | Hist. sim |
|---|---|---|---|---|---|---|
| S&P 500 | 2.60 | 2.22 | 1.93 | 2.19 | **1.62** | 1.52 |
| FTSE 100 | 2.26 | 1.87 | 1.73 | 1.83 | **1.52** | 1.45 |
| DAX | 2.01 | 1.72 | 1.82 | 1.69 | **1.52** | 1.55 |
| Nikkei 225 | 1.30 | 1.59 | 1.41 | 1.60 | **1.25** | 1.36 |
| Hang Seng | 1.78 | 1.72 | 1.52 | 1.62 | **1.47** | 1.51 |
| Sensex | 1.11 | 1.46 | 1.10 | 1.29 | **1.01** | 1.25 |
| NIFTY 50 | 0.44 | 1.58 | 0.92 | 1.02 | 0.92 | 1.05 |

### Violations, Kupiec p and Christoffersen p

| Market | Model | Violations | Kupiec p | Christoffersen p | Passes both |
|---|---|---|---|---|---|
| S&P 500 | GBM | 228 | 0 | 2.3e-6 | No |
| S&P 500 | GARCH(1,1) | 195 | 0 | 0.0057 | No |
| S&P 500 | HMM 2-state | 169 | 1.1e-14 | 0.0072 | No |
| S&P 500 | MS-GARCH | 192 | 0 | 0.20 | No |
| S&P 500 | HMM-gated GARCH | 142 | 9.3e-8 | 0.12 | No |
| S&P 500 | Historical simulation | 133 | 6.5e-6 | 7.7e-7 | No |
| FTSE 100 | GBM | 199 | 0 | 6.6e-10 | No |
| FTSE 100 | GARCH(1,1) | 165 | 2.0e-13 | 0.051 | No |
| FTSE 100 | HMM 2-state | 152 | 5.4e-10 | 0.18 | No |
| FTSE 100 | MS-GARCH | 161 | 2.5e-12 | 0.55 | No |
| FTSE 100 | HMM-gated GARCH | 134 | 4.9e-6 | 0.52 | No |
| FTSE 100 | Historical simulation | 128 | 6.2e-5 | 3.8e-10 | No |
| DAX | GBM | 177 | 1.1e-16 | 2.3e-9 | No |
| DAX | GARCH(1,1) | 152 | 6.1e-10 | 0.42 | No |
| DAX | HMM 2-state | 161 | 2.9e-12 | 0.55 | No |
| DAX | MS-GARCH | 149 | 3.2e-9 | 0.27 | No |
| DAX | HMM-gated GARCH | 134 | 5.4e-6 | 0.98 | No |
| DAX | Historical simulation | 137 | 1.4e-6 | 1.9e-7 | No |
| Nikkei 225 | GBM | 111 | 0.0077 | 0.0038 | No |
| Nikkei 225 | GARCH(1,1) | 136 | 4.0e-7 | 0.26 | No |
| Nikkei 225 | HMM 2-state | 120 | 0.00039 | 0.12 | No |
| Nikkei 225 | MS-GARCH | 137 | 2.4e-7 | 0.27 | No |
| Nikkei 225 | HMM-gated GARCH | 107 | 0.024 | 0.21 | No |
| Nikkei 225 | Historical simulation | 116 | 0.0016 | 0.00019 | No |
| Hang Seng | GBM | 153 | 5.6e-11 | 0 | No |
| Hang Seng | GARCH(1,1) | 148 | 1.0e-9 | 0.0013 | No |
| Hang Seng | HMM 2-state | 131 | 5.7e-6 | 0.069 | No |
| Hang Seng | MS-GARCH | 139 | 1.3e-7 | 0.28 | No |
| Hang Seng | HMM-gated GARCH | 126 | 4.8e-5 | 0.43 | No |
| Hang Seng | Historical simulation | 130 | 8.8e-6 | 8.2e-11 | No |
| Sensex | GBM | 75 | 0.35 | 1.0e-7 | No |
| Sensex | GARCH(1,1) | 98 | 0.00043 | 0.65 | No |
| Sensex | HMM 2-state | 74 | 0.42 | 0.058 | **Yes** |
| Sensex | MS-GARCH | 87 | 0.021 | 0.90 | No |
| Sensex | HMM-gated GARCH | 68 | 0.93 | 0.036 | No |
| Sensex | Historical simulation | 84 | 0.049 | 6.5e-8 | No |
| NIFTY 50 | GBM | 18 | 4.5e-5 | 0.69 | No |
| NIFTY 50 | GARCH(1,1) | 65 | 0.00056 | 0.98 | No |
| NIFTY 50 | HMM 2-state | 38 | 0.62 | 0.40 | **Yes** |
| NIFTY 50 | MS-GARCH | 42 | 0.89 | 0.35 | **Yes** |
| NIFTY 50 | HMM-gated GARCH | 38 | 0.62 | 0.40 | **Yes** |
| NIFTY 50 | Historical simulation | 43 | 0.77 | 0.00091 | No |

Markets passed (both tests): HMM 2 of 7, MS-GARCH 1, HMM-gated GARCH 1, all others 0.
**No model passes on the five markets with data from 1990.** With normal errors every model is
too narrow over 30+ years; fat-tailed errors would fix this but are outside the syllabus.

### Violation rate by sub-period (%)

| Market | Period | GBM | GARCH | HMM | MS-GARCH | Gated | Hist. sim |
|---|---|---|---|---|---|---|---|
| S&P 500 | pre-2008 | 2.70 | 2.01 | 1.46 | 2.08 | 1.34 | 1.51 |
| S&P 500 | GFC | 23.97 | 4.11 | 15.07 | 4.79 | 4.11 | 10.96 |
| S&P 500 | post-GFC | 1.69 | 2.35 | 1.92 | 2.19 | 1.78 | 1.15 |
| S&P 500 | COVID | 18.09 | 7.45 | 10.64 | 6.38 | 7.45 | 10.64 |
| FTSE 100 | pre-2008 | 2.62 | 1.51 | 1.56 | 1.63 | 1.36 | 1.61 |
| FTSE 100 | GFC | 16.11 | 3.36 | 9.40 | 3.36 | 3.36 | 6.04 |
| FTSE 100 | post-GFC | 1.31 | 2.12 | 1.58 | 1.91 | 1.55 | 1.15 |
| FTSE 100 | COVID | 18.28 | 5.38 | 4.30 | 5.38 | 3.23 | 12.90 |
| DAX | pre-2008 | 2.40 | 1.41 | 1.71 | 1.44 | 1.29 | 1.61 |
| DAX | GFC | 13.61 | 4.76 | 7.48 | 4.08 | 2.72 | 6.12 |
| DAX | post-GFC | 1.25 | 1.90 | 1.66 | 1.81 | 1.61 | 1.32 |
| DAX | COVID | 16.13 | 4.30 | 5.38 | 3.23 | 3.23 | 11.83 |
| Nikkei 225 | pre-2008 | 0.81 | 1.25 | 0.94 | 1.27 | 0.89 | 1.25 |
| Nikkei 225 | GFC | 15.00 | 2.86 | 10.71 | 2.86 | 3.57 | 7.86 |
| Nikkei 225 | post-GFC | 1.19 | 1.88 | 1.51 | 1.86 | 1.51 | 1.14 |
| Nikkei 225 | COVID | 7.69 | 6.59 | 4.40 | 2.20 | 4.40 | 7.69 |
| Hang Seng | pre-2008 | 2.20 | 1.72 | 1.52 | 1.67 | 1.47 | 1.70 |
| Hang Seng | GFC | 14.69 | 2.10 | 3.50 | 2.10 | 1.40 | 4.20 |
| Hang Seng | post-GFC | 0.81 | 1.64 | 1.45 | 1.50 | 1.45 | 1.15 |
| Hang Seng | COVID | 7.61 | 3.26 | 4.35 | 3.26 | 4.35 | 7.61 |
| Sensex | pre-2008 | 1.52 | 1.52 | 1.28 | 1.38 | 1.24 | 1.33 |
| Sensex | GFC | 9.49 | 2.19 | 3.65 | 2.92 | 1.46 | 5.11 |
| Sensex | post-GFC | 0.42 | 1.32 | 0.86 | 1.13 | 0.81 | 0.97 |
| Sensex | COVID | 10.11 | 6.74 | 7.87 | 5.62 | 5.62 | 11.24 |
| NIFTY 50 | post-GFC | 0.44 | 1.58 | 0.92 | 1.02 | 0.92 | 1.05 |
| NIFTY 50 | COVID | 11.24 | 6.74 | 4.49 | 6.74 | 4.49 | 12.36 |

### HMM-gated GARCH vs the plain HMM

| Market | Violations HMM -> gated | GFC violations | COVID violations | Diebold-Mariano p | Verdict |
|---|---|---|---|---|---|
| S&P 500 | 169 -> 142 | 22 -> 6 | 10 -> 7 | 0.010 | Gated better, significant |
| FTSE 100 | 152 -> 134 | 14 -> 5 | 4 -> 3 | 0.014 | Gated better, significant |
| DAX | 161 -> 134 | 11 -> 4 | 5 -> 3 | 0.003 | Gated better, significant |
| Nikkei 225 | 120 -> 107 | 15 -> 5 | 4 -> 4 | 0.081 | Gated better, not significant |
| Hang Seng | 131 -> 126 | 5 -> 2 | 4 -> 4 | 0.16 | Gated better, not significant |
| Sensex | 74 -> 68 | 5 -> 2 | 7 -> 5 | 0.16 | Gated better, not significant |
| NIFTY 50 | 38 -> 38 | not in sample | 4 -> 4 | 0.33 | Tie |

Why: the HMM learns its crisis volatility from history. Before 2008 the S&P 500 had five calm
years, so the HMM's crisis level was too low when the GFC hit (VaR flat around 3.5% in autumn
2008). The gate takes the larger of that level and GARCH's current volatility, so VaR rises with
the crash. Because the gated tick loss is lower in all seven markets, the gain is not just from a
wider VaR.

### Basel traffic light: red-zone years (yellow-zone years)

| Market | GBM | GARCH | HMM | MS-GARCH | Gated | Hist. sim |
|---|---|---|---|---|---|---|
| S&P 500 | 10 (6) | 3 (16) | 4 (12) | 2 (19) | 0 (15) | 3 (11) |
| FTSE 100 | 7 (9) | 1 (16) | 2 (13) | 1 (14) | 1 (12) | 5 (6) |
| DAX | 8 (4) | 0 (14) | 3 (12) | 0 (14) | 0 (11) | 5 (8) |
| Nikkei 225 | 1 (8) | 2 (9) | 1 (5) | 0 (11) | 0 (6) | 1 (10) |
| Hang Seng | 5 (5) | 1 (12) | 1 (8) | 1 (12) | 1 (7) | 2 (8) |
| Sensex | 2 (3) | 1 (6) | 0 (7) | 1 (6) | 0 (5) | 1 (7) |
| NIFTY 50 | 0 (1) | 0 (6) | 0 (1) | 0 (4) | 0 (1) | 1 (4) |
| **Total red** | **33** | **8** | **11** | **5** | **2** | **18** |

---

## 06. Forecast skill and run time

Source: `06_forecast_skill_and_runtime/ALL_forecast_skill.csv`, `ALL_timings.csv`.
Tick loss at tau = 0.01 (x 10^4, lower is better); skill = % reduction vs GBM; rank 1 = best in market.

| Market | GBM | GARCH | HMM | MS-GARCH | Gated | Hist. sim |
|---|---|---|---|---|---|---|
| S&P 500 | 5.238 (6) | 3.700 (3) | 4.192 (4) | 3.673 (2) | **3.633 (1)** | 4.539 (5) |
| FTSE 100 | 4.724 (6) | 3.411 (2) | 3.709 (4) | **3.323 (1)** | 3.423 (3) | 4.106 (5) |
| DAX | 5.618 (6) | 4.094 (2) | 4.388 (4) | **4.078 (1)** | 4.170 (3) | 5.133 (5) |
| Nikkei 225 | 5.663 (6) | 4.644 (3) | 4.860 (4) | 4.613 (2) | **4.604 (1)** | 5.646 (5) |
| Hang Seng | 6.345 (6) | 4.991 (2) | 5.120 (4) | 5.021 (3) | **4.946 (1)** | 5.897 (5) |
| Sensex | 5.564 (6) | 4.327 (2) | 4.466 (4) | **4.224 (1)** | 4.339 (3) | 5.236 (5) |
| NIFTY 50 | 4.617 (6) | **3.444 (1)** | 3.670 (3) | 3.690 (4) | 3.582 (2) | 4.324 (5) |

| Model | Skill vs GBM | Mean rank | Total run time, 7 markets | Per forecast |
|---|---|---|---|---|
| MS-GARCH 2-regime | 20-30% | 2.00 | 201 min | 156-265 ms |
| HMM-gated GARCH | 19-31% | 2.00 | 22.9 min | 19-27 ms |
| GARCH(1,1) | 18-29% | 2.14 | 3.2 min | 3-4 ms |
| HMM 2-state | 14-22% | 3.86 | 24.3 min | 19-29 ms |
| Historical simulation | 0-13% | 5.00 | 0.5 min | 0.3-0.7 ms |
| GBM | baseline | 6.00 | 0.7 min | 0.6-0.8 ms |

Diebold-Mariano vs the plain HMM: every GARCH-family model is significantly better on S&P 500,
FTSE 100 and DAX (p <= 0.03). GBM and historical simulation are significantly worse in every market.
Run times include CPU contention (seven markets ran in parallel on one machine).

GARCH is sharp but too narrow; the HMM is wide enough but coarse. HMM-gated GARCH is the only model
near the top on both coverage and tick loss, and it matches MS-GARCH's rank at about one-ninth of
the run time.

---

## 07. Regimes and transition matrices

Source: `07_regimes_and_transitions/ALL_regimes.csv`, `ALL_transition_matrices.csv`,
`ALL_crisis_windows.csv`. Full-sample fits (descriptive).

| Market | Days | Crash days | HMM crisis share | Crash days inside HMM crisis | Calm / crisis daily vol | HMM crisis spells (median, longest days) | MS-GARCH crisis share | Crash days inside MS-GARCH crisis |
|---|---|---|---|---|---|---|---|---|
| S&P 500 | 9,277 | 115 (1.2%) | 25.1% | 100% | 0.68% / 1.93% | 252 (4, 110) | 4.0% | 52% |
| FTSE 100 | 9,306 | 108 (1.2%) | 21.6% | 100% | 0.70% / 1.88% | 204 (5, 111) | 10.6% | 81% |
| DAX | 9,322 | 204 (2.2%) | 26.6% | 100% | 0.86% / 2.23% | 230 (4, 148) | 13.0% | 71% |
| Nikkei 225 | 9,037 | 237 (2.6%) | 20.2% | 90% | 1.06% / 2.56% | 236 (4, 56) | 8.4% | 77% |
| Hang Seng | 9,091 | 242 (2.7%) | 20.4% | 93% | 1.05% / 2.73% | 187 (4, 160) | 4.3% | 52% |
| Sensex | 7,228 | 174 (2.4%) | 24.6% | 100% | 0.86% / 2.41% | 204 (4, 129) | 2.8% | 47% |
| NIFTY 50 | 4,613 | 78 (1.7%) | 15.0% | 100% | 0.82% / 2.65% | 89 (4, 116) | 3.2% | 46% |

### Transition matrices (P[i, j] = P(regime j tomorrow | regime i today))

| Market | HMM P(calm->calm) | HMM P(crisis->crisis) | Expected calm / crisis stay (days) | Stationary crisis share | MS-GARCH P(crisis->crisis) | MS-GARCH crisis stay (days) | HMM flag switches per year |
|---|---|---|---|---|---|---|---|
| S&P 500 | 0.989 | 0.968 | 87 / 31 | 26.6% | 0.573 | 2.3 | 13.7 |
| FTSE 100 | 0.990 | 0.966 | 97 / 29 | 23.3% | 0.880 | 8.4 | 11.0 |
| DAX | 0.989 | 0.971 | 91 / 35 | 27.7% | 0.869 | 7.6 | 12.4 |
| Nikkei 225 | 0.986 | 0.954 | 71 / 22 | 23.4% | 0.811 | 5.3 | 13.2 |
| Hang Seng | 0.992 | 0.971 | 124 / 34 | 21.7% | 0.847 | 6.5 | 10.4 |
| Sensex | 0.987 | 0.965 | 78 / 28 | 26.7% | 0.723 | 3.6 | 14.2 |
| NIFTY 50 | 0.991 | 0.956 | 111 / 23 | 17.0% | 0.751 | 4.0 | 9.7 |

Expected stay = 1 / (1 - P(stay)); stationary crisis share = P(calm->crisis) / (P(calm->crisis) + P(crisis->calm)).

| Check | Result | Verdict |
|---|---|---|
| Two regimes in every market | Crisis regime 2.4-3.2x as volatile | Pass |
| Crash days fall in the crisis regime | 90-100% (HMM) | Pass |
| Regimes are persistent | P(stay) >= 0.95 in every market | Pass |
| Stationary share matches observed share | 17-28% implied vs 15-27% observed | Pass |
| MS-GARCH's crisis regime is a crisis period | 3-13% of days, stays of 2-8 days: it marks extreme days, not crises | Fail |
| Daily flags are as stable as the matrix implies | Flags switch 10-14 times a year, more than implied; single noisy days cross P(crisis) = 0.5 | Fail (spells are merged across gaps of up to 5 days for the length study) |
| 3-state HMM (NIFTY) | BIC prefers 2 states; two of the three states swap almost every day (see 02) | Rejected; 2 states kept |

### Share of each documented crisis window flagged as crisis by the HMM (%)

| Window | S&P 500 | FTSE 100 | DAX | Nikkei | Hang Seng | Sensex | NIFTY |
|---|---|---|---|---|---|---|---|
| 1997 Asian crisis | 36 | 47 | 76 | 45 | 70 | 38 | before data |
| 1998 Russia/LTCM | 83 | 95 | 97 | 71 | 92 | 85 | before data |
| 2000-02 dot-com bust | 58 | 44 | 58 | 38 | 31 | 43 | before data |
| 2008 GFC | 99 | 95 | 94 | 87 | 100 | 98 | 97 |
| 2011 euro/US downgrade | 98 | 88 | 100 | 14 | 90 | 68 | 62 |
| 2015-16 China slowdown | 50 | 57 | 68 | 41 | 36 | 16 | 14 |
| 2020 COVID | 88 | 95 | 95 | 60 | 41 | 67 | 64 |
| 2022 rate-hike onset | 78 | 43 | 46 | 14 | 60 | 24 | 22 |

---

## 08. Chapman-Kolmogorov check

Source: `08_chapman_kolmogorov/CK_summary.csv` (detail per horizon in `CK_crash_days.csv`,
`CK_regime_flags.csv`); code `src/chapman_kolmogorov.py`.

For a Markov chain, P^(n) = P^n. Each model's n-step prediction is compared with the empirical
n-step frequency at horizons 1-40 trading days. A model passes at a horizon if its value lies
inside the 95% moving-block-bootstrap band of the empirical value (250-day blocks, 1,000
replications). Full-sample fits, so this is a goodness-of-fit check.

### A. Crash days: P(crash at t+n | crash at t), horizons passed out of 40

| Market | Poisson | Observed Markov chain on the crash flag | HMM 2-state | HMM mean abs. error |
|---|---|---|---|---|
| S&P 500 | 19 | 20 | 39 | 0.055 |
| FTSE 100 | 12 | 14 | 34 | 0.044 |
| DAX | 3 | 4 | 35 | 0.033 |
| Nikkei 225 | 20 | 21 | 39 | 0.013 |
| Hang Seng | 6 | 7 | 39 | 0.019 |
| Sensex | 17 | 18 | 37 | 0.032 |
| NIFTY 50 | 38 | 38 | 39 | 0.044 |
| **All, of 280** | **115 (41%)** | **122 (44%)** | **262 (94%)** | |

- After a crash, the chance of another crash averages 8-13% over the next 20 days, against a
  1-3% base rate (3-9x). The observed chain's memory dies within 2-3 days; the HMM's hidden chain
  reproduces the slow decay.
- NIFTY cannot discriminate: its crashes sit in 2008-09, so the bootstrap band reaches 0.
  Without NIFTY: HMM 223/240, chain 84/240, Poisson 77/240.
- The HMM's curve sits below the empirical one in the Western markets: normal errors give crash
  days too little probability (the same thin-tail problem as in VaR coverage).

### B. Daily HMM regime flags: P(same regime at t+n | regime at t), horizons passed out of 40

| Market | Flags' own one-step matrix, calm | Same, crisis | HMM fitted matrix, calm | HMM fitted matrix, crisis |
|---|---|---|---|---|
| S&P 500 | 2 | 1 | 34 | 30 |
| FTSE 100 | 3 | 2 | 34 | 31 |
| DAX | 2 | 2 | 33 | 31 |
| Nikkei 225 | 7 | 1 | 21 | 28 |
| Hang Seng | 2 | 2 | 34 | 31 |
| Sensex | 3 | 2 | 31 | 32 |
| NIFTY 50 | 3 | 37 | 15 | 35 |
| **All, of 280** | **22** | **47** | **202** | **218** |

The daily flags are not themselves a Markov chain (noise flips them), but the HMM's fitted matrix
describes the underlying regime at 75% of horizon checks, failing mainly at n = 1-5.

---

## 09. Crisis timing and length

Source: `09_crisis_timing_and_length/VALIDATION_*.csv`. Full-sample labels, so descriptive;
the real-time evidence is the backtest in 05.

Yardsticks, fixed before looking at the labels: 83 bear markets from each market's own prices
(20%-down / 20%-up rule); NBER US recession dates; Thai baht devaluation 2 Jul 1997, Russian
default 17 Aug 1998, LTCM recapitalisation 23 Sep 1998, WHO pandemic declaration 11 Mar 2020.

### Timing against dated events (trading days to the first HMM crisis flag; negative = already in crisis)

| Event | S&P 500 | FTSE 100 | DAX | Nikkei | Hang Seng | Sensex | NIFTY | Verdict |
|---|---|---|---|---|---|---|---|---|
| WHO pandemic, 11 Mar 2020 | -12 | -12 | -12 | -9 | -2 | -2 | -2 | Before it in 7 of 7 |
| LTCM rescue, 23 Sep 1998 | -18 | -34 | -35 | -18 | -84 | +1 | - | Before it in 5 of 6 |
| Russian default, 17 Aug 1998 | -4 | -8 | -8 | +8 | -57 | 0 | - | Before or same day in 5 of 6 |
| Thai baht devaluation, 2 Jul 1997 | +13 | +1 | +14 | +27 | +32 | +28 | - | Late in 6 of 6 |

| NBER recession (S&P 500) | Share flagged by HMM | Share flagged by MS-GARCH |
|---|---|---|
| 2001 | 53% | 2% |
| 2007-09 | 82% | 4% |
| 2020 | 100% | 14% |

### Overlap with bear-market days

| Market | Bear-day share | HMM recall | HMM precision | Corr(P(crisis), bear) | MS-GARCH recall |
|---|---|---|---|---|---|
| S&P 500 | 11.4% | 65% | 30% | 0.37 | 6% |
| FTSE 100 | 20.5% | 45% | 43% | 0.32 | 20% |
| DAX | 21.8% | 47% | 38% | 0.27 | 23% |
| Nikkei 225 | 34.5% | 30% | 51% | 0.22 | 12% |
| Hang Seng | 30.7% | 31% | 47% | 0.20 | 8% |
| Sensex | 22.2% | 46% | 42% | 0.30 | 6% |
| NIFTY 50 | 16.1% | 34% | 37% | 0.26 | 7% |

### Summary

| Check | Result | Verdict |
|---|---|---|
| Start lag after a bear-market peak | Median 8 trading days; 54% within 10 days, 70% within 30 | Pass |
| Sudden crashes | COVID +3 days (S&P 500, DAX); 1998 +2 (FTSE), +5 (DAX); GFC +8 | Pass |
| Crisis length vs bear length | Median 130 vs 164 days (ratio 1.17); 65% within +/-50%; rank correlation 0.67 | Pass |
| NBER 2007-09 length | 377 days actual vs 393 HMM (4 Jan 2008 - 27 Jul 2009), 4% long | Pass |
| NBER 2001 length | 166 days vs 154 (7% short) | Pass |
| End of crisis | Median 33 trading days after the price trough (volatility stays high in the rebound) | Expected for a volatility regime |
| COVID crash length | 21-24 days peak to trough vs 94-99 day episode | Fail (about 4x too long) |
| Slow low-volatility bears | Nikkei 1994-95: 60 of 263 days flagged; Hang Seng 2010-11: 90 of 225 | Fail (missed) |
| MS-GARCH length | Flags only 2-14% of crisis days; apparent length match comes from scattered flags | Fail (artifact) |

---

## 10. Trading applications

Source: `10_trading_applications/TRADE_*.csv`; code `src/trading_applications.py`.
Every forecast is the out-of-sample one from 05 (daily vol = VaR / 2.326). "Trailing 21-day vol"
is a no-model practitioner baseline. Pass criteria were written in the code before the run and
parameters were not tuned.

### Trade 1, options: selling one-month variance at the VIX level

Index options usually price more volatility than follows, so desks sell one-month variance
(variance swaps, delta-hedged straddles). VIX and India VIX are the 30-day variance-swap rates by
construction, so they are the strike K. Every 21 trading days the desk sells one month of
variance and holds it to expiry. P&L per $1 of vega notional, in vol points, with a 0.5
vol-point bid-ask:

    P&L = ((K - 0.5)^2 - RV) / (2K),   RV = (252/21) * sum of 21 squared daily returns * 100^2

- **Always short:** sell every month (the industry carry trade; benchmark).
- **Filter:** sell only if the model's forecast vol < K, otherwise stay flat.
- **Spread:** sell if forecast < K, buy if forecast > K.
- **Pass** = smaller max drawdown, smaller worst month and Sharpe >= Always short. The Sharpe
  gain's p-value comes from a one-sided block bootstrap of months (6-month blocks, 5,000 reps).

| Market | Strategy | Months selling | Mean P&L | Sharpe | Worst month | Max drawdown | GFC P&L | Feb 2018 P&L | COVID P&L | Sharpe gain p | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| NIFTY 50 (195 months, 2010-2026) | Always short | 100% | 1.45 | 0.71 | -60.7 | 111.6 | - | 5.0 | -75.8 | | Benchmark |
| | Filter: HMM-gated GARCH | 41% | 1.15 | 1.30 | -10.3 | 12.2 | - | 5.6 | 11.6 | 0.26 | **Pass** |
| | Filter: GARCH(1,1) | 83% | 1.76 | 1.65 | -13.4 | 19.1 | - | 5.0 | 25.5 | 0.16 | **Pass** |
| | Filter: MS-GARCH | 53% | 1.07 | 1.18 | -10.9 | 13.1 | - | 5.0 | 25.5 | 0.32 | **Pass** |
| | Filter: HMM 2-state | 42% | 0.89 | 0.57 | -60.7 | 67.4 | - | 5.6 | -40.3 | 0.72 | Fail |
| | Filter: Trailing 21-day vol | 86% | 1.58 | 0.95 | -60.7 | 61.3 | - | 5.0 | -35.2 | 0.10 | Fail |
| | Filter: GBM | 9% | 0.15 | 0.11 | -60.7 | 60.7 | - | 0.0 | -26.4 | 0.99 | Fail |
| | Filter: Historical simulation | 44% | 0.64 | 0.40 | -60.7 | 62.4 | - | 5.0 | -38.0 | 0.93 | Fail |
| | Spread: GARCH(1,1) | 100% | 1.90 | 0.94 | -13.4 | 38.2 | - | 5.0 | 123.8 | 0.36 | **Pass** |
| S&P 500 (417 months, 1992-2026) | Always short | 100% | 2.29 | 0.96 | -91.9 | 111.4 | -71.7 | -21.0 | -73.4 | | Benchmark |
| | Filter: HMM-gated GARCH | 69% | 1.85 | 0.96 | -91.9 | 105.8 | 24.3 | 0.0 | -93.7 | 0.51 | Fail |
| | Filter: GARCH(1,1) | 91% | 2.32 | 1.16 | -91.9 | 105.8 | 24.3 | -19.5 | -93.7 | 0.30 | Fail |
| | Filter: MS-GARCH | 91% | 2.29 | 1.13 | -91.9 | 105.8 | -4.7 | 5.3 | -93.7 | 0.22 | Fail |
| | Filter: HMM 2-state | 73% | 1.81 | 0.81 | -91.9 | 103.4 | -36.7 | 0.0 | -73.4 | 0.95 | Fail |
| | Filter: Trailing 21-day vol | 88% | 2.32 | 1.18 | -91.9 | 110.0 | 24.3 | -19.5 | -93.7 | 0.27 | Fail |
| | Filter: GBM | 57% | 1.51 | 0.66 | -91.9 | 111.4 | -71.7 | 2.9 | -61.9 | 1.00 | Fail |
| | Filter: Historical simulation | 51% | 1.08 | 0.52 | -91.9 | 96.7 | -44.0 | -22.0 | -71.5 | 1.00 | Fail |

P&L in vol points per $1 of vega notional. All 14 "Spread" rows are in `TRADE_vrp_summary.csv`;
only GARCH on NIFTY passes. Long volatility pays the premium every month.

The trades that decided the result (from `TRADE_vrp_trades.csv`):

| Entry | Market | Implied K | Gated forecast | Trailing 21-day | Realised vol | Always-short P&L | What happened |
|---|---|---|---|---|---|---|---|
| 13 Mar 2020 | NIFTY 50 | 41.2 | 50.2 | 36.9 | 81.6 | -60.7 | Models stayed out; trailing rule sold and lost 60.7 |
| 11 Feb 2020 | NIFTY 50 | 14.2 | 15.0 | 14.4 | 39.9 | -49.4 | Gated, GARCH and trailing all stayed out (forecasts just above K) |
| 2 Mar 2020 | S&P 500 | 40.1 | 30.3 | 24.5 | 94.5 | -91.9 | Every filter sold: no model saw the second leg coming |
| 24 Sep 2008 | S&P 500 | 35.7 | 39.5 | 41.2 | 81.3 | -75.1 | Gated, GARCH and trailing stayed out |

### Trade 2, quant equity: volatility targeting

A fund holds the index under a daily 99% VaR budget of 2% of NAV: exposure = min(1, 2% / VaR
forecast). No leverage, cash earns 0, cost 5 bp per unit of turnover.
**Pass** = (1) days losing more than the budget not significantly above 1% (one-sided binomial
p >= 0.05); (2) max drawdown below a constant position with the same average exposure (chosen with
hindsight, which favours the benchmark); (3) Sharpe >= buy-and-hold. Sharpe gain p-value from a
daily block bootstrap (250-day blocks, 2,000 reps).

HMM-gated GARCH in every market:

| Market | Avg. exposure | CAGR (B&H) | Vol (B&H) | Sharpe (B&H) | Max DD | Same-exposure DD | B&H DD | GFC return (B&H) | COVID return (B&H) | Days over budget (p) | Turnover / year | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S&P 500 | 0.85 | 7.2% (8.8%) | 12.1% (18.1%) | 0.64 (0.56) | 44.1% | 50.2% | 56.8% | -14.6% (-37.8%) | -4.6% (-8.3%) | 1.16% (0.073) | 6.2 | **Pass** |
| FTSE 100 | 0.87 | 2.9% (4.4%) | 12.3% (17.1%) | 0.29 (0.34) | 47.5% | 47.3% | 52.6% | -11.8% (-30.3%) | -9.7% (-16.7%) | 1.26% (0.010) | 5.4 | Fail |
| DAX | 0.74 | 6.1% (8.3%) | 13.1% (21.7%) | 0.52 (0.48) | 52.8% | 60.6% | 72.7% | -15.2% (-36.4%) | -7.5% (-10.4%) | 1.38% (0.000) | 7.4 | Fail (budget) |
| Nikkei 225 | 0.63 | 1.4% (3.2%) | 12.9% (23.4%) | 0.17 (0.25) | 52.6% | 47.2% | 68.9% | -17.6% (-38.0%) | -2.9% (-5.9%) | 1.17% (0.065) | 6.5 | Fail |
| Hang Seng | 0.66 | 2.3% (5.1%) | 13.6% (24.9%) | 0.24 (0.33) | 50.4% | 48.9% | 65.2% | -11.1% (-36.1%) | -12.0% (-12.2%) | 1.41% (0.000) | 8.4 | Fail |
| Sensex | 0.68 | 7.1% (11.1%) | 11.8% (21.8%) | 0.64 (0.59) | 31.0% | 46.0% | 60.9% | -9.9% (-33.3%) | -9.5% (-15.4%) | 0.97% (0.63) | 5.1 | **Pass** |
| NIFTY 50 | 0.79 | 5.9% (9.6%) | 11.5% (16.4%) | 0.56 (0.64) | 21.7% | 31.5% | 38.4% | - | -9.7% (-15.0%) | 0.90% (0.76) | 4.0 | Fail (Sharpe) |

All models, Sharpe ratio (buy-and-hold in the first column):

| Market | Buy and hold | GBM | GARCH | HMM | MS-GARCH | Gated | Hist. sim | Trailing 21-day |
|---|---|---|---|---|---|---|---|---|
| S&P 500 | 0.558 | 0.557 | 0.643 | 0.606 | 0.617 | 0.637 | 0.574 | 0.634 |
| FTSE 100 | 0.340 | 0.345 | 0.289 | 0.289 | 0.299 | 0.291 | 0.371 | 0.293 |
| DAX | 0.479 | 0.488 | 0.528 | 0.513 | 0.522 | 0.516 | 0.497 | 0.518 |
| Nikkei 225 | 0.252 | 0.251 | 0.145 | 0.186 | 0.148 | 0.171 | 0.207 | 0.180 |
| Hang Seng | 0.325 | 0.344 | 0.283 | 0.242 | 0.258 | 0.236 | 0.366 | 0.314 |
| Sensex | 0.592 | 0.611 | 0.674 | 0.634 | 0.667 | 0.641 | 0.585 | 0.686 |
| NIFTY 50 | 0.642 | 0.632 | 0.584 | 0.558 | 0.559 | 0.558 | 0.497 | 0.596 |

All models, max drawdown (%):

| Market | Buy and hold | GBM | GARCH | HMM | MS-GARCH | Gated | Hist. sim | Trailing 21-day |
|---|---|---|---|---|---|---|---|---|
| S&P 500 | 56.8 | 49.3 | 44.0 | 42.4 | 44.8 | 44.1 | 36.0 | 44.4 |
| FTSE 100 | 52.6 | 47.8 | 48.4 | 46.6 | 48.2 | 47.5 | 35.4 | 48.4 |
| DAX | 72.7 | 55.1 | 52.1 | 53.3 | 51.9 | 52.8 | 44.1 | 51.9 |
| Nikkei 225 | 68.9 | 43.7 | 58.5 | 52.3 | 59.2 | 52.6 | 47.7 | 58.2 |
| Hang Seng | 65.2 | 41.9 | 48.5 | 49.9 | 48.8 | 50.4 | 39.2 | 49.2 |
| Sensex | 60.9 | 37.0 | 33.3 | 31.9 | 32.3 | 31.0 | 34.4 | 40.6 |
| NIFTY 50 | 38.4 | 26.0 | 22.9 | 21.7 | 21.9 | 21.7 | 33.9 | 23.3 |

- **Crash protection holds everywhere:** the GFC loss is cut by 54-70% in the six markets that
  cover it, and the COVID loss is smaller in all seven.
- **Higher Sharpe in only 3 of 7 markets** (S&P 500, DAX, Sensex), none significant (gated
  bootstrap p >= 0.15; smallest across all models 0.11).
- Across all 49 model-and-market runs, **5 pass all three criteria**: gated on S&P 500 and Sensex;
  GBM, HMM and MS-GARCH on Sensex.
- **Our models roughly tie the trailing 21-day rule** in volatility targeting (S&P 500 Sharpe
  0.637 vs 0.634).
- **Slow bears are not helped:** the 2000-02 drawdown is about -44% vs -49% for buy-and-hold.
- Costs are negligible: turnover is about 6x NAV a year, roughly 3 bp a year at 5 bp per unit.

### What the trading test shows

The practical value of the models is **risk control, not return prediction**: they tell a trader
when not to sell options and when to cut exposure, and that roughly halves crisis losses. Where the
forecast must react fast (NIFTY options, March 2020), the model-based filters beat a simple
trailing-volatility rule. Limits: the options P&L uses VIX as the variance-swap price, not traded
option quotes; one 0.5 vol-point cost is assumed; no Sharpe gain is statistically significant with
16-35 years of data; cash earns 0; index returns exclude dividends (for both strategy and benchmark).

---

## 11. Official NSE data experiments (teammate, commit f6441e4)

Source: `11_nse_data_experiments/` (`exp1/`, `exp2/`, `validation_exp1_exp2.json`, `audits/`).
NIFTY 50 rebuilt from official NSE data back to 1990 (`data/processed/nifty50_nse_1990_2026.csv`)
and audited against Yahoo. GBM, GARCH and historical simulation were rerun with different
training-start dates on a common evaluation window. Every run's validation status is PASS.

| Experiment | Training start | Model | Forecasts | Violations | Rate | Kupiec p | Christoffersen p | Verdict |
|---|---|---|---|---|---|---|---|---|
| Exp 1 (2010-01-20 to 2026-10-30) | 2008 | GBM | 4,163 | 18 | 0.43% | 3.3e-5 | 0.69 | Fail |
| | 2008 | GARCH | 4,163 | 69 | 1.66% | 9.8e-5 | 0.89 | Fail |
| | 2008 | Historical simulation | 4,163 | 43 | 1.03% | 0.83 | 0.082 | Pass |
| | 2008 | HMM 2-state | 4,163 | 40 | 0.96% | 0.80 | 0.38 | **Pass** |
| | 1990 | GBM | 4,163 | 15 | 0.36% | 1.8e-6 | 0.74 | Fail |
| | 1990 | GARCH | 4,163 | 53 | 1.27% | 0.089 | 0.24 | **Pass** |
| | 1990 | Historical simulation | 4,163 | 43 | 1.03% | 0.83 | 0.082 | Pass |
| Exp 2 (2011-06-28 to 2026-10-30) | 1990 (pre-launch) | GBM | 3,802 | 15 | 0.39% | 1.9e-5 | 0.73 | Fail |
| | 1990 | GARCH | 3,802 | 49 | 1.29% | 0.087 | 0.26 | Pass |
| | 1990 | Historical simulation | 3,802 | 43 | 1.13% | 0.43 | 0.098 | Pass |
| | 1996 (launch) | GBM | 3,802 | 20 | 0.53% | 0.0012 | 0.65 | Fail |
| | 1996 | GARCH | 3,802 | 49 | 1.29% | 0.087 | 0.26 | Pass |
| | 1996 | Historical simulation | 3,802 | 43 | 1.13% | 0.43 | 0.098 | Pass |
| | 2009 (free float) | GBM | 3,802 | 43 | 1.13% | 0.43 | 9.3e-5 | Fail |
| | 2009 | GARCH | 3,802 | 65 | 1.71% | 6.5e-5 | 0.91 | Fail |
| | 2009 | Historical simulation | 3,802 | 43 | 1.13% | 0.43 | 0.098 | Pass |

- A longer training history fixes GARCH on NIFTY: a 2008/2009 start fails, a 1990/1996 start
  passes. The difference is significant (McNemar exact p = 3.1e-5: 16 extra violations, all one-sided).
- Historical simulation does not depend on the start date (500-day window).
- Exp 1 GARCH: 17 of 199 refits had persistence alpha + beta >= 0.999 (warning, not an error).
- The NSE-based GBM count (18) reproduces the Yahoo-based study exactly, and the NSE-based HMM
  (40 violations, 0.96%) confirms the Yahoo result (38, 0.92%): the HMM passes on both data sources.
- Figures for both experiments (VaR timelines, rolling and annual exception rates, loss heatmaps,
  bootstrap loss intervals) are in `exp1_exp2/`, with their numbers in `visualization_metrics.csv`.
  `pilot_exp1/` holds the pilot run.

---

## 12. Run logs and tests

`12_run_logs/` holds the run log and console output of every market, the aggregation,
crisis-length and transition studies, and `pytest_output.txt`: **40 of 40 unit tests pass**
(parameter recovery, backtest harness, MS-GARCH, HMM-gated GARCH, NSE experiment calendar and
runner, extended data loaders, Chapman-Kolmogorov, trading-test building blocks).

---

## Master scorecard

| # | Study | Test | Result | Verdict |
|---|---|---|---|---|
| 1 | Returns are normal | Jarque-Bera, kurtosis | Excess kurtosis 15.9, JB p = 0 | Rejected (intended) |
| 2 | Volatility clusters | Ljung-Box on r^2 | p = 0 | Confirmed |
| 3 | Realised vol tracks India VIX | Correlation | 0.87 | Pass |
| 4 | NIFTY VaR: GBM | Kupiec + Christoffersen | 0.44%, too conservative | Fail |
| 5 | NIFTY VaR: GARCH(1,1) | Kupiec + Christoffersen | 1.58%, too narrow | Fail |
| 6 | NIFTY VaR: HMM 2-state | Kupiec + Christoffersen | 0.92%; p = 0.62 / 0.40 | Pass |
| 7 | NIFTY VaR: MS-GARCH | Kupiec + Christoffersen | 1.02%; p = 0.89 / 0.35 | Pass |
| 8 | NIFTY VaR: HMM-gated GARCH | Kupiec + Christoffersen | 0.92%; p = 0.62 / 0.40 | Pass |
| 9 | NIFTY VaR: historical simulation | Kupiec + Christoffersen | 1.05%; violations cluster | Fail |
| 10 | 3-state HMM | BIC, regime persistence | BIC prefers 2; two states swap daily | Rejected (intended) |
| 11 | Crash days are Poisson | Dispersion test | Index 17.6, p = 0 | Rejected (intended) |
| 12 | Hawkes beats Poisson | Likelihood ratio | LR = 160, branching 0.76 | Pass |
| 13 | HMM estimates unbiased | 50-sample parameter recovery | 3 of 4 within 1 std; calm vol +2% | Pass (caveat) |
| 14 | Unit tests | pytest | 40 of 40 | Pass |
| 15 | Correct 1% coverage, 7 markets | Kupiec + Christoffersen | Best: HMM 2 of 7; none on 1990-start markets | Fail |
| 16 | Fewest violations | Count per market | HMM-gated GARCH in 6 of 7 | Pass |
| 17 | Gated beats HMM | Diebold-Mariano | Better in 6 of 7, significant in 3 | Partial |
| 18 | Gated fixes crisis onset | GFC violations | S&P 500 22 -> 6; fewer in every market | Pass |
| 19 | Basel red-zone years | Traffic light | Gated 2, HMM 11, GBM 33 | Pass (best) |
| 20 | Forecast skill | Tick-loss rank | Gated and MS-GARCH tied at 2.00 | Pass (joint best) |
| 21 | Speed | Run time | Gated 23 min vs MS-GARCH 201 min | Pass |
| 22 | Calm/crisis regimes in every market | HMM fit | Crisis 2.4-3.2x as volatile, 15-27% of days | Pass |
| 23 | Crash days in the crisis regime | Share | 90-100% | Pass |
| 24 | Regimes persistent | Transition matrix | P(stay) >= 0.95 | Pass |
| 25 | MS-GARCH regime is a crisis period | Share and stay | 3-13% of days, 2-8 day stays | Fail |
| 26 | HMM covers the 2008 GFC | Window overlap | 87-100%, all 7 markets | Pass |
| 27 | HMM covers COVID | Window overlap | 41-95% | Partial |
| 28 | Crisis start | Lag after bear peak | Median 8 days | Pass |
| 29 | Crisis before WHO declaration | Event date | 7 of 7 markets | Pass |
| 30 | Asian crisis 1997 caught early | Event date | Late in 6 of 6 | Fail |
| 31 | NBER recessions flagged | Overlap | 53% / 82% / 100% | Pass (2001 partial) |
| 32 | Crisis length vs bear markets | Episode length | Ratio 1.17, 65% within +/-50% | Pass |
| 33 | NBER 2007-09 length | Episode length | 4% off | Pass |
| 34 | COVID crash length | Episode length | About 4x too long | Fail |
| 35 | Slow bear markets detected | Overlap | Nikkei 1994, Hang Seng 2010 missed | Fail |
| 36 | NSE data: GARCH with 1990 history | Kupiec + Christoffersen | 1.27%, p = 0.089 / 0.24 | Pass |
| 37 | Crash days follow the HMM's hidden Markov chain | Chapman-Kolmogorov | 262 of 280 horizons (94%) | Pass |
| 38 | Crash days follow an observed Markov chain | Chapman-Kolmogorov | 122 of 280 (44%); memory dies in 2-3 days | Rejected (intended) |
| 39 | HMM fitted matrix predicts n-step regime flags | Chapman-Kolmogorov | 420 of 560 (75%); misses n = 1-5 | Partial |
| 40 | Daily regime flags are themselves Markov | Chapman-Kolmogorov | 69 of 560 (12%); flag noise | Rejected (intended) |
| 41 | Options: model filter improves selling volatility (NIFTY) | Drawdown, worst month, Sharpe | Gated: worst month -10.3 vs -60.7, Sharpe 1.30 vs 0.71 (p = 0.26) | Pass (not significant) |
| 42 | Options: model filter improves selling volatility (S&P 500) | Drawdown, worst month, Sharpe | GFC +24 vs -72, but no filter avoids March 2020 (-91.9) | Fail |
| 43 | Volatility targeting with gated VaR, 7 markets | Budget, drawdown, Sharpe | Passes in 2 of 7; GFC loss cut 54-70% everywhere | Partial |
| 44 | Leverage effect (NIFTY) | Corr(return, next-day vol change) | -0.014 | Not found |
| 45 | NSE data: HMM 2-state, 2008 start | Kupiec + Christoffersen | 0.96%, p = 0.80 / 0.38 | Pass |

**Totals (45 studies): 26 pass, 4 partial, 10 fail or not found, 5 rejected hypotheses where
rejection is the intended finding.**

---

## Known limitations

- **Thin tails.** With normal errors no model gets 1% coverage over 30+ years (1.25-2.6%
  violations on the 1990-start markets). Fat-tailed errors would fix this but are outside the
  syllabus.
- **Gated gain is significant in 3 of 7 markets only** (Diebold-Mariano: S&P 500, FTSE 100, DAX).
- **Regime labels in 07-09 are full-sample fits**; only the backtests (05, 10) are strictly out of sample.
- **NIFTY's Yahoo sample is short** (2008 on), so the GFC is not backtested on NIFTY; 11 uses NSE data to 1990.
- **HMM VaR uses a moment-matched normal**, which understates the exact mixture quantile when the regime is uncertain (2.85% vs 3.99% on 4 Oct 2022).
- **No expected-shortfall backtest**, although Basel now uses 97.5% ES.
- **No robustness checks** on the gate rule, refit frequency, the 0.5 crisis threshold or the 5-day merge gap.
- **Trading test simplifications:** VIX as the variance-swap price, a fixed 0.5 vol-point cost,
  cash at 0%, price indices without dividends.

---

## Reproduce (only if you want to; every output is already in this folder)

```
py -3.11 -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt
python notebooks/01_stylized_facts.py          # 01
python -m src.model_comparison                 # 02
python -m src.extremes                         # 03
python -m src.simulation_study                 # 04
python -m src.multi_market --download          # 05-07: data
python -m src.multi_market --market sp500      # repeat for each market (MS-GARCH is the slow part)
python -m src.multi_market --aggregate
python -m src.regime_transitions               # 07
python -m src.chapman_kolmogorov               # 08
python -m src.crisis_validation                # 09
python -m src.crisis_duration                  # 09
python -m src.trading_applications             # 10
python -m src.make_multi_market_figures
python -m pytest tests -q                      # 12
python -m src.collect_results                  # rebuilds this folder
```
