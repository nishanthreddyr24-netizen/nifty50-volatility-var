# Report and presentation guide

This file gives the team everything needed to write the 10-12 page report and the 15-minute
presentation. It follows the course's required structure section by section. Each section has:

- **What the course asks** (from the project brief),
- **Draft text** to adapt (write it in your own words; every number is from our results),
- **Numbers** and **tables** to quote,
- **Figure(s) to use**: every image is 300 dpi and saved in one folder,
  [`final_results/13_report_figures/`](final_results/13_report_figures/).

All numbers trace back to a file in [`final_results/`](final_results/) (see its
[README](final_results/README.md)). Main market for the report: **NIFTY 50, 2008-2026**. The seven-market
study and the trading test are the "real data at scale" and "interpretation" parts.

---

## 0. Checklist: where each course requirement is answered

| Course requirement | Where in the report | Figure / table |
|---|---|---|
| Real-world problem, why randomness, question | Sec. 1-2 | RF01 |
| Random variables, process, state space, time, assumptions, distributions | Sec. 6 | Table 6.1 |
| Markov chain: transition matrix | Sec. 5.1, 9 | RF02, RF03(a), fig06 |
| Markov chain: n-step probabilities | Sec. 5.1, 9 | RF03(b), fig11 |
| Markov chain: stationary distribution | Sec. 5.1, 9 | RF02, RF03(b) |
| Markov chain: state-transition diagram, heatmap, state probabilities vs time, long-run distribution | Sec. 9 | RF02, RF03, RF05 |
| Poisson: inter-arrival distribution | Sec. 5.2, 11 | RF06(b) |
| Poisson: event-count distribution, expected number of events | Sec. 5.2, 11 | RF06(a)(c) |
| Poisson: event-count plot, inter-arrival histogram, empirical vs theoretical | Sec. 11 | RF06 |
| Own Python code | Sec. 8 | Appendix |
| Simulate the process, multiple sample paths | Sec. 9 | RF07, RF08, RF09 |
| Estimate parameters | Sec. 7 | RF11, RF12 |
| Compare simulation with theory | Sec. 9 | RF03(b), RF04, RF10, Table 9.1 |
| Real data (public source) | Sec. 4 | Table 4.1 |
| Parameter estimation and validation, quantified | Sec. 7, 11 | Table 11.1 |
| Interpretation, limitations, extensions | Sec. 12-13 | fig12, fig13 |

---

## Figure list (all in `final_results/13_report_figures/`, 300 dpi)

Use about 10 figures in the main text and put the rest in the appendix.

| Figure | File | Shows | Use in | Priority |
|---|---|---|---|---|
| RF01 | `RF01_pipeline.png` | The course pipeline mapped to our project | Introduction, slide 2 | Main |
| RF02 | `RF02_state_transition_diagram.png` | 2-state Markov chain: probabilities, stays, long-run shares | Stochastic model | **Main** |
| RF03 | `RF03_transition_matrix_and_nstep.png` | (a) transition heatmap; (b) P^n theory vs 10,000 simulated chains converging to the stationary distribution | Theory vs simulation | **Main** |
| RF04 | `RF04_sojourn_times.png` | Stay lengths: geometric theory vs simulation vs observed | Simulation results | Main |
| RF05 | `RF05_regime_probability_nifty.png` | NIFTY price, filtered P(crisis) over time, India VIX | Real data results | **Main** |
| RF06 | `RF06_poisson_checks.png` | Poisson checks: N(t), inter-arrival survival function, yearly counts vs Poisson | Validation of Poisson | **Main** |
| RF07 | `RF07_poisson_vs_hawkes_simulation.png` | 1,000 Poisson vs 1,000 Hawkes simulations vs observed; Hawkes intensity | Simulation results | **Main** |
| RF08 | `RF08_simulated_sample_paths.png` | Actual returns vs one simulated path of GBM, GARCH, HMM | Simulation results | **Main** |
| RF09 | `RF09_simulated_price_paths.png` | 500 simulated price paths (GBM vs HMM) vs actual NIFTY | Simulation results | Appendix |
| RF10 | `RF10_theory_vs_simulation_vs_data.png` | ACF of squared returns and return density: theory vs simulation vs data | Theory vs data | **Main** |
| RF11 | `RF11_parameter_recovery.png` | 100 simulated samples refitted with our EM: no bias | Parameter estimation | Main |
| RF12 | `RF12_em_convergence.png` | Our EM: monotone log-likelihood, matches hmmlearn | Computational implementation | Appendix |
| RF13 | `RF13_nifty_covid_var.png` | NIFTY COVID crash: VaR forecasts and violations | Real data results | **Main** |
| fig01 | `fig01_violation_rate_by_market.png` | Violation rate, 6 models x 7 markets | Real data results | **Main** |
| fig04 | `fig04_basel_red_years.png` | Basel red-zone years by model | Validation | Main |
| fig05 | `fig05_forecast_skill.png` | Forecast skill vs GBM | Validation | Appendix |
| fig06 | `fig06_transition_matrices.png` | Transition matrices, 7 markets | Real data results | Main |
| fig07 | `fig07_regime_timeline.png` | Regime timeline, 7 markets | Real data results | Appendix |
| fig09 | `fig09_crisis_duration.png` | Crisis length vs bear-market length | Validation | Appendix |
| fig11 | `fig11_chapman_kolmogorov.png` | Chapman-Kolmogorov check on crash days, 7 markets | Validation | **Main** |
| fig12 | `fig12_vrp_trade.png` | Options trade P&L | Interpretation | Main |
| fig13 | `fig13_vol_targeting.png` | Volatility targeting, S&P 500 | Interpretation | Appendix |
| 01 | `01_price_and_returns.png` | NIFTY price and daily returns | Data description | Main |
| 02 | `02_fat_tails.png` | Return histogram and QQ plot vs normal | Data description | Appendix |
| 04 | `04_vol_vs_vix.png` | Realised volatility vs India VIX | Data description | Appendix |

Suggested page budget (12 pages): title and abstract 0.5; introduction and problem 1; literature 1;
data 1; mathematical background and model 2.5; methodology and implementation 1; simulation results
1.5; real-data results 1.5; validation 1; discussion, limitations, conclusion 1; references 0.5.
Code goes in the appendix (not counted).

---

## Title

Pick one:
- **"Calm and Crisis: A Hidden Markov Model of Market Volatility and Crash Clustering, with Value-at-Risk Forecasting"**
- "Do Market Crashes Arrive as a Poisson Process? Regime-Switching Models of Risk for NIFTY 50 and Six World Indices"
- "Predicting Risk, Not Returns: Markov Regime-Switching Volatility Models for Value-at-Risk"

## Abstract (about 200 words; adapt)

> Daily stock returns are close to unpredictable, but their risk is not. We model the volatility of
> the NIFTY 50 index (2008-2026) as a two-state hidden Markov chain with calm and crisis regimes, and
> compare it with a random walk (geometric Brownian motion) and GARCH(1,1). The fitted chain is highly
> persistent: the calm regime lasts 111 trading days on average and the crisis regime 23, and the
> stationary distribution places 17% of days in crisis. We derive the n-step transition
> probabilities, stationary distribution, sojourn times and the implied kurtosis and autocorrelation
> of squared returns, and confirm each with simulation. Crash days (log return <= -3%) are not a
> Poisson process: the variance of yearly counts is 17.6 times the mean, 44% of gaps between
> crashes are five days or less against 8% under Poisson, and a self-exciting Hawkes process fits
> far better (likelihood ratio 160). The hidden chain passes a Chapman-Kolmogorov check on crash
> days at 94% of horizons against 44% for an observed Markov chain. Out of sample, regime models give
> correctly calibrated 99% Value-at-Risk on NIFTY 50; across seven markets and 30+ years, a regime-gated
> GARCH model has the fewest violations and Basel red-zone years. Used in trading, the forecasts cut
> crisis losses by 54-70% but do not significantly raise risk-adjusted returns.

---

## 1. Introduction

**Draft text:**

> Every bank, fund and exchange must answer one question each day: how much could we lose tomorrow?
> The standard answer is Value-at-Risk (VaR), the loss exceeded on only 1% of days. A VaR model
> needs a probability model for returns, and the simplest one, the random walk of Bachelier and the
> geometric Brownian motion (GBM) of Samuelson and Black-Scholes, assumes independent, normally
> distributed returns with constant volatility. Real markets break this assumption: large moves
> cluster in time (2008, 2020) and calm years are long and quiet. We model this with a stochastic
> process in which the market switches between hidden calm and crisis states according to a Markov
> chain, and we ask whether the theory of Markov chains and Poisson processes explains what we see
> in the data and gives better risk forecasts.

Figure: **RF01** (the pipeline: problem, data, probability model, process, analysis, simulation,
validation, interpretation). One sentence under it: "The project follows the full modelling
pipeline; each box is a section of this report."

Contributions to state:
1. A two-state hidden Markov model of NIFTY 50 volatility, with derived n-step, stationary, sojourn,
   kurtosis and autocorrelation properties, each checked by simulation.
2. A test of the Poisson assumption for crash days, and a self-exciting Hawkes alternative.
3. Out-of-sample VaR backtests of six models on seven indices (about 54,000 forecasts per model).
4. Our own Baum-Welch EM estimator, verified against a library implementation.
5. Two trading applications (selling options volatility, volatility targeting).

---

## 2. Problem Statement

**What is the real-world problem?** Forecasting the risk of an equity index: how large tomorrow's
loss can be (99% VaR), and whether extreme losses arrive at random or in clusters.

**Why is randomness involved?** Tomorrow's return is unknown. Its distribution changes over time:
NIFTY's daily volatility is about 0.84% in calm periods and about 2.52% in crises. The state of the
market (calm or crisis) is itself random and not directly observed.

**Questions** (state these as the report's research questions):
1. Can a two-state Markov chain describe how markets move between calm and crisis, and do its
   theoretical properties (transition matrix, n-step probabilities, stationary distribution, stay
   lengths) match the data?
2. Do crash days arrive as a Poisson process? If not, what process fits?
3. Do regime models give better out-of-sample 99% VaR than the random walk and GARCH?
4. Is any of this useful in practice (risk limits, options trading)?

---

## 3. Literature Review

Write about one paragraph per stream (about 1 page in total).

**Random walk and its failures.** Bachelier (1900) and Samuelson (1965) model prices as a random
walk / GBM, the basis of Black and Scholes (1973). Mandelbrot (1963) showed returns have fat
tails; Cont (2001) lists the stylized facts: heavy tails, volatility clustering, and near-zero
autocorrelation of returns but strong autocorrelation of squared returns.

**Time-varying volatility.** Engle (1982) introduced ARCH and Bollerslev (1986) GARCH, in which
today's variance depends on yesterday's shock and variance.

**Regime switching and hidden Markov models.** Hamilton (1989) introduced Markov regime switching in
economics. HMMs are estimated with the Baum-Welch algorithm (Baum et al., 1970), a case of EM
(Dempster, Laird and Rubin, 1977); Rabiner (1989) is the standard tutorial. Rydén, Teräsvirta and
Åsbrink (1998) showed that HMMs reproduce most stylized facts of daily returns but struggle with the
slow decay of the autocorrelation of squared returns, which is exactly what we find (Section 9).
Haas, Mittnik and Paolella (2004) proposed the Markov-switching GARCH we use. Ang and Bekaert (2002)
showed regimes matter for asset allocation.

**Point processes for extreme events.** The Poisson process is the baseline for random event
arrivals (Ross, 2014). Hawkes (1971) introduced self-exciting processes in which each event raises
the rate of future events; Ogata (1981) gave the thinning algorithm we use to simulate them.
Aït-Sahalia, Cacho-Diaz and Laeven (2015) used Hawkes processes for financial contagion.

**VaR and its validation.** Jorion (2007) covers VaR. Kupiec (1995) tests the violation rate,
Christoffersen (1998) the independence of violations, and the Basel Committee (1996) sets the
traffic-light rule used by regulators. Diebold and Mariano (1995) compare forecast accuracy, and
Gneiting (2011) shows the tick (quantile) loss is the right score for a quantile forecast.

**Applications.** Moreira and Muir (2017) show volatility-managed portfolios; Carr and Wu (2009)
document the variance risk premium that the options trade in Section 13 exploits.

**Our contribution relative to this:** we put the course's Markov-chain and Poisson theory (n-step
probabilities, stationary distribution, sojourn times, Chapman-Kolmogorov, inter-arrival
distribution) directly against market data, and test the models out of sample on seven markets.

---

## 4. Data Description

**Draft text:**

> We use daily closing prices of seven equity indices from Yahoo Finance (public): NIFTY 50
> (2008-2026, our main series), Sensex, S&P 500, FTSE 100, DAX, Nikkei 225 and Hang Seng (from 1990),
> plus the CBOE VIX and India VIX implied-volatility indices. We also rebuilt NIFTY 50 back to
> 1990 from official National Stock Exchange (NSE) files and audited it against Yahoo. We
> work with daily log returns r_t = ln(S_t / S_{t-1}).

**Table 4.1 (data):**

| Series | Source | Period | Daily returns | Use |
|---|---|---|---|---|
| NIFTY 50 (^NSEI) | Yahoo Finance | 2008-01-01 to 2026-10-23 | 4,613 | Main market |
| NIFTY 50 (official) | NSE | 1990-2026 | 8,832 | Robustness (Exp 1/2) |
| S&P 500, FTSE 100, DAX, Nikkei 225, Hang Seng | Yahoo Finance | 1990-2026 | 9,037-9,322 each | Seven-market study |
| Sensex | Yahoo Finance | 1997-2026 | 7,228 | Seven-market study |
| VIX, India VIX | Yahoo Finance | 1990 / 2008-2026 | | Validation, options trade |

All price files are in `final_results/00_input_data/`.

**Table 4.2 (NIFTY 50 stylized facts):**

| Statistic | Value | Meaning |
|---|---|---|
| Mean daily return | 0.028% (7.1% a year) | Small drift |
| Daily volatility | 1.28% (20.3% a year) | |
| Skewness | -0.32 | Large falls more common than large rises |
| Excess kurtosis | 15.9 (normal: 0) | Very fat tails |
| Worst / best day | -13.9% / +16.3% | |
| Jarque-Bera | 48,800, p = 0 | Normality rejected |
| Ljung-Box on squared returns, lag 10 | p = 0 | Volatility clusters |
| Corr(21-day realised vol, India VIX) | 0.87 | Our volatility measure tracks the market's |

Figures: **01_price_and_returns.png** (main); optional 02_fat_tails.png (QQ plot) and
04_vol_vs_vix.png.

Data cleaning: no observations were removed. 0-14 zero-return days per market (holidays) are kept;
no return exceeds 15% except one day each on Hang Seng, Sensex and NIFTY (all genuine market moves).

---

## 5. Mathematical Background

Write these derivations in the report; they are the "theoretical analysis" the course asks for.

### 5.1 Discrete-time Markov chains (the regime process)

Let S_t in {0 = calm, 1 = crisis} with transition matrix

$$P = \begin{pmatrix} 1-a & a \\ b & 1-b \end{pmatrix}, \qquad a = P(S_{t+1}=1 \mid S_t=0),\; b = P(S_{t+1}=0 \mid S_t=1).$$

**Chapman-Kolmogorov:** $P^{(m+n)} = P^{(m)} P^{(n)}$, so $P^{(n)} = P^n$.

**Closed form for two states** (diagonalise P: eigenvalues 1 and $\lambda = 1-a-b$):

$$P^n = \frac{1}{a+b}\begin{pmatrix} b & a \\ b & a \end{pmatrix} + \frac{(1-a-b)^n}{a+b}\begin{pmatrix} a & -a \\ -b & b \end{pmatrix}.$$

**Stationary distribution** (solve $\pi P = \pi$, $\pi_0+\pi_1=1$): $\pi = \left(\frac{b}{a+b}, \frac{a}{a+b}\right)$,
and $P^n \to \mathbf{1}\pi$ geometrically at rate $|\lambda|^n$.

**Sojourn (stay) time** in state i is geometric: $P(T_i = k) = p_{ii}^{k-1}(1-p_{ii})$, with
$E[T_i] = 1/(1-p_{ii})$.

### 5.2 Poisson process (crash-day arrivals)

N(t) = number of crash days by time t, with rate lambda:
- $N(t) \sim \text{Poisson}(\lambda t)$, so $E[N(t)] = \text{Var}[N(t)] = \lambda t$ (dispersion index 1).
- Inter-arrival times are i.i.d. $\text{Exponential}(\lambda)$: mean $1/\lambda$, coefficient of variation 1,
  $P(\text{gap} \le x) = 1-e^{-\lambda x}$.
- Counts in disjoint intervals are independent (no clustering).

Dispersion test: with yearly counts $x_1..x_n$, $D = \sum (x_i - \bar x)^2/\bar x \sim \chi^2_{n-1}$ under Poisson.

### 5.3 Hawkes (self-exciting) process

$$\lambda(t) = \mu + \alpha \sum_{t_i < t} e^{-\beta (t - t_i)}.$$

Each event adds alpha to the rate, which decays at rate beta (half-life ln2 / beta). Branching ratio
$n = \alpha/\beta$ = expected number of events triggered by one event. If n < 1 the process is
stationary with mean rate $\mu/(1-n)$, so $E[N(T)] \approx \mu T/(1-n)$. Log-likelihood:
$\ell = \sum_i \ln \lambda(t_i) - \int_0^T \lambda(s)\,ds$. Poisson is the special case alpha = 0.

### 5.4 Return models

- **GBM / random walk:** $dS_t = \mu S_t dt + \sigma S_t dW_t$, so $r_t \sim N(\mu - \sigma^2/2, \sigma^2)$ i.i.d.
- **GARCH(1,1):** $r_t = \mu + \varepsilon_t$, $\varepsilon_t = \sigma_t z_t$, $\sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2$;
  persistence $\alpha+\beta$, unconditional variance $\omega/(1-\alpha-\beta)$.
- **Hidden Markov model:** $r_t \mid S_t = i \sim N(\mu_i, \sigma_i^2)$, S_t the Markov chain of 5.1.
  The stationary return distribution is the mixture $\sum_i \pi_i N(\mu_i, \sigma_i^2)$.

**Derived HMM properties** (compared with data in Section 9):

Excess kurtosis of the mixture (with $m = \sum \pi_i \mu_i$, $v = \sum \pi_i(\sigma_i^2+\mu_i^2) - m^2$):
$$\kappa = \frac{\sum_i \pi_i\left[(\mu_i-m)^4 + 6(\mu_i-m)^2\sigma_i^2 + 3\sigma_i^4\right]}{v^2} - 3 > 0.$$

Autocorrelation of squared returns, with $g(i) = \mu_i^2 + \sigma_i^2$:
$$\text{Corr}(r_t^2, r_{t+k}^2) = \frac{\text{Var}_\pi\big(g(S)\big)}{\text{Var}(r_t^2)}\,\lambda^k, \qquad \lambda = 1-a-b,$$
so volatility clustering decays geometrically, at the chain's second eigenvalue.

### 5.5 Filtering and VaR

Forward filter (computed recursively, uses only past data):
$\alpha_t(j) \propto \left[\sum_i \alpha_{t-1}(i) P_{ij}\right] \phi(r_t; \mu_j, \sigma_j)$, $\alpha_t(j) = P(S_t = j \mid r_1..r_t)$.
One-day-ahead regime probabilities: $\alpha_t P$. VaR at 99% with a normal approximation:
$\text{VaR}_{t+1} = 2.326\,\hat\sigma_{t+1} - \hat\mu_{t+1}$.

---

## 6. Stochastic Model (course item B)

**Table 6.1: formulation**

| Item | Regime model (main) | Crash-day model |
|---|---|---|
| Random variables | $r_t$ daily log return; $S_t$ hidden regime | $N(t)$ number of crash days by t; gaps $\tau_k$ between crashes |
| Stochastic process | Hidden Markov chain $\{S_t\}$ with Gaussian emissions $\{r_t\}$ | Counting process $\{N(t)\}$: Poisson vs Hawkes |
| State space | $S_t \in \{\text{calm}, \text{crisis}\}$; $r_t \in \mathbb{R}$ | $N(t) \in \{0,1,2,\dots\}$ |
| Time parameter | Discrete, t = trading days | Continuous t (calendar days) / trading days |
| Assumptions | Markov property; time-homogeneous P; returns independent given the regime; normal within a regime | Poisson: independent, stationary increments, rate lambda. Hawkes: rate jumps after each event |
| Distributions | $r_t \mid S_t=i \sim N(\mu_i,\sigma_i^2)$; stays geometric | Poisson counts, exponential gaps (null); Hawkes (alternative) |

**Fitted NIFTY 50 model** (Figure **RF02**, the state-transition diagram):

| Parameter | Calm | Crisis |
|---|---|---|
| Mean return | +0.056%/day (+14.1% a year) | -0.107%/day (-26.8% a year) |
| Volatility | 0.844%/day (13.4% a year) | 2.522%/day (40.0% a year) |
| Stay probability $p_{ii}$ | 0.9910 | 0.9562 |
| Leave probability | a = 0.0090 | b = 0.0438 |
| Expected stay $1/(1-p_{ii})$ | 111 trading days | 23 trading days |
| Stationary share $\pi_i$ | 83.0% | 17.0% |

Second eigenvalue $\lambda = 1-a-b = 0.9472$ (memory half-life 12.8 trading days).

Other fitted models (NIFTY 50, full sample): GBM volatility 20.3% a year; GARCH(1,1)
$\alpha = 0.092$, $\beta = 0.899$, persistence 0.991 (half-life 76 days), unconditional
volatility 20.6% a year.

**Number of regimes (BIC):** 2 states: BIC -28,989 (7 parameters); 3 states: BIC -28,953
(14 parameters). BIC prefers 2. In the 3-state fit two states swap almost daily
(P = 0.635 and 0.996) and have the same average India VIX (16.7), so the third state is not a real
regime.

---

## 7. Methodology

### 7.1 Parameter estimation

| Model | Method | Detail |
|---|---|---|
| GBM | Maximum likelihood (closed form) | Sample mean and standard deviation of log returns |
| GARCH(1,1) | Quasi maximum likelihood | Normal likelihood, `arch` package |
| HMM | Maximum likelihood by Baum-Welch EM | Our own implementation (`src/em_hmm.py`); `hmmlearn` used in the backtests |
| MS-GARCH | Maximum likelihood, Hamilton filter | Haas et al. (2004), 4 random starts |
| Poisson | MLE: $\hat\lambda$ = events / time | |
| Hawkes | Maximum likelihood (L-BFGS-B) | Exponential kernel, O(n) recursion |
| Number of regimes | BIC | |

**EM (Baum-Welch), to state in the report:** E-step: forward-backward gives
$\gamma_t(i) = P(S_t=i \mid \text{data})$ and $\xi_t(i,j) = P(S_t=i, S_{t+1}=j \mid \text{data})$.
M-step: $\hat P_{ij} = \sum_t \xi_t(i,j) / \sum_t \gamma_t(i)$,
$\hat\mu_i = \sum_t \gamma_t(i) r_t / \sum_t \gamma_t(i)$,
$\hat\sigma_i^2 = \sum_t \gamma_t(i)(r_t-\hat\mu_i)^2 / \sum_t \gamma_t(i)$. The log-likelihood
never decreases.

### 7.2 Out-of-sample forecasting

Rolling backtest: on each day t, every model is fitted only to returns before t (500-day minimum
history; refit every 21 days for GARCH and 63 for HMM and MS-GARCH) and forecasts tomorrow's
99% VaR. A violation is a day with $r_t < -\text{VaR}_t$. This gives 4,113 forecasts on NIFTY and
about 54,000 per model across seven markets.

### 7.3 Validation methods

| Method | What it checks |
|---|---|
| Kupiec likelihood-ratio test | Violation rate = 1% |
| Christoffersen test | Violations independent (it fits a 2-state Markov chain to the 0/1 hit sequence) |
| Basel traffic light | Red zone = 10+ violations in a 250-day year |
| Tick loss + Diebold-Mariano | Forecast accuracy and its significance |
| Dispersion test, KS test | Poisson count and inter-arrival distributions |
| Likelihood ratio | Hawkes vs Poisson |
| Chapman-Kolmogorov check | n-step model predictions vs empirical n-step frequencies, 40 horizons, block-bootstrap bands |
| Simulation (parameter recovery) | Estimator is unbiased |
| Crisis dating | Regime labels vs 83 bear markets, NBER recessions, dated events |

### 7.4 Simulation design

- Markov chain: 10,000 chains from each starting state (n-step probabilities); one chain of
  2,000,000 days (stay lengths).
- Return paths: 300-500 paths of 4,613 days from each fitted model (GBM, GARCH, HMM).
- Crash process: 1,000 Poisson and 1,000 Hawkes paths over the NIFTY sample (Hawkes by Ogata thinning).
- Parameter recovery: 100 samples of 4,613 days from the fitted HMM, each refitted with our EM.

---

## 8. Computational Implementation

**Draft text:**

> All analysis is in Python 3.11 (numpy, pandas, scipy, matplotlib; arch for GARCH; hmmlearn as a
> reference). We wrote our own code for: the rolling backtest and all VaR tests; the HMM forward
> filter; the Baum-Welch EM estimator; MS-GARCH; the HMM-gated GARCH; the Poisson and Hawkes
> estimation and the Hawkes simulator; the Chapman-Kolmogorov check with block bootstrap; all
> simulations; and the trading tests. Every forecast uses only past data. 43 unit tests cover the
> estimators and tests, and all pass.

**Own EM vs library (Figure RF12):** our EM converges monotonically in 46 iterations to
log-likelihood 14,526.715, identical to hmmlearn with its default variance prior switched off.
hmmlearn's default prior (covars_prior = 0.01) gives 14,524.039 and inflates the calm volatility
from 0.814% to 0.844% a day. This is a nice point for the report: implementing the estimator
ourselves revealed a hidden default in the library.

**Run time** (all seven markets): MS-GARCH 201 min, HMM 24 min, HMM-gated GARCH 23 min, GARCH 3.2 min,
GBM 0.7 min, historical simulation 0.5 min.

Code structure (put in the appendix): see the "Appendix - Code" section below.

---

## 9. Simulation Results

All simulated data, with a README per simulation, is in
[`final_results/14_simulations/`](final_results/14_simulations/) (code: `src/simulators.py`).

### 9.1 Markov chain: theory vs simulation (Figure RF03)

| n (days ahead) | 1 | 5 | 10 | 20 | 50 | 100 | 250 |
|---|---|---|---|---|---|---|---|
| P(crisis at n \| crisis now), theory | 0.956 | 0.803 | 0.652 | 0.451 | 0.225 | 0.174 | 0.170 |
| P(crisis at n \| calm now), theory | 0.009 | 0.041 | 0.071 | 0.113 | 0.159 | 0.170 | 0.170 |

10,000 simulated chains match the closed-form $P^n$ to within 0.011 at every horizon, and both
rows converge to the stationary share 0.170. Interpretation: once in a crisis, the market is
still more likely than not in crisis 10 trading days later (0.65); after about 100 days the
starting state no longer matters.

### 9.2 Stay lengths (Figure RF04)

| Regime | Theory mean (geometric) | Simulated mean | Observed (Viterbi path) |
|---|---|---|---|
| Calm | 111 days | 111.3 days | 184 days (21 spells, median 135) |
| Crisis | 23 days | 23.0 days | 37 days (20 spells, median 14.5) |

Simulation confirms the geometric theory. Observed stays are longer than the theory says, and
their distribution has a heavier tail (a few very long calm periods and the 2008-09 crisis), so a
time-homogeneous chain is an approximation; duration-dependent regimes are an extension (Section 15).

### 9.3 Sample paths (Figures RF08 and RF09)

One simulated path of each model next to the real NIFTY returns: GBM has no quiet or wild
periods; GARCH and the HMM both produce bursts of volatility like the real data. The HMM's bursts
switch on and off sharply (regimes), GARCH's rise and fade smoothly. RF09 shows 500 price paths:
the actual NIFTY stays inside both models' 5-95% bands, so price levels alone cannot tell the
models apart; the difference is in risk (volatility and tails).

### 9.4 Theory vs simulation vs data (Figure RF10, Table 9.1)

**Table 9.1** (300 simulated paths per model, each 4,613 days)

| | Excess kurtosis | ACF of $r^2$, lag 1 | ACF of $r^2$, lag 20 |
|---|---|---|---|
| NIFTY 50, actual | **15.9** | **0.161** | **0.086** |
| GBM, simulated (theory: 0, 0, 0) | 0.0 (95%: -0.1 to 0.1) | 0.000 | 0.000 |
| GARCH(1,1), simulated | 3.1 (0.9 to 10.1) | 0.247 | 0.181 |
| HMM, theory (formulas in 5.4) | 4.83 | 0.223 | 0.079 |
| HMM, simulated | 4.81 (3.7 to 6.1) | 0.225 | 0.077 |

- **Theory = simulation** for the HMM (4.83 vs 4.81; 0.223 vs 0.225): the derivations are right.
- **HMM vs data:** it matches the ACF of squared returns up to about lag 20 (0.079 vs 0.086) but
  decays too fast after that (lag 100: 0.001 vs 0.060). Real volatility has longer memory than one
  geometric rate $\lambda^k$ can produce, a known result (Rydén et al., 1998).
- **Tails:** the HMM gives 4.8 excess kurtosis, a large improvement on GBM's 0, but well below the
  data's 15.9. Normal distributions inside each regime are too thin-tailed; this is the same reason
  VaR coverage fails on long samples (Section 10).

### 9.5 Poisson vs Hawkes simulation (Figure RF07)

| Model | Simulated dispersion index of yearly counts, 95% range | Share of simulations at least as clustered as NIFTY (17.6) |
|---|---|---|
| Poisson | 0.46 - 1.70 | 0% of 1,000 |
| Hawkes (fitted) | 2.14 - 29.6 | 11.8% of 1,000 |

A Poisson process essentially never produces the observed clustering; the fitted Hawkes process
produces it routinely. The Hawkes model's expected number of crashes, $\mu T/(1-n)$ = 78.0,
equals the observed 78.

### 9.6 Parameter recovery (Figure RF11)

100 samples simulated from the fitted HMM (true values = our EM estimates on NIFTY), each refitted
with our EM:

| Parameter | True | Mean estimate | Std across fits | Bias / (std / sqrt(100)) |
|---|---|---|---|---|
| Calm mean $\mu_0$ | 0.000594 | 0.000596 | 0.000128 | 0.15 |
| Crisis mean $\mu_1$ | -0.001090 | -0.001208 | 0.000978 | -1.21 |
| Calm vol $\sigma_0$ | 0.008143 | 0.008145 | 0.000092 | 0.25 |
| Crisis vol $\sigma_1$ | 0.024264 | 0.024289 | 0.000612 | 0.41 |
| P(calm to crisis) | 0.01007 | 0.01025 | 0.00183 | 1.02 |
| P(crisis to calm) | 0.04429 | 0.04502 | 0.00879 | 0.83 |

No parameter shows significant bias (all |t| < 2). The means are the least precise (the crisis mean
has a standard deviation as large as itself), which is why VaR depends mostly on the volatilities.
(An earlier 50-sample study with hmmlearn's defaults found +2.0% bias in calm volatility; our own
EM shows this came from the library's default prior, not from EM.)

---

## 10. Real Data Results

### 10.1 Regimes in NIFTY 50 (Figure RF05)

The filtered crisis probability (computed day by day from past data only) is near 1 throughout
2008-09 and in March-June 2020, and near 0 in 2017-2019. The most likely regime path (Viterbi)
puts 100% of the GFC window (Sep 2008 - Mar 2009) and 68.5% of the COVID window (15 Feb - 30 Jun 2020)
in the crisis regime (filtered, real-time: 97% and 64%). India VIX averages 34.9 in the crisis
regime vs 16.9 in calm, although India VIX is not used to fit the model.
**All 78 crash days fall in the filtered crisis regime** (75 of 78 on the Viterbi path).

### 10.2 VaR backtest, NIFTY 50 (Figure RF13, Table 10.1)

**Table 10.1:** 4,113 out-of-sample forecasts (2010-2026); expected violation rate 1.00%

| Model | Violations | Rate | Kupiec p | Christoffersen p | COVID violations (89 days) | Verdict |
|---|---|---|---|---|---|---|
| GBM | 18 | 0.44% | 0.00005 | 0.69 | 10 | Fail (too conservative) |
| GARCH(1,1) | 65 | 1.58% | 0.0006 | 0.98 | 6 | Fail (too narrow) |
| **HMM 2-state** | 38 | 0.92% | 0.62 | 0.40 | **4** | **Pass** |
| MS-GARCH | 42 | 1.02% | 0.89 | 0.35 | 6 | Pass |
| HMM-gated GARCH | 38 | 0.92% | 0.62 | 0.40 | 4 | Pass |
| Historical simulation | 43 | 1.05% | 0.77 | 0.0009 | 11 | Fail (violations cluster) |

RF13 shows why: in March 2020 GBM's VaR stays flat at about 3%, while the regime models' VaR widens
to 10-13% as the crisis regime takes over.

Robustness on official NSE data: the HMM gives 40 violations (0.96%, Kupiec p = 0.80,
Christoffersen p = 0.38); GARCH passes once its history starts in 1990 (1.27%, p = 0.089).

### 10.3 Seven markets, 30+ years (Figure fig01, Table 10.2)

**Table 10.2:** violation rate (%), expected 1.00

| Market | GBM | GARCH | HMM | MS-GARCH | HMM-gated GARCH | Hist. sim |
|---|---|---|---|---|---|---|
| S&P 500 | 2.60 | 2.22 | 1.93 | 2.19 | **1.62** | 1.52 |
| FTSE 100 | 2.26 | 1.87 | 1.73 | 1.83 | **1.52** | 1.45 |
| DAX | 2.01 | 1.72 | 1.82 | 1.69 | **1.52** | 1.55 |
| Nikkei 225 | 1.30 | 1.59 | 1.41 | 1.60 | **1.25** | 1.36 |
| Hang Seng | 1.78 | 1.72 | 1.52 | 1.62 | **1.47** | 1.51 |
| Sensex | 1.11 | 1.46 | 1.10 | 1.29 | **1.01** | 1.25 |
| NIFTY 50 | 0.44 | 1.58 | 0.92 | 1.02 | 0.92 | 1.05 |

- HMM-gated GARCH (the HMM, but in the crisis regime volatility = max(HMM crisis volatility, GARCH
  volatility)) has the fewest violations in 6 of 7 markets and the fewest Basel red-zone years:
  2, against 11 for the HMM and 33 for GBM (fig04).
- On the five markets with data from 1990, no model reaches 1%: normal errors are too thin-tailed
  over 30+ years.
- In the 2008 GFC on the S&P 500, the gate cut violations from 22 to 6 (fig02, fig03).

### 10.4 Regimes in every market (Figure fig06)

| Market | P(calm to calm) | P(crisis to crisis) | Calm / crisis stay (days) | Crisis share | Crash days in crisis |
|---|---|---|---|---|---|
| S&P 500 | 0.989 | 0.968 | 87 / 31 | 25.1% | 100% |
| FTSE 100 | 0.990 | 0.966 | 97 / 29 | 21.6% | 100% |
| DAX | 0.989 | 0.971 | 91 / 35 | 26.6% | 100% |
| Nikkei 225 | 0.986 | 0.954 | 71 / 22 | 20.2% | 90% |
| Hang Seng | 0.992 | 0.971 | 124 / 34 | 20.4% | 93% |
| Sensex | 0.987 | 0.965 | 78 / 28 | 24.6% | 100% |
| NIFTY 50 | 0.991 | 0.956 | 111 / 23 | 15.0% | 100% |

The same two-regime structure appears in every market; stationary crisis shares (17-28%) match the
observed shares (15-27%).

---

## 11. Model Validation

**Table 11.1: every validation, quantified**

| Check | Result | Verdict |
|---|---|---|
| Normality of returns | Excess kurtosis 15.9, Jarque-Bera p = 0 | Rejected |
| Crash counts Poisson (dispersion) | Variance/mean 17.6; D = 317.6, 18 df, p = 0 | Rejected |
| Inter-arrival times exponential (KS) | KS = 0.52, p = 7e-20; median gap 7 days vs 41 under Poisson; CV 2.54 vs 1 | Rejected |
| Zero-crash years | 6 observed vs 0.31 expected under Poisson | Rejected |
| Hawkes vs Poisson | LR = 160.3 (2 df), p = 0 | Hawkes preferred |
| Hawkes expected count | 78.0 vs 78 observed | Pass |
| HMM theory vs simulation (P^n, stays, kurtosis, ACF) | Agreement within simulation error | Pass |
| HMM vs data: ACF of r^2 up to lag 20 / beyond | 0.079 vs 0.086 at lag 20; too fast after | Partial |
| HMM vs data: kurtosis | 4.8 vs 15.9 | Fail (thin tails within regimes) |
| Chapman-Kolmogorov on crash days (fig11) | HMM 262/280 horizons (94%), observed chain 122 (44%), Poisson 115 (41%) | HMM passes |
| Number of regimes | BIC prefers 2 | 2 states |
| Parameter recovery (own EM) | All biases insignificant (|t| < 1.3) | Pass |
| Own EM vs hmmlearn | Same log-likelihood (14,526.715) | Pass |
| NIFTY VaR, HMM | 0.92%, Kupiec p = 0.62, Christoffersen p = 0.40 | Pass |
| Seven markets, 1% coverage | No model on the five 1990-start markets | Fail |
| Fewest violations / Basel red years | Gated: 6 of 7 markets; 2 red years | Pass |
| Crisis dating | Median 8 trading days after bear-market peaks; in crisis before the WHO pandemic declaration in 7 of 7 markets; NBER 2007-09 length within 4% | Pass |
| Slow, low-volatility bear markets | Nikkei 1994-95, Hang Seng 2010-11 missed | Fail |

Figures: **RF06** (Poisson checks), **fig11** (Chapman-Kolmogorov), **fig04** (Basel).

Chapman-Kolmogorov explanation for the report: "After a crash, the chance of another crash stays at
8-13% for weeks (3-9 times the base rate). An observed Markov chain on the crash flag forgets the
crash within 2-3 days, so its n-step predictions fall to the base rate. The hidden chain remembers
the regime, so its predicted probability decays slowly, at rate lambda^n, as the data do."

---

## 12. Discussion

Points to make:
1. **The Markov-chain view works.** Every market shows two persistent regimes; the theory (P^n,
   stationary distribution, geometric stays) is confirmed by simulation and roughly matches the data.
2. **Crashes are not Poisson.** Every Poisson prediction fails (dispersion, inter-arrival
   distribution, zero-crash years). Clustering comes from the regime: crashes happen almost only in
   the crisis state, so crash arrivals are a Poisson process whose rate is switched by a Markov
   chain (a Markov-modulated Poisson process). The Hawkes process is a second, self-exciting
   explanation; both fit far better than Poisson.
3. **Hidden beats observed.** The observed 0/1 crash chain has too short a memory; the hidden chain
   carries it (Chapman-Kolmogorov 94% vs 44%).
4. **What the models miss:** within-regime tails (kurtosis 4.8 vs 15.9) and long memory in
   volatility (ACF beyond 20 days). These explain the VaR coverage failures over 30+ years.
5. **Interpretation for practice (Section 13):** regime models are most useful as a risk-control
   signal.

---

## 13. Interpretation: real-world use (trading test)

Both trades use only the out-of-sample forecasts from the backtest.

**Selling options volatility** (one-month variance at the VIX level, the variance risk premium
trade; figure **fig12**). On NIFTY (India VIX, 2010-2026), selling only when the HMM-gated GARCH
forecast is below implied volatility cut the worst month from -60.7 to -10.3 vol points, the
maximum drawdown from 111.6 to 12.2, and raised the Sharpe ratio from 0.71 to 1.30 (bootstrap
p = 0.26, not significant). On 13 March 2020 India VIX was 41.2, the model forecast 50.2 and stayed
out; realised volatility was 81.6. On the S&P 500 every filter failed in March 2020 (no model
foresaw the second leg of the crash), but the GARCH-type filters made +24.3 vol points in the GFC
where always selling lost 71.7.

**Volatility targeting** (index exposure = min(1, 2% / VaR); figure fig13). The GFC loss was cut by
54-70% in all six markets that cover it (S&P 500: -14.6% vs -37.8% buy-and-hold), but the Sharpe
ratio improved in only 3 of 7 markets, never significantly.

**Takeaway sentence:** "The stochastic model's value is in risk control: knowing when the market is
in its crisis regime cuts losses roughly in half, but it does not predict returns."

---

## 14. Limitations

- Normal distributions within each regime are too thin-tailed (kurtosis 4.8 vs 15.9); no model gets
  1% coverage over 30+ years.
- Constant transition probabilities give geometric stay lengths and a single decay rate; observed
  stays are longer and volatility memory decays more slowly.
- Regime labels and the Markov-chain properties are full-sample fits; only the backtests and trading
  tests are strictly out of sample.
- The NIFTY Yahoo sample starts in 2008, so the GFC is only in its training window (the NSE data to
  1990 partly fixes this).
- The HMM VaR uses a normal approximation to the mixture, which understates VaR when the regime is
  uncertain (2.85% vs 3.99% exact on 4 Oct 2022).
- Options P&L uses VIX as the variance-swap price, not traded option quotes.

## 15. Extensions

- Student-t errors within each regime (fat tails) or more regimes chosen by BIC.
- Duration-dependent (semi-Markov) regimes for the stay-length mismatch.
- Time-varying transition probabilities driven by VIX.
- Markov-modulated Poisson process fitted directly to crash days.
- Expected-shortfall backtests (the current Basel standard).

## 16. Conclusion (draft)

> A two-state hidden Markov chain captures the main feature of market risk: long calm periods
> interrupted by shorter, much more volatile crises. Its derived properties (n-step probabilities,
> stationary distribution, stay lengths, kurtosis and volatility clustering) agree with simulation,
> and largely with seven markets' data. Crash days are clearly not a Poisson process; their
> clustering is explained by the hidden regime and by self-excitation. Out of sample, regime models
> give well-calibrated 99% VaR for NIFTY 50 and the most robust VaR across seven markets, and in
> trading they roughly halve crisis losses. Thin tails within regimes and the long memory of
> volatility remain open problems.

---

## 17. References

(Check page numbers against Google Scholar before submitting.)

1. Aït-Sahalia, Y., Cacho-Diaz, J., and Laeven, R. J. A. (2015). Modeling financial contagion using mutually exciting jump processes. *Journal of Financial Economics*, 117(3), 585-606.
2. Ang, A., and Bekaert, G. (2002). International asset allocation with regime shifts. *Review of Financial Studies*, 15(4), 1137-1187.
3. Bachelier, L. (1900). Théorie de la spéculation. *Annales scientifiques de l'École Normale Supérieure*, 17, 21-86.
4. Basel Committee on Banking Supervision (1996). *Supervisory framework for the use of "backtesting" in conjunction with the internal models approach to market risk capital requirements.* Bank for International Settlements.
5. Baum, L. E., Petrie, T., Soules, G., and Weiss, N. (1970). A maximization technique occurring in the statistical analysis of probabilistic functions of Markov chains. *Annals of Mathematical Statistics*, 41(1), 164-171.
6. Black, F., and Scholes, M. (1973). The pricing of options and corporate liabilities. *Journal of Political Economy*, 81(3), 637-654.
7. Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. *Journal of Econometrics*, 31(3), 307-327.
8. Carr, P., and Wu, L. (2009). Variance risk premiums. *Review of Financial Studies*, 22(3), 1311-1341.
9. Christoffersen, P. F. (1998). Evaluating interval forecasts. *International Economic Review*, 39(4), 841-862.
10. Cont, R. (2001). Empirical properties of asset returns: stylized facts and statistical issues. *Quantitative Finance*, 1(2), 223-236.
11. Dempster, A. P., Laird, N. M., and Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. *Journal of the Royal Statistical Society, Series B*, 39(1), 1-38.
12. Diebold, F. X., and Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business and Economic Statistics*, 13(3), 253-263.
13. Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. *Econometrica*, 50(4), 987-1007.
14. Gneiting, T. (2011). Making and evaluating point forecasts. *Journal of the American Statistical Association*, 106(494), 746-762.
15. Haas, M., Mittnik, S., and Paolella, M. S. (2004). A new approach to Markov-switching GARCH models. *Journal of Financial Econometrics*, 2(4), 493-530.
16. Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. *Econometrica*, 57(2), 357-384.
17. Hawkes, A. G. (1971). Spectra of some self-exciting and mutually exciting point processes. *Biometrika*, 58(1), 83-90.
18. Jarque, C. M., and Bera, A. K. (1987). A test for normality of observations and regression residuals. *International Statistical Review*, 55(2), 163-172.
19. Jorion, P. (2007). *Value at Risk: The New Benchmark for Managing Financial Risk* (3rd ed.). McGraw-Hill.
20. Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations. *Annals of Statistics*, 17(3), 1217-1241.
21. Kupiec, P. H. (1995). Techniques for verifying the accuracy of risk measurement models. *Journal of Derivatives*, 3(2), 73-84.
22. Ljung, G. M., and Box, G. E. P. (1978). On a measure of lack of fit in time series models. *Biometrika*, 65(2), 297-303.
23. Mandelbrot, B. (1963). The variation of certain speculative prices. *Journal of Business*, 36(4), 394-419.
24. Moreira, A., and Muir, T. (2017). Volatility-managed portfolios. *Journal of Finance*, 72(4), 1611-1644.
25. Ogata, Y. (1981). On Lewis' simulation method for point processes. *IEEE Transactions on Information Theory*, 27(1), 23-31.
26. Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition. *Proceedings of the IEEE*, 77(2), 257-286.
27. Ross, S. M. (2014). *Introduction to Probability Models* (11th ed.). Academic Press.
28. Rydén, T., Teräsvirta, T., and Åsbrink, S. (1998). Stylized facts of daily return series and the hidden Markov model. *Journal of Applied Econometrics*, 13(3), 217-244.
29. Samuelson, P. A. (1965). Rational theory of warrant pricing. *Industrial Management Review*, 6(2), 13-39.
30. Schwarz, G. (1978). Estimating the dimension of a model. *Annals of Statistics*, 6(2), 461-464.

Data: Yahoo Finance (finance.yahoo.com); National Stock Exchange of India (nseindia.com); CBOE
(cboe.com) for VIX.

---

## 18. Appendix - Code

Repository: https://github.com/nishanthreddyr24-netizen/nifty50-volatility-var

| File | What it does |
|---|---|
| `src/loaders.py`, `src/returns.py` | Download and cache prices; log returns, summary statistics |
| `notebooks/01_stylized_facts.py` | Fat tails, clustering, VIX check, figures 01-05 |
| `src/gbm.py` | GBM fit (MLE), simulation, normal VaR/ES |
| `src/garch.py` | GARCH(1,1) fit and rolling forecasts |
| `src/hmm.py` | HMM fit, own forward filter, Viterbi, BIC, VaR forecasts |
| `src/em_hmm.py` | **Our own Baum-Welch EM** |
| `src/msgarch.py`, `src/hmm_garch.py` | MS-GARCH (Hamilton filter) and HMM-gated GARCH |
| `src/var_backtest.py` | Rolling backtest, Kupiec, Christoffersen, Basel |
| `src/extremes.py` | Poisson dispersion test, Hawkes MLE, likelihood-ratio test |
| `src/simulation_study.py` | HMM parameter recovery (hmmlearn) |
| `src/report_figures.py` | Theory vs simulation, Poisson/Hawkes simulation, own-EM recovery, report figures RF01-RF13 |
| `src/chapman_kolmogorov.py` | Chapman-Kolmogorov check with block bootstrap |
| `src/multi_market.py` | Seven-market backtest |
| `src/crisis_validation.py`, `src/crisis_duration.py`, `src/regime_transitions.py` | Crisis dating, crisis length, transition matrices |
| `src/trading_applications.py` | Options and volatility-targeting tests |
| `src/extended_loaders.py`, `scripts/` | NSE data to 1990, Exp 1 and Exp 2 |
| `tests/` | Unit tests |

For the printed appendix, include `src/em_hmm.py`, the core of `src/hmm.py` (forward filter), and
`src/extremes.py`: they are short and show our own implementation of the course theory.

---

## Presentation plan (15 minutes)

| Time | Slide | Content | Figure |
|---|---|---|---|
| 0:00-1:00 | 1. Problem | "How much could we lose tomorrow, and do crashes cluster?" VaR in one sentence | RF01 |
| 1:00-2:00 | 2. Data | 7 indices, 1990-2026, NIFTY main; fat tails (kurtosis 15.9) and clustering | 01_price_and_returns |
| 2:00-3:00 | 3. Model | Hidden 2-state Markov chain; fitted P, stays 111 and 23 days, pi = (0.83, 0.17) | **RF02** |
| 3:00-4:00 | 4. Theory | P^n closed form converging to pi; geometric stays; Poisson vs Hawkes | **RF03** |
| 4:00-5:00 | 5. Methods and code | EM (own code), rolling out-of-sample backtest, tests | RF12 + short live demo |
| 5:00-6:00 | 6. Code demo | Run `python -m src.extremes` or show `src/em_hmm.py` | terminal |
| 6:00-7:00 | 7. Crashes are not Poisson | Dispersion 17.6; 44% of gaps under 5 days; Hawkes fits | **RF06**, RF07 |
| 7:00-8:00 | 8. Theory vs simulation vs data | Kurtosis and ACF: what the HMM gets right and wrong | **RF10** |
| 8:00-9:00 | 9. Real data: regimes and VaR | Crisis probability over time; COVID VaR | **RF05**, RF13 |
| 9:00-10:00 | 10. Validation | NIFTY table; seven markets; Chapman-Kolmogorov 94% vs 44% | fig01, fig11 |
| 10:00-11:00 | 11. Interpretation and conclusion | Risk control: crisis losses halved; limitations (thin tails, long memory) | fig12 |
| 11:00-12:00 | 12. Individual contributions | One line per member | |
| 12:00-15:00 | Q&A | | |

Demo commands (run before the talk; outputs take a few seconds):
```
python -m src.extremes          # Poisson test and Hawkes fit, prints in seconds
python -m pytest tests -q       # all unit tests pass
```

### Likely questions and answers

- **Why 2 regimes and not 3?** BIC prefers 2 (-28,989 vs -28,953); in the 3-state fit two states
  swap daily and have the same India VIX, so the third state is noise.
- **Is the regime really Markov?** The Chapman-Kolmogorov check on the hidden chain passes at 94% of
  horizons for crash days and 75% for the regime flags; the noisy daily flags themselves are not
  Markov (12%), which is why we use the filtered probability, not the 0/1 flag.
- **Why not just use GARCH?** GARCH's VaR is too narrow (1.58% violations on NIFTY) and its
  simulated volatility memory is too long (ACF 0.18 at lag 20 vs 0.09 in the data). The gated model
  combines both.
- **Isn't this look-ahead?** No: every forecast uses only data before its date; the regime labels
  in the descriptive sections are full-sample and are labelled as such.
- **Why does no model pass on 30 years of data?** Normal errors are too thin-tailed: HMM kurtosis 4.8
  vs 15.9. Student-t errors are the extension.
- **Did you write the EM yourselves?** Yes (`src/em_hmm.py`); it gives exactly the same likelihood
  as hmmlearn without its default prior, and it showed that prior biases the calm volatility by 3.7%.
- **How is this different from a Poisson model of crashes?** Poisson is rejected by four separate
  tests; crash arrivals behave like a Poisson process whose rate is switched by the hidden Markov chain.

