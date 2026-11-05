# Simulations

Every simulation in the project, with the simulated data, the figures and the theory each one is
checked against. Nothing needs to be rerun: all files here are outputs.

| Sub-folder | Question the simulation answers | Main result |
|---|---|---|
| [`markov_chain/`](markov_chain/) | Do the derived Markov-chain formulas (n-step probabilities, stationary distribution, stay lengths) hold? | Simulation matches theory to within 0.011; observed stays are longer than the theory |
| [`poisson_hawkes/`](poisson_hawkes/) | Can a Poisson process, or a Hawkes process, produce the observed clustering of crash days? | Poisson: 0 of 1,000 paths; Hawkes: 118 of 1,000 |
| [`return_paths/`](return_paths/) | What do GBM, GARCH and HMM return paths look like, and do they reproduce fat tails and volatility clustering? | HMM theory = simulation; HMM matches clustering up to ~20 days but not the full fat tails |
| [`parameter_recovery/`](parameter_recovery/) | Does our EM estimator recover the true parameters? | No significant bias on any of 6 parameters (100 replications) |

All simulations use the 2-state HMM, GARCH(1,1), GBM, Poisson and Hawkes models fitted to
**NIFTY 50 daily log returns, 2008-01-01 to 2026-10-23 (4,613 days)**. Random seed 2026
(`numpy.random.default_rng`), so every number is reproducible.

**Code:** the simulators and theory formulas are in [`src/simulators.py`](../../src/simulators.py);
the runs that produced these files are in [`src/report_figures.py`](../../src/report_figures.py);
our EM estimator is [`src/em_hmm.py`](../../src/em_hmm.py). Rerun everything with
`python -m src.report_figures` (about 5 minutes).

**Fitted model used as "truth"** (NIFTY 50): calm volatility 0.844%/day, crisis 2.522%/day;
P(calm to crisis) = 0.0090, P(crisis to calm) = 0.0438; stationary crisis share 0.170; expected stays
111 and 23 trading days. GARCH(1,1): alpha = 0.092, beta = 0.899. GBM: daily mean 0.0282%,
volatility 1.279%. Hawkes (calendar days): mu = 0.00271, alpha = 0.0288, beta = 0.0379.

---

## 1. `markov_chain/`: the regime chain

| File | Contents |
|---|---|
| `nstep_theory_vs_simulation.csv` | n = 0-250 days ahead: closed-form P^n (theory) and the share of 10,000 simulated chains in crisis, starting from crisis and from calm; stationary share |
| `sojourn_theory_vs_simulation.csv` | Stay length 1-399 days, per regime: geometric probability (theory) and frequency in one simulated chain of 2,000,000 days |
| `observed_stays_viterbi.csv` | Every observed calm and crisis stay on the NIFTY Viterbi path (21 calm, 20 crisis) |
| `RF02_state_transition_diagram.png` | The chain with its probabilities, mean stays and long-run shares |
| `RF03_transition_matrix_and_nstep.png` | Heatmap of P; P^n theory vs simulation converging to the stationary distribution |
| `RF04_sojourn_times.png` | Stay-length distributions: theory vs simulation vs observed |

Theory: $P^n = \frac{1}{a+b}\begin{pmatrix} b & a \\ b & a \end{pmatrix} + \frac{(1-a-b)^n}{a+b}\begin{pmatrix} a & -a \\ -b & b \end{pmatrix}$,
stationary $\pi = (b, a)/(a+b)$, stays geometric with mean $1/(1-p_{ii})$.

| Check | Theory | Simulation | Observed |
|---|---|---|---|
| P(crisis in 10 days \| crisis now) | 0.652 | within 0.011 | |
| P(crisis in 50 days \| crisis now) | 0.225 | within 0.011 | |
| Long-run crisis share | 0.170 | 0.171 (2,000,000 days) | 0.150 (filtered flags) |
| Mean calm stay | 111.1 days | 111.3 | 184 (Viterbi, 21 stays) |
| Mean crisis stay | 22.8 days | 23.0 | 37 (Viterbi, 20 stays) |

Reading: the formulas are confirmed. Real stays are longer and more varied than a constant-probability
chain allows (a known limitation; semi-Markov regimes are the extension).

---

## 2. `poisson_hawkes/`: crash days (log return <= -3%)

| File | Contents |
|---|---|
| `observed_crash_days.csv` | The 78 NIFTY crash days with their returns |
| `observed_interarrival_gaps.csv` | The 77 gaps between crash days (trading days) and the exponential survival probability at each |
| `observed_yearly_counts.csv` | Crash days per calendar year (19 years) and the Poisson probability of each count |
| `simulated_paths_summary.csv` | 1,000 Poisson and 1,000 Hawkes paths over the same period: number of events and dispersion index of yearly counts per path |
| `simulated_event_times_first20_paths.csv` | Event times (calendar day and date) of the first 20 simulated paths of each model |
| `RF06_poisson_checks.png` | Counting process N(t), inter-arrival survival function, yearly counts vs Poisson |
| `RF07_poisson_vs_hawkes_simulation.png` | Dispersion index across simulations vs observed; fitted Hawkes intensity |

| Check | Poisson theory | Observed | Hawkes simulation |
|---|---|---|---|
| Variance / mean of yearly counts | 1 (simulated 95%: 0.46-1.70) | **17.6** | 95%: 2.14-29.6 |
| Paths at least as clustered as observed | 0 of 1,000 | | 118 of 1,000 |
| Mean gap between crashes | 59.1 days | 57.8 | |
| Median gap | 41.0 days | **7** | |
| Gaps of 5 days or less | 8.1% | **44.2%** | |
| Years with no crash | 0.31 expected | **6** | |
| Expected number of crashes | 78 (fitted) | 78 | mu T / (1 - n) = 78.0 |

Poisson was simulated by drawing N ~ Poisson(lambda T) and placing the events uniformly; Hawkes by
Ogata thinning. Reading: a Poisson process essentially never produces the observed clustering; the
fitted Hawkes process does.

---

## 3. `return_paths/`: simulated returns and prices

| File | Contents |
|---|---|
| `simulated_return_paths.csv` | 4,613 days: actual NIFTY returns and one simulated path each from GBM, GARCH(1,1) and the HMM (with its hidden state) |
| `simulated_price_quantiles_500_paths.csv` | Daily 5/25/50/75/95% quantiles of 500 simulated GBM and 500 HMM price paths, and the actual NIFTY level |
| `simulation_stats.csv` | Excess kurtosis and autocorrelation of squared returns: data, 300 simulated paths per model (mean and 95% range), HMM theory |
| `acf_squared_returns.csv` | ACF of squared returns at lags 1-100: data, HMM theory, and simulation means for HMM, GARCH, GBM |
| `RF08_simulated_sample_paths.png` | Actual returns vs one path of each model |
| `RF09_simulated_price_paths.png` | 500 price paths per model vs actual |
| `RF10_theory_vs_simulation_vs_data.png` | ACF of squared returns and return density: theory vs simulation vs data |

| Statistic | NIFTY actual | GBM sim. | GARCH sim. | HMM theory | HMM sim. |
|---|---|---|---|---|---|
| Excess kurtosis | **15.9** | 0.0 | 3.1 | 4.83 | 4.81 |
| ACF of r^2, lag 1 | **0.161** | 0.000 | 0.247 | 0.223 | 0.225 |
| ACF of r^2, lag 20 | **0.086** | 0.000 | 0.181 | 0.079 | 0.077 |

Reading: the HMM formulas are confirmed by simulation. The HMM reproduces volatility clustering up
to about 20 days and much of the fat tails, but not all (4.8 vs 15.9); GARCH's clustering lasts too
long; GBM has neither.

---

## 4. `parameter_recovery/`: is the estimator trustworthy?

| File | Contents |
|---|---|
| `own_em_estimates_100_replications.csv` | Our EM's 6 estimates on each of 100 simulated NIFTY-length samples |
| `parameter_recovery_own_em_summary.csv` | True value, mean estimate, standard deviation, bias and bias t-statistic per parameter |
| `em_loglikelihood_by_iteration.csv` | Our EM's log-likelihood at each iteration on the real data (46 iterations, never decreasing) |
| `simulation_study_output.txt` | The earlier 50-replication study with hmmlearn's default settings |
| `RF11_parameter_recovery.png` | Histograms of the 100 estimates vs the true values |
| `RF12_em_convergence.png` | EM convergence, our code vs hmmlearn |

| Parameter | True | Mean of 100 estimates | Bias t-statistic |
|---|---|---|---|
| Calm mean | 0.000594 | 0.000596 | 0.15 |
| Crisis mean | -0.001090 | -0.001208 | -1.21 |
| Calm volatility | 0.008143 | 0.008145 | 0.25 |
| Crisis volatility | 0.024264 | 0.024289 | 0.41 |
| P(calm to crisis) | 0.01007 | 0.01025 | 1.02 |
| P(crisis to calm) | 0.04429 | 0.04502 | 0.83 |

Reading: no significant bias (all |t| < 2). The earlier hmmlearn study found +2% bias in calm
volatility; that came from hmmlearn's default variance prior (`covars_prior = 0.01`), not from EM.
Without the prior, hmmlearn and our EM reach the same maximum log-likelihood, 14,526.715.
