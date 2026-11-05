"""Collect every result of the project into one ordered folder: final_results/.

Nothing is recomputed here: files are copied from where each study writes them
(reports/, results/), so final_results/ is a read-only snapshot that anyone can
browse without running code. A MANIFEST.csv lists every file with its size,
source path and a one-line description.

Usage:  python -m src.collect_results
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REP = ROOT / "reports"
MM = REP / "multi_market"
FIG = MM / "figures"
SM = REP / "single_market"
SIMS = REP / "simulations"
RFIG = REP / "report_figures"
OUT = ROOT / "final_results"
REPO = "nishanthreddyr24-netizen/nifty50-volatility-var"
RAW_URL = f"https://raw.githubusercontent.com/{REPO}/master/final_results/"
WEB_URL = f"https://github.com/{REPO}/blob/master/final_results/"
MARKETS = ["sp500", "ftse100", "dax", "nikkei225", "hangseng", "sensex", "nifty50"]

# (destination folder, [(source, description)])
PLAN: list[tuple[str, list[tuple[Path, str]]]] = [
    ("00_input_data", [
        (ROOT / "data" / "raw" / "sp500_1990.csv", "S&P 500 (^GSPC) daily OHLCV, 1990-2026 (Yahoo Finance)"),
        (ROOT / "data" / "raw" / "ftse100_1990.csv", "FTSE 100 (^FTSE) daily OHLCV, 1990-2026 (Yahoo Finance)"),
        (ROOT / "data" / "raw" / "dax_1990.csv", "DAX (^GDAXI) daily OHLCV, 1990-2026 (Yahoo Finance)"),
        (ROOT / "data" / "raw" / "nikkei225_1990.csv", "Nikkei 225 (^N225) daily OHLCV, 1990-2026 (Yahoo Finance)"),
        (ROOT / "data" / "raw" / "hangseng_1990.csv", "Hang Seng (^HSI) daily OHLCV, 1990-2026 (Yahoo Finance)"),
        (ROOT / "data" / "raw" / "sensex_1997.csv", "Sensex (^BSESN) daily OHLCV, 1997-2026 (Yahoo Finance)"),
        (ROOT / "data" / "raw" / "nifty50.csv", "NIFTY 50 (^NSEI) daily OHLCV, 2008-2026 (Yahoo Finance)"),
        (ROOT / "data" / "raw" / "sp500.csv", "S&P 500 (^GSPC) daily OHLCV, 2008-2026 (Yahoo Finance; single-market study)"),
        (ROOT / "data" / "raw" / "vix_1990.csv", "CBOE VIX (^VIX) daily, 1990-2026 (Yahoo Finance)"),
        (ROOT / "data" / "raw" / "india_vix.csv", "India VIX (^INDIAVIX) daily, 2008-2026 (Yahoo Finance)"),
        (ROOT / "data" / "processed" / "nifty50_nse_1990_2026.csv", "NIFTY 50 from official NSE data, 1990-2026 (teammate's series, audited against Yahoo)"),
    ]),
    ("01_stylized_facts", [
        (SM / "stylized_facts_output.txt", "NIFTY 50 summary stats, Jarque-Bera, Ljung-Box, VIX and leverage correlations"),
        (REP / "figures" / "01_price_and_returns.png", "NIFTY 50 price and daily log returns"),
        (REP / "figures" / "02_fat_tails.png", "Return histogram vs normal, QQ plot"),
        (REP / "figures" / "03_acf_clustering.png", "ACF of returns vs squared returns"),
        (REP / "figures" / "04_vol_vs_vix.png", "21-day realised vol vs India VIX"),
        (REP / "figures" / "05_crisis_windows.png", "2008 GFC and 2020 COVID windows"),
    ]),
    ("02_nifty_model_comparison", [
        (SM / "model_comparison_output.txt", "Six-model VaR backtest on NIFTY 50 (Kupiec, Christoffersen, Basel, COVID window, regimes)"),
        (SM / "hmm_regime_output.txt", "Full-sample HMM: regimes, 2 vs 3 states (BIC), India VIX and crisis-window checks, mixture VaR"),
        (MM / "nifty50_backtest.csv", "NIFTY 50 backtest table per model (seven-market run, identical numbers)"),
        (MM / "nifty50_var_paths.csv", "NIFTY 50 daily return, VaR and hit for every model"),
    ]),
    ("03_extreme_events_poisson_hawkes", [
        (SM / "extremes_output.txt", "Crash days per year, Poisson dispersion test, Hawkes fit and LR test"),
    ]),
    ("04_hmm_simulation_study", [
        (SM / "simulation_study_output.txt", "HMM parameter recovery: 50 simulated samples refit by EM"),
    ]),
    ("05_seven_market_var_backtest", [
        (MM / "ALL_backtest.csv", "Every market x model: violations, Kupiec, Christoffersen, Basel zones, sub-periods"),
        *[(MM / f"{m}_backtest.csv", f"{m}: backtest table per model") for m in MARKETS],
        *[(MM / f"{m}_var_paths.csv", f"{m}: daily return, VaR and hit for every model") for m in MARKETS],
        (FIG / "fig01_violation_rate_by_market.png", "Violation rate by market and model"),
        (FIG / "fig02_hmm_vs_gated_by_period.png", "HMM vs HMM-gated GARCH violations by period"),
        (FIG / "fig03_sp500_gfc_var_paths.png", "S&P 500 VaR paths through the GFC"),
        (FIG / "fig04_basel_red_years.png", "Basel red-zone years by model"),
    ]),
    ("06_forecast_skill_and_runtime", [
        (MM / "ALL_forecast_skill.csv", "Tick loss, skill vs GBM, rank, Diebold-Mariano vs HMM"),
        (MM / "ALL_timings.csv", "Run time and ms per forecast, every model and market"),
        (FIG / "fig05_forecast_skill.png", "Forecast skill vs GBM"),
        (FIG / "fig10_runtime.png", "Run time per model"),
    ]),
    ("07_regimes_and_transitions", [
        (MM / "ALL_regimes.csv", "Regime summary per market: crisis share, crash days in crisis, regime vols"),
        (MM / "ALL_crisis_windows.csv", "Share of each documented crisis window flagged by HMM and MS-GARCH"),
        (MM / "ALL_transition_matrices.csv", "Fitted and empirical transition matrices, expected stays, switches per year"),
        *[(MM / f"{m}_regimes.csv", f"{m}: regime summary") for m in MARKETS],
        *[(MM / f"{m}_crisis_windows.csv", f"{m}: crisis-window coverage") for m in MARKETS],
        *[(MM / f"{m}_regime_flags.csv", f"{m}: daily crash flag, HMM P(crisis), HMM and MS-GARCH crisis flags") for m in MARKETS],
        (FIG / "fig06_transition_matrices.png", "Transition matrices, all markets"),
        (FIG / "fig07_regime_timeline.png", "Regime timeline, all markets"),
        (FIG / "fig08_crisis_days_by_market.png", "Crisis share and crash days by market"),
    ]),
    ("08_chapman_kolmogorov", [
        (MM / "CK_summary.csv", "Horizons passed out of 40 per market and model"),
        (MM / "CK_crash_days.csv", "Crash days: empirical n-step probability, bootstrap band, each model's value, pass flag"),
        (MM / "CK_regime_flags.csv", "Regime flags: empirical n-step probability vs one-step matrix power and HMM matrix power"),
        (FIG / "fig11_chapman_kolmogorov.png", "Chapman-Kolmogorov check on crash days, 7 markets"),
    ]),
    ("09_crisis_timing_and_length", [
        (MM / "VALIDATION_bear_markets.csv", "83 bear markets (20% rule) with HMM / MS-GARCH start lags"),
        (MM / "VALIDATION_overlap.csv", "Day-level overlap of crisis flags with bear markets"),
        (MM / "VALIDATION_events.csv", "Lags against dated events and NBER recession coverage"),
        (MM / "VALIDATION_crisis_duration.csv", "Crisis episode length vs bear-market length"),
        (FIG / "fig09_crisis_duration.png", "Crisis length vs bear-market length"),
    ]),
    ("10_trading_applications", [
        (MM / "TRADE_vrp_summary.csv", "Options trade (sell one-month variance at VIX): every strategy, Sharpe, drawdown, crisis P&L, pass flags"),
        (MM / "TRADE_vrp_trades.csv", "Every monthly options trade: strike, realised vol, forecasts, P&L per strategy"),
        (MM / "TRADE_voltarget_summary.csv", "Volatility targeting: every market x model, Sharpe, drawdown, budget breaches, pass flags"),
        (FIG / "fig12_vrp_trade.png", "Cumulative P&L of the options trade, S&P 500 and NIFTY 50"),
        (FIG / "fig13_vol_targeting.png", "S&P 500 volatility targeting: growth and drawdown"),
    ]),
    ("11_nse_data_experiments", [
        (ROOT / "results" / "validation_exp1_exp2.json", "Validation status of every Exp 1 / Exp 2 run"),
        (ROOT / "results" / "exp1", "Exp 1: 2008 vs 1990 training start on NSE data (GBM, GARCH, historical simulation)"),
        (ROOT / "results" / "exp2", "Exp 2: 1990 / 1996 / 2009 training start, paired McNemar comparisons"),
        (ROOT / "results" / "pilot_exp1", "Pilot run of Exp 1 (GBM, historical simulation)"),
        (REP / "figures" / "exp1_exp2", "Exp 1 / Exp 2 figures (VaR timelines, exception rates, loss heatmaps, bootstrap intervals) and visualization_metrics.csv"),
        (REP / "audits", "NSE vs Yahoo data audits, weekend-date audit, calendar preflight"),
        (REP / "reproducibility", "Loss comparison between Exp 1 and Exp 2"),
    ]),
    ("13_report_figures", [
        *[(REP / "report_figures" / f, d) for f, d in [
            ("RF01_pipeline.png", "Report figure: the modelling pipeline mapped to the project"),
            ("RF02_state_transition_diagram.png", "Report figure: 2-state Markov chain diagram (NIFTY 50)"),
            ("RF03_transition_matrix_and_nstep.png", "Report figure: transition heatmap; P^n theory vs 10,000 simulated chains"),
            ("RF04_sojourn_times.png", "Report figure: stay lengths, geometric theory vs simulation vs observed"),
            ("RF05_regime_probability_nifty.png", "Report figure: NIFTY price, filtered P(crisis), India VIX"),
            ("RF06_poisson_checks.png", "Report figure: Poisson checks (counting process, inter-arrival survival, yearly counts)"),
            ("RF07_poisson_vs_hawkes_simulation.png", "Report figure: 1,000 Poisson vs 1,000 Hawkes simulations; Hawkes intensity"),
            ("RF08_simulated_sample_paths.png", "Report figure: actual returns vs simulated GBM, GARCH, HMM paths"),
            ("RF09_simulated_price_paths.png", "Report figure: 500 simulated price paths, GBM vs HMM, vs actual"),
            ("RF10_theory_vs_simulation_vs_data.png", "Report figure: ACF of squared returns and density, theory vs simulation vs data"),
            ("RF11_parameter_recovery.png", "Report figure: parameter recovery with our own EM, 100 samples"),
            ("RF12_em_convergence.png", "Report figure: our EM log-likelihood by iteration vs hmmlearn"),
            ("RF13_nifty_covid_var.png", "Report figure: NIFTY COVID crash, VaR forecasts and violations"),
            ("report_numbers.json", "Every theory, simulation and estimation number behind RF01-RF13"),
            ("simulation_stats.csv", "Kurtosis and ACF of squared returns: data, GBM, GARCH, HMM simulations, HMM theory"),
            ("parameter_recovery_own_em.csv", "Parameter recovery with our own EM: true value, mean, std, bias"),
        ]],
        *[(FIG / f, "Report figure (copy): " + f) for f in [
            "fig01_violation_rate_by_market.png", "fig02_hmm_vs_gated_by_period.png", "fig03_sp500_gfc_var_paths.png",
            "fig04_basel_red_years.png", "fig05_forecast_skill.png", "fig06_transition_matrices.png",
            "fig07_regime_timeline.png", "fig09_crisis_duration.png", "fig11_chapman_kolmogorov.png",
            "fig12_vrp_trade.png", "fig13_vol_targeting.png"]],
        *[(REP / "figures" / f, "Report figure (copy): " + f) for f in [
            "01_price_and_returns.png", "02_fat_tails.png", "04_vol_vs_vix.png"]],
    ]),
    ("14_simulations", [
        (REP / "simulations" / "README.md", "What each simulation answers, its data files, theory and results"),
    ]),
    ("14_simulations/markov_chain", [
        (SIMS / "markov_chain" / "nstep_theory_vs_simulation.csv", "P^n theory vs 10,000 simulated chains, n = 0-250"),
        (SIMS / "markov_chain" / "sojourn_theory_vs_simulation.csv", "Stay-length probabilities: geometric theory vs 2,000,000-day simulated chain"),
        (SIMS / "markov_chain" / "observed_stays_viterbi.csv", "Observed calm and crisis stays on the NIFTY Viterbi path"),
        (RFIG / "RF02_state_transition_diagram.png", "State-transition diagram"),
        (RFIG / "RF03_transition_matrix_and_nstep.png", "Transition heatmap; n-step theory vs simulation"),
        (RFIG / "RF04_sojourn_times.png", "Stay lengths: theory vs simulation vs observed"),
    ]),
    ("14_simulations/poisson_hawkes", [
        (SIMS / "poisson_hawkes" / "observed_crash_days.csv", "The 78 NIFTY crash days"),
        (SIMS / "poisson_hawkes" / "observed_interarrival_gaps.csv", "Gaps between crash days and exponential survival"),
        (SIMS / "poisson_hawkes" / "observed_yearly_counts.csv", "Crash days per year and Poisson probabilities"),
        (SIMS / "poisson_hawkes" / "simulated_paths_summary.csv", "1,000 Poisson and 1,000 Hawkes paths: event count and dispersion"),
        (SIMS / "poisson_hawkes" / "simulated_event_times_first20_paths.csv", "Event times of the first 20 simulated paths of each model"),
        (RFIG / "RF06_poisson_checks.png", "Poisson checks"),
        (RFIG / "RF07_poisson_vs_hawkes_simulation.png", "Poisson vs Hawkes simulations"),
    ]),
    ("14_simulations/return_paths", [
        (SIMS / "return_paths" / "simulated_return_paths.csv", "Actual returns and one simulated path each of GBM, GARCH, HMM"),
        (SIMS / "return_paths" / "simulated_price_quantiles_500_paths.csv", "Quantiles of 500 GBM and 500 HMM price paths vs actual"),
        (SIMS / "return_paths" / "simulation_stats.csv", "Kurtosis and ACF of squared returns: data, simulations, HMM theory"),
        (SIMS / "return_paths" / "acf_squared_returns.csv", "ACF of squared returns, lags 1-100: data, theory, simulations"),
        (RFIG / "RF08_simulated_sample_paths.png", "Simulated return paths"),
        (RFIG / "RF09_simulated_price_paths.png", "Simulated price paths"),
        (RFIG / "RF10_theory_vs_simulation_vs_data.png", "Theory vs simulation vs data"),
    ]),
    ("14_simulations/parameter_recovery", [
        (SIMS / "parameter_recovery" / "own_em_estimates_100_replications.csv", "Our EM's estimates on 100 simulated samples"),
        (SIMS / "parameter_recovery" / "parameter_recovery_own_em_summary.csv", "Bias summary per parameter"),
        (SIMS / "parameter_recovery" / "em_loglikelihood_by_iteration.csv", "Our EM's log-likelihood by iteration"),
        (SM / "simulation_study_output.txt", "Earlier 50-replication study with hmmlearn defaults"),
        (RFIG / "RF11_parameter_recovery.png", "Parameter recovery histograms"),
        (RFIG / "RF12_em_convergence.png", "EM convergence vs hmmlearn"),
    ]),
    ("12_run_logs", [
        *[(MM / f"{m}_log.txt", f"{m}: run log of the seven-market backtest") for m in MARKETS],
        *[(MM / f"{m}_stdout.txt", f"{m}: console output of the seven-market backtest") for m in MARKETS],
        (MM / "AGGREGATE_stdout.txt", "Console output of the cross-market aggregation"),
        (MM / "DURATION_stdout.txt", "Console output of the crisis-length study"),
        (MM / "TRANSITIONS_stdout.txt", "Console output of the transition-matrix study"),
        (SM / "pytest_output.txt", "Unit test run"),
    ]),
]


def write_csv_index(manifest: pd.DataFrame) -> None:
    """CSV_INDEX.md: every CSV with a GitHub link, a direct-download link and its size."""
    csv = manifest[manifest["file"].str.endswith(".csv")]
    lines = [
        "# Every CSV, with direct download links",
        "",
        "Generated by `python -m src.collect_results`. Load any file straight into pandas, no clone needed:",
        "",
        "```python",
        "import pandas as pd",
        f'df = pd.read_csv("{RAW_URL}05_seven_market_var_backtest/ALL_backtest.csv")',
        "```",
        "",
        f"{len(csv)} CSV files. Rows exclude the header.",
        "",
    ]
    for folder, g in csv.groupby(csv["file"].str.split("/").str[0], sort=True):
        lines += [f"## {folder}", "", "| File | Download | Rows | Description |", "|---|---|---|---|"]
        for _, r in g.iterrows():
            with open(OUT / r["file"], encoding="utf-8", errors="ignore") as fh:
                n = sum(1 for _ in fh) - 1
            name = r["file"].split("/", 1)[1]
            lines.append(f"| [{name}]({WEB_URL}{r['file']}) | [raw]({RAW_URL}{r['file']}) | {n:,} | {r['description']} |")
        lines.append("")
    (OUT / "CSV_INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if OUT.exists():
        for child in OUT.iterdir():
            if child.name != "README.md":   # the hand-written README is kept (CSV_INDEX.md is regenerated)
                shutil.rmtree(child) if child.is_dir() else child.unlink()
    OUT.mkdir(exist_ok=True)
    rows, missing = [], []
    for folder, items in PLAN:
        dest_dir = OUT / folder
        dest_dir.mkdir(parents=True, exist_ok=True)
        for src, desc in items:
            if not src.exists():
                missing.append(str(src.relative_to(ROOT)))
                continue
            dest = dest_dir / src.name
            if src.is_dir():
                shutil.copytree(src, dest)
                files = [p for p in dest.rglob("*") if p.is_file()]
            else:
                shutil.copy2(src, dest)
                files = [dest]
            for f in files:
                rel_in_src = f.relative_to(dest) if src.is_dir() else None
                rows.append({"file": f.relative_to(OUT).as_posix(),
                             "size_kb": round(f.stat().st_size / 1024, 1),
                             "source": (src / rel_in_src if rel_in_src else src).relative_to(ROOT).as_posix(),
                             "description": desc})
    manifest = pd.DataFrame(rows)
    manifest.to_csv(OUT / "MANIFEST.csv", index=False)
    write_csv_index(manifest)
    print(f"{len(rows)} files copied into {OUT.relative_to(ROOT)}/")
    if missing:
        print("MISSING:", *missing, sep="\n  ")


if __name__ == "__main__":
    main()
