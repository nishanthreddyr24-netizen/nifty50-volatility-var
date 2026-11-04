"""Generate publication-style visualisations for NIFTY VaR Experiments 1 and 2.

Reads existing forecast CSVs only. It does not fit models or modify result files.

Expected input layout:
    results/exp1/<variant>/{gbm,garch,historical_simulation}.csv
    results/exp2/<variant>/{gbm,garch,historical_simulation}.csv

Run from the project root (or invoke this script by its absolute path):
    python scripts/visualize_exp1_exp2.py

The script writes PNG figures and a compact metrics CSV under:
    reports/figures/exp1_exp2/

Dependencies: numpy, pandas, matplotlib.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Safe for PowerShell / non-interactive execution.
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TAU = 0.01                         # 1% lower-tail quantile (99% VaR)
ROLLING_WINDOW = 250               # Approximately one trading year
BOOTSTRAP_REPS = 2_000
BOOTSTRAP_BLOCK_LENGTHS = (10, 20, 40)
BOOTSTRAP_BATCH_SIZE = 150         # Bound memory use during vectorised bootstrap
RANDOM_SEED = 20261102

EXPERIMENTS = {
    "exp1": {
        "title": "Experiment 1 — Historical data extension",
        "variants": ["BASELINE_CONFIGURED_2008", "EXTENDED_1990"],
        "expected_n": 4163,
        "expected_first": "2010-01-20",
    },
    "exp2": {
        "title": "Experiment 2 — Historical sample-start sensitivity",
        "variants": ["PRELAUNCH_1990", "LAUNCHED_1996", "FREEFLOAT_2009"],
        "expected_n": 3802,
        "expected_first": "2011-06-28",
    },
}
MODELS = ["gbm", "garch", "historical_simulation"]
MODEL_LABELS = {
    "gbm": "GBM",
    "garch": "GARCH(1,1)",
    "historical_simulation": "Historical Simulation",
}
REQUIRED_COLUMNS = {"date", "return", "var", "es", "hit"}


# ---------------------------------------------------------------------------
# Input loading and checks
# ---------------------------------------------------------------------------
def pinball_loss(returns: np.ndarray, var: np.ndarray) -> np.ndarray:
    """Pinball loss for the lower 1% return quantile q = -VaR."""
    q = -np.asarray(var, dtype=float)
    r = np.asarray(returns, dtype=float)
    return np.where(r <= q, (1.0 - TAU) * (q - r), TAU * (r - q))


def fz0_loss(returns: np.ndarray, var: np.ndarray, es: np.ndarray) -> np.ndarray:
    """Patton-Ziegel-Chen FZ0 score for lower-tail VaR and ES.

    The stored VaR and ES are positive loss magnitudes. Convert them to the
    corresponding negative return thresholds before applying the FZ0 formula.
    Lower (more negative) scores are better under this convention.
    """
    r = np.asarray(returns, dtype=float)
    v = -np.asarray(var, dtype=float)
    e = -np.asarray(es, dtype=float)
    if np.any(e >= 0):
        raise ValueError("FZ0 requires negative lower-tail ES on the return scale")
    return np.where(r <= v, (r - v) / (TAU * e), 0.0) + v / e + np.log(-e) - 1.0


def load_all_runs(root: Path) -> dict[tuple[str, str, str], pd.DataFrame]:
    runs: dict[tuple[str, str, str], pd.DataFrame] = {}
    for exp, cfg in EXPERIMENTS.items():
        expected_n = cfg["expected_n"]
        expected_first = pd.Timestamp(cfg["expected_first"])
        for variant in cfg["variants"]:
            for model in MODELS:
                path = root / "results" / exp / variant / f"{model}.csv"
                if not path.exists():
                    raise FileNotFoundError(
                        f"Missing input CSV: {path}\n"
                        "Run the experiment runner first; this script does not create forecasts."
                    )
                table = pd.read_csv(path)  # default parser avoids deprecated float_precision
                missing = REQUIRED_COLUMNS.difference(table.columns)
                if missing:
                    raise ValueError(f"{path} is missing columns: {sorted(missing)}")
                table["date"] = pd.to_datetime(table["date"], errors="raise")
                for col in ("return", "var", "es"):
                    table[col] = pd.to_numeric(table[col], errors="raise")
                table["hit"] = pd.to_numeric(table["hit"], errors="raise").astype(int)

                if len(table) != expected_n:
                    raise ValueError(f"{path}: expected {expected_n} rows, found {len(table)}")
                if table["date"].iloc[0] != expected_first:
                    raise ValueError(
                        f"{path}: expected first date {expected_first.date()}, "
                        f"found {table['date'].iloc[0].date()}"
                    )
                if table["date"].duplicated().any() or not table["date"].is_monotonic_increasing:
                    raise ValueError(f"{path}: dates must be unique and ascending")

                values = table[["return", "var", "es"]].to_numpy(dtype=float)
                if not np.isfinite(values).all():
                    raise ValueError(f"{path}: non-finite return, VaR, or ES values")
                if (table["var"] <= 0).any() or (table["es"] <= 0).any():
                    raise ValueError(f"{path}: VaR and ES must be positive loss magnitudes")
                if (table["es"] + 1e-12 < table["var"]).any():
                    raise ValueError(f"{path}: ES must be greater than or equal to VaR")

                expected_hit = (-table["return"].to_numpy() > table["var"].to_numpy()).astype(int)
                if not np.array_equal(expected_hit, table["hit"].to_numpy()):
                    raise ValueError(f"{path}: hit values do not match (-return > VaR)")

                table["pinball"] = pinball_loss(table["return"].to_numpy(), table["var"].to_numpy())
                table["fz0"] = fz0_loss(
                    table["return"].to_numpy(), table["var"].to_numpy(), table["es"].to_numpy()
                )
                table["year"] = table["date"].dt.year
                table["month"] = table["date"].dt.to_period("M").dt.to_timestamp()
                runs[(exp, variant, model)] = table

        # All variants must be evaluated on exactly the same dates and returns.
        reference = runs[(exp, cfg["variants"][0], MODELS[0])]
        for variant in cfg["variants"][1:]:
            candidate = runs[(exp, variant, MODELS[0])]
            if not reference["date"].equals(candidate["date"]):
                raise ValueError(f"{exp}: forecast dates do not align for {variant}")
            if not np.allclose(reference["return"], candidate["return"], rtol=0, atol=1e-15):
                raise ValueError(f"{exp}: realised returns do not align for {variant}")

        # The HS model should be identical across historical-start variants on common dates.
        hs_ref = runs[(exp, cfg["variants"][0], "historical_simulation")]
        for variant in cfg["variants"][1:]:
            hs = runs[(exp, variant, "historical_simulation")]
            for col in ("var", "es"):
                if not np.allclose(hs_ref[col], hs[col], rtol=0, atol=1e-12):
                    raise ValueError(f"{exp}: HS {col} differs across variants; inspect outputs")

    return runs


def save_figure(fig: plt.Figure, out_dir: Path, filename: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {path.relative_to(out_dir.parents[1])}")


# ---------------------------------------------------------------------------
# 1. VaR timeline with observed breaches
# ---------------------------------------------------------------------------
def plot_var_timeline(exp: str, runs: dict, out_dir: Path) -> None:
    cfg = EXPERIMENTS[exp]
    fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)
    for ax, model in zip(axes, MODELS):
        first = runs[(exp, cfg["variants"][0], model)]
        dates = first["date"]
        ax.plot(dates, 100 * first["return"], color="0.35", lw=0.45, alpha=0.65,
                label="Realised return", zorder=1)
        for variant in cfg["variants"]:
            tab = runs[(exp, variant, model)]
            line, = ax.plot(tab["date"], -100 * tab["var"], lw=0.95,
                            label=f"{variant} VaR threshold")
            breach = tab["hit"].to_numpy(dtype=bool)
            ax.scatter(tab.loc[breach, "date"], 100 * tab.loc[breach, "return"],
                       s=15, marker="x", linewidths=0.9, color=line.get_color(),
                       label=f"{variant} breaches", zorder=3)
        ax.axhline(0, color="black", lw=0.55, alpha=0.65)
        ax.set_ylabel("Daily return (%)")
        ax.set_title(MODEL_LABELS[model], loc="left", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.22)
        ax.legend(loc="lower left", fontsize=6.8, ncol=2, frameon=False)
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].set_xlabel("Forecast date")
    fig.suptitle(f"{cfg['title']} — realised returns, VaR thresholds and breaches", y=1.01,
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, out_dir, f"{exp}_01_var_timeline.png")


# ---------------------------------------------------------------------------
# 2. Rolling exception rates
# ---------------------------------------------------------------------------
def plot_rolling_coverage(exp: str, runs: dict, out_dir: Path) -> None:
    cfg = EXPERIMENTS[exp]
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for ax, model in zip(axes, MODELS):
        for variant in cfg["variants"]:
            tab = runs[(exp, variant, model)]
            rolling = tab["hit"].rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW).mean() * 100
            ax.plot(tab["date"], rolling, lw=1.15, label=variant)
        ax.axhline(1.0, color="black", ls="--", lw=1.0, label="Nominal 1%")
        ax.set_ylabel("Exceptions (%)")
        ax.set_title(f"{MODEL_LABELS[model]} — {ROLLING_WINDOW}-forecast rolling window",
                     loc="left", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.22)
        ax.legend(fontsize=7, ncol=2, frameon=False)
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].set_xlabel("Forecast date")
    fig.suptitle(f"{cfg['title']} — rolling VaR exception rate", y=1.01,
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, out_dir, f"{exp}_02_rolling_exception_rate.png")


# ---------------------------------------------------------------------------
# 3. Annual exception rates
# ---------------------------------------------------------------------------
def plot_annual_coverage(exp: str, runs: dict, out_dir: Path) -> None:
    cfg = EXPERIMENTS[exp]
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    for ax, model in zip(axes, MODELS):
        for variant in cfg["variants"]:
            tab = runs[(exp, variant, model)]
            annual = tab.groupby("year")["hit"].mean() * 100
            ax.plot(annual.index, annual.values, marker="o", ms=4, lw=1.2, label=variant)
        ax.axhline(1.0, color="black", ls="--", lw=1.0, label="Nominal 1%")
        ax.set_ylabel("Exceptions (%)")
        ax.set_title(MODEL_LABELS[model], loc="left", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.22)
        ax.legend(fontsize=7, ncol=2, frameon=False)
    axes[-1].set_xlabel("Calendar year")
    axes[-1].set_xticks(sorted({
        int(y) for variant in cfg["variants"]
        for y in runs[(exp, variant, MODELS[0])]["year"].unique()
    })[::2])
    fig.suptitle(f"{cfg['title']} — annual exception rates (target = 1%)", y=1.01,
                 fontsize=14, fontweight="bold")
    fig.text(0.5, 0.005, "Note: 2026 is a partial year; data end on 30 October 2026.",
             ha="center", fontsize=8, color="0.35")
    fig.tight_layout(rect=(0, 0.025, 1, 0.99))
    save_figure(fig, out_dir, f"{exp}_03_annual_exception_rate.png")


# ---------------------------------------------------------------------------
# 4. Risk-efficiency / calibration scatter
# ---------------------------------------------------------------------------
def plot_risk_efficiency(exp: str, runs: dict, out_dir: Path) -> None:
    cfg = EXPERIMENTS[exp]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    markers = ["o", "s", "^", "D"]
    for ax, model in zip(axes, MODELS):
        for i, variant in enumerate(cfg["variants"]):
            tab = runs[(exp, variant, model)]
            mean_var = 100 * tab["var"].mean()
            exception_rate = 100 * tab["hit"].mean()
            ax.scatter(mean_var, exception_rate, s=85, marker=markers[i],
                       edgecolor="black", linewidth=0.5, label=variant, zorder=3)
            ax.annotate(variant.replace("_", "\n"), (mean_var, exception_rate),
                        xytext=(5, 5), textcoords="offset points", fontsize=6.8)
        ax.axhline(1.0, color="black", ls="--", lw=1.0, label="Target exception rate")
        ax.set_title(MODEL_LABELS[model], fontweight="bold")
        ax.set_xlabel("Mean forecast VaR (%)")
        ax.set_ylabel("Observed exception rate (%)")
        ax.grid(True, alpha=0.22)
        ax.legend(fontsize=6.6, frameon=False)
    fig.suptitle(f"{cfg['title']} — risk level versus realised coverage", y=1.02,
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, out_dir, f"{exp}_04_risk_efficiency_scatter.png")


# ---------------------------------------------------------------------------
# 5. Pinball and FZ0 loss heatmaps
# ---------------------------------------------------------------------------
def plot_loss_heatmaps(exp: str, runs: dict, out_dir: Path) -> None:
    cfg = EXPERIMENTS[exp]
    variants = cfg["variants"]
    pin = np.array([[runs[(exp, v, m)]["pinball"].mean() * 1e4 for v in variants]
                    for m in MODELS])
    fz = np.array([[runs[(exp, v, m)]["fz0"].mean() for v in variants]
                   for m in MODELS])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, matrix, title, fmt in (
        (axes[0], pin, "Mean pinball loss (×10⁻⁴)", ".4f"),
        (axes[1], fz, "Mean FZ0 score", ".4f"),
    ):
        image = ax.imshow(matrix, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(variants)), [v.replace("_", "\n") for v in variants], fontsize=8)
        ax.set_yticks(range(len(MODELS)), [MODEL_LABELS[m] for m in MODELS])
        ax.set_title(title, fontweight="bold")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, format(matrix[i, j], fmt), ha="center", va="center",
                        color="white" if matrix[i, j] < np.nanmedian(matrix) else "black",
                        fontsize=8)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"{cfg['title']} — forecast scoring rules (lower is better)", y=1.02,
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, out_dir, f"{exp}_05_loss_heatmaps.png")


# ---------------------------------------------------------------------------
# 6. Distribution of VaR and ES forecasts
# ---------------------------------------------------------------------------
def plot_var_es_distributions(exp: str, runs: dict, out_dir: Path) -> None:
    cfg = EXPERIMENTS[exp]
    variants = cfg["variants"]
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    for row, model in enumerate(MODELS):
        for col, field in enumerate(("var", "es")):
            data = [100 * runs[(exp, v, model)][field].to_numpy() for v in variants]
            axes[row, col].boxplot(data, tick_labels=[v.replace("_", "\n") for v in variants],
                                   showfliers=False, patch_artist=False, widths=0.58)
            axes[row, col].set_title(
                f"{MODEL_LABELS[model]} — {field.upper()} distribution",
                loc="left", fontsize=9.5, fontweight="bold"
            )
            axes[row, col].set_ylabel("Forecast magnitude (% return)")
            axes[row, col].grid(True, axis="y", alpha=0.22)
            axes[row, col].tick_params(axis="x", labelsize=7)
    fig.suptitle(f"{cfg['title']} — distributions of daily VaR and ES forecasts", y=1.01,
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, out_dir, f"{exp}_06_var_es_distributions.png")


# ---------------------------------------------------------------------------
# 7. Cumulative paired loss differences over time
# ---------------------------------------------------------------------------
def history_pairs(exp: str) -> list[tuple[str, str]]:
    variants = EXPERIMENTS[exp]["variants"]
    if exp == "exp1":
        return [("EXTENDED_1990", "BASELINE_CONFIGURED_2008")]
    return [
        ("PRELAUNCH_1990", "LAUNCHED_1996"),
        ("PRELAUNCH_1990", "FREEFLOAT_2009"),
        ("LAUNCHED_1996", "FREEFLOAT_2009"),
    ]


def plot_cumulative_loss_differences(exp: str, runs: dict, out_dir: Path) -> None:
    cfg = EXPERIMENTS[exp]
    pairs = history_pairs(exp)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    metrics = [("pinball", "Cumulative pinball difference (×10⁻⁴)", 1e4),
               ("fz0", "Cumulative FZ0 difference", 1.0)]
    for col, (metric, title, scale) in enumerate(metrics):
        for row, model in enumerate(("gbm", "garch")):
            ax = axes[row, col]
            for a, b in pairs:
                left, right = runs[(exp, a, model)], runs[(exp, b, model)]
                if not left["date"].equals(right["date"]):
                    raise ValueError(f"{exp}/{model}: dates do not align for {a} versus {b}")
                diff = (left[metric].to_numpy() - right[metric].to_numpy()) * scale
                ax.plot(left["date"], np.cumsum(diff), lw=1.25, label=f"{a} − {b}")
            ax.axhline(0, color="black", lw=0.8, ls="--")
            ax.set_title(f"{MODEL_LABELS[model]} — {title}", loc="left", fontsize=9.5,
                         fontweight="bold")
            ax.set_ylabel("Cumulative difference")
            ax.grid(True, alpha=0.22)
            ax.legend(fontsize=6.8, frameon=False)
    for ax in axes[-1, :]:
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.set_xlabel("Forecast date")
    fig.suptitle(f"{cfg['title']} — when forecast-loss differences accumulate", y=1.01,
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, out_dir, f"{exp}_07_cumulative_loss_differences.png")


# ---------------------------------------------------------------------------
# 8. Paired stationary-bootstrap confidence intervals
# ---------------------------------------------------------------------------
def stationary_bootstrap_means(
    differences: np.ndarray,
    mean_block: int,
    reps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Circular stationary-bootstrap means, generated in memory-bounded batches."""
    d = np.asarray(differences, dtype=float)
    n = len(d)
    if n < 2:
        raise ValueError("At least two paired observations are required")
    if mean_block < 1:
        raise ValueError("mean_block must be >= 1")
    out = np.empty(reps, dtype=float)
    positions = np.arange(n)[None, :]
    for first in range(0, reps, BOOTSTRAP_BATCH_SIZE):
        count = min(BOOTSTRAP_BATCH_SIZE, reps - first)
        starts = rng.integers(0, n, size=(count, n))
        restart = rng.random((count, n)) < (1.0 / mean_block)
        restart[:, 0] = True
        last_restart = np.maximum.accumulate(
            np.where(restart, np.arange(n)[None, :], 0), axis=1
        )
        segment_start = np.take_along_axis(starts, last_restart, axis=1)
        sample_idx = (segment_start + positions - last_restart) % n
        out[first:first + count] = d[sample_idx].mean(axis=1)
    return out


def plot_bootstrap_intervals(exp: str, runs: dict, out_dir: Path) -> None:
    cfg = EXPERIMENTS[exp]
    pairs = history_pairs(exp)
    contrasts: list[tuple[str, np.ndarray, np.ndarray]] = []
    for model in ("gbm", "garch"):
        for a, b in pairs:
            left, right = runs[(exp, a, model)], runs[(exp, b, model)]
            if not left["date"].equals(right["date"]):
                raise ValueError(f"{exp}/{model}: dates do not align for {a} versus {b}")
            label = f"{MODEL_LABELS[model]}: {a} − {b}"
            contrasts.append((label, left["pinball"].to_numpy() - right["pinball"].to_numpy(),
                              left["fz0"].to_numpy() - right["fz0"].to_numpy()))

    fig, axes = plt.subplots(1, 2, figsize=(15, max(5.5, 1.0 * len(contrasts) + 2.0)))
    rng = np.random.default_rng(RANDOM_SEED + (1 if exp == "exp1" else 2))
    offsets = {10: -0.19, 20: 0.0, 40: 0.19}
    markers = {10: "o", 20: "s", 40: "^"}
    metric_specs = [("pinball", 0, 1e4, "Pinball difference (×10⁻⁴)"),
                    ("fz0", 1, 1.0, "FZ0 difference")]
    for ax, (_, col, scale, xlabel) in zip(axes, metric_specs):
        for i, (label, pin_diff, fz_diff) in enumerate(contrasts):
            diff = pin_diff if col == 0 else fz_diff
            estimate = float(diff.mean()) * scale
            for block in BOOTSTRAP_BLOCK_LENGTHS:
                boot = stationary_bootstrap_means(diff, block, BOOTSTRAP_REPS, rng)
                lo, hi = np.quantile(boot, [0.025, 0.975]) * scale
                y = i + offsets[block]
                ax.hlines(y, lo, hi, lw=1.5, alpha=0.85)
                ax.plot(estimate, y, marker=markers[block], ms=5.5, linestyle="none",
                        label=f"Mean block {block}" if i == 0 else None)
        ax.axvline(0, color="black", lw=0.9, ls="--", label="No mean difference")
        ax.set_yticks(range(len(contrasts)), [x[0] for x in contrasts])
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.grid(True, axis="x", alpha=0.22)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle(
        f"{cfg['title']} — paired stationary-bootstrap 95% confidence intervals\n"
        "Difference = first variant − second variant; negative favours the first",
        y=1.03, fontsize=13, fontweight="bold"
    )
    fig.tight_layout()
    save_figure(fig, out_dir, f"{exp}_08_bootstrap_loss_intervals.png")


# ---------------------------------------------------------------------------
# Metrics table and orchestration
# ---------------------------------------------------------------------------
def write_metrics_csv(runs: dict, out_dir: Path) -> Path:
    rows = []
    for (exp, variant, model), tab in runs.items():
        rows.append({
            "experiment": exp,
            "variant": variant,
            "model": MODEL_LABELS[model],
            "n_forecasts": len(tab),
            "violations": int(tab["hit"].sum()),
            "exception_rate_pct": 100 * float(tab["hit"].mean()),
            "mean_var_pct": 100 * float(tab["var"].mean()),
            "mean_es_pct": 100 * float(tab["es"].mean()),
            "pinball_x1e4": 1e4 * float(tab["pinball"].mean()),
            "fz0": float(tab["fz0"].mean()),
            "first_forecast": tab["date"].iloc[0].date().isoformat(),
            "last_forecast": tab["date"].iloc[-1].date().isoformat(),
        })
    path = out_dir / "visualization_metrics.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent.parent,
                        help="Project root containing results/ (default: script parent)")
    parser.add_argument("--skip-bootstrap", action="store_true",
                        help="Skip the bootstrap interval plots; all other figures are still created")
    args = parser.parse_args()

    root = args.project_root.resolve()
    out_dir = root / "reports" / "figures" / "exp1_exp2"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Project root: {root}")
    print("Loading and validating the 15 existing forecast tables...")
    runs = load_all_runs(root)
    print("Input validation passed. No model fitting will be performed.")

    for exp in EXPERIMENTS:
        print(f"\nGenerating {exp} figures...")
        plot_var_timeline(exp, runs, out_dir)
        plot_rolling_coverage(exp, runs, out_dir)
        plot_annual_coverage(exp, runs, out_dir)
        plot_risk_efficiency(exp, runs, out_dir)
        plot_loss_heatmaps(exp, runs, out_dir)
        plot_var_es_distributions(exp, runs, out_dir)
        plot_cumulative_loss_differences(exp, runs, out_dir)
        if not args.skip_bootstrap:
            plot_bootstrap_intervals(exp, runs, out_dir)

    metrics_path = write_metrics_csv(runs, out_dir)
    print(f"\nMetrics table: {metrics_path}")
    print(f"Figures directory: {out_dir}")
    print("Complete. Source result CSVs were read only; no experiment outputs were changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
