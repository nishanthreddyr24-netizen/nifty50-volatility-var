# %% [markdown]
# # Stylized Facts of NIFTY 50 Returns
#
# This notebook checks the empirical facts that motivate moving beyond GBM:
# fat tails, volatility clustering, the leverage effect, negative skew, and
# near-zero autocorrelation of returns but strong autocorrelation of squared
# returns. It is written as a `# %%`-delimited script (jupytext "light"
# format) so it is plain, diffable, runnable code; convert to a classic
# `.ipynb` with `jupytext --to notebook 01_stylized_facts.py` if a notebook
# UI is wanted, or just run it as a script.

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sstats
from statsmodels.graphics.tsaplots import plot_acf

sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))

from src.loaders import load_close
from src.returns import log_returns, rolling_vol, summary_stats

FIG_DIR = Path(__file__).resolve().parent.parent / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["figure.figsize"] = (10, 5)

# %% [markdown]
# ## Load data and compute log returns

# %%
prices = load_close("nifty50")
returns = log_returns(prices)
vix = load_close("india_vix")

print(f"NIFTY 50: {len(prices)} prices, {prices.index.min().date()} to {prices.index.max().date()}")
print(f"Log returns: {len(returns)} observations")

stats = summary_stats(returns)
print(stats)

# %% [markdown]
# ## 1. Price and return series

# %%
fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
axes[0].plot(prices.index, prices.values, lw=0.8)
axes[0].set_title("NIFTY 50 Close Price")
axes[0].set_ylabel("Level")

axes[1].plot(returns.index, returns.values, lw=0.4, color="darkred")
axes[1].set_title("Daily Log Returns")
axes[1].set_ylabel("Log return")
plt.tight_layout()
plt.savefig(FIG_DIR / "01_price_and_returns.png", dpi=120)
plt.close()

# %% [markdown]
# ## 2. Fat tails: histogram + QQ plot vs Normal
#
# If returns were really Normal (as GBM assumes), the histogram would be
# bell-shaped with no excess kurtosis, and the QQ plot would lie on the
# 45-degree line. Finance returns typically show a sharply peaked center
# and much fatter tails than Normal (excess kurtosis > 0).

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(returns, bins=100, density=True, alpha=0.6, label="Empirical")
x = np.linspace(returns.min(), returns.max(), 500)
axes[0].plot(x, sstats.norm.pdf(x, returns.mean(), returns.std()), "r-", lw=2, label="Normal fit")
axes[0].set_title(f"Return distribution (excess kurtosis = {stats['excess_kurtosis']:.2f})")
axes[0].legend()

sstats.probplot(returns, dist="norm", plot=axes[1])
axes[1].set_title("QQ plot vs Normal")

plt.tight_layout()
plt.savefig(FIG_DIR / "02_fat_tails.png", dpi=120)
plt.close()

jb_stat, jb_p = sstats.jarque_bera(returns)
print(f"Jarque-Bera stat={jb_stat:.1f}, p-value={jb_p:.2e} -> reject Normality: {jb_p < 0.05}")

# %% [markdown]
# ## 3. Volatility clustering: ACF of returns vs squared returns
#
# Near-zero autocorrelation of raw returns (market is hard to predict
# linearly) but strong, persistent autocorrelation of squared returns
# (today's volatility predicts tomorrow's volatility) is the core stylized
# fact motivating GARCH/HMM regime models over GBM's constant volatility.

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plot_acf(returns, lags=40, ax=axes[0], title="ACF of returns")
plot_acf(returns**2, lags=40, ax=axes[1], title="ACF of squared returns")
plt.tight_layout()
plt.savefig(FIG_DIR / "03_acf_clustering.png", dpi=120)
plt.close()

from statsmodels.stats.diagnostic import acorr_ljungbox

lb_returns = acorr_ljungbox(returns, lags=[10, 20], return_df=True)
lb_squared = acorr_ljungbox(returns**2, lags=[10, 20], return_df=True)
print("Ljung-Box on returns (p-values):\n", lb_returns["lb_pvalue"])
print("Ljung-Box on squared returns (p-values):\n", lb_squared["lb_pvalue"])

# %% [markdown]
# ## 4. Rolling realized volatility vs India VIX
#
# An external, market-implied check: if our realized-volatility regimes are
# sensible, they should track India VIX (which the market prices from NIFTY
# options, independent of our own return series).

# %%
rv21 = rolling_vol(returns, window=21, annualize=True)

fig, ax1 = plt.subplots(figsize=(11, 5))
ax1.plot(rv21.index, rv21.values * 100, label="21d realized vol (annualized %)", color="steelblue")
ax1.set_ylabel("Realized vol (%)")
ax2 = ax1.twinx()
ax2.plot(vix.index, vix.values, label="India VIX", color="darkorange", alpha=0.7)
ax2.set_ylabel("India VIX")
fig.legend(loc="upper right")
plt.title("21-day realized volatility vs India VIX")
plt.tight_layout()
plt.savefig(FIG_DIR / "04_vol_vs_vix.png", dpi=120)
plt.close()

common_idx = rv21.dropna().index.intersection(vix.index)
corr = np.corrcoef(rv21.loc[common_idx], vix.loc[common_idx])[0, 1]
print(f"Correlation(21d realized vol, India VIX) = {corr:.3f}")

# %% [markdown]
# ## 5. Leverage effect: correlation between return and next-day volatility change
#
# Negative returns tend to be followed by *increases* in volatility more
# than positive returns of the same magnitude (bad news raises risk more
# than good news lowers it) -- this asymmetry motivates GJR-GARCH over plain
# GARCH later.

# %%
vol_change = rv21.diff()
leverage_corr = pd.concat([returns, vol_change.shift(-1)], axis=1).dropna()
leverage_corr.columns = ["return", "next_vol_change"]
corr_leverage = leverage_corr.corr().iloc[0, 1]
print(f"Correlation(return_t, vol_change_{{t+1}}) = {corr_leverage:.3f} (negative => leverage effect)")

# %% [markdown]
# ## 6. Crisis markers: 2008 GFC and 2020 COVID crash

# %%
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(returns.index, rolling_vol(returns, 21).values * 100, lw=1)
ax.axvspan(pd.Timestamp("2008-09-01"), pd.Timestamp("2009-03-31"), color="red", alpha=0.15, label="2008 GFC")
ax.axvspan(pd.Timestamp("2020-02-15"), pd.Timestamp("2020-06-30"), color="purple", alpha=0.15, label="2020 COVID")
ax.set_title("21-day realized volatility with crisis windows marked")
ax.set_ylabel("Annualized vol (%)")
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "05_crisis_windows.png", dpi=120)
plt.close()

print(f"\nFigures written to {FIG_DIR}")
