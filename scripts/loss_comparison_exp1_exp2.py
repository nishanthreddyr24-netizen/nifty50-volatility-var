"""Read-only loss comparison for Experiment 1/2 priority runs: pinball, FZ0, paired stationary bootstrap."""
import os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
R = ROOT / "results"
TAU, B, BLOCK = 0.01, 2000, 20
MODELS = ("gbm", "garch", "historical_simulation")
LAYOUT = {"exp1": ["BASELINE_CONFIGURED_2008", "EXTENDED_1990"],
          "exp2": ["PRELAUNCH_1990", "LAUNCHED_1996", "FREEFLOAT_2009"]}


def pinball(r, var):                      # q = -VaR is the return quantile
    q = -var
    return np.where(r <= q, (1 - TAU) * (q - r), TAU * (r - q))


def fz0(r, var, es):                      # Patton-Ziegel-Chen FZ0, lower tail, v,e < 0
    v, e = -var, -es
    return np.where(r <= v, (r - v) / (TAU * e), 0.0) + v / e + np.log(-e) - 1.0


def sb_means(d, rng):                     # vectorised stationary bootstrap of mean(d)
    n = len(d); pos = np.arange(n)
    start = rng.integers(n, size=(B, n))
    new = rng.random((B, n)) < 1.0 / BLOCK
    new[:, 0] = True
    last = np.maximum.accumulate(np.where(new, pos, 0), axis=1)
    base = np.take_along_axis(start, last, axis=1)
    return d[(base + pos - last) % n].mean(axis=1)


def holm(p):
    order = np.argsort(p); out = np.empty(len(p)); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(p) - rank) * p[i])); out[i] = run
    return out


runs = {}
for exp, variants in LAYOUT.items():
    for v in variants:
        for m in MODELS:
            t = pd.read_csv(R / exp / v / f"{m}.csv")
            r, var, es = (t[c].to_numpy(float) for c in ("return", "var", "es"))
            assert np.array_equal(t["hit"].to_numpy(), (-r > var).astype(int))
            runs[(exp, v, m)] = dict(dates=t["date"], pin=pinball(r, var), fz=fz0(r, var, es), var=var.mean())

print("PER-RUN MEAN LOSSES (pinball x1e4, FZ0; lower is better)")
print(pd.DataFrame([dict(exp=k[0], variant=k[1], model=k[2], mean_var_pct=round(100 * v["var"], 3),
                         pinball_e4=round(1e4 * v["pin"].mean(), 4), fz0=round(v["fz"].mean(), 4))
                    for k, v in runs.items()]).to_string(index=False))

rng = np.random.default_rng(12345)


def contrast(ka, kb, label, family):
    a, b = runs[ka], runs[kb]
    assert a["dates"].equals(b["dates"])
    row = dict(contrast=label, family=family)
    for name, scale in (("pin", 1e4), ("fz", 1.0)):
        d = a[name] - b[name]; m = sb_means(d, rng)
        lo, hi = np.quantile(m, [0.025, 0.975])
        p = min(1.0, 2 * (min((m <= 0).sum(), (m >= 0).sum()) + 1) / (B + 1))
        row[f"d_{name}"] = round(scale * d.mean(), 4)
        row[f"ci_{name}"] = f"[{scale * lo:.4f}, {scale * hi:.4f}]"
        row[f"p_{name}"] = p
    return row


rows = []
for m in ("gbm", "garch"):
    rows.append(contrast(("exp1", "EXTENDED_1990", m), ("exp1", "BASELINE_CONFIGURED_2008", m), f"exp1 {m}: EXTENDED - BASELINE", "history"))
    for a, b in (("PRELAUNCH_1990", "LAUNCHED_1996"), ("PRELAUNCH_1990", "FREEFLOAT_2009"), ("LAUNCHED_1996", "FREEFLOAT_2009")):
        rows.append(contrast(("exp2", a, m), ("exp2", b, m), f"exp2 {m}: {a} - {b}", "history"))
for exp, v in (("exp1", "EXTENDED_1990"), ("exp2", "PRELAUNCH_1990")):
    for ma, mb in (("garch", "historical_simulation"), ("gbm", "historical_simulation"), ("garch", "gbm")):
        rows.append(contrast((exp, v, ma), (exp, v, mb), f"{exp} {v}: {ma} - {mb}", "model"))

df = pd.DataFrame(rows)
mask = df["family"] == "history"
df["p_fz_holm"] = np.nan
df.loc[mask, "p_fz_holm"] = holm(df.loc[mask, "p_fz"].to_numpy())
pd.set_option("display.width", 250)
print("\nPAIRED DIFFERENCES (first minus second; negative = first is better). Exploratory except FZ0 history family (Holm).")
print(df.drop(columns="family").to_string(index=False, float_format=lambda x: f"{x:.4g}"))