import json, os
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
R = ROOT / "results"
S = {e: json.loads((R / e / "summary_priority.json").read_text()) for e in ("exp1", "exp2")}

rows = []
for e, s in S.items():
    for key, v in s["runs"].items():
        variant, model = key.split("|")
        rows.append(dict(exp=e, variant=variant, model=model, n=v["n_forecasts"], viol=v["n_violations"],
                         rate_pct=round(100 * v["violation_rate"], 3), kupiec_p=round(v["kupiec_p"], 4),
                         christ_p=round(v["christoffersen_p"], 4), mean_var_pct=round(100 * v["mean_var"], 3),
                         mean_es_pct=round(100 * v["mean_es"], 3)))
print(pd.DataFrame(rows).to_string(index=False))

for e in ("exp1", "exp2"):
    for model in ("gbm", "garch"):
        cols, covid = {}, {}
        for d in sorted((R / e).iterdir()):
            f = d / f"{model}.csv"
            if d.is_dir() and f.exists():
                t = pd.read_csv(f); dt = pd.to_datetime(t["date"])
                cols[d.name] = t.groupby(dt.dt.year)["hit"].sum()
                covid[d.name] = int(t.loc[(dt >= "2020-02-15") & (dt <= "2020-06-30"), "hit"].sum())
        table = pd.DataFrame(cols).fillna(0).astype(int)
        print(f"\n{e} {model.upper()}: violations by year (only years with any)")
        print(table[table.sum(axis=1) > 0].to_string())
        print("COVID window (2020-02-15..06-30) hits:", covid)

print("\nGARCH fits with alpha+beta >= 0.999 (training_end, persistence):")
for e in ("exp1", "exp2"):
    for d in sorted((R / e).iterdir()):
        f = d / "garch.json"
        if d.is_dir() and f.exists():
            m = json.loads(f.read_text())
            logs = [x for x in m["refit_logs"] if x.get("component") == "GARCH"]
            pars = (m.get("launcher") or {}).get("garch_fit_parameters") or []
            hi = [(l["training_end"], round(p["persistence"], 5)) for l, p in zip(logs, pars) if p["persistence"] >= 0.999]
            print(f"  {e}/{d.name}: {len(hi)} fits", hi[:6])

print("\nPaired (a_only/b_only = hit in only one variant):")
for e, s in S.items():
    for k, v in s["paired_comparisons"].items():
        if k.startswith(("GBM", "GARCH")):
            print(f"  {e} {k}: a_only={v['a_only']} b_only={v['b_only']} mcnemar_p={v['mcnemar_exact_p']} mean_var_diff={v['mean_var_diff_a_minus_b']:.5f}")