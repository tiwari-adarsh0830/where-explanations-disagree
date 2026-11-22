"""Trivial baselines for agreement with patching, in both tasks and every patching setting.

Any per-head score that tracks depth can agree with patching, because later heads tend to matter
more. These baselines say how much agreement comes for free:
  layer      the head's layer number (0-11)
  activity   1 - attention from the final token to BOS, i.e. how much of the final token's
             attention the head spends anywhere except the start-of-text "sink"
Spearman on |score| over the 144 heads, as for the methods, with 95% bootstrap intervals over
prompts (1000 resamples, numpy seed 0; layer has no sampling noise, so only the reference varies).

Run:  python pipeline/analysis_baselines.py results results/analysis_baselines.json
"""
import json, sys
import numpy as np
from scipy.stats import spearmanr
D = sys.argv[1] if len(sys.argv) > 1 else "results"
OUT = sys.argv[2] if len(sys.argv) > 2 else "results/analysis_baselines.json"
rng = np.random.default_rng(0); B = 1000
rho = lambda a, b: float(spearmanr(np.abs(a), np.abs(b)).correlation)
ci = lambda v: [round(float(x), 3) for x in np.percentile(v, [2.5, 97.5])]
layer = np.repeat(np.arange(12), 12).astype(float)
out = {}
for task, vf, mf, end in (("ioi", f"{D}/patching_variants_per_prompt.json", f"{D}/attn_map.json", "END"),
                          ("gt", f"{D}/gt/patching_variants_per_prompt.json", f"{D}/gt/attn_map.json", "XX end")):
    V = json.load(open(vf)); M = json.load(open(mf))
    A = np.array(M["attn"]["all"]); e = M["labels"].index(end)
    activity = 1 - A[:, e, 0]
    out[task] = {}
    for v in V["variants"]:
        S = np.array([p["scores"][v] for p in V["prompts"]]); n = len(S); boots = [rng.integers(0, n, n) for _ in range(B)]
        out[task][v] = {name: {"spearman": round(rho(sc, S.mean(0)), 3), "spearman_ci95": ci([rho(sc, S[b].mean(0)) for b in boots])}
                        for name, sc in (("layer", layer), ("activity", activity))}
json.dump(out, open(OUT, "w"), indent=1)
for t, d in out.items():
    for v, r in d.items(): print(t, f"{v:14s}", {k: (x["spearman"], x["spearman_ci95"]) for k, x in r.items()})
print("wrote", OUT)
