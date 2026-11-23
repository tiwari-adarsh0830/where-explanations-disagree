"""Analysis of the five patching settings (pipeline/patching_variants.py).

Questions:
  1. Do the patching settings agree with each other about which heads matter?
  2. Does each explanation method's agreement with "patching" depend on which setting
     is used as the reference? Does the ranking of methods change?
  3. Which heads move the most between settings, and which circuit roles do they belong to?
  4. Group effects (name movers, backups, both) under each setting, per name order.

Spearman and top-10 overlap use absolute mean scores over prompts, as in analysis_v4.
95% intervals: 1000 bootstrap resamples of prompts (numpy seed 0).
Note: attribution patching is a gradient estimate of noise_abc specifically, so its
agreement with the other settings is not a like-for-like comparison.

Run:  python pipeline/analysis_variants.py results/patching_variants_per_prompt.json \
          results/ioi_v3_results_per_prompt.json results/attn_scores.json results/analysis_variants.json
"""
import json, sys
import numpy as np
from scipy.stats import spearmanr

VAR = sys.argv[1] if len(sys.argv) > 1 else "results/patching_variants_per_prompt.json"
V3 = sys.argv[2] if len(sys.argv) > 2 else "results/ioi_v3_results_per_prompt.json"
ATTN = sys.argv[3] if len(sys.argv) > 3 else "results/attn_scores.json"
OUT = sys.argv[4] if len(sys.argv) > 4 else "results/analysis_variants.json"
B = 1000
rng = np.random.default_rng(0)
H = 12

dv, d3, da = json.load(open(VAR)), json.load(open(V3)), json.load(open(ATTN))
PV, P3, PA = dv["prompts"], d3["prompts"], da["prompts"]
N = len(PV)
assert [p["text"] for p in PV] == [p["text"] for p in P3[:N]] == [p["text"] for p in PA[:N]], "prompt order differs"
VARIANTS = dv["variants"]
S = {v: np.array([p["scores"][v] for p in PV]) for v in VARIANTS}            # [N, 144]
GR = {v: {k: np.array([p["group"][v][k] for p in PV]) for k in dv["groups"]} for v in VARIANTS}
attn = np.array([p["attn"] for p in PA[:N]]); vw = np.array([p["vw"] for p in PA[:N]])
dla = np.array([p["dla"] for p in P3[:N]]); attr = np.array([p["attr"] for p in P3[:N]])
order = np.array([p["order"] for p in PV])
METHODS = {
    "attention (end>io)": lambda s: attn[s][:, :, 0].mean(0),
    "attention (best of 6)": lambda s: attn[s].mean(0).max(1),
    "value-weighted (best of 6)": lambda s: vw[s].mean(0).max(1),
    "dla": lambda s: dla[s].mean(0),
    "attr_patching": lambda s: attr[s].mean(0),
}
LABEL = {"noise_abc": "noising, ABC corruption, all positions (v3 reference)",
         "denoise_abc": "denoising, ABC corruption, all positions",
         "noise_flip": "noising, S2 replaced by IO name, all positions",
         "noise_mean": "mean ablation over 32 ABC prompts, all positions",
         "noise_abc_end": "noising, ABC corruption, final position only"}

def lab(i): return f"{i // H}.{i % H}"
def idx(s): l, h = map(int, s.split(".")); return l * H + h
roles = {r: [idx(s) for s in hs] for r, hs in d3["roles"].items()}
roles["previous_token"] = [idx("2.2"), idx("4.11")]
role_of = {i: r for r, hs in roles.items() for i in hs}
def rho(a, b): return float(spearmanr(np.abs(a), np.abs(b)).correlation)
def top10(a, b): return len(set(np.argsort(-np.abs(a), kind="stable")[:10]) & set(np.argsort(-np.abs(b), kind="stable")[:10]))
def ranks(x): r = np.empty(144, int); r[np.argsort(-np.abs(x), kind="stable")] = np.arange(1, 145); return r
def ci(vals): lo, hi = np.percentile(vals, [2.5, 97.5]); return [round(float(lo), 3), round(float(hi), 3)]
def sel_of(c): return np.arange(N) if c == "pooled" else np.where(order == c)[0]

out = {"n_prompts": N, "bootstrap_reps": B, "variants": {v: LABEL[v] for v in VARIANTS},
       "mean_logit_diff": {k: round(float(np.mean([p["logit_diff"][k] for p in PV])), 3) for k in PV[0]["logit_diff"]}}
M = {v: S[v].mean(0) for v in VARIANTS}

# ---------- 1. agreement between settings ----------
boots = [rng.integers(0, N, N) for _ in range(B)]
pair = {}
for a in VARIANTS:
    for b in VARIANTS:
        if a >= b: continue
        bs = [rho(S[a][s].mean(0), S[b][s].mean(0)) for s in boots]
        pair[f"{a} vs {b}"] = {"spearman": round(rho(M[a], M[b]), 3), "spearman_ci95": ci(bs), "top10_overlap": top10(M[a], M[b])}
out["variant_agreement"] = pair
out["top10_by_variant"] = {v: [{"head": lab(i), "role": role_of.get(i, "other"), "score": round(float(M[v][i]), 3)}
                               for i in np.argsort(-np.abs(M[v]), kind="stable")[:10]] for v in VARIANTS}

# ---------- 2. each method against each setting ----------
mv = {}
for c in ("pooled", "BABA", "ABBA"):
    sel = sel_of(c); bsel = [sel[rng.integers(0, len(sel), len(sel))] for _ in range(B)]
    mv[c] = {}
    for v in VARIANTS:
        ref = S[v][sel].mean(0); mv[c][v] = {}
        for m, f in METHODS.items():
            bs = [rho(f(s), S[v][s].mean(0)) for s in bsel]
            mv[c][v][m] = {"spearman": round(rho(f(sel), ref), 3), "spearman_ci95": ci(bs), "top10_overlap": top10(f(sel), ref)}
        mv[c][v]["method_order"] = sorted(METHODS, key=lambda m: -mv[c][v][m]["spearman"])
out["methods_vs_variant"] = mv

# ---------- 3. role ranks and biggest movers ----------
R = {v: ranks(M[v]) for v in VARIANTS}
out["role_ranks"] = {r: {lab(i): {v: int(R[v][i]) for v in VARIANTS} for i in hs} for r, hs in roles.items()}
out["role_median_rank"] = {r: {v: float(np.median([R[v][i] for i in hs])) for v in VARIANTS} for r, hs in roles.items()}
movers = {}
for v in VARIANTS:
    if v == "noise_abc": continue
    cand = [i for i in range(144) if min(R["noise_abc"][i], R[v][i]) <= 20]
    cand.sort(key=lambda i: -abs(np.log(R[v][i] / R["noise_abc"][i])))
    movers[v] = [{"head": lab(i), "role": role_of.get(i, "other"), "rank_noise_abc": int(R["noise_abc"][i]),
                  "rank_here": int(R[v][i]), "score_noise_abc": round(float(M["noise_abc"][i]), 3),
                  "score_here": round(float(M[v][i]), 3)} for i in cand[:8]]
out["biggest_rank_movers_vs_noise_abc"] = movers

# ---------- 4. group effects ----------
nm, bk = roles["name_movers"], roles["backup_nm"]
ge = {}
for v in VARIANTS:
    ge[v] = {}
    for c in ("pooled", "BABA", "ABBA"):
        sel = sel_of(c); Mc = S[v][sel].mean(0); row = {}
        for k in dv["groups"]:
            x = GR[v][k][sel]
            row[k] = {"mean": round(float(x.mean()), 3),
                      "ci95": ci([x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)])}
        row["sum_single_name_movers"] = round(float(Mc[nm].sum()), 3)
        row["sum_single_backups"] = round(float(Mc[bk].sum()), 3)
        sep = row["name_movers"]["mean"] + row["backups"]["mean"]
        row["separately"] = round(sep, 3)
        row["both_over_separately"] = round(row["both"]["mean"] / sep, 2) if abs(sep) >= 0.1 else None
        ge[v][c] = row
out["group_effects"] = ge
json.dump(out, open(OUT, "w"), indent=1)

# short printout
print("mean logit diff:", out["mean_logit_diff"])
print("\nsettings vs each other (Spearman on |score|, top-10 overlap)")
for k, r in pair.items(): print(f"  {k:32s} {r['spearman']:.3f} {r['spearman_ci95']}  top10 {r['top10_overlap']}")
print("\nmethods vs each setting, pooled (Spearman)")
print(" " * 28 + "".join(f"{v:>15s}" for v in VARIANTS))
for m in METHODS: print(f"  {m:26s}" + "".join(f"{mv['pooled'][v][m]['spearman']:15.3f}" for v in VARIANTS))
print("\ngroup effects, pooled / BABA / ABBA (name movers | both)")
for v in VARIANTS:
    print(f"  {v:14s}" + "  ".join(f"{c}: {ge[v][c]['name_movers']['mean']:+.2f} | {ge[v][c]['both']['mean']:+.2f}" for c in ("pooled", "BABA", "ABBA")))
print("\nwrote", OUT)
