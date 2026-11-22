"""Analysis for the greater-than task (pipeline/gt_check.py output), mirroring analysis_v4 and analysis_variants.

1. Agreement of each explanation method with patching, against both settings (noising and
   denoising with the "01" corruption): Spearman on |mean score| and top-10 overlap, with
   95% bootstrap intervals over prompts (1000 resamples, numpy seed 0).
2. Noising vs denoising agreement.
3. Ranks of the seven circuit heads from Hanna et al. (2023) under every method and setting.
4. Group effects with bootstrap intervals: layer 5-7 year heads, layer 8-9 year heads, both,
   MLPs 8-11; against the sum of single-head effects and the two head groups separately.
5. By start-year bucket (YY 02-19, 20-39, ...): clean metric and agreement.

Run:  python pipeline/analysis_gt.py results/gt results/analysis_gt.json
"""
import json, sys
import numpy as np
from scipy.stats import spearmanr

DIR = sys.argv[1] if len(sys.argv) > 1 else "results/gt"
OUT = sys.argv[2] if len(sys.argv) > 2 else "results/analysis_gt.json"
B = 1000
rng = np.random.default_rng(0)
H = 12
d = json.load(open(f"{DIR}/per_prompt.json")); da = json.load(open(f"{DIR}/attn_scores.json")); dv = json.load(open(f"{DIR}/patching_variants_per_prompt.json"))
P, PA, PV = d["prompts"], da["prompts"], dv["prompts"]
N = len(P)
assert [p["text"] for p in P] == [p["text"] for p in PA] == [p["text"] for p in PV]
attn = np.array([p["attn"] for p in PA]); vw = np.array([p["vw"] for p in PA])
dla = np.array([p["dla"] for p in P]); attr = np.array([p["attr"] for p in P])
S = {v: np.array([p["scores"][v] for p in PV]) for v in dv["variants"]}
G = {v: {k: np.array([p["group"][v][k] for p in PV]) for k in dv["groups"]} for v in dv["variants"]}
bucket = np.array([p["tpl"] for p in P]); metric = np.array([p["logit_diff"] for p in P]); mcor = np.array([p["metric_corrupt"] for p in P])
PAIRS = d["pairs"]
METHODS = {
    "attention (end>yy)": lambda s: attn[s][:, :, 0].mean(0),
    f"attention (best of {len(PAIRS)})": lambda s: attn[s].mean(0).max(1),
    f"value-weighted (best of {len(PAIRS)})": lambda s: vw[s].mean(0).max(1),
    "dla": lambda s: dla[s].mean(0),
    "attr_patching": lambda s: attr[s].mean(0),
}
def lab(i): return f"{i // H}.{i % H}"
def idx(s): l, h = map(int, s.split(".")); return l * H + h
def rho(a, b): return float(spearmanr(np.abs(a), np.abs(b)).correlation)
def top10(a, b): return len(set(np.argsort(-np.abs(a))[:10]) & set(np.argsort(-np.abs(b))[:10]))
def ranks(x): r = np.empty(144, int); r[np.argsort(-np.abs(x))] = np.arange(1, 145); return r
def ci(v): lo, hi = np.percentile(v, [2.5, 97.5]); return [round(float(lo), 3), round(float(hi), 3)]
circuit = [idx(h) for hs in d["roles"].values() for h in hs]
ALL = np.arange(N); boots = [rng.integers(0, N, N) for _ in range(B)]

out = {"n_prompts": N, "bootstrap_reps": B, "metric": d["metric"],
       "mean_metric": {"clean": round(float(metric.mean()), 3), "clean_sd": round(float(metric.std()), 3), "corrupt_01": round(float(mcor.mean()), 3)},
       "pairs": PAIRS}
# 1-2
agree = {}
for v in S:
    agree[v] = {}
    for m, f in METHODS.items():
        agree[v][m] = {"spearman": round(rho(f(ALL), S[v].mean(0)), 3), "top10_overlap": top10(f(ALL), S[v].mean(0)),
                       "spearman_ci95": ci([rho(f(s), S[v][s].mean(0)) for s in boots])}
    agree[v]["method_order"] = sorted(METHODS, key=lambda m: -agree[v][m]["spearman"])
out["agreement"] = agree
out["noise_vs_denoise"] = {"spearman": round(rho(S["noise_01"].mean(0), S["denoise_01"].mean(0)), 3),
                           "spearman_ci95": ci([rho(S["noise_01"][s].mean(0), S["denoise_01"][s].mean(0)) for s in boots]),
                           "top10_overlap": top10(S["noise_01"].mean(0), S["denoise_01"].mean(0))}
out["attention_by_pair"] = {p: round(rho(attn[:, :, j].mean(0), S["noise_01"].mean(0)), 3) for j, p in enumerate(PAIRS)}
# 3
scores = {**{m: f(ALL) for m, f in METHODS.items()}, **{f"patching ({v})": S[v].mean(0) for v in S}}
R = {k: ranks(v) for k, v in scores.items()}
out["circuit_head_ranks"] = {lab(i): {k: int(R[k][i]) for k in R} for i in circuit}
out["circuit_heads_in_top10"] = {k: int(sum(R[k][i] <= 10 for i in circuit)) for k in R}
out["top10_by_setting"] = {v: [{"head": lab(i), "circuit": i in circuit, "score": round(float(S[v].mean(0)[i]), 4)}
                               for i in np.argsort(-np.abs(S[v].mean(0)))[:10]] for v in S}
out["dla_sign_of_circuit_heads"] = {lab(i): round(float(dla[:, i].mean()), 3) for i in circuit}
# 4
early, late = [idx(h) for h in d["roles"]["yy_heads_early"]], [idx(h) for h in d["roles"]["yy_heads_late"]]
ge = {}
for v in S:
    Mv = S[v].mean(0); row = {}
    for k, x in G[v].items():
        row[k] = {"mean": round(float(x.mean()), 3), "ci95": ci([x[s].mean() for s in boots])}
    row["sum_single_early"] = round(float(Mv[early].sum()), 3); row["sum_single_late"] = round(float(Mv[late].sum()), 3)
    sep = row["early_heads"]["mean"] + row["late_heads"]["mean"]; row["separately"] = round(sep, 3)
    row["both_over_separately"] = round(row["both"]["mean"] / sep, 2) if abs(sep) >= 0.05 else None
    ge[v] = row
out["group_effects"] = ge
# 5
out["by_bucket"] = {d["buckets"][b]: {"n": int((bucket == b).sum()), "clean_metric": round(float(metric[bucket == b].mean()), 3),
                                      **{m: round(rho(f(np.where(bucket == b)[0]), S["noise_01"][bucket == b].mean(0)), 3) for m, f in METHODS.items()}}
                    for b in range(len(d["buckets"]))}
json.dump(out, open(OUT, "w"), indent=1)

print("metric:", out["mean_metric"])
for v in S:
    print(f"\nagainst {v}:")
    for m in METHODS: print(f"  {m:26s} {agree[v][m]['spearman']:.3f} {agree[v][m]['spearman_ci95']}  top10 {agree[v][m]['top10_overlap']}")
print("\nnoise vs denoise:", out["noise_vs_denoise"])
print("attention by pair:", out["attention_by_pair"])
print("circuit heads in top 10:", out["circuit_heads_in_top10"])
print("circuit head ranks:"); [print(" ", h, r) for h, r in out["circuit_head_ranks"].items()]
print("DLA of circuit heads:", out["dla_sign_of_circuit_heads"])
print("groups:"); [print(" ", v, {k: (r[k]['mean'], r[k]['ci95']) for k in G[v]}, "separately", r["separately"], "ratio", r["both_over_separately"]) for v, r in ge.items()]
print("by bucket:"); [print(" ", b, r) for b, r in out["by_bucket"].items()]
print("wrote", OUT)
