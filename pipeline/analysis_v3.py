"""Analyses on the v3 per-prompt output. No model runs, just reads the JSON.

1. Role-aware attention: attention at each query>key pair, and how agreement
   with activation patching changes when a head is scored at its best pair.
2. Bootstrap CIs on Spearman and top-10 overlap, resampling prompts and,
   more conservatively, resampling base templates (prompts from one template
   are not independent).
3. Attribution patching vs activation patching: deviation by layer and role,
   biggest deviators, and how consistent the sign of the error is per prompt.
4. Per-template and per-order breakdown.

Run:  python pipeline/analysis_v3.py results/ioi_v3_results_per_prompt.json results/analysis_v3.json
"""
import json, sys
import numpy as np
from scipy.stats import spearmanr

IN = sys.argv[1] if len(sys.argv) > 1 else "results/ioi_v3_results_per_prompt.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "results/analysis_v3.json"
B = 1000
rng = np.random.default_rng(0)

d = json.load(open(IN))
P = d["prompts"]; PAIRS = d["pairs"]; N = len(P); L = H = 12
attn = np.array([p["attn"] for p in P])            # [N, 144, 5]
dla = np.array([p["dla"] for p in P])              # [N, 144]
attr = np.array([p["attr"] for p in P])
truth = np.array([p["truth"] for p in P])
tpl = np.array([p["tpl"] for p in P]); order = np.array([p["order"] for p in P])
ldiff = np.array([p["logit_diff"] for p in P])
gdrop = {k: np.array([p["group_drop"][k] for p in P]) for k in P[0]["group_drop"]}

def lab(i): return f"{i // H}.{i % H}"
def idx(s): l, h = map(int, s.split(".")); return l * H + h

# role labels; previous-token heads (2.2, 4.11) are from Wang et al. 2023 but not in the pipeline's ROLE dict
roles = {r: [idx(s) for s in hs] for r, hs in d["roles"].items()}
roles["previous_token"] = [idx("2.2"), idx("4.11")]
role_of = {i: r for r, hs in roles.items() for i in hs}

def rho(score, ref): return float(spearmanr(np.abs(score), np.abs(ref)).correlation)
def top10(score, ref):
    return len(set(np.argsort(-np.abs(ref))[:10]) & set(np.argsort(-np.abs(score))[:10]))
def ranks(score):
    r = np.empty(144, int); r[np.argsort(-np.abs(score))] = np.arange(1, 145); return r

out = {"n_prompts": N, "bootstrap_reps": B}

# ---------- 1. role-aware attention ----------
A = attn.mean(0)                                    # [144, 5]
T = truth.mean(0)
attn_scores = {f"attn[{p}]": A[:, j] for j, p in enumerate(PAIRS)}
attn_scores["attn[max pair]"] = A.max(1)
out["attention_by_pair"] = {
    "spearman_vs_patching": {k: round(rho(v, T), 3) for k, v in attn_scores.items()},
    "top10_overlap": {k: top10(v, T) for k, v in attn_scores.items()},
    "role_heads": {lab(i): {"role": role_of[i], "patching_rank": int(ranks(T)[i]),
                            "attn": {p: round(float(A[i, j]), 3) for j, p in enumerate(PAIRS)},
                            "rank_end>io": int(ranks(A[:, 0])[i]), "rank_max_pair": int(ranks(A.max(1))[i])}
                   for r, hs in roles.items() for i in hs},
}

# ---------- 2. bootstrap CIs ----------
methods = {"attention (end>io)": attn[:, :, 0], "attention (max pair)": None, "dla": dla, "attr_patching": attr}
def stats_for(sel):
    Ts = truth[sel].mean(0); res = {}
    for m, X in methods.items():
        s = attn[sel].mean(0).max(1) if X is None else X[sel].mean(0)
        res[m] = (rho(s, Ts), top10(s, Ts))
    return res
point = stats_for(np.arange(N))
base_tpl = np.unique(tpl)                           # base templates (each has a BABA and an ABBA group)
boot_p = [stats_for(rng.integers(0, N, N)) for _ in range(B)]
boot_t = []
for _ in range(B):
    pick = rng.choice(base_tpl, len(base_tpl))
    boot_t.append(stats_for(np.concatenate([np.where(tpl == t)[0] for t in pick])))
def ci(boot, m, k):
    v = np.array([b[m][k] for b in boot]); lo, hi = np.percentile(v, [2.5, 97.5]); return [round(float(lo), 3), round(float(hi), 3)]
out["agreement_with_patching"] = {m: {"spearman": round(point[m][0], 3), "top10_overlap": point[m][1],
                                      "spearman_ci95_prompts": ci(boot_p, m, 0), "spearman_ci95_templates": ci(boot_t, m, 0),
                                      "top10_ci95_prompts": ci(boot_p, m, 1), "top10_ci95_templates": ci(boot_t, m, 1)}
                                  for m in methods}

# ---------- 3. attribution patching deviation ----------
Am = attr.mean(0); dev = Am - T
out["attr_patching_deviation"] = {
    "mean_abs_dev_by_layer": [round(float(np.abs(dev[l * H:(l + 1) * H]).mean()), 3) for l in range(L)],
    "by_role": {r: {"attr_sum": round(float(Am[hs].sum()), 3), "patching_sum": round(float(T[hs].sum()), 3),
                    "ratio": round(float(Am[hs].sum() / T[hs].sum()), 2) if abs(T[hs].sum()) > 1e-6 else None}
                for r, hs in roles.items()},
    "top10_abs_dev": [{"head": lab(i), "role": role_of.get(i, "other"), "attr": round(float(Am[i]), 3),
                       "patching": round(float(T[i]), 3),
                       "share_prompts_attr_below_patching": round(float((attr[:, i] < truth[:, i]).mean()), 2)}
                      for i in np.argsort(-np.abs(dev))[:10]],
    "sum_single_head_patching": round(float(T.sum()), 3),
    "sum_attr_patching": round(float(Am.sum()), 3),
    "clean_logit_diff": round(float(ldiff.mean()), 3),
}

# ---------- 4. per template / per order ----------
def block(sel):
    Ts = truth[sel].mean(0)
    return {"n": int(sel.sum()), "accuracy": round(float((ldiff[sel] > 0).mean()), 3),
            "mean_logit_diff": round(float(ldiff[sel].mean()), 3),
            "spearman": {"attention (end>io)": round(rho(attn[sel, :, 0].mean(0), Ts), 3),
                         "attention (max pair)": round(rho(attn[sel].mean(0).max(1), Ts), 3),
                         "dla": round(rho(dla[sel].mean(0), Ts), 3), "attr_patching": round(rho(attr[sel].mean(0), Ts), 3)},
            "group_drop": {k: round(float(v[sel].mean()), 3) for k, v in gdrop.items()}}
out["by_order"] = {o: block(order == o) for o in ("BABA", "ABBA")}
out["by_template"] = {str(int(t)): block(tpl == t) for t in base_tpl}
sp = {m: [v["spearman"][m] for v in out["by_template"].values()] for m in out["by_order"]["BABA"]["spearman"]}
out["by_template_spread"] = {m: {"min": min(v), "median": round(float(np.median(v)), 3), "max": max(v)} for m, v in sp.items()}

json.dump(out, open(OUT, "w"), indent=1)

# ---------- readable summary ----------
print(f"{N} prompts, {len(base_tpl)} base templates, {B} bootstrap reps\n")
print("Agreement with activation patching (Spearman on |score|, 95% CI over prompts / over templates; top-10 overlap)")
for m, v in out["agreement_with_patching"].items():
    print(f"  {m:22s} rho {v['spearman']:.3f}  [{v['spearman_ci95_prompts'][0]:.2f}, {v['spearman_ci95_prompts'][1]:.2f}]"
          f" / [{v['spearman_ci95_templates'][0]:.2f}, {v['spearman_ci95_templates'][1]:.2f}]   top10 {v['top10_overlap']}"
          f"  [{v['top10_ci95_templates'][0]:.0f}, {v['top10_ci95_templates'][1]:.0f}]")
print("\nAttention at each pair: Spearman vs patching")
for k, v in out["attention_by_pair"]["spearman_vs_patching"].items():
    print(f"  {k:18s} {v:.3f}   top10 {out['attention_by_pair']['top10_overlap'][k]}")
print("\nRole heads: patching rank, attention rank at end>io vs at best pair, and attention per pair")
for h, v in out["attention_by_pair"]["role_heads"].items():
    if v["role"] in ("s_inhibition", "induction", "duplicate", "previous_token", "name_movers"):
        best = max(v["attn"], key=v["attn"].get)
        print(f"  {h:5s} {v['role']:15s} patch {v['patching_rank']:3d}  attn end>io {v['rank_end>io']:3d} -> max pair {v['rank_max_pair']:3d}"
              f"   best pair {best} = {v['attn'][best]:.2f} (end>io {v['attn']['end>io']:.2f})")
dv = out["attr_patching_deviation"]
print("\nAttribution patching minus patching: mean |dev| by layer")
print("  " + "  ".join(f"L{l}:{x:.2f}" for l, x in enumerate(dv["mean_abs_dev_by_layer"])))
print("By role (sum over role heads): attr vs patching")
for r, v in dv["by_role"].items(): print(f"  {r:15s} {v['attr_sum']:7.3f} vs {v['patching_sum']:7.3f}  ratio {v['ratio']}")
print("Biggest deviators")
for v in dv["top10_abs_dev"]:
    print(f"  {v['head']:5s} {v['role']:15s} attr {v['attr']:6.3f}  patch {v['patching']:6.3f}  attr<patch in {v['share_prompts_attr_below_patching']:.0%} of prompts")
print(f"Sum of single-head patching {dv['sum_single_head_patching']} vs attr sum {dv['sum_attr_patching']} vs clean logit diff {dv['clean_logit_diff']}")
print("\nBy name order")
for o, v in out["by_order"].items(): print(f"  {o}: n {v['n']}, acc {v['accuracy']}, logit diff {v['mean_logit_diff']}, spearman {v['spearman']}, groups {v['group_drop']}")
print("Per base template (8 prompts each): Spearman min / median / max")
for m, v in out["by_template_spread"].items(): print(f"  {m:22s} {v['min']:.2f} / {v['median']:.2f} / {v['max']:.2f}")
print(f"\nwrote {OUT}")
