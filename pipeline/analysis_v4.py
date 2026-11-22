"""Analysis v4: everything the D3 page and the paper report, from one script.

Reads the v3 per-prompt output (patching, DLA, attribution patching, group knockouts)
and attention_patterns.py's attn_scores.json (attention and value-weighted attention at
six positions). No model runs. Supersedes analysis_v3.py, which stays because it produced
the numbers in the Nov 3 update (attention at five positions).

1. Attention at each of the six positions, raw and value-weighted, and what each head
   ranks at its best position.
2. Agreement with activation patching (Spearman on |score|, top-10 overlap) for every
   method, with 95% bootstrap intervals over prompts and over base templates.
3. Attribution patching vs patching: error by layer and role, biggest deviators.
4. Group knockouts per name order, with bootstrap intervals, against the sum of
   single-head effects and against the two groups knocked out separately.
5. Per name order and per template: agreement, and how often the order effects repeat
   across the 15 templates.

Run:  python pipeline/analysis_v4.py results/ioi_v3_results_per_prompt.json results/attn_scores.json results/analysis_v4.json
"""
import json, sys
import numpy as np
from scipy.stats import spearmanr

IN = sys.argv[1] if len(sys.argv) > 1 else "results/ioi_v3_results_per_prompt.json"
ATTN = sys.argv[2] if len(sys.argv) > 2 else "results/attn_scores.json"
OUT = sys.argv[3] if len(sys.argv) > 3 else "results/analysis_v4.json"
B = 1000
rng = np.random.default_rng(0)

d = json.load(open(IN)); a = json.load(open(ATTN))
P = d["prompts"]; N = len(P); L = H = 12
assert [p["text"] for p in P] == [q["text"] for q in a["prompts"]], "attn_scores.json prompts do not match the v3 file"
PAIRS = a["pairs"]                                  # six query>key positions
attn = np.array([q["attn"] for q in a["prompts"]])  # [N, 144, 6]
vw = np.array([q["vw"] for q in a["prompts"]])      # [N, 144, 6], value-weighted
dla = np.array([p["dla"] for p in P]); attr = np.array([p["attr"] for p in P]); truth = np.array([p["truth"] for p in P])
tpl = np.array([p["tpl"] for p in P]); order = np.array([p["order"] for p in P])
ldiff = np.array([p["logit_diff"] for p in P])
gdrop = {k: np.array([p["group_drop"][k] for p in P]) for k in P[0]["group_drop"]}
FIVE = [PAIRS.index(p) for p in ("end>io", "end>s1", "end>s2", "s2>s1", "s2>s1+1")]   # the v3 positions

def lab(i): return f"{i // H}.{i % H}"
def idx(s): l, h = map(int, s.split(".")); return l * H + h
roles = {r: [idx(s) for s in hs] for r, hs in d["roles"].items()}
roles["previous_token"] = [idx("2.2"), idx("4.11")]  # Wang et al. 2023, not in the pipeline's role list
role_of = {i: r for r, hs in roles.items() for i in hs}

def rho(score, ref): return float(spearmanr(np.abs(score), np.abs(ref)).correlation)
# stable sorts: exact ties (for example two heads with zero attention) get the same order on every numpy version
def top10(score, ref): return len(set(np.argsort(-np.abs(ref), kind="stable")[:10]) & set(np.argsort(-np.abs(score), kind="stable")[:10]))
def ranks(score):
    r = np.empty(144, int); r[np.argsort(-np.abs(score), kind="stable")] = np.arange(1, 145); return r

# every per-head score, as a function of a set of prompts
METHODS = {
    "attention (end>io)": lambda s: attn[s][:, :, 0].mean(0),
    "attention (best of 5, as in v3)": lambda s: attn[s][:, :, FIVE].mean(0).max(1),
    "attention (best of 6)": lambda s: attn[s].mean(0).max(1),
    "value-weighted (end>io)": lambda s: vw[s][:, :, 0].mean(0),
    "value-weighted (best of 6)": lambda s: vw[s].mean(0).max(1),
    "dla": lambda s: dla[s].mean(0),
    "attr_patching": lambda s: attr[s].mean(0),
}
ALL = np.arange(N)
out = {"n_prompts": N, "bootstrap_reps": B, "pairs": PAIRS}

# ---------- 1. attention at each position ----------
A, V, T = attn.mean(0), vw.mean(0), truth.mean(0)
by_pair = {}
for j, p in enumerate(PAIRS):
    by_pair[f"attn[{p}]"] = A[:, j]; by_pair[f"vw[{p}]"] = V[:, j]
by_pair["attn[best of 5]"] = A[:, FIVE].max(1); by_pair["attn[best of 6]"] = A.max(1); by_pair["vw[best of 6]"] = V.max(1)
out["attention_by_pair"] = {
    "spearman_vs_patching": {k: round(rho(v, T), 3) for k, v in by_pair.items()},
    "top10_overlap": {k: top10(v, T) for k, v in by_pair.items()},
    "role_heads": {lab(i): {"role": role_of[i], "patching_rank": int(ranks(T)[i]),
                            "attn": {p: round(float(A[i, j]), 3) for j, p in enumerate(PAIRS)},
                            "best_position": PAIRS[int(A[i].argmax())],
                            "rank_end>io": int(ranks(A[:, 0])[i]), "rank_best_of_5": int(ranks(A[:, FIVE].max(1))[i]),
                            "rank_best_of_6": int(ranks(A.max(1))[i]), "rank_vw_best_of_6": int(ranks(V.max(1))[i])}
                   for r, hs in roles.items() for i in hs},
}

# ---------- 2. agreement with patching, bootstrap over prompts and over templates ----------
def stats_for(sel):
    Ts = truth[sel].mean(0)
    return {m: (rho(f(sel), Ts), top10(f(sel), Ts)) for m, f in METHODS.items()}
point = stats_for(ALL)
base_tpl = np.unique(tpl)
boot_p = [stats_for(rng.integers(0, N, N)) for _ in range(B)]
boot_t = [stats_for(np.concatenate([np.where(tpl == t)[0] for t in rng.choice(base_tpl, len(base_tpl))])) for _ in range(B)]
def ci(boot, m, k):
    v = np.array([b[m][k] for b in boot]); lo, hi = np.percentile(v, [2.5, 97.5]); return [round(float(lo), 3), round(float(hi), 3)]
out["agreement_with_patching"] = {m: {"spearman": round(point[m][0], 3), "top10_overlap": point[m][1],
                                      "spearman_ci95_prompts": ci(boot_p, m, 0), "spearman_ci95_templates": ci(boot_t, m, 0),
                                      "top10_ci95_prompts": ci(boot_p, m, 1), "top10_ci95_templates": ci(boot_t, m, 1)}
                                  for m in METHODS}

# ---------- 3. attribution patching deviation ----------
Am = attr.mean(0); dev = Am - T
out["attr_patching_deviation"] = {
    "mean_abs_dev_by_layer": [round(float(np.abs(dev[l * H:(l + 1) * H]).mean()), 3) for l in range(L)],
    "by_role": {r: {"attr_sum": round(float(Am[hs].sum()), 3), "patching_sum": round(float(T[hs].sum()), 3),
                    "ratio": round(float(Am[hs].sum() / T[hs].sum()), 2) if abs(T[hs].sum()) > 1e-6 else None}
                for r, hs in roles.items()},
    "top10_abs_dev": [{"head": lab(i), "role": role_of.get(i, "other"), "attr": round(float(Am[i]), 3), "patching": round(float(T[i]), 3),
                       "share_prompts_attr_below_patching": round(float((attr[:, i] < truth[:, i]).mean()), 2)}
                      for i in np.argsort(-np.abs(dev))[:10]],
    "sum_single_head_patching": round(float(T.sum()), 3), "sum_attr_patching": round(float(Am.sum()), 3),
    "clean_logit_diff": round(float(ldiff.mean()), 3),
}

# ---------- 4. group knockouts ----------
nm, bk = roles["name_movers"], roles["backup_nm"]
def boot_mean(v):
    m = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(B)]; lo, hi = np.percentile(m, [2.5, 97.5])
    return [round(float(lo), 3), round(float(hi), 3)]
groups = {}
for c in ("pooled", "BABA", "ABBA"):
    sel = ALL if c == "pooled" else np.where(order == c)[0]; Tc = truth[sel].mean(0)
    rows = {}
    for key, heads in (("name_movers", nm), ("backups", bk), ("both", nm + bk)):
        v = gdrop[key][sel]
        rows[key] = {"together": round(float(v.mean()), 3), "ci95": boot_mean(v), "sum_of_single_heads": round(float(Tc[heads].sum()), 3)}
    rows["both"]["name_movers_plus_backups_separately"] = round(rows["name_movers"]["together"] + rows["backups"]["together"], 3)
    sep = rows["both"]["name_movers_plus_backups_separately"]   # a ratio is meaningless when the two separate effects cancel out
    rows["both"]["ratio_to_separately"] = round(rows["both"]["together"] / sep, 2) if abs(sep) >= 0.1 else None
    groups[c] = {"n": int(len(sel)), "clean_logit_diff": round(float(ldiff[sel].mean()), 3), **rows}
out["group_knockouts"] = groups

# ---------- 5. per order and per template ----------
def block(sel):
    Ts = truth[sel].mean(0)
    return {"n": int(len(sel)), "accuracy": round(float((ldiff[sel] > 0).mean()), 3), "mean_logit_diff": round(float(ldiff[sel].mean()), 3),
            "spearman": {m: round(rho(f(sel), Ts), 3) for m, f in METHODS.items()},
            "group_drop": {k: round(float(v[sel].mean()), 3) for k, v in gdrop.items()}}
out["by_order"] = {o: block(np.where(order == o)[0]) for o in ("BABA", "ABBA")}
out["by_template"] = {str(int(t)): block(np.where(tpl == t)[0]) for t in base_tpl}
cells = {(int(t), o): block(np.where((tpl == t) & (order == o))[0]) for t in base_tpl for o in ("BABA", "ABBA")}
out["by_template_and_order"] = {f"T{t + 1} {o}": v for (t, o), v in cells.items()}
cnt = lambda f: int(sum(f(int(t)) for t in base_tpl))
out["order_effects_across_templates"] = {
    "templates": int(len(base_tpl)), "prompts_per_cell": int(cells[(0, "BABA")]["n"]),
    "name_mover_knockout_positive_in_BABA": cnt(lambda t: cells[(t, "BABA")]["group_drop"]["name_movers"] > 0),
    "name_mover_knockout_negative_in_ABBA": cnt(lambda t: cells[(t, "ABBA")]["group_drop"]["name_movers"] < 0),
    "attention_end>io_agrees_better_in_ABBA": cnt(lambda t: cells[(t, "ABBA")]["spearman"]["attention (end>io)"] > cells[(t, "BABA")]["spearman"]["attention (end>io)"]),
    "attention_best_of_6_agrees_better_in_ABBA": cnt(lambda t: cells[(t, "ABBA")]["spearman"]["attention (best of 6)"] > cells[(t, "BABA")]["spearman"]["attention (best of 6)"]),
    "clean_logit_diff_range": [round(min(v["mean_logit_diff"] for v in cells.values()), 2), round(max(v["mean_logit_diff"] for v in cells.values()), 2)],
}
json.dump(out, open(OUT, "w"), indent=1)

# ---------- readable summary ----------
print(f"{N} prompts, {len(base_tpl)} base templates, {B} bootstrap reps, positions: {', '.join(PAIRS)}\n")
print("Agreement with activation patching (Spearman on |score|; 95% CI over prompts / over templates; top-10 overlap)")
for m, v in out["agreement_with_patching"].items():
    print(f"  {m:32s} {v['spearman']:.3f}  [{v['spearman_ci95_prompts'][0]:.2f}, {v['spearman_ci95_prompts'][1]:.2f}] / "
          f"[{v['spearman_ci95_templates'][0]:.2f}, {v['spearman_ci95_templates'][1]:.2f}]   top10 {v['top10_overlap']}")
print("\nRole heads: patching rank vs attention rank (end>io, best of 5, best of 6, value-weighted best of 6), best position")
for h, v in out["attention_by_pair"]["role_heads"].items():
    print(f"  {h:5s} {v['role']:15s} patch {v['patching_rank']:3d} | {v['rank_end>io']:3d} {v['rank_best_of_5']:3d} {v['rank_best_of_6']:3d} {v['rank_vw_best_of_6']:3d} | {v['best_position']}")
print("\nGroup knockouts (together [95% CI] / sum of single heads)")
for c, g in groups.items():
    print(f"  {c:6s} " + "  ".join(f"{k}: {g[k]['together']:+.3f} [{g[k]['ci95'][0]:+.2f}, {g[k]['ci95'][1]:+.2f}] / {g[k]['sum_of_single_heads']:+.3f}" for k in ("name_movers", "backups", "both"))
          + (f"  | both vs separately {g['both']['ratio_to_separately']}x" if g['both']['ratio_to_separately'] is not None else f"  | separately {g['both']['name_movers_plus_backups_separately']:+.3f}, so no ratio"))
print("\nOrder effects across templates:", out["order_effects_across_templates"])
print(f"\nwrote {OUT}")
