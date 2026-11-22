"""Where should attention be read? Agreement with patching at every query > key position.

So far attention was scored at a few positions we picked by hand (six for IOI, five for
greater-than). This script scores it everywhere:
  IOI:  prompts differ in length across templates, so positions are aligned by role:
        BOS, S1, S1+1, IO, IO+1, S2, S2+1, END (8 x 8 query > key cells; attention to a later
        token is 0 by the causal mask, so some cells are 0 for one name order).
  greater-than: every prompt has the same 13 tokens, so all 13 x 13 token pairs.
For each cell, the per-head score is mean attention at that cell, and agreement is Spearman on
|score| against single-head patching over the 144 heads (as everywhere else).

It also checks how much of the "best position" gain is selection. Rules compared:
  hand-picked: each head's max over our hand-picked positions (the "best of 6/5" score)
  all cells:   each head's max over every cell (no choice involved, but more chances)
  greedy k:    pick k cells on one half of the data to maximize agreement of the max over
               those cells, then measure agreement on the other half (k = 1..6)
Halves: IOI splits the 15 base templates (8 / 7), greater-than splits prompts at random.
200 random splits, numpy seed 0.

Writes results/attn_map.json and results/gt/attn_map.json (mean attention per cell for the
page, which computes agreement against whichever patching setting is selected) and
results/analysis_attn_map.json.

Run:  python pipeline/attention_map.py data/ioi_lists.json results
"""
import json, os, random, sys, time
import numpy as np
import torch
from scipy.stats import spearmanr
from transformer_lens import HookedTransformer

lists = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "data/ioi_lists.json"))
OUT = sys.argv[2] if len(sys.argv) > 2 else "results"
SPLITS = 200
random.seed(0); torch.manual_seed(0)
t0 = time.time()
model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
L, H = model.cfg.n_layers, model.cfg.n_heads
torch.set_grad_enabled(False)
KEEP = lambda nm: nm.endswith("hook_pattern") or nm.endswith("hook_v")

def vw_of(cache, l):
    pat = cache[f"blocks.{l}.attn.hook_pattern"]; v = cache[f"blocks.{l}.attn.hook_v"]
    on = torch.einsum("nkhd,hdm->nkhm", v, model.W_O[l]).norm(dim=-1)
    vw = pat * on.permute(0, 2, 1)[:, :, None, :]
    return pat, vw / vw.sum(-1, keepdim=True).clamp_min(1e-9)

# ---------------- IOI: same prompts as ioi_check_v3.py ----------------
def one_tok(w):
    return len(model.to_tokens(" " + w, prepend_bos=False)[0]) == 1
names = [n for n in lists["NAMES"] if one_tok(n)]
places = [p for p in lists["PLACES"] if one_tok(p)]
objects = [o for o in lists["OBJECTS"] if one_tok(o)]
def fill(tpl, A, B, C, place, obj):
    t = tpl.replace("[PLACE]", place).replace("[OBJECT]", obj)
    t = t[: -len(" [A]")]
    first_b = t.index("[B]")
    t = t[:first_b] + B + t[first_b + 3:]
    return t.replace("[A]", A).replace("[B]", C)
ROLES = ["BOS", "S1", "S1+1", "IO", "IO+1", "S2", "S2+1", "END"]
R = len(ROLES)
ioi_attn, ioi_vw, ioi_meta = [], [], []
for ti, tpl in enumerate(lists["BABA_TEMPLATES"]):
    for order in ("BABA", "ABBA"):
        base = tpl
        if order == "ABBA":
            i, j = base.index("[B]"), base.index("[A]")
            base = base[:i] + "[TMP]" + base[i + 3:j] + "[B]" + base[j + 3:]
            base = base.replace("[TMP]", "[A]", 1)
        clean, io = [], []
        for _ in range(4):
            A, B, X, Y, Z = random.sample(names, 5)
            place, obj = random.choice(places), random.choice(objects)
            clean.append(fill(base, A, B, B, place, obj)); io.append((A, B))
        tc = model.to_tokens(clean)
        _, cache = model.run_with_cache(tc, names_filter=KEEP)
        T = tc.shape[1]
        pos = []
        for r, (A, B) in enumerate(io):
            ia, ib = model.to_single_token(" " + A), model.to_single_token(" " + B)
            s = (tc[r] == ib).nonzero().flatten().tolist(); p_io = (tc[r] == ia).nonzero()[0].item()
            p = {"BOS": 0, "S1": s[0], "S1+1": s[0] + 1, "IO": p_io, "IO+1": p_io + 1, "S2": s[1], "S2+1": s[1] + 1, "END": T - 1}
            assert len(set(p.values())) == R, (clean[r], p)
            pos.append([p[k] for k in ROLES])
        pos = torch.tensor(pos)                                         # [4, R]
        A_ = torch.zeros(4, L, H, R, R); V_ = torch.zeros(4, L, H, R, R)
        for l in range(L):
            pat, vw = vw_of(cache, l)                                   # [4, H, T, T]
            for n in range(4):
                idx = pos[n]
                A_[n, l] = pat[n][:, idx][:, :, idx]; V_[n, l] = vw[n][:, idx][:, :, idx]
        ioi_attn.append(A_); ioi_vw.append(V_)
        ioi_meta += [{"text": c, "tpl": ti, "order": order} for c in clean]
ioi_attn = torch.cat(ioi_attn).reshape(-1, L * H, R, R).numpy(); ioi_vw = torch.cat(ioi_vw).reshape(-1, L * H, R, R).numpy()
v3 = json.load(open(f"{OUT}/ioi_v3_results_per_prompt.json"))
assert [p["text"] for p in v3["prompts"]] == [m["text"] for m in ioi_meta], "prompt order differs from the v3 file"
print(f"IOI: {len(ioi_meta)} prompts, {time.time() - t0:.0f}s", flush=True)

# ---------------- greater-than: same prompts as gt_check.py ----------------
gt = json.load(open(f"{OUT}/gt/per_prompt.json"))
gt_texts = [p["text"] for p in gt["prompts"]]
gt_attn, gt_vw = [], []
for i in range(0, len(gt_texts), 48):
    tc = model.to_tokens(gt_texts[i:i + 48])
    _, cache = model.run_with_cache(tc, names_filter=KEEP)
    a = torch.zeros(tc.shape[0], L, H, tc.shape[1], tc.shape[1]); v = torch.zeros_like(a)
    for l in range(L):
        pat, vw = vw_of(cache, l); a[:, l] = pat; v[:, l] = vw
    gt_attn.append(a); gt_vw.append(v)
gt_attn = torch.cat(gt_attn).reshape(len(gt_texts), L * H, 13, 13).numpy(); gt_vw = torch.cat(gt_vw).reshape(len(gt_texts), L * H, 13, 13).numpy()
GT_TOKS = gt["examples"][0]["toks"]; POS = gt["examples"][0]["pos"]
GT_LABELS = [("YY" if k == POS["yy"] else "XX start" if k == POS["xx1"] else "XX end" if k == POS["end"] else "NOUN" if k == POS["noun"]
              else "BOS" if t == "<|endoftext|>" else t.strip()) for k, t in enumerate(GT_TOKS)]
_seen = {}
for k, t in enumerate(GT_LABELS):                       # "the" and "year" appear twice: number the second one
    _seen[t] = _seen.get(t, 0) + 1
    if _seen[t] > 1: GT_LABELS[k] = f"{t} ({_seen[t]})"
print(f"greater-than: {len(gt_texts)} prompts, {time.time() - t0:.0f}s", flush=True)

# ---------------- agreement maps and the selection check ----------------
def rho(a, b):
    a, b = np.abs(a), np.abs(b)
    if np.ptp(a) == 0: return None
    return float(spearmanr(a, b).correlation)
def cell_map(att, truth):       # att [n, 144, Q, K], truth [n, 144] -> Q x K Spearman
    M, T = att.mean(0), truth.mean(0)
    return [[rho(M[:, q, k], T) for k in range(M.shape[2])] for q in range(M.shape[1])]
def best_over(M, cells):        # per-head max over the given (q, k) cells of the mean attention M [144, Q, K]
    return np.max(np.stack([M[:, q, k] for q, k in cells], 1), 1)
def valid_cells(att):
    M = att.mean(0); return [(q, k) for q in range(M.shape[1]) for k in range(M.shape[2]) if np.ptp(M[:, q, k]) > 0]
def greedy(M, T, cells, kmax):
    chosen, cur = [], None
    for _ in range(kmax):
        def score(c):
            v = M[:, c[0], c[1]] if cur is None else np.maximum(cur, M[:, c[0], c[1]])
            return rho(v, T) or -1
        best = max((c for c in cells if c not in chosen), key=score)
        chosen.append(best); cur = best_over(M, chosen)
    return chosen
def selection_check(att, truth, hand, groups, rng):
    cells = valid_cells(att)
    Mall = att.mean(0)
    out = {"hand_picked": rho(best_over(Mall, hand), truth.mean(0)), "all_cells": rho(best_over(Mall, cells), truth.mean(0)),
           "n_cells": len(cells), "splits": SPLITS}
    ug = np.unique(groups); held = {k: [] for k in range(1, 7)}; insample = {k: [] for k in range(1, 7)}; hand_held = []
    picks = {}
    for s in range(SPLITS):
        g_a = rng.choice(ug, len(ug) // 2 + len(ug) % 2, replace=False)
        A = np.isin(groups, g_a); B = ~A
        MA, MB, TA, TB = att[A].mean(0), att[B].mean(0), truth[A].mean(0), truth[B].mean(0)
        ch = greedy(MA, TA, cells, 6)
        for k in range(1, 7):
            insample[k].append(rho(best_over(MA, ch[:k]), TA))
            held[k].append(rho(best_over(MB, ch[:k]), TB))
        hand_held.append(rho(best_over(MB, hand), TB))
        for c in ch[:3]: picks[c] = picks.get(c, 0) + 1
    q = lambda v: [round(float(np.mean(v)), 3), [round(float(x), 3) for x in np.percentile(v, [2.5, 97.5])]]
    out["greedy"] = {k: {"in_sample_mean_ci": q(insample[k]), "held_out_mean_ci": q(held[k])} for k in range(1, 7)}
    out["hand_picked_held_out_mean_ci"] = q(hand_held)
    out["most_picked_cells"] = sorted(picks.items(), key=lambda x: -x[1])[:8]
    return out

rng = np.random.default_rng(0)
var = json.load(open(f"{OUT}/patching_variants_per_prompt.json"))
ioi_ref = {v: np.array([p["scores"][v] for p in var["prompts"]]) for v in var["variants"]}
ri = {r: i for i, r in enumerate(ROLES)}
HAND_IOI = [(ri["END"], ri["IO"]), (ri["END"], ri["S1"]), (ri["END"], ri["S2"]), (ri["S2"], ri["S1"]), (ri["S2"], ri["S1+1"]), (ri["S1+1"], ri["S1"])]
order = np.array([m["order"] for m in ioi_meta]); tpl = np.array([m["tpl"] for m in ioi_meta])
gvar = json.load(open(f"{OUT}/gt/patching_variants_per_prompt.json"))
gt_ref = {v: np.array([p["scores"][v] for p in gvar["prompts"]]) for v in gvar["variants"]}
HAND_GT = [(POS["end"], POS["yy"]), (POS["end"], POS["xx1"]), (POS["end"], POS["noun"]), (POS["yy"], POS["xx1"]), (POS["yy"], POS["noun"])]

def lab_ioi(c): return f"{ROLES[c[0]]}>{ROLES[c[1]]}"
def lab_gt(c): return f"{GT_LABELS[c[0]]}>{GT_LABELS[c[1]]}"
analysis = {"ioi": {"roles": ROLES, "maps": {}, "top_cells": {}}, "gt": {"tokens": GT_LABELS, "maps": {}, "top_cells": {}}}
for v, tr in ioi_ref.items():
    for c in ("pooled", "BABA", "ABBA"):
        sel = np.arange(len(order)) if c == "pooled" else np.where(order == c)[0]
        m = cell_map(ioi_attn[sel], tr[sel]); analysis["ioi"]["maps"][f"{v}/{c}"] = m
        flat = sorted(((m[q][k], (q, k)) for q in range(R) for k in range(R) if m[q][k] is not None), reverse=True)[:5]
        analysis["ioi"]["top_cells"][f"{v}/{c}"] = [[lab_ioi(cc), round(x, 3)] for x, cc in flat]
for v, tr in gt_ref.items():
    m = cell_map(gt_attn, tr); analysis["gt"]["maps"][v] = m
    flat = sorted(((m[q][k], (q, k)) for q in range(13) for k in range(13) if m[q][k] is not None), reverse=True)[:5]
    analysis["gt"]["top_cells"][v] = [[lab_gt(cc), round(x, 3)] for x, cc in flat]
print(f"maps done, {time.time() - t0:.0f}s", flush=True)
sc = selection_check(ioi_attn, ioi_ref["noise_abc"], HAND_IOI, tpl, rng)
sc["most_picked_cells"] = [[lab_ioi(c), n] for c, n in sc["most_picked_cells"]]; analysis["ioi"]["selection_noise_abc"] = sc
print("IOI selection:", json.dumps({k: sc[k] for k in ("hand_picked", "all_cells", "hand_picked_held_out_mean_ci")}), flush=True)
for k in range(1, 7): print(f"  greedy k={k}:", sc["greedy"][k])
print("  most picked:", sc["most_picked_cells"], f"{time.time() - t0:.0f}s", flush=True)
gid = rng.permutation(len(gt_texts)) % 20                # 20 random groups of prompts, split 10 / 10
sg = selection_check(gt_attn, gt_ref["noise_01"], HAND_GT, gid, rng)
sg["most_picked_cells"] = [[lab_gt(c), n] for c, n in sg["most_picked_cells"]]; analysis["gt"]["selection_noise_01"] = sg
print("GT selection:", json.dumps({k: sg[k] for k in ("hand_picked", "all_cells", "hand_picked_held_out_mean_ci")}))
for k in range(1, 7): print(f"  greedy k={k}:", sg["greedy"][k])
print("  most picked:", sg["most_picked_cells"])
for t in ("ioi", "gt"):
    for k, v in analysis[t]["top_cells"].items(): print(t, k, v)

# ---------------- files for the page ----------------
r3 = lambda a: np.round(a, 3).tolist()
conds = {"all": np.arange(len(order)), "BABA": np.where(order == "BABA")[0], "ABBA": np.where(order == "ABBA")[0]}
json.dump({"labels": ROLES, "hand": [list(c) for c in HAND_IOI], "note": "attn[cond][head][query][key] = mean attention over prompts; roles aligned per prompt",
           "attn": {c: r3(ioi_attn[s].mean(0)) for c, s in conds.items()}, "vw": {c: r3(ioi_vw[s].mean(0)) for c, s in conds.items()}},
          open(f"{OUT}/attn_map.json", "w"), separators=(",", ":"))
json.dump({"labels": GT_LABELS, "hand": [list(c) for c in HAND_GT], "note": "attn[cond][head][query][key] = mean attention over prompts",
           "attn": {"all": r3(gt_attn.mean(0))}, "vw": {"all": r3(gt_vw.mean(0))}},
          open(f"{OUT}/gt/attn_map.json", "w"), separators=(",", ":"))
json.dump(analysis, open(f"{OUT}/analysis_attn_map.json", "w"), indent=1)
print(f"wrote {OUT}/attn_map.json, {OUT}/gt/attn_map.json, {OUT}/analysis_attn_map.json in {time.time() - t0:.0f}s")
