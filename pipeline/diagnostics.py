"""Assumption checks: the measurements that say why each explanation method disagrees with patching.

Each per-head score equals "importance" only if an assumption holds. This script measures, for
every head, how badly each assumption is broken, using interventions that do not reuse the score
itself. Both tasks, same prompts as the earlier pipelines (IOI: same code and seed as
ioi_check_v3.py; greater-than: rebuilt from results/gt/per_prompt.json and checked).

Per head and prompt, under noising with the task's main corruption (IOI: ABC; greater-than: "01"):
  full      activation patching, all positions (alpha = 1): reproduces the v3 / gt reference
  half      the same patch at half strength, z = clean + 0.5 * (corrupted - clean)
            -> linearity: a quadratic through f(0), f(0.5), f(1) gives the slope at 0 (what
               attribution patching estimates) and the curvature
  end       patching at the final position only (greater-than; IOI already has it in
            patching_variants_per_prompt.json as noise_abc_end)
            -> does the head act at the final token (what final-token attention assumes)?
  direct    direct-path effect: the head's corrupted output replaces its clean output only in
            the final residual stream (frozen final LayerNorm scale), so nothing downstream sees it
            -> how much of the effect is direct (what DLA assumes)?
Pairwise interactions for the 24 heads with the largest mean |full| effect:
  pair      both heads patched together; interaction = pair - (single i + single j)
            -> does a head's effect depend on other heads (what single-head patching assumes)?

Writes results/diagnostics_ioi.json and results/gt/diagnostics.json (per-prompt arrays).
Run:  python pipeline/diagnostics.py data/ioi_lists.json results [ioi|gt|both] [n_groups_for_test]
"""
import json, os, random, sys, time
import torch
from transformer_lens import HookedTransformer

lists = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "data/ioi_lists.json"))
OUT = sys.argv[2] if len(sys.argv) > 2 else "results"
WHICH = sys.argv[3] if len(sys.argv) > 3 else "both"
TEST = int(sys.argv[4]) if len(sys.argv) > 4 else None
K = 24
t0 = time.time()
model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
L, H = model.cfg.n_layers, model.cfg.n_heads
torch.set_grad_enabled(False)
ZK = [f"blocks.{l}.attn.hook_z" for l in range(L)]
KEEP = lambda nm: nm.endswith("hook_z") or nm.endswith("hook_resid_pre") or nm == "ln_final.hook_scale"

def last_logits(resid):
    return model.unembed(model.ln_final(resid[:, -1:, :]))[:, 0]
def run_last(x, start=None):
    return last_logits(model(x, start_at_layer=start, stop_at_layer=L) if start is not None else model(x, stop_at_layer=L))

def sweep(b, alpha=1.0, end_only=False):
    """Noising every head, 12 heads of a layer per batch. Returns [n, L, H] of clean metric - patched metric."""
    n = b["tc"].shape[0]; out = torch.zeros(n, L, H)
    for l in range(L):
        resid = b["cc"][f"blocks.{l}.hook_resid_pre"].repeat(H, 1, 1)
        zc, zx = b["cc"][ZK[l]], b["cx"][ZK[l]]
        src = zc + alpha * (zx - zc)
        def f(z, hook, src=src):
            for h in range(H):
                if end_only: z[h * n:(h + 1) * n, -1, h] = src[:, -1, h]
                else: z[h * n:(h + 1) * n, :, h] = src[:, :, h]
            return z
        with model.hooks(fwd_hooks=[(ZK[l], f)]):
            m = b["metric"](run_last(resid, l), H).reshape(H, n).T
        out[:, l] = b["mc"][:, None] - m
    return out

def direct(b):
    """Direct-path noising effect of every head (final residual only, frozen final LayerNorm)."""
    n = b["tc"].shape[0]; out = torch.zeros(n, L, H)
    scale = b["cc"]["ln_final.hook_scale"][:, -1]                          # [n, 1]
    for l in range(L):
        d = torch.einsum("nhd,hdm->nhm", b["cx"][ZK[l]][:, -1] - b["cc"][ZK[l]][:, -1], model.W_O[l])   # [n, H, d]
        d = (d - d.mean(-1, keepdim=True)) / scale[:, None, :]
        for h in range(H):
            out[:, l, h] = b["mc"] - b["metric"](b["lc"] + d[:, h] @ model.W_U, 1)
    return out

def pairs_effect(b, pairs, chunk=46):
    """Noising two heads together, many pairs per forward pass. Returns [n, n_pairs]."""
    n = b["tc"].shape[0]; out = torch.zeros(n, len(pairs))
    for c0 in range(0, len(pairs), chunk):
        ps = pairs[c0:c0 + chunk]; P = len(ps)
        hooks = []
        for l in range(L):
            rows = [(k, h) for k, (a, bb) in enumerate(ps) for (ll, h) in (a, bb) if ll == l]
            if not rows: continue
            def f(z, hook, rows=rows, l=l):
                zx = b["cx"][ZK[l]]
                for k, h in rows: z[k * n:(k + 1) * n, :, h] = zx[:, :, h]
                return z
            hooks.append((ZK[l], f))
        with model.hooks(fwd_hooks=hooks):
            m = b["metric"](run_last(b["tc"].repeat(P, 1)), P).reshape(P, n).T
        out[:, c0:c0 + P] = b["mc"][:, None] - m
    return out

def prepare(tc, tx, metric):
    _, cc = model.run_with_cache(tc, names_filter=KEEP, return_type=None)
    _, cx = model.run_with_cache(tx, names_filter=KEEP, return_type=None)
    lc = run_last(tc)
    b = {"tc": tc, "tx": tx, "cc": cc, "cx": cx, "lc": lc, "metric": metric}
    b["mc"] = metric(lc, 1); b["mx"] = metric(run_last(tx), 1)
    return b

def run(batches, top_pairs, with_end):
    res = {k: [] for k in ("full", "half", "end", "direct", "pair", "mc", "mx")}
    for i, mk in enumerate(batches):
        b = mk()
        res["full"].append(sweep(b)); res["half"].append(sweep(b, 0.5))
        if with_end: res["end"].append(sweep(b, end_only=True))
        res["direct"].append(direct(b)); res["pair"].append(pairs_effect(b, top_pairs))
        res["mc"].append(b["mc"]); res["mx"].append(b["mx"])
        print(f"  batch {i + 1}/{len(batches)} at {time.time() - t0:.0f}s", flush=True)
        del b
    return {k: torch.cat(v) for k, v in res.items() if v}

def top_k_pairs(mean_effect):
    idx = mean_effect.abs().flatten().topk(K).indices.tolist()
    heads = [(i // H, i % H) for i in sorted(idx)]
    return heads, [(heads[a], heads[c]) for a in range(K) for c in range(a + 1, K)]

r4 = lambda t: [round(float(x), 4) for x in t.flatten().tolist()]
def write(path, meta, R, heads, pairs, extra):
    n = len(meta)
    json.dump({"note": "per-head arrays: 144 entries, index = layer*12 + head; effects are clean metric minus patched metric (noising)",
               "top_heads": [f"{l}.{h}" for l, h in heads], "pairs": [[f"{a[0]}.{a[1]}", f"{c[0]}.{c[1]}"] for a, c in pairs], **extra,
               "prompts": [dict(meta[i], metric_clean=round(R["mc"][i].item(), 4), metric_corrupt=round(R["mx"][i].item(), 4),
                                full=r4(R["full"][i]), half=r4(R["half"][i]), direct=r4(R["direct"][i]),
                                **({"end": r4(R["end"][i])} if "end" in R else {}), pair=r4(R["pair"][i])) for i in range(n)]},
              open(path, "w"), separators=(",", ":"))

# ------------------------------- IOI -------------------------------
if WHICH in ("ioi", "both"):
    random.seed(0); torch.manual_seed(0)
    def one_tok(w): return len(model.to_tokens(" " + w, prepend_bos=False)[0]) == 1
    names = [n for n in lists["NAMES"] if one_tok(n)]
    places = [p for p in lists["PLACES"] if one_tok(p)]
    objects = [o for o in lists["OBJECTS"] if one_tok(o)]
    def fill(tpl, A, B, C, place, obj):
        t = tpl.replace("[PLACE]", place).replace("[OBJECT]", obj)
        t = t[: -len(" [A]")]
        first_b = t.index("[B]")
        t = t[:first_b] + B + t[first_b + 3:]
        return t.replace("[A]", A).replace("[B]", C)
    groups = []
    for ti, tpl in enumerate(lists["BABA_TEMPLATES"]):
        for order in ("BABA", "ABBA"):
            base = tpl
            if order == "ABBA":
                i, j = base.index("[B]"), base.index("[A]")
                base = base[:i] + "[TMP]" + base[i + 3:j] + "[B]" + base[j + 3:]
                base = base.replace("[TMP]", "[A]", 1)
            clean, corr, io, s = [], [], [], []
            for _ in range(4):
                A, B, X, Y, Z = random.sample(names, 5)
                place, obj = random.choice(places), random.choice(objects)
                clean.append(fill(base, A, B, B, place, obj)); corr.append(fill(base, X, Y, Z, place, obj))
                io.append(model.to_single_token(" " + A)); s.append(model.to_single_token(" " + B))
            groups.append(dict(clean=clean, corr=corr, io=torch.tensor(io), s=torch.tensor(s), tpl=ti, order=order))
    if TEST: groups = groups[:TEST]
    def ioi_batch(g):
        def mk():
            io, s = g["io"], g["s"]
            def metric(last, rep):
                i = torch.arange(last.shape[0]); return last[i, io.repeat(rep)] - last[i, s.repeat(rep)]
            return prepare(model.to_tokens(g["clean"]), model.to_tokens(g["corr"]), metric)
        return mk
    v3 = json.load(open(f"{OUT}/ioi_v3_results_per_prompt.json"))["prompts"]
    meta = [{"text": c, "tpl": g["tpl"], "order": g["order"]} for g in groups for c in g["clean"]]
    assert [p["text"] for p in v3[:len(meta)]] == [m["text"] for m in meta], "IOI prompts differ from v3"
    ref = torch.tensor([p["truth"] for p in v3]).mean(0)                 # pick the top heads from the full reference
    heads, pairs = top_k_pairs(ref)
    print(f"IOI: {len(meta)} prompts, {len(pairs)} pairs among {K} heads", flush=True)
    R = run([ioi_batch(g) for g in groups], pairs, with_end=False)
    err = (R["full"].reshape(len(meta), -1) - torch.tensor([p["truth"] for p in v3[:len(meta)]])).abs().max().item()
    print(f"IOI full vs v3 truth: max abs diff {err:.5f}", flush=True); assert err < 2e-3
    write(f"{OUT}/diagnostics_ioi.json" if not TEST else f"{OUT}/diagnostics_ioi_test.json", meta, R, heads, pairs,
          {"task": "ioi", "corruption": "ABC (three fresh names)"})

# --------------------------- greater-than ---------------------------
if WHICH in ("gt", "both"):
    gt = json.load(open(f"{OUT}/gt/per_prompt.json"))["prompts"]
    if TEST: gt = gt[:48 * TEST]
    corr = []
    for p in gt:
        start = f"{p['century']}{p['yy']:02d}"; i = p["text"].index(start)
        corr.append(p["text"][:i] + f"{p['century']}01" + p["text"][i + 4:])
    YEAR = torch.tensor([model.to_single_token(f"{y:02d}") for y in range(100)])
    def gt_batch(rows):
        def mk():
            yy = torch.tensor([gt[r]["yy"] for r in rows]); mask = (torch.arange(100)[None, :] > yy[:, None]).float()
            def metric(last, rep):
                p = last.softmax(-1)[:, YEAR]; m = mask.repeat(rep, 1)
                return (p * m).sum(-1) - (p * (1 - m)).sum(-1)
            tc = model.to_tokens([gt[r]["text"] for r in rows]); tx = model.to_tokens([corr[r] for r in rows])
            assert tc.shape == tx.shape
            return prepare(tc, tx, metric)
        return mk
    chunks = [list(range(i, min(i + 24, len(gt)))) for i in range(0, len(gt), 24)]   # 24 x 12 heads = 288 rows per pass
    ref = torch.tensor([p["truth"] for p in json.load(open(f"{OUT}/gt/per_prompt.json"))["prompts"]]).mean(0)
    heads, pairs = top_k_pairs(ref)
    print(f"greater-than: {len(gt)} prompts, {len(pairs)} pairs among {K} heads", flush=True)
    R = run([gt_batch(c) for c in chunks], pairs, with_end=True)
    err = (R["full"].reshape(len(gt), -1) - torch.tensor([p["truth"] for p in gt])).abs().max().item()
    print(f"greater-than full vs gt truth: max abs diff {err:.5f}", flush=True); assert err < 2e-3
    meta = [{"text": p["text"], "tpl": p["tpl"], "yy": p["yy"]} for p in gt]
    write(f"{OUT}/gt/diagnostics.json" if not TEST else f"{OUT}/gt/diagnostics_test.json", meta, R, heads, pairs,
          {"task": "gt", "corruption": "start year changed to XX01"})
print(f"done in {time.time() - t0:.0f}s")
