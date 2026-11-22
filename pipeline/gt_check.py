"""Second task: greater-than (Hanna, Liu & Variengien, NeurIPS 2023), same five scores per head.

Prompt: "The [NOUN] lasted from the year [XX][YY] to the year [XX]". GPT-2 should put its
probability on two-digit years after YY. Setup follows Hanna et al.:
  - nouns: their 120 FrameNet nouns (data/gt_nouns.txt, from github.com/hannamw/gpt2-greater-than,
    cache/potential_nouns.txt); all are single tokens, so every prompt has the same 13 tokens
  - century XX in 11..17, YY in 02..98, only years GPT-2 splits into [XX][YY], and each
    century's lowest and highest such year left out
  - metric: probability difference, sum of p(y) for y > YY minus sum for y <= YY (range -1..1)
  - corruption: the "01" prompt, YY replaced by 01 ("from the year 1701 to the year 17")
Balanced over YY: two prompts per start year 02..98 (194 prompts), random century and noun.

Per head and prompt (same meaning as the IOI pipeline):
  attn   attention at five query>key positions: end>yy, end>xx1, end>noun, yy>xx1, yy>noun
         (end = the final XX token, where the year is predicted)
  vw     the same, value-weighted (Kobayashi et al. 2020)
  dla    direct logit attribution onto mean W_U[years > YY] - mean W_U[years <= YY]
         (a logit-space version of the metric, since probability difference is not linear)
  attr   attribution patching: gradient of the probability difference times (corrupted - clean) z
  truth  activation patching, noising: clean metric - metric with the head's "01" output
  denoise_01  the reverse: metric with the head's clean output in the "01" run - "01" metric
Group effects (noising and denoising): the four layer 5-7 year heads, the three layer 8-9 year
heads, all seven, and MLPs 8-11 (which Hanna et al. find compute the comparison).

Writes results/gt/per_prompt.json, results/gt/attn_scores.json, results/gt/attn/L{l}_H{h}.json.
Run:  python pipeline/gt_check.py data/gt_nouns.txt results/gt
"""
import json, os, random, sys, time
import torch
from transformer_lens import HookedTransformer

NOUNS = [l.strip() for l in open(sys.argv[1] if len(sys.argv) > 1 else "data/gt_nouns.txt") if l.strip()]
OUT = sys.argv[2] if len(sys.argv) > 2 else "results/gt"
PER_YY = int(sys.argv[3]) if len(sys.argv) > 3 else 2
random.seed(0); torch.manual_seed(0)
t0 = time.time()
model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
L, H = model.cfg.n_layers, model.cfg.n_heads
torch.set_grad_enabled(False)
os.makedirs(f"{OUT}/attn", exist_ok=True)

# ---------- dataset ----------
assert all(len(model.to_tokens(" " + n, prepend_bos=False)[0]) == 1 for n in NOUNS), "a noun is not one token"
def valid(c, yy):
    t = model.to_str_tokens(f" {c}{yy:02d}", prepend_bos=False)
    return len(t) == 2 and len(t[1]) == 2
pool = {}
for c in range(11, 18):
    ys = [yy for yy in range(100) if valid(c, yy)]
    for yy in ys[1:-1]:                       # drop the century's lowest and highest valid year
        if 2 <= yy <= 98:
            pool.setdefault(yy, []).append(c)
assert all(valid(c, 1) for c in range(11, 18)), "'01' corruption must tokenize as two tokens"
prompts, corr, yys, nouns_used, cents = [], [], [], [], []
for _ in range(PER_YY):
    for yy in range(2, 99):
        c, noun = random.choice(pool[yy]), random.choice(NOUNS)
        prompts.append(f"The {noun} lasted from the year {c}{yy:02d} to the year {c}")
        corr.append(f"The {noun} lasted from the year {c}01 to the year {c}")
        yys.append(yy); nouns_used.append(noun); cents.append(c)
tc, tx = model.to_tokens(prompts), model.to_tokens(corr)
assert tc.shape == tx.shape and tc.shape[1] == 13, tc.shape
N, T = tc.shape
YEAR_TOK = torch.tensor([model.to_single_token(f"{y:02d}") for y in range(100)])
YY = torch.tensor(yys)
MASK = (torch.arange(100)[None, :] > YY[:, None]).float()           # [N, 100], years after YY
POS = {"noun": 2, "xx1": 7, "yy": 8, "end": T - 1}
assert model.to_str_tokens(tc[0])[POS["yy"]] == f"{yys[0]:02d}"
PAIRS = ["end>yy", "end>xx1", "end>noun", "yy>xx1", "yy>noun"]
QK = [(POS["end"], POS["yy"]), (POS["end"], POS["xx1"]), (POS["end"], POS["noun"]), (POS["yy"], POS["xx1"]), (POS["yy"], POS["noun"])]
BUCKETS = [(2, 19), (20, 39), (40, 59), (60, 79), (80, 98)]
bucket = [next(b for b, (lo, hi) in enumerate(BUCKETS) if lo <= yy <= hi) for yy in yys]
print(f"{N} prompts, {T} tokens, example {prompts[0]!r}", flush=True)

def last_logits(resid):
    """unembed the final position only (full-vocab logits at every position would not fit in memory)"""
    return model.unembed(model.ln_final(resid[:, -1:, :]))[:, 0]
def run_last(x, start_at_layer=None):
    return last_logits(model(x, start_at_layer=start_at_layer, stop_at_layer=L) if start_at_layer is not None else model(x, stop_at_layer=L))
def pdiff(logits, rows=None):
    """probability difference per row; logits are final-position [b, vocab]; rows index into the prompt list"""
    rows = torch.arange(logits.shape[0]) % N if rows is None else rows
    if logits.dim() == 3: logits = logits[:, -1]
    p = logits.softmax(-1)[:, YEAR_TOK]                              # [b, 100]
    m = MASK[rows]
    return (p * m).sum(-1) - (p * (1 - m)).sum(-1)

ZK = [f"blocks.{l}.attn.hook_z" for l in range(L)]
# cache only what is used (a full cache with vocab-sized logits for 194 prompts does not fit in 7 GB)
KEEP = ("hook_z", "hook_pattern", "hook_v", "hook_resid_pre", "hook_mlp_out", "ln_final.hook_scale")
_, cc = model.run_with_cache(tc, names_filter=lambda nm: nm.endswith(KEEP), return_type=None)
_, cx = model.run_with_cache(tx, names_filter=lambda nm: nm.endswith(KEEP), return_type=None)
pd_c, pd_x = pdiff(run_last(tc)), pdiff(run_last(tx))
print(f"probability difference: clean {pd_c.mean():.3f} (sd {pd_c.std():.3f}), '01' {pd_x.mean():.3f}", flush=True)

# ---------- attention and value-weighted attention ----------
attn = torch.zeros(N, L, H, len(PAIRS)); vwa = torch.zeros(N, L, H, len(PAIRS))
pattern_files = {(l, h): [] for l in range(L) for h in range(H)}
for l in range(L):
    pat = cc[f"blocks.{l}.attn.hook_pattern"]
    v = cc[f"blocks.{l}.attn.hook_v"]
    on = torch.einsum("nkhd,hdm->nkhm", v, model.W_O[l]).norm(dim=-1)
    vw = pat * on.permute(0, 2, 1)[:, :, None, :]; vw = vw / vw.sum(-1, keepdim=True).clamp_min(1e-9)
    for j, (q, k) in enumerate(QK):
        attn[:, l, :, j] = pat[:, :, q, k]; vwa[:, l, :, j] = vw[:, :, q, k]
    for b in range(len(BUCKETS)):
        sel = torch.tensor([i for i in range(N) if bucket[i] == b])
        mp, mv = pat[sel].mean(0), vw[sel].mean(0)
        for h in range(H):
            tri = lambda M: [[int(round(100 * float(M[h, q, k]))) for k in range(q + 1)] for q in range(T)]
            pattern_files[(l, h)].append({"tpl": b, "order": "all", "attn": tri(mp), "vw": tri(mv)})

import resource
def mem(tag):  # ru_maxrss is in bytes on macOS and in kilobytes on Linux
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // (1024 * 1024 if sys.platform == "darwin" else 1024)
    print(f"  [{tag}] peak {peak} MB, {time.time() - t0:.0f}s", flush=True)
mem("attention done")
# ---------- DLA ----------
# each head's output at the final position, centered and divided by the final LayerNorm scale
# (what cache.apply_ln_to_stack does), computed per layer to keep memory small
WU = model.W_U[:, YEAR_TOK]                                            # [d, 100]
dirs = (WU @ (MASK / MASK.sum(1, keepdim=True)).T) - (WU @ ((1 - MASK) / (1 - MASK).sum(1, keepdim=True)).T)   # [d, N]
scale = cc["ln_final.hook_scale"][:, -1]                               # [N, 1]
dla = torch.zeros(N, L, H)
for l in range(L):
    res = torch.einsum("nhd,hdm->nhm", cc[ZK[l]][:, -1], model.W_O[l])  # [N, H, d]
    res = (res - res.mean(-1, keepdim=True)) / scale[:, None, :]
    dla[:, l] = (res * dirs.T[:, None, :]).sum(-1)

mem("dla done")
# ---------- attribution patching ----------
CHUNK = 48                                   # prompts per chunk, to stay inside a 7 GB machine
CHUNKS = [torch.arange(i, min(i + CHUNK, N)) for i in range(0, N, CHUNK)]
attr = torch.zeros(N, L, H)
for rows in CHUNKS:
    torch.set_grad_enabled(True)
    grads = {}
    def keep(name):
        def h(t, hook):
            t.register_hook(lambda gr: grads.__setitem__(name, gr.detach())); return t
        return h
    with model.hooks(fwd_hooks=[(k, keep(k)) for k in ZK]):
        pdiff(run_last(tc[rows]), rows).sum().backward()
    model.zero_grad(set_to_none=True)
    torch.set_grad_enabled(False)
    for l in range(L):
        attr[rows, l] = -(grads[ZK[l]] * (cx[ZK[l]][rows] - cc[ZK[l]][rows])).sum(dim=(1, 3))

mem("attr done")
# ---------- activation patching, single heads (12 heads of a layer in one batch) ----------
def patch_layer(run_cache, src, base, sign):
    out = torch.zeros(N, L, H)
    for rows in CHUNKS:
        n = len(rows)
        for l in range(L):
            resid = run_cache[f"blocks.{l}.hook_resid_pre"][rows].repeat(H, 1, 1)
            s = src[ZK[l]][rows]
            def f(z, hook, s=s, n=n):
                for h in range(H):
                    z[h * n:(h + 1) * n, :, h] = s[:, :, h]
                return z
            with model.hooks(fwd_hooks=[(ZK[l], f)]):
                pl = pdiff(run_last(resid, start_at_layer=l), rows.repeat(H)).reshape(H, n).T
            out[rows, l] = (base[rows][:, None] - pl) if sign > 0 else (pl - base[rows][:, None])
    return out
truth = patch_layer(cc, cx, pd_c, +1); mem("noising done")
denoise = patch_layer(cx, cc, pd_x, -1); mem("denoising done")

# ---------- group effects ----------
EARLY = [(5, 1), (5, 5), (6, 9), (7, 10)]
LATE = [(8, 8), (8, 11), (9, 1)]
GROUPS = {"early_heads": EARLY, "late_heads": LATE, "both": EARLY + LATE}
def group(tokens, src, heads, base, sign, mlps=()):
    hooks = []
    for (l, h) in heads:
        def f(z, hook, l=l, h=h):
            z[:, :, h] = src[ZK[l]][:, :, h]; return z
        hooks.append((ZK[l], f))
    for l in mlps:
        k = f"blocks.{l}.hook_mlp_out"
        def g(x, hook, k=k):
            return src[k].clone()
        hooks.append((k, g))
    with model.hooks(fwd_hooks=hooks):
        pl = pdiff(run_last(tokens))
    return base - pl if sign > 0 else pl - base
gd = {"noise_01": {}, "denoise_01": {}}
for name, heads in GROUPS.items():
    gd["noise_01"][name] = group(tc, cx, heads, pd_c, +1)
    gd["denoise_01"][name] = group(tx, cc, heads, pd_x, -1)
gd["noise_01"]["mlps_8_11"] = group(tc, cx, [], pd_c, +1, mlps=range(8, 12))
gd["denoise_01"]["mlps_8_11"] = group(tx, cc, [], pd_x, -1, mlps=range(8, 12))

# ---------- write ----------
r4 = lambda t: [round(float(x), 4) for x in t.flatten().tolist()]
ROLES = {"yy_heads_early": [f"{l}.{h}" for l, h in EARLY], "yy_heads_late": [f"{l}.{h}" for l, h in LATE]}
toks = model.to_str_tokens(tc[0])
examples = [{"tpl": b, "order": "all", "toks": model.to_str_tokens(tc[bucket.index(b)]), "pos": POS} for b in range(len(BUCKETS))]
per = [{"text": prompts[i], "tpl": bucket[i], "order": "all", "yy": yys[i], "century": cents[i], "noun": nouns_used[i],
        "logit_diff": round(pd_c[i].item(), 4), "metric_corrupt": round(pd_x[i].item(), 4),
        "group_drop": {k: round(v[i].item(), 4) for k, v in gd["noise_01"].items()},
        "attn": [[round(float(x), 4) for x in row] for row in attn[i].reshape(L * H, -1).tolist()],
        "dla": r4(dla[i]), "attr": r4(attr[i]), "truth": r4(truth[i])} for i in range(N)]
json.dump({"task": "greater_than", "metric": "probability difference: p(year > YY) - p(year <= YY)",
           "note": "per-head arrays have 144 entries, index = layer*12 + head; logit_diff holds the task metric (probability difference)",
           "pairs": PAIRS, "roles": ROLES, "buckets": [f"{lo:02d}-{hi:02d}" for lo, hi in BUCKETS], "examples": examples,
           "prompts": per}, open(f"{OUT}/per_prompt.json", "w"), separators=(",", ":"))
json.dump({"pairs": PAIRS, "groups": [{"tpl": b, "order": "all", "toks": examples[b]["toks"], "pos": POS} for b in range(len(BUCKETS))],
           "prompts": [{"text": prompts[i], "tpl": bucket[i], "order": "all",
                        "attn": [[round(float(x), 4) for x in row] for row in attn[i].reshape(L * H, -1).tolist()],
                        "vw": [[round(float(x), 4) for x in row] for row in vwa[i].reshape(L * H, -1).tolist()]} for i in range(N)]},
          open(f"{OUT}/attn_scores.json", "w"), separators=(",", ":"))
json.dump({"variants": ["noise_01", "denoise_01"], "groups": {**{k: [f"{l}.{h}" for l, h in v] for k, v in GROUPS.items()}, "mlps_8_11": ["MLP 8", "MLP 9", "MLP 10", "MLP 11"]},
           "prompts": [{"text": prompts[i], "scores": {"noise_01": r4(truth[i]), "denoise_01": r4(denoise[i])},
                        "group": {v: {k: round(x[i].item(), 4) for k, x in gd[v].items()} for v in gd}} for i in range(N)]},
          open(f"{OUT}/patching_variants_per_prompt.json", "w"), separators=(",", ":"))
for (l, h), items in pattern_files.items():
    json.dump({"head": f"{l}.{h}", "groups": items}, open(f"{OUT}/attn/L{l}_H{h}.json", "w"), separators=(",", ":"))

# quick summary
from scipy.stats import spearmanr
def rho(a, b): return spearmanr(a.abs().flatten(), b.abs().flatten()).correlation
Tm = truth.mean(0)
print("Spearman vs patching (|mean|): attn end>yy %.3f, attn best %.3f, vw best %.3f, dla %.3f, attr %.3f, denoise %.3f" % (
    rho(attn[..., 0].mean(0), Tm), rho(attn.mean(0).max(-1).values, Tm), rho(vwa.mean(0).max(-1).values, Tm),
    rho(dla.mean(0), Tm), rho(attr.mean(0), Tm), rho(denoise.mean(0), Tm)))
top = Tm.abs().flatten().topk(10).indices.tolist()
print("patching top 10:", [f"{i // H}.{i % H}:{Tm.flatten()[i]:+.3f}" for i in top])
print("groups (noise / denoise):", {k: (round(gd['noise_01'][k].mean().item(), 3), round(gd['denoise_01'][k].mean().item(), 3)) for k in gd["noise_01"]})
print(f"wrote {OUT} in {time.time() - t0:.0f}s")
