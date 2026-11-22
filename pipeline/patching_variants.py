"""Does the reference move? Activation patching under five settings.

Every comparison so far scores the explanation methods against one patching setup:
ABC corruption (three fresh names), noising (run the clean prompt, swap in one head's
corrupted output), at all positions. Each of those is a choice. This script reruns
single-head patching and the group knockouts under five settings, on exactly the same
120 prompts as ioi_check_v3.py (same code, same seed; checked against the v3 file).

  noise_abc      the v3 reference: clean run, head output from the ABC prompt, all positions.
                 Recomputed here and checked against v3's "truth" as a sanity test.
  denoise_abc    the reverse: run the ABC prompt, put the head's clean output back, and
                 measure how much logit diff comes back. Noising asks "is this head
                 necessary?", denoising asks "is it sufficient?" (Heimersheim & Nanda 2024).
                 They should differ most for redundant heads such as the backup name movers.
  noise_flip     a different corruption: the second mention of S is replaced with the IO name
                 ("..., Mary gave a drink to" instead of "..., John gave a drink to"), so the
                 corrupted prompt prefers the other name. Same template, same tokens elsewhere.
  noise_mean     mean ablation: the head's output is replaced by its average over 32 fresh ABC
                 prompts of the same template and name order (the corruption Wang et al. use).
  noise_abc_end  ABC noising at the final position only. Heads that act earlier in the
                 sentence (induction, duplicate, previous-token, S-inhibition at S2) should
                 drop out, which shows how much the "all positions" choice matters.

Score per head and prompt:
  noising   = clean logit diff - patched logit diff  (positive = the head helped)
  denoising = patched logit diff - corrupted logit diff  (positive = the head restores it)

To keep it fast on CPU, the 12 heads of a layer are patched in one batch: the prompts are
repeated 12 times and copy j gets head j patched, starting the forward pass at that layer.

Writes results/patching_variants_per_prompt.json (same prompt order as the v3 file).

Run:  python pipeline/patching_variants.py data/ioi_lists.json results
"""
import json, os, random, sys, time
import torch
from transformer_lens import HookedTransformer

lists = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "ioi_lists.json"))
OUT = sys.argv[2] if len(sys.argv) > 2 else "results"
PER_TEMPLATE = int(sys.argv[3]) if len(sys.argv) > 3 else 4
ONLY_GROUPS = int(sys.argv[4]) if len(sys.argv) > 4 else None   # for quick tests
N_MEAN = 32
random.seed(0); torch.manual_seed(0)
t0 = time.time()
model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
L, H = model.cfg.n_layers, model.cfg.n_heads
torch.set_grad_enabled(False)

# ---------- prompts: identical to ioi_check_v3.py (do not change the order of random calls) ----------
def one_tok(w):
    return len(model.to_tokens(" " + w, prepend_bos=False)[0]) == 1
names = [n for n in lists["NAMES"] if one_tok(n)]
places = [p for p in lists["PLACES"] if one_tok(p)]
objects = [o for o in lists["OBJECTS"] if one_tok(o)]

def fill(tpl, A, B, C, place, obj):
    # BABA template: "[B] and [A] ... [B] ... to [A]" ; prompt stops before the final [A]
    t = tpl.replace("[PLACE]", place).replace("[OBJECT]", obj)
    assert t.endswith(" [A]")
    t = t[: -len(" [A]")]
    first_b = t.index("[B]")
    t = t[:first_b] + B + t[first_b + 3:]          # first mention of B
    t = t.replace("[A]", A).replace("[B]", C)      # second mention of B -> C (C=B for clean)
    return t

groups = []
for ti, tpl in enumerate(lists["BABA_TEMPLATES"]):
    for order in ("BABA", "ABBA"):
        base = tpl
        if order == "ABBA":
            i, j = base.index("[B]"), base.index("[A]")
            base = base[:i] + "[TMP]" + base[i + 3:j] + "[B]" + base[j + 3:]
            base = base.replace("[TMP]", "[A]", 1)
        clean, corr, flip, io, s = [], [], [], [], []
        for _ in range(PER_TEMPLATE):
            A, B, X, Y, Z = random.sample(names, 5)
            place, obj = random.choice(places), random.choice(objects)
            c = fill(base, A, B, B, place, obj)
            x = fill(base, X, Y, Z, place, obj)      # ABC corruption (same as v3)
            clean.append(c); corr.append(x)
            flip.append(fill(base, A, B, A, place, obj))   # S2 -> IO name: prefers B now
            io.append(model.to_single_token(" " + A)); s.append(model.to_single_token(" " + B))
        tc, tx = model.to_tokens(clean), model.to_tokens(corr)
        if tc.shape != tx.shape or len(set(len(model.to_tokens(p)[0]) for p in clean)) != 1:
            continue
        tf = model.to_tokens(flip)
        assert tf.shape == tc.shape, "flip prompt changed the token count"
        groups.append(dict(tc=tc, tx=tx, tf=tf, io=torch.tensor(io), s=torch.tensor(s), text=clean,
                           tpl=ti, order=order, base=base))
# extra ABC prompts for mean ablation, from a separate generator so the prompts above stay as in v3
mrng = random.Random(12345)
for g in groups:
    T = g["tc"].shape[1]; pool = []
    while len(pool) < N_MEAN:
        X, Y, Z = mrng.sample(names, 3)
        t = model.to_tokens(fill(g["base"], X, Y, Z, mrng.choice(places), mrng.choice(objects)))
        if t.shape[1] == T:
            pool.append(t[0])
    g["tm"] = torch.stack(pool)
if ONLY_GROUPS:
    groups = groups[:ONLY_GROUPS]
N = sum(len(g["io"]) for g in groups)
print(f"{len(groups)} groups, {N} prompts, built in {time.time() - t0:.0f}s", flush=True)

def ld(logits, g):
    last = logits[:, -1]; i = torch.arange(last.shape[0]); n = len(g["io"])
    io, s = g["io"].repeat(last.shape[0] // n), g["s"].repeat(last.shape[0] // n)
    return last[i, io] - last[i, s]

ZK = [f"blocks.{l}.attn.hook_z" for l in range(L)]
NM = [(9, 6), (9, 9), (10, 0)]
BACKUP = [(9, 0), (9, 7), (10, 1), (10, 2), (10, 6), (10, 10), (11, 2), (11, 9)]
GROUPS = {"name_movers": NM, "backups": BACKUP, "both": NM + BACKUP}
VARIANTS = ["noise_abc", "denoise_abc", "noise_flip", "noise_mean", "noise_abc_end"]

def zcache(tokens):
    logits, cache = model.run_with_cache(tokens, names_filter=lambda nm: nm.endswith("hook_z") or nm.endswith("hook_resid_pre"))
    return logits, cache

def single_heads(run_cache, src, g, base_ld, sign, end_only=False):
    """Patch each head's z (all positions or final only) from src[layer] into the run cached in run_cache."""
    n, T = g["tc"].shape; out = torch.zeros(n, L, H)
    for l in range(L):
        resid = run_cache[f"blocks.{l}.hook_resid_pre"].repeat(H, 1, 1)        # [H*n, T, d]
        s = src[l]                                                          # [n, T, H, dh]
        def patch(z, hook, s=s):
            for h in range(H):
                rows = slice(h * n, (h + 1) * n)
                if end_only:
                    z[rows, -1, h] = s[:, -1, h]
                else:
                    z[rows, :, h] = s[:, :, h]
            return z
        with model.hooks(fwd_hooks=[(ZK[l], patch)]):
            pl = ld(model(resid, start_at_layer=l), g).reshape(H, n).T            # [n, H]
        out[:, l] = sign * (base_ld[:, None] - pl) if sign > 0 else (pl - base_ld[:, None])
    return out

def group_effect(tokens, src, heads, g, base_ld, sign, end_only=False):
    hooks = []
    for (l, h) in heads:
        def f(z, hook, l=l, h=h):
            if end_only:
                z[:, -1, h] = src[l][:, -1, h]
            else:
                z[:, :, h] = src[l][:, :, h]
            return z
        hooks.append((ZK[l], f))
    with model.hooks(fwd_hooks=hooks):
        pl = ld(model(tokens), g)
    return base_ld - pl if sign > 0 else pl - base_ld

per = {v: [] for v in VARIANTS}; grp = {v: {k: [] for k in GROUPS} for v in VARIANTS}
lds = {"clean": [], "abc": [], "flip": []}; meta = []
for gi, g in enumerate(groups):
    n = len(g["io"])
    lc, cc = zcache(g["tc"]); lx, cx = zcache(g["tx"]); lf, cf = zcache(g["tf"])
    _, cm = model.run_with_cache(g["tm"], names_filter=lambda nm: nm.endswith("hook_z"))
    ld_c, ld_x, ld_f = ld(lc, g), ld(lx, g), ld(lf, g)
    lds["clean"].append(ld_c); lds["abc"].append(ld_x); lds["flip"].append(ld_f)
    src_x = [cx[k] for k in ZK]; src_c = [cc[k] for k in ZK]; src_f = [cf[k] for k in ZK]
    src_m = [cm[k].mean(0, keepdim=True).expand(n, -1, -1, -1) for k in ZK]
    plan = {  # variant: (cache of the run being patched, tokens of that run, source of the patch, base ld, sign, end only)
        "noise_abc": (cc, g["tc"], src_x, ld_c, +1, False),
        "denoise_abc": (cx, g["tx"], src_c, ld_x, -1, False),
        "noise_flip": (cc, g["tc"], src_f, ld_c, +1, False),
        "noise_mean": (cc, g["tc"], src_m, ld_c, +1, False),
        "noise_abc_end": (cc, g["tc"], src_x, ld_c, +1, True),
    }
    for v, (rc, toks, src, base_ld, sign, end) in plan.items():
        per[v].append(single_heads(rc, src, g, base_ld, sign, end))
        for k, heads in GROUPS.items():
            grp[v][k].append(group_effect(toks, src, heads, g, base_ld, sign, end))
    for i in range(n):
        meta.append({"text": g["text"][i], "tpl": g["tpl"], "order": g["order"]})
    print(f"group {gi + 1}/{len(groups)} done at {time.time() - t0:.0f}s", flush=True)

P = {v: torch.cat(x) for v, x in per.items()}
G = {v: {k: torch.cat(x) for k, x in d.items()} for v, d in grp.items()}
LD = {k: torch.cat(x) for k, x in lds.items()}
r4 = lambda t: [round(float(x), 4) for x in t.flatten().tolist()]
prompts = [dict(meta[i], logit_diff={k: round(LD[k][i].item(), 4) for k in LD},
                scores={v: r4(P[v][i]) for v in VARIANTS},
                group={v: {k: round(G[v][k][i].item(), 4) for k in GROUPS} for v in VARIANTS})
           for i in range(len(meta))]
note = ("scores[variant] has 144 entries per prompt, index = layer*12 + head. Noising variants: clean logit diff "
        "minus patched. denoise_abc: patched minus ABC-corrupted logit diff. group[variant]: the same for the "
        "name movers, the backup name movers, and both groups patched together.")
path = f"{OUT}/patching_variants_per_prompt.json" if not ONLY_GROUPS else f"{OUT}/patching_variants_test.json"
json.dump({"note": note, "variants": VARIANTS, "groups": {k: [f"{l}.{h}" for l, h in v] for k, v in GROUPS.items()},
           "n_mean_prompts": N_MEAN, "prompts": prompts}, open(path, "w"), separators=(",", ":"))

# checks against the v3 file: same prompts, and noise_abc reproduces v3's patching scores
v3 = f"{OUT}/ioi_v3_results_per_prompt.json"
if os.path.exists(v3):
    ref = json.load(open(v3))["prompts"][:len(meta)]
    assert [p["text"] for p in ref] == [m["text"] for m in meta], "prompt order differs from the v3 file"
    tr = torch.tensor([p["truth"] for p in ref])
    err = (tr - P["noise_abc"].reshape(len(meta), -1)).abs().max().item()
    gerr = max(abs(p["group_drop"][k] - G["noise_abc"][k][i].item()) for i, p in enumerate(ref) for k in GROUPS)
    print(f"prompt texts match; noise_abc vs v3 truth: max abs diff {err:.5f} (heads), {gerr:.5f} (groups)")
    assert err < 2e-3 and gerr < 2e-3, "noise_abc does not reproduce v3"
print(f"wrote {path} in {time.time() - t0:.0f}s")
