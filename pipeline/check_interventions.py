"""Independent check of the interventions behind the assumption checks (pipeline/diagnostics.py).

Recomputes, with ordinary TransformerLens hooks and full forward passes instead of the batched code in
diagnostics.py, (1) the direct-path effect of several heads on the first IOI group (4 prompts) and the
first 6 greater-than prompts, (2) six pair effects on the first IOI group, (3) half-strength patching (behind the
linearity check) through attn.hook_result in full forward passes, and (4) value patching at the target token (behind the
"what it reads" check) written out as a change to z through the clean attention pattern, and compares them with
results/diagnostics_ioi.json, results/gt/diagnostics.json and the two diagnostics_values files.
Expected: max differences around 1e-4 (Nov 22: 5e-5 to 6e-5 for all four).
Direct path: run the clean prompt; at the final position add (corrupted - clean) output of the head to
blocks.11.hook_resid_post and pin ln_final's scale to its clean value. Effect = clean metric - patched metric.

Run:  python pipeline/check_interventions.py      (about two minutes on a laptop CPU)
"""
import json, random, sys, torch
from transformer_lens import HookedTransformer
torch.set_grad_enabled(False)
model = HookedTransformer.from_pretrained("gpt2-small", device="cpu"); model.set_use_attn_result(True)
lists = json.load(open("data/ioi_lists.json"))
src = open("pipeline/diagnostics.py").read()
a = src.index('    random.seed(0); torch.manual_seed(0)'); b = src.index('    if TEST: groups = groups[:TEST]')
block = "\n".join(l[4:] for l in src[a:b].split("\n"))
exec(block)            # rebuilds the IOI groups exactly as diagnostics.py does
D = json.load(open("results/diagnostics_ioi.json"))["prompts"]
def path_patch(tc, tx, metric, heads):
    _, cc = model.run_with_cache(tc); _, cx = model.run_with_cache(tx)
    mc = metric(model(tc)[:, -1])
    out = {}
    for (l, h) in heads:
        delta = cx[f"blocks.{l}.attn.hook_result"][:, -1, h] - cc[f"blocks.{l}.attn.hook_result"][:, -1, h]
        sc = cc["ln_final.hook_scale"]
        def add(x, hook): x[:, -1] += delta; return x
        def pin(x, hook): return sc
        lg = model.run_with_hooks(tc, fwd_hooks=[("blocks.11.hook_resid_post", add), ("ln_final.hook_scale", pin)])[:, -1]
        out[(l, h)] = (mc - metric(lg)).tolist()
    return out
heads = [(9, 9), (9, 6), (10, 0), (5, 5), (8, 6), (10, 7), (11, 10), (3, 0)]
g = groups[0]; io, s = g["io"], g["s"]
met = lambda last: last[torch.arange(last.shape[0]), io] - last[torch.arange(last.shape[0]), s]
res = path_patch(model.to_tokens(g["clean"]), model.to_tokens(g["corr"]), met, heads)
mx = 0
print("IOI, first 4 prompts: independent vs diagnostics.py direct (mean over the 4)")
for (l, h), v in res.items():
    ref = [D[k]["direct"][l * 12 + h] for k in range(4)]; mx = max(mx, max(abs(x - y) for x, y in zip(v, ref)))
    print(f"  {l}.{h}: {sum(v)/4:+.4f} vs {sum(ref)/4:+.4f}")
print("max abs diff per prompt:", round(mx, 5))
# greater-than
gt = json.load(open("results/gt/per_prompt.json"))["prompts"][:6]; G = json.load(open("results/gt/diagnostics.json"))["prompts"]
corr = []
for p in gt:
    st = f"{p['century']}{p['yy']:02d}"; i = p["text"].index(st); corr.append(p["text"][:i] + f"{p['century']}01" + p["text"][i + 4:])
YEAR = torch.tensor([model.to_single_token(f"{y:02d}") for y in range(100)])
yy = torch.tensor([p["yy"] for p in gt]); mask = (torch.arange(100)[None] > yy[:, None]).float()
def gmet(last):
    pr = last.softmax(-1)[:, YEAR]; return (pr * mask).sum(-1) - (pr * (1 - mask)).sum(-1)
assert all(G[k]["text"] == gt[k]["text"] for k in range(6))
res = path_patch(model.to_tokens([p["text"] for p in gt]), model.to_tokens(corr), gmet, [(9, 1), (8, 11), (7, 10), (6, 9), (5, 5), (10, 7)])
mx = 0
print("greater-than, first 6 prompts:")
for (l, h), v in res.items():
    ref = [G[k]["direct"][l * 12 + h] for k in range(6)]; mx = max(mx, max(abs(x - y) for x, y in zip(v, ref)))
    full = sum(G[k]["full"][l * 12 + h] for k in range(6)) / 6
    print(f"  {l}.{h}: direct {sum(v)/6:+.4f} vs {sum(ref)/6:+.4f}   (full effect {full:+.4f})")
print("max abs diff per prompt:", round(mx, 5))

# ---- pair effects (noising two heads together) ----
D = json.load(open("results/diagnostics_ioi.json")); P = D["prompts"]; pairs = D["pairs"]
g = groups[0]; tc, tx = model.to_tokens(g["clean"]), model.to_tokens(g["corr"]); io, s = g["io"], g["s"]
met = lambda lg: lg[torch.arange(4), -1, io] - lg[torch.arange(4), -1, s]
_, cx = model.run_with_cache(tx); mc = met(model(tc))
idx = lambda h: tuple(map(int, h.split(".")))
mx = 0
for k in [0, 7, 50, 120, 200, 275]:
    (l1, h1), (l2, h2) = idx(pairs[k][0]), idx(pairs[k][1])
    def hk(z, hook):
        l = int(hook.name.split(".")[1])
        for (L_, H_) in ((l1, h1), (l2, h2)):
            if L_ == l: z[:, :, H_] = cx[hook.name][:, :, H_]
        return z
    names = {f"blocks.{l1}.attn.hook_z", f"blocks.{l2}.attn.hook_z"}
    eff = (mc - met(model.run_with_hooks(tc, fwd_hooks=[(n, hk) for n in names]))).tolist()
    ref = [P[i]["pair"][k] for i in range(4)]; mx = max(mx, max(abs(x - y) for x, y in zip(eff, ref)))
    print(pairs[k], [round(x, 3) for x in eff], [round(x, 3) for x in ref])
print("pairs, max abs diff:", round(mx, 5))

# ---- half-strength patching (behind the linearity check), by a different route ----
# diagnostics.py interpolates the head's z at hook_z and starts each pass from that layer's cached residual stream.
# Here: full forward passes from the embeddings, interpolating the head's own output at attn.hook_result
# (result = z W_O per head, so the two are the same intervention if both codes are right).
def half_patch(tc, tx, metric, heads, alpha=0.5):
    _, cc = model.run_with_cache(tc); _, cx = model.run_with_cache(tx); mc = metric(model(tc)[:, -1]); out = {}
    for (l, h) in heads:
        nm = f"blocks.{l}.attn.hook_result"; src = cc[nm][:, :, h] + alpha * (cx[nm][:, :, h] - cc[nm][:, :, h])
        def hk(x, hook, src=src, h=h): x[:, :, h] = src; return x
        out[(l, h)] = (mc - metric(model.run_with_hooks(tc, fwd_hooks=[(nm, hk)])[:, -1])).tolist()
    return out
D = json.load(open("results/diagnostics_ioi.json"))["prompts"]
g = groups[0]; io, s = g["io"], g["s"]
met = lambda last: last[torch.arange(last.shape[0]), io] - last[torch.arange(last.shape[0]), s]
tc_i, tx_i = model.to_tokens(g["clean"]), model.to_tokens(g["corr"])
res = half_patch(tc_i, tx_i, met, [(5, 5), (8, 6), (9, 9), (3, 0), (4, 11), (10, 7)])
mx = max(abs(a - D[k]["half"][l * 12 + h]) for (l, h), v in res.items() for k, a in enumerate(v))
print("IOI half strength, first 4 prompts (mean): independent vs diagnostics.py")
for (l, h), v in res.items(): print(f"  {l}.{h}: {sum(v)/4:+.4f} vs {sum(D[k]['half'][l*12+h] for k in range(4))/4:+.4f}")
print("half strength IOI, max abs diff:", round(mx, 5))
tc_g, tx_g = model.to_tokens([p["text"] for p in gt]), model.to_tokens(corr)
res = half_patch(tc_g, tx_g, gmet, [(9, 1), (7, 10), (5, 5), (8, 11)])
mx = max(abs(a - G[k]["half"][l * 12 + h]) for (l, h), v in res.items() for k, a in enumerate(v))
print("half strength greater-than, max abs diff:", round(mx, 5))

# ---- value patching at the target token (behind the "what it reads" check), by a different route ----
# diagnostics_values.py swaps the head's value vector at one source position at hook_v. The same intervention, written
# out: z[q] = sum_k pattern[q, k] v[k], so changing v at position p changes z[q] by pattern[q, p] * (v_corr[p] - v_clean[p])
# for every query q, with the pattern left clean. Here that change is added at hook_z in a full forward pass.
def value_patch(tc, tx, metric, heads, pos):
    _, cc = model.run_with_cache(tc); _, cx = model.run_with_cache(tx); mc = metric(model(tc)[:, -1]); out = {}
    i = torch.arange(tc.shape[0])
    for (l, h) in heads:
        pat = cc[f"blocks.{l}.attn.hook_pattern"][i, h, :, pos]                    # [n, q]: attention from every query to p
        dv = cx[f"blocks.{l}.attn.hook_v"][i, pos, h] - cc[f"blocks.{l}.attn.hook_v"][i, pos, h]   # [n, d_head]
        delta = pat[:, :, None] * dv[:, None, :]                                    # [n, q, d_head]
        def hk(z, hook, delta=delta, h=h): z[:, :, h] = z[:, :, h] + delta; return z
        out[(l, h)] = (mc - metric(model.run_with_hooks(tc, fwd_hooks=[(f"blocks.{l}.attn.hook_z", hk)])[:, -1])).tolist()
    return out
V = json.load(open("results/diagnostics_values_ioi.json"))["prompts"]
pos_io = torch.tensor([[j for j, x in enumerate(t) if x == io[k].item()][0] for k, t in enumerate(tc_i.tolist())])
pos_s2 = torch.tensor([[j for j, x in enumerate(t) if x == s[k].item()][1] for k, t in enumerate(tc_i.tolist())])
mx = 0
for role, pos in (("IO", pos_io), ("S2", pos_s2)):
    res = value_patch(tc_i, tx_i, met, [(9, 9), (10, 7), (8, 6), (7, 9), (5, 5)], pos)
    for (l, h), v in res.items():
        ref = [V[k][role][l * 12 + h] for k in range(4)]; mx = max(mx, max(abs(a - b_) for a, b_ in zip(v, ref)))
        print(f"  values at {role}, {l}.{h}: {sum(v)/4:+.4f} vs {sum(ref)/4:+.4f}")
print("value patching IOI, max abs diff:", round(mx, 5))
VG = json.load(open("results/gt/diagnostics_values.json"))["prompts"]; assert all(VG[k]["text"] == gt[k]["text"] for k in range(6))
pos_yy = torch.tensor([[j for j, x in enumerate(t) if x == YEAR[gt[k]["yy"]].item()][0] for k, t in enumerate(tc_g.tolist())])
res = value_patch(tc_g, tx_g, gmet, [(9, 1), (7, 10), (5, 5), (6, 9)], pos_yy)
mx = max(abs(a - VG[k]["YY"][l * 12 + h]) for (l, h), v in res.items() for k, a in enumerate(v))
print("value patching greater-than (at YY), max abs diff:", round(mx, 5))
