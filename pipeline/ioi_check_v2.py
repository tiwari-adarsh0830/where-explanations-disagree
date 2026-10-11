"""Project 8b feasibility check, v2: the paper's own IOI templates.

GPT-2 small. Templates, names, places and objects are the ones from Wang et al.'s
released dataset code (BABA templates plus their ABBA mirror). Corruption is the
paper's ABC distribution: same template, three fresh names.

For every head we compare three explanations against activation patching:
  attention  : mean attention from the final token to the IO name
  dla        : direct logit attribution to logit(IO) - logit(S)
  attr       : attribution patching (gradient estimate of the patching drop)
  truth      : activation patching, one head at a time (drop in logit diff)
Plus a group check: knock out the three name movers together and see whether
the backup name movers compensate.

Run:  pip install "transformer_lens<4" scipy
      python ioi_check_v2.py ioi_lists.json
"""
import json, random, sys, time
import torch
from scipy.stats import spearmanr
from transformer_lens import HookedTransformer

lists = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "ioi_lists.json"))
OUT = sys.argv[2] if len(sys.argv) > 2 else "ioi_v2_results.json"
PER_TEMPLATE = int(sys.argv[3]) if len(sys.argv) > 3 else 4
random.seed(0); torch.manual_seed(0)
t0 = time.time()
model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
L, H = model.cfg.n_layers, model.cfg.n_heads
torch.set_grad_enabled(False)

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

groups = []  # one group per (template, order): equal token length inside a group
for tpl in lists["BABA_TEMPLATES"]:
    for order in ("BABA", "ABBA"):
        base = tpl
        if order == "ABBA":  # swap the order of the first two names
            i, j = base.index("[B]"), base.index("[A]")
            base = base[:i] + "[TMP]" + base[i + 3:j] + "[B]" + base[j + 3:]
            base = base.replace("[TMP]", "[A]", 1)
        clean, corr, io, s = [], [], [], []
        for _ in range(PER_TEMPLATE):
            A, B, X, Y, Z = random.sample(names, 5)
            place, obj = random.choice(places), random.choice(objects)
            if order == "BABA":
                c = fill(base, A, B, B, place, obj)
                x = fill(base, X, Y, Z, place, obj)      # ABC corruption
            else:
                # base is "[A] and [B] ... [B] ... to [A]" after the swap; fill handles first [B]
                c = fill(base, A, B, B, place, obj)
                x = fill(base, X, Y, Z, place, obj)
            clean.append(c); corr.append(x)
            io.append(model.to_single_token(" " + A)); s.append(model.to_single_token(" " + B))
        tc, tx = model.to_tokens(clean), model.to_tokens(corr)
        if tc.shape != tx.shape or len(set(len(model.to_tokens(p)[0]) for p in clean)) != 1:
            continue
        io_t, s_t = torch.tensor(io), torch.tensor(s)
        io_pos = torch.tensor([(row == t).nonzero()[0].item() for row, t in zip(tc, io_t)])
        groups.append(dict(tc=tc, tx=tx, io=io_t, s=s_t, io_pos=io_pos, text=clean))
N = sum(len(g["io"]) for g in groups)
print(f"{len(groups)} groups, {N} prompts, example: {groups[0]['text'][0]!r} / {groups[1]['text'][0]!r}", flush=True)

def ld(logits, g):
    last = logits[:, -1]; i = torch.arange(len(g["io"]))
    return last[i, g["io"]] - last[i, g["s"]]

attn = torch.zeros(L, H); dla = torch.zeros(L, H); attr = torch.zeros(L, H); truth = torch.zeros(L, H)
lds, accs = [], []
group_drop = {}
NM = [(9, 6), (9, 9), (10, 0)]
BACKUP = [(9, 0), (9, 7), (10, 1), (10, 2), (10, 6), (10, 10), (11, 2), (11, 9)]

for gi, g in enumerate(groups):
    n = len(g["io"]); w = n / N; idx = torch.arange(n)
    logits_c, cc = model.run_with_cache(g["tc"])
    _, cx = model.run_with_cache(g["tx"])
    base = ld(logits_c, g)
    lds.append(base.sum().item()); accs.append((base > 0).sum().item())
    for l in range(L):
        attn[l] += w * cc[f"blocks.{l}.attn.hook_pattern"][idx, :, -1, g["io_pos"]].mean(0)
    stack = cc.stack_head_results(layer=-1, pos_slice=-1)
    stack = cc.apply_ln_to_stack(stack, layer=-1, pos_slice=-1)
    dirs = model.W_U[:, g["io"]] - model.W_U[:, g["s"]]
    dla += w * (stack * dirs.T.unsqueeze(0)).sum(-1).mean(-1).reshape(L, H)
    # attribution patching
    torch.set_grad_enabled(True)
    grads = {}
    def keep(name):
        def h(t, hook):
            t.register_hook(lambda gr: grads.__setitem__(name, gr.detach())); return t
        return h
    with model.hooks(fwd_hooks=[(f"blocks.{l}.attn.hook_z", keep(f"blocks.{l}.attn.hook_z")) for l in range(L)]):
        ld(model(g["tc"]), g).mean().backward()
    torch.set_grad_enabled(False)
    for l in range(L):
        k = f"blocks.{l}.attn.hook_z"
        attr[l] += w * -(grads[k] * (cx[k] - cc[k])).sum(dim=(0, 1, 3))  # predicted drop
    # activation patching, starting each run at the patched layer
    for l in range(L):
        k = f"blocks.{l}.attn.hook_z"; resid = cc[f"blocks.{l}.hook_resid_pre"]
        for h in range(H):
            def patch(z, hook, h=h):
                z[:, :, h] = cx[k][:, :, h]; return z
            with model.hooks(fwd_hooks=[(k, patch)]):
                out = model(resid, start_at_layer=l)
            truth[l, h] += w * (base - ld(out, g)).mean()
    # group knockouts
    def knock(heads):
        hooks = []
        for (l, h) in heads:
            k = f"blocks.{l}.attn.hook_z"
            def f(z, hook, h=h, k=k):
                z[:, :, h] = cx[k][:, :, h]; return z
            hooks.append((k, f))
        with model.hooks(fwd_hooks=hooks):
            return (base - ld(model(g["tc"]), g)).mean().item()
    for name, heads in (("name_movers", NM), ("backups", BACKUP), ("both", NM + BACKUP)):
        group_drop[name] = group_drop.get(name, 0) + w * knock(heads)
    print(f"group {gi + 1}/{len(groups)} done at {time.time() - t0:.0f}s", flush=True)

def lab(i): return f"{i // H}.{i % H}"
def cmp(score, name):
    rho = spearmanr(score.abs().flatten(), truth.abs().flatten()).correlation
    tt = set(truth.abs().flatten().topk(10).indices.tolist()); ss = set(score.abs().flatten().topk(10).indices.tolist())
    return {"method": name, "spearman": round(float(rho), 3), "top10_overlap": len(tt & ss)}
def rank_of(score, heads):
    flat = score.abs().flatten(); order = flat.argsort(descending=True).tolist()
    r = {i: k + 1 for k, i in enumerate(order)}
    return {f"{l}.{h}": r[l * H + h] for l, h in heads}
ROLE = {"name_movers": NM, "negative_nm": [(10, 7), (11, 10)], "s_inhibition": [(7, 3), (7, 9), (8, 6), (8, 10)],
        "induction": [(5, 5), (6, 9)], "duplicate": [(0, 1), (3, 0)], "backup_nm": BACKUP}
res = {
    "prompts": N, "templates": len(groups), "accuracy": round(sum(accs) / N, 3), "mean_logit_diff": round(sum(lds) / N, 3),
    "comparisons": [cmp(attn, "attention"), cmp(dla, "dla"), cmp(attr, "attr_patching")],
    "truth_top10": [lab(i) for i in truth.abs().flatten().topk(10).indices.tolist()],
    "attn_top10": [lab(i) for i in attn.abs().flatten().topk(10).indices.tolist()],
    "ranks": {role: {m: rank_of(sc, hs) for m, sc in (("attn", attn), ("dla", dla), ("truth", truth))} for role, hs in ROLE.items()},
    "single_head_sum": {"name_movers": round(sum(truth[l, h].item() for l, h in NM), 3),
                        "backups": round(sum(truth[l, h].item() for l, h in BACKUP), 3)},
    "group_drop": {k: round(v, 3) for k, v in group_drop.items()},
    "runtime_sec": round(time.time() - t0),
}
json.dump({"summary": res, "grids": {k: [[round(float(v), 4) for v in row] for row in t.tolist()]
           for k, t in (("attn", attn), ("dla", dla), ("attr", attr), ("truth", truth))}}, open(OUT, "w"))
print(json.dumps(res, indent=1))
