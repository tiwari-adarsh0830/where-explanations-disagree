"""Project 8b IOI pipeline, v3: v2 plus per-prompt scores, template labels and attention at 5 query>key pairs.

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
for ti, tpl in enumerate(lists["BABA_TEMPLATES"]):
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
        s_hits = [(row == t).nonzero().flatten().tolist() for row, t in zip(tc, s_t)]
        assert all(len(p) == 2 for p in s_hits), "S should appear exactly twice"
        s1_pos = torch.tensor([p[0] for p in s_hits]); s2_pos = torch.tensor([p[1] for p in s_hits])
        groups.append(dict(tc=tc, tx=tx, io=io_t, s=s_t, io_pos=io_pos, text=clean,
                           tpl=ti, order=order, s1_pos=s1_pos, s2_pos=s2_pos,
                           toks=model.to_str_tokens(tc[0])))
N = sum(len(g["io"]) for g in groups)
print(f"{len(groups)} groups, {N} prompts, example: {groups[0]['text'][0]!r} / {groups[1]['text'][0]!r}", flush=True)
def ld(logits, g):
    last = logits[:, -1]; i = torch.arange(len(g["io"]))
    return last[i, g["io"]] - last[i, g["s"]]

NM = [(9, 6), (9, 9), (10, 0)]
BACKUP = [(9, 0), (9, 7), (10, 1), (10, 2), (10, 6), (10, 10), (11, 2), (11, 9)]
# attention is read at five query>key pairs per head (end = final token, where the answer is predicted)
PAIRS = ["end>io", "end>s1", "end>s2", "s2>s1", "s2>s1+1"]
pp = {k: [] for k in ("attn", "dla", "attr", "truth")}   # per prompt: [n, L, H] (attn: [n, L, H, 5])
pp_ld, pp_meta, examples = [], [], []
pp_group = {k: [] for k in ("name_movers", "backups", "both")}

for gi, g in enumerate(groups):
    n = len(g["io"]); idx = torch.arange(n)
    logits_c, cc = model.run_with_cache(g["tc"])
    _, cx = model.run_with_cache(g["tx"])
    base = ld(logits_c, g)
    pp_ld.append(base)
    # attention at all five query>key pairs
    endp = torch.full((n,), g["tc"].shape[1] - 1)
    qk = [(endp, g["io_pos"]), (endp, g["s1_pos"]), (endp, g["s2_pos"]),
          (g["s2_pos"], g["s1_pos"]), (g["s2_pos"], g["s1_pos"] + 1)]
    a = torch.zeros(n, L, H, len(PAIRS))
    for l in range(L):
        pat = cc[f"blocks.{l}.attn.hook_pattern"]          # [n, H, query, key]
        for j, (q, k) in enumerate(qk):
            a[:, l, :, j] = pat[idx, :, q, k]
    pp["attn"].append(a)
    # direct logit attribution, per prompt
    stack = cc.stack_head_results(layer=-1, pos_slice=-1)
    stack = cc.apply_ln_to_stack(stack, layer=-1, pos_slice=-1)
    dirs = model.W_U[:, g["io"]] - model.W_U[:, g["s"]]
    pp["dla"].append((stack * dirs.T.unsqueeze(0)).sum(-1).T.reshape(n, L, H))
    # attribution patching: backward on the SUM so each prompt keeps its own unscaled gradient
    torch.set_grad_enabled(True)
    grads = {}
    def keep(name):
        def h(t, hook):
            t.register_hook(lambda gr: grads.__setitem__(name, gr.detach())); return t
        return h
    with model.hooks(fwd_hooks=[(f"blocks.{l}.attn.hook_z", keep(f"blocks.{l}.attn.hook_z")) for l in range(L)]):
        ld(model(g["tc"]), g).sum().backward()
    torch.set_grad_enabled(False)
    at = torch.zeros(n, L, H)
    for l in range(L):
        k = f"blocks.{l}.attn.hook_z"
        at[:, l] = -(grads[k] * (cx[k] - cc[k])).sum(dim=(1, 3))  # predicted drop, per prompt
    pp["attr"].append(at)
    # activation patching, starting each run at the patched layer
    tr = torch.zeros(n, L, H)
    for l in range(L):
        k = f"blocks.{l}.attn.hook_z"; resid = cc[f"blocks.{l}.hook_resid_pre"]
        for h in range(H):
            def patch(z, hook, h=h):
                z[:, :, h] = cx[k][:, :, h]; return z
            with model.hooks(fwd_hooks=[(k, patch)]):
                out = model(resid, start_at_layer=l)
            tr[:, l, h] = base - ld(out, g)
    pp["truth"].append(tr)
    # group knockouts, per prompt
    def knock(heads):
        hooks = []
        for (l, h) in heads:
            k = f"blocks.{l}.attn.hook_z"
            def f(z, hook, h=h, k=k):
                z[:, :, h] = cx[k][:, :, h]; return z
            hooks.append((k, f))
        with model.hooks(fwd_hooks=hooks):
            return base - ld(model(g["tc"]), g)
    for name, heads in (("name_movers", NM), ("backups", BACKUP), ("both", NM + BACKUP)):
        pp_group[name].append(knock(heads))
    for i in range(n):
        pp_meta.append({"tpl": g["tpl"], "order": g["order"], "text": g["text"][i]})
    examples.append({"tpl": g["tpl"], "order": g["order"], "toks": g["toks"],
                     "io_pos": g["io_pos"][0].item(), "s1_pos": g["s1_pos"][0].item(), "s2_pos": g["s2_pos"][0].item()})
    print(f"group {gi + 1}/{len(groups)} done at {time.time() - t0:.0f}s", flush=True)

# stack per-prompt values; plain means over prompts must reproduce v2 exactly
P = {k: torch.cat(v) for k, v in pp.items()}
base_all = torch.cat(pp_ld)
G = {k: torch.cat(v) for k, v in pp_group.items()}
attn_pairs = P["attn"].mean(0)                       # [L, H, 5]
attn, dla, attr, truth = attn_pairs[..., 0], P["dla"].mean(0), P["attr"].mean(0), P["truth"].mean(0)
lds, accs = [base_all.sum().item()], [(base_all > 0).sum().item()]
group_drop = {k: v.mean().item() for k, v in G.items()}

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
def r4(t): return [[round(float(v), 4) for v in row] for row in t.tolist()]
json.dump({"summary": res,
           "grids": {k: r4(t) for k, t in (("attn", attn), ("dla", dla), ("attr", attr), ("truth", truth))},
           "attn_pairs": {name: r4(attn_pairs[..., j]) for j, name in enumerate(PAIRS)}}, open(OUT, "w"))
PP_OUT = OUT[:-5] + "_per_prompt.json" if OUT.endswith(".json") else OUT + "_per_prompt.json"
flat = lambda t: [round(float(v), 4) for v in t.flatten().tolist()]
json.dump({"note": "per-head arrays have 144 entries, index = layer*12 + head; attn has 144 x 5 (pairs order below)",
           "pairs": PAIRS, "roles": {r: [f"{l}.{h}" for l, h in hs] for r, hs in ROLE.items()},
           "examples": examples,
           "prompts": [dict(pp_meta[i], logit_diff=round(base_all[i].item(), 4),
                            group_drop={k: round(G[k][i].item(), 4) for k in G},
                            attn=[[round(float(v), 4) for v in row] for row in P["attn"][i].reshape(L * H, len(PAIRS)).tolist()],
                            dla=flat(P["dla"][i]), attr=flat(P["attr"][i]), truth=flat(P["truth"][i]))
                       for i in range(len(pp_meta))]}, open(PP_OUT, "w"))
print("wrote", OUT, "and", PP_OUT)
print(json.dumps(res, indent=1))
