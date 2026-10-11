# Run in Colab or locally: pip install "transformer_lens<4" scipy   (TransformerLens 4.x removed HookedTransformer)
# Full head-by-head patching takes ~12 min on 2 CPU cores; much faster on a GPU (not timed).
"""Feasibility smoke test for project 8b.

GPT-2 small, IOI prompts. For every attention head, compute three cheap
"explanations" of why the model says the indirect object, then compare each
against activation patching (the causal ground truth).

  1. attention  : attention weight from the final token to the IO name
  2. DLA        : direct logit attribution of the head to logit(IO) - logit(S)
  3. attr_patch : gradient-based attribution patching estimate
  truth: activation patching (replace one head's output with its value on a
         name-swapped prompt, measure how much the logit difference drops)
"""
import json, time, random
import torch
from scipy.stats import spearmanr
from transformer_lens import HookedTransformer

torch.set_grad_enabled(False)
random.seed(0)
t0 = time.time()
print("loading", flush=True)
model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
print("loaded", round(time.time()-t0,1), flush=True)
model.set_use_attn_result(True)
L, H = model.cfg.n_layers, model.cfg.n_heads

names = [" John", " Mary", " Tom", " James", " Dan", " Sarah", " Paul", " Anna",
         " Mark", " Kate", " David", " Emma", " Peter", " Laura", " Mike", " Lisa"]
names = [n for n in names if len(model.to_tokens(n, prepend_bos=False)[0]) == 1]
templates = [
    "When{A} and{B} went to the store,{S} gave a drink to",
    "When{A} and{B} went to the park,{S} gave a ball to",
    "After{A} and{B} went to the office,{S} handed a report to",
    "Then,{A} and{B} had a long argument, and afterwards{S} said to",
]
clean, corrupt, io_tok, s_tok, io_pos = [], [], [], [], []
for i in range(24):
    tpl = templates[i % len(templates)]
    a, b = random.sample(names, 2)
    # IO = A, subject repeated = B
    clean.append(tpl.format(A=a, B=b, S=b))
    corrupt.append(tpl.format(A=a, B=b, S=a))  # swap: correct answer flips to B
    io_tok.append(model.to_single_token(a))
    s_tok.append(model.to_single_token(b))
toks_c = model.to_tokens(clean)
toks_x = model.to_tokens(corrupt)
assert toks_c.shape == toks_x.shape, "prompt lengths differ"
io_tok = torch.tensor(io_tok); s_tok = torch.tensor(s_tok)
# IO token position = first occurrence of IO token in clean prompt
io_pos = torch.tensor([(row == t).nonzero()[0].item() for row, t in zip(toks_c, io_tok)])
N = len(clean); idx = torch.arange(N)

def logit_diff(logits):
    last = logits[:, -1]
    return (last[idx, io_tok] - last[idx, s_tok])

logits_c, cache_c = model.run_with_cache(toks_c)
logits_x, cache_x = model.run_with_cache(toks_x)
ld_clean = logit_diff(logits_c).mean().item()
ld_corr = logit_diff(logits_x).mean().item()
acc = (logit_diff(logits_c) > 0).float().mean().item()

# 1. attention from final position to IO token, per head
attn = torch.zeros(L, H)
for l in range(L):
    pat = cache_c[f"blocks.{l}.attn.hook_pattern"]  # [N, H, q, k]
    attn[l] = pat[idx, :, -1, io_pos].mean(0)

# 2. direct logit attribution per head
res_stack, labels = cache_c.stack_head_results(layer=-1, pos_slice=-1, return_labels=True)
res_stack = cache_c.apply_ln_to_stack(res_stack, layer=-1, pos_slice=-1)  # [LH, N, d]
dirs = model.W_U[:, io_tok] - model.W_U[:, s_tok]  # [d, N]
dla = (res_stack * dirs.T.unsqueeze(0)).sum(-1).mean(-1).reshape(L, H)

# 3. attribution patching (gradient x activation difference) on hook_z
torch.set_grad_enabled(True)
grads = {}
def save_grad(name):
    def hook(t, hook):
        t.register_hook(lambda g: grads.__setitem__(name, g.detach()))
        return t
    return hook
hooks = [(f"blocks.{l}.attn.hook_z", save_grad(f"blocks.{l}.attn.hook_z")) for l in range(L)]
with model.hooks(fwd_hooks=hooks):
    out = model(toks_c)
    logit_diff(out).mean().backward()
torch.set_grad_enabled(False)
attr = torch.zeros(L, H)
for l in range(L):
    n = f"blocks.{l}.attn.hook_z"
    delta = cache_x[n] - cache_c[n]  # [N, pos, H, d_head]
    attr[l] = (grads[n] * delta).sum(dim=(0, 1, 3))  # predicted change in mean logit diff (grad already has the 1/N from .mean())

# Truth: activation patching, one head at a time (corrupt -> clean)
print("explanations done", round(time.time()-t0,1), flush=True)
patch = torch.zeros(L, H)
for l in range(L):
    n = f"blocks.{l}.attn.hook_z"
    for h in range(H):
        def hook(z, hook, h=h):
            z[:, :, h] = cache_x[n][:, :, h]
            return z
        out = model.run_with_hooks(toks_c, fwd_hooks=[(n, hook)])
        patch[l, h] = logit_diff(out).mean() - ld_clean  # negative = head matters
    print("layer", l, round(time.time()-t0,1), flush=True)

truth = -patch  # importance: how much logit diff drops
def compare(score, name, absval=True):
    s = score.abs() if absval else score
    rho = spearmanr(s.flatten().numpy(), truth.abs().flatten().numpy()).correlation
    top_t = set(truth.abs().flatten().topk(10).indices.tolist())
    top_s = set(s.flatten().topk(10).indices.tolist())
    return {"method": name, "spearman_vs_patching": round(float(rho), 3),
            "top10_overlap": len(top_t & top_s),
            "top5_heads": [f"{i//H}.{i%H}" for i in s.flatten().topk(5).indices.tolist()]}

def heads(t, k=8):
    return [{"head": f"{i//H}.{i%H}", "value": round(float(t.flatten()[i]), 3)}
            for i in t.flatten().abs().topk(k).indices.tolist()]

res = {
    "n_prompts": N, "accuracy_IO_over_S": round(acc, 3),
    "mean_logit_diff_clean": round(ld_clean, 3), "mean_logit_diff_corrupt": round(ld_corr, 3),
    "truth_top_heads_by_patching": heads(truth),
    "comparisons": [compare(attn, "attention to IO"), compare(dla, "direct logit attribution"),
                    compare(attr, "attribution patching (gradient)")],
    "attention_top_heads": heads(attn),
    "runtime_sec": round(time.time() - t0, 1),
}
# Disagreement example: heads with high attention to IO but small causal effect
a, t = attn.flatten(), truth.abs().flatten()
hi_attn = a.topk(12).indices.tolist()
res["high_attention_low_effect"] = [
    {"head": f"{i//H}.{i%H}", "attn_to_IO": round(float(a[i]), 3), "patch_effect": round(float(t[i]), 3)}
    for i in hi_attn if t[i] < 0.1 * t.max()][:5]
json.dump({"attn": attn.tolist(), "dla": dla.tolist(), "attr": attr.tolist(), "truth": truth.tolist()},
          open("ioi_grids.json", "w"))
print(json.dumps(res, indent=1))
