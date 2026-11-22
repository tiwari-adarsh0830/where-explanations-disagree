"""Attention patterns for the T3 view, plus two extra per-head attention scores.

Forward passes only (no patching), so it takes well under a minute on CPU.
Prompts are built with exactly the same code and random seed as ioi_check_v3.py,
and the script checks that its prompt texts match results/ioi_v3_results_per_prompt.json.

Writes
  results/attn/L{layer}_H{head}.json   one file per head: for each of the 30 template x order
                                       groups, the head's attention pattern averaged over the
                                       group's 4 prompts (all 4 have the same length), plus a
                                       value-weighted pattern. Lower triangle only, as integer
                                       percentages (each row sums to about 100).
  results/attn_scores.json             per prompt and head: attention and value-weighted attention
                                       at 6 query>key positions (v3's five plus s1+1>s1 for
                                       previous-token heads).

Value-weighted attention follows Kobayashi et al. (2020): weight each attention entry by the
size of what that key position actually writes, ||v_k W_O||, then renormalize each row.

Run:  python pipeline/attention_patterns.py data/ioi_lists.json results
"""
import json, random, sys, time
import torch
import os
from transformer_lens import HookedTransformer

lists = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "ioi_lists.json"))
OUT = sys.argv[2] if len(sys.argv) > 2 else "results"
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

PAIRS = ["end>io", "end>s1", "end>s2", "s2>s1", "s2>s1+1", "s1+1>s1"]
os.makedirs(f"{OUT}/attn", exist_ok=True)
pattern_files = {(l, h): [] for l in range(L) for h in range(H)}
per_prompt = []
for g in groups:
    n, T = g["tc"].shape
    _, cache = model.run_with_cache(g["tc"], names_filter=lambda nm: nm.endswith("hook_pattern") or nm.endswith("hook_v"))
    endp = torch.full((n,), T - 1)
    qk = [(endp, g["io_pos"]), (endp, g["s1_pos"]), (endp, g["s2_pos"]),
          (g["s2_pos"], g["s1_pos"]), (g["s2_pos"], g["s1_pos"] + 1), (g["s1_pos"] + 1, g["s1_pos"])]
    idx = torch.arange(n)
    a_pairs = torch.zeros(n, L, H, len(PAIRS)); vw_pairs = torch.zeros(n, L, H, len(PAIRS))
    for l in range(L):
        pat = cache[f"blocks.{l}.attn.hook_pattern"]                    # [n, H, q, k]
        v = cache[f"blocks.{l}.attn.hook_v"]                                 # [n, k, H, d_head]
        out_norm = torch.einsum("nkhd,hdm->nkhm", v, model.W_O[l]).norm(dim=-1)   # [n, k, H]: size of what each key writes
        vw = pat * out_norm.permute(0, 2, 1)[:, :, None, :]             # [n, H, q, k]
        vw = vw / vw.sum(-1, keepdim=True).clamp_min(1e-9)              # renormalize each query row
        for j, (q, k) in enumerate(qk):
            a_pairs[:, l, :, j] = pat[idx, :, q, k]; vw_pairs[:, l, :, j] = vw[idx, :, q, k]
        mean_pat, mean_vw = pat.mean(0), vw.mean(0)                     # [H, q, k], averaged over the group's prompts
        for h in range(H):
            tri = lambda M: [[int(round(100 * float(M[h, q, k]))) for k in range(q + 1)] for q in range(T)]
            pattern_files[(l, h)].append({"tpl": g["tpl"], "order": g["order"], "attn": tri(mean_pat), "vw": tri(mean_vw)})
    for i in range(n):
        per_prompt.append({"text": g["text"][i], "tpl": g["tpl"], "order": g["order"],
                           "attn": [[round(float(x), 4) for x in row] for row in a_pairs[i].reshape(L * H, -1).tolist()],
                           "vw": [[round(float(x), 4) for x in row] for row in vw_pairs[i].reshape(L * H, -1).tolist()]})

groups_meta = [{"tpl": g["tpl"], "order": g["order"], "toks": g["toks"], "io_pos": g["io_pos"][0].item(),
                "s1_pos": g["s1_pos"][0].item(), "s2_pos": g["s2_pos"][0].item(), "example": g["text"][0]} for g in groups]
for (l, h), items in pattern_files.items():
    json.dump({"head": f"{l}.{h}", "groups": items}, open(f"{OUT}/attn/L{l}_H{h}.json", "w"), separators=(",", ":"))
json.dump({"note": "per-head arrays have 144 rows, index = layer*12 + head; each row has one value per pair",
           "pairs": PAIRS, "groups": groups_meta, "prompts": per_prompt}, open(f"{OUT}/attn_scores.json", "w"), separators=(",", ":"))

# prompts must line up one to one with the v3 per-prompt file the frontend already uses
v3 = f"{OUT}/ioi_v3_results_per_prompt.json"
if os.path.exists(v3):
    ref = [p["text"] for p in json.load(open(v3))["prompts"]]
    assert ref == [p["text"] for p in per_prompt], "prompt order differs from the v3 file"
    print("prompt texts match", v3)
print(f"wrote {OUT}/attn/ ({len(pattern_files)} files) and {OUT}/attn_scores.json in {time.time() - t0:.0f}s")
