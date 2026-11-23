"""Does it hold on another model? Every score and every assumption check in one script, for any GPT-2 size or Pythia model.

The main results come from five scripts written for GPT-2 small (ioi_check_v3.py, gt_check.py, attention_patterns.py,
patching_variants.py, diagnostics.py, diagnostics_values.py). This script recomputes, for any HookedTransformer model, the
quantities the contrast test needs, with the same prompts, corruptions and definitions:

  attn_end   attention from the final token to the target (IOI: IO name, greater-than: YY)
  dla        direct logit attribution (IOI: onto W_U[IO] - W_U[S]; greater-than: onto mean W_U[years > YY] - mean W_U[years <= YY]),
             final LayerNorm applied with the clean run's scale
  attr       attribution patching: gradient of the metric at hook_z times (corrupted z - clean z)
  full       single-head noising, all positions (clean run, the head's corrupted output patched in): clean metric - patched metric
  half       the same at half strength (z = clean + 0.5 (corrupted - clean))      -> linearity check
  end        noising at the final position only                                 -> position check
  denoise    corrupted run, the head's clean output patched in: patched metric - corrupted metric
  direct     the head's corrupted output only in the final residual stream, final LayerNorm scale frozen   -> direct check
  val_target value patching at the target position only (pattern and everything else clean)               -> target check
  pair       both heads noised together, for all 276 pairs among the 24 heads with the largest mean |full|  -> redundancy check

Prompts: IOI uses the same code and seed as diagnostics.py (120 prompts, 15 templates x BABA/ABBA, ABC corruption);
greater-than uses the 194 prompts in results/gt/per_prompt.json with the "01" corruption. All GPT-2 sizes share one
tokenizer, so the prompts are the same tokens for every size.
Other tokenizers (Pythia): IOI keeps only names that are one token in this model, as diagnostics.py already does, so the
same seed draws different names (88 of the 99 names for Pythia); greater-than keeps only prompts whose start year splits
into the century token and a two-digit token, clean and corrupted (146 of 194 for Pythia: " 1604" becomes " 160" + "4"
there). A start-of-text token is prepended for every model, as TransformerLens does by default for GPT-2 (it does not for
Pythia). Metrics: IOI logit(IO) - logit(S); greater-than probability
on years above YY minus at or below.

Check: with gpt2-small the output matches the committed results (see check_model_check.py).

Writes OUTDIR/ioi.json and OUTDIR/gt.json (per-prompt arrays, index = layer * n_heads + head).
Run:  python pipeline/model_check.py gpt2-medium results/gpt2-medium [ioi|gt|both] [n_test_batches]
      python pipeline/model_check.py pythia-410m results/pythia-410m
      Measured on the 2-core cloud machine: one IOI batch (4 prompts) of gpt2-medium takes about 5 minutes, so IOI alone is
      about 2.5 hours there. A MacBook ran the small-model pipeline about 4x faster, so expect very roughly 40 minutes for
      IOI and a bit more for greater-than; the progress lines show the real pace after the first batch.
"""
import json, os, random, sys, time
import torch
from transformer_lens import HookedTransformer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt2-medium"
OUT = sys.argv[2] if len(sys.argv) > 2 else f"results/{MODEL}"
WHICH = sys.argv[3] if len(sys.argv) > 3 else "both"
TEST = int(sys.argv[4]) if len(sys.argv) > 4 else None
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
K = 24
t0 = time.time()
torch.set_grad_enabled(False)
model = HookedTransformer.from_pretrained(MODEL, device="cpu")
model.eval()
model.cfg.default_prepend_bos = True          # GPT-2's default already; Pythia's default is False, so set it for the same setup
L, H = model.cfg.n_layers, model.cfg.n_heads
ZK = [f"blocks.{l}.attn.hook_z" for l in range(L)]
VK = [f"blocks.{l}.attn.hook_v" for l in range(L)]
PK = [f"blocks.{l}.attn.hook_pattern" for l in range(L)]
KEEP = lambda nm: (nm.endswith("hook_z") or nm.endswith("hook_v") or nm.endswith("hook_resid_pre")
                   or nm.endswith("hook_pattern") or nm == "ln_final.hook_scale")
os.makedirs(OUT, exist_ok=True)
print(f"{MODEL}: {L} layers x {H} heads, d_model {model.cfg.d_model}", flush=True)

def last_logits(resid):
    return model.unembed(model.ln_final(resid[:, -1:, :]))[:, 0]
def run_last(x, start=None):
    return last_logits(model(x, start_at_layer=start, stop_at_layer=L) if start is not None else model(x, stop_at_layer=L))

def prepare(tc, tx, metric, tgt):
    _, cc = model.run_with_cache(tc, names_filter=KEEP, return_type=None)
    _, cx = model.run_with_cache(tx, names_filter=KEEP, return_type=None)
    b = {"tc": tc, "tx": tx, "cc": cc, "cx": cx, "metric": metric, "tgt": tgt, "lc": run_last(tc)}
    b["mc"] = metric(b["lc"], 1); b["mx"] = metric(run_last(tx), 1)
    return b

def sweep(b, alpha=1.0, end_only=False, denoise=False):
    """One head at a time, H heads of a layer per forward pass. Noising: clean run, corrupted z in (returns clean - patched).
    Denoising: corrupted run, clean z in (returns patched - corrupted)."""
    base, other = ("cx", "cc") if denoise else ("cc", "cx")
    n = b["tc"].shape[0]; out = torch.zeros(n, L, H)
    for l in range(L):
        resid = b[base][f"blocks.{l}.hook_resid_pre"].repeat(H, 1, 1)
        z0, z1 = b[base][ZK[l]], b[other][ZK[l]]
        src = z0 + alpha * (z1 - z0)
        def f(z, hook, src=src):
            for h in range(H):
                if end_only: z[h * n:(h + 1) * n, -1, h] = src[:, -1, h]
                else: z[h * n:(h + 1) * n, :, h] = src[:, :, h]
            return z
        with model.hooks(fwd_hooks=[(ZK[l], f)]):
            m = b["metric"](run_last(resid, l), H).reshape(H, n).T
        out[:, l] = (m - b["mx"][:, None]) if denoise else (b["mc"][:, None] - m)
    return out

def sweep_values(b):
    """Each head's values at the target position replaced by the corrupted run's (pattern and the rest clean)."""
    n = b["tc"].shape[0]; out = torch.zeros(n, L, H); idx = torch.arange(n); pos = b["tgt"]
    for l in range(L):
        resid = b["cc"][f"blocks.{l}.hook_resid_pre"].repeat(H, 1, 1)
        vx = b["cx"][VK[l]]
        def f(v, hook, vx=vx):
            for h in range(H): v[h * n + idx, pos, h] = vx[idx, pos, h]
            return v
        with model.hooks(fwd_hooks=[(VK[l], f)]):
            m = b["metric"](run_last(resid, l), H).reshape(H, n).T
        out[:, l] = b["mc"][:, None] - m
    return out

def direct(b):
    n = b["tc"].shape[0]; out = torch.zeros(n, L, H)
    scale = b["cc"]["ln_final.hook_scale"][:, -1]
    for l in range(L):
        d = torch.einsum("nhd,hdm->nhm", b["cx"][ZK[l]][:, -1] - b["cc"][ZK[l]][:, -1], model.W_O[l])
        d = (d - d.mean(-1, keepdim=True)) / scale[:, None, :]
        for h in range(H):
            out[:, l, h] = b["mc"] - b["metric"](b["lc"] + d[:, h] @ model.W_U, 1)
    return out

def dla(b, dirs):
    """dirs: [n, d_model] logit-space direction per prompt."""
    n = b["tc"].shape[0]; out = torch.zeros(n, L, H)
    scale = b["cc"]["ln_final.hook_scale"][:, -1]
    for l in range(L):
        r = torch.einsum("nhd,hdm->nhm", b["cc"][ZK[l]][:, -1], model.W_O[l])
        r = (r - r.mean(-1, keepdim=True)) / scale[:, None, :]
        out[:, l] = (r * dirs[:, None, :]).sum(-1)
    return out

def attr(b):
    grads = {}
    def keep(name):
        def h(t, hook):
            t.register_hook(lambda gr: grads.__setitem__(name, gr.detach())); return t
        return h
    with torch.enable_grad():
        with model.hooks(fwd_hooks=[(k, keep(k)) for k in ZK]):
            b["metric"](run_last(b["tc"]), 1).sum().backward()
    model.zero_grad(set_to_none=True)
    n = b["tc"].shape[0]; out = torch.zeros(n, L, H)
    for l in range(L):
        out[:, l] = -(grads[ZK[l]] * (b["cx"][ZK[l]] - b["cc"][ZK[l]])).sum(dim=(1, 3))
    return out

def attn_end(b):
    n = b["tc"].shape[0]; idx = torch.arange(n); out = torch.zeros(n, L, H)
    for l in range(L):
        out[:, l] = b["cc"][PK[l]][idx, :, -1, b["tgt"]]
    return out

def pairs_effect(b, pairs, chunk=46):
    n = b["tc"].shape[0]; out = torch.zeros(n, len(pairs))
    for c0 in range(0, len(pairs), chunk):
        ps = pairs[c0:c0 + chunk]; P = len(ps); hooks = []
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

SINGLE = ("attn_end", "dla", "attr", "full", "half", "end", "denoise", "direct", "val_target")
def run(batches, label):
    res = {k: [] for k in SINGLE + ("mc", "mx")}
    for i, mk in enumerate(batches):
        b, dirs = mk()
        res["attn_end"].append(attn_end(b)); res["dla"].append(dla(b, dirs)); res["attr"].append(attr(b))
        res["full"].append(sweep(b)); res["half"].append(sweep(b, 0.5)); res["end"].append(sweep(b, end_only=True))
        res["denoise"].append(sweep(b, denoise=True)); res["direct"].append(direct(b)); res["val_target"].append(sweep_values(b))
        res["mc"].append(b["mc"]); res["mx"].append(b["mx"])
        el = time.time() - t0
        print(f"  {label} batch {i + 1}/{len(batches)} at {el:.0f}s", flush=True)
        del b
    R = {k: torch.cat(v) for k, v in res.items()}
    ref = R["full"].mean(0)
    idx = ref.abs().flatten().topk(K).indices.tolist()
    heads = [(i // H, i % H) for i in sorted(idx)]
    pairs = [(heads[a], heads[c]) for a in range(K) for c in range(a + 1, K)]
    pr = []
    for i, mk in enumerate(batches):
        b, _ = mk(); pr.append(pairs_effect(b, pairs)); del b
        print(f"  {label} pairs {i + 1}/{len(batches)} at {time.time() - t0:.0f}s", flush=True)
    R["pair"] = torch.cat(pr)
    return R, heads, pairs

r4 = lambda t: [round(float(x), 4) for x in t.flatten().tolist()]
def write(path, meta, R, heads, pairs, extra):
    json.dump({"model": MODEL, "n_layers": L, "n_heads": H,
               "note": f"per-head arrays: {L * H} entries, index = layer*{H} + head",
               "top_heads": [f"{l}.{h}" for l, h in heads], "pairs": [[f"{a[0]}.{a[1]}", f"{c[0]}.{c[1]}"] for a, c in pairs], **extra,
               "prompts": [dict(meta[i], metric_clean=round(R["mc"][i].item(), 4), metric_corrupt=round(R["mx"][i].item(), 4),
                                **{k: r4(R[k][i]) for k in SINGLE + ("pair",)}) for i in range(len(meta))]},
              open(path, "w"), separators=(",", ":"))
    print(f"wrote {path}", flush=True)

def summary_line(task, R):
    print(f"{task}: clean metric {R['mc'].mean():.3f}, corrupted {R['mx'].mean():.3f}; "
          f"largest mean noising effects: " +
          ", ".join(f"{i // H}.{i % H} {R['full'].mean(0).flatten()[i]:+.2f}" for i in R["full"].mean(0).abs().flatten().topk(6).indices.tolist()), flush=True)

# ------------------------------- IOI -------------------------------
if WHICH in ("ioi", "both"):
    lists = json.load(open(os.path.join(DATA, "data/ioi_lists.json")))
    src = open(os.path.join(DATA, "pipeline/diagnostics.py")).read()
    a = src.index('    random.seed(0); torch.manual_seed(0)'); z = src.index('    if TEST: groups = groups[:TEST]')
    exec("\n".join(l[4:] for l in src[a:z].split("\n")))                 # same prompts as diagnostics.py
    if TEST: groups = groups[:TEST]
    def ioi_batch(g):
        def mk():
            io, s = g["io"], g["s"]
            def metric(last, rep):
                i = torch.arange(last.shape[0]); return last[i, io.repeat(rep)] - last[i, s.repeat(rep)]
            tc, tx = model.to_tokens(g["clean"]), model.to_tokens(g["corr"]); assert tc.shape == tx.shape
            tgt = []
            for k in range(tc.shape[0]):
                t = tc[k].tolist(); ip = [j for j, x in enumerate(t) if x == io[k].item()]
                assert len(ip) == 1, g["clean"][k]; tgt.append(ip[0])
            dirs = (model.W_U[:, io] - model.W_U[:, s]).T
            return prepare(tc, tx, metric, torch.tensor(tgt)), dirs
        return mk
    meta = [{"text": c, "tpl": g["tpl"], "order": g["order"]} for g in groups for c in g["clean"]]
    print(f"IOI: {len(meta)} prompts, {len(names)} of {len(lists['NAMES'])} names are one token", flush=True)
    R, heads, pairs = run([ioi_batch(g) for g in groups], "IOI")
    summary_line("IOI", R)
    write(f"{OUT}/ioi{'_test' if TEST else ''}.json", meta, R, heads, pairs,
          {"task": "ioi", "target": "IO", "corruption": "ABC (three fresh names)", "metric": "logit(IO) - logit(S)"})

# --------------------------- greater-than ---------------------------
if WHICH in ("gt", "both"):
    gt = json.load(open(os.path.join(DATA, "results/gt/per_prompt.json")))["prompts"]
    CH = 24 if H <= 12 else 12
    YEAR = torch.tensor([model.to_single_token(f"{y:02d}") for y in range(100)])
    def splits(cen, yy):                       # " CCYY" -> " CC" + "YY" in this model's tokenizer
        return model.to_tokens(f" {cen}{yy:02d}", prepend_bos=False)[0].tolist() == [model.to_single_token(f" {cen}"), YEAR[yy].item()]
    n_all = len(gt); gt = [p for p in gt if splits(p["century"], p["yy"]) and splits(p["century"], 1)]
    dropped_gt = n_all - len(gt)
    if dropped_gt: print(f"greater-than: {dropped_gt} of {n_all} prompts dropped (start year not split into century + two digits in this tokenizer)", flush=True)
    if TEST: gt = gt[:CH * TEST]
    corr = []
    for p in gt:
        st = f"{p['century']}{p['yy']:02d}"; i = p["text"].index(st); corr.append(p["text"][:i] + f"{p['century']}01" + p["text"][i + 4:])
    WU = model.W_U[:, YEAR]
    def gt_batch(rows):
        def mk():
            yy = torch.tensor([gt[r]["yy"] for r in rows]); mask = (torch.arange(100)[None, :] > yy[:, None]).float()
            def metric(last, rep):
                p = last.softmax(-1)[:, YEAR]; m = mask.repeat(rep, 1)
                return (p * m).sum(-1) - (p * (1 - m)).sum(-1)
            tc = model.to_tokens([gt[r]["text"] for r in rows]); tx = model.to_tokens([corr[r] for r in rows]); assert tc.shape == tx.shape
            tgt = []
            for k, r in enumerate(rows):
                t = tc[k].tolist(); cen = model.to_single_token(f" {gt[r]['century']}")
                j = [q for q in range(1, len(t)) if t[q] == YEAR[gt[r]["yy"]].item() and t[q - 1] == cen]
                assert len(j) == 1, gt[r]["text"]; tgt.append(j[0])
            dirs = (mask / mask.sum(1, keepdim=True)) @ WU.T - ((1 - mask) / (1 - mask).sum(1, keepdim=True)) @ WU.T
            return prepare(tc, tx, metric, torch.tensor(tgt)), dirs
        return mk
    chunks = [list(range(i, min(i + CH, len(gt)))) for i in range(0, len(gt), CH)]
    print(f"greater-than: {len(gt)} prompts", flush=True)
    R, heads, pairs = run([gt_batch(c) for c in chunks], "greater-than")
    summary_line("greater-than", R)
    write(f"{OUT}/gt{'_test' if TEST else ''}.json", [{"text": p["text"], "tpl": p["tpl"], "yy": p["yy"]} for p in gt], R, heads, pairs,
          {"task": "gt", "target": "YY", "corruption": "start year changed to XX01", "metric": "P(year > YY) - P(year <= YY)",
           "prompts_dropped": dropped_gt})
print(f"done in {time.time() - t0:.0f}s", flush=True)
