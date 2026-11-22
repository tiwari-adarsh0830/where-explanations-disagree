"""What does each head read? Value patching at one source position.

The position check (final-position patching) tests where a head acts. Final-token attention also assumes
the head reads the target (IO in IOI, YY in greater-than). This script tests that with an intervention that
does not use the attention score: for every head, replace only the head's value vectors at one source
position with their values from the corrupted prompt (attention pattern and everything else clean), and
measure the change in the metric (clean minus patched, noising, main corruption). Done for each role position
(IOI: IO, S1, S1+1, S2, END; greater-than: YY and END, where END is the final " XX" token). The target check uses the target position:
  violation = |full - value_at_target| / (|full| + |value_at_target|)
Same prompts and corruptions as pipeline/diagnostics.py (its IOI prompt code is reused verbatim).

Writes results/diagnostics_values_ioi.json and results/gt/diagnostics_values.json.
Run:  python pipeline/diagnostics_values.py data/ioi_lists.json results [ioi|gt|both] [n_batches_for_test]
"""
import json, os, sys, time, random
import torch
from transformer_lens import HookedTransformer

src = open(__file__.replace("diagnostics_values.py", "diagnostics.py")).read()
exec(src[src.index("lists = json.load"):src.index("def sweep(")])          # model, L, H, run_last, argv handling
VK = [f"blocks.{l}.attn.hook_v" for l in range(L)]
KEEPV = lambda nm: nm.endswith("hook_v") or nm.endswith("hook_resid_pre")
t0 = time.time()

def prepare(tc, tx, metric):
    _, cc = model.run_with_cache(tc, names_filter=KEEPV, return_type=None)
    _, cx = model.run_with_cache(tx, names_filter=KEEPV, return_type=None)
    b = {"tc": tc, "cc": cc, "cx": cx, "metric": metric}
    b["mc"] = metric(run_last(tc), 1)
    return b

def sweep_values(b, pos):
    """Patch head h's values at source position pos[i] (per prompt) with corrupted values; 12 heads per pass."""
    n = b["tc"].shape[0]; out = torch.zeros(n, L, H); idx = torch.arange(n)
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

def run(batches, roles):
    res = {r: [] for r in roles}; res["mc"] = []
    for i, mk in enumerate(batches):
        b, P = mk()
        for r in roles: res[r].append(sweep_values(b, P[r]))
        res["mc"].append(b["mc"])
        print(f"  batch {i + 1}/{len(batches)} at {time.time() - t0:.0f}s", flush=True)
        del b
    return {k: torch.cat(v) for k, v in res.items()}

r4 = lambda t: [round(float(x), 4) for x in t.flatten().tolist()]
def write(path, texts, R, roles, extra):
    json.dump({"note": "value patching at one source position per role; per-head arrays of 144 (index = layer*12 + head); clean metric minus patched metric",
               "roles": roles, **extra,
               "prompts": [dict(text=texts[i], metric_clean=round(R["mc"][i].item(), 4), **{r: r4(R[r][i]) for r in roles}) for i in range(len(texts))]},
              open(path, "w"), separators=(",", ":"))

if WHICH in ("ioi", "both"):
    a = src.index('    random.seed(0); torch.manual_seed(0)'); z = src.index('    if TEST: groups = groups[:TEST]')
    exec("\n".join(l[4:] for l in src[a:z].split("\n")))                 # same groups as diagnostics.py
    if TEST: groups = groups[:TEST]
    roles = ["IO", "S1", "S1+1", "S2", "END"]
    def ioi_batch(g):
        def mk():
            io, s = g["io"], g["s"]
            def metric(last, rep):
                i = torch.arange(last.shape[0]); return last[i, io.repeat(rep)] - last[i, s.repeat(rep)]
            tc, tx = model.to_tokens(g["clean"]), model.to_tokens(g["corr"]); assert tc.shape == tx.shape
            P = {r: [] for r in roles}
            for k in range(tc.shape[0]):
                t = tc[k].tolist(); sp = [j for j, x in enumerate(t) if x == s[k].item()]; ip = [j for j, x in enumerate(t) if x == io[k].item()]
                assert len(sp) == 2 and len(ip) == 1, (g["clean"][k], sp, ip)
                P["IO"].append(ip[0]); P["S1"].append(sp[0]); P["S1+1"].append(sp[0] + 1); P["S2"].append(sp[1]); P["END"].append(len(t) - 1)
            return prepare(tc, tx, metric), {r: torch.tensor(v) for r, v in P.items()}
        return mk
    print(f"IOI: {sum(len(g['clean']) for g in groups)} prompts", flush=True)
    R = run([ioi_batch(g) for g in groups], roles)
    write(f"{OUT}/diagnostics_values_ioi{'_test' if TEST else ''}.json", [c for g in groups for c in g["clean"]], R, roles,
          {"task": "ioi", "target": "IO", "corruption": "ABC (three fresh names)"})

if WHICH in ("gt", "both"):
    gt = json.load(open(f"{OUT}/gt/per_prompt.json"))["prompts"]
    if TEST: gt = gt[:24 * TEST]
    corr = []
    for p in gt:
        st = f"{p['century']}{p['yy']:02d}"; i = p["text"].index(st); corr.append(p["text"][:i] + f"{p['century']}01" + p["text"][i + 4:])
    YEAR = torch.tensor([model.to_single_token(f"{y:02d}") for y in range(100)])
    roles = ["YY", "END"]
    def gt_batch(rows):
        def mk():
            yy = torch.tensor([gt[r]["yy"] for r in rows]); mask = (torch.arange(100)[None, :] > yy[:, None]).float()
            def metric(last, rep):
                p = last.softmax(-1)[:, YEAR]; m = mask.repeat(rep, 1)
                return (p * m).sum(-1) - (p * (1 - m)).sum(-1)
            tc = model.to_tokens([gt[r]["text"] for r in rows]); tx = model.to_tokens([corr[r] for r in rows]); assert tc.shape == tx.shape
            P = {r: [] for r in roles}
            for k, r in enumerate(rows):
                t = tc[k].tolist(); cen = model.to_single_token(f" {gt[r]['century']}")
                j = [q for q in range(1, len(t)) if t[q] == YEAR[gt[r]["yy"]].item() and t[q - 1] == cen]
                assert len(j) == 1, gt[r]["text"]
                P["YY"].append(j[0]); P["END"].append(len(t) - 1)
            return prepare(tc, tx, metric), {r: torch.tensor(v) for r, v in P.items()}
        return mk
    chunks = [list(range(i, min(i + 24, len(gt)))) for i in range(0, len(gt), 24)]
    print(f"greater-than: {len(gt)} prompts", flush=True)
    R = run([gt_batch(c) for c in chunks], roles)
    write(f"{OUT}/gt/diagnostics_values{'_test' if TEST else ''}.json", [p["text"] for p in gt], R, roles,
          {"task": "gt", "target": "YY", "corruption": "01"})
print(f"done in {time.time() - t0:.0f}s", flush=True)
