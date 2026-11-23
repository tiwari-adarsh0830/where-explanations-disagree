"""The contrast test and its validity checks on another model size, next to the GPT-2 small numbers.

Reads OUTDIR/ioi.json and OUTDIR/gt.json from pipeline/model_check.py and runs the same analysis as the main results:
the contrast test (misranked vs right heads, 2% rule, top-30 pool, 27 cutoff choices, bootstrap), the matching table,
layer control, the one-sided Mann-Whitney and permutation tests with Holm correction over the 8 planned tests (10 with
the "reads the target" check), and the split-half test. Every definition is loaded from analysis_diagnostics.py and
analysis_validity.py rather than copied, so a change there changes this too. The only differences: the layer and head
counts come from the model, and only final-token attention is read (best-position and value-weighted attention are not
part of the checks, so they are left out).

Writes OUTDIR/analysis.json and prints the results next to GPT-2 small's (results/analysis_validity.json).
Run:  python pipeline/analysis_model.py results/gpt2-medium [_test]
      python pipeline/analysis_model.py results/pythia-410m          (IOI only; see the Holm note below)
"""
import json, os, sys
import numpy as np

OUTDIR = sys.argv[1] if len(sys.argv) > 1 else "results/gpt2-medium"
SFX = sys.argv[2] if len(sys.argv) > 2 else ""
HERE = os.path.dirname(os.path.abspath(__file__))

ns = {}
sys_argv = sys.argv; sys.argv = [sys_argv[0], OUTDIR, SFX]
src = open(os.path.join(HERE, "analysis_diagnostics.py")).read()
exec(src[:src.index("\nout = {}\n")], ns)                    # measures, disagreement, analyse, CHECK_OF, ... as in the main analysis
vsrc = open(os.path.join(HERE, "analysis_validity.py")).read()
exec(vsrc[vsrc.index("def contrast_counts"):vsrc.index("\nout = {}\n")], ns)   # contrast_counts, summarise
exec(vsrc[vsrc.index("def holm"):vsrc.index("tests = []")], ns)                # holm
sys.argv = sys_argv

files = {t: f"{OUTDIR}/{t}{SFX}.json" for t in ("ioi", "gt") if os.path.exists(f"{OUTDIR}/{t}{SFX}.json")}
first = json.load(open(next(iter(files.values()))))
L, H = first["n_layers"], first["n_heads"]; NH = L * H
ns.update(H=H, NH=NH, LAYER=np.repeat(np.arange(L), H).astype(float))
def within_layer(a, b):
    v = [ns["rho"](np.abs(a[l * H:(l + 1) * H]), np.abs(b[l * H:(l + 1) * H])) for l in range(L)]
    v = [x for x in v if x == x]; return float(np.mean(v)) if v else float("nan")
ns["within_layer"] = within_layer
CACHE = {}
def task_data(task):
    d = CACHE.setdefault(task, json.load(open(files[task]))); P = d["prompts"]
    g = lambda k: np.array([p[k] for p in P])
    X = {k: g(k) for k in ("full", "half", "direct", "end", "denoise", "dla", "attr", "pair")}
    X["attn"] = g("attn_end")[..., None]; X["vw"] = X["attn"]           # only final-token attention is read
    X["order"] = g("order") if task == "ioi" else None
    X["val"] = {d["target"]: g("val_target")}; X["target_role"] = d["target"]
    X["top"] = [ns["idx"](h) for h in d["top_heads"]]; X["pairs"] = [(ns["idx"](a), ns["idx"](c)) for a, c in d["pairs"]]
    X["n"] = len(P); X["tpl"] = g("tpl")
    return X
ns["task_data"] = task_data
measures, disagreement, CHECK_OF = ns["measures"], ns["disagreement"], ns["CHECK_OF"]

out = {"model": first["model"], "n_layers": L, "n_heads": H}
for task in files:
    res, _ = ns["analyse"](task)
    for ref in res["layer_control"].values():
        for k in ("attn_best", "vw_best"): ref.pop(k, None)
    P = CACHE[task]["prompts"]
    res["metric_clean_mean"] = round(float(np.mean([p["metric_clean"] for p in P])), 4)
    res["metric_corrupt_mean"] = round(float(np.mean([p["metric_corrupt"] for p in P])), 4)
    F = np.mean([p["full"] for p in P], 0)
    res["top10_patching"] = [f"{i // H}.{i % H}" for i in np.argsort(-np.abs(F), kind="stable")[:10]]
    out[task] = res

# split-half, exactly as analysis_validity.py part 1 (IOI split by base template, greater-than by random halves)
for task in files:
    X = task_data(task); n = X["n"]; rng = np.random.default_rng(7)
    m_all, v_all, _ = measures(X, np.arange(n)); cross = {vk: [] for vk in CHECK_OF if vk in v_all}
    for _ in range(200):                              # NSPLIT in analysis_validity.py
        if task == "ioi":
            t = rng.permutation(15); inA = np.isin(X["tpl"], t[:7 + rng.integers(0, 2)])
        else:
            inA = np.zeros(n, bool); inA[rng.permutation(n)[: n // 2]] = True
        if inA.all() or not inA.any(): continue           # only with a tiny test file
        (mA, vA, _), (mB, vB, _) = measures(X, np.where(inA)[0]), measures(X, np.where(~inA)[0])
        for vk in cross:
            cross[vk] += [ns["contrast_counts"](mA, vB, CHECK_OF[vk], vk), ns["contrast_counts"](mB, vA, CHECK_OF[vk], vk)]
    out[task]["split_half"] = {vk: ns["summarise"](r) for vk, r in cross.items()}

# multiple testing, exactly as analysis_validity.py part 2
from scipy.stats import fisher_exact, mannwhitneyu
tests = []; prng = np.random.default_rng(3)
for task in files:
    X = task_data(task); m, v, _ = measures(X, np.arange(X["n"]))
    for vk in ("position", "direct", "linearity", "redundancy", "target"):
        meth = CHECK_OF[vk]; g, rel = disagreement(m, meth); ok = rel & ~np.isnan(v[vk])
        dI, aI = ok & (g >= 1), ok & (g < 1); br = v[vk] > 0.5
        bd, nd, ba, na = int((br & dI).sum()), int(dI.sum()), int((br & aI).sum()), int(aI.sum())
        if not (nd and na): tests.append({"task": task, "check": vk, "counts": f"{bd}/{nd} vs {ba}/{na}", "note": "no misranked or no right heads"}); continue
        pf = float(fisher_exact([[bd, nd - bd], [ba, na - ba]], alternative="greater")[1])
        xd, xa = v[vk][dI], v[vk][aI]
        pm = float(mannwhitneyu(xd, xa, alternative="greater").pvalue)
        pool = np.r_[xd, xa]; obs = xd.mean() - xa.mean(); cnt = 0
        for _ in range(20000):
            sh = prng.permutation(pool); cnt += (sh[:nd].mean() - sh[nd:].mean()) >= obs - 1e-12
        tests.append({"task": task, "check": vk, "counts": f"{bd}/{nd} vs {ba}/{na}", "mean_violation": [round(float(xd.mean()), 3), round(float(xa.mean()), 3)],
                      "fisher_p": round(pf, 4), "mannwhitney_p": round(pm, 4), "permutation_p": round((cnt + 1) / 20001, 4), "planned": vk != "target"})
valid = [t for t in tests if "fisher_p" in t]
# A model run on one task only (Pythia: IOI only) keeps the same bar as the two-task runs: the other task's tests count as
# p = 1 in the Holm family (4 planned + 1 later test per task), so the correction is as strict as for GPT-2 small and medium.
pad = {"planned_8": 4, "all_10": 5} if len(files) == 1 else {"planned_8": 0, "all_10": 0}
out["holm_padding"] = pad
for fam, sel in (("planned_8", [t for t in valid if t["planned"]]), ("all_10", valid)):
    if not sel: continue
    for key in ("fisher_p", "mannwhitney_p", "permutation_p"):
        adj = ns["holm"](np.r_[np.array([t[key] for t in sel]), np.ones(pad[fam])])[:len(sel)]
        for t, a in zip(sel, adj): t.setdefault("holm", {}).setdefault(fam, {})[key] = round(float(a), 4)
out["multiple_testing"] = tests
json.dump(out, open(f"{OUTDIR}/analysis{SFX}.json", "w"), indent=1)

# ---------------- print, next to GPT-2 small ----------------
small = {}
try:
    sv = json.load(open(os.path.join(HERE, "..", "results/analysis_validity.json")))
    for t in sv["multiple_testing"]: small[(t["task"], t["check"])] = t
    for task in ("ioi", "gt"):
        for vk, r in sv[task]["checks"].items(): small.setdefault((task, vk), {})["split"] = r["cross_half"]
except FileNotFoundError:
    pass
def fam(t): return t["holm"]["planned_8" if t.get("planned") else "all_10"]["mannwhitney_p"] if "holm" in t else None
print(f"\n{first['model']} ({L} layers x {H} heads)")
for task in files:
    r = out[task]
    print(f"\n===== {task}: {r['n_prompts']} prompts; clean metric {r['metric_clean_mean']}, corrupted {r['metric_corrupt_mean']} =====")
    print("patching top 10:", ", ".join(r["top10_patching"]))
    print("agreement with patching (Spearman | partial, layer held out):",
          "; ".join(f"{k} {x['spearman']:.2f} | {x['partial_layer_held_out']:.2f}" for k, x in r["layer_control"]["noising"].items() if isinstance(x, dict)),
          f"; layer alone {r['layer_control']['noising']['layer_baseline']:.2f}")
    print(f"{'check':12s} {'counts':>16s} {'corrected p':>12s} {'split-half':>12s} || GPT-2 small: {'counts':>14s} {'corrected p':>12s} {'split-half':>12s}")
    for vk in ("position", "direct", "linearity", "redundancy", "target"):
        t = next((x for x in tests if x["task"] == task and x["check"] == vk), None)
        if t is None: continue
        sh = r["split_half"].get(vk); s = small.get((task, vk), {}); ssh = s.get("split")
        f = lambda z: "n/a" if not z else f"{z['misranked_break_more']}/{z['tests']}"
        cp = fam(t); scp = fam(s) if "holm" in s else None
        print(f"{vk:12s} {t['counts']:>16s} {('-' if cp is None else f'{cp:.3f}'):>12s} {f(sh):>12s} || {'':13s}{s.get('counts', '-'):>14s} "
              f"{('-' if scp is None else f'{scp:.3f}'):>12s} {f(ssh):>12s}")
print(f"\nwrote {OUTDIR}/analysis{SFX}.json")
