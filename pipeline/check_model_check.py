"""Check pipeline/model_check.py against the committed GPT-2 small results.

model_check.py recomputes every score and check in one script so it can run on other model sizes. Run it on gpt2-small
first (a few batches are enough) and this script compares each quantity, prompt by prompt and head by head, with the
files the main analysis uses. The largest absolute difference should be at the rounding level (about 1e-4).

With --analysis it also checks pipeline/analysis_model.py: it writes GPT-2 small's committed numbers in model_check.py's
format and checks that analysis_model.py gives exactly the numbers in results/analysis_diagnostics.json and the corrected
p-values in results/analysis_validity.json.

Run:  python pipeline/model_check.py gpt2-small /tmp/mc_small both 2
      python pipeline/check_model_check.py /tmp/mc_small results [--analysis]
"""
import json, os, subprocess, sys, tempfile
import numpy as np

ANALYSIS = "--analysis" in sys.argv; args = [a for a in sys.argv[1:] if a != "--analysis"]
M = args[0] if len(args) > 0 else "/tmp/mc_small"
D = args[1] if len(args) > 1 else "results"
def load(p): return json.load(open(p))
def pp(d): return d["prompts"]
worst = 0.0
def cmp(name, a, b):
    global worst
    a, b = np.asarray(a, float), np.asarray(b, float); d = float(np.abs(a - b).max()); scale = float(np.abs(b).max())
    worst = max(worst, d / max(scale, 1e-9))
    print(f"  {name:32s} max |diff| {d:.2e}   (largest value {scale:.2f})")

for task in ("ioi", "gt"):
    suffix = "_test"
    try: mc = load(f"{M}/{task}{suffix}.json")
    except FileNotFoundError:
        try: mc = load(f"{M}/{task}.json")
        except FileNotFoundError: continue
    P = pp(mc); n = len(P); texts = [p["text"] for p in P]
    get = lambda k: np.array([p[k] for p in P])
    print(f"{task}: {n} prompts from {M}")
    if task == "ioi":
        dg = pp(load(f"{D}/diagnostics_ioi.json"))[:n]; v3 = pp(load(f"{D}/ioi_v3_results_per_prompt.json"))[:n]
        at = pp(load(f"{D}/attn_scores.json"))[:n]; var = pp(load(f"{D}/patching_variants_per_prompt.json"))[:n]
        vv = load(f"{D}/diagnostics_values_ioi.json"); vals = pp(vv)[:n]; full_ref = load(f"{D}/diagnostics_ioi.json")
        assert texts == [p["text"] for p in dg] == [p["text"] for p in v3] == [p["text"] for p in at] == [p["text"] for p in var] == [p["text"] for p in vals]
        cmp("full vs v3 truth", get("full"), [p["truth"] for p in v3])
        cmp("half vs diagnostics", get("half"), [p["half"] for p in dg])
        cmp("direct vs diagnostics", get("direct"), [p["direct"] for p in dg])
        cmp("end vs patching_variants", get("end"), [p["scores"]["noise_abc_end"] for p in var])
        cmp("denoise vs patching_variants", get("denoise"), [p["scores"]["denoise_abc"] for p in var])
        cmp("dla vs v3", get("dla"), [p["dla"] for p in v3])
        cmp("attr vs v3", get("attr"), [p["attr"] for p in v3])
        cmp("attn_end vs attn_scores end>io", get("attn_end"), [[r[0] for r in p["attn"]] for p in at])
        cmp("val_target vs diagnostics_values IO", get("val_target"), [p["IO"] for p in vals])
    else:
        dg = pp(load(f"{D}/gt/diagnostics.json"))[:n]; g = pp(load(f"{D}/gt/per_prompt.json"))[:n]
        at = pp(load(f"{D}/gt/attn_scores.json"))[:n]; var = pp(load(f"{D}/gt/patching_variants_per_prompt.json"))[:n]
        vals = pp(load(f"{D}/gt/diagnostics_values.json"))[:n]; full_ref = load(f"{D}/gt/diagnostics.json")
        assert texts == [p["text"] for p in dg] == [p["text"] for p in g] == [p["text"] for p in at] == [p["text"] for p in vals]
        cmp("full vs gt truth", get("full"), [p["truth"] for p in g])
        cmp("half vs diagnostics", get("half"), [p["half"] for p in dg])
        cmp("direct vs diagnostics", get("direct"), [p["direct"] for p in dg])
        cmp("end vs diagnostics", get("end"), [p["end"] for p in dg])
        cmp("denoise vs patching_variants", get("denoise"), [p["scores"]["denoise_01"] for p in var])
        cmp("dla vs gt_check", get("dla"), [p["dla"] for p in g])
        cmp("attr vs gt_check", get("attr"), [p["attr"] for p in g])
        cmp("attn_end vs attn_scores end>yy", get("attn_end"), [[r[0] for r in p["attn"]] for p in at])
        cmp("val_target vs diagnostics_values YY", get("val_target"), [p["YY"] for p in vals])
    # pairs: the top 24 heads are picked from this run's prompts, so compare only the pairs both files contain
    ref_pairs = {tuple(x): k for k, x in enumerate(full_ref["pairs"])}
    common = [(k, ref_pairs[tuple(x)]) for k, x in enumerate(mc["pairs"]) if tuple(x) in ref_pairs]
    if common:
        cmp(f"pair ({len(common)} shared pairs)", [[p["pair"][a] for a, _ in common] for p in P], [[p["pair"][b] for _, b in common] for p in dg])
print(f"\nlargest difference relative to the largest value: {worst:.1e}  ->  {'OK' if worst < 1e-3 else 'MISMATCH'}")

if ANALYSIS:
    tmp = tempfile.mkdtemp()
    for task in ("ioi", "gt"):
        g = "" if task == "ioi" else "gt/"
        dg = load(f"{D}/{g}diagnostics{'_ioi' if task == 'ioi' else ''}.json"); P = pp(dg); n = len(P)
        sc = pp(load(f"{D}/ioi_v3_results_per_prompt.json" if task == "ioi" else f"{D}/gt/per_prompt.json"))[:n]
        at = pp(load(f"{D}/{g}attn_scores.json"))[:n]; var = pp(load(f"{D}/{g}patching_variants_per_prompt.json"))[:n]
        vals = pp(load(f"{D}/{g}diagnostics_values{'_ioi' if task == 'ioi' else ''}.json"))[:n]
        rows = []
        for p, q, a, w, x in zip(P, sc, at, var, vals):
            r = dict(text=p["text"], tpl=p["tpl"], metric_clean=p["metric_clean"], metric_corrupt=p["metric_corrupt"],
                     attn_end=[v[0] for v in a["attn"]], dla=q["dla"], attr=q["attr"], full=p["full"], half=p["half"], direct=p["direct"], pair=p["pair"])
            if task == "ioi": r.update(order=p["order"], end=w["scores"]["noise_abc_end"], denoise=w["scores"]["denoise_abc"], val_target=x["IO"])
            else: r.update(yy=p["yy"], end=p["end"], denoise=w["scores"]["denoise_01"], val_target=x["YY"])
            rows.append(r)
        json.dump({"model": "gpt2-small (committed files)", "n_layers": 12, "n_heads": 12, "top_heads": dg["top_heads"], "pairs": dg["pairs"],
                   "task": task, "target": "IO" if task == "ioi" else "YY", "prompts": rows}, open(f"{tmp}/{task}.json", "w"))
    subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis_model.py"), tmp],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    a, c, v = load(f"{tmp}/analysis.json"), load(f"{D}/analysis_diagnostics.json"), load(f"{D}/analysis_validity.json")
    bad = []
    for task in ("ioi", "gt"):
        for k in ("matching", "contrast", "contrast_all_heads", "top10_disagreements", "slope0_vs_attr_spearman"):
            if a[task][k] != c[task][k]: bad.append(f"{task} {k}")
        for ref in ("noising", "denoising"):
            for meth in ("attn_end", "dla", "attr"):
                if a[task]["layer_control"][ref][meth] != c[task]["layer_control"][ref][meth]: bad.append(f"{task} layer control {ref} {meth}")
    key = lambda t: (t["task"], t["check"])
    vt = {key(t): t for t in v["multiple_testing"]}
    for t in a["multiple_testing"]:
        if "holm" in t and t.get("holm") != vt[key(t)].get("holm"): bad.append(f"corrected p {key(t)}")
    print("analysis_model.py on the committed GPT-2 small numbers:", "identical to analysis_diagnostics.json and analysis_validity.json" if not bad else f"DIFFERENT: {bad}")
