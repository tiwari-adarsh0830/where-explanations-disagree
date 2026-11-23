"""Do the assumption checks explain where each method disagrees with patching?

For each head, a violation score in [0, 1] says how badly the assumption behind a method breaks
(computed from pipeline/diagnostics.py, under noising with the task's main corruption):
  position   final-token attention assumes the head acts at the final token:
             |all-position effect - final-position effect| / (|all| + |final|)
  direct     DLA assumes the effect is direct:
             |total effect - direct-path effect| / (|total| + |direct|)
  linearity  attribution patching assumes the effect is linear in the activation. From
             f(0.5) and f(1): slope at 0  a = 4 f(0.5) - f(1), curvature  b = 2 f(1) - 4 f(0.5);
             violation = |b| / (|a| + |b|)
  target     final-token attention also assumes the head reads the target (IO / YY): value patching at the
             target position only (pipeline/diagnostics_values.py), |full - value_at_target| / (|full| + |value_at_target|);
             position_or_target is the larger of the two attention checks
  redundancy single-head noising assumes no other head changes this head's effect (top 24 heads):
             sum_j |INT_ij| / (sum_j |INT_ij| + |single_i|),  INT_ij = pair_ij - single_i - single_j
Disagreement of a method with patching for a head: |log2 rank(method) - log2 rank(patching)|,
on |score|, over heads that rank in the top 30 for either (the tail is noise) and whose patching effect is at least
2% of the largest (below that every violation ratio is a ratio of two near-zero numbers; the unfiltered contrast is
also written, as contrast_all_heads, for transparency).

Tests, with 95% bootstrap intervals over prompts (500 resamples, numpy seed 0):
  1. matching: Spearman between each violation and each method's disagreement. If the checks
     are right, each method's disagreement follows its own assumption (the diagonal) more than
     the others.
  2. top-10: of the (up to) 10 heads where a method disagrees most, counting only heads whose
     ranks differ by at least a factor of 2, how many have a violation above 0.5.
  2b. contrast: the same count for every head whose ranks differ by 2x or more, against the heads
     (same top-30 pool) whose ranks are within 2x, with a one-sided Fisher exact test. A check
     only explains disagreements if it breaks more often where the method disagrees.
  2c. robustness of 2b: the same test under 27 choices of cutoffs (pool: top 20/30/40; misranked:
     ranks 1.5x/2x/3x apart; broken: violation above 0.3/0.5/0.7), counting how often the
     misranked heads break more often (direction) and how often p < 0.1; plus a bootstrap over
     prompts (300 resamples) of the difference in broken rates at the default cutoffs.
  3. layer control: every method's agreement with patching as plain Spearman, as a partial
     Spearman with layer held out, and as within-layer Spearman averaged over layers.
  4. rank intervals for every head under every method (for the page).
Writes results/analysis_diagnostics.json (IOI and greater-than) and per-head summaries for the
page: results/diagnostics_summary.json and results/gt/diagnostics_summary.json.

Run:  python pipeline/analysis_diagnostics.py results
"""
import json, sys
import numpy as np
from scipy.stats import spearmanr, rankdata, fisher_exact

D = sys.argv[1] if len(sys.argv) > 1 else "results"
SUFFIX = sys.argv[2] if len(sys.argv) > 2 else ""          # "_test" to run on the test files
B = 500; H = 12; NH = 144; TOP = 30
FMIN = 0.02     # heads whose patching effect is below 2% of the largest are left out of the tests: a violation ratio of two near-zero numbers is noise (the page already calls them "too small to judge")
rng = np.random.default_rng(0)
EPS = 1e-9

def rho(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.ptp(a) == 0 or np.ptp(b) == 0: return float("nan")
    return float(spearmanr(a, b).correlation)
def ranks(x): return rankdata(-np.abs(x), method="ordinal")
def ci(v):
    v = np.asarray([x for x in v if x == x]); return [round(float(x), 3) for x in np.percentile(v, [2.5, 97.5])] if len(v) else None
def idx(s): l, h = map(int, s.split(".")); return l * H + h
def partial_rho(a, b, z):
    ra, rb, rz = rankdata(np.abs(a)), rankdata(np.abs(b)), rankdata(z)
    def resid(y, x): X = np.c_[np.ones_like(x), x]; return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    return float(np.corrcoef(resid(ra, rz), resid(rb, rz))[0, 1])
LAYER = np.repeat(np.arange(12), 12).astype(float)
def within_layer(a, b):
    v = [rho(np.abs(a[l * H:(l + 1) * H]), np.abs(b[l * H:(l + 1) * H])) for l in range(12)]
    v = [x for x in v if x == x]; return float(np.mean(v)) if v else float("nan")

def task_data(task):
    if task == "ioi":
        dg = json.load(open(f"{D}/diagnostics_ioi{SUFFIX}.json")); P = dg["prompts"]; n = len(P)
        v3 = json.load(open(f"{D}/ioi_v3_results_per_prompt.json"))["prompts"][:n]
        at = json.load(open(f"{D}/attn_scores.json"))["prompts"][:n]
        var = json.load(open(f"{D}/patching_variants_per_prompt.json"))["prompts"][:n]
        assert [p["text"] for p in P] == [p["text"] for p in v3] == [p["text"] for p in at] == [p["text"] for p in var]
        X = {"end": np.array([p["scores"]["noise_abc_end"] for p in var]), "denoise": np.array([p["scores"]["denoise_abc"] for p in var]),
             "flip": np.array([p["scores"]["noise_flip"] for p in var]),
             "dla": np.array([p["dla"] for p in v3]), "attr": np.array([p["attr"] for p in v3]),
             "order": np.array([p["order"] for p in P])}
    else:
        dg = json.load(open(f"{D}/gt/diagnostics{SUFFIX}.json")); P = dg["prompts"]; n = len(P)
        g = json.load(open(f"{D}/gt/per_prompt.json"))["prompts"][:n]
        at = json.load(open(f"{D}/gt/attn_scores.json"))["prompts"][:n]
        var = json.load(open(f"{D}/gt/patching_variants_per_prompt.json"))["prompts"][:n]
        assert [p["text"] for p in P] == [p["text"] for p in g] == [p["text"] for p in at]
        X = {"end": np.array([p["end"] for p in P]), "denoise": np.array([p["scores"]["denoise_01"] for p in var]),
             "dla": np.array([p["dla"] for p in g]), "attr": np.array([p["attr"] for p in g]), "order": None}
    X.update({"full": np.array([p["full"] for p in P]), "half": np.array([p["half"] for p in P]),
              "direct": np.array([p["direct"] for p in P]), "pair": np.array([p["pair"] for p in P]),
              "attn": np.array([p["attn"] for p in at]), "vw": np.array([p["vw"] for p in at])})
    vf = f"{D}/diagnostics_values_ioi{SUFFIX}.json" if task == "ioi" else f"{D}/gt/diagnostics_values{SUFFIX}.json"
    try:
        vv = json.load(open(vf)); assert [p["text"] for p in vv["prompts"]] == [p["text"] for p in P]
        X["val"] = {r: np.array([p[r] for p in vv["prompts"]]) for r in vv["roles"]}; X["target_role"] = vv["target"]
    except FileNotFoundError:
        pass
    X["top"] = [idx(h) for h in dg["top_heads"]]; X["pairs"] = [(idx(a), idx(c)) for a, c in dg["pairs"]]
    X["n"] = n; X["tpl"] = np.array([p["tpl"] for p in P])
    return X

def measures(X, s):
    """Per-head means and violation scores over the prompts in s."""
    m = {k: X[k][s].mean(0) for k in ("full", "half", "direct", "end", "denoise", "dla", "attr")}
    A = X["attn"][s].mean(0); V = X["vw"][s].mean(0)
    m["attn_end"], m["attn_best"], m["vw_best"] = A[:, 0], A.max(1), V.max(1)
    F = m["full"]
    a, b = 4 * m["half"] - F, 2 * F - 4 * m["half"]
    m["slope0"], m["curv"] = a, b
    v = {"position": np.abs(F - m["end"]) / (np.abs(F) + np.abs(m["end"]) + EPS),
         "direct": np.abs(F - m["direct"]) / (np.abs(F) + np.abs(m["direct"]) + EPS),
         "linearity": np.abs(b) / (np.abs(a) + np.abs(b) + EPS)}
    if "val" in X:
        for r, arr in X["val"].items(): m["val_" + r] = arr[s].mean(0)
        vt = m["val_" + X["target_role"]]
        v["target"] = np.abs(F - vt) / (np.abs(F) + np.abs(vt) + EPS)
        v["position_or_target"] = np.maximum(v["position"], v["target"])
    pm = X["pair"][s].mean(0); INT = np.zeros((NH, NH)) * np.nan
    for k, (i, j) in enumerate(X["pairs"]):
        INT[i, j] = INT[j, i] = pm[k] - F[i] - F[j]
    red = np.full(NH, np.nan)
    for i in X["top"]:
        tot = np.nansum(np.abs(INT[i])); red[i] = tot / (tot + abs(F[i]) + EPS)
    v["redundancy"] = red
    return m, v, INT

METHOD_OF = {"position": "attn_end", "direct": "dla", "linearity": "attr", "redundancy": "denoise"}
CHECK_OF = {**METHOD_OF, "target": "attn_end", "position_or_target": "attn_end"}     # every check and the method it is about
METHOD_NAME = {"attn_end": "final-token attention", "dla": "DLA", "attr": "attribution patching", "denoise": "denoising (vs noising)",
               "attn_best": "best-position attention", "vw_best": "value-weighted best position"}

def disagreement(m, method, fmin=FMIN):
    rm, rr = ranks(m[method]), ranks(m["full"]); F = np.abs(m["full"])
    rel = ((rm <= TOP) | (rr <= TOP)) & (F >= fmin * F.max())
    return np.abs(np.log2(rm) - np.log2(rr)), rel

def matching(m, v):
    out = {}
    for vk in v:
        out[vk] = {}
        for vm_k, meth in METHOD_OF.items():
            g, rel = disagreement(m, meth)
            ok = rel & ~np.isnan(v[vk])
            out[vk][meth] = rho(v[vk][ok], g[ok]) if ok.sum() > 5 else float("nan")
    return out

def analyse(task):
    X = task_data(task); n = X["n"]; ALL = np.arange(n)
    m, v, INT = measures(X, ALL)
    res = {"n_prompts": n, "top_heads": [f"{i // H}.{i % H}" for i in X["top"]]}
    res["matching"] = {vk: {mk: round(x, 3) for mk, x in d.items()} for vk, d in matching(m, v).items()}
    boots = [rng.integers(0, n, n) for _ in range(B)]
    bm = [matching(*measures(X, s)[:2]) for s in boots]
    res["matching_ci95"] = {vk: {mk: ci([b[vk][mk] for b in bm]) for mk in METHOD_OF.values()} for vk in v}
    # top-10 disagreements per method and how many have a large violation of the method's own assumption
    top10 = {}
    for vk, meth in METHOD_OF.items():
        g, rel = disagreement(m, meth)
        order = [i for i in np.argsort(-g, kind="stable") if rel[i] and g[i] >= 1][:10]     # at least a factor of 2 apart in rank
        rm, rr = ranks(m[meth]), ranks(m["full"])
        top10[meth] = {"assumption": vk, "n_with_violation_over_0.5": int(sum((v[vk][i] > 0.5) for i in order if v[vk][i] == v[vk][i])),
                       "n_checked": int(sum(v[vk][i] == v[vk][i] for i in order)),
                       "heads": [{"head": f"{i // H}.{i % H}", "rank_method": int(rm[i]), "rank_patching": int(rr[i]),
                                  "violation": None if v[vk][i] != v[vk][i] else round(float(v[vk][i]), 3)} for i in order]}
    res["top10_disagreements"] = top10
    contrast, contrast_all = {}, {}
    for vk, meth in CHECK_OF.items():
        if vk not in v: continue
        g0, rel0 = disagreement(m, meth, 0.0); ok0 = rel0 & ~np.isnan(v[vk]); br0 = v[vk] > 0.5
        d0, a0 = ok0 & (g0 >= 1), ok0 & (g0 < 1); bd0, nd0, ba0, na0 = int((br0 & d0).sum()), int(d0.sum()), int((br0 & a0).sum()), int(a0.sum())
        contrast_all[vk] = {"broken_where_disagree": bd0, "n_disagree": nd0, "broken_where_agree": ba0, "n_agree": na0,
                            "fisher_p_one_sided": round(float(fisher_exact([[bd0, nd0 - bd0], [ba0, na0 - ba0]], alternative="greater")[1]), 4) if nd0 and na0 else None}
        g, rel = disagreement(m, meth); ok = rel & ~np.isnan(v[vk]); br = v[vk] > 0.5
        dI, aI = ok & (g >= 1), ok & (g < 1)
        bd, nd, ba, na = int((br & dI).sum()), int(dI.sum()), int((br & aI).sum()), int(aI.sum())
        p = fisher_exact([[bd, nd - bd], [ba, na - ba]], alternative="greater")[1] if nd and na else None
        contrast[vk] = {"method": meth, "assumption": vk, "broken_where_disagree": bd, "n_disagree": nd, "broken_where_agree": ba, "n_agree": na,
                          "fisher_p_one_sided": None if p is None else round(float(p), 4)}
    def ctab(mm, vv, meth, vk, top, gap, thr):
        rm, rr = ranks(mm[meth]), ranks(mm["full"]); g = np.abs(np.log2(rm) - np.log2(rr))
        F = np.abs(mm["full"]); ok = ((rm <= top) | (rr <= top)) & ~np.isnan(vv[vk]) & (F >= FMIN * F.max()); br = vv[vk] > thr
        dI, aI = ok & (g >= np.log2(gap)), ok & (g < np.log2(gap))
        return int((br & dI).sum()), int(dI.sum()), int((br & aI).sum()), int(aI.sum())
    rb = np.random.default_rng(1); bsamp = [rb.integers(0, n, n) for _ in range(300)]; bmeas = [measures(X, s)[:2] for s in bsamp]
    for vk, meth in CHECK_OF.items():
        if vk not in v: continue
        dirn = sig = tot = 0
        for top in (20, 30, 40):
            for gap in (1.5, 2, 3):
                for thr in (0.3, 0.5, 0.7):
                    bd, nd, ba, na = ctab(m, v, meth, vk, top, gap, thr)
                    if not (nd and na): continue
                    tot += 1; dirn += bd / nd > ba / na
                    sig += fisher_exact([[bd, nd - bd], [ba, na - ba]], alternative="greater")[1] < 0.1
        diffs = []
        for mm, vv in bmeas:
            bd, nd, ba, na = ctab(mm, vv, meth, vk, TOP, 2, 0.5)
            if nd and na: diffs.append(bd / nd - ba / na)
        contrast[vk]["robustness"] = {"settings": tot, "misranked_break_more": int(dirn), "p_below_0.1": int(sig),
                                        "boot_diff_ci95": ci(diffs) if len(diffs) >= 50 else None, "boot_n": len(diffs)}
    res["contrast"] = contrast; res["contrast_all_heads"] = contrast_all; res["fmin"] = FMIN
    # layer control, against noising (the reference the checks use) and denoising
    lc = {}
    for ref_name, ref in (("noising", m["full"]), ("denoising", m["denoise"])):
        lc[ref_name] = {}
        for meth in ("attn_end", "attn_best", "vw_best", "dla", "attr"):
            sc = m[meth]
            lc[ref_name][meth] = {"spearman": round(rho(np.abs(sc), np.abs(ref)), 3),
                                  "partial_layer_held_out": round(partial_rho(sc, ref, LAYER), 3),
                                  "within_layer_mean": round(within_layer(sc, ref), 3)}
        lc[ref_name]["layer_baseline"] = round(rho(LAYER, np.abs(ref)), 3)
    lc_b = {k: [] for k in ("attn_end", "attn_best", "vw_best", "dla", "attr")}
    for s in boots[:200]:
        mm = measures(X, s)[0]
        for meth in lc_b: lc_b[meth].append(partial_rho(mm[meth], mm["full"], LAYER))
    for meth in lc_b: lc["noising"][meth]["partial_ci95"] = ci(lc_b[meth])
    res["layer_control"] = lc
    # descriptive: linearity check against attribution patching
    res["slope0_vs_attr_spearman"] = round(rho(np.abs(m["slope0"]), np.abs(m["attr"])), 3)
    # per-head summary for the page, with rank intervals under every score
    R = {k: [] for k in ("full", "denoise", "attn_end", "attn_best", "vw_best", "dla", "attr")}
    for s in boots[:300]:
        mm = measures(X, s)[0]
        for k in R: R[k].append(ranks(mm[k]))
    rank_ci = {k: np.percentile(np.array(R[k]), [2.5, 97.5], axis=0).round().astype(int).tolist() for k in R}
    r3 = lambda a: [None if x != x else round(float(x), 4) for x in np.asarray(a).tolist()]
    summary = {"task": task, "reference": "noising, " + ("ABC corruption" if task == "ioi" else "“01” corruption"),
               "note": "per-head arrays: 144 entries, index = layer*12 + head; violations in [0, 1], null where not computed",
               "effects": {k: r3(m[k]) for k in ("full", "end", "direct", "slope0", "curv", "denoise", "dla", "attr") + tuple(k for k in m if k.startswith("val_"))},
               "target_role": X.get("target_role"),
               "violations": {k: r3(x) for k, x in v.items()},
               "top_heads": res["top_heads"], "interaction": [[None if INT[i, j] != INT[i, j] else round(float(INT[i, j]), 4) for j in X["top"]] for i in X["top"]],
               "rank_ci": rank_ci, "matching": res["matching"], "matching_ci95": res["matching_ci95"],
               "top10": res["top10_disagreements"], "contrast": res["contrast"], "contrast_all_heads": res["contrast_all_heads"], "fmin": FMIN,
               "layer_control": res["layer_control"]}
    if X["order"] is not None:    # IOI: per name order too, for the page's filter
        summary["by_order"] = {}
        for o in ("BABA", "ABBA"):
            s = np.where(X["order"] == o)[0]; mo, vo, INTo = measures(X, s)
            summary["by_order"][o] = {"effects": {k: r3(mo[k]) for k in ("full", "end", "direct", "slope0", "curv", "denoise", "dla", "attr") + tuple(k for k in mo if k.startswith("val_"))},
                                      "violations": {k: r3(x) for k, x in vo.items()},
                                      "interaction": [[None if INTo[i, j] != INTo[i, j] else round(float(INTo[i, j]), 4) for j in X["top"]] for i in X["top"]]}
    return res, summary

out = {}
for task in ("ioi", "gt"):
    res, summary = analyse(task); out[task] = res
    json.dump(summary, open(f"{D}/diagnostics_summary{SUFFIX}.json" if task == "ioi" else f"{D}/gt/diagnostics_summary{SUFFIX}.json", "w"), separators=(",", ":"))
    print(f"\n===== {task} ({res['n_prompts']} prompts) =====")
    print("matching (rows: violation, cols: disagreement of method):")
    cols = list(METHOD_OF.values()); print(" " * 12 + "".join(f"{c:>12s}" for c in cols))
    for vk, d in res["matching"].items(): print(f"{vk:12s}" + "".join(f"{d[c]:12.3f}" for c in cols))
    print("contrast (violation > 0.5 where ranks are 2x apart vs within 2x):")
    for meth, c in res["contrast"].items():
        r = c["robustness"]
        print(f"  {meth:18s} {c['broken_where_disagree']}/{c['n_disagree']} vs {c['broken_where_agree']}/{c['n_agree']}  Fisher p = {c['fisher_p_one_sided']}"
              f"  | 27 cutoffs: direction {r['misranked_break_more']}/{r['settings']}, p<0.1 {r['p_below_0.1']}/{r['settings']}; bootstrap diff {r['boot_diff_ci95']} ({r['boot_n']} resamples)")
    for meth, t in res["top10_disagreements"].items():
        print(f"top-10 disagreements, {meth}: {t['n_with_violation_over_0.5']}/{t['n_checked']} with {t['assumption']} violation > 0.5:",
              [(h['head'], h['rank_method'], h['rank_patching'], h['violation']) for h in t['heads'][:6]])
    print("layer control (noising):", {k: (x["spearman"], x["partial_layer_held_out"], x["within_layer_mean"]) if isinstance(x, dict) else x for k, x in res["layer_control"]["noising"].items()})
    print("slope at 0 vs attribution patching:", res["slope0_vs_attr_spearman"])
json.dump(out, open(f"{D}/analysis_diagnostics{SUFFIX}.json", "w"), indent=1)
print("\nwrote", f"{D}/analysis_diagnostics{SUFFIX}.json")
