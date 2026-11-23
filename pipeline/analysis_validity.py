"""Validity checks for the contrast test in analysis_diagnostics.py: shared noise (split-half) and multiple testing.

Part 1, split-half: is the link between "misranked" and "assumption breaks" shared noise?

In the main analysis both the rank gap (method rank vs patching rank) and the violation score are computed from the same
prompts, and both use the same patching effect. Noise in that effect can move both at once and create a link that is not
there. Here the prompts are split in two: half A decides which heads are misranked (ranks, the top-30 pool and the 2% rule
all from A), half B gives the violation scores. IOI is split by base template (15 templates, 7 or 8 per half, both name
orders kept together), greater-than by random halves of the prompts. 200 splits, each used both ways round (A then B, and
B then A), so 400 tests per check. As a control, the same test with both parts taken from one half ("same half") shows
what the half-size data give when the noise is shared, so the drop from "same half" to "cross half" is what shared noise
contributed.

All definitions (violations, rank gap, top-30 pool, 2x cutoff, 0.5 threshold, 2% rule) come from analysis_diagnostics.py,
which is loaded rather than copied.

Part 2, multiple testing: the contrast is tested for several checks in two tasks. Holm-adjusted p-values for (a) the eight
tests planned before the results (each method's own assumption, two tasks: position, direct, linearity, redundancy) and
(b) a stricter family of ten that adds the "reads the target" check added later. The Fisher test throws away the size of
each violation, so the same family is also tested with a one-sided Mann-Whitney test of the violation scores of misranked
against right heads, and with a permutation test (20,000 shuffles of the misranked/right labels) of the difference in mean
violation; all Holm-adjusted.

Writes results/analysis_validity.json.
Run:  python pipeline/analysis_validity.py results
"""
import json, sys
import numpy as np
from scipy.stats import fisher_exact

src = open("pipeline/analysis_diagnostics.py").read()
exec(src[:src.index("\nout = {}\n")])      # task_data, measures, CHECK_OF, TOP, FMIN, ranks, ... exactly as in the main analysis

NSPLIT = 200

def contrast_counts(m_def, v_eval, meth, vk):
    """Misranked vs right decided on m_def; broken (> 0.5) read from v_eval."""
    rm, rr = ranks(m_def[meth]), ranks(m_def["full"]); g = np.abs(np.log2(rm) - np.log2(rr))
    F = np.abs(m_def["full"]); ok = ((rm <= TOP) | (rr <= TOP)) & (F >= FMIN * F.max()) & ~np.isnan(v_eval[vk])
    br = v_eval[vk] > 0.5
    dI, aI = ok & (g >= 1), ok & (g < 1)
    return int((br & dI).sum()), int(dI.sum()), int((br & aI).sum()), int(aI.sum())

def summarise(rows):
    rows = [r for r in rows if r[1] and r[3]]
    if not rows: return None
    diff = [bd / nd - ba / na for bd, nd, ba, na in rows]
    ps = [fisher_exact([[bd, nd - bd], [ba, na - ba]], alternative="greater")[1] for bd, nd, ba, na in rows]
    med = lambda k: float(np.median([r[k] for r in rows]))
    return {"tests": len(rows), "misranked_break_more": int(sum(d > 0 for d in diff)), "p_below_0.1": int(sum(p < 0.1 for p in ps)),
            "p_below_0.05": int(sum(p < 0.05 for p in ps)), "median_p": round(float(np.median(ps)), 3),
            "diff_median": round(float(np.median(diff)), 3), "diff_2.5_97.5": [round(float(x), 3) for x in np.percentile(diff, [2.5, 97.5])],
            "median_counts": f"{med(0):.0f}/{med(1):.0f} vs {med(2):.0f}/{med(3):.0f}"}

out = {}
for task in ("ioi", "gt"):
    X = task_data(task); n = X["n"]; rng = np.random.default_rng(7)
    m_all, v_all, _ = measures(X, np.arange(n))
    cross = {vk: [] for vk in CHECK_OF if vk in v_all}; same = {vk: [] for vk in cross}
    for _ in range(NSPLIT):
        if task == "ioi":
            t = rng.permutation(15); inA = np.isin(X["tpl"], t[:7 + rng.integers(0, 2)])
        else:
            inA = np.zeros(n, bool); inA[rng.permutation(n)[: n // 2]] = True
        A, B = np.where(inA)[0], np.where(~inA)[0]
        (mA, vA, _), (mB, vB, _) = measures(X, A), measures(X, B)
        for vk in cross:
            meth = CHECK_OF[vk]
            cross[vk] += [contrast_counts(mA, vB, meth, vk), contrast_counts(mB, vA, meth, vk)]
            same[vk] += [contrast_counts(mA, vA, meth, vk), contrast_counts(mB, vB, meth, vk)]
    out[task] = {"n_prompts": n, "splits": NSPLIT, "split_by": "base template" if task == "ioi" else "random half of prompts",
                 "checks": {vk: {"method": CHECK_OF[vk], "full_data": dict(zip(("broken_where_disagree", "n_disagree", "broken_where_agree", "n_agree"),
                                                                              contrast_counts(m_all, v_all, CHECK_OF[vk], vk))),
                                 "cross_half": summarise(cross[vk]), "same_half": summarise(same[vk])} for vk in cross}}
    print(f"\n===== {task}: {n} prompts, {NSPLIT} splits x 2 directions =====")
    print(f"{'check':20s} {'full data':>16s} | {'cross half: dir, p<.1, median diff [95%]':>48s} | {'same half: dir, p<.1, median diff':>36s}")
    for vk, r in out[task]["checks"].items():
        fd = r["full_data"]; c, s = r["cross_half"], r["same_half"]
        fmt = lambda z: "n/a" if z is None else f"{z['misranked_break_more']}/{z['tests']}, {z['p_below_0.1']}/{z['tests']}, {z['diff_median']:+.2f} {z['diff_2.5_97.5']}"
        print(f"{vk:20s} {fd['broken_where_disagree']:>3d}/{fd['n_disagree']:<3d} vs {fd['broken_where_agree']:>2d}/{fd['n_agree']:<3d} | {fmt(c):>48s} | {fmt(s):>36s}")
# The matching table (Spearman of each violation with each method's rank gap) can be inflated the same way, so it gets the
# same treatment: rank gap from one half, violation from the other, median over splits; and the same-half version as control.
def match_cross(m_def, v_eval):
    res = {}
    for vk in v_eval:
        res[vk] = {}
        for meth in METHOD_OF.values():
            g, rel = disagreement(m_def, meth); ok = rel & ~np.isnan(v_eval[vk])
            res[vk][meth] = rho(v_eval[vk][ok], g[ok]) if ok.sum() > 5 else float("nan")
    return res
for task in ("ioi", "gt"):
    X = task_data(task); n = X["n"]; rng = np.random.default_rng(11)
    cr, sm = [], []
    for _ in range(NSPLIT):
        if task == "ioi":
            t = rng.permutation(15); inA = np.isin(X["tpl"], t[:7 + rng.integers(0, 2)])
        else:
            inA = np.zeros(n, bool); inA[rng.permutation(n)[: n // 2]] = True
        (mA, vA, _), (mB, vB, _) = measures(X, np.where(inA)[0]), measures(X, np.where(~inA)[0])
        cr += [match_cross(mA, vB), match_cross(mB, vA)]; sm += [match_cross(mA, vA), match_cross(mB, vB)]
    agg = lambda L: {vk: {meth: round(float(np.nanmedian([d[vk][meth] for d in L])), 3) for meth in METHOD_OF.values()} for vk in L[0]}
    out[task]["matching_cross_half_median"] = agg(cr); out[task]["matching_same_half_median"] = agg(sm)
    print(f"\n===== {task}: matching table, median Spearman over splits (cross half | same half) =====")
    cols = list(METHOD_OF.values()); print(" " * 20 + "".join(f"{c:>16s}" for c in cols))
    for vk in out[task]["matching_cross_half_median"]:
        print(f"{vk:20s}" + "".join(f"{out[task]['matching_cross_half_median'][vk][c]:>8.2f} | {out[task]['matching_same_half_median'][vk][c]:<5.2f}" for c in cols))
from scipy.stats import mannwhitneyu
def holm(ps):
    order = np.argsort(ps, kind="stable"); k = len(ps); adj = np.empty(k); run = 0.0
    for r, i in enumerate(order): run = max(run, min(1.0, (k - r) * ps[i])); adj[i] = run
    return adj
tests = []
prng = np.random.default_rng(3)
for task in ("ioi", "gt"):
    X = task_data(task); m, v, _ = measures(X, np.arange(X["n"]))
    for vk in ("position", "direct", "linearity", "redundancy", "target"):
        meth = CHECK_OF[vk]; g, rel = disagreement(m, meth); ok = rel & ~np.isnan(v[vk])
        dI, aI = ok & (g >= 1), ok & (g < 1); br = v[vk] > 0.5
        bd, nd, ba, na = int((br & dI).sum()), int(dI.sum()), int((br & aI).sum()), int(aI.sum())
        if not (nd and na): tests.append({"task": task, "check": vk, "note": "no misranked heads"}); continue
        pf = float(fisher_exact([[bd, nd - bd], [ba, na - ba]], alternative="greater")[1])
        xd, xa = v[vk][dI], v[vk][aI]
        pm = float(mannwhitneyu(xd, xa, alternative="greater").pvalue)
        pool = np.r_[xd, xa]; obs = xd.mean() - xa.mean(); cnt = 0
        for _ in range(20000):
            sh = prng.permutation(pool); cnt += (sh[:nd].mean() - sh[nd:].mean()) >= obs - 1e-12
        tests.append({"task": task, "check": vk, "counts": f"{bd}/{nd} vs {ba}/{na}", "mean_violation": [round(float(xd.mean()), 3), round(float(xa.mean()), 3)],
                      "fisher_p": round(pf, 4), "mannwhitney_p": round(pm, 4), "permutation_p": round((cnt + 1) / 20001, 4), "planned": vk != "target"})
valid = [t for t in tests if "fisher_p" in t]
for fam, sel in (("planned_8", [t for t in valid if t["planned"]]), ("all_10", valid)):
    for key in ("fisher_p", "mannwhitney_p", "permutation_p"):
        adj = holm(np.array([t[key] for t in sel]))
        for t, a in zip(sel, adj): t.setdefault("holm", {}).setdefault(fam, {})[key] = round(float(a), 4)
out["multiple_testing"] = tests
print("\n===== multiple testing (one-sided; Holm within the family of 8 planned tests / all 10) =====")
for t in tests:
    if "fisher_p" not in t: print(f"  {t['task']:4s} {t['check']:11s} {t['note']}"); continue
    h8, h10 = t["holm"].get("planned_8", {}), t["holm"]["all_10"]
    print(f"  {t['task']:4s} {t['check']:11s} {t['counts']:>15s} mean viol {t['mean_violation']}  Fisher {t['fisher_p']:.4f} (Holm {h8.get('fisher_p','-')} / {h10['fisher_p']})"
          f"  MW {t['mannwhitney_p']:.4f} (Holm {h8.get('mannwhitney_p','-')} / {h10['mannwhitney_p']})  perm {t['permutation_p']:.4f} (Holm {h8.get('permutation_p','-')} / {h10['permutation_p']})")
json.dump(out, open(f"{D}/analysis_validity.json", "w"), indent=1)
print("\nwrote", f"{D}/analysis_validity.json")
# Add the corrected p-values and the split-half summary to the per-task summaries the page reads (run this script after
# analysis_diagnostics.py, which writes those files). Corrected p = Holm-adjusted one-sided Mann-Whitney, within the
# 8 planned tests for the four original checks and within all 10 for the later "reads the target" check.
for task in ("ioi", "gt"):
    f = f"{D}/diagnostics_summary.json" if task == "ioi" else f"{D}/gt/diagnostics_summary.json"
    S = json.load(open(f))
    for t in tests:
        if t["task"] != task or "fisher_p" not in t or t["check"] not in S["contrast"]: continue
        fam = "planned_8" if t["planned"] else "all_10"
        sh = out[task]["checks"][t["check"]]["cross_half"]
        S["contrast"][t["check"]]["validity"] = {"mannwhitney_p": t["mannwhitney_p"], "holm_p": t["holm"][fam]["mannwhitney_p"], "holm_family": fam,
                                                 "permutation_holm_p": t["holm"][fam]["permutation_p"],
                                                 "split_half": None if sh is None else {"tests": sh["tests"], "misranked_break_more": sh["misranked_break_more"], "p_below_0.1": sh["p_below_0.1"]}}
    json.dump(S, open(f, "w"), separators=(",", ":"))
    print("added corrected p-values and split-half results to", f)
