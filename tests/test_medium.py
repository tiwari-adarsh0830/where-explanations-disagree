"""The other-models table on the assumption-check card (GPT-2 small, GPT-2 medium, Pythia-410M), and the intro line built from it.

Every cell is recomputed here from the result files (results/*diagnostics_summary.json for small,
results/gpt2-medium/analysis.json and results/pythia-410m/analysis.json for the others) with the same verdict rule the page
states, and compared with the page. Pythia ran IOI only, so greater-than shows small and medium.
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import R, D3, CHROME, BASE
import json
from playwright.sync_api import sync_playwright

OTHER = [("medium", "GPT-2 medium", json.load(open(os.path.join(R, "results/gpt2-medium/analysis.json")))),
         ("pythia", "Pythia-410M", json.load(open(os.path.join(R, "results/pythia-410m/analysis.json"))))]
ORDER = ["position", "target", "direct", "linearity", "redundancy"]
METHOD = {"position": "attn_end", "target": "attn_end", "direct": "dla", "linearity": "attr", "redundancy": "denoise"}

def pf(p): return "–" if p is None else "p < 0.001" if p < 0.001 else f"p = {p:.{3 if p < 0.1 else 2}f}"
def cnt(c):
    if not c: return "–"
    return f"{c['broken_where_disagree']} of {c['n_disagree']} vs {c['broken_where_agree']} of {c['n_agree']}" if c["n_disagree"] else f"none misranked; {c['broken_where_agree']} of {c['n_agree']} right break"
def verdict(c, holm):
    if not c or not c["n_disagree"] or not c["n_agree"]: return ("untestable", "no head misranked" if c and not c["n_disagree"] else "not testable")
    m = min(c["n_disagree"], c["n_agree"])
    few = f" (only {m} {'misranked' if c['n_disagree'] < c['n_agree'] else 'right'} heads)" if m < 5 else ""
    fp = c["fisher_p_one_sided"]; up = c["broken_where_disagree"] / c["n_disagree"] > c["broken_where_agree"] / c["n_agree"]
    if c["broken_where_agree"] / c["n_agree"] >= 0.5: return ("everywhere", "breaks even where it agrees")
    if holm is not None and holm < 0.05 and fp is not None and fp < 0.05: return ("separates", "separates right from wrong" + few)
    if up and ((holm is not None and holm < 0.1) or (fp is not None and fp < 0.1)): return ("direction", "points the right way" + few)
    if up: return ("leans", "leans the right way (no test below 0.1)" + few)
    return ("none", "does not separate")
def other_holm(M, task, v):
    t = next((x for x in M["multiple_testing"] if x["task"] == task and x["check"] == v), None)
    return t["holm"]["planned_8" if t["planned"] else "all_10"]["mannwhitney_p"] if t and "holm" in t else None

errs = []; total_bad = 0
with sync_playwright() as p:
    b = p.chromium.launch(executable_path=CHROME)
    for task, url in (("ioi", ""), ("gt", "?task=gt")):
        S = json.load(open(os.path.join(R, "results/diagnostics_summary.json" if task == "ioi" else "results/gt/diagnostics_summary.json")))
        pg = b.new_page(viewport={"width": 1320, "height": 1000})
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.route("**/cdn.jsdelivr.net/npm/d3@7", lambda r: r.fulfill(body=D3, content_type="application/javascript"))
        pg.goto(BASE + url); pg.wait_for_selector("#medtbl"); pg.wait_for_timeout(300)
        rows = {(r[0], r[1]): [c for c in r[2:]] for r in pg.eval_on_selector_all("#medtbl tr[data-check]",
                "rs => rs.map(r => [r.dataset.check, r.dataset.model, ...[...r.querySelectorAll('td')].map(c => c.innerText)])")}
        OM = [(k, n, M) for k, n, M in OTHER if task in M]
        bad = 0; changed = []; outl_want = []
        for v in ORDER:
            cs = S["contrast"].get(v) or S["contrast"].get(METHOD[v]); hs = cs.get("validity", {}).get("holm_p") if cs else None
            vs = verdict(cs, hs)
            want = {("small"): ["GPT-2 small", cnt(cs), pf(cs["fisher_p_one_sided"] if cs else None), pf(hs), vs[1]]}
            differs = False
            for k, n, M in OM:
                cm = M[task]["contrast"].get(v); hm = other_holm(M, task, v); vm = verdict(cm, hm)
                want[k] = [n, cnt(cm), pf(cm["fisher_p_one_sided"] if cm else None), pf(hm), ("changed: " if vm[0] != vs[0] else "") + vm[1]]
                if vm[0] != vs[0]: differs = True; outl_want.append((v, k))
            for k, w in want.items():
                got = rows.get((v, k))
                if got != w: bad += 1; print(f"   {task} {v} {k}: page {got} vs expected {w}")
            if differs: changed.append(v)
        bad += len(rows) != len(ORDER) * (1 + len(OM))
        outlined = [tuple(x) for x in pg.eval_on_selector_all("#medtbl tr[data-check]", "rs => rs.filter(r => r.querySelector('td.vd').classList.contains('chg')).map(r => [r.dataset.check, r.dataset.model])")]
        bad += sorted(outlined) != sorted(outl_want)
        two = len(OM) == 1
        lead_change, lead_same = ("A second model changes part of this.", "A second model gives the same verdicts.") if two else ("Two other models change part of this.", "Two other models give the same verdicts.")
        intro = [x for x in pg.eval_on_selector_all("#introfind li", "l => l.map(x => x.innerText)") if x.startswith(lead_change) or x.startswith(lead_same)]
        bad += not intro
        if intro:
            bad += ("no check changes its verdict" in intro[0]) != (not changed)
            bad += intro[0].startswith(lead_change) != bool(changed)
            bad += ("findings above describe GPT-2 small" in intro[0]) != bool(changed)
            bad += (("hold for both sizes" if two else "hold for all three models") in intro[0]) == bool(changed)
            bad += (" but points the right way in" in intro[0] and "points the right way in small but points" in intro[0])   # shared verdicts are grouped
        title = pg.inner_text("#medtitle"); bad += ("Pythia-410M" in title) != True
        if task == "gt": bad += "IOI only" not in title
        pg.click("#introfind a[href='#medtitle']"); pg.wait_for_timeout(300)
        top = pg.evaluate("document.getElementById('medtitle').getBoundingClientRect().top")
        bad += not (-5 < top < 300)
        # phone: the new table must not push the page sideways, whether the page loads on a phone or is resized to one
        pg.set_viewport_size({"width": 390, "height": 844}); pg.wait_for_timeout(700)
        over = pg.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
        pg.reload(); pg.wait_for_selector("#medtbl"); pg.wait_for_timeout(300)
        over += pg.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
        print(f"== {task}: rows {len(rows)}, checks that change {changed}, outlined {outlined}; intro: {intro[0][:230] if intro else 'MISSING'}")
        print(f"   phone overflow (px): {over}")
        print(f"   mismatches: {bad}")
        total_bad += bad
        pg.close()
    b.close()
print("console errors:", errs)
