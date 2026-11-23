"""The GPT-2 small vs medium table on the assumption-check card, and the intro line built from it.

Every cell is recomputed here from the result files (results/*diagnostics_summary.json for small,
results/gpt2-medium/analysis.json for medium) with the same verdict rule the page states, and compared with the page.
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import R, D3, CHROME, BASE
import json
from playwright.sync_api import sync_playwright

MED = json.load(open(os.path.join(R, "results/gpt2-medium/analysis.json")))
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
def med_holm(task, v):
    t = next((x for x in MED["multiple_testing"] if x["task"] == task and x["check"] == v), None)
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
        rows = {r[0]: r[1:] for r in pg.eval_on_selector_all("#medtbl tr[data-check]", "rs => rs.map(r => [r.dataset.check, ...[...r.children].slice(1).map(c => c.innerText)])")}
        bad = 0; changed = []
        for v in ORDER:
            cs = S["contrast"].get(v) or S["contrast"].get(METHOD[v]); cm = MED[task]["contrast"].get(v)
            hs = cs.get("validity", {}).get("holm_p") if cs else None; hm = med_holm(task, v)
            vs, vm = verdict(cs, hs), verdict(cm, hm)
            fs = cs["fisher_p_one_sided"] if cs else None; fm = cm["fisher_p_one_sided"] if cm else None
            want = [cnt(cs), pf(fs), pf(hs), vs[1], cnt(cm), pf(fm), pf(hm), ("changed: " if vs[0] != vm[0] else "") + vm[1]]
            got = rows.get(v)
            if got != want: bad += 1; print(f"   {task} {v}: page {got} vs expected {want}")
            if vs[0] != vm[0]: changed.append(v)
        outlined = pg.eval_on_selector_all("#medtbl tr[data-check]", "rs => rs.filter(r => r.children[8].classList.contains('chg')).map(r => r.dataset.check)")
        bad += sorted(outlined) != sorted(changed)
        intro = [t for t in pg.eval_on_selector_all("#introfind li", "l => l.map(x => x.innerText)") if t.startswith("A second model")]
        bad += not intro
        if intro:
            bad += ("no check changes its verdict" in intro[0]) != (not changed)
            bad += intro[0].startswith("A second model changes part of this.") != bool(changed)
            bad += ("findings above describe GPT-2 small" in intro[0]) != bool(changed)
            bad += ("hold for both sizes" in intro[0]) == bool(changed)
        pg.click("#introfind a[href='#medtitle']"); pg.wait_for_timeout(300)
        top = pg.evaluate("document.getElementById('medtitle').getBoundingClientRect().top")
        bad += not (-5 < top < 300)
        # phone: the new table must not push the page sideways, whether the page loads on a phone or is resized to one
        pg.set_viewport_size({"width": 390, "height": 844}); pg.wait_for_timeout(700)
        over = pg.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
        pg.reload(); pg.wait_for_selector("#medtbl"); pg.wait_for_timeout(300)
        over += pg.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
        print(f"== {task}: rows {len(rows)}, verdict changes {changed}, outlined {outlined}; intro: {intro[0][:230] if intro else 'MISSING'}")
        print(f"   phone overflow (px): {over}")
        print(f"   mismatches: {bad}")
        total_bad += bad
        pg.close()
    b.close()
print("console errors:", errs)
