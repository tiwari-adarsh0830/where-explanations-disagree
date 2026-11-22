"""The 'what it reads' check (value patching at the target token) and the pinned pair in the interaction card."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import R, D3, OUT, CHROME, BASE
import json
from playwright.sync_api import sync_playwright
errs = []
with sync_playwright() as p:
    b = p.chromium.launch(executable_path=CHROME)
    for url, head in (("?head=8.6", "8.6"), ("?task=gt&head=9.1", "9.1")):
        pg = b.new_page(viewport={"width": 1320, "height": 1000})
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.route("**/cdn.jsdelivr.net/npm/d3@7", lambda r: r.fulfill(body=D3, content_type="application/javascript"))
        pg.goto(BASE + url); pg.wait_for_selector("#t5 svg"); pg.wait_for_timeout(400)
        rows = pg.eval_on_selector_all("#whybody tr", "r => r.length - 1")
        target = [t for t in pg.eval_on_selector_all("#whybody tr td:last-child", "l => l.map(x => x.innerText)") if "reads" in t or "carries" in t or "No single position" in t]
        print(f"== {url}: why rows {rows}; target row: {target[0][:200] if target else 'MISSING'}")
        print("   contrast rows:", pg.eval_on_selector_all("#contrasttbl tr th:first-child", "l => l.map(x => x.innerText)")[1:])
        print("   violation grids:", pg.eval_on_selector_all("#vgrids h3", "l => l.map(x => x.innerText)"))
        pg.click('#evkey button[data-v="target"]'); pg.wait_for_timeout(200)
        print("   evidence (reads target):", pg.inner_text("#evstat"))
        pg.locator("#intsvg rect.ic").nth(5).click(force=True); pg.wait_for_timeout(300)
        print("   pinned:", pg.inner_text("#intpair")[:220])
        # corrected p-values and split-half results on the page must match the summary file, and the intro must follow them
        summ = json.load(open(os.path.join(R, "results/gt/diagnostics_summary.json" if "gt" in url else "results/diagnostics_summary.json")))
        hdr = pg.eval_on_selector_all("#contrasttbl tr:first-child th", "l => l.map(x => x.innerText)")
        cells = {r[0]: r for r in pg.eval_on_selector_all("#contrasttbl tr", "rs => rs.slice(1).map(r => [...r.children].map(c => c.innerText))")}
        bad = 0
        for vk, row in (("linearity", "Attribution patching"), ("direct", "DLA")):
            V = summ["contrast"][vk].get("validity")
            if V and row in cells:
                want = "p < 0.001" if V["holm_p"] < 0.001 else f"p = {V['holm_p']:.{3 if V['holm_p'] < 0.1 else 2}f}"
                bad += cells[row][hdr.index("corrected p")] != want
                sh = V["split_half"]; bad += sh is not None and cells[row][hdr.index("split-half")] != f"same direction {sh['misranked_break_more']}/{sh['tests']}, p < 0.1 in {sh['p_below_0.1']}"
        intro = [t for t in pg.eval_on_selector_all("#introfind li", "l => l.map(x => x.innerText)") if "checkable" in t][0]
        if "gt" not in url:
            bad += "Checks that separate right from wrong: for attribution patching" not in intro
            bad += "Pointing the right way, but weakly: for final-token attention" not in intro
        else:
            bad += "Checks that separate" in intro
        print("   corrected p column and intro:", "ok" if not bad else f"{bad} problem(s)")
        print(f"   mismatches: {int(not (rows >= 5 and target)) + bad}")
        pg.close()
    b.close()
print("console errors:", errs)
