import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import R, D3, OUT, CHROME, BASE
import json
from playwright.sync_api import sync_playwright
D3 = D3
an = json.load(open(f"{R}/results/analysis_v3.json"))
v3 = json.load(open(f"{R}/results/ioi_v3_results.json"))
exp = {"Pooled": v3["summary"]["group_drop"], "BABA": an["by_order"]["BABA"]["group_drop"], "ABBA": an["by_order"]["ABBA"]["group_drop"]}
key = {"Name movers": "name_movers", "Backup name movers": "backups", "Both groups": "both"}
errs = []
with sync_playwright() as p:
    b = p.chromium.launch(executable_path=CHROME)
    for scheme in ("light", "dark"):
        pg = b.new_page(viewport={"width": 1320, "height": 1000}, color_scheme=scheme)
        pg.on("console", lambda m: m.type in ("error", "warning") and errs.append(m.text))
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.route("**/cdn.jsdelivr.net/npm/d3@7", lambda r: r.fulfill(body=D3, content_type="application/javascript"))
        pg.goto(BASE); pg.wait_for_selector("#t4 svg")
        if scheme == "light":
            rows = pg.eval_on_selector_all("#t4tbl tr", "rs => rs.slice(1).map(r => [...r.children].map(td => td.textContent))")
            bad = 0
            for r in rows:
                e = exp[r[0]][key[r[1]]]; got = float(r[2].replace("\u2212","-"))
                ok = abs(got - e) < 0.0015; bad += not ok
                print(f"{r[0]:6s} {r[1]:20s} together {got:+.3f} expected {e:+.3f} {'OK' if ok else 'MISMATCH'}  CI {r[3]}  sum singles {r[4]}  sep {r[5]}")
            print("mismatches:", bad)
            print("note:", pg.inner_text("#t4note"))
            # select head 9.9 (a name mover) in the first grid and check the dot appears in T4
            pg.locator("#grids .gridbox").nth(0).locator("g.c").nth(9*12+9).click()
            pg.mouse.move(2, 2)
            print("selected-head dots in T4:", pg.locator("#t4 circle").count())
            pg.click('#cond button[data-v="ABBA"]')
            print("active panel:", pg.inner_text("#t4 .panel.active h3"))
            pg.evaluate("document.getElementById('tip').style.opacity=0")
        pg.locator("#groups").screenshot(path=f"{OUT}/t4_{scheme}.png")
        pg.close()
    m = b.new_page(viewport={"width": 390, "height": 900})
    m.route("**/cdn.jsdelivr.net/npm/d3@7", lambda r: r.fulfill(body=D3, content_type="application/javascript"))
    m.goto(BASE); m.wait_for_selector("#t4 svg")
    print("mobile overflow px:", m.evaluate("document.documentElement.scrollWidth - innerWidth"))
    m.locator("#groups").screenshot(path=f"{OUT}/t4_mobile.png")
    b.close()
print("console errors:", errs)
