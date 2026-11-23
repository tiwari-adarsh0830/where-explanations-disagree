"""Fixes from the Nov 22 page review: link targets clear the sticky toolbar, Unpin keeps the view in place, a cell click
pins without clearing the head, the Why panel uses the statistics' 2x rule and says so, the tour shows top-10 overlap
where its text talks about the top 10, one name per method, and each wide table scrolls on its own on a phone.
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import D3, CHROME, BASE
from playwright.sync_api import sync_playwright

errs = []; bad = 0
def page(b, url, w=1120, h=732):
    pg = b.new_page(viewport={"width": w, "height": h})
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.route("**/cdn.jsdelivr.net/npm/d3@7", lambda r: r.fulfill(body=D3, content_type="application/javascript"))
    pg.goto(BASE + url); pg.wait_for_selector("#medtbl"); pg.wait_for_timeout(400)
    return pg
with sync_playwright() as p:
    b = p.chromium.launch(executable_path=CHROME)
    # 1. intro link to the medium table lands below the sticky toolbar
    pg = page(b, "")
    pg.evaluate("document.querySelector('details').open = true")
    pg.click("#introfind a[href='#medtitle']"); pg.wait_for_timeout(500)
    tb = pg.evaluate("document.querySelector('.controls').getBoundingClientRect().bottom")
    top = pg.evaluate("document.getElementById('medtitle').getBoundingClientRect().top")
    ok1 = tb <= top < tb + 60; bad += not ok1
    print(f"link target below toolbar: toolbar bottom {tb:.0f}, title top {top:.0f} -> {'ok' if ok1 else 'HIDDEN'}")
    # 2. clicking a matrix cell pins the pair and keeps a head selected, even when clicked twice
    cell = pg.locator("#intsvg rect.ic").nth(30)
    cell.click(force=True); pg.wait_for_timeout(300); sel1 = pg.evaluate("state.sel")
    cell.click(force=True); pg.wait_for_timeout(300); sel2 = pg.evaluate("state.sel")
    ok2 = sel1 is not None and sel1 == sel2 and pg.evaluate("!!state.pair"); bad += not ok2
    print(f"cell click keeps the head: {sel1} then {sel2} -> {'ok' if ok2 else 'TOGGLED'}")
    # 3. Unpin keeps the pinned-pair line where it was
    pg.locator("#unpin").scroll_into_view_if_needed(); pg.wait_for_timeout(200)
    y0 = pg.evaluate("document.getElementById('intpair').getBoundingClientRect().top")
    pg.click("#unpin"); pg.wait_for_timeout(300)
    y1 = pg.evaluate("document.getElementById('intpair').getBoundingClientRect().top")
    ok3 = abs(y1 - y0) < 3 and pg.evaluate("state.pair") is None; bad += not ok3
    print(f"unpin keeps the view: {y0:.0f} -> {y1:.0f} -> {'ok' if ok3 else 'JUMPED'}")
    # 4. Why panel: 2x rule stated, sub-labels, single-head ranks named, no "-0.00"
    pg.evaluate("state.sel = 65; render()"); pg.wait_for_timeout(300)
    why = pg.inner_text("#whybody")
    ok4 = "within 2×" in why and "(where it acts)" in why and "(what it reads)" in why and "denoising rank" in why and "noising rank" in why and "−0.00" not in why and "-0.00" not in why
    bad += not ok4; print("why panel wording:", "ok" if ok4 else "PROBLEM")
    # 5. one name per method in the three tables and the evidence line
    names = pg.eval_on_selector_all("#contrasttbl tr th:first-child, #medtbl tr th:first-child", "l => l.map(x => x.innerText)")
    ok5 = not any(n.startswith("Denoising") for n in names) and any(n.startswith("Single-head patching") for n in names); bad += not ok5
    print("one name per method:", "ok" if ok5 else names)
    # 6. head box gives feedback
    pg.fill("#headin", "9.9"); pg.press("#headin", "Enter"); pg.wait_for_timeout(300)
    ok6 = "see why" in pg.inner_text("#headstatus"); bad += not ok6; print("head box feedback:", "ok" if ok6 else "MISSING")
    pg.close()
    # 7. tour step 5 shows top-10 overlap, step 6 goes back to Spearman, step 7 brings the pinned pair into view
    pg = page(b, "", 1320, 1000)
    pg.evaluate("showTour(4)"); pg.wait_for_timeout(600); m5 = pg.evaluate("state.t6metric")
    pg.evaluate("showTour(5)"); pg.wait_for_timeout(600); m6 = pg.evaluate("state.t6metric")
    pg.evaluate("showTour(6)"); pg.wait_for_timeout(1200)
    r = pg.evaluate("(() => { const r = document.getElementById('intpair').getBoundingClientRect(); return [r.top, r.bottom, innerHeight]; })()")
    ok7 = m5 == "top" and m6 == "rho" and 0 <= r[0] and r[1] <= r[2]; bad += not ok7
    print(f"tour: step 5 metric {m5}, step 6 {m6}, pinned text in view {0 <= r[0] and r[1] <= r[2]} -> {'ok' if ok7 else 'PROBLEM'}")
    pg.close()
    # 8. phone: tables scroll inside their own wrapper, titles stay put, the swipe hint shows
    for url in ("", "?task=gt"):
        pg = page(b, url, 390, 844)
        over = pg.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
        wraps = pg.evaluate("[...document.querySelectorAll('#matchwrap .tscroll')].map(d => d.scrollWidth > d.clientWidth)")
        title_scrolls = pg.evaluate("(() => { const t = document.getElementById('medtitle'); return t.closest('.tscroll') !== null; })()")
        hint = pg.evaluate("[...document.querySelectorAll('.swipehint')].some(e => getComputedStyle(e).display !== 'none')")
        ok8 = over == 0 and len(wraps) == 3 and any(wraps) and not title_scrolls and hint; bad += not ok8
        print(f"phone {url or 'ioi'}: overflow {over}, table wrappers {len(wraps)} (scrolling {sum(wraps)}), title outside wrapper {not title_scrolls}, swipe hint {hint} -> {'ok' if ok8 else 'PROBLEM'}")
        pg.close()
    b.close()
print(f"mismatches: {bad}")
print("console errors:", errs)
