"""Figure export: every card's "Save SVG" button gives a standalone, light-theme SVG, and the page is left as it was."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import R, D3, OUT, CHROME, BASE
import xml.etree.ElementTree as ET
from playwright.sync_api import sync_playwright
NS = "{http://www.w3.org/2000/svg}"
errs, bad = [], 0
with sync_playwright() as p:
    b = p.chromium.launch(executable_path=CHROME)
    for url in ("?head=5.5", "?task=gt&head=9.1"):
        ctx = b.new_context(viewport={"width": 1320, "height": 1000}, color_scheme="dark", accept_downloads=True)
        pg = ctx.new_page()
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.route("**/cdn.jsdelivr.net/npm/d3@7", lambda r: r.fulfill(body=D3, content_type="application/javascript"))
        pg.goto(BASE + url); pg.wait_for_selector("#t5 svg"); pg.wait_for_timeout(500)
        href0 = pg.evaluate("location.href"); bg0 = pg.evaluate("getComputedStyle(document.body).backgroundColor")
        ids = pg.eval_on_selector_all("section.card[id] > .svgbtn", "l => l.filter(b => b.offsetParent).map(b => b.parentElement.id)")
        print(f"== {url}: {len(ids)} cards with a visible button: {ids}")
        for cid in ids:
            with pg.expect_download() as dl:
                pg.click(f"#{cid} > .svgbtn")
            d = dl.value; path = os.path.join(OUT, d.suggested_filename); d.save_as(path)
            txt = open(path, encoding="utf-8").read()
            try:
                root = ET.fromstring(txt.split("?>", 1)[1])
            except ET.ParseError as e:
                print(f"   {cid}: XML parse error {e}"); bad += 1; continue
            charts = root.find(f"{NS}g[@id='charts']"); texts = root.find(f"{NS}g[@id='text']")
            n_svg = len(charts) if charts is not None else 0; n_txt = len(texts) if texts is not None else 0
            bgfill = root.find(f"{NS}rect").get("fill")
            problems = []
            if "var(--" in txt: problems.append("unresolved CSS variable")
            if bgfill.replace(" ", "") != "rgb(252,252,251)": problems.append(f"background {bgfill} is not the light surface")
            if n_svg + n_txt == 0: problems.append("empty")
            for t in ("Save SVG", "Table view"):
                if t in txt: problems.append(f"contains '{t}'")
            # render the file on its own and screenshot it
            import base64
            w, h = int(float(root.get("width"))), int(float(root.get("height")))
            vp = ctx.new_page(); vp.set_viewport_size({"width": w, "height": h})
            vp.set_content(f'<body style="margin:0"><img src="data:image/svg+xml;base64,{base64.b64encode(txt.encode()).decode()}"></body>')
            vp.wait_for_timeout(150); vp.screenshot(path=path.replace(".svg", ".png")); vp.close()
            print(f"   {cid}: {d.suggested_filename} {len(txt)//1024} KB, {n_svg} charts, {n_txt} text lines" + (f"  PROBLEMS: {problems}" if problems else ""))
            bad += bool(problems)
        restored = pg.evaluate("document.documentElement.getAttribute('data-theme')") is None and pg.evaluate("getComputedStyle(document.body).backgroundColor") == bg0
        same_url = pg.evaluate("location.href") == href0
        print(f"   theme restored: {restored}; URL unchanged: {same_url}")
        bad += (not restored) + (not same_url)
        ctx.close()
    # phone width with a head selected (the Why table used to push the page sideways)
    for url in ("?head=5.5", "?task=gt&head=9.1"):
        m = b.new_page(viewport={"width": 390, "height": 900})
        m.route("**/cdn.jsdelivr.net/npm/d3@7", lambda r: r.fulfill(body=D3, content_type="application/javascript"))
        m.goto(BASE + url); m.wait_for_selector("#t5 svg"); m.wait_for_timeout(500)
        ov = m.evaluate("document.documentElement.scrollWidth - innerWidth"); print(f"   mobile overflow with {url}: {ov}"); bad += ov > 0
        m.locator("#why").screenshot(path=os.path.join(OUT, "mobile_why_" + ("gt" if "gt" in url else "ioi") + ".png")); m.close()
    b.close()
print(f"mismatches: {bad}")
print("console errors:", errs)
