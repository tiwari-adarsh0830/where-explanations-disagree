import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import R, D3, OUT, CHROME, BASE
from playwright.sync_api import sync_playwright
errs=[]
with sync_playwright() as p:
  b=p.chromium.launch(executable_path=CHROME)
  for w in (1320,390):
    pg=b.new_page(viewport={"width":w,"height":900}); pg.on("pageerror",lambda e: errs.append(e.stack))
    pg.route("**/cdn.jsdelivr.net/npm/d3@7",lambda r: r.fulfill(body=D3,content_type="application/javascript"))
    pg.goto(BASE); pg.wait_for_selector("#t5 svg")
    pg.fill("#headin","9.9"); pg.press("#headin","Enter"); pg.wait_for_timeout(200)
    print(w,"typed 9.9 ->", pg.inner_text("#dtitle"), "| url", pg.url.split("frontend/")[1])
    pg.fill("#headin","13.2"); pg.press("#headin","Enter"); print("  bad input invalid:", pg.get_attribute("#headin","aria-invalid"), "| still", pg.inner_text("#dtitle"))
    pg.focus('#grids svg[data-m="dla"]'); pg.keyboard.press("ArrowRight"); pg.keyboard.press("ArrowDown"); pg.wait_for_timeout(200)
    print("  arrows ->", pg.inner_text("#dtitle"), "| focus kept:", pg.evaluate("document.activeElement.dataset.m"), "| headin", pg.input_value("#headin"))
    print("  aria:", pg.get_attribute('#grids svg[data-m="dla"]',"aria-label"))
    pg.keyboard.press("Escape"); pg.wait_for_timeout(200); print("  escape ->", pg.inner_text("#dtitle"))
    print("  overflow:", pg.evaluate("document.documentElement.scrollWidth - innerWidth"))
    pg.close()
  b.close()
print("errors:", errs)
