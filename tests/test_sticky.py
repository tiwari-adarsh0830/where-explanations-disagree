import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import R, D3, OUT, CHROME, BASE
from playwright.sync_api import sync_playwright
errs=[]
with sync_playwright() as p:
  b=p.chromium.launch(executable_path=CHROME)
  for url in (BASE,BASE+"?task=gt"):
    pg=b.new_page(viewport={"width":1320,"height":900})
    pg.on("console",lambda m: m.type in("error","warning") and errs.append(m.text)); pg.on("pageerror",lambda e: errs.append(str(e)))
    pg.route("**/cdn.jsdelivr.net/npm/d3@7",lambda r: r.fulfill(body=D3,content_type="application/javascript"))
    pg.goto(url); pg.wait_for_selector("#t6 svg")
    print(url.split("/")[-1] or "ioi", "t6controls position:", pg.eval_on_selector(".t6controls","e=>getComputedStyle(e).position"))
    for target in ("#refs","#groups","#templates"):
      pg.eval_on_selector(target,"e=>window.scrollTo(0, e.getBoundingClientRect().top + scrollY + 200)"); pg.wait_for_timeout(150)
      hit=pg.evaluate("""()=>{const r=document.querySelector('#task button[data-v=ioi]').getBoundingClientRect();
        const el=document.elementFromPoint(r.x+r.width/2, r.y+r.height/2); return el ? (el.closest('#task') ? 'task toggle' : el.outerHTML.slice(0,60)) : 'nothing'}""")
      print(f"  scrolled into {target}: point on IOI button hits -> {hit}")
    pg.evaluate("window.scrollTo(0,0)")
    print("  t4 panel header:", pg.inner_text("#t4 .panel .clean"))
    if "gt" in url:
      pg.locator("#t6 svg g").filter(has_text="denoising,").first.click(); pg.wait_for_timeout(300)
      print("  after denoise:", pg.inner_text("#t4 .panel .clean"), "|", pg.inner_text("#groups summary"))
    else:
      pg.select_option("#ref","noise_flip"); pg.wait_for_timeout(300)
      print("  name flip panels:", pg.eval_on_selector_all("#t4 .panel .clean","e=>e.map(x=>x.textContent)"))
      pg.select_option("#ref","noise_mean"); pg.wait_for_timeout(300)
      print("  mean ablation:", pg.inner_text("#t4 .panel .clean"))
    pg.close()
  b.close()
print("console errors:",errs)
