import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import R, D3, OUT, CHROME, BASE
import json
from playwright.sync_api import sync_playwright
f55=json.load(open(f"{R}/results/attn/L5_H5.json")); AS=json.load(open(f"{R}/results/attn_scores.json"))
meta=[g for g in AS["groups"] if g["tpl"]==0 and g["order"]=="BABA"][0]
item=[g for g in f55["groups"] if g["tpl"]==0 and g["order"]=="BABA"][0]
exp_cell=item["attn"][meta["s2_pos"]][meta["s1_pos"]+1]; exp_vw=item["vw"][meta["s2_pos"]][meta["s1_pos"]+1]
errs=[]
with sync_playwright() as p:
  b=p.chromium.launch(executable_path=CHROME)
  for scheme in ("light","dark"):
    pg=b.new_page(viewport={"width":1320,"height":1000},color_scheme=scheme)
    pg.on("console",lambda m: m.type in("error","warning") and errs.append(m.text)); pg.on("pageerror",lambda e: errs.append(str(e)))
    pg.route("**/cdn.jsdelivr.net/npm/d3@7",lambda r: r.fulfill(body=D3,content_type="application/javascript"))
    pg.goto(BASE); pg.wait_for_selector("#t5 svg")
    if scheme=="light":
      for k in ("attn_end","attn_best","vw_best","dla","attr"):
        pg.select_option("#method",k); print(f"{k:10s}", pg.inner_text("#stat"))
      print("grids drawn:", pg.locator("#grids .gridbox").count())
      print("T3 before selection:", pg.inner_text("#t3empty"))
    pg.locator("#grids .gridbox").nth(0).locator("g.c").nth(65).click(); pg.mouse.move(2,2)
    pg.wait_for_selector("#t3 svg rect"); pg.wait_for_timeout(300)
    if scheme=="light":
      print("T3 note:", pg.inner_text("#t3note"))
      print("expected S2 -> token after S1 for 5.5, T1 BABA:", exp_cell, "% raw,", exp_vw, "% value-weighted")
      pg.click('#t3mode button[data-v="vw"]'); pg.wait_for_timeout(200); print("VW note:", pg.inner_text("#t3note"))
      pg.click('#t3mode button[data-v="attn"]'); pg.select_option("#t3tpl","4"); pg.click('#t3order button[data-v="ABBA"]'); pg.wait_for_timeout(200)
      print("T5 ABBA note:", pg.inner_text("#t3note"))
      print("global filter still pooled:", pg.get_attribute('#cond button[data-v="all"]',"aria-pressed"))
      pg.click('#cond button[data-v="BABA"]'); pg.wait_for_timeout(200)
      print("after global BABA, T3 order pressed:", pg.get_attribute('#t3order button[data-v="BABA"]',"aria-pressed"))
      pg.click('#cond button[data-v="all"]'); pg.select_option("#t3tpl","0"); pg.click('#t3order button[data-v="BABA"]')
      # head 4.11 should now rank 1 under best pair
      pg.select_option("#method","attn_best")
      pg.locator("#grids .gridbox").nth(0).locator("g.c").nth(65).click()  # deselect 5.5
      pg.locator("#grids .gridbox").nth(0).locator("g.c").nth(4*12+11).click(); pg.mouse.move(2,2); pg.wait_for_timeout(400)
      print("4.11 panel:", pg.inner_text("#dbody").split("\n")[:12])
      print("4.11 T3 note:", pg.inner_text("#t3note"))
      pg.locator("#grids .gridbox").nth(0).locator("g.c").nth(4*12+11).click(); pg.locator("#grids .gridbox").nth(0).locator("g.c").nth(65).click(); pg.mouse.move(2,2); pg.wait_for_timeout(300)
    pg.evaluate("document.getElementById('tip').style.opacity=0")
    pg.locator("#attention").screenshot(path=f"{OUT}/t3_{scheme}.png")
    pg.close()
  m=b.new_page(viewport={"width":390,"height":900}); m.route("**/cdn.jsdelivr.net/npm/d3@7",lambda r: r.fulfill(body=D3,content_type="application/javascript"))
  m.goto(BASE); m.wait_for_selector("#t5 svg")
  print("mobile page overflow px:", m.evaluate("document.documentElement.scrollWidth - innerWidth"))
  b.close()
print("console errors:",errs)
