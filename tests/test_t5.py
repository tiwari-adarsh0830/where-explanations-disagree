import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import R, D3, OUT, CHROME, BASE
import json, numpy as np
from scipy.stats import spearmanr
from playwright.sync_api import sync_playwright
P=json.load(open(f"{R}/results/ioi_v3_results_per_prompt.json"))["prompts"]
exp={}
for t in range(15):
  for o in ("BABA","ABBA"):
    ps=[p for p in P if p["tpl"]==t and p["order"]==o]
    tr=np.mean([p["truth"] for p in ps],0); A=np.mean([p["attn"] for p in ps],0)
    rho=spearmanr(np.abs(A[:,0]),np.abs(tr)).correlation
    exp[(f"T{t+1}",o)]=(np.mean([p["logit_diff"] for p in ps]), rho, np.mean([p["group_drop"]["name_movers"] for p in ps]), tr[5*12+5])
errs=[]
with sync_playwright() as p:
  b=p.chromium.launch(executable_path=CHROME)
  for scheme in ("light","dark"):
    pg=b.new_page(viewport={"width":1320,"height":1000},color_scheme=scheme)
    pg.on("console",lambda m: m.type in("error","warning") and errs.append(m.text)); pg.on("pageerror",lambda e: errs.append(str(e)))
    pg.route("**/cdn.jsdelivr.net/npm/d3@7",lambda r: r.fulfill(body=D3,content_type="application/javascript"))
    pg.goto(BASE); pg.wait_for_selector("#t5 svg")
    if scheme=="light":
      pg.locator("#grids .gridbox").nth(0).locator("g.c").nth(65).click(); pg.mouse.move(2,2)  # select 5.5
      rows=pg.eval_on_selector_all("#t5tbl tr","rs=>rs.slice(1).map(r=>[...r.children].map(td=>td.textContent))")
      hdr=pg.eval_on_selector_all("#t5tbl tr:first-child th","h=>h.map(x=>x.textContent)")
      print("columns:",hdr, "rows:",len(rows))
      bad=0
      for r in rows:
        e=exp[(r[0],r[1])]; got=[float(v.replace("−","-")) for v in r[3:7]]
        ok=all(abs(g-x)<0.006 for g,x in zip(got,e)); bad+=not ok
        if not ok: print("MISMATCH",r,e)
      print("mismatches:",bad)
      print("example row T5:",[r for r in rows if r[0]=="T5"])
      pg.evaluate("document.getElementById('tip').style.opacity=0")
    pg.locator("#templates").screenshot(path=f"{OUT}/t5_{scheme}.png")
    pg.close()
  m=b.new_page(viewport={"width":390,"height":900}); m.route("**/cdn.jsdelivr.net/npm/d3@7",lambda r: r.fulfill(body=D3,content_type="application/javascript"))
  m.goto(BASE); m.wait_for_selector("#t5 svg")
  print("mobile page overflow px:", m.evaluate("document.documentElement.scrollWidth - innerWidth"))
  b.close()
print("console errors:",errs)
# overflow check with a head selected, at two desktop widths
with sync_playwright() as p:
  b=p.chromium.launch(executable_path=CHROME)
  for w in (1251, 1100):
    pg=b.new_page(viewport={"width":w,"height":1000})
    pg.route("**/cdn.jsdelivr.net/npm/d3@7",lambda r: r.fulfill(body=D3,content_type="application/javascript"))
    pg.goto(BASE); pg.wait_for_selector("#t5 svg")
    pg.locator("#grids .gridbox").nth(0).locator("g.c").nth(65).click(); pg.mouse.move(2,2)
    o=pg.evaluate("(()=>{const w=document.querySelector('.t5wrap');return w.scrollWidth-w.clientWidth})()")
    print(f"width {w}: T5 sideways overflow {o}px, panels {pg.locator('#t5 text.ph').count()}")
    if w==1251: pg.evaluate("document.getElementById('tip').style.opacity=0"); pg.locator("#templates").screenshot(path=f"{OUT}/t5_4col.png")
    pg.close()
  b.close()
