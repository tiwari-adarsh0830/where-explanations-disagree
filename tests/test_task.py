import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import R, D3, OUT, CHROME, BASE
import json, numpy as np
from scipy.stats import spearmanr
from playwright.sync_api import sync_playwright
P=json.load(open(f"{R}/results/gt/per_prompt.json")); A=json.load(open(f"{R}/results/gt/attn_scores.json"))["prompts"]
V=json.load(open(f"{R}/results/gt/patching_variants_per_prompt.json"))["prompts"]
tr=np.mean([p["truth"] for p in P["prompts"]],0); at=np.mean([a["attn"] for a in A],0); vw=np.mean([a["vw"] for a in A],0)
dla=np.mean([p["dla"] for p in P["prompts"]],0); attr=np.mean([p["attr"] for p in P["prompts"]],0); den=np.mean([v["scores"]["denoise_01"] for v in V],0)
rho=lambda a,b: spearmanr(np.abs(a),np.abs(b)).correlation
exp={"attn_end":rho(at[:,0],tr),"attn_best":rho(at.max(1),tr),"vw_best":rho(vw.max(1),tr),"dla":rho(dla,tr),"attr":rho(attr,tr)}
print("expected:",{k:round(v,3) for k,v in exp.items()}, "attn_end vs denoise", round(rho(at[:,0],den),3))
errs=[]
def page(b, url, scheme="light", w=1320):
    pg=b.new_page(viewport={"width":w,"height":1000},color_scheme=scheme)
    pg.on("console",lambda m: m.type in("error","warning") and errs.append((url,m.text))); pg.on("pageerror",lambda e: errs.append((url,str(e))))
    pg.route("**/cdn.jsdelivr.net/npm/d3@7",lambda r: r.fulfill(body=D3,content_type="application/javascript"))
    pg.goto(url); pg.wait_for_selector("#t5 svg"); return pg
with sync_playwright() as p:
  b=p.chromium.launch(executable_path=CHROME)
  # IOI unchanged
  pg=page(b,BASE)
  print("IOI title:",pg.title(),"| task pressed:",pg.eval_on_selector("#task button[aria-pressed=true]","e=>e.dataset.v"))
  print("IOI stat:",pg.inner_text("#stat")); print("IOI t4note:",pg.inner_text("#t4note")[:120])
  print("IOI stage options:",pg.eval_on_selector_all("#stage option","o=>o.map(x=>x.textContent)"))
  pg.click("#task button[data-v=gt]"); pg.wait_for_selector("#t5 svg"); pg.wait_for_timeout(500)
  print("after click url:",pg.url,"| title:",pg.title()); pg.close()
  for scheme in ("light","dark"):
    pg=page(b,BASE+"?task=gt",scheme)
    if scheme=="light":
      print("intro:",pg.inner_text(".sub")[:200])
      print("cond hidden:",not pg.is_visible("#cond"),"| t3 order hidden:",not pg.is_visible("#t3order"))
      print("ref options:",pg.eval_on_selector_all("#ref option","o=>o.map(x=>x.textContent)"))
      print("grid titles:",pg.eval_on_selector_all("#grids h3","h=>h.map(x=>x.textContent)"))
      got={}
      for k in exp:
        pg.select_option("#method",k); got[k]=pg.inner_text("#stat")
      for k in got: print(f"  {k:9s} page: {got[k][:60]}  exp {exp[k]:.3f}")
      pg.select_option("#method","attn_end")
      print("t4 panels:",pg.eval_on_selector_all("#t4 .panel h3","h=>h.map(x=>x.textContent)"))
      print("t4 table:",pg.eval_on_selector_all("#t4tbl tr","rs=>rs.map(r=>[...r.children].map(c=>c.textContent).join(' | '))"))
      print("t4 note:",pg.inner_text("#t4note"))
      print("t5 table:",pg.eval_on_selector_all("#t5tbl tr","rs=>rs.map(r=>[...r.children].map(c=>c.textContent).join(' | '))"))
      print("t6 note:",pg.inner_text("#t6note"))
      print("t6 table:",pg.eval_on_selector_all("#t6tbl tr","rs=>rs.map(r=>[...r.children].map(c=>c.textContent).join(' | '))"))
      # select 9.1 and check detail + T3
      pg.locator("#grids .gridbox").nth(0).locator("g.c").nth(9*12+1).click(); pg.wait_for_selector("#t3 svg"); pg.mouse.move(2,2)
      print("detail:",pg.inner_text("#dbody")[:300].replace("\n"," / "))
      print("t3 options:",pg.eval_on_selector_all("#t3tpl option","o=>o.map(x=>x.textContent)"))
      print("t3 note:",pg.inner_text("#t3note"))
      print("t3 role labels:",pg.eval_on_selector_all("#t3 text.tok.role","t=>[...new Set(t.map(x=>x.textContent))]"))
      pg.select_option("#t3tpl","4"); pg.wait_for_timeout(300); print("t3 note bucket 5:",pg.inner_text("#t3note"))
      print("t5 cols with head:",pg.eval_on_selector_all("#t5tbl tr:first-child th","h=>h.map(x=>x.textContent)"))
      # switch to denoising
      pg.locator("#t6 svg g").filter(has_text="denoising,").first.click(); pg.wait_for_timeout(300)
      print("den stat:",pg.inner_text("#stat")[:60],"| t4what:",pg.inner_text("#t4what")[:80])
      print("den t4 note:",pg.inner_text("#t4note"))
      print("den t5 header:",pg.eval_on_selector_all("#t5tbl tr:first-child th","h=>h.map(x=>x.textContent)"))
      pg.evaluate("document.getElementById('tip').style.opacity=0")
      for sec,name in (("#grids","grids"),("#attention","t3"),("#groups","t4"),("#templates","t5"),("#refs","t6")):
        pg.locator(sec).first.screenshot(path=f"{OUT}/gt_{name}.png")
    else:
      pg.locator("#groups").screenshot(path=f"{OUT}/gt_t4_dark.png")
    pg.close()
  m=page(b,BASE+"?task=gt",w=390)
  print("mobile overflow:",m.evaluate("document.documentElement.scrollWidth - innerWidth"))
  b.close()
print("console errors:",errs)
