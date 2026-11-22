import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import R, D3, OUT, CHROME, BASE
import json, numpy as np
from scipy.stats import spearmanr
from playwright.sync_api import sync_playwright
V=json.load(open(f"{R}/results/patching_variants_per_prompt.json")); P3=json.load(open(f"{R}/results/ioi_v3_results_per_prompt.json"))["prompts"]
A=json.load(open(f"{R}/results/attn_scores.json"))["prompts"]
vars_=V["variants"]
def means(sel):
    S={v: np.mean([V["prompts"][i]["scores"][v] for i in sel],0) for v in vars_}
    at=np.mean([A[i]["attn"] for i in sel],0); vw=np.mean([A[i]["vw"] for i in sel],0)
    M={"attn_end":at[:,0],"attn_best":at.max(1),"vw_best":vw.max(1),"dla":np.mean([P3[i]["dla"] for i in sel],0),"attr":np.mean([P3[i]["attr"] for i in sel],0)}
    return S,M
rho=lambda a,b: spearmanr(np.abs(a),np.abs(b)).correlation
S,M=means(range(120))
MAP=json.load(open(f"{R}/results/attn_map.json")); AM=np.array(MAP["attn"]["all"])
M["base_layer"]=np.repeat(np.arange(12),12).astype(float); M["base_act"]=1-AM[:,MAP["labels"].index("END"),0]
exp=[[rho(M[m],S[c]) for c in vars_] for m in ["attn_end","attn_best","vw_best","dla","attr","base_layer","base_act"]]+[[None if r==c else rho(S[r],S[c]) for c in vars_] for r in vars_]
errs=[]
with sync_playwright() as p:
  b=p.chromium.launch(executable_path=CHROME)
  for scheme in ("light","dark"):
    pg=b.new_page(viewport={"width":1320,"height":1000},color_scheme=scheme)
    pg.on("console",lambda m: m.type in("error","warning") and errs.append(m.text)); pg.on("pageerror",lambda e: errs.append(str(e)))
    pg.route("**/cdn.jsdelivr.net/npm/d3@7",lambda r: r.fulfill(body=D3,content_type="application/javascript"))
    pg.goto(BASE); pg.wait_for_selector("#t6 svg")
    if scheme=="light":
      print("ref control visible:", pg.is_visible("#ref"), "options:", pg.eval_on_selector_all("#ref option","o=>o.map(x=>x.value)"))
      rows=pg.eval_on_selector_all("#t6tbl tr","rs=>rs.slice(1).map(r=>[...r.children].map(td=>td.textContent))")
      bad=0
      for r,e in zip(rows,exp):
        for g,x in zip(r[1:],e):
          if x is None: bad+= g!="—"
          elif abs(float(g)-x)>0.002: bad+=1; print("MISMATCH",r[0],g,x)
      print("matrix rows",len(rows),"mismatches",bad)
      print("note:",pg.inner_text("#t6note"))
      s0=pg.inner_text("#t4note") if pg.query_selector("#t4note") else ""
      st0=pg.inner_text(".stat") if pg.query_selector(".stat") else ""
      # click the denoising column header
      pg.locator("#t6 svg g").filter(has_text="denoising,").first.click(); pg.wait_for_timeout(300)
      print("ref now:", pg.eval_on_selector("#ref","e=>e.value"))
      print("intro:", pg.inner_text("#refdesc")[:90])
      print("t4what:", pg.inner_text("#t4what")[:90])
      print("note after:", pg.inner_text("#t6note"))
      print("scatter stat before/after:", st0[:80], "|", pg.inner_text(".stat")[:80])
      # expected T2 rho for default method attn_end vs denoise
      print("expected attn_end vs denoise:", round(rho(M["attn_end"],S["denoise_abc"]),3))
      pg.click("#t6metric button[data-v=top]"); pg.wait_for_timeout(200)
      print("top10 view first row:", pg.eval_on_selector_all("#t6tbl tr","rs=>[...rs[1].children].map(td=>td.textContent)"))
      pg.click("#cond button[data-v=ABBA]"); pg.wait_for_timeout(300)
      print("ABBA note:", pg.inner_text("#t6note"))
      print("t4 ABBA:", pg.inner_text("#t4note")[:200])
      pg.click("#t6metric button[data-v=rho]"); pg.click("#cond button[data-v=all]"); pg.wait_for_timeout(200)
      pg.evaluate("document.getElementById('tip').style.opacity=0")
    pg.locator("#refs").screenshot(path=f"{OUT}/t6_{scheme}.png")
    pg.close()
  m=b.new_page(viewport={"width":390,"height":900}); m.route("**/cdn.jsdelivr.net/npm/d3@7",lambda r: r.fulfill(body=D3,content_type="application/javascript"))
  m.goto(BASE); m.wait_for_selector("#t6 svg")
  print("mobile page overflow px:", m.evaluate("document.documentElement.scrollWidth - innerWidth"))
  m.locator("#refs").screenshot(path=f"{OUT}/t6_mobile.png")
  b.close()
print("console errors:",errs)
