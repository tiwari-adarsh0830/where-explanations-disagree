import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import R, D3, OUT, CHROME, BASE
import json, numpy as np
from scipy.stats import spearmanr
from playwright.sync_api import sync_playwright
errs=[]
rho=lambda a,b: spearmanr(np.abs(a),np.abs(b)).correlation
def exp(task):
    if task=="ioi":
        m=json.load(open(f"{R}/results/attn_map.json")); T=np.mean([p["truth"] for p in json.load(open(f"{R}/results/ioi_v3_results_per_prompt.json"))["prompts"]],0)
    else:
        m=json.load(open(f"{R}/results/gt/attn_map.json")); T=np.mean([p["truth"] for p in json.load(open(f"{R}/results/gt/per_prompt.json"))["prompts"]],0)
    A=np.array(m["attn"]["all"]); L=m["labels"]; best=None
    for q in range(len(L)):
        for k in range(len(L)):
            v=A[:,q,k]
            if np.ptp(v)<1e-9: continue
            r=rho(v,T)
            if best is None or r>best[2]: best=(L[q],L[k],r)
    lay=rho(np.repeat(np.arange(12),12).astype(float),T); e=L.index("END" if task=="ioi" else "XX end"); act=rho(1-A[:,e,0],T)
    return best, lay, act, m
with sync_playwright() as p:
  b=p.chromium.launch(executable_path=CHROME)
  for task,url in (("ioi",BASE),("gt",BASE+"?task=gt")):
    best,lay,act,m=exp(task)
    print(f"== {task}: expected best {best[0]}>{best[1]} {best[2]:.3f}, layer {lay:.3f}, activity {act:.3f}")
    for scheme in ("light","dark"):
      pg=b.new_page(viewport={"width":1320,"height":1000},color_scheme=scheme)
      pg.on("console",lambda m_: m_.type in("error","warning") and errs.append(m_.text)); pg.on("pageerror",lambda e_: errs.append(str(e_)))
      pg.route("**/cdn.jsdelivr.net/npm/d3@7",lambda r: r.fulfill(body=D3,content_type="application/javascript"))
      pg.goto(url); pg.wait_for_selector("#mapsvg svg")
      if scheme=="light":
        print("map note:",pg.inner_text("#mapnote"))
        print("stat:",pg.inner_text("#stat").replace("\n"," / "))
        print("method options:",pg.eval_on_selector_all("#method option","o=>o.map(x=>x.value)"))
        print("t6 note:",pg.inner_text("#t6note")[-140:])
        print("t6 rows:",pg.eval_on_selector_all("#t6tbl tr td:first-child","t=>t.map(x=>x.textContent)"))
        # click best cell
        L=m["labels"]; q=L.index(best[0]); k=L.index(best[1])
        cells=pg.locator("#mapsvg svg g g")
        n=cells.count()
        # find the cell whose tooltip names it: hover each until match is costly; use DOM order: iterate rows q then k, skipping blanks
        A=np.array(m["attn"]["all"]); order=[(qq,kk) for qq in range(len(L)) for kk in range(len(L)) if np.ptp(A[:,qq,kk])>=1e-9]
        idx=order.index((q,k)); cells.nth(idx).click(); pg.wait_for_timeout(300)
        print("after click, method:",pg.eval_on_selector("#method","e=>e.value"),"|",pg.inner_text("#stat").split("\n")[0])
        print("t5 rho title:",pg.eval_on_selector_all("#t5 text.ph","t=>t.map(x=>x.textContent)")[1])
        pg.click("#mapmode button[data-v=vw]"); pg.wait_for_timeout(300)
        print("vw note:",pg.inner_text("#mapnote")[:120], "| method name:", pg.eval_on_selector("#method option:checked","e=>e.textContent"))
        if task=="ioi":
          pg.click("#cond button[data-v=ABBA]"); pg.wait_for_timeout(300); print("ABBA map note:",pg.inner_text("#mapnote")[:150])
          pg.select_option("#ref","noise_abc_end"); pg.wait_for_timeout(300); print("final-pos note:",pg.inner_text("#mapnote"))
        pg.evaluate("document.getElementById('tip').style.opacity=0")
      pg.locator("#map").screenshot(path=f"{OUT}/map_{task}_{scheme}.png")
      pg.close()
  mm=b.new_page(viewport={"width":390,"height":900}); mm.route("**/cdn.jsdelivr.net/npm/d3@7",lambda r: r.fulfill(body=D3,content_type="application/javascript"))
  mm.goto(BASE+"?task=gt"); mm.wait_for_selector("#mapsvg svg"); print("mobile overflow:",mm.evaluate("document.documentElement.scrollWidth - innerWidth"))
  b.close()
print("console errors:",errs)
