"""Headless checks for the Nov 21-22 additions: intro, notes, tour, URL state, prompts card, matrix intervals, assumption cards."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import R, D3, OUT, CHROME, BASE

import json, numpy as np
from scipy.stats import spearmanr
from playwright.sync_api import sync_playwright
errs=[]
def page(b, url, w=1320, scheme="light"):
    pg=b.new_page(viewport={"width":w,"height":1000},color_scheme=scheme)
    pg.on("console",lambda m: m.type in("error","warning") and errs.append((url,m.text))); pg.on("pageerror",lambda e: errs.append((url,str(e))))
    pg.route("**/cdn.jsdelivr.net/npm/d3@7",lambda r: r.fulfill(body=D3,content_type="application/javascript"))
    pg.goto(url); pg.wait_for_selector("#t5 svg"); return pg
with sync_playwright() as p:
  b=p.chromium.launch(executable_path=CHROME)
  for task in ("ioi","gt"):
    url=BASE+("?task=gt" if task=="gt" else "")
    pg=page(b,url)
    print(f"===== {task}")
    print("intro findings:", pg.eval_on_selector_all("#introfind li","l=>l.map(x=>x.innerText.slice(0,160))"))
    print("notes on cards:", pg.eval_on_selector_all("details.about","d=>d.length"), "| cards:", pg.eval_on_selector_all("section.card[id]","s=>s.filter(x=>!x.hidden).map(x=>x.id)"))
    # tour
    pg.click("#tourbtn"); n=0
    while True:
        pg.wait_for_timeout(500)
        st=pg.inner_text("#tstep"); tt=pg.inner_text("#ttitle"); n+=1
        print(f"  tour {st}: {tt} | url {pg.url.split('frontend/')[1]} | method {pg.eval_on_selector('#method','e=>e.value')} | ref {pg.eval_on_selector('#ref','e=>e.value')}")
        if pg.inner_text("#tnext")=="Finish": pg.click("#tnext"); break
        pg.click("#tnext")
    print("  tour hidden after finish:", pg.eval_on_selector("#tour","e=>e.hidden"))
    # URL state round trip
    pg.select_option("#method","dla"); pg.wait_for_timeout(200)
    shared=pg.url; print("shared url:", shared.split("frontend/")[1])
    pg2=page(b,shared); pg2.wait_for_timeout(300)
    print("  restored: method", pg2.eval_on_selector("#method","e=>e.value"), "| ref", pg2.eval_on_selector("#ref","e=>e.value"), "| head title:", pg2.inner_text("#dtitle"), "| stat:", pg2.inner_text("#stat").split("\n")[0][:70])
    pg2.close()
    # prompts card
    print("pstat:", pg.inner_text("#pstat")); print("hstat:", pg.inner_text("#hstat"))
    pg.locator("#pstrip circle").last.click(force=True); pg.wait_for_timeout(200); print("pdetail:", pg.inner_text("#pdetail")[:200].replace("\n"," / "))
    # matrix interval tooltip
    pg.locator("#t6 svg g rect").nth(3).hover(force=True); pg.wait_for_timeout(300); print("t6 tooltip:", pg.inner_text("#tip").replace("\n"," / ")[:200])
    pg.evaluate("document.getElementById('tip').style.opacity=0")
    pg.locator("#intro-card").screenshot(path=f"{OUT}/v2_{task}_intro.png")
    pg.locator("#prompts").screenshot(path=f"{OUT}/v2_{task}_prompts.png")
    pg.close()
  # dark + mobile
  pg=page(b,BASE+"?head=5.5&ref=denoise_abc",scheme="dark"); pg.wait_for_timeout(300)
  print("dark deep link: ref", pg.eval_on_selector("#ref","e=>e.value"), "| why rows", pg.eval_on_selector_all("#whybody tr","r=>r.length"))
  pg.locator("#why").screenshot(path=f"{OUT}/v2_why_dark.png"); pg.close()
  m=page(b,BASE,w=390); print("mobile overflow:", m.evaluate("document.documentElement.scrollWidth - innerWidth"))
  m.locator("#intro-card").screenshot(path=f"{OUT}/v2_intro_mobile.png"); m.close()
  b.close()
print("console errors:", errs)
