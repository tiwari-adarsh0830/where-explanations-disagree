"""Accessibility audit with axe-core (WCAG 2 A/AA rules), light and dark, both tasks, with a head selected."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import R, D3, OUT, CHROME, BASE

import json
from playwright.sync_api import sync_playwright
AXE_PATH=os.environ.get("AXE_JS", os.path.join(R, "tests", "node_modules", "axe-core", "axe.min.js"))
if not os.path.exists(AXE_PATH): print("skipped: axe-core not installed (cd tests && npm i axe-core@4)"); sys.exit(0)
AXE=open(AXE_PATH).read()
with sync_playwright() as p:
  b=p.chromium.launch(executable_path=CHROME)
  for url in ("?head=5.5","?task=gt&head=5.1"):
    for scheme in ("light","dark"):
      pg=b.new_page(viewport={"width":1320,"height":1000},color_scheme=scheme)
      pg.route("**/cdn.jsdelivr.net/npm/d3@7",lambda r: r.fulfill(body=D3,content_type="application/javascript"))
      pg.goto(BASE+url); pg.wait_for_selector("#t5 svg"); pg.wait_for_timeout(400)
      pg.evaluate("document.querySelectorAll('details').forEach(d=>d.open=true)"); pg.wait_for_timeout(300)
      pg.add_script_tag(content=AXE)
      r=pg.evaluate("axe.run(document,{runOnly:{type:'tag',values:['wcag2a','wcag2aa','wcag21aa']}}).then(r=>r.violations.map(v=>({id:v.id,impact:v.impact,n:v.nodes.length,help:v.help,eg:v.nodes.slice(0,4).map(n=>n.target.join(' ')+' :: '+(n.failureSummary||'').split('\\n').slice(1,2).join(''))})))")
      print(f"===== {url} {scheme}: {len(r)} rule(s) violated")
      for v in r: print(f"  [{v['impact']}] {v['id']} x{v['n']}: {v['help']}"); [print("     ",e[:230]) for e in v['eg']]
      pg.close()
  b.close()
