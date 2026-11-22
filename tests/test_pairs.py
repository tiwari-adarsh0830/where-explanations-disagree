"""Pair interaction labels depend on the signs of the two single effects; checked for every measured pair against Python.
Also checks that ?pair=a,b pins a pair, that #card opens the page at that card, and that the tour's interaction step pins its pair."""
import os, sys, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import R, D3, OUT, CHROME, BASE
from playwright.sync_api import sync_playwright

def kind(sa, sb, v):
    S, T = sa + sb, sa + sb + v
    if abs(v) < 0.1 * max(abs(sa) + abs(sb), 1e-9): return "none"
    if sa * sb > 0: return "backup" if abs(T) > abs(S) else "path"
    return "dep"

errs, bad = [], 0
with sync_playwright() as p:
    b = p.chromium.launch(executable_path=CHROME)
    for task, summ, url in (("ioi", "results/diagnostics_summary.json", "?head=9.9&pair=9.9,10.7#interact"),
                            ("gt", "results/gt/diagnostics_summary.json", "?task=gt&head=9.1&pair=7.10,9.1#interact")):
        G = json.load(open(os.path.join(R, summ)))
        heads, M, eff = G["top_heads"], G["interaction"], G["effects"]["full"]
        idx = lambda h: int(h.split(".")[0]) * 12 + int(h.split(".")[1])
        expect = {}
        for a in range(len(heads)):
            for c in range(a + 1, len(heads)):
                if M[a][c] is not None: expect[(heads[a], heads[c])] = (eff[idx(heads[a])], eff[idx(heads[c])], M[a][c])
        pg = b.new_page(viewport={"width": 1320, "height": 900})
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.route("**/cdn.jsdelivr.net/npm/d3@7", lambda r: r.fulfill(body=D3, content_type="application/javascript"))
        pg.goto(BASE + url); pg.wait_for_selector("#t5 svg"); pg.wait_for_timeout(500)
        got = pg.evaluate("pairs => pairs.map(([sa, sb, v]) => pairKind(sa, sb, v).kind)", list(expect.values()))
        want = [kind(*t) for t in expect.values()]
        mism = [(k, g, w) for k, g, w in zip(expect, got, want) if g != w]
        counts = {k: want.count(k) for k in ("backup", "path", "dep", "none")}
        print(f"== {task}: {len(want)} pairs, labels {counts}, mismatches with Python: {len(mism)} {mism[:3]}")
        bad += len(mism)
        pinned = pg.inner_text("#intpair"); top = pg.evaluate("document.getElementById('interact').getBoundingClientRect().top")
        print(f"   pinned from URL: {pinned[:120]}...")
        print(f"   opened at the card: top {top:.0f} px; URL keeps pair and hash: {pg.evaluate('location.search + location.hash')}")
        bad += ("Pinned pair" not in pinned) + (not (0 <= top < 200)) + ("#interact" not in pg.evaluate("location.hash"))
        if task == "ioi":
            for s_ in ("cover for each other\" : \"same pathway\"",):
                pass
            # 9.9 + 10.7: one helps, one hurts, so no backup or pathway label
            if "covers for" in pinned or "same pathway" in pinned: print("   PROBLEM: mixed-sign pair got a backup/pathway label"); bad += 1
            pg.click("#unpin"); pg.wait_for_timeout(100)
            print("   after Unpin, URL:", pg.evaluate("location.search + location.hash")); bad += "pair=" in pg.evaluate("location.search")
            # tour step that pins 9.9 + 10.7
            pg.click("#tourbtn"); n = pg.evaluate("TOURS.ioi.length")
            for _ in range(n - 1): pg.click("#tnext"); pg.wait_for_timeout(80)
            pg.wait_for_timeout(400); t = pg.inner_text("#intpair")
            print("   last tour step pins:", t[:60]); bad += not t.startswith("Pinned pair 9.9 + 10.7")
        pg.close()
    b.close()
print(f"mismatches: {bad}")
print("console errors:", errs)
