"""Shared settings for the headless page tests.

Run from the repo root with tests/run_all.sh, which starts a local server on port 8767.
D3: the tests serve tests/d3.min.js (d3 v7, ISC license) in place of the CDN copy so they run offline.
CHROME: optional path to a Chromium binary; leave unset to use Playwright's own browser.
"""
import os
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D3 = open(os.path.join(R, "tests", "d3.min.js")).read()
OUT = os.environ.get("TEST_OUT", os.path.join(R, "tests", "out")); os.makedirs(OUT, exist_ok=True)
CHROME = os.environ.get("CHROME") or None
BASE = os.environ.get("BASE_URL", "http://localhost:8767/frontend/")
