#!/usr/bin/env bash
# Headless checks of the page against the committed results.
# Needs: pip install playwright numpy scipy && python -m playwright install chromium
# Optional accessibility audit: (cd tests && npm i axe-core@4)
# Usage, from the repo root: bash tests/run_all.sh   (screenshots go to tests/out/)
set -u
cd "$(dirname "$0")/.."
python3 -m http.server 8767 >/dev/null 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null' EXIT
sleep 1
fail=0
for t in t3 t4 t5 t6 task sticky map v2 keys target svg pairs medium review axe; do
  out=$(python3 "tests/test_$t.py" 2>&1); code=$?
  bad=$(printf '%s\n' "$out" | grep -E "errors: \[.+\]|mismatches: [1-9]|overflow[^:]*: [1-9]|: [1-9][0-9]* rule\(s\) violated|Traceback")
  if [ $code -ne 0 ] || [ -n "$bad" ]; then
    echo "FAIL  $t"; printf '%s\n' "$out" | tail -15 | sed 's/^/      /'; fail=1
  else
    echo "ok    $t"
  fi
done
[ $fail -eq 0 ] && echo "all checks passed" || { echo "some checks failed"; exit 1; }
