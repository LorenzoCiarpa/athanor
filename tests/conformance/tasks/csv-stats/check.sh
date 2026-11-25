#!/bin/sh
# stats.py exists, and result.txt holds the mean score to two decimals.
test -f stats.py || { echo "no stats.py"; exit 1; }
test -f result.txt || { echo "no result.txt"; exit 1; }
python3 - <<'PY' || exit 1
import csv, sys
rows = list(csv.DictReader(open("data.csv")))
want = f"{sum(float(r['score']) for r in rows) / len(rows):.2f}"
got = open("result.txt").read().strip()
if got != want:
    sys.exit(f"result.txt holds {got!r}; the mean is {want}")
PY
