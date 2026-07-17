#!/usr/bin/env bash
# Regenerate the offline research dashboard from prices.txt:
#   1. re-export every scorecard strategy's per-day positions -> books.json
#   2. rebuild dashboard.html (all 7 books' entries/exits + Compare tab)
# Run from anywhere:  ./refresh_dashboard.sh
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
[ -x "$PY" ] || PY=python            # fall back to whatever python is on PATH
export PYTHONPATH=.

echo "[1/2] exporting strategy books (the Markov leg fits an MLE per day, ~2 min)..."
"$PY" export_books.py

echo "[2/2] building dashboard.html..."
RUNS_ARG=""
[ -f runs.json ] && RUNS_ARG="--runs runs.json"
"$PY" dashboard.py --books books.json $RUNS_ARG

echo
echo "done -> open dashboard.html in a browser:"
echo "   explorer.exe dashboard.html     # from a WSL terminal"
