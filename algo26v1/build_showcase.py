#!/usr/bin/env python
"""Inject showcase_data.json into showcase_template.html -> showcase_dashboard.html.

Mirrors dashboard.py: the template carries a `/*__DATA__*/null` placeholder that we
replace with the compact JSON payload, producing one self-contained offline page.

    python showcase_data.py       # (re)compute the backtests -> showcase_data.json
    python build_showcase.py      # embed into showcase_dashboard.html
"""
import json, os

here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "showcase_data.json")) as f:
    data = json.load(f)
with open(os.path.join(here, "showcase_template.html")) as f:
    tpl = f.read()

html = tpl.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
out = os.path.join(here, "showcase_dashboard.html")
with open(out, "w") as f:
    f.write(html)
print(f"wrote {out}  ({len(html)//1024} KB)")
print(f"open: file://{out}")
