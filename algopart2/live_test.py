"""live_test.py — SAFE_live (submission build) must be POSITION-IDENTICAL to SAFE_rotate while
being fast enough to submit. Verifies:
 (1) TIMING     cold first call + incremental per-day cost, both builds (subprocess-isolated)
 (2) PARITY/real   day-by-day positions over t=141..750 (spans the readiness boundary at 143)
 (3) PARITY/regime day-by-day positions over the injected momentum panel 750..900 — the ONLY
     scenario where rotation, xsac fast-gate and (potentially) kill paths actually activate;
     equality here proves the lean cache changes no decision. Also reports LV regime PnL.
"""
import subprocess, sys, time
import numpy as np, pandas as pd

PY = sys.executable

# ---------- (1) timing, cold, in subprocesses ----------
SNIP = """
import time, numpy as np, pandas as pd
import {MOD} as M
prc = pd.read_csv('prices.txt', sep=r'\\s+', header=0).values.T
t0 = time.perf_counter(); M.getMyPosition(prc[:, :740]); t1 = time.perf_counter()
inc = []
for t in range(741, 751):
    s = time.perf_counter(); M.getMyPosition(prc[:, :t]); inc.append(time.perf_counter() - s)
print('{MOD}: cold first call (t=740) = %.2fs   incremental/day = %.0fms' % (t1 - t0, np.mean(inc) * 1000))
"""
print("(1) TIMING (cold subprocesses)")
for mod in ("SAFE_live", "SAFE_rotate"):
    out = subprocess.run([PY, "-c", SNIP.replace("{MOD}", mod)], capture_output=True, text=True, timeout=1200)
    print("   ", (out.stdout.strip() or out.stderr.strip().splitlines()[-1]))

# ---------- (2) parity on real data ----------
import SAFE_live as LV
import SAFE_rotate as R
prc = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nDays = prc.shape

print("\n(2) PARITY — real data, day-by-day t=141..750 (readiness boundary = "
      f"{LV.WARMUP + LV.ROT_W + LV.ROT_P})")
md = 0; bad = []
for t in range(141, nDays + 1):
    a = LV.getMyPosition(prc[:, :t]); b = R.getMyPosition(prc[:, :t])
    d = int(np.abs(a - b).max())
    if d > md: md = d
    if d: bad.append(t)
print(f"    max |SAFE_live - SAFE_rotate| = {md} shares over {nDays-140} days"
      + (f"   MISMATCH DAYS: {bad[:10]}" if bad else "   (identical)"))

# ---------- (3) parity on the injected momentum regime ----------
def make_ext(kind="momentum", T_ext=150, mom=0.6, seed=1):
    rng = np.random.default_rng(seed)
    logp = np.log(prc).copy(); vol = np.diff(logp[1:], axis=1).std()
    names = logp[1:, :].copy(); K = 5
    for _ in range(T_ext):
        trail = names[:, -1] - names[:, -K]; tc = trail - trail.mean()
        drift = mom * (tc / (tc.std() + 1e-9)) * vol; drift -= drift.mean()
        noise = rng.normal(0, vol, 50); noise -= noise.mean()
        names = np.concatenate([names, (names[:, -1] + drift + noise)[:, None]], axis=1)
    full = np.exp(np.concatenate([names.mean(0, keepdims=True), names], axis=0))
    full[:, :nDays] = prc
    return full

full = make_ext()
for M in (LV, R):
    M._SIG.clear(); M._RET.clear(); M._ICD.clear(); M._AZ.clear(); M._XC.clear()

commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
print("\n(3) PARITY — injected momentum regime, day-by-day t=750..900 (rotation/xsac ACTIVE)")
md = 0; bad = []; cash=0.0; cp=np.zeros(nInst); value=0.0; comm=0.0; pll=[]
rotated_days = 0
for t in range(nDays, nDays + 151):
    cur = full[:, t - 1]
    a = LV.getMyPosition(full[:, :t]) if t < nDays + 150 else cp
    b = R.getMyPosition(full[:, :t]) if t < nDays + 150 else cp
    d = int(np.abs(np.asarray(a) - np.asarray(b)).max())
    if d > md: md = d
    if d: bad.append(t)
    if t < nDays + 150 and LV._choose(t) != "champ": rotated_days += 1
    dP = np.asarray(a) - cp; cash -= cur.dot(dP) + comm
    comm = np.sum(cur * np.abs(dP) * commRate); cp = np.asarray(a, dtype=float)
    pl = cash + cp.dot(cur) - value; value = cash + cp.dot(cur)
    if t > nDays: pll.append(pl)
print(f"    max |SAFE_live - SAFE_rotate| = {md} shares over 150 regime days"
      + (f"   MISMATCH DAYS: {bad[:10]}" if bad else "   (identical)"))
print(f"    SAFE_live regime PnL = {sum(pll):,.0f}   (reference: 691,946)   rotated days: {rotated_days}")
