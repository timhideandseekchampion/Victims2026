"""
================================================================================
### H-candidate v7: is the pairwise boost's TREND (recent vs older |corr|)  ###
###                  a useful causal confidence multiplier on BOOST_K?      ###
================================================================================
Structurally DIFFERENT from H3 (test_h3_leader_stability.py / test_h3_stage2_backtest.py):
H3 asks whether the argmax leader IDENTITY has stayed the same for many consecutive days
("stability" = persistence of *which* stock is the leader). This file does not use that
counter at all. Instead, for a pair that ALREADY qualifies under v6's own unchanged
significance test (expanding-window |corr|, Bonferroni-corrected, min-history=480), it asks
whether that SAME pair's lag-1 |corr| MAGNITUDE has been trending stronger or weaker lately --
comparing a trailing rolling-window |corr| estimate against an older rolling-window |corr|
estimate ending strictly before it. Rationale: an expanding-window correlation is slow to
reflect recent decay -- a pair that cleared the bar mostly on old history but has been
weakening recently should be trusted less than one that is currently strengthening, even
though both clear today's expanding-window bar identically.

Step 1 (qualifying set -- UNCHANGED): reuses SAFE_llboost_v6's own constants and exact
_pairwise_boost logic (imported directly, sanity-checked below to reproduce its boost values
bit-for-bit) to decide WHICH (leader i, follower j) pairs qualify each day and what the raw
boost value is. This file adds nothing to that gate and removes nothing from it.

Step 2 (trend signal -- NEW): for each qualifying (i, j) pair on day k, using the exact same
lag-1 alignment already used by the significance test (a = rs[i, :T-1], b = rs[j, 1:T], causal
by construction since these are the very arrays v6's own gate already relies on), take
  recent  = |corr(a[-Lr:],        b[-Lr:])|              (trailing Lr-day window)
  older   = |corr(a[-(Lr+Lo):-Lr], b[-(Lr+Lo):-Lr])|      (the Lo-day window immediately BEFORE
                                                            the recent window -- non-overlapping,
                                                            fully in the past relative to "recent")
If there isn't enough history yet for both windows (T-1 < Lr+Lo, or either window's std is ~0),
the multiplier falls back to 1.0 (full trust, i.e. identical to shipped v6) -- this only ever
activates once there is enough data to actually judge a trend, never fabricates one.

Two multiplier designs swept, each only ever <= 1.0 (this variant never boosts a pair MORE than
v6 already does -- only tempers it when recent evidence looks weaker than older evidence):
  RATIO  : mult = clip(recent / (older + eps), FLOOR, 1.0)
  GATE   : mult = 1.0 if recent >= older else GATE_VAL      (hard step at the strengthen/weaken line)

Score convention identical to validate_llboost_v6_full.py: window(POS,S,E), commRate=1e-4
(inst0=2e-5), dlr=10_000 (inst0=100_000), score=mu*sr^2/(sr^2+1), OLD=(500,750), NEW=(750,nt),
rolling mean/floor over end_days=range(400,nt+1,10) (61 windows), n_worse vs the ACTUAL shipped
SAFE_llboost_v6.getMyPosition (built directly, not approximated).
================================================================================
"""
import time
import numpy as np
import pandas as pd

import SAFE_llboost_v6 as V6

P = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
NUMTEST = 250


def score(mu, sd):
    if mu <= 0 or sd < 1e-10: return float(mu)
    sr = np.sqrt(250) * mu / sd
    return float(mu * sr ** 2 / (sr ** 2 + 1.0))


def window(POS, S, E):
    curPos = np.zeros(nInst); comm_vec = np.zeros(nInst); prevCur = None; tot = []
    for tt in range(S, E + 1):
        cur = P[:, tt - 1]
        newPos = (POS[:, tt - 1].copy() if tt < E else curPos.copy())
        if tt > S:
            pl = curPos * (cur - prevCur) - comm_vec
            tot.append(float(pl.sum()))
        dP = newPos - curPos
        comm_vec = commRate * np.abs(dP) * cur
        prevCur = cur; curPos = newPos
    tot = np.array(tot)
    return float(score(tot.mean(), tot.std()))


end_days = list(range(400, nt + 1, 10))
OLD = (500, 750); NEW = (750, nt)

logp = np.log(P)
r = np.diff(logp, axis=1)          # (nInst, nt-1)
rs_full = r[1:]                    # (n_idio, nt-1) idio returns, ALGO excluded
n_idio, Tmax = rs_full.shape

# ================================================================================
# Step 0: baseline = REAL SAFE_llboost_v6.getMyPosition (not an approximation)
# ================================================================================
FIRST_DAY = 148  # matches validate_llboost_v6_full.py: covers every rolling window


def build_pos_module(mod, first_day):
    POS = np.zeros((nInst, nt))
    for k in range(first_day, nt):
        POS[:, k] = mod.getMyPosition(P[:, :k + 1])
    return POS


print("=== building real SAFE_llboost_v6 baseline positions (getMyPosition, unmodified) ===")
t0 = time.time()
POS_v6 = build_pos_module(V6, FIRST_DAY)
print(f"  done in {time.time()-t0:.0f}s")


def report(nm, POS, base_scs=None):
    wo = window(POS, *OLD); wn = window(POS, *NEW)
    scs = np.array([window(POS, E - NUMTEST, E) for E in end_days])
    line = f"{nm:<34}OLD={wo:>7.1f}  NEW={wn:>7.1f}  rmean={scs.mean():>7.1f}  rfloor={scs.min():>7.1f}"
    if base_scs is not None:
        nworse = int((scs < base_scs).sum())
        line += f"  n_worse={nworse}/{len(scs)}"
    print(line)
    return scs, wo, wn


base_scs, base_old, base_new = report("SAFE_llboost_v6 (shipped, baseline)", POS_v6)

# ================================================================================
# Step 1: precompute idio ridge+blend WZ (no boost) and ALGO leg -- IDENTICAL to v6,
# reusing v6's own functions/constants so there is zero risk of drift from the shipped file.
# ================================================================================
print("\n=== precomputing idio WZ (ridge+blend) + ALGO leg (v6's own functions, unchanged) ===")
t0 = time.time()
WZ = {}
for k in range(V6.WARMUP, nt):
    rr = r[:, :k]
    fs = []
    for hl in V6.HALF_LIVES:
        B, mx, my = V6._ewls_ridge(rr[:, :-1].T, rr[1:, 1:].T, hl, V6.RIDGE_A)
        pred = my + (rr[:, -1] - mx) @ B
        fi = pred - pred.mean()
        fs.append(fi / (fi.std() + 1e-12))
    wz = np.mean(fs, 0)
    if V6.BLEND > 0:
        rv_ = logp[1:, k] - logp[1:, k - V6.REV_W]
        rv_ = rv_ - rv_.mean()
        rv = -rv_ / (rv_.std() + 1e-12)
        wz = (1 - V6.BLEND) * wz + V6.BLEND * rv
    WZ[k] = wz

algo_pos = np.zeros(nt)
for k in range(V6.WARMUP, nt):
    cur0 = P[0, k]; lim0 = int(dlr[0] / cur0)
    algo_pos[k] = np.clip(V6._algo_vol_shares(logp[0, :k + 1], cur0, dlr[0]), -lim0, lim0)
print(f"  done in {time.time()-t0:.0f}s")

# ================================================================================
# Step 2: precompute LEADER_AT[k] = {j: (i, bv)} -- v6's exact qualifying-pair logic
# (sanity-checked below to reproduce V6._pairwise_boost's raw boost values exactly).
# ================================================================================
print("\n=== precomputing v6's exact qualifying-pair map (leader identity + raw boost value) ===")
t0 = time.time()


def leader_map_at(k):
    T = k
    rs_k = rs_full[:, :T]
    Xi_full = rs_k[:, :-1]; Yj = rs_k[:, 1:]
    n_samples = Xi_full.shape[1]
    thr = V6._sig_threshold(n_samples)
    vol_causal = np.nanstd(Xi_full, axis=1)
    cand_idx = np.argsort(-vol_causal)[:V6.BOOST_N_CANDIDATES]
    Xi = Xi_full[cand_idx]
    C = V6._corrmat(Xi, Yj)
    out = {}
    for j in range(n_idio):
        col = C[:, j].copy()
        cand_pos = np.where(cand_idx == j)[0]
        if len(cand_pos):
            col[cand_pos[0]] = np.nan
        if np.all(np.isnan(col)):
            continue
        ci = int(np.nanargmax(np.abs(col)))
        if abs(col[ci]) <= thr:
            continue
        i = cand_idx[ci]
        lead = rs_k[i]
        scale = np.nanstd(lead[max(0, T - 1 - V6.BOOST_SCALE_W):T - 1]) + 1e-12
        lead_boost = np.sign(lead) * (np.abs(lead) / scale) ** V6.BOOST_P
        a = max(0, T - 1 - V6.BOOST_IC_L)
        xs = lead_boost[a:T - 1]; ys = rs_k[j, a + 1:T]
        ok = ~np.isnan(xs) & ~np.isnan(ys)
        if ok.sum() < 60 or xs[ok].std() < 1e-12:
            continue
        ic = float(np.corrcoef(xs[ok], ys[ok])[0, 1])
        if ic <= 0:
            continue
        out[j] = (i, float(lead_boost[-1]))
    return out


LEADER_AT = {k: leader_map_at(k) for k in range(V6.BOOST_MIN_DAY, nt)}
n_records = sum(len(v) for v in LEADER_AT.values())
print(f"  done in {time.time()-t0:.0f}s; {n_records} qualifying (day, follower) instances")

print("  sanity check vs V6._pairwise_boost at 5 sampled days ...")
for k in [V6.BOOST_MIN_DAY, 500, 600, 800, nt - 1]:
    boost_v6 = V6._pairwise_boost(rs_full[:, :k])
    recon = np.zeros(n_idio)
    for j, (i, bv) in LEADER_AT[k].items():
        recon[j] = bv
    ok = np.allclose(boost_v6, recon)
    print(f"    day {k}: exact match = {ok}  (n qualify = {len(LEADER_AT[k])})")

# ================================================================================
# Step 2b: sanity check -- reconstruction with trend multiplier forced to 1.0 everywhere
# must reproduce the real v6 baseline exactly (proves the WZ/algo/leader-map plumbing is
# correct before trusting any deltas from the trend multiplier below).
# ================================================================================


def build_pos_trend(mult_fn):
    """mult_fn(k, j, i) -> float multiplier in [0,1] applied to K*bv for that pair/day."""
    POS = np.zeros((nInst, nt))
    for k in range(V6.WARMUP, nt):
        cur = P[:, k]; lim = (dlr / cur).astype(int)
        wz = WZ[k].copy()
        if k >= V6.BOOST_MIN_DAY:
            for j, (i, bv) in LEADER_AT[k].items():
                m = mult_fn(k, j, i)
                wz[j] += V6.BOOST_K * m * bv
        POS[1:, k] = np.clip(np.sign(wz) * (dlr[1:] / cur[1:]), -lim[1:], lim[1:])
    POS[0, :] = algo_pos
    return POS


print("\n=== sanity check: reconstruction (mult==1 always) vs real v6 baseline ===")
POS_recon = build_pos_trend(lambda k, j, i: 1.0)
print("  idio+algo positions identical (days >= FIRST_DAY):",
      np.allclose(POS_recon[:, FIRST_DAY:], POS_v6[:, FIRST_DAY:]))
report("reconstruction (mult=1, sanity)", POS_recon, base_scs)

# ================================================================================
# Step 3: trend signal -- recent-vs-older rolling |corr| on the SAME (i,j) alignment
# already used by v6's own significance test. Precompute per (Lr, Lo) window combo once,
# reused across every design/threshold swept on top of it (cheap).
# ================================================================================
print("\n=== precomputing recent-vs-older |corr| trend for each qualifying (k,j) pair ===")
t0 = time.time()

TREND_CACHE = {}


def trend_table(Lr, Lo):
    if (Lr, Lo) in TREND_CACHE:
        return TREND_CACHE[(Lr, Lo)]
    TR = {}
    for k, entries in LEADER_AT.items():
        T = k
        if not entries:
            continue
        row = {}
        for j, (i, bv) in entries.items():
            a_full = rs_full[i, :T - 1]
            b_full = rs_full[j, 1:T]
            n_av = len(a_full)
            if n_av < Lr + Lo:
                row[j] = None  # not enough history yet -> caller falls back to mult=1.0
                continue
            ra, rb = a_full[-Lr:], b_full[-Lr:]
            oa, ob = a_full[-(Lr + Lo):-Lr], b_full[-(Lr + Lo):-Lr]
            if ra.std() < 1e-12 or rb.std() < 1e-12 or oa.std() < 1e-12 or ob.std() < 1e-12:
                row[j] = None
                continue
            recent = abs(float(np.corrcoef(ra, rb)[0, 1]))
            older = abs(float(np.corrcoef(oa, ob)[0, 1]))
            row[j] = (recent, older)
        TR[k] = row
    TREND_CACHE[(Lr, Lo)] = TR
    return TR


WINDOW_COMBOS = [(150, 150), (200, 200), (200, 300), (250, 250)]
for Lr, Lo in WINDOW_COMBOS:
    trend_table(Lr, Lo)
print(f"  done in {time.time()-t0:.0f}s for {len(WINDOW_COMBOS)} window combos")


def make_ratio_mult(Lr, Lo, floor):
    TR = trend_table(Lr, Lo)

    def mult_fn(k, j, i):
        row = TR.get(k)
        rec = row.get(j) if row else None
        if rec is None:
            return 1.0
        recent, older = rec
        return float(np.clip(recent / (older + 1e-6), floor, 1.0))
    return mult_fn


def make_gate_mult(Lr, Lo, gate_val):
    TR = trend_table(Lr, Lo)

    def mult_fn(k, j, i):
        row = TR.get(k)
        rec = row.get(j) if row else None
        if rec is None:
            return 1.0
        recent, older = rec
        return 1.0 if recent >= older else gate_val
    return mult_fn


# ================================================================================
# Step 4: sweep designs
# ================================================================================
print("\n=== RATIO design: mult = clip(recent/(older+eps), FLOOR, 1.0) ===")
results = {}
for Lr, Lo in WINDOW_COMBOS:
    for floor in (0.0, 0.3, 0.5, 0.7):
        nm = f"ratio Lr={Lr} Lo={Lo} floor={floor}"
        POS = build_pos_trend(make_ratio_mult(Lr, Lo, floor))
        scs, wo, wn = report(nm, POS, base_scs)
        results[nm] = (wo, wn, scs.mean(), scs.min(), int((scs < base_scs).sum()))

print("\n=== GATE design: mult = 1.0 if recent>=older else GATE_VAL ===")
for Lr, Lo in WINDOW_COMBOS:
    for gate_val in (0.0, 0.3, 0.5):
        nm = f"gate  Lr={Lr} Lo={Lo} gate={gate_val}"
        POS = build_pos_trend(make_gate_mult(Lr, Lo, gate_val))
        scs, wo, wn = report(nm, POS, base_scs)
        results[nm] = (wo, wn, scs.mean(), scs.min(), int((scs < base_scs).sum()))

# ================================================================================
# Step 5: identify any variant beating v6 on OLD + NEW + rolling-mean simultaneously
# ================================================================================
print("\n=== candidates beating shipped v6 on OLD AND NEW AND rolling-mean simultaneously ===")
winners = [(nm, v) for nm, v in results.items()
           if v[0] > base_old and v[1] > base_new and v[2] > base_scs.mean()]
if not winners:
    print("  none -- no variant clears all three headline metrics at once.")
else:
    for nm, v in winners:
        print(f"  {nm}: OLD={v[0]:.1f} NEW={v[1]:.1f} rmean={v[2]:.1f} rfloor={v[3]:.1f} n_worse={v[4]}/61")

    print("\n=== neighbor-stability check around each winner ===")
    for nm, v in winners:
        # parse out Lr, Lo, and the design param from the name; re-sweep small neighborhood
        print(f"  -- neighbors of: {nm}")
        if nm.startswith("ratio"):
            parts = nm.split()
            Lr0 = int(parts[1].split("=")[1]); Lo0 = int(parts[2].split("=")[1])
            floor0 = float(parts[3].split("=")[1])
            for dLr in (-50, 0, 50):
                for dLo in (-50, 0, 50):
                    for dF in (-0.2, 0.0, 0.2):
                        Lr, Lo, floor = Lr0 + dLr, Lo0 + dLo, float(np.clip(floor0 + dF, 0.0, 1.0))
                        if Lr < 50 or Lo < 50:
                            continue
                        POS = build_pos_trend(make_ratio_mult(Lr, Lo, floor))
                        scs, wo, wn = report(f"    nbr ratio Lr={Lr} Lo={Lo} floor={floor:.1f}", POS, base_scs)
        else:
            parts = nm.split()
            Lr0 = int(parts[1].split("=")[1]); Lo0 = int(parts[2].split("=")[1])
            gate0 = float(parts[3].split("=")[1])
            for dLr in (-50, 0, 50):
                for dLo in (-50, 0, 50):
                    for dG in (-0.2, 0.0, 0.2):
                        Lr, Lo, gate = Lr0 + dLr, Lo0 + dLo, float(np.clip(gate0 + dG, 0.0, 1.0))
                        if Lr < 50 or Lo < 50:
                            continue
                        POS = build_pos_trend(make_gate_mult(Lr, Lo, gate))
                        scs, wo, wn = report(f"    nbr gate  Lr={Lr} Lo={Lo} gate={gate:.1f}", POS, base_scs)

print("\n=== done ===")
