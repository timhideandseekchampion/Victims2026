"""
test_v7cand_ew_idio.py

FOLLOW-UP to test_v7cand_double_ic_idio.py. There, the ALGO leg's EW-blend IC estimator
(`IC_BLEND` / `IC_EW_HL=(20,45)` / `IC_EW_W=200`: the mean of two exponentially-weighted ICs at
half-lives 20 and 45) was only ever used as a SECOND OPINION -- a confirmation that could remove
boosts but never add one. That test says nothing about whether the estimator itself is any good;
it only says a veto built from it doesn't pay.

THIS test asks the separate question: is exponential recency weighting -- specifically the 20/45
two-half-life blend the ALGO leg uses -- a BETTER PRIMARY estimator than the flat equal-weighted
windows the idio book uses everywhere? Unlike the veto, a replacement can go both ways: a pair whose
flat 250-day IC is <= 0 but whose recent EW IC is > 0 now gets boosted where v7 skips it.

Four placements -- every flat/equal-weighted estimator on the idio side:

  E1) BOOST IC GATE. v7: `ic = corr(lead_boost[t-250:t], follower_next_ret) > 0`, flat 250-day
      window, every sample weighted equally. Replace with the EW-blend estimator (mean of EW corrs
      at the two half-lives) over W samples. Leader SELECTION is untouched, so the candidate set is
      identical to v7 and the only thing that changes is which pairs pass the gate.
      Half-life pairs swept from the ALGO's own (20,45) up to (125,250), plus HL=87 as the
      centre-of-mass-matched control (a flat 250-window has COM 125 samples back; an EW weight with
      half-life h has COM ~ 1.44h, so h=87 is the same "average age of information" as v7's window,
      differing ONLY in the shape of the taper -- this isolates taper-shape from lookback-length).

  E2) LEADER SELECTION. v7 picks each follower's leader by the equal-weighted lag-1 correlation over
      the ENTIRE available history (`_corrmat(Xi_full, Yj)`), then tests it against a Bonferroni
      threshold at that full sample size. Replace with an EW-weighted correlation, with the
      significance threshold recomputed at the EFFECTIVE sample size n_eff = (sum w)^2 / sum(w^2) --
      so a shorter half-life is honestly penalised for the information it throws away rather than
      quietly clearing an unchanged bar. (The trailing-VOL ranking that defines the 39-name
      candidate pool is left flat: EWMA/short-window vol ranking is already tested-dead, README
      "Alternative boost-pool mechanics uniformly lose".)

  E3) RIDGE HALF-LIVES. The ridge is already exponentially weighted, but at HALF_LIVES =
      (250, 500, 1000, 2000) -- an order of magnitude slower than 20/45. Add / substitute the fast
      half-lives. (batch80 rejected extra half-lives at 100 and 4000; 20 and 45 were never tried --
      they are far outside the range anything has probed.)

  E4) REVERSAL BLEND. `REV_W=10` is a flat 10-day return. Replace with an EW-weighted average of
      past returns at half-lives 5/10/20/45 -- the same "taper instead of a hard window" change as
      E1, applied to the 30% reversal leg instead of the boost gate.

Baseline = SAFE_llboost_v7 (COMBINE_GAIN=16.0), same backtest-equivalent reconstruction and same
scoring convention as test_v7cand_double_ic_idio.py / validate_llboost_v7_full.py. The ALGO leg is
identical in every variant. Bar, per repo policy: beat v7 on OLD, NEW and rolling-mean JOINTLY.
"""
import numpy as np, pandas as pd, time
import SAFE_llboost_v7 as V7

P_ = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P_.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
NUMTEST = 250
logp = np.log(P_)
r = np.diff(logp, axis=1)
rs = r[1:]
nIdio = rs.shape[0]

BOOST_K, BOOST_MIN_DAY, BOOST_IC_L, WARMUP = V7.BOOST_K, V7.BOOST_MIN_DAY, V7.BOOST_IC_L, V7.WARMUP


def score(mu, sd):
    if mu <= 0 or sd < 1e-10: return float(mu)
    sr = np.sqrt(250) * mu / sd
    return float(mu * sr ** 2 / (sr ** 2 + 1.0))


def window(POS, S, E):
    curPos = np.zeros(nInst); comm_vec = np.zeros(nInst); prevCur = None; tot = []
    for tt in range(S, E + 1):
        cur = P_[:, tt - 1]
        newPos = (POS[:, tt - 1].copy() if tt < E else curPos.copy())
        if tt > S:
            tot.append(float((curPos * (cur - prevCur) - comm_vec).sum()))
        dP = newPos - curPos
        comm_vec = commRate * np.abs(dP) * cur
        prevCur = cur; curPos = newPos
    tot = np.array(tot)
    return float(score(tot.mean(), tot.std()))


end_days = list(range(400, nt + 1, 10))
OLD = (500, 750); NEW = (750, nt)
scs_curve = lambda POS: np.array([window(POS, E - NUMTEST, E) for E in end_days])


def ew_w(m, hl):
    return (0.5 ** (1.0 / hl)) ** np.arange(m - 1, -1, -1)


def wcorr(x, y, w, min_n=60):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < min_n: return np.nan
    x, y, w = x[ok], y[ok], w[ok]; sw = w.sum()
    mx = (w * x).sum() / sw; my = (w * y).sum() / sw
    vx = (w * (x - mx) ** 2).sum() / sw; vy = (w * (y - my) ** 2).sum() / sw
    if vx < 1e-24 or vy < 1e-24: return np.nan
    return float((w * (x - mx) * (y - my)).sum() / sw / np.sqrt(vx * vy))


def corrmat_w(X, Y, w):
    """EW-weighted analogue of V7._corrmat: rows of X vs rows of Y, weights w over the time axis."""
    sw = w.sum()
    mx = (X * w).sum(1, keepdims=True) / sw; my = (Y * w).sum(1, keepdims=True) / sw
    Xc = X - mx; Yc = Y - my
    vx = (w * Xc * Xc).sum(1, keepdims=True) / sw
    vy = (w * Yc * Yc).sum(1, keepdims=True) / sw
    C = (Xc * w) @ Yc.T / sw
    return C / (np.sqrt(vx) + 1e-24) / (np.sqrt(vy).T + 1e-24)


# ==================================================================================================
# shared precompute
# ==================================================================================================
def build_WZ(half_lives=V7.HALF_LIVES, rev_hl=None):
    """v7 ridge+blend forecast. rev_hl=None -> shipped flat REV_W=10 reversal; else EW reversal."""
    WZ = np.full((nIdio, nt), np.nan)
    for t in range(WARMUP, nt):
        rr = r[:, :t]
        fs = []
        for hl in half_lives:
            B, mx, my = V7._ewls_ridge(rr[:, :-1].T, rr[1:, 1:].T, hl, V7.RIDGE_A)
            pred = my + (rr[:, -1] - mx) @ B
            fi = pred - pred.mean()
            fs.append(fi / (fi.std() + 1e-12))
        wz = np.mean(fs, 0)
        if rev_hl is None:
            rr_ = logp[1:, t] - logp[1:, t - V7.REV_W]
        else:
            m = min(t, 6 * rev_hl)
            w = ew_w(m, rev_hl)
            rr_ = (rs[:, t - m:t] * w).sum(1) / w.sum() * m   # EW mean return, scaled to a total
        rr_ = rr_ - rr_.mean()
        WZ[:, t] = (1 - V7.BLEND) * wz + V7.BLEND * (-rr_ / (rr_.std() + 1e-12))
    return WZ


print("=== precompute: v7 WZ, ALGO leg ===", flush=True)
t0 = time.time()
WZ_v7 = build_WZ()
algo_pos = np.zeros(nt)
for k in range(130, nt):
    cur0 = P_[0, k]; lim0 = int(dlr[0] / cur0)
    algo_pos[k] = np.clip(V7._algo_vol_shares(logp[0, :k + 1], cur0, dlr[0]), -lim0, lim0)
print(f"  done ({time.time()-t0:.0f}s)", flush=True)


def build_pos(BOOST, WZ=None):
    WZ = WZ_v7 if WZ is None else WZ
    POS = np.zeros((nInst, nt))
    for k in range(WARMUP, nt):
        cur = P_[:, k]; lim = (dlr / cur).astype(int)
        wz = WZ[:, k].copy()
        if k >= BOOST_MIN_DAY:
            wz = wz + BOOST_K * BOOST[:, k]
        POS[1:, k] = np.clip(np.sign(wz) * (dlr[1:] / cur[1:]), -lim[1:], lim[1:])
    POS[0, :] = algo_pos
    return POS


results = []


def evaluate(nm, POS, base_scs, base, extra=""):
    wo = window(POS, *OLD); wn = window(POS, *NEW); scs = scs_curve(POS)
    line = (f"{nm:<40}OLD={wo:>7.1f}  NEW={wn:>7.1f}  rmean={scs.mean():>7.1f}  "
            f"rfloor={scs.min():>7.1f}  n_worse={int((scs<base_scs).sum())}/{len(scs)}")
    if extra: line += f"   {extra}"
    print(line, flush=True)
    results.append(dict(name=nm, wo=wo, wn=wn, rm=scs.mean(), rf=scs.min(),
                        passed=(wo > base[0]) and (wn > base[1]) and (scs.mean() > base[2])))
    return scs


# ==================================================================================================
# E1) EW-blend IC as the boost's PRIMARY gate (leader selection untouched)
# ==================================================================================================
EW_GATES = {"flat250 (v7)": None,
            "ew(20,45)/250": ((20, 45), 250),
            "ew(20,45)/500": ((20, 45), 500),
            "ew(45,90)/250": ((45, 90), 250),
            "ew(87)/250 [COM-matched]": ((87,), 250),
            "ew(87,174)/500": ((87, 174), 500),
            "ew(125,250)/500": ((125, 250), 500)}

print("\n=== precompute: boost candidates + every gate IC (leader selection = v7) ===", flush=True)
t0 = time.time()
BOOST_RAW = np.zeros((nIdio, nt))       # boost value for every follower with a SIGNIFICANT leader
GATE = {nm: np.full((nIdio, nt), np.nan) for nm in EW_GATES}
for k in range(BOOST_MIN_DAY, nt):
    rsl = rs[:, :k]; n, T = rsl.shape
    Xi_full = rsl[:, :-1]; Yj = rsl[:, 1:]
    thr = V7._sig_threshold(Xi_full.shape[1])
    cand_idx = np.argsort(-np.nanstd(Xi_full, axis=1))[:V7.BOOST_N_CANDIDATES]
    C = V7._corrmat(Xi_full[cand_idx], Yj)
    for j in range(n):
        col = C[:, j].copy()
        cp = np.where(cand_idx == j)[0]
        if len(cp): col[cp[0]] = np.nan
        if np.all(np.isnan(col)): continue
        ci = int(np.nanargmax(np.abs(col)))
        if abs(col[ci]) <= thr: continue
        lead = rsl[cand_idx[ci]]
        scale = np.nanstd(lead[max(0, T - 1 - V7.BOOST_SCALE_W):T - 1]) + 1e-12
        lb = np.sign(lead) * (np.abs(lead) / scale) ** V7.BOOST_P
        a = max(0, T - 1 - BOOST_IC_L)
        xs = lb[a:T - 1]; ys = rsl[j, a + 1:T]
        ok = ~np.isnan(xs) & ~np.isnan(ys)
        if ok.sum() < 60 or xs[ok].std() < 1e-12: continue
        BOOST_RAW[j, k] = lb[-1]
        GATE["flat250 (v7)"][j, k] = float(np.corrcoef(xs[ok], ys[ok])[0, 1])
        # the EW gates need the longest window any of them asks for
        aw = max(0, T - 1 - 500)
        xw = lb[aw:T - 1]; yw = rsl[j, aw + 1:T]
        for nm, spec in EW_GATES.items():
            if spec is None: continue
            hls, W = spec
            xv, yv = xw[-W:], yw[-W:]
            vals = [wcorr(xv, yv, ew_w(len(xv), hl)) for hl in hls]
            GATE[nm][j, k] = np.nan if any(not np.isfinite(v) for v in vals) else float(np.mean(vals))
print(f"  done ({time.time()-t0:.0f}s)", flush=True)

eligible = BOOST_RAW != 0.0
n_elig = eligible[:, BOOST_MIN_DAY:].sum()
v7_on = eligible & (GATE["flat250 (v7)"] > 0)
n_v7 = v7_on[:, BOOST_MIN_DAY:].sum()
print(f"  {n_elig} stock-days have a significant leader; v7's flat-250 gate passes {n_v7} "
      f"({100*n_v7/n_elig:.0f}%)")

print("\n=== sanity: backtest-equivalent v7 vs shipped README (830.3 / 888.5 / 876.8 / 674.4) ===")
POS_base = build_pos(np.where(v7_on, BOOST_RAW, 0.0))
base_scs = scs_curve(POS_base)
base = (window(POS_base, *OLD), window(POS_base, *NEW), base_scs.mean())
print(f"v7 baseline (backtest-equiv)             OLD={base[0]:>7.1f}  NEW={base[1]:>7.1f}  "
      f"rmean={base[2]:>7.1f}  rfloor={base_scs.min():>7.1f}")

print("\n### E1) EW-blend IC REPLACING the boost's flat 250-day gate ###")
for nm, spec in EW_GATES.items():
    if spec is None: continue
    on = eligible & (GATE[nm] > 0)
    added = int((on & ~v7_on)[:, BOOST_MIN_DAY:].sum())
    removed = int((v7_on & ~on)[:, BOOST_MIN_DAY:].sum())
    evaluate(f"E1 {nm}", build_pos(np.where(on, BOOST_RAW, 0.0)), base_scs, base,
             f"+{added} / -{removed} boosts vs v7")

# ==================================================================================================
# E2) EW-weighted LEADER SELECTION (threshold recomputed at the effective sample size)
# ==================================================================================================
print("\n### E2) EW-weighted leader-selection correlation (Bonferroni bar at n_eff) ###")


def boost_ew_selection(hl):
    B = np.zeros((nIdio, nt)); neff_seen = []
    for k in range(BOOST_MIN_DAY, nt):
        rsl = rs[:, :k]; n, T = rsl.shape
        Xi_full = rsl[:, :-1]; Yj = rsl[:, 1:]
        m = Xi_full.shape[1]
        w = ew_w(m, hl)
        neff = (w.sum() ** 2) / (w * w).sum()
        neff_seen.append(neff)
        thr = V7._sig_threshold(int(neff))
        cand_idx = np.argsort(-np.nanstd(Xi_full, axis=1))[:V7.BOOST_N_CANDIDATES]
        C = corrmat_w(Xi_full[cand_idx], Yj, w)
        for j in range(n):
            col = C[:, j].copy()
            cp = np.where(cand_idx == j)[0]
            if len(cp): col[cp[0]] = np.nan
            if np.all(np.isnan(col)): continue
            ci = int(np.nanargmax(np.abs(col)))
            if abs(col[ci]) <= thr: continue
            lead = rsl[cand_idx[ci]]
            scale = np.nanstd(lead[max(0, T - 1 - V7.BOOST_SCALE_W):T - 1]) + 1e-12
            lb = np.sign(lead) * (np.abs(lead) / scale) ** V7.BOOST_P
            a = max(0, T - 1 - BOOST_IC_L)
            xs = lb[a:T - 1]; ys = rsl[j, a + 1:T]
            ok = ~np.isnan(xs) & ~np.isnan(ys)
            if ok.sum() < 60 or xs[ok].std() < 1e-12: continue
            if float(np.corrcoef(xs[ok], ys[ok])[0, 1]) <= 0: continue
            B[j, k] = lb[-1]
    return B, float(np.mean(neff_seen))


for hl in (20, 45, 90, 250, 500):
    t0 = time.time()
    B, neff = boost_ew_selection(hl)
    on = int((B != 0.0)[:, BOOST_MIN_DAY:].sum())
    evaluate(f"E2 EW-select HL={hl}", build_pos(B), base_scs, base,
             f"n_eff~{neff:.0f} (v7 flat: {nt-BOOST_MIN_DAY+BOOST_MIN_DAY-2}), {on} boosts vs {n_v7}")

# ==================================================================================================
# E3) fast half-lives in the ridge ensemble
# ==================================================================================================
print("\n### E3) 20/45-day half-lives in the ridge ensemble (shipped: 250/500/1000/2000) ###")
for tag, hls in (("+45", V7.HALF_LIVES + (45,)),
                 ("+20,45", V7.HALF_LIVES + (20, 45)),
                 ("(20,45) only", (20, 45)),
                 ("(45,250,500,1000,2000)", (45,) + V7.HALF_LIVES)):
    WZ = build_WZ(half_lives=hls)
    evaluate(f"E3 ridge HL {tag}", build_pos(np.where(v7_on, BOOST_RAW, 0.0), WZ), base_scs, base)

# ==================================================================================================
# E4) EW reversal leg (shipped: flat 10-day return)
# ==================================================================================================
print("\n### E4) EW-tapered reversal leg (shipped: flat REV_W=10) ###")
for hl in (5, 10, 20, 45):
    WZ = build_WZ(rev_hl=hl)
    evaluate(f"E4 EW reversal HL={hl}", build_pos(np.where(v7_on, BOOST_RAW, 0.0), WZ), base_scs, base)

# ==================================================================================================
print("\n=== ranking: must beat v7 on OLD, NEW and rolling-mean JOINTLY ===")
print(f"baseline: OLD={base[0]:.1f} NEW={base[1]:.1f} rmean={base[2]:.1f} rfloor={base_scs.min():.1f}")
passing = [c for c in results if c["passed"]]
for c in passing:
    print(f"  PASS  {c['name']:<38} OLD={c['wo']:.1f} NEW={c['wn']:.1f} rmean={c['rm']:.1f} "
          f"rfloor={c['rf']:.1f}")
print(f"\n{len(passing)}/{len(results)} variants beat v7 on OLD+NEW+rmean jointly.")
print("\nTop 6 by rolling mean:")
for c in sorted(results, key=lambda c: -c["rm"])[:6]:
    print(f"  {c['name']:<38} OLD={c['wo']:>7.1f} NEW={c['wn']:>7.1f} rmean={c['rm']:>7.1f} "
          f"rfloor={c['rf']:>7.1f}")
