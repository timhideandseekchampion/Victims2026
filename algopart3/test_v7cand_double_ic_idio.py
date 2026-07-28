"""
test_v7cand_double_ic_idio.py

IDEA UNDER TEST (user's question): SAFE_llboost_v7's ALGO leg decides whether to trade the index at
all with a DOUBLE IC test -- `_side()` takes the sign from a fast 90-day simple IC (`IC_FAST`) and
then REFUSES TO TRADE (returns 0.0) unless a structurally different second estimator of the same IC
-- the mean of two exponentially-weighted ICs at half-lives (20, 45) over a 200-day window
(`IC_BLEND`/`IC_EW_HL`/`IC_EW_W`) -- AGREES ON THE SIGN. Two estimators, one veto. That gate is a
validated part of the shipped book. Would porting the same two-estimator-agreement veto to the
IDIOSYNCRATIC stat-arb side improve it?

This is NOT any of the previously-rejected trailing-IC ideas -- all of those used ONE estimator:
  - test_v7cand_adaptive_boostk.py: pooled trailing IC as a MAGNITUDE multiplier on BOOST_K.
  - test_gated_pair_boost.py:       a single per-pair trailing-IC on/off gate.
  - test_partial_pooling_boost.py:  shrink a pair's own trailing IC toward a population mean.
  - test_h4_margin_scaled_final.py: |corr|/threshold margin -- structural, not realized.
The distinctive content of `_side` is DISAGREEMENT BETWEEN TWO DIFFERENT ESTIMATORS of the same
quantity as a stand-down signal -- an estimator-instability veto, not a level or a strength dial.
Nothing on the idio side has ever been gated that way.

THREE PLACEMENTS, all fully causal, all scored on the same harness as every other test in this file:

  A) PAIR level (closest structural analogue). The pairwise boost already has exactly ONE IC gate:
     `ic = corr(lead_boost[t-IC_L:t], follower[t-IC_L+1:t+1]) > 0` at BOOST_IC_L=250. Add the second
     estimator on the SAME (x=lead_boost, y=follower-next-return) series and require agreement:
       A-fast(L) : also require a simple IC over the last L samples to be > 0
       A-ew(HL,W): also require the mean of EW ICs at half-lives HL over the last W samples to be > 0
                   -- (20,45)/200 is the EXACT ALGO port (same estimator `_side` uses)
       A-both    : require both of the above
     Strictly a tightening: it can only remove boosts, never add one.

  B) PER-STOCK level (the ALGO leg's own structure, applied per idio name). ALGO measures the
     trailing IC of ITS OWN feature against ITS OWN next-day return; the idio analogue is the
     trailing IC of stock j's traded score wz[j] against stock j's own next-day return.
       B-veto : if the fast simple IC and the EW-blend IC disagree in sign -> flatten stock j today
                (the literal `_side` port: disagreement -> 0.0)
       B-flip : as B-veto, but additionally take the SIGN from the fast IC (the full `_side` port --
                sf * feature, where a negative trailing IC inverts the name)
       B-pos  : require BOTH estimators > 0 (strict directional version -- a signal that has stopped
                working on BOTH clocks stands down)

  C) BOOK level. Same rule, but on the IC pooled across all 50 idio names; disagreement flattens the
     whole idio book for the day (the direct "should we trade this book at all today" analogue of
     "should we trade ALGO at all today").

Baseline = SAFE_llboost_v7 (COMBINE_GAIN=16.0), reconstructed backtest-equivalently (WZ +
_pairwise_boost + _algo_vol_shares, all imported from V7 verbatim) and sanity-checked against the
shipped README numbers (OLD=830.3 NEW=888.5 rmean=876.8 rfloor=674.4). The ALGO leg is IDENTICAL in
every variant -- only the idio book changes -- so any score difference is attributable to the idio
side alone. Scoring convention matches validate_llboost_v7_full.py: window(POS,S,E), commRate=1e-4
(inst0=2e-5), dlr=10_000 (inst0=100_000), score=mu*sr^2/(sr^2+1), OLD=(500,750), NEW=(750,nt),
rolling mean/floor over end_days=range(400,nt+1,10) (61 windows), n_worse counted against baseline.

Per repo policy ([[ic-vs-score-lesson]]): a variant is only interesting if it beats the baseline on
OLD, NEW and rolling-mean JOINTLY -- never on IC or on one window alone.
"""
import numpy as np, pandas as pd, time
import SAFE_llboost_v7 as V7

P_ = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P_.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
NUMTEST = 250
logp = np.log(P_)
r = np.diff(logp, axis=1)           # (nInst, nt-1)
rs = r[1:]                          # (50, nt-1) idio-only, matches V7._pairwise_boost's input
nIdio = rs.shape[0]

BOOST_K = V7.BOOST_K
BOOST_MIN_DAY = V7.BOOST_MIN_DAY
BOOST_IC_L = V7.BOOST_IC_L
WARMUP = V7.WARMUP


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
            pl = curPos * (cur - prevCur) - comm_vec
            tot.append(float(pl.sum()))
        dP = newPos - curPos
        comm_vec = commRate * np.abs(dP) * cur
        prevCur = cur; curPos = newPos
    tot = np.array(tot)
    return float(score(tot.mean(), tot.std()))


end_days = list(range(400, nt + 1, 10))
OLD = (500, 750); NEW = (750, nt)


def scs_curve(POS):
    return np.array([window(POS, E - NUMTEST, E) for E in end_days])


def report(nm, POS, base_scs=None, extra=""):
    wo = window(POS, *OLD); wn = window(POS, *NEW)
    scs = scs_curve(POS)
    line = f"{nm:<44}OLD={wo:>7.1f}  NEW={wn:>7.1f}  rmean={scs.mean():>7.1f}  rfloor={scs.min():>7.1f}"
    if base_scs is not None:
        line += f"  n_worse={int((scs < base_scs).sum())}/{len(scs)}"
    if extra:
        line += f"   {extra}"
    print(line, flush=True)
    return scs


# ==================================================================================================
# weighted-correlation helper (mirrors _ic / _ic_ew from the ALGO leg, vectorised over rows)
# ==================================================================================================
def wcorr_rows(X, Y, w, min_n=60):
    """Row-wise weighted Pearson corr of X vs Y (both (n, m)), weights w (m,). NaN-aware per row.
    Returns (n,) array, NaN where fewer than min_n valid pairs or degenerate variance."""
    ok = np.isfinite(X) & np.isfinite(Y)
    W = np.where(ok, w[None, :], 0.0)
    sw = W.sum(1)
    Xz = np.where(ok, X, 0.0); Yz = np.where(ok, Y, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mx = (W * Xz).sum(1) / sw; my = (W * Yz).sum(1) / sw
        dx = np.where(ok, Xz - mx[:, None], 0.0); dy = np.where(ok, Yz - my[:, None], 0.0)
        vx = (W * dx * dx).sum(1) / sw; vy = (W * dy * dy).sum(1) / sw
        cxy = (W * dx * dy).sum(1) / sw
        out = cxy / np.sqrt(vx * vy)
    bad = (ok.sum(1) < min_n) | (vx < 1e-24) | (vy < 1e-24) | ~np.isfinite(out)
    out[bad] = np.nan
    return out


def ew_weights(m, hl):
    """Most-recent-sample weight 1, decaying back in time -- same convention as V7._ic_ew."""
    return (0.5 ** (1.0 / hl)) ** np.arange(m - 1, -1, -1)


# ==================================================================================================
# 1) precompute: v7 ridge+blend WZ forecast, v7 ALGO leg, v7 boost (all verbatim V7 building blocks)
# ==================================================================================================
print("=== precompute: ridge+blend WZ forecast (v7-identical) ===", flush=True)
t0 = time.time()
WZ = np.full((nIdio, nt), np.nan)
for t in range(WARMUP, nt):
    rr = r[:, :t]
    fs = []
    for hl in V7.HALF_LIVES:
        B, mx, my = V7._ewls_ridge(rr[:, :-1].T, rr[1:, 1:].T, hl, V7.RIDGE_A)
        pred = my + (rr[:, -1] - mx) @ B
        fi = pred - pred.mean()
        fs.append(fi / (fi.std() + 1e-12))
    wz = np.mean(fs, 0)
    if V7.BLEND > 0:
        rv_ = logp[1:, t] - logp[1:, t - V7.REV_W]
        rv_ = rv_ - rv_.mean()
        rv = -rv_ / (rv_.std() + 1e-12)
        wz = (1 - V7.BLEND) * wz + V7.BLEND * rv
    WZ[:, t] = wz
print(f"  done ({time.time()-t0:.0f}s)", flush=True)

print("=== precompute: v7 ALGO leg (COMBINE_GAIN=16.0) -- identical in every variant ===", flush=True)
t0 = time.time()
algo_pos = np.zeros(nt)
for k in range(130, nt):
    cur0 = P_[0, k]; lim0 = int(dlr[0] / cur0)
    algo_pos[k] = np.clip(V7._algo_vol_shares(logp[0, :k + 1], cur0, dlr[0]), -lim0, lim0)
print(f"  done ({time.time()-t0:.0f}s)", flush=True)

# --------------------------------------------------------------------------------------------------
# boost + the SECOND-ESTIMATOR ICs, computed on exactly the same (x, y) series the shipped gate uses
# --------------------------------------------------------------------------------------------------
FAST_LS = (60, 90, 120, 180)
EW_SPECS = {"ew(20,45)/200": ((20, 45), 200),     # the exact ALGO `_side` estimator
            "ew(20,45)/250": ((20, 45), 250),     # same half-lives, window matched to BOOST_IC_L
            "ew(30,60)/200": ((30, 60), 200)}     # slower half-lives, robustness axis


def boost_with_ics(rsl):
    """V7._pairwise_boost, verbatim in every decision it makes, PLUS the second-estimator ICs
    computed on the identical (x=lead_boost, y=follower next return) series the shipped gate uses.
    Returns (boost, ic_fast{L->arr}, ic_ew{name->arr}); ICs are NaN where no boost is active."""
    n, T = rsl.shape
    boost = np.zeros(n)
    ic_fast = {L: np.full(n, np.nan) for L in FAST_LS}
    ic_ew = {nm: np.full(n, np.nan) for nm in EW_SPECS}
    if T < BOOST_MIN_DAY:
        return boost, ic_fast, ic_ew
    Xi_full = rsl[:, :-1]; Yj = rsl[:, 1:]
    n_samples = Xi_full.shape[1]
    thr = V7._sig_threshold(n_samples)
    vol_causal = np.nanstd(Xi_full, axis=1)
    cand_idx = np.argsort(-vol_causal)[:V7.BOOST_N_CANDIDATES]
    Xi = Xi_full[cand_idx]
    C = V7._corrmat(Xi, Yj)
    for j in range(n):
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
        lead = rsl[i]
        scale = np.nanstd(lead[max(0, T - 1 - V7.BOOST_SCALE_W):T - 1]) + 1e-12
        lead_boost = np.sign(lead) * (np.abs(lead) / scale) ** V7.BOOST_P
        a = max(0, T - 1 - BOOST_IC_L)
        xs = lead_boost[a:T - 1]; ys = rsl[j, a + 1:T]
        ok = ~np.isnan(xs) & ~np.isnan(ys)
        if ok.sum() < 60 or xs[ok].std() < 1e-12:
            continue
        ic = float(np.corrcoef(xs[ok], ys[ok])[0, 1])
        if ic <= 0:
            continue
        boost[j] = lead_boost[-1]
        # ---- second estimators on the SAME series (only needed where the shipped gate passed) ----
        for L in FAST_LS:
            xf = xs[-L:]; yf = ys[-L:]
            v = wcorr_rows(xf[None, :], yf[None, :], np.ones(len(xf)))[0]
            ic_fast[L][j] = v
        for nm, (hls, W) in EW_SPECS.items():
            xw = xs[-W:]; yw = ys[-W:]
            vals = [wcorr_rows(xw[None, :], yw[None, :], ew_weights(len(xw), hl))[0] for hl in hls]
            ic_ew[nm][j] = np.nan if any(not np.isfinite(v) for v in vals) else float(np.mean(vals))
    return boost, ic_fast, ic_ew


print("=== precompute: v7 boost map + second-estimator ICs ===", flush=True)
t0 = time.time()
BOOST_AT = np.zeros((nIdio, nt))
ICF_AT = {L: np.full((nIdio, nt), np.nan) for L in FAST_LS}
ICE_AT = {nm: np.full((nIdio, nt), np.nan) for nm in EW_SPECS}
for k in range(BOOST_MIN_DAY, nt):
    b, icf, ice = boost_with_ics(rs[:, :k])
    BOOST_AT[:, k] = b
    for L in FAST_LS: ICF_AT[L][:, k] = icf[L]
    for nm in EW_SPECS: ICE_AT[nm][:, k] = ice[nm]
print(f"  done ({time.time()-t0:.0f}s, {nt-BOOST_MIN_DAY} days)", flush=True)

active = BOOST_AT != 0.0
n_active_days = active[:, BOOST_MIN_DAY:].sum()
print(f"  baseline boost coverage: {n_active_days} stock-days active "
      f"({n_active_days/(nt-BOOST_MIN_DAY):.1f} names/day of {nIdio})")


# ==================================================================================================
# 2) position builders
# ==================================================================================================
def build_pos(boost_mask=None, idio_mult=None):
    """boost_mask: (nIdio, nt) bool -- keep boost only where True (default: v7, keep all).
    idio_mult:   (nIdio, nt) float in {-1,0,1} -- multiplies the traded sign (default: all 1)."""
    POS = np.zeros((nInst, nt))
    for k in range(WARMUP, nt):
        cur = P_[:, k]; lim = (dlr / cur).astype(int)
        wz = WZ[:, k].copy()
        if k >= BOOST_MIN_DAY:
            b = BOOST_AT[:, k]
            if boost_mask is not None:
                b = np.where(boost_mask[:, k], b, 0.0)
            wz = wz + BOOST_K * b
        sgn = np.sign(wz)
        if idio_mult is not None:
            sgn = sgn * idio_mult[:, k]
        POS[1:, k] = np.clip(sgn * (dlr[1:] / cur[1:]), -lim[1:], lim[1:])
    POS[0, :] = algo_pos
    return POS


print("\n=== sanity check: backtest-equivalent v7 vs shipped README numbers "
      "(830.3 / 888.5 / 876.8 / 674.4) ===")
base_scs = report("v7 baseline (backtest-equiv)", build_pos())
base_wo = window(build_pos(), *OLD); base_wn = window(build_pos(), *NEW); base_rm = base_scs.mean()

results = []


def evaluate(nm, POS, extra=""):
    scs = report(nm, POS, base_scs, extra)
    wo = window(POS, *OLD); wn = window(POS, *NEW)
    passed = (wo > base_wo) and (wn > base_wn) and (scs.mean() > base_rm)
    results.append(dict(name=nm, wo=wo, wn=wn, rm=scs.mean(), rf=scs.min(),
                        nworse=int((scs < base_scs).sum()), passed=passed))
    return scs


# ==================================================================================================
# A) PAIR-LEVEL double IC: second estimator must confirm the shipped 250-day gate
# ==================================================================================================
print("\n### A) PAIR level -- the boost's single IC>0 gate + a second-estimator confirmation ###")
for L in FAST_LS:
    conf = ICF_AT[L] > 0
    mask = conf | ~np.isfinite(ICF_AT[L])        # undecidable -> keep (v7 default philosophy)
    kept = (active & mask)[:, BOOST_MIN_DAY:].sum()
    evaluate(f"A-fast(L={L})", build_pos(boost_mask=mask),
             f"boosts kept {kept}/{n_active_days} ({100*kept/n_active_days:.0f}%)")

for nm in EW_SPECS:
    conf = ICE_AT[nm] > 0
    mask = conf | ~np.isfinite(ICE_AT[nm])
    kept = (active & mask)[:, BOOST_MIN_DAY:].sum()
    tag = "  <-- exact ALGO _side estimator" if nm == "ew(20,45)/200" else ""
    evaluate(f"A-{nm}", build_pos(boost_mask=mask),
             f"boosts kept {kept}/{n_active_days} ({100*kept/n_active_days:.0f}%){tag}")

for L in (90, 120):
    nm = "ew(20,45)/200"
    mask = ((ICF_AT[L] > 0) | ~np.isfinite(ICF_AT[L])) & ((ICE_AT[nm] > 0) | ~np.isfinite(ICE_AT[nm]))
    kept = (active & mask)[:, BOOST_MIN_DAY:].sum()
    evaluate(f"A-both(fast{L}+{nm})", build_pos(boost_mask=mask),
             f"boosts kept {kept}/{n_active_days} ({100*kept/n_active_days:.0f}%)")


# ==================================================================================================
# B) PER-STOCK double IC on the traded score wz (the literal `_side` port, per idio name)
# ==================================================================================================
print("\n### B) PER-STOCK level -- trailing IC of wz[j] vs stock j's own next-day return ###")
print("    (signal WZ[:,t] earns rs[:,t]; newest usable pair at decision day k is t=k-1 -- causal)")

t0 = time.time()
FIRST_IC_DAY = WARMUP + 60          # earliest day any estimator can have 60 pairs


def stock_ic_maps(fast_L, ew_hls, ew_W):
    """(icf, ice) each (nIdio, nt): trailing IC of wz[j] vs j's realised next-day return, using only
    pairs strictly available at decision day k."""
    icf = np.full((nIdio, nt), np.nan)
    ice = np.full((nIdio, nt), np.nan)
    for k in range(FIRST_IC_DAY, nt):
        hi = k                                   # pairs t in [lo, k-1]  ->  slice [lo:hi]
        lo_f = max(WARMUP, hi - fast_L)
        if hi - lo_f >= 60:
            icf[:, k] = wcorr_rows(WZ[:, lo_f:hi], rs[:, lo_f:hi], np.ones(hi - lo_f))
        lo_w = max(WARMUP, hi - ew_W)
        m = hi - lo_w
        if m >= 60:
            vals = [wcorr_rows(WZ[:, lo_w:hi], rs[:, lo_w:hi], ew_weights(m, hl)) for hl in ew_hls]
            ice[:, k] = np.mean(vals, 0)
    return icf, ice


for fast_L in FAST_LS:
    icf, ice = stock_ic_maps(fast_L, (20, 45), 200)
    decidable = np.isfinite(icf) & np.isfinite(ice)
    disagree = decidable & ((icf > 0) != (ice > 0))
    both_neg = decidable & (icf <= 0) & (ice <= 0)
    dd = decidable[:, 150:].sum()
    pct_dis = 100 * disagree[:, 150:].sum() / max(dd, 1)

    # B-veto: disagreement -> flat (literal `_side` port)
    mult = np.where(disagree, 0.0, 1.0)
    evaluate(f"B-veto(fast{fast_L}, ew20/45-200)", build_pos(idio_mult=mult),
             f"flat on {pct_dis:.1f}% of decidable stock-days")

    # B-flip: full `_side` port -- sign taken from the fast IC as well
    mult_f = np.where(disagree, 0.0, np.where(decidable & (icf < 0), -1.0, 1.0))
    evaluate(f"B-flip(fast{fast_L}, ew20/45-200)", build_pos(idio_mult=mult_f),
             f"inverted on {100*(decidable&(icf<0)&~disagree)[:,150:].sum()/max(dd,1):.1f}%")

    # B-pos: strict -- stand down unless BOTH estimators are positive
    mult_p = np.where(disagree | both_neg, 0.0, 1.0)
    evaluate(f"B-pos (fast{fast_L}, ew20/45-200)", build_pos(idio_mult=mult_p),
             f"flat on {100*(disagree|both_neg)[:,150:].sum()/max(dd,1):.1f}%")


# ==================================================================================================
# C) BOOK-LEVEL double IC -- "should we trade the idio book at all today", the direct ALGO analogue
# ==================================================================================================
print("\n### C) BOOK level -- IC pooled across all 50 names; disagreement flattens the whole book ###")


def book_ic_series(fast_L, ew_hls, ew_W):
    icf = np.full(nt, np.nan); ice = np.full(nt, np.nan)
    for k in range(FIRST_IC_DAY, nt):
        hi = k
        lo_f = max(WARMUP, hi - fast_L)
        if hi - lo_f >= 20:
            X = WZ[:, lo_f:hi].ravel()[None, :]; Y = rs[:, lo_f:hi].ravel()[None, :]
            icf[k] = wcorr_rows(X, Y, np.ones(X.shape[1]))[0]
        lo_w = max(WARMUP, hi - ew_W); m = hi - lo_w
        if m >= 20:
            X = WZ[:, lo_w:hi]; Y = rs[:, lo_w:hi]
            vals = []
            for hl in ew_hls:
                w = np.repeat(ew_weights(m, hl)[None, :], nIdio, 0).ravel()
                vals.append(wcorr_rows(X.ravel()[None, :], Y.ravel()[None, :], w)[0])
            ice[k] = float(np.mean(vals))
    return icf, ice


for fast_L in FAST_LS:
    bicf, bice = book_ic_series(fast_L, (20, 45), 200)
    dec = np.isfinite(bicf) & np.isfinite(bice)
    dis = dec & ((bicf > 0) != (bice > 0))
    neg = dec & (bicf <= 0) & (bice <= 0)
    mult = np.where(dis, 0.0, 1.0)[None, :] * np.ones((nIdio, 1))
    evaluate(f"C-veto(fast{fast_L}, ew20/45-200)", build_pos(idio_mult=mult),
             f"book flat {dis[150:].sum()} days ({100*dis[150:].sum()/max(dec[150:].sum(),1):.1f}%)")
    mult_p = np.where(dis | neg, 0.0, 1.0)[None, :] * np.ones((nIdio, 1))
    evaluate(f"C-pos (fast{fast_L}, ew20/45-200)", build_pos(idio_mult=mult_p),
             f"book flat {(dis|neg)[150:].sum()} days")
print(f"  (per-stock + book IC maps: {time.time()-t0:.0f}s)")


# ==================================================================================================
# verdict
# ==================================================================================================
print("\n=== ranking: must beat the v7 baseline on OLD, NEW and rolling-mean JOINTLY ===")
print(f"baseline: OLD={base_wo:.1f} NEW={base_wn:.1f} rmean={base_rm:.1f} rfloor={base_scs.min():.1f}")
passing = [c for c in results if c["passed"]]
for c in passing:
    print(f"  PASS  {c['name']:<40} OLD={c['wo']:.1f} NEW={c['wn']:.1f} rmean={c['rm']:.1f} "
          f"rfloor={c['rf']:.1f} n_worse={c['nworse']}/{len(base_scs)}")
print(f"\n{len(passing)}/{len(results)} variants beat v7 on OLD+NEW+rmean jointly.")
if passing:
    best = max(passing, key=lambda c: c["rm"])
    print(f"Best: {best['name']}  rmean={best['rm']:.1f} (baseline {base_rm:.1f}, "
          f"+{best['rm']-base_rm:.1f})")
else:
    print("Closest by rolling mean:")
    for c in sorted(results, key=lambda c: -c["rm"])[:5]:
        print(f"  {c['name']:<40} OLD={c['wo']:>7.1f} NEW={c['wn']:>7.1f} rmean={c['rm']:>7.1f} "
              f"rfloor={c['rf']:>7.1f} n_worse={c['nworse']}/{len(base_scs)}")
