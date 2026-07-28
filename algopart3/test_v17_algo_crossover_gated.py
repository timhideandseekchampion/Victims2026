"""
test_v17_algo_crossover_gated.py

Final rigor pass on the ALGO crossover idea: does gating the vote by its OWN trailing realized IC
(only trust it when it's recently been working, mirroring the double-IC philosophy already used in
ALGO's `_side()`, and the same "size a signal by its own trailing edge" idea already validated for
ALGO's vol/momentum sub-signals) rescue what a fixed-weight blend could not?

Precedent to weigh against: `test_v7cand_adaptive_boostk.py` tried this exact idea (gate/scale the
PAIRWISE BOOST by its own trailing IC) and found the boost's edge "too stable" for a gate to find a
regime to exploit -- 0/64 configs passed. This crossover signal is the OPPOSITE case: the diagnostic
just found its trailing IC genuinely oscillates in sign 2-5 times over the file, which is exactly the
condition under which a trailing-performance gate COULD help (if it can correctly identify which
regime is which, causally, without lag costing more than it saves).
"""
import numpy as np, pandas as pd, time
import SAFE_llboost_v10 as V10

P_ = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P_.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
NUMTEST = 250
logp = np.log(P_)
r = np.diff(logp, axis=1)
rs = r[1:]
nIdio = rs.shape[0]
lpA = logp[0]
T = len(lpA)
WARMUP, BOOST_MIN_DAY, BOOST_K = V10.WARMUP, V10.BOOST_MIN_DAY, V10.BOOST_K
RIDGE_A = V10.RIDGE_A
HALF_LIVES = V10.HALF_LIVES


def score(mu, sd):
    if mu <= 0 or sd < 1e-10: return float(mu)
    sr = np.sqrt(250) * mu / sd
    return float(mu * sr ** 2 / (sr ** 2 + 1.0))


def wscore(POS, S, E):
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
    return score(tot.mean(), tot.std())


end_days = list(range(400, nt + 1, 10))
OLD = (500, 750); NEW = (750, nt)
scs_curve = lambda POS: np.array([wscore(POS, E - NUMTEST, E) for E in end_days])

print("=== precompute: idio book + ALGO raw target (unchanged, reused verbatim from v10) ===",
      flush=True)
t0 = time.time()
days = list(range(WARMUP, nt))
REV = np.zeros((nIdio, nt))
for t in days:
    rv_ = logp[1:, t] - logp[1:, t - V10.REV_W]
    rv_ = rv_ - rv_.mean()
    REV[:, t] = -rv_ / (rv_.std() + 1e-12)

BOOST = np.zeros((nIdio, nt))
for k in range(BOOST_MIN_DAY, nt):
    BOOST[:, k] = V10._pairwise_boost(rs[:, :k])

WZ_V10 = np.full((nIdio, nt), np.nan)
for t in days:
    rr_ = r[:, :t]
    X = rr_[:, :-1].T
    Y = V10._beta_adjusted_target(rr_)
    xq = rr_[:, -1]
    fs = []
    for hl in HALF_LIVES:
        B, mx, my = V10._ewls_ridge(X, Y, hl, RIDGE_A)
        pred = my + (xq - mx) @ B
        fi = pred - pred.mean()
        fs.append(fi / (fi.std() + 1e-12))
    wz = np.mean(fs, 0)
    wz = (1 - V10.BLEND) * wz + V10.BLEND * REV[:, t]
    if t >= BOOST_MIN_DAY:
        wz = wz + BOOST_K * BOOST[:, t]
    rs_sig = V10._rank_stability_signal(logp[:, :t + 1])
    if rs_sig is not None:
        s_std = rs_sig.std()
        s_z = (rs_sig - rs_sig.mean()) / (s_std + 1e-12) if s_std > 1e-12 else np.zeros_like(rs_sig)
        wz = (1 - V10.RS_WEIGHT) * wz + V10.RS_WEIGHT * s_z * (np.abs(wz).mean() + 1e-12)
    WZ_V10[:, t] = wz


def algo_raw(lpA_):
    Tt = len(lpA_)
    if Tt < V10.VOL_WIN + V10.VOL_Z + 60: return 0.0
    rr = np.diff(lpA_)
    vol = np.full(Tt, np.nan); vol[V10.VOL_WIN:] = V10._roll_std(rr, V10.VOL_WIN)
    tnow = Tt - 1
    lo = max(V10.VOL_WIN + V10.VOL_Z, tnow - V10.IC_LOOKBACK)
    volz = np.full(Tt, np.nan)
    for s in range(lo, Tt):
        wv = vol[s - V10.VOL_Z:s]
        volz[s] = (vol[s] - wv.mean()) / (wv.std() + 1e-12)
    ret1_ = np.full(Tt, np.nan); ret1_[:Tt - 1] = lpA_[1:] - lpA_[:-1]

    def _ic(feat, L):
        a = max(0, tnow - L); xs = feat[a:tnow]; ys = ret1_[a:tnow]
        ok = ~np.isnan(xs) & ~np.isnan(ys)
        if ok.sum() < 60: return None
        xs, ys = xs[ok], ys[ok]
        if xs.std() < 1e-12: return None
        return float(np.corrcoef(xs, ys)[0, 1])

    def _ic_ew(feat, HL, W):
        a = max(0, tnow - W); xs = feat[a:tnow]; ys = ret1_[a:tnow]
        w = (0.5 ** (1.0 / HL)) ** ((tnow - 1) - np.arange(a, tnow))
        ok = ~np.isnan(xs) & ~np.isnan(ys)
        if ok.sum() < 60: return None
        xs, ys, w = xs[ok], ys[ok], w[ok]; sw = w.sum()
        mx = (w * xs).sum() / sw; my = (w * ys).sum() / sw
        cxy = (w * (xs - mx) * (ys - my)).sum() / sw
        vx = (w * (xs - mx) ** 2).sum() / sw; vy = (w * (ys - my) ** 2).sum() / sw
        if vx < 1e-24 or vy < 1e-24: return None
        return float(cxy / np.sqrt(vx * vy))

    def _side(feat, fhv):
        icf = _ic(feat, V10.IC_FAST)
        if icf is None: return None
        sf = 1.0 if icf >= 0 else -1.0
        ics = [_ic_ew(feat, hl, V10.IC_EW_W) for hl in V10.IC_EW_HL]
        if any(x is None for x in ics): return sf * fhv
        ice = float(np.mean(ics))
        return (sf * fhv) if (ice >= 0) == (icf >= 0) else 0.0

    fh = np.clip(volz[tnow], -3, 3) / 3.0
    if np.isnan(fh): return 0.0
    sig = _side(volz, fh)
    if sig is None: return 0.0
    mom_lb = V10.MOM_LB_SHORT if fh > 0 else V10.MOM_LB_LONG
    mom = np.full(Tt, np.nan); mom[mom_lb:] = lpA_[mom_lb:] - lpA_[:-mom_lb]
    z10 = np.full(Tt, np.nan)
    for s in range(max(mom_lb + V10.VOL_Z, tnow - V10.IC_EW_W), Tt):
        wm = mom[s - V10.VOL_Z:s]; z10[s] = (mom[s] - wm.mean()) / (wm.std() + 1e-12)
    fhm = np.clip(z10[tnow], -3, 3) / 3.0
    msig = _side(z10, fhm) if not np.isnan(fhm) else None
    if msig is not None:
        return V10.COMBINE_GAIN * (sig + msig) * 100_000.0
    return V10.SWITCH_GAIN * sig * 100_000.0


AV_RAW = np.zeros(nt)
for k in range(130, nt):
    AV_RAW[k] = algo_raw(logp[0, :k + 1])
print(f"  done ({time.time()-t0:.0f}s)", flush=True)

ret1 = np.full(T, np.nan); ret1[:T - 1] = lpA[1:] - lpA[:-1]


def crossover_vote_full(short_w, long_w):
    vote = np.zeros(T)
    for k in range(max(short_w, long_w) + 5, T):
        long_ret = lpA[k] - lpA[k - long_w]
        short_ret = lpA[k] - lpA[k - short_w]
        if long_ret == 0 or short_ret == 0:
            continue
        if np.sign(long_ret) != np.sign(short_ret):
            vote[k] = np.sign(long_ret)
    return vote


def gated_shares(short_w, long_w, weight, gate_w, min_trades=20):
    """Blend the crossover vote in only on days its OWN trailing IC (over gate_w days, causal, using
    only vote/return pairs strictly before today) is positive; otherwise leave ALGO's target
    untouched (identical to v10)."""
    vote = crossover_vote_full(short_w, long_w)
    out = np.zeros(nt)
    prev = 0; prev_t = -1
    for k in range(130, nt):
        cur0 = P_[0, k]; lim = int(dlr[0] / cur0)
        av = AV_RAW[k]
        have_prev = (k == prev_t + 1)
        if (have_prev and k >= V10.DEADBAND_MIN_DAY
                and abs(av) < V10.DEADBAND_THRESH_FRAC * dlr[0]):
            av_c = float(np.clip(prev * cur0, -dlr[0], dlr[0]))
        else:
            av_c = float(np.clip(av, -dlr[0], dlr[0]))
        v = vote[k] if k < len(vote) else 0.0
        if v != 0:
            lo = max(0, k - gate_w)
            seg_v = vote[lo:k]; seg_r = ret1[lo:k]
            m = (seg_v != 0) & np.isfinite(seg_r)
            trailing_ic = None
            if m.sum() >= min_trades:
                trailing_ic = float(np.corrcoef(seg_v[m], seg_r[m])[0, 1])
            if trailing_ic is not None and trailing_ic > 0:
                av_c = (1 - weight) * av_c + weight * v * dlr[0]
        sh = int(np.clip(av_c / cur0, -lim, lim))
        out[k] = sh; prev = sh; prev_t = k
    return out


def build_pos(algo_shares_arr):
    POS = np.zeros((nInst, nt))
    for t in days:
        cur = P_[:, t]; lim = (dlr / cur).astype(int)
        POS[1:, t] = np.clip(np.sign(WZ_V10[:, t]) * (dlr[1:] / cur[1:]), -lim[1:], lim[1:])
    POS[0, :] = algo_shares_arr
    return POS


def algo_baseline_shares():
    out = np.zeros(nt)
    prev = 0; prev_t = -1
    for k in range(130, nt):
        cur0 = P_[0, k]; lim = int(dlr[0] / cur0)
        av = AV_RAW[k]
        have_prev = (k == prev_t + 1)
        if (have_prev and k >= V10.DEADBAND_MIN_DAY
                and abs(av) < V10.DEADBAND_THRESH_FRAC * dlr[0]):
            sh = prev
        else:
            av_c = float(np.clip(av, -dlr[0], dlr[0]))
            sh = int(np.clip(av_c / cur0, -lim, lim))
        out[k] = sh; prev = sh; prev_t = k
    return out


print("\n=== sanity check ===")
POS_base = build_pos(algo_baseline_shares())
base_scs = scs_curve(POS_base)
base_wo, base_wn = wscore(POS_base, *OLD), wscore(POS_base, *NEW)
print(f"  baseline: OLD={base_wo:.1f}  NEW={base_wn:.1f}  rmean={base_scs.mean():.1f}  "
      f"rfloor={base_scs.min():.1f}   (v10 docstring: 871.0/912.6/909.8/709.7)")
ok = abs(base_wo - 871.0) < 0.5 and abs(base_wn - 912.6) < 0.5
print("  OK -- matches v10." if ok else "  *** WARNING: mismatch. ***")


def evaluate(nm, short_w, long_w, weight, gate_w, verbose=True):
    Pz = build_pos(gated_shares(short_w, long_w, weight, gate_w)); scs = scs_curve(Pz)
    wo = wscore(Pz, *OLD); wn = wscore(Pz, *NEW)
    passed = (wo > base_wo) and (wn > base_wn) and (scs.mean() > base_scs.mean())
    nworse = int((scs < base_scs).sum())
    if verbose:
        tag = "  <== PASS" if passed else ""
        print(f"  {nm:<40}OLD={wo:7.1f}  NEW={wn:7.1f}  rmean={scs.mean():7.1f}  rfloor={scs.min():7.1f}  "
              f"n_worse={nworse}/{len(scs)}{tag}")
    return dict(wo=wo, wn=wn, rm=scs.mean(), nworse=nworse, passed=passed)


print("\n=== trailing-IC-gated crossover: (short_w, long_w) x gate_w x weight ===")
results = []
for sw, lw in ((5, 10), (5, 15), (8, 22)):
    for gate_w in (150, 250, 350):
        for w in (0.1, 0.2, 0.3):
            r_ = evaluate(f"short{sw}_long{lw} gate_w={gate_w} w={w}", sw, lw, w, gate_w, verbose=False)
            results.append((sw, lw, gate_w, w, r_))
            tag = "  <== PASS" if r_["passed"] else ""
            print(f"  short{sw}_long{lw} gate_w={gate_w:<4} w={w:<4} OLD={r_['wo']:7.1f} "
                  f"NEW={r_['wn']:7.1f} rmean={r_['rm']:7.1f} n_worse={r_['nworse']}/61{tag}")

passing = [x for x in results if x[4]["passed"]]
print(f"\n{len(passing)}/{len(results)} gated configs beat v10 on OLD+NEW+rmean jointly.")
