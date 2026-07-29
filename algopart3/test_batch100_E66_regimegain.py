"""
test_batch100_E66_regimegain.py

E66: regime-conditional COMBINE_GAIN for ALGO -- a different gain in HIGH vs LOW realized-vol regimes,
instead of the single shipped COMBINE_GAIN=16.0 applied uniformly on every day the combine (sig+msig)
path fires.

"Regime" = sign of the day's fast vol-z feature fh = clip(volz[tnow],-3,3)/3 -- the SAME quantity V10
already uses to pick the momentum lookback (MOM_LB_SHORT if fh>0 else MOM_LB_LONG), i.e. fh>0 is
already treated as a distinct ("high realized vol") regime by the shipped code; this extends that same
regime split to the combine gain: av = GAIN_HI*(sig+msig)*100000 if fh>0 else GAIN_LO*(sig+msig)*100000.
GAIN_HI=GAIN_LO=16.0 reproduces v10 exactly (sanity check).

Because the change is to the av-computation FORMULA (not a single swappable constant), this is a
faithful line-by-line copy of V10._algo_vol_shares' VOL_MODE="switch"/VOL_COMBINE=True branch (the only
branch V10's shipped constants ever take), reusing V10._roll_std and every other V10 constant verbatim,
with only the final gain line parameterized -- and with its own LOCAL state dict (not V10's module
globals), so it doesn't collide with the shared precompute's already-populated algo_pos state. The idio
side of the book is unaffected by this idea -- reused verbatim from batch100_d6x_shared.py.
"""
import numpy as np, time
import SAFE_llboost_v10 as V10
import batch100_d6x_shared as SH

logp, nt, dlr, P_ = SH.logp, SH.nt, SH.dlr, SH.P_
score, wscore, scs_curve, OLD, NEW = SH.score, SH.wscore, SH.scs_curve, SH.OLD, SH.NEW

print(f"\nSANITY_CHECK_PASSED (shared baseline, idio side identical for this idea) = {SH.SANITY_OK}")

assert V10.VOL_MODE == "switch" and V10.VOL_COMBINE, \
    "this copy only reimplements the switch+combine branch V10 actually ships"


def _algo_vol_shares_regime(lpA, cur0, cap_dol, gain_hi, gain_lo, state):
    """Faithful copy of V10._algo_vol_shares' VOL_MODE='switch', VOL_COMBINE=True branch, with the
    single COMBINE_GAIN constant replaced by a regime-conditional GAIN_HI/GAIN_LO pair (regime = sign
    of fh, the same fast vol-z feature already used to pick MOM_LB_SHORT vs MOM_LB_LONG)."""
    T = len(lpA)
    tnow = T - 1
    have_prev = (tnow == state['prev_t'] + 1)
    if T < V10.VOL_WIN + V10.VOL_Z + 60:
        state['prev_t'] = tnow; state['prev_shares'] = 0
        return 0
    r = np.diff(lpA)
    vol = np.full(T, np.nan)
    vol[V10.VOL_WIN:] = V10._roll_std(r, V10.VOL_WIN)
    lo = max(V10.VOL_WIN + V10.VOL_Z, tnow - V10.IC_LOOKBACK)
    volz = np.full(T, np.nan)
    for s in range(lo, T):
        wv = vol[s - V10.VOL_Z:s]
        volz[s] = (vol[s] - wv.mean()) / (wv.std() + 1e-12)
    ret1 = np.full(T, np.nan); ret1[:T - 1] = lpA[1:] - lpA[:-1]

    def _ic(feat, L):
        a = max(0, tnow - L); xs = feat[a:tnow]; ys = ret1[a:tnow]
        ok = ~np.isnan(xs) & ~np.isnan(ys)
        if ok.sum() < 60: return None
        xs, ys = xs[ok], ys[ok]
        if xs.std() < 1e-12: return None
        return float(np.corrcoef(xs, ys)[0, 1])

    def _ic_ew(feat, HL, W):
        a = max(0, tnow - W); xs = feat[a:tnow]; ys = ret1[a:tnow]
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
        if not V10.IC_BLEND: return sf * fhv
        ics = [x for x in (_ic_ew(feat, hl, V10.IC_EW_W) for hl in V10.IC_EW_HL) if x is not None]
        if len(ics) < len(V10.IC_EW_HL): return sf * fhv
        ice = float(np.mean(ics))
        return (sf * fhv) if (ice >= 0) == (icf >= 0) else 0.0

    fh = np.clip(volz[tnow], -3, 3) / 3.0
    if np.isnan(fh):
        state['prev_t'] = tnow; state['prev_shares'] = 0
        return 0
    sig = _side(volz, fh)
    if sig is None:
        state['prev_t'] = tnow; state['prev_shares'] = 0
        return 0
    mom_lb = V10.MOM_LB_SHORT if fh > 0 else V10.MOM_LB_LONG
    mom = np.full(T, np.nan); mom[mom_lb:] = lpA[mom_lb:] - lpA[:-mom_lb]
    z10 = np.full(T, np.nan)
    for s in range(max(mom_lb + V10.VOL_Z, tnow - V10.IC_EW_W), T):
        wm = mom[s - V10.VOL_Z:s]; z10[s] = (mom[s] - wm.mean()) / (wm.std() + 1e-12)
    fhm = np.clip(z10[tnow], -3, 3) / 3.0
    msig = _side(z10, fhm) if not np.isnan(fhm) else None
    gain = gain_hi if fh > 0 else gain_lo
    if msig is not None:
        av = gain * (sig + msig) * 100_000.0
    else:
        av = V10.SWITCH_GAIN * sig * 100_000.0

    av = float(np.clip(av, -cap_dol, cap_dol))
    lim = int(cap_dol / cur0)
    if have_prev and tnow >= V10.DEADBAND_MIN_DAY and abs(av) < V10.DEADBAND_THRESH_FRAC * cap_dol:
        shares = int(np.clip(state['prev_shares'], -lim, lim))
    else:
        shares = int(np.clip(av / cur0, -lim, lim))
    state['prev_shares'] = shares
    state['prev_t'] = tnow
    return shares


def compute_algo(gain_hi, gain_lo):
    state = {'prev_t': -1, 'prev_shares': 0}
    algo = np.zeros(nt)
    for k in range(130, nt):
        cur0 = P_[0, k]; lim0 = int(dlr[0] / cur0)
        algo[k] = np.clip(_algo_vol_shares_regime(logp[0, :k + 1], cur0, dlr[0], gain_hi, gain_lo, state),
                           -lim0, lim0)
    return algo


def build_pos_with_algo(algo):
    POS = SH.POS_BASE.copy()
    POS[0, :] = algo
    return POS


print("\n=== sanity check: GAIN_HI=GAIN_LO=COMBINE_GAIN=16.0 (mechanism OFF, uniform gain) -- must "
      "exactly match the shared-cache algo_pos array AND reproduce SAFE_llboost_v10's official numbers "
      "===")
t0 = time.time()
G = V10.COMBINE_GAIN
algo_base = compute_algo(G, G)
exact_match = np.array_equal(algo_base, SH.algo_pos)
print(f"  array-identical to shared algo_pos: {exact_match}  [{time.time()-t0:.0f}s]")
POS_base2 = build_pos_with_algo(algo_base)
scs_base2 = scs_curve(POS_base2)
wo0, wn0 = wscore(POS_base2, *OLD), wscore(POS_base2, *NEW)
print(f"  OLD={wo0:.1f}  NEW={wn0:.1f}  rmean={scs_base2.mean():.1f}  rfloor={scs_base2.min():.1f}   "
      f"(v10 docstring: 871.0/912.6/909.8/709.7)")
SANITY_OK = exact_match and abs(wo0 - 871.0) < 0.5 and abs(wn0 - 912.6) < 0.5 and SH.SANITY_OK
print("  OK -- matches v10 exactly." if SANITY_OK else
      "  *** WARNING: baseline does NOT reproduce v10 -- do not trust results below. ***")


def evaluate(nm, gain_hi, gain_lo):
    t0 = time.time()
    algo = compute_algo(gain_hi, gain_lo)
    Pz = build_pos_with_algo(algo)
    scs = scs_curve(Pz)
    wo = wscore(Pz, *OLD); wn = wscore(Pz, *NEW)
    passed = (wo > wo0) and (wn > wn0) and (scs.mean() > scs_base2.mean())
    nworse = int((scs < scs_base2).sum())
    tag = "  <== PASS" if passed else ""
    print(f"  {nm:<28}OLD={wo:7.1f}  NEW={wn:7.1f}  rmean={scs.mean():7.1f}  rfloor={scs.min():7.1f}  "
          f"n_worse={nworse}/{len(scs)}{tag}  [{time.time()-t0:.0f}s]")
    return dict(name=nm, wo=wo, wn=wn, rm=scs.mean(), rf=scs.min(), nworse=nworse, passed=passed)


print(f"\n=== CANDIDATE: regime-conditional gain, (GAIN_HI, GAIN_LO) around shipped uniform {G} ===")
CONFIGS = [(12.0, G), (20.0, G), (G, 12.0), (G, 20.0)]
results = [evaluate(f"HI={hi},LO={lo}", hi, lo) for hi, lo in CONFIGS]

passing = [c for c in results if c["passed"]]
print(f"\n{len(passing)}/{len(results)} regime-gain configs beat v10 on OLD+NEW+rmean jointly.")
for c in sorted(results, key=lambda c: -c["rm"]):
    print(f"  {c['name']:<28} OLD={c['wo']:>7.1f} NEW={c['wn']:>7.1f} rmean={c['rm']:>7.1f} "
          f"rfloor={c['rf']:>7.1f} n_worse={c['nworse']}/61")

print(f"\nSANITY_CHECK_PASSED={SANITY_OK}")
