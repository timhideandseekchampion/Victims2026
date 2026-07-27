"""Unconventional gate: instead of formal significance (kills real-but-modest signals due to low
statistical power at n=90) or 2-window sign-agreement (lets noise through ~83% of the time), use
BREADTH of directional agreement across MANY independent lookback windows (8 windows spanning
30-200 days). A real, structural signal should show consistent direction across a wide range of
timescales; a pure-noise candidate should only agree across an arbitrary subset by chance. This
keeps the "consensus on direction, not magnitude" philosophy that makes the existing gate work, but
raises the bar from 2-window agreement to an 8-window supermajority vote.
"""
import numpy as np, pandas as pd
import SAFE, SAFE_llvol as M

P = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
NUMTEST = 250
logp = np.log(P)


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
    return {"mu": float(tot.mean()), "sd": float(tot.std()), "score": score(tot.mean(), tot.std())}


print("building idio book (shipped, unchanged) ...")
idio_pos = np.zeros((nInst, nt))
for k in range(130, nt):
    cur = P[:, k]; lim = (dlr / cur).astype(int)
    full = np.asarray(SAFE.getMyPosition(P[:, :k + 1])); p = full.copy(); p[0] = 0
    idio_pos[:, k] = np.clip(p, -lim, lim).astype(int)
print("done")

lpA = logp[0]; r = np.diff(lpA); T = len(lpA)
vol = np.full(T, np.nan); vol[M.VOL_WIN:] = M._roll_std(r, M.VOL_WIN)
volz = np.full(T, np.nan)
for s in range(M.VOL_WIN + M.VOL_Z, T):
    wv = vol[s - M.VOL_Z:s]; volz[s] = (vol[s] - wv.mean()) / (wv.std() + 1e-12)
ret1 = np.full(T, np.nan); ret1[:T - 1] = lpA[1:] - lpA[:-1]
mom = np.full(T, np.nan); mom[M.MOM_LB:] = lpA[M.MOM_LB:] - lpA[:-M.MOM_LB]
z10 = np.full(T, np.nan)
for s in range(M.MOM_LB + M.VOL_Z, T):
    wm = mom[s - M.VOL_Z:s]; z10[s] = (mom[s] - wm.mean()) / (wm.std() + 1e-12)
REV_K = 5
revmove = np.full(T, np.nan); revmove[REV_K:] = lpA[REV_K:] - lpA[:-REV_K]
zrev = np.full(T, np.nan)
for s in range(REV_K + M.VOL_Z, T):
    wv = revmove[s - M.VOL_Z:s]; zrev[s] = (revmove[s] - wv.mean()) / (wv.std() + 1e-12)
zrev_faded = -zrev

VOTE_WINDOWS = (30, 50, 70, 90, 110, 130, 150, 180)


def _ic(feat, tnow, L):
    a = max(0, tnow - L); xs = feat[a:tnow]; ys = ret1[a:tnow]
    ok = ~np.isnan(xs) & ~np.isnan(ys)
    if ok.sum() < max(30, L // 3): return None
    xs, ys = xs[ok], ys[ok]
    if xs.std() < 1e-12: return None
    return float(np.corrcoef(xs, ys)[0, 1])


def _side_vote(feat, tnow, fhv, need_frac):
    """Sign from the primary 90-day IC; active only if >= need_frac of VOTE_WINDOWS agree with it."""
    ics = [_ic(feat, tnow, L) for L in VOTE_WINDOWS]
    valid = [x for x in ics if x is not None]
    if len(valid) < 5: return None
    primary = _ic(feat, tnow, 90)
    if primary is None: return None
    sf = 1.0 if primary >= 0 else -1.0
    agree = sum(1 for x in valid if (1.0 if x >= 0 else -1.0) == sf)
    frac = agree / len(valid)
    return (sf * fhv) if frac >= need_frac else 0.0


OLD = (500, 750); NEW = (750, nt); end_days = list(range(400, nt + 1, 10))


def build_pos(use_reversion, gain, need_frac):
    POS = idio_pos.copy()
    n_rev = n_vol = n_mom = n_days = 0
    for k in range(180, nt):
        cur = P[:, k]; lim = (dlr / cur).astype(int)
        fhv = np.clip(volz[k], -3, 3) / 3.0 if not np.isnan(volz[k]) else np.nan
        fhm = np.clip(z10[k], -3, 3) / 3.0 if not np.isnan(z10[k]) else np.nan
        fhr = np.clip(zrev_faded[k], -3, 3) / 3.0 if not np.isnan(zrev_faded[k]) else np.nan
        sig = _side_vote(volz, k, fhv, need_frac) if not np.isnan(fhv) else None
        msig = _side_vote(z10, k, fhm, need_frac) if not np.isnan(fhm) else None
        rsig = _side_vote(zrev_faded, k, fhr, need_frac) if (use_reversion and not np.isnan(fhr)) else None
        if sig: n_vol += 1
        if msig: n_mom += 1
        if rsig: n_rev += 1
        n_days += 1
        parts = [x for x in (sig, msig, rsig) if x is not None]
        if not parts:
            av = 0.0
        elif len(parts) == 1:
            av = M.SWITCH_GAIN * parts[0] * 100_000.0
        else:
            av = gain * sum(parts) * 100_000.0
        POS[0, k] = int(np.clip(np.clip(av, -dlr[0], dlr[0]) / cur[0], -lim[0], lim[0]))
    return POS, n_rev, n_vol, n_mom, n_days


def report(nm, POS):
    wo = window(POS, *OLD); wn = window(POS, *NEW)
    scs = np.array([window(POS, E - NUMTEST, E)["score"] for E in end_days])
    print(f"{nm:<40}OLD={wo['score']:>8.1f}  NEW={wn['score']:>8.1f}  "
          f"rmean={scs.mean():>8.1f}  rfloor={scs.min():>8.1f}")
    return scs


print(f"\n--- vol+mom only, supermajority-gated (no reversion yet) ---")
for need_frac in (0.55, 0.625, 0.7, 0.75, 0.8, 0.875):
    POS, _, n_vol, n_mom, n_days = build_pos(False, M.COMBINE_GAIN, need_frac)
    scs = report(f"vol+mom, vote>={need_frac:.3f}", POS)
    print(f"    vol active {n_vol}/{n_days} ({100*n_vol/n_days:.0f}%)  mom active {n_mom}/{n_days} ({100*n_mom/n_days:.0f}%)")

print("\n--- shipped reference ---")
base_POS, *_ = build_pos(False, M.COMBINE_GAIN, need_frac=0.0)  # 0.0 = always passes = plain switch, no vote
base_scs = report("no-gate reference (always active)", base_POS)

print("\n--- NOW add reversion, supermajority-gated: does the vote suppress the fake signal? ---")
for need_frac in (0.55, 0.625, 0.7, 0.75, 0.8, 0.875):
    POS, n_rev, n_vol, n_mom, n_days = build_pos(True, M.COMBINE_GAIN, need_frac)
    scs = report(f"vol+mom+rev, vote>={need_frac:.3f}", POS)
    nworse = int((scs < base_scs).sum())
    print(f"    rev active {n_rev}/{n_days} ({100*n_rev/n_days:.0f}%)  "
          f"n_worse_vs_no-gate-ref={nworse}/{len(scs)}")

vol_only_ref, _, n_vol0, n_mom0, n_days0 = build_pos(False, M.COMBINE_GAIN, 0.0)
print(f"\n(for reference, at need_frac=0.0 i.e. no filtering at all: rev would show its RAW activity rate)")
_, n_rev0, _, _, _ = build_pos(True, M.COMBINE_GAIN, 0.0)
print(f"reversion active with NO vote filter at all: {n_rev0}/{n_days0} ({100*n_rev0/n_days0:.0f}%)")
