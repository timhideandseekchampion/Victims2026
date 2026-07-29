"""Scratch sweep: replace the pairwise boost's candidate-SELECTION correlation (currently full,
undecayed history) with a WINDOWED one, to test whether it can reselect a genuinely new leader fast
enough to matter (see README's change-point section) without hurting real-data performance. Not a
committed test_*.py file -- exploratory only, deleted or promoted after the sweep.
"""
import numpy as np, pandas as pd
from scipy import stats
import SAFE_llboost_v11 as V11

P_ = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P_.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
NUMTEST = 250
logp = np.log(P_)
r_full = np.diff(logp, axis=1)


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
OLD_W = (500, 750); NEW_W = (750, nt)
scs_curve = lambda POS: np.array([wscore(POS, E - NUMTEST, E) for E in end_days])


def pairwise_boost_windowed(rs, sel_w):
    """Same as V11._pairwise_boost, except the candidate-selection correlation matrix uses only the
    trailing `sel_w` days (None = full history, i.e. reproduces V11 exactly)."""
    n, T = rs.shape
    boost = np.zeros(n)
    if T < V11.BOOST_MIN_DAY:
        return boost
    lo_sel = 0 if sel_w is None else max(0, T - 1 - sel_w)
    Xi_full = rs[:, lo_sel:-1]; Yj = rs[:, lo_sel + 1:]
    n_samples = Xi_full.shape[1]
    thr = V11._sig_threshold(n_samples)
    vol_causal = np.nanstd(Xi_full, axis=1)
    cand_idx = np.argsort(-vol_causal)[:V11.BOOST_N_CANDIDATES]
    Xi = Xi_full[cand_idx]
    C = V11._corrmat(Xi, Yj)
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
        lead = rs[i]
        scale = np.nanstd(lead[max(0, T - 1 - V11.BOOST_SCALE_W):T - 1]) + 1e-12
        lead_boost = np.sign(lead) * (np.abs(lead) / scale) ** V11.BOOST_P
        a = max(0, T - 1 - V11.BOOST_IC_L)
        xs = lead_boost[a:T - 1]; ys = rs[j, a + 1:T]
        ok = ~np.isnan(xs) & ~np.isnan(ys)
        if ok.sum() < 60 or xs[ok].std() < 1e-12:
            continue
        ic = float(np.corrcoef(xs[ok], ys[ok])[0, 1])
        if ic <= 0:
            continue
        boost[j] = lead_boost[-1]
    return boost


def idio_signal_windowed(prcSoFar, sel_w):
    logp_ = np.log(prcSoFar)
    r = logp_[:, 1:] - logp_[:, :-1]
    Y = V11._beta_adjusted_target(r)
    fs = []
    for hl in V11.HALF_LIVES:
        B, mx, my = V11._ewls_ridge(r[:, :-1].T, Y, hl, V11.RIDGE_A)
        pred = my + (r[:, -1] - mx) @ B
        fi = pred - pred.mean()
        fs.append(fi / (fi.std() + 1e-12))
    wz = np.mean(fs, 0)
    if V11.BLEND > 0:
        rr = logp_[1:, -1] - logp_[1:, -1 - V11.REV_W]
        rr = rr - rr.mean()
        rv = -rr / (rr.std() + 1e-12)
        wz = (1 - V11.BLEND) * wz + V11.BLEND * rv
    boost = pairwise_boost_windowed(r[1:], sel_w)
    wz = wz + V11.BOOST_K * boost
    rs_sig = V11._rank_stability_signal(logp_)
    if rs_sig is not None:
        s_std = rs_sig.std()
        s_z = (rs_sig - rs_sig.mean()) / (s_std + 1e-12) if s_std > 1e-12 else np.zeros_like(rs_sig)
        wz = (1 - V11.RS_WEIGHT) * wz + V11.RS_WEIGHT * s_z * (np.abs(wz).mean() + 1e-12)
    return wz


def build_idio_pos(sel_w):
    """POS[:, t] = position decided using info through day t (prc has t+1 columns) -- matches
    wscore's convention (POS[:,t] applied to the day t->t+1 move). An earlier off-by-one version of
    this file stored at t-1, which silently corrupted every score computed here (verified: real
    data must reproduce 871.0/912.6 at sel_w=None, and only does so with this indexing)."""
    POS = np.zeros((nInst, nt))
    for t in range(V11.WARMUP, nt):
        prc = P_[:, :t + 1]
        cur = prc[:, -1]
        wz = idio_signal_windowed(prc, sel_w)
        pos = np.zeros(nInst)
        pos[1:] = np.sign(wz) * (dlr[1:] / cur[1:])
        lim = (dlr / cur).astype(int)
        POS[:, t] = np.clip(pos, -lim, lim).astype(int)
    return POS


if __name__ == "__main__":
    print("=== sanity: sel_w=None must reproduce v10/v11's own idio-only real-data numbers ===")
    for sel_w in [None, 1500, 1000, 750, 500, 375, 250]:
        POS = build_idio_pos(sel_w)
        wo = wscore(POS, *OLD_W); wn = wscore(POS, *NEW_W); rm = scs_curve(POS).mean()
        print(f"  sel_w={str(sel_w):5s}: OLD={wo:7.1f}  NEW={wn:7.1f}  rmean={rm:7.1f}")
