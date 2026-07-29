"""Scratch sweep: EXPONENTIALLY-DECAYED candidate-selection correlation (half-life based, same
math style as _ewls_ridge) instead of a hard trailing window, to see if it can adapt faster than
the full-history version without giving up real-data edge the way a hard window does (see
test_v12cand_boost_sel_window.py: hard windows below ~750 days degrade OLD/NEW/rmean monotonically).
"""
import numpy as np, pandas as pd
import SAFE_llboost_v11 as V11
from test_v12cand_boost_sel_window import P_, dlr, nInst, nt, wscore, scs_curve, OLD_W, NEW_W

commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5


def _corrmat_weighted(X, Y, w):
    """Weighted Pearson correlation matrix, same shape convention as V11._corrmat."""
    sw = w.sum()
    mx = (w[None, :] * X).sum(1, keepdims=True) / sw
    my = (w[None, :] * Y).sum(1, keepdims=True) / sw
    Xc, Yc = X - mx, Y - my
    vx = (w[None, :] * Xc * Xc).sum(1) / sw; vy = (w[None, :] * Yc * Yc).sum(1) / sw
    cov = (Xc * w[None, :]) @ Yc.T / sw
    denom = np.sqrt(vx[:, None] * vy[None, :]) + 1e-12
    return cov / denom


def pairwise_boost_decayed(rs, hl):
    """hl=None reproduces V11._pairwise_boost exactly (equal weight). hl=<half-life in days>
    exponentially decays the candidate-selection correlation (NOT the trailing-250 validation IC,
    which stays as-is -- only the selection step is being tested here)."""
    n, T = rs.shape
    boost = np.zeros(n)
    if T < V11.BOOST_MIN_DAY:
        return boost
    Xi_full = rs[:, :-1]; Yj = rs[:, 1:]
    n_samples = Xi_full.shape[1]
    if hl is None:
        w = np.ones(n_samples)
    else:
        lam = 0.5 ** (1.0 / hl)
        w = lam ** np.arange(n_samples - 1, -1, -1)
    # effective sample size (Kish) for the significance threshold, in place of raw n_samples
    n_eff = float(w.sum() ** 2 / (w ** 2).sum())
    thr = V11._sig_threshold(max(10, int(n_eff)))
    vol_causal = np.sqrt(np.average((Xi_full - np.average(Xi_full, axis=1, weights=w, keepdims=True)) ** 2,
                                     axis=1, weights=w))
    cand_idx = np.argsort(-vol_causal)[:V11.BOOST_N_CANDIDATES]
    Xi = Xi_full[cand_idx]
    C = _corrmat_weighted(Xi, Yj, w)
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


def idio_signal_decayed(prcSoFar, hl):
    logp_ = np.log(prcSoFar)
    r = logp_[:, 1:] - logp_[:, :-1]
    Y = V11._beta_adjusted_target(r)
    fs = []
    for h in V11.HALF_LIVES:
        B, mx, my = V11._ewls_ridge(r[:, :-1].T, Y, h, V11.RIDGE_A)
        pred = my + (r[:, -1] - mx) @ B
        fi = pred - pred.mean()
        fs.append(fi / (fi.std() + 1e-12))
    wz = np.mean(fs, 0)
    if V11.BLEND > 0:
        rr = logp_[1:, -1] - logp_[1:, -1 - V11.REV_W]
        rr = rr - rr.mean()
        rv = -rr / (rr.std() + 1e-12)
        wz = (1 - V11.BLEND) * wz + V11.BLEND * rv
    boost = pairwise_boost_decayed(r[1:], hl)
    wz = wz + V11.BOOST_K * boost
    rs_sig = V11._rank_stability_signal(logp_)
    if rs_sig is not None:
        s_std = rs_sig.std()
        s_z = (rs_sig - rs_sig.mean()) / (s_std + 1e-12) if s_std > 1e-12 else np.zeros_like(rs_sig)
        wz = (1 - V11.RS_WEIGHT) * wz + V11.RS_WEIGHT * s_z * (np.abs(wz).mean() + 1e-12)
    return wz


def build_idio_pos_decayed(hl):
    POS = np.zeros((nInst, nt))
    for t in range(V11.WARMUP, nt):
        prc = P_[:, :t + 1]
        cur = prc[:, -1]
        wz = idio_signal_decayed(prc, hl)
        pos = np.zeros(nInst)
        pos[1:] = np.sign(wz) * (dlr[1:] / cur[1:])
        lim = (dlr / cur).astype(int)
        POS[:, t] = np.clip(pos, -lim, lim).astype(int)
    return POS


if __name__ == "__main__":
    print("=== sanity: hl=None must reproduce v11's own idio-only real-data numbers (717.4/688.1/742.0) ===")
    for hl in [None, 2000, 1000, 750, 500, 375, 250]:
        POS = build_idio_pos_decayed(hl)
        wo = wscore(POS, *OLD_W); wn = wscore(POS, *NEW_W); rm = scs_curve(POS).mean()
        print(f"  hl={str(hl):5s}: OLD={wo:7.1f}  NEW={wn:7.1f}  rmean={rm:7.1f}")
