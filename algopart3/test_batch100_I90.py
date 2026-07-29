"""
test_batch100_I90.py

I90 (DIAGNOSTIC): leave-one-day-out sensitivity check on the BOOST_N_CANDIDATES=39 "isolated
spike" finding from test_v19cand_boost_ncandidates.py (also re-confirmed in the README's
"Re-sweeping BOOST_N_CANDIDATES" section): does removing any single day from the OLD/NEW scoring
change whether N=39 looks like a spike (uniquely best) or a plateau (ties/near-ties with neighbors)?

Reuses batch100_common_gi's cached WZ_PRE / RS_RAW / algo_pos (independent of BOOST_N_CANDIDATES) --
per house convention, only the boost itself (which DOES depend on N) is recomputed, once per
candidate N, exactly like test_v19cand_boost_ncandidates.py's own `boost_at_day`.
"""
import numpy as np
import SAFE_llboost_v10 as V10
import batch100_common_gi as G

P_, dlr, nInst, nIdio = G.P_, G.dlr, G.nInst, G.nIdio
WZ_PRE, RS_RAW, algo_pos = G.WZ_PRE, G.RS_RAW, G.algo_pos
RS_WEIGHT = G.RS_WEIGHT
days, rs = G.days, G.rs
OLD, NEW = G.OLD, G.NEW
score = G.score

print(G.print_sanity("(I90, via batch100_common_gi)"))

NEIGHBORS = [35, 37, 38, 39, 40, 41, 42]  # exact neighborhood from the README re-sweep table


def boost_at_day(k, n_candidates):
    """Exact copy of V10._pairwise_boost's body, n_candidates parameterized (same technique as
    test_v19cand_boost_ncandidates.py's boost_at_day)."""
    rs_k = rs[:, :k]
    T = k
    Xi_full = rs_k[:, :-1]; Yj = rs_k[:, 1:]
    n_samples = Xi_full.shape[1]
    thr = V10._sig_threshold(n_samples) if n_candidates == V10.BOOST_N_CANDIDATES else _sig_thr(n_samples, n_candidates)
    vol_causal = np.nanstd(Xi_full, axis=1)
    cand_idx = np.argsort(-vol_causal)[:n_candidates]
    Xi = Xi_full[cand_idx]
    C = V10._corrmat(Xi, Yj)
    boost = np.zeros(nIdio)
    for j in range(nIdio):
        col = C[:, j].copy()
        cp = np.where(cand_idx == j)[0]
        if len(cp): col[cp[0]] = np.nan
        if np.all(np.isnan(col)): continue
        ci = int(np.nanargmax(np.abs(col)))
        if abs(col[ci]) <= thr: continue
        i = cand_idx[ci]
        lead = rs_k[i]
        scale = np.nanstd(lead[max(0, T - 1 - V10.BOOST_SCALE_W):T - 1]) + 1e-12
        lead_boost = np.sign(lead) * (np.abs(lead) / scale) ** V10.BOOST_P
        a = max(0, T - 1 - V10.BOOST_IC_L)
        xs = lead_boost[a:T - 1]; ys = rs_k[j, a + 1:T]
        ok = ~np.isnan(xs) & ~np.isnan(ys)
        if ok.sum() < 60 or xs[ok].std() < 1e-12: continue
        ic = float(np.corrcoef(xs[ok], ys[ok])[0, 1])
        if ic <= 0: continue
        boost[j] = lead_boost[-1]
    return boost


def _sig_thr(n_samples, n_candidates):
    from scipy import stats
    if n_samples < 10: return 1.0
    alpha_adj = V10.BOOST_ALPHA / n_candidates
    tcrit = stats.t.ppf(1 - alpha_adj / 2, df=n_samples - 2)
    return float(tcrit / np.sqrt(n_samples - 2 + tcrit ** 2))


def build_pos(n_candidates):
    POS = np.zeros((nInst, G.nt))
    for t in days:
        wz = WZ_PRE[:, t].copy()
        if t >= V10.BOOST_MIN_DAY:
            wz = wz + V10.BOOST_K * boost_at_day(t, n_candidates)
        s = RS_RAW[:, t]
        if np.isfinite(s).all():
            sstd = s.std()
            s_z = (s - s.mean()) / (sstd + 1e-12) if sstd > 1e-12 else np.zeros(nIdio)
            wz = (1 - RS_WEIGHT) * wz + RS_WEIGHT * s_z * (np.abs(wz).mean() + 1e-12)
        cur = P_[:, t]; lim = (dlr / cur).astype(int)
        POS[1:, t] = np.clip(np.sign(wz) * (dlr[1:] / cur[1:]), -lim[1:], lim[1:])
    POS[0, :] = algo_pos
    return POS


def daily_pnl(POS, S, E):
    commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
    curPos = np.zeros(nInst); comm_vec = np.zeros(nInst); prevCur = None; tot = []
    for tt in range(S, E + 1):
        cur = P_[:, tt - 1]
        newPos = (POS[:, tt - 1].copy() if tt < E else curPos.copy())
        if tt > S:
            tot.append(float((curPos * (cur - prevCur) - comm_vec).sum()))
        dP = newPos - curPos
        comm_vec = commRate * np.abs(dP) * cur
        prevCur = cur; curPos = newPos
    return np.array(tot)


print(f"\n=== building POS + daily PnL (OLD, NEW) for N in {NEIGHBORS} ===", flush=True)
PNL_OLD, PNL_NEW = {}, {}
base_score = {}
for n in NEIGHBORS:
    POS = build_pos(n)
    PNL_OLD[n] = daily_pnl(POS, *OLD)
    PNL_NEW[n] = daily_pnl(POS, *NEW)
    wo = score(PNL_OLD[n].mean(), PNL_OLD[n].std())
    wn = score(PNL_NEW[n].mean(), PNL_NEW[n].std())
    base_score[n] = (wo, wn)
    print(f"  N={n:<3} OLD={wo:7.1f}  NEW={wn:7.1f}")

best_old_n = max(NEIGHBORS, key=lambda n: base_score[n][0])
best_new_n = max(NEIGHBORS, key=lambda n: base_score[n][1])
print(f"\nBaseline (no day removed): OLD-best N={best_old_n} ({base_score[best_old_n][0]:.1f}), "
      f"NEW-best N={best_new_n} ({base_score[best_new_n][1]:.1f})")
print("(N=39 is expected to be the OLD-best per the README table; NEW-best tends to favor slightly "
      "higher N there -- the 'spike' claim is about the rolling-mean metric, checked at the day level "
      "on OLD/NEW below)")


def loo_flip_rate(pnl_dict, baseline_best):
    n_days = len(next(iter(pnl_dict.values())))
    flips = 0
    for d in range(n_days):
        scores = {}
        for n, arr in pnl_dict.items():
            sub = np.delete(arr, d)
            scores[n] = score(sub.mean(), sub.std())
        winner = max(scores, key=scores.get)
        if winner != baseline_best:
            flips += 1
    return flips, n_days


flips_old, n_old = loo_flip_rate(PNL_OLD, best_old_n)
flips_new, n_new = loo_flip_rate(PNL_NEW, best_new_n)
print(f"\n=== leave-one-day-out: does removing a single day change which N wins OLD / NEW? ===")
print(f"  OLD window: {flips_old}/{n_old} single-day removals flip the OLD-best N away from {best_old_n}")
print(f"  NEW window: {flips_new}/{n_new} single-day removals flip the NEW-best N away from {best_new_n}")

# margin: how close is the runner-up, at baseline, on OLD (the metric where N=39 actually leads)?
old_sorted = sorted(NEIGHBORS, key=lambda n: -base_score[n][0])
print(f"\nOLD ranking (baseline): " + ", ".join(f"N={n}:{base_score[n][0]:.1f}" for n in old_sorted))
print(f"  margin of N={old_sorted[0]} over runner-up N={old_sorted[1]}: "
      f"{base_score[old_sorted[0]][0]-base_score[old_sorted[1]][0]:.1f}")
