"""Does the exponentially-decayed candidate-selection correlation (test_v12cand_boost_sel_decay.py)
actually re-discover a genuinely new leader faster than the undecayed baseline, in the rotate-mode
change-point scenario? Reuses the same tracked 20 (follower, old-leader, new-leader) pairs from
changepoint_synthetic.py. Tracks the boost's candidate-selection outcome directly (no full
getMyPosition/PnL machinery needed for this diagnostic).
"""
import numpy as np
from changepoint_synthetic import simulate, followers, leaders_old
import SAFE_llboost_v11 as V11
from test_v12cand_boost_sel_decay import _corrmat_weighted


def track_selection(rs, hl, track_followers):
    """Returns {j: selected_leader_idx or -1} for one day's call, using the SAME candidate-pool
    logic as pairwise_boost_decayed but returning the raw argmax leader per tracked follower,
    regardless of significance (to see who WOULD be picked, then separately note if it clears the
    Bonferroni bar)."""
    n, T = rs.shape
    out = {}
    if T < V11.BOOST_MIN_DAY:
        return {j: (-1, False) for j in track_followers}
    Xi_full = rs[:, :-1]; Yj = rs[:, 1:]
    n_samples = Xi_full.shape[1]
    if hl is None:
        w = np.ones(n_samples)
    else:
        lam = 0.5 ** (1.0 / hl)
        w = lam ** np.arange(n_samples - 1, -1, -1)
    n_eff = float(w.sum() ** 2 / (w ** 2).sum())
    thr = V11._sig_threshold(max(10, int(n_eff)))
    vol_causal = np.sqrt(np.average((Xi_full - np.average(Xi_full, axis=1, weights=w, keepdims=True)) ** 2,
                                     axis=1, weights=w))
    cand_idx = np.argsort(-vol_causal)[:V11.BOOST_N_CANDIDATES]
    Xi = Xi_full[cand_idx]
    C = _corrmat_weighted(Xi, Yj, w)
    for j in track_followers:
        col = C[:, j].copy()
        cand_pos = np.where(cand_idx == j)[0]
        if len(cand_pos):
            col[cand_pos[0]] = np.nan
        if np.all(np.isnan(col)):
            out[j] = (-1, False)
            continue
        ci = int(np.nanargmax(np.abs(col)))
        i = cand_idx[ci]
        out[j] = (int(i), bool(abs(col[ci]) > thr))
    return out


def reselect_day(mode, seed, hl, nt_pre=1000, nt_post=2000):
    out, idio, algo_ret, W_new, leaders_new = simulate(nt_pre, nt_post, mode, seed=seed)
    nInst, nt = out.shape
    logp = np.log(out)
    r = logp[:, 1:] - logp[:, :-1]
    rs = r[1:]

    days_found = {j: None for j in followers}
    streak = {j: 0 for j in followers}
    for T in range(nt_pre + 1, nt):
        sel = track_selection(rs[:, :T], hl, list(followers))
        for idx, j in enumerate(followers):
            i_new_true = leaders_new[idx]
            leader, passed = sel[j]
            if leader == i_new_true and passed:
                streak[j] += 1
                if streak[j] >= 10 and days_found[j] is None:
                    days_found[j] = T - 9 - nt_pre
            else:
                streak[j] = 0
    return days_found


if __name__ == "__main__":
    for hl in [None, 1000, 500]:
        print(f"\n=== hl={hl} ===")
        for seed in [123, 124]:
            days = reselect_day("rotate", seed, hl, nt_pre=1000, nt_post=2000)
            found = [d for d in days.values() if d is not None]
            print(f"  seed={seed}: {len(found)}/20 reselected within 2000d, "
                  f"days-post-change (sorted)={sorted(found)}")
