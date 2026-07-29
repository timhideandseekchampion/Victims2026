"""
investigate_rotate_gap.py -- why does v13's gated decayed-selection boost fallback barely help in
the rotate change-point scenario (mean frac_saved only -1.5% -> -0.7%, still net negative), when
rotate mode -- a genuinely NEW leader replacing an old one -- is exactly the failure mode it targets?

Two candidate explanations, checked directly rather than guessed:
  1. The fallback's own GATE rarely opens in rotate mode -- i.e. the full-history path keeps
     "succeeding" (finding SOME significant candidate, even if wrong) so the decayed search never
     gets a turn. Instrumented by monkey-patching `_pairwise_boost` with a counter (same logic as
     SAFE_llboost_v13.py, copied verbatim, not modifying the shipped file) tracking how many of the
     50 followers get filled via the full-history path vs the decayed path vs neither, per day.
  2. Separately: does the momentum/xsac insurance layer (v14 Part B) even engage in rotate mode?
     Already known from test_v14_changepoint.py's own output (non_champ_days 8-89/599 in rotate vs
     232-283/599 in reverse) -- much lower engagement -- consistent with an EARLIER finding in
     algothon-protection-stack memory: the idio book stays net POSITIVE post-change in rotate mode
     (pure opportunity cost, not an active loss), so a PnL-sum "champ sick" trigger barely fires
     there by construction. Not re-derived here, just cited for context in the printed summary.

Run: python3 investigate_rotate_gap.py
"""
import numpy as np
from scipy import stats
from changepoint_synthetic import simulate, W_old
import SAFE_llboost_v13 as V13

NT_PRE, NT_POST = 1000, 600
SEEDS = [123, 124, 125, 126]

_counts = {"full": 0, "decayed": 0, "none": 0, "days": 0}


def _sig_threshold(n_samples):
    if n_samples < 10:
        return 1.0
    alpha_adj = V13.BOOST_ALPHA / V13.BOOST_N_CANDIDATES
    tcrit = stats.t.ppf(1 - alpha_adj / 2, df=n_samples - 2)
    return float(tcrit / np.sqrt(n_samples - 2 + tcrit ** 2))


def _instrumented_pairwise_boost(rs):
    """Verbatim copy of SAFE_llboost_v13._pairwise_boost with a fill-source counter added."""
    n, T = rs.shape
    boost = np.zeros(n)
    if T < V13.BOOST_MIN_DAY:
        return boost
    Xi_full = rs[:, :-1]; Yj = rs[:, 1:]
    n_samples = Xi_full.shape[1]

    thr_full = _sig_threshold(n_samples)
    vol_causal_full = np.nanstd(Xi_full, axis=1)
    cand_idx_full = np.argsort(-vol_causal_full)[:V13.BOOST_N_CANDIDATES]
    Xi_f = Xi_full[cand_idx_full]
    C_full = V13._corrmat(Xi_f, Yj)

    lam = 0.5 ** (1.0 / V13.BOOST_SEL_FALLBACK_HL)
    w = lam ** np.arange(n_samples - 1, -1, -1)
    n_eff = float(w.sum() ** 2 / (w ** 2).sum())
    thr_dec = _sig_threshold(max(10, int(n_eff)))
    mean_dec = np.average(Xi_full, axis=1, weights=w)
    vol_causal_dec = np.sqrt(np.average((Xi_full - mean_dec[:, None]) ** 2, axis=1, weights=w))
    cand_idx_dec = np.argsort(-vol_causal_dec)[:V13.BOOST_N_CANDIDATES]
    Xi_d = Xi_full[cand_idx_dec]
    C_dec = V13._corrmat_weighted(Xi_d, Yj, w)

    _counts["days"] += 1
    for j in range(n):
        filled = False

        col = C_full[:, j].copy()
        cand_pos = np.where(cand_idx_full == j)[0]
        if len(cand_pos):
            col[cand_pos[0]] = np.nan
        if not np.all(np.isnan(col)):
            ci = int(np.nanargmax(np.abs(col)))
            if abs(col[ci]) > thr_full:
                i = cand_idx_full[ci]
                lead_boost, ic = V13._leader_boost_and_ic(rs, i, j, T)
                if lead_boost is not None and ic is not None and ic > 0:
                    boost[j] = lead_boost[-1]
                    filled = True

        if filled:
            _counts["full"] += 1
            continue

        colD = C_dec[:, j].copy()
        cand_posD = np.where(cand_idx_dec == j)[0]
        if len(cand_posD):
            colD[cand_posD[0]] = np.nan
        if np.all(np.isnan(colD)):
            _counts["none"] += 1
            continue
        ciD = int(np.nanargmax(np.abs(colD)))
        if abs(colD[ciD]) <= thr_dec:
            _counts["none"] += 1
            continue
        iD = cand_idx_dec[ciD]
        lead_boost, ic = V13._leader_boost_and_ic(rs, iD, j, T)
        if lead_boost is None or ic is None or ic <= 0:
            _counts["none"] += 1
            continue
        boost[j] = lead_boost[-1]
        _counts["decayed"] += 1
    return boost


V13._pairwise_boost = _instrumented_pairwise_boost


def reset_module_state(mod):
    for name in ("_SIG", "_RET", "_ICD", "_PN"):
        if hasattr(mod, name):
            getattr(mod, name).clear()
    if hasattr(mod, "_PREV_ALGO_SHARES"):
        mod._PREV_ALGO_SHARES = 0
    if hasattr(mod, "_PREV_T"):
        mod._PREV_T = -1
    if hasattr(mod, "_DLR"):
        mod._DLR = None


if __name__ == "__main__":
    print("=== rotate mode: how often does the DECAYED fallback path actually engage? ===")
    for seed in SEEDS:
        _counts["full"] = _counts["decayed"] = _counts["none"] = _counts["days"] = 0
        out, idio, algo_ret, W_new, leaders_new = simulate(NT_PRE, NT_POST, "rotate", seed=seed)
        nInst, nt = out.shape
        reset_module_state(V13)
        for t in range(V13.WARMUP, nt):
            V13.getMyPosition(out[:, :t + 1])
        total = _counts["full"] + _counts["decayed"] + _counts["none"]
        print(f"  seed={seed}: over {_counts['days']} days x 50 followers = {total} follower-days -- "
              f"full-history-filled={_counts['full']} ({100*_counts['full']/total:.1f}%)  "
              f"decayed-filled={_counts['decayed']} ({100*_counts['decayed']/total:.1f}%)  "
              f"unfilled={_counts['none']} ({100*_counts['none']/total:.1f}%)")

    print("\n=== for context: reverse mode, same instrumentation ===")
    for seed in SEEDS:
        _counts["full"] = _counts["decayed"] = _counts["none"] = _counts["days"] = 0
        out, idio, algo_ret, W_new, leaders_new = simulate(NT_PRE, NT_POST, "reverse", seed=seed)
        nInst, nt = out.shape
        reset_module_state(V13)
        for t in range(V13.WARMUP, nt):
            V13.getMyPosition(out[:, :t + 1])
        total = _counts["full"] + _counts["decayed"] + _counts["none"]
        print(f"  seed={seed}: full-history-filled={_counts['full']} ({100*_counts['full']/total:.1f}%)  "
              f"decayed-filled={_counts['decayed']} ({100*_counts['decayed']/total:.1f}%)  "
              f"unfilled={_counts['none']} ({100*_counts['none']/total:.1f}%)")
