"""Combine the ridge (shipped, strong) with the leader-only GBM (weaker alone, IC=0.0366, but a
DIFFERENT model class -- possibly different errors). First check how correlated their day-to-day
disagreements actually are (if GBM is just a noisier copy of the ridge, combining won't help; if its
errors are meaningfully different, a confirm-gate could). Then test: trade only when both agree in
sign (else flat), same eval-mirroring accounting as all night.
"""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
import SAFE

P = pd.read_csv("prices.txt", sep=r"\s+", header=0)
names = list(P.columns)
Praw = P.values.T.astype(float)
nInst, nt = Praw.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
NUMTEST = 250
logp = np.log(Praw)
r = np.diff(logp, axis=1)
T = r.shape[1]


def score(mu, sd):
    if mu <= 0 or sd < 1e-10: return float(mu)
    sr = np.sqrt(250) * mu / sd
    return float(mu * sr ** 2 / (sr ** 2 + 1.0))


def window(POS, S, E):
    curPos = np.zeros(nInst); comm_vec = np.zeros(nInst); prevCur = None; tot = []
    for tt in range(S, E + 1):
        cur = Praw[:, tt - 1]
        newPos = (POS[:, tt - 1].copy() if tt < E else curPos.copy())
        if tt > S:
            pl = curPos * (cur - prevCur) - comm_vec
            tot.append(float(pl.sum()))
        dP = newPos - curPos
        comm_vec = commRate * np.abs(dP) * cur
        prevCur = cur; curPos = newPos
    tot = np.array(tot)
    return {"mu": float(tot.mean()), "sd": float(tot.std()), "score": score(tot.mean(), tot.std())}


def corrmat(X, Y):
    Xc = X - X.mean(1, keepdims=True); Yc = Y - Y.mean(1, keepdims=True)
    Xs = Xc / (Xc.std(1, keepdims=True) + 1e-12); Ys = Yc / (Yc.std(1, keepdims=True) + 1e-12)
    return (Xs @ Ys.T) / X.shape[1]


CHECKPOINTS_LEADER = list(range(150, T, 50))
LEADER_AT = {}
for cp in CHECKPOINTS_LEADER:
    Xi = r[1:, :cp - 1]; Yj = r[1:, 1:cp]
    n = nInst - 1
    C = corrmat(Xi, Yj)
    leader = {}
    for j in range(n):
        col = C[:, j].copy(); col[j] = np.nan
        i = int(np.nanargmax(np.abs(col))); leader[j + 1] = i + 1
    LEADER_AT[cp] = leader


def leader_for_day(k):
    valid = [c for c in CHECKPOINTS_LEADER if c <= k]
    return LEADER_AT[max(valid)] if valid else {}


print("rebuilding the leader-only panel (same as the tuned version, IC=0.0366) ...")
VOL_W = 20
own_vol = np.full((nInst, T), np.nan)
for j in range(nInst):
    for t in range(VOL_W, T):
        own_vol[j, t] = r[j, t - VOL_W:t].std()
mom5 = np.full((nInst, T), np.nan); mom20 = np.full((nInst, T), np.nan)
for j in range(nInst):
    mom5[j, 5:] = np.array([r[j, t - 5:t].sum() for t in range(5, T)])
    mom20[j, 20:] = np.array([r[j, t - 20:t].sum() for t in range(20, T)])
xs_mean_others = np.array([r[1:, t].mean() for t in range(T)])
cross_rank = np.full((nInst, T), np.nan)
for t in range(T):
    cross_rank[1:, t] = pd.Series(r[1:, t]).rank(pct=True).values
import SAFE_llvol as M
vol0 = np.full(T, np.nan); vol0[M.VOL_WIN - 1:] = M._roll_std(r[0], M.VOL_WIN)
volz0 = np.full(T, np.nan)
for s in range(M.VOL_WIN + M.VOL_Z, T):
    wv = vol0[s - M.VOL_Z:s]; volz0[s] = (vol0[s] - wv.mean()) / (wv.std() + 1e-12)

rows = []
for j in range(1, nInst):
    for t in range(200, T - 1):
        leader = leader_for_day(t)
        li = leader.get(j, 0)
        lret = r[li, t] if li else 0.0
        lret_lag1 = r[li, t - 1] if li else 0.0
        rows.append([j, t, r[j, t], r[j, t - 1], own_vol[j, t], r[0, t], cross_rank[j, t],
                      mom5[j, t], mom20[j, t], xs_mean_others[t], lret, lret_lag1,
                      volz0[t] if not np.isnan(volz0[t]) else 0.0, r[j, t + 1]])
panel = pd.DataFrame(rows, columns=["stock", "t", "own_ret", "own_ret_lag1", "own_vol", "algo_ret",
                                     "xs_rank", "mom5", "mom20", "xs_mean", "leader_ret",
                                     "leader_ret_lag1", "algo_volz", "target"]).dropna()
panel["stock"] = panel["stock"].astype("category")
FEATS = ["stock", "own_ret", "own_ret_lag1", "own_vol", "algo_ret", "xs_rank", "mom5", "mom20",
         "xs_mean", "leader_ret", "leader_ret_lag1", "algo_volz"]

CHECKPOINTS = list(range(300, T - 1, 100))
gbm_sig = {}
for cp in CHECKPOINTS:
    train = panel[panel["t"] < cp]
    test = panel[(panel["t"] >= cp) & (panel["t"] < cp + 100)]
    if len(test) == 0: continue
    gbm = HistGradientBoostingRegressor(max_iter=100, max_depth=3, learning_rate=0.02,
                                         l2_regularization=5.0, categorical_features=["stock"],
                                         random_state=0).fit(train[FEATS], train["target"])
    gp = gbm.predict(test[FEATS])
    for (stk, t), gv in zip(zip(test["stock"], test["t"]), gp):
        gbm_sig[(stk, t)] = gv
print(f"GBM signal computed for {len(gbm_sig)} (stock, day) pairs")

print("\ncomputing SAFE's own wz forecast (causal, per stock per day) ...")
WZ = {}
for t in range(SAFE.WARMUP, nt):
    rr = r[:, :t]
    fs = []
    for hl in SAFE.HALF_LIVES:
        B, mx, my = SAFE._ewls_ridge(rr[:, :-1].T, rr[1:, 1:].T, hl, SAFE.RIDGE_A)
        pred = my + (rr[:, -1] - mx) @ B
        fi = pred - pred.mean()
        fs.append(fi / (fi.std() + 1e-12))
    wz = np.mean(fs, 0)
    if SAFE.BLEND > 0:
        rv_ = logp[1:, t] - logp[1:, t - SAFE.REV_W]
        rv_ = rv_ - rv_.mean()
        rv = -rv_ / (rv_.std() + 1e-12)
        wz = (1 - SAFE.BLEND) * wz + SAFE.BLEND * rv
    WZ[t] = wz
print("done")

# --- correlation / agreement diagnostics ---
first_t = min(t for (_, t) in gbm_sig)
pairs = [(j, t) for (j, t) in gbm_sig if t + 1 in WZ]
ridge_vals = np.array([WZ[t + 1][j - 1] for (j, t) in pairs])
gbm_vals = np.array([gbm_sig[(j, t)] for (j, t) in pairs])
print(f"\ncorr(ridge wz, GBM pred) on the same (stock,day) pairs: {np.corrcoef(ridge_vals, gbm_vals)[0,1]:.3f}")
agree = (np.sign(ridge_vals) == np.sign(gbm_vals))
print(f"sign agreement rate: {agree.mean()*100:.1f}%  ({agree.sum()}/{len(agree)})")

OLD = (500, 750); NEW = (750, nt); end_days = list(range(400, nt + 1, 10))
FULL = (first_t + 2, nt)


def build_pos(mode):
    POS = np.zeros((nInst, nt))
    for k in range(first_t + 2, nt):
        cur = Praw[:, k]; lim = (dlr / cur).astype(int)
        t = k - 1
        if t + 1 not in WZ: continue
        wz = WZ[t + 1]
        for j in range(1, nInst):
            rsign = np.sign(wz[j - 1])
            if mode == "ridge_only":
                s = rsign
            elif mode == "confirm_gate":
                if (j, t) in gbm_sig:
                    gsign = np.sign(gbm_sig[(j, t)])
                    s = rsign if gsign == rsign else 0.0
                else:
                    s = rsign
            POS[j, k] = np.clip(s * (dlr[j] / cur[j]), -lim[j], lim[j])
    return POS


def report(nm, POS):
    wo = window(POS, *OLD); wn = window(POS, *NEW); wf = window(POS, *FULL)
    scs = [window(POS, E - NUMTEST, E)["score"] for E in end_days if E - NUMTEST >= first_t]
    print(f"{nm:<20}FULL={wf['score']:>8.1f}  OLD={wo['score']:>8.1f}  NEW={wn['score']:>8.1f}  "
          f"rmean={np.mean(scs):>8.1f}  rfloor={min(scs):>8.1f}")


print()
report("ridge only", build_pos("ridge_only"))
report("confirm-gate (both agree)", build_pos("confirm_gate"))

# --- additive blend instead of a gate: z-score both signals, sum with a weight, trade the sign ---
print("\n--- additive blend (z-score both, sum with weight w on GBM, trade sign) ---")


def build_pos_blend(w):
    POS = np.zeros((nInst, nt))
    for k in range(first_t + 2, nt):
        cur = Praw[:, k]; lim = (dlr / cur).astype(int)
        t = k - 1
        if t + 1 not in WZ: continue
        wz = WZ[t + 1]
        wz_z = wz / (np.std(wz) + 1e-12)
        for j in range(1, nInst):
            rz = wz_z[j - 1]
            if (j, t) in gbm_sig:
                gz = gbm_sig[(j, t)]   # already roughly standardized in scale from training target units
                combined = (1 - w) * rz + w * gz * (np.std(wz_z) / (np.std(gbm_vals) + 1e-12))
            else:
                combined = rz
            POS[j, k] = np.clip(np.sign(combined) * (dlr[j] / cur[j]), -lim[j], lim[j])
    return POS


for w in (0.1, 0.2, 0.3, 0.4, 0.5):
    report(f"blend w_gbm={w}", build_pos_blend(w))

print("\n--- finer weight sweep ---")
for w in (0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.25):
    report(f"blend w_gbm={w}", build_pos_blend(w))

print("\n--- does GBM agreement actually mark HIGHER-QUALITY ridge trades? (validates/refutes 'confirmation') ---")
agree_mask = np.sign(ridge_vals) == np.sign(gbm_vals)
actual_rets = np.array([r[j, t + 1] for (j, t) in pairs])
ridge_side_pnl = np.sign(ridge_vals) * actual_rets
print(f"ridge-side realized edge when GBM AGREES:    mean={ridge_side_pnl[agree_mask].mean():+.5f}  n={agree_mask.sum()}")
print(f"ridge-side realized edge when GBM DISAGREES: mean={ridge_side_pnl[~agree_mask].mean():+.5f}  n={(~agree_mask).sum()}")

print("\n--- is GBM's edge concentrated in the STRONG-leader pairs (validates lead-lag) vs weak ones? ---")
Xi_full = r[1:, :-1]; Yj_full = r[1:, 1:]
Cfull = corrmat(Xi_full, Yj_full)
best_corr_full = {}
for jj in range(nInst - 1):
    col = Cfull[:, jj].copy(); col[jj] = np.nan
    best_corr_full[jj + 1] = np.nanmax(np.abs(col))
strong_js = {j for j, c in best_corr_full.items() if c > 0.10}
is_strong = np.array([1 if j in strong_js else 0 for (j, t) in pairs])
gbm_ic_strong = np.corrcoef(gbm_vals[is_strong == 1], actual_rets[is_strong == 1])[0, 1]
gbm_ic_weak = np.corrcoef(gbm_vals[is_strong == 0], actual_rets[is_strong == 0])[0, 1]
print(f"GBM IC on STRONG-leader stocks (n={int((is_strong==1).sum())}): {gbm_ic_strong:.4f}")
print(f"GBM IC on WEAK-leader stocks   (n={int((is_strong==0).sum())}): {gbm_ic_weak:.4f}")

# --- OLD vs NEW split of the agree/disagree realized-edge diagnostic (grounds the desize plan) ---
print("\n--- agree/disagree realized edge, split by OLD vs NEW ---")
pairs_arr = np.array(pairs)
for lbl, (s, e) in [("OLD", OLD), ("NEW", NEW)]:
    m = (pairs_arr[:, 1] >= s) & (pairs_arr[:, 1] < e)
    am = agree_mask & m; dm = (~agree_mask) & m
    print(f"  {lbl}: n_agree={am.sum():<6d} edge_agree={ridge_side_pnl[am].mean():+.5f}   "
          f"n_disagree={dm.sum():<6d} edge_disagree={ridge_side_pnl[dm].mean():+.5f}")
print(f"  std of per-trade ridge-side pnl: agree={ridge_side_pnl[agree_mask].std():.5f}  "
      f"disagree={ridge_side_pnl[~agree_mask].std():.5f}")

# --- desize (never flip sign; shrink size on GBM disagreement, keep full size on agreement) ---
print("\n--- desize (never flip sign; shrink size on GBM disagreement, keep full size on agreement) ---")


def build_pos_desize(disagree_frac):
    """Trade sign(ridge) always. Full $10k when GBM agrees in sign; disagree_frac * $10k when GBM
    disagrees (or no GBM signal available -> treat as agree, matching how build_pos/build_pos_blend
    already handle missing gbm_sig entries)."""
    POS = np.zeros((nInst, nt))
    for k in range(first_t + 2, nt):
        cur = Praw[:, k]; lim = (dlr / cur).astype(int)
        t = k - 1
        if t + 1 not in WZ: continue
        wz = WZ[t + 1]
        for j in range(1, nInst):
            rsign = np.sign(wz[j - 1])
            if (j, t) in gbm_sig:
                gsign = np.sign(gbm_sig[(j, t)])
                frac = 1.0 if gsign == rsign else disagree_frac
            else:
                frac = 1.0
            POS[j, k] = np.clip(rsign * frac * (dlr[j] / cur[j]), -lim[j], lim[j])
    return POS


assert np.allclose(build_pos_desize(0.0), build_pos("confirm_gate")), "frac=0 must equal confirm_gate exactly"
assert np.allclose(build_pos_desize(1.0), build_pos("ridge_only")), "frac=1 must equal ridge_only exactly"
print("sanity check passed: desize(0)==confirm_gate, desize(1)==ridge_only")

print("\n--- coarse sweep ---")
report("ridge only (ref)", build_pos("ridge_only"))
for f in (0.15, 0.3, 0.45, 0.6, 0.75, 0.9):
    report(f"desize frac={f}", build_pos_desize(f))

print("\n\n=== NEW IDEA: apply GBM ONLY where there is NO significant pairwise leader today (complementary, ===")
print("    not competing) -- motivated by 'GBM IC on WEAK-leader stocks (0.0405) > STRONG (0.0330)' above.===")
print("    Uses the CAUSAL Bonferroni significance gate from SAFE_llboost.py (not the crude 0.10 full-")
print("    sample threshold used in the diagnostic above), fresh every day, no stale checkpoints.")

from scipy import stats

BOOST_ALPHA = 0.05; BOOST_N_CANDIDATES = 49
BOOST_P = 2.0; BOOST_SCALE_W = 1000; BOOST_IC_L = 190; BOOST_K = 1.5; BOOST_MIN_DAY = 500
rs = r[1:]


def sig_threshold(n_samples):
    if n_samples < 10: return 1.0
    alpha_adj = BOOST_ALPHA / BOOST_N_CANDIDATES
    tcrit = stats.t.ppf(1 - alpha_adj / 2, df=n_samples - 2)
    return float(tcrit / np.sqrt(n_samples - 2 + tcrit ** 2))


print("\nprecomputing causal significance-gate boost (matching SAFE_llboost.py exactly) ...")
import time
t0 = time.time()
BOOST_AT = {}  # day k -> {j: boost_val} for stocks WITH a qualifying significant leader
for k in range(BOOST_MIN_DAY, nt):
    T_ = k
    Xi = rs[:, :T_ - 1]; Yj = rs[:, 1:T_]
    n_samples = Xi.shape[1]
    thr = sig_threshold(n_samples)
    C = corrmat(Xi, Yj)
    entry = {}
    for j in range(nInst - 1):
        col = C[:, j].copy(); col[j] = np.nan
        i = int(np.nanargmax(np.abs(col)))
        if abs(col[i]) <= thr:
            continue
        lead = rs[i, :T_]
        scale = np.nanstd(lead[max(0, T_ - 1 - BOOST_SCALE_W):T_ - 1]) + 1e-12
        lead_boost = np.sign(lead) * (np.abs(lead) / scale) ** BOOST_P
        a = max(0, T_ - 1 - BOOST_IC_L)
        xs = lead_boost[a:T_ - 1]; ys = rs[j, a + 1:T_]
        ok = ~np.isnan(xs) & ~np.isnan(ys)
        if ok.sum() < 60 or xs[ok].std() < 1e-12:
            continue
        ic = float(np.corrcoef(xs[ok], ys[ok])[0, 1])
        if ic <= 0:
            continue
        entry[j] = lead_boost[-1]
    BOOST_AT[k] = entry
print(f"done ({time.time()-t0:.0f}s)")


print("computing ALGO leg (was missing -- caught via full day-by-day comparison against shipped) ...")
algo_pos = np.zeros(nt)
for k in range(130, nt):
    cur0 = Praw[0, k]; lim0 = int(dlr[0] / cur0)
    algo_pos[k] = np.clip(M._algo_vol_shares(logp[0, :k + 1], cur0, dlr[0]), -lim0, lim0)


def build_pos_complementary(mode, gbm_w=0.15):
    """mode='confirm': confirm-gate GBM only on non-boosted stocks (flat if GBM disagrees).
       mode='blend': additive z-blend GBM only on non-boosted stocks.
       Boosted stocks: pure ridge + significance-boost, untouched."""
    POS = np.zeros((nInst, nt))
    for k in range(first_t + 2, nt):
        cur = Praw[:, k]; lim = (dlr / cur).astype(int)
        t = k - 1
        if t + 1 not in WZ: continue
        wz = WZ[t + 1].copy()
        boosted = BOOST_AT.get(k, {}) if k >= BOOST_MIN_DAY else {}
        for j in range(1, nInst):
            if (j - 1) in boosted:
                wz[j - 1] += BOOST_K * boosted[j - 1]
        wz_z = (wz - wz.mean()) / (wz.std() + 1e-12)
        for j in range(1, nInst):
            rsign = np.sign(wz[j - 1])
            if (j - 1) in boosted:
                sig = wz[j - 1]  # untouched: ridge + significance-boost only
            elif (j, t) in gbm_sig:
                gsign = np.sign(gbm_sig[(j, t)])
                if mode == "confirm":
                    sig = wz[j - 1] if gsign == rsign else 0.0
                else:  # blend
                    gz = gbm_sig[(j, t)]
                    sig = (1 - gbm_w) * wz_z[j - 1] + gbm_w * gz * (np.std(wz_z) / (np.std(gbm_vals) + 1e-12))
            else:
                sig = wz[j - 1]
            POS[j, k] = np.clip(np.sign(sig) * (dlr[j] / cur[j]), -lim[j], lim[j])
    POS[0, :] = algo_pos
    return POS


def build_pos_boost_only():
    """reference: ridge + significance-boost, no GBM anywhere (the current shipped mechanism,
    restricted to this file's backtest range for a fair comparison)."""
    POS = np.zeros((nInst, nt))
    for k in range(first_t + 2, nt):
        cur = Praw[:, k]; lim = (dlr / cur).astype(int)
        t = k - 1
        if t + 1 not in WZ: continue
        wz = WZ[t + 1].copy()
        boosted = BOOST_AT.get(k, {}) if k >= BOOST_MIN_DAY else {}
        for j in range(1, nInst):
            if (j - 1) in boosted:
                wz[j - 1] += BOOST_K * boosted[j - 1]
        POS[1:, k] = np.clip(np.sign(wz) * (dlr[1:] / cur[1:]), -lim[1:], lim[1:])
    POS[0, :] = algo_pos
    return POS


def report2(nm, POS, base_scs=None):
    wo = window(POS, *OLD); wn = window(POS, *NEW); wf = window(POS, *FULL)
    scs = np.array([window(POS, E - NUMTEST, E)["score"] for E in end_days if E - NUMTEST >= first_t])
    line = (f"{nm:<24}FULL={wf['score']:>8.1f}  OLD={wo['score']:>8.1f}  NEW={wn['score']:>8.1f}  "
            f"rmean={scs.mean():>8.1f}  rfloor={scs.min():>8.1f}")
    if base_scs is not None:
        nworse = int((scs < base_scs).sum())
        line += f"  n_worse={nworse}/{len(scs)}"
    print(line)
    return scs


print("\n=== reference: ridge + significance-boost, no GBM (current shipped mechanism) ===")
base_scs_c = report2("ridge+boost (ref)", build_pos_boost_only())

print("\n=== complementary GBM: confirm-gate on NON-boosted stocks only ===")
report2("compl. confirm-gate", build_pos_complementary("confirm"), base_scs_c)

print("\n=== complementary GBM: additive blend on NON-boosted stocks only ===")
for w in (0.1, 0.15, 0.2, 0.3, 0.4):
    report2(f"compl. blend w={w}", build_pos_complementary("blend", w), base_scs_c)

print("\n--- sanity check: build_pos_boost_only() must now reproduce the known SAFE_llboost numbers exactly ---")
import SAFE_llboost as SHIPPED
POS_check = build_pos_boost_only()
mismatches = sum(
    not np.array_equal(np.sign(np.asarray(SHIPPED.getMyPosition(Praw[:, :day + 1]))), np.sign(POS_check[:, day]))
    for day in range(500, nt, 25)
)
print(f"sign mismatches on a 25-day-stride spot check (500-999): {mismatches}/{len(range(500, nt, 25))} (must be 0)")
wo = window(POS_check, *OLD); wn = window(POS_check, *NEW)
print(f"build_pos_boost_only() OLD={wo['score']:.1f} NEW={wn['score']:.1f}  "
      f"(must match known SAFE_llboost: OLD=774.1 NEW=828.6)")
