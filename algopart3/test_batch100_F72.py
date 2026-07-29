"""
test_batch100_F72.py

F72: per-name KURTOSIS (4th moment) as a signal/tilt, distinct from the already-rejected
skewness signal (test_skewness_signal.py) which tested per-stock return SKEWNESS as a predictor
of that stock's OWN next-day return (raw, |skew|->|return|, and self-relative z-scored forms; all
rejected). Same rigor bar and same three hypotheses, translated to excess kurtosis (Fisher, tail
heaviness / "fat-tailedness", 0 for a normal distribution):

  H1: raw kurtosis(t) -> own next-day return r(t+1), pooled across all 50 idio names.
  H2: |kurtosis| (tail-heaviness MAGNITUDE) -> next-day |return| (a vol-proxy / clustering test:
      do currently fat-tailed names have bigger next-day moves?).
  H3: kurtosis z-scored relative to the name's OWN trailing kurtosis history -> next-day return.

Stage 1 (this file, primary): pooled IC + circular-shift-per-stock permutation test + H1/H2
persistence split, IDENTICAL methodology to test_skewness_signal.py -- a quick IC-based
pre-check, per instructions for a new-signal idea, before deciding whether a full wz-blend
candidate test is warranted.

Stage 2 (only run if Stage 1 clears significance): build the strongest form found into an
ADDITIVE tilt on wz using the same RS_WEIGHT-style blend mechanic as v10's shipped rank-stability
signal and F71's decile overlay (REQUIRED because v10's idio positions are pure sign(wz) -- a
tilt that never flips a sign near wz~0 has zero effect, per the README's dispersion-signal
caveat), and score it against v10 exactly like every other candidate in this batch.
"""
import numpy as np, pandas as pd, time
import SAFE_llboost_v10 as V10

P = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P.shape
logp = np.log(P)
r = np.diff(logp, axis=1)
ridio = r[1:]  # (50, T)
n, T = ridio.shape

KURT_W = 60


def rolling_kurt(x, w):
    out = np.full(len(x), np.nan)
    for t in range(w, len(x)):
        seg = x[t - w:t]
        m = seg.mean(); s = seg.std()
        if s < 1e-12:
            continue
        out[t] = float(np.mean(((seg - m) / s) ** 4) - 3.0)
    return out


print("=== building causal rolling excess kurtosis per stock (60-day trailing window) ===")
kurt = np.full((n, T), np.nan)
t0 = time.time()
for j in range(n):
    kurt[j] = rolling_kurt(ridio[j], KURT_W)
print(f"done ({time.time()-t0:.0f}s). pooled mean kurt={np.nanmean(kurt):.3f}  std={np.nanstd(kurt):.3f}"
      f"  min={np.nanmin(kurt):.2f}  max={np.nanmax(kurt):.2f}")


def pooled_ic_perm(feat, target, label, n_perm=300):
    rng = np.random.default_rng(0)
    rows_x, rows_y = [], []
    for t in range(T - 1):
        fx = feat[:, t]; fy = target[:, t + 1]
        ok = ~np.isnan(fx) & ~np.isnan(fy)
        if ok.sum() == 0: continue
        rows_x.append(fx[ok]); rows_y.append(fy[ok])
    X = np.concatenate(rows_x); Y = np.concatenate(rows_y)
    ic = float(np.corrcoef(X, Y)[0, 1])
    half_t = T // 2

    def sub_ic(t0_, t1_):
        rx, ry = [], []
        for t in range(t0_, min(t1_, T - 1)):
            fx = feat[:, t]; fy = target[:, t + 1]
            ok = ~np.isnan(fx) & ~np.isnan(fy)
            if ok.sum() == 0: continue
            rx.append(fx[ok]); ry.append(fy[ok])
        xs = np.concatenate(rx); ys = np.concatenate(ry)
        return float(np.corrcoef(xs, ys)[0, 1])

    ic1 = sub_ic(0, half_t); ic2 = sub_ic(half_t, T)
    perm_ics = np.empty(n_perm)
    for p in range(n_perm):
        feat_shift = np.empty_like(feat)
        for j in range(n):
            shift = rng.integers(1, T - 1)
            feat_shift[j] = np.roll(feat[j], shift)
        rx, ry = [], []
        for t in range(T - 1):
            fx = feat_shift[:, t]; fy = target[:, t + 1]
            ok = ~np.isnan(fx) & ~np.isnan(fy)
            if ok.sum() == 0: continue
            rx.append(fx[ok]); ry.append(fy[ok])
        xs = np.concatenate(rx); ys = np.concatenate(ry)
        perm_ics[p] = np.corrcoef(xs, ys)[0, 1]
    pval = float((np.abs(perm_ics) >= abs(ic)).mean())
    print(f"{label}: IC={ic:+.4f}  H1={ic1:+.4f}  H2={ic2:+.4f}  perm p={pval:.3f}  perm_std={perm_ics.std():.4f}")
    return ic, pval


print("\n=== H1: raw kurtosis(t) -> own next-day return (pooled across all 50 idio stocks) ===")
ic1, p1 = pooled_ic_perm(kurt, ridio, "  kurt -> r(t+1)")

print("\n=== H2: |kurtosis| (tail-heaviness MAGNITUDE) -> next-day |return| (vol/clustering proxy) ===")
abs_kurt = np.abs(kurt)
abs_ret = np.abs(ridio)
ic2, p2 = pooled_ic_perm(abs_kurt, abs_ret, "  |kurt| -> |r(t+1)|")

print("\n=== H3: kurtosis z-scored (relative to the stock's OWN trailing kurtosis history) -> next-day return ===")
KURT_Z_W = 120
kurtz = np.full((n, T), np.nan)
for j in range(n):
    for t in range(KURT_W + KURT_Z_W, T):
        w = kurt[j, t - KURT_Z_W:t]
        ok = ~np.isnan(w)
        if ok.sum() > 30:
            kurtz[j, t] = (kurt[j, t] - w[ok].mean()) / (w[ok].std() + 1e-12)
ic3, p3 = pooled_ic_perm(kurtz, ridio, "  kurtz -> r(t+1)")

SIG_THRESH_P = 0.05
any_significant = any(p < SIG_THRESH_P for p in (p1, p2, p3))
print(f"\n=== Stage 1 verdict: significant (perm p<{SIG_THRESH_P})? H1={p1<SIG_THRESH_P} "
      f"H2={p2<SIG_THRESH_P} H3={p3<SIG_THRESH_P} ===")

if not any_significant:
    print("\nNone of the three forms clear permutation significance -- stopping at Stage 1, no "
          "Stage 2 candidate blend test run (consistent with the skewness signal's own rejection "
          "and the time budget for this batch).")
else:
    print("\n>>> At least one form cleared Stage 1 significance -- proceeding to Stage 2 full "
          "candidate blend test using the strongest/most significant form found. <<<")

    # ---------------------------------------------------------------------------------
    # Stage 2: full v10 precompute + additive RS-style blend tilt using the best kurtosis form
    # ---------------------------------------------------------------------------------
    best_form = min([("raw", p1, kurt), ("absmag", p2, abs_kurt), ("zrel", p3, kurtz)],
                     key=lambda x: x[1])
    form_name, form_p, FEAT = best_form
    print(f"Best form by permutation p-value: {form_name} (p={form_p:.3f})")

    P_ = P
    commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
    dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
    NUMTEST = 250
    rs = r[1:]
    nIdio = rs.shape[0]
    WARMUP, BOOST_MIN_DAY, BOOST_K = V10.WARMUP, V10.BOOST_MIN_DAY, V10.BOOST_K
    RIDGE_A, HALF_LIVES = V10.RIDGE_A, V10.HALF_LIVES
    RS_SHORT_W, RS_LONG_W, RS_WEIGHT = V10.RS_SHORT_W, V10.RS_LONG_W, V10.RS_WEIGHT

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

    print("\n=== precompute: full v10 idio WZ, verbatim ===", flush=True)
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

    algo_pos = np.zeros(nt)
    for k in range(130, nt):
        cur0 = P_[0, k]; lim0 = int(dlr[0] / cur0)
        algo_pos[k] = np.clip(V10._algo_vol_shares(logp[0, :k + 1], cur0, dlr[0]), -lim0, lim0)

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
        if t >= max(RS_SHORT_W, RS_LONG_W) + 5:
            short_ret = logp[1:, t] - logp[1:, t - RS_SHORT_W]
            long_ret = logp[1:, t] - logp[1:, t - RS_LONG_W]
            sz = short_ret - short_ret.mean(); sstd = sz.std()
            lz = long_ret - long_ret.mean(); lstd = lz.std()
            if sstd > 1e-12 and lstd > 1e-12:
                sz = sz / sstd; lz = lz / lstd
                disagree = np.sign(lz) != np.sign(sz)
                rs_sig = np.where(disagree, -sz, 0.0)
                s_std = rs_sig.std()
                s_z = (rs_sig - rs_sig.mean()) / (s_std + 1e-12) if s_std > 1e-12 else np.zeros(nIdio)
                wz = (1 - RS_WEIGHT) * wz + RS_WEIGHT * s_z * (np.abs(wz).mean() + 1e-12)
        WZ_V10[:, t] = wz
    print(f"  done ({time.time()-t0:.0f}s)", flush=True)

    def build_pos_baseline():
        POS = np.zeros((nInst, nt))
        for t in days:
            wz = WZ_V10[:, t]
            cur = P_[:, t]; lim = (dlr[1:] / cur[1:]).astype(int)
            POS[1:, t] = np.clip(np.sign(wz) * (dlr[1:] / cur[1:]), -lim, lim)
        POS[0, :] = algo_pos
        return POS

    print("\n=== sanity check ===")
    POS_base = build_pos_baseline()
    base_scs = scs_curve(POS_base)
    base_wo, base_wn = wscore(POS_base, *OLD), wscore(POS_base, *NEW)
    print(f"  baseline: OLD={base_wo:.1f}  NEW={base_wn:.1f}  rmean={base_scs.mean():.1f}  "
          f"rfloor={base_scs.min():.1f}   (v10 docstring: 871.0/912.6/909.8/709.7)")
    SANITY_OK = abs(base_wo - 871.0) < 0.5 and abs(base_wn - 912.6) < 0.5
    print("  OK." if SANITY_OK else "  *** WARNING: sanity check FAILED ***")

    if SANITY_OK:
        def build_pos_tilt(weight, sign_flip=1.0):
            POS = np.zeros((nInst, nt))
            for t in days:
                wz = WZ_V10[:, t].copy()
                f = FEAT[:, t] * sign_flip
                if np.isfinite(f).all():
                    fstd = f.std()
                    f_z = (f - f.mean()) / (fstd + 1e-12) if fstd > 1e-12 else np.zeros(nIdio)
                    wz = (1 - weight) * wz + weight * f_z * (np.abs(wz).mean() + 1e-12)
                cur = P_[:, t]; lim = (dlr[1:] / cur[1:]).astype(int)
                POS[1:, t] = np.clip(np.sign(wz) * (dlr[1:] / cur[1:]), -lim, lim)
            POS[0, :] = algo_pos
            return POS

        print("\n=== Stage 2 sweep: WEIGHT x sign(IC) for the best kurtosis form ===")
        sflip = 1.0 if ic1 >= 0 or form_name != "raw" else -1.0
        results = []
        for w in (0.01, 0.02, 0.05, 0.1):
            for sf in (1.0, -1.0):
                Pz = build_pos_tilt(w, sf); scs = scs_curve(Pz)
                wo = wscore(Pz, *OLD); wn = wscore(Pz, *NEW)
                passed = (wo > base_wo) and (wn > base_wn) and (scs.mean() > base_scs.mean())
                nworse = int((scs < base_scs).sum())
                tag = "  <== PASS" if passed else ""
                print(f"  weight={w},sign={sf:+.0f}  OLD={wo:7.1f}  NEW={wn:7.1f}  "
                      f"rmean={scs.mean():7.1f}  rfloor={scs.min():7.1f}  n_worse={nworse}/61{tag}")
                results.append(dict(w=w, sf=sf, wo=wo, wn=wn, rm=scs.mean(), rf=scs.min(),
                                     nworse=nworse, passed=passed))
        npass = sum(1 for c in results if c["passed"])
        best = max(results, key=lambda c: c["rm"])
        print(f"\n{npass}/{len(results)} configs pass. Best by rmean: weight={best['w']},sign={best['sf']:+.0f} "
              f"rmean={best['rm']:.1f} vs v10={base_scs.mean():.1f}")
