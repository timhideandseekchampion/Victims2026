"""Sift through the ridge's OWN parameters (HALF_LIVES ensemble composition, RIDGE_A, BLEND, REV_W)
to find the highest achievable pooled cross-sectional IC -- not score, IC specifically, since we just
found IC and score don't always move together (GBM had similar IC to ridge but wildly different
score). Precompute each CANDIDATE half-life's own (non-ensembled) forecast ONCE, then mix-and-match
subsets cheaply to test many ensemble compositions without refitting the ridge from scratch each time.
"""
import numpy as np, pandas as pd
import SAFE

P = pd.read_csv("prices.txt", sep=r"\s+", header=0)
Praw = P.values.T.astype(float)
nInst, nt = Praw.shape
logp = np.log(Praw)
r = np.diff(logp, axis=1)
T = r.shape[1]

CAND_HALF_LIVES = (60, 100, 150, 250, 375, 500, 750, 1000, 1500, 2000, 3000, 5000)
RIDGE_A = 0.1


def single_hl_forecast(hl, ridge_a=RIDGE_A):
    """Per-day, per-stock z-scored ridge forecast at ONE half-life (no ensembling, no blend)."""
    F = np.full((nt, nInst - 1), np.nan)  # index t -> forecast for r[1:, t]
    for t in range(SAFE.WARMUP, nt):
        rr = r[:, :t]
        B, mx, my = SAFE._ewls_ridge(rr[:, :-1].T, rr[1:, 1:].T, hl, ridge_a)
        pred = my + (rr[:, -1] - mx) @ B
        fi = pred - pred.mean()
        F[t] = fi / (fi.std() + 1e-12)
    return F


print("precomputing single-half-life forecasts (one ridge fit per day per half-life) ...")
FCACHE = {}
for hl in CAND_HALF_LIVES:
    FCACHE[hl] = single_hl_forecast(hl)
    print(f"  hl={hl} done")


def pooled_ic(F):
    """F: (nt-1, 50) forecast-for-day-t array. Pool all (stock,day) pairs, corr vs actual r[j,t]."""
    rows_x = []; rows_y = []
    for t in range(200, nt - 2):
        if np.all(np.isnan(F[t + 1])): continue
        rows_x.append(F[t + 1]); rows_y.append(r[1:, t + 1])
    X = np.concatenate(rows_x); Y = np.concatenate(rows_y)
    ok = ~np.isnan(X) & ~np.isnan(Y)
    return float(np.corrcoef(X[ok], Y[ok])[0, 1]), int(ok.sum())


print("\n--- standalone IC per half-life (no blend, no ensemble) ---")
for hl in CAND_HALF_LIVES:
    ic, n = pooled_ic(FCACHE[hl])
    print(f"  hl={hl:>5}: IC={ic:.4f}  (n={n})")

print("\n--- shipped ensemble (250,500,1000,2000) for reference ---")
ship = np.mean([FCACHE[hl] for hl in (250, 500, 1000, 2000)], axis=0)
ic_ship, _ = pooled_ic(ship)
print(f"  ensemble(250,500,1000,2000): IC={ic_ship:.4f}")

print("\n--- trying other ensemble compositions ---")
import itertools
candidates = [
    (250, 500, 1000, 2000),
    (150, 375, 750, 1500),
    (100, 250, 500, 1000),
    (250, 500, 1000, 3000),
    (250, 500, 1500, 3000),
    (150, 500, 1000, 2000),
    (60, 250, 1000, 3000),
    (250, 1000),
    (500, 2000),
    (250, 500, 1000, 2000, 3000),
    (150, 250, 500, 1000, 2000),
    (1000, 2000, 3000, 5000),
    (60, 100, 150),
    (2000, 3000, 5000),
]
results = []
for combo in candidates:
    ens = np.mean([FCACHE[hl] for hl in combo], axis=0)
    ic, _ = pooled_ic(ens)
    results.append((ic, combo))
    print(f"  {str(combo):<32} IC={ic:.4f}")
results.sort(key=lambda x: -x[0])
print(f"\nbest ensemble found: {results[0][1]}  IC={results[0][0]:.4f}  (shipped: {ic_ship:.4f})")

print("\n--- does the higher-IC ensemble actually score better? (test for score, not just IC) ---")
import SAFE as SAFEMOD

commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
NUMTEST = 250


def score_fn(mu, sd):
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
    return {"mu": float(tot.mean()), "sd": float(tot.std()), "score": score_fn(tot.mean(), tot.std())}


def build_pos_from_ensemble(combo):
    ens = np.mean([FCACHE[hl] for hl in combo], axis=0)  # (nt, 50), index t = forecast for day t
    POS = np.zeros((nInst, nt))
    for k in range(SAFE.WARMUP, nt):
        cur = Praw[:, k]; lim = (dlr / cur).astype(int)
        wz = ens[k]
        if np.all(np.isnan(wz)): continue
        POS[1:, k] = np.clip(np.sign(wz) * (dlr[1:] / cur[1:]), -lim[1:], lim[1:])
    return POS


OLD = (500, 750); NEW = (750, nt); end_days = list(range(400, nt + 1, 10))


def report(nm, POS):
    wo = window(POS, *OLD); wn = window(POS, *NEW)
    scs = [window(POS, E - NUMTEST, E)["score"] for E in end_days]
    print(f"{nm:<32}OLD={wo['score']:>8.1f}  NEW={wn['score']:>8.1f}  "
          f"rmean={np.mean(scs):>8.1f}  rfloor={min(scs):>8.1f}")


report("shipped (250,500,1000,2000)", build_pos_from_ensemble((250, 500, 1000, 2000)))
report("best-IC (1000,2000,3000,5000)", build_pos_from_ensemble((1000, 2000, 3000, 5000)))
report("pure long hl=2000 alone", build_pos_from_ensemble((2000,)))
report("pure long hl=5000 alone", build_pos_from_ensemble((5000,)))

print("\n--- now WITH the shipped BLEND=0.3 reversion component added back, vs the TRUE shipped baseline ---")


def build_pos_full(combo, blend=SAFE.BLEND, rev_w=SAFE.REV_W):
    ens = np.mean([FCACHE[hl] for hl in combo], axis=0)  # (nt, 50)
    POS = np.zeros((nInst, nt))
    for k in range(SAFE.WARMUP, nt):
        cur = Praw[:, k]; lim = (dlr / cur).astype(int)
        wz = ens[k].copy()
        if np.all(np.isnan(wz)): continue
        if blend > 0 and k - rev_w >= 0:
            rr_ = logp[1:, k] - logp[1:, k - rev_w]
            rr_ = rr_ - rr_.mean()
            rv = -rr_ / (rr_.std() + 1e-12)
            wz = (1 - blend) * wz + blend * rv
        POS[1:, k] = np.clip(np.sign(wz) * (dlr[1:] / cur[1:]), -lim[1:], lim[1:])
    return POS


report("TRUE shipped (ridge+blend)", build_pos_full((250, 500, 1000, 2000)))
report("best-IC hl + shipped blend", build_pos_full((1000, 2000, 3000, 5000)))
report("hl=2000 alone + shipped blend", build_pos_full((2000,)))
report("hl=5000 alone + shipped blend", build_pos_full((5000,)))

print("\n--- sweeping single very-long half-lives (+ shipped blend), full rolling-window comparison ---")


def full_scs_curve(POS):
    return np.array([window(POS, E - NUMTEST, E)["score"] for E in end_days])


base_POS = build_pos_full((250, 500, 1000, 2000))
base_scs = full_scs_curve(base_POS)
wo0 = window(base_POS, *OLD); wn0 = window(base_POS, *NEW)
print(f"TRUE shipped: OLD={wo0['score']:.1f} NEW={wn0['score']:.1f} rmean={base_scs.mean():.1f} rfloor={base_scs.min():.1f}")

for hl in (2000, 2500, 3000, 4000, 5000, 6000, 8000, 10000):
    if hl not in FCACHE:
        FCACHE[hl] = single_hl_forecast(hl)
    POS = build_pos_full((hl,))
    scs = full_scs_curve(POS)
    wo = window(POS, *OLD); wn = window(POS, *NEW)
    nworse = int((scs < base_scs).sum())
    print(f"hl={hl:>6} alone: OLD={wo['score']:>7.1f} NEW={wn['score']:>7.1f} "
          f"rmean={scs.mean():>7.1f} rfloor={scs.min():>7.1f}  n_worse_vs_shipped={nworse}/{len(scs)}")
