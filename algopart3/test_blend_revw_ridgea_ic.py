"""Sift RIDGE_A, BLEND, REV_W for highest IC (and then verify score + full distribution, since we
just learned IC-optimal != score-optimal for half-lives, and aggregate score can hide a
majority-of-windows loss). Reuses the shipped half-life ensemble (250,500,1000,2000).
"""
import numpy as np, pandas as pd
import SAFE

P = pd.read_csv("prices.txt", sep=r"\s+", header=0)
Praw = P.values.T.astype(float)
nInst, nt = Praw.shape
logp = np.log(Praw)
r = np.diff(logp, axis=1)
T = r.shape[1]
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
NUMTEST = 250
OLD = (500, 750); NEW = (750, nt); end_days = list(range(400, nt + 1, 10))


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


def single_hl_forecast(hl, ridge_a):
    F = np.full((nt, nInst - 1), np.nan)
    for t in range(SAFE.WARMUP, nt):
        rr = r[:, :t]
        B, mx, my = SAFE._ewls_ridge(rr[:, :-1].T, rr[1:, 1:].T, hl, ridge_a)
        pred = my + (rr[:, -1] - mx) @ B
        fi = pred - pred.mean()
        F[t] = fi / (fi.std() + 1e-12)
    return F


def pooled_ic(signal_by_day):
    rows_x = []; rows_y = []
    for t in range(200, nt - 2):
        if np.all(np.isnan(signal_by_day[t + 1])): continue
        rows_x.append(signal_by_day[t + 1]); rows_y.append(r[1:, t + 1])
    X = np.concatenate(rows_x); Y = np.concatenate(rows_y)
    ok = ~np.isnan(X) & ~np.isnan(Y)
    return float(np.corrcoef(X[ok], Y[ok])[0, 1])


HALF_LIVES = (250, 500, 1000, 2000)

print("building shipped half-life ensemble at RIDGE_A=0.1 (reused for BLEND/REV_W sweep) ...")
FCACHE_01 = {hl: single_hl_forecast(hl, 0.1) for hl in HALF_LIVES}
ridge_ens_01 = np.mean([FCACHE_01[hl] for hl in HALF_LIVES], axis=0)
print("done")


def signal_with_blend(ridge_ens, blend, rev_w):
    sig = np.full((nt, nInst - 1), np.nan)
    for k in range(SAFE.WARMUP, nt):
        wz = ridge_ens[k].copy()
        if np.all(np.isnan(wz)): continue
        if blend > 0 and k - rev_w >= 0:
            rr_ = logp[1:, k] - logp[1:, k - rev_w]
            rr_ = rr_ - rr_.mean()
            rv = -rr_ / (rr_.std() + 1e-12)
            wz = (1 - blend) * wz + blend * rv
        sig[k] = wz
    return sig


def build_pos(signal):
    POS = np.zeros((nInst, nt))
    for k in range(SAFE.WARMUP, nt):
        cur = Praw[:, k]; lim = (dlr / cur).astype(int)
        wz = signal[k]
        if np.all(np.isnan(wz)): continue
        POS[1:, k] = np.clip(np.sign(wz) * (dlr[1:] / cur[1:]), -lim[1:], lim[1:])
    return POS


def full_scs(POS):
    return np.array([window(POS, E - NUMTEST, E)["score"] for E in end_days])


base_sig = signal_with_blend(ridge_ens_01, SAFE.BLEND, SAFE.REV_W)
base_POS = build_pos(base_sig)
base_scs = full_scs(base_POS)
wo0 = window(base_POS, *OLD); wn0 = window(base_POS, *NEW)
print(f"\nTRUE shipped (BLEND=0.3,REV_W=10): IC={pooled_ic(base_sig):.4f}  "
      f"OLD={wo0['score']:.1f} NEW={wn0['score']:.1f} rmean={base_scs.mean():.1f} rfloor={base_scs.min():.1f}")

print("\n--- BLEND sweep (REV_W=10 fixed) ---")
for blend in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
    sig = signal_with_blend(ridge_ens_01, blend, SAFE.REV_W)
    POS = build_pos(sig)
    scs = full_scs(POS)
    wo = window(POS, *OLD); wn = window(POS, *NEW)
    nworse = int((scs < base_scs).sum())
    print(f"blend={blend:.1f}: IC={pooled_ic(sig):.4f}  OLD={wo['score']:>7.1f} NEW={wn['score']:>7.1f} "
          f"rmean={scs.mean():>7.1f} rfloor={scs.min():>7.1f}  n_worse={nworse}/{len(scs)}")

print("\n--- REV_W sweep (BLEND=0.3 fixed) ---")
for rev_w in (3, 5, 7, 10, 15, 20, 30, 40, 60):
    sig = signal_with_blend(ridge_ens_01, SAFE.BLEND, rev_w)
    POS = build_pos(sig)
    scs = full_scs(POS)
    wo = window(POS, *OLD); wn = window(POS, *NEW)
    nworse = int((scs < base_scs).sum())
    print(f"rev_w={rev_w:>3}: IC={pooled_ic(sig):.4f}  OLD={wo['score']:>7.1f} NEW={wn['score']:>7.1f} "
          f"rmean={scs.mean():>7.1f} rfloor={scs.min():>7.1f}  n_worse={nworse}/{len(scs)}")

print("\n--- RIDGE_A sweep (shipped half-lives + BLEND=0.3 + REV_W=10, full distribution check) ---")
for ridge_a in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0):
    ens = np.mean([single_hl_forecast(hl, ridge_a) for hl in HALF_LIVES], axis=0)
    sig = signal_with_blend(ens, SAFE.BLEND, SAFE.REV_W)
    POS = build_pos(sig)
    scs = full_scs(POS)
    wo = window(POS, *OLD); wn = window(POS, *NEW)
    nworse = int((scs < base_scs).sum())
    mark = "  <-- shipped" if ridge_a == 0.1 else ""
    print(f"ridge_a={ridge_a:.2f}: IC={pooled_ic(sig):.4f}  OLD={wo['score']:>7.1f} NEW={wn['score']:>7.1f} "
          f"rmean={scs.mean():>7.1f} rfloor={scs.min():>7.1f}  n_worse={nworse}/{len(scs)}{mark}")
