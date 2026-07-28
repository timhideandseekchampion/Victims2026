"""
test_v7_leak_diagnostic.py

Not a candidate -- a MEASUREMENT. Before trying anything else on SAFE_llboost_v7, ask where the
score actually goes, so effort is spent on the biggest leak rather than the most interesting idea.
Score is `mu * sr^2/(sr^2+1)`, so there are exactly three ways to raise it: more gross PnL, less
commission, or less variance. This decomposes all three on the shipped v7 book.

  1. GROSS / COMMISSION / NET, and the score with commission set to zero -- a hard upper bound on
     what ANY turnover-reduction idea (deadbands, hysteresis, trade throttling) could ever be worth.
  2. LEG ATTRIBUTION -- PnL is additive across instruments, so zeroing one leg gives an exact split
     of both mean and variance between the $100k ALGO leg and the $500k idio book.
  3. THE SHARPE PENALTY -- how far `sr^2/(sr^2+1)` is below 1, i.e. how much score a variance cut
     buys relative to an equal-percentage mean increase (the elasticity ratio).
  4. NET FACTOR EXPOSURE -- the idio book is `sign(wz)` at full $10k on all 50 names, and the count
     of longs vs shorts is not constrained. In a one-factor market an imbalance of k names is a
     k*$10k naked bet on the common factor with no expected return: pure variance. Measures the
     imbalance and how much of daily PnL variance it explains.
  5. Directly scores the cheapest fix for 4 -- force exactly 25 long / 25 short (the 25 highest wz
     long, 25 lowest short). This keeps full-conviction sign sizing AND full deployment (the two
     principles that have won every sizing test in this repo); it only reallocates the handful of
     names nearest the median, which are the lowest-conviction ones by construction.
"""
import numpy as np, pandas as pd, time
import SAFE_llboost_v7 as V7

P_ = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P_.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
NUMTEST = 250
logp = np.log(P_)
r = np.diff(logp, axis=1)
rs = r[1:]
nIdio = rs.shape[0]
WARMUP, BOOST_MIN_DAY, BOOST_K = V7.WARMUP, V7.BOOST_MIN_DAY, V7.BOOST_K


def score(mu, sd):
    if mu <= 0 or sd < 1e-10: return float(mu)
    sr = np.sqrt(250) * mu / sd
    return float(mu * sr ** 2 / (sr ** 2 + 1.0))


def pnl_series(POS, S, E, comm_on=True):
    curPos = np.zeros(nInst); comm_vec = np.zeros(nInst); prevCur = None
    gross, comm = [], []
    for tt in range(S, E + 1):
        cur = P_[:, tt - 1]
        newPos = (POS[:, tt - 1].copy() if tt < E else curPos.copy())
        if tt > S:
            gross.append(float((curPos * (cur - prevCur)).sum()))
            comm.append(float(comm_vec.sum()))
        dP = newPos - curPos
        comm_vec = (commRate * np.abs(dP) * cur) if comm_on else np.zeros(nInst)
        prevCur = cur; curPos = newPos
    return np.array(gross), np.array(comm)


def wscore(POS, S, E, comm_on=True):
    g, c = pnl_series(POS, S, E, comm_on)
    net = g - c
    return score(net.mean(), net.std())


end_days = list(range(400, nt + 1, 10))
OLD = (500, 750); NEW = (750, nt)
scs_curve = lambda POS: np.array([wscore(POS, E - NUMTEST, E) for E in end_days])

# ---------------- rebuild the shipped v7 book ----------------
print("=== rebuilding v7 (backtest-equivalent) ===", flush=True)
t0 = time.time()
WZ = np.full((nIdio, nt), np.nan)
for t in range(WARMUP, nt):
    rr = r[:, :t]
    fs = []
    for hl in V7.HALF_LIVES:
        B, mx, my = V7._ewls_ridge(rr[:, :-1].T, rr[1:, 1:].T, hl, V7.RIDGE_A)
        pred = my + (rr[:, -1] - mx) @ B
        fi = pred - pred.mean()
        fs.append(fi / (fi.std() + 1e-12))
    wz = np.mean(fs, 0)
    rv_ = logp[1:, t] - logp[1:, t - V7.REV_W]
    rv_ = rv_ - rv_.mean()
    WZ[:, t] = (1 - V7.BLEND) * wz + V7.BLEND * (-rv_ / (rv_.std() + 1e-12))

BOOST = np.zeros((nIdio, nt))
for k in range(BOOST_MIN_DAY, nt):
    BOOST[:, k] = V7._pairwise_boost(rs[:, :k])

algo_pos = np.zeros(nt)
for k in range(130, nt):
    cur0 = P_[0, k]; lim0 = int(dlr[0] / cur0)
    algo_pos[k] = np.clip(V7._algo_vol_shares(logp[0, :k + 1], cur0, dlr[0]), -lim0, lim0)

WZB = WZ.copy()
WZB[:, BOOST_MIN_DAY:] = WZ[:, BOOST_MIN_DAY:] + BOOST_K * BOOST[:, BOOST_MIN_DAY:]


def build(sgn_fn=None):
    POS = np.zeros((nInst, nt))
    for k in range(WARMUP, nt):
        cur = P_[:, k]; lim = (dlr / cur).astype(int)
        s = np.sign(WZB[:, k]) if sgn_fn is None else sgn_fn(WZB[:, k])
        POS[1:, k] = np.clip(s * (dlr[1:] / cur[1:]), -lim[1:], lim[1:])
    POS[0, :] = algo_pos
    return POS


POS = build()
print(f"  done ({time.time()-t0:.0f}s)")
base_scs = scs_curve(POS)
print(f"  v7: OLD={wscore(POS,*OLD):.1f}  NEW={wscore(POS,*NEW):.1f}  "
      f"rmean={base_scs.mean():.1f}  rfloor={base_scs.min():.1f}   (README: 830.3/888.5/876.8/674.4)")

# ==================================================================================================
print("\n" + "=" * 96)
print("1) GROSS / COMMISSION / NET  --  and the score with commission switched off")
print("=" * 96)
for tag, (S, E) in (("OLD 500-750", OLD), ("NEW 750-1000", NEW)):
    g, c = pnl_series(POS, S, E)
    net = g - c
    sc = score(net.mean(), net.std())
    sc0 = score(g.mean(), g.std())
    print(f"  {tag}:  gross/day ${g.mean():7.1f}   commission/day ${c.mean():6.1f} "
          f"({100*c.mean()/g.mean():4.1f}% of gross)   net ${net.mean():7.1f}")
    print(f"{'':14}score {sc:7.1f}   |   zero-commission score {sc0:7.1f}   "
          f"-> ANY turnover idea is capped at +{sc0-sc:.1f}")

# ==================================================================================================
print("\n" + "=" * 96)
print("2) LEG ATTRIBUTION (exact: PnL is additive across instruments)")
print("=" * 96)
POS_algo = POS.copy(); POS_algo[1:, :] = 0.0
POS_idio = POS.copy(); POS_idio[0, :] = 0.0
for tag, (S, E) in (("OLD 500-750", OLD), ("NEW 750-1000", NEW)):
    ga, ca = pnl_series(POS_algo, S, E); na = ga - ca
    gi, ci = pnl_series(POS_idio, S, E); ni = gi - ci
    g, c = pnl_series(POS, S, E); net = g - c
    print(f"  {tag}:")
    print(f"    ALGO leg ($100k cap): mean ${na.mean():7.1f}  sd ${na.std():7.1f}  "
          f"comm ${ca.mean():5.1f}  standalone score {score(na.mean(), na.std()):7.1f}")
    print(f"    idio book ($500k):    mean ${ni.mean():7.1f}  sd ${ni.std():7.1f}  "
          f"comm ${ci.mean():5.1f}  standalone score {score(ni.mean(), ni.std()):7.1f}")
    print(f"    corr(ALGO, idio) = {np.corrcoef(na, ni)[0,1]:+.3f}   combined sd ${net.std():.1f} "
          f"(sum of legs' sds ${na.std()+ni.std():.1f})")

# ==================================================================================================
print("\n" + "=" * 96)
print("3) THE SHARPE PENALTY: how much is a variance cut worth vs a mean increase?")
print("=" * 96)
for tag, (S, E) in (("OLD 500-750", OLD), ("NEW 750-1000", NEW)):
    g, c = pnl_series(POS, S, E); net = g - c
    mu, sd = net.mean(), net.std()
    sr = np.sqrt(250) * mu / sd
    frac = sr ** 2 / (sr ** 2 + 1)
    # d(score)/d(mu) * mu/score  and  d(score)/d(sd) * sd/score  (elasticities)
    e_mu = 1 + 2 * (1 - frac)
    e_sd = -2 * (1 - frac)
    print(f"  {tag}: annSharpe {sr:.2f}  ->  frac = sr^2/(sr^2+1) = {frac:.4f}  "
          f"(score = {100*frac:.1f}% of mean PnL; {mu*(1-frac):.1f}/day given up to the penalty)")
    print(f"{'':14}elasticity: +1% mean = +{e_mu:.2f}% score,  -1% sd = +{abs(e_sd):.2f}% score  "
          f"-> mean is worth {e_mu/max(abs(e_sd),1e-9):.1f}x a variance cut of the same size")

# ==================================================================================================
print("\n" + "=" * 96)
print("4) NET FACTOR EXPOSURE of the sign-sized idio book")
print("=" * 96)
sgn = np.sign(WZB)
n_long = (sgn > 0).sum(0); n_short = (sgn < 0).sum(0)
imb = n_long - n_short
netdol = np.zeros(nt)
for k in range(WARMUP, nt):
    cur = P_[1:, k]; lim = (dlr[1:] / cur).astype(int)
    netdol[k] = float((np.clip(np.sign(WZB[:, k]) * (dlr[1:] / cur), -lim, lim) * cur).sum())
sl = slice(500, nt)
fac = rs.mean(0)          # cross-sectional mean return = the common factor in a one-factor market
print(f"  long/short name imbalance (days 500+): mean {imb[sl].mean():+.1f} names, "
      f"sd {imb[sl].std():.1f}, range [{imb[sl].min():+d}, {imb[sl].max():+d}]")
print(f"  net dollar exposure:      mean ${netdol[sl].mean():+,.0f}, sd ${netdol[sl].std():,.0f}, "
      f"max |${np.abs(netdol[sl]).max():,.0f}| against ${500_000:,} gross")
fac_pnl = netdol[500:nt - 1] * fac[500:nt - 1]
gi, ci = pnl_series(POS_idio, 500, nt); ni = gi - ci
m = min(len(fac_pnl), len(ni))
print(f"  PnL from that naked factor bet: mean ${fac_pnl.mean():+.1f}/day, sd ${fac_pnl.std():.1f}/day")
print(f"  it explains {100*np.corrcoef(fac_pnl[:m], ni[:m])[0,1]**2:.1f}% of idio-book PnL variance "
      f"(corr {np.corrcoef(fac_pnl[:m], ni[:m])[0,1]:+.3f})")

# ==================================================================================================
print("\n" + "=" * 96)
print("5) FIX: force exactly 25 long / 25 short (highest-wz long, lowest-wz short)")
print("=" * 96)


def sgn_balanced(wz):
    order = np.argsort(-wz)
    s = np.empty(len(wz))
    s[order[:len(wz) // 2]] = 1.0
    s[order[len(wz) // 2:]] = -1.0
    return s


POS_bal = build(sgn_balanced)
bal_scs = scs_curve(POS_bal)
flipped = (np.sign(WZB[:, 500:]) != np.array([sgn_balanced(WZB[:, k]) for k in range(500, nt)]).T)
print(f"  changes the sign of {100*flipped.mean():.1f}% of stock-days (only names nearest the median)")
print(f"  v7        : OLD={wscore(POS,*OLD):7.1f}  NEW={wscore(POS,*NEW):7.1f}  "
      f"rmean={base_scs.mean():7.1f}  rfloor={base_scs.min():7.1f}")
print(f"  25/25 bal : OLD={wscore(POS_bal,*OLD):7.1f}  NEW={wscore(POS_bal,*NEW):7.1f}  "
      f"rmean={bal_scs.mean():7.1f}  rfloor={bal_scs.min():7.1f}  "
      f"n_worse={int((bal_scs<base_scs).sum())}/{len(bal_scs)}")
gb, cb = pnl_series(POS_bal, *NEW); nb = gb - cb
g, c = pnl_series(POS, *NEW); nn = g - c
print(f"  NEW window: mean ${nn.mean():.1f} -> ${nb.mean():.1f}   sd ${nn.std():.1f} -> ${nb.std():.1f}")

# soft version: only rebalance when the imbalance is large
print("\n  softer variant -- only rebalance when |imbalance| exceeds a threshold:")
for THR in (4, 6, 8, 10):
    def mk(thr):
        def f(wz):
            s = np.sign(wz)
            if abs(int((s > 0).sum() - (s < 0).sum())) <= thr:
                return s
            return sgn_balanced(wz)
        return f
    Pb = build(mk(THR)); sc = scs_curve(Pb)
    print(f"    THR={THR:<3} OLD={wscore(Pb,*OLD):7.1f}  NEW={wscore(Pb,*NEW):7.1f}  "
          f"rmean={sc.mean():7.1f}  rfloor={sc.min():7.1f}  n_worse={int((sc<base_scs).sum())}/{len(sc)}")
