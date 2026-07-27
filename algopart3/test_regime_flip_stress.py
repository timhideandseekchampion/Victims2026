"""Stress test: SAFE_llvol.py's own docstring flags that 'high vol -> higher next return' is the
OPPOSITE of real markets (leverage effect) and is 'plausibly a synthetic-generator artifact.' The
adaptive switch (VOL_MODE='switch', sign = trailing IC) is the safety net for that risk - but it is
not instant. IC_FAST=90 (equal-weight) blended with a fast EW-IC (half-lives 20/45) needs enough
trailing days of the WRONG regime to accumulate before it flips sign.

This builds a synthetic continuation of ALGO's price path where the vol->return relationship is
DELIBERATELY REVERSED (elevated vol -> negative next return, i.e. the real-market leverage-effect
sign) appended after real history, and runs the actual _algo_vol_shares() function from
SAFE_llvol.py day-by-day across the transition to measure:
  (a) how many days it takes the switch to flip from the legacy (+) side to the new (-) side,
  (b) how much $ the leg loses during that lag vs an oracle that flips on day 1 of the regime change.
"""
import numpy as np, pandas as pd
import SAFE_llvol as M

np.random.seed(7)
P = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
lpA_real = np.log(P[0])                       # real ALGO log-price history (1000 days)
cur0_ref = P[0, -1]

BETA = 0.35                                   # generative strength of the (reversed) vol->return effect
BASE_SIG = 0.012                              # baseline daily vol (~ALGO's historical scale)
N_SYN = 400                                   # synthetic days appended

def simulate_reversed(lpA_hist, n_syn):
    lp = list(lpA_hist)
    logsig = np.log(BASE_SIG)
    for _ in range(n_syn):
        logsig = 0.97 * logsig + 0.03 * np.log(BASE_SIG) + np.random.normal(0, 0.06)
        sig = np.exp(logsig)
        r_recent = np.diff(lp[-80:])
        vol20 = np.array([r_recent[max(0, i - 20):i].std() for i in range(20, len(r_recent))])
        if len(vol20) >= 60:
            z = (vol20[-1] - vol20[-60:].mean()) / (vol20[-60:].std() + 1e-12)
        else:
            z = 0.0
        mu = -BETA * np.clip(z, -3, 3) * sig               # REVERSED: elevated vol -> negative drift
        ret = mu + np.random.normal(0, sig)
        lp.append(lp[-1] + ret)
    return np.array(lp)

lpA_syn = simulate_reversed(lpA_real, N_SYN)
T0 = len(lpA_real)

# empirical check: what IC did this generative process actually produce, using the SAME rolling
# vol-z definition the strategy itself uses (VOL_WIN=20, VOL_Z=60), over the synthetic segment only
r_full = np.diff(lpA_syn)
vol_full = np.full(len(lpA_syn), np.nan)
vol_full[20:] = M._roll_std(r_full, 20)
z_full = np.full(len(lpA_syn), np.nan)
for s in range(80, len(lpA_syn)):
    w = vol_full[s - 60:s]
    z_full[s] = (vol_full[s] - w.mean()) / (w.std() + 1e-12)
ret1_full = np.full(len(lpA_syn), np.nan); ret1_full[:-1] = lpA_syn[1:] - lpA_syn[:-1]
xs = z_full[T0:-1]; ys = ret1_full[T0:-1]
ok = ~np.isnan(xs) & ~np.isnan(ys)
emp_ic = np.corrcoef(xs[ok], ys[ok])[0, 1]
print(f"synthetic segment empirical vol->return IC ~ {emp_ic:.3f} (target: clearly negative)")

cap_dol = 100_000.0
shares = []
for t in range(T0, len(lpA_syn)):
    s = M._algo_vol_shares(lpA_syn[:t + 1], np.exp(lpA_syn[t]), cap_dol)
    shares.append(s)
shares = np.array(shares)
cur_syn = np.exp(lpA_syn[T0:])
daily_ret_dol = shares[:-1] * (cur_syn[1:] - cur_syn[:-1])

# oracle: sign each day should be -1 (since generative mu is always -BETA*z*sig, negative-correlated to z)
# approximate oracle position: full cap in the TRUE-regime direction, sized like SWITCH_GAIN*|z|/3
first_correct = None
for i, s in enumerate(shares):
    if s < 0:                      # position is finally on the "reversed regime" (short) side
        first_correct = i
        break
print(f"days into the reversed regime before the leg first goes NET SHORT: "
      f"{first_correct if first_correct is not None else 'never (within ' + str(N_SYN) + ' days)'}")

# lag cost: cumulative $ PnL of the ACTUAL leg vs a same-magnitude oracle that is always short from day 0
oracle_shares = -np.abs(shares)               # same sizing/magnitude schedule, but always correctly-signed
actual_pnl = np.cumsum(daily_ret_dol)
oracle_pnl = np.cumsum(oracle_shares[:-1] * (cur_syn[1:] - cur_syn[:-1]))
print(f"leg PnL over {N_SYN} synthetic days: actual ${actual_pnl[-1]:,.0f}  vs same-|size| oracle ${oracle_pnl[-1]:,.0f}")
print(f"day-by-day actual PnL, first 60 days of the new regime:")
window = 60
cum = np.cumsum(daily_ret_dol[:window])
for d in (10, 20, 30, 45, 60):
    if d <= len(cum):
        print(f"  day {d:>3}: cum actual PnL ${cum[d-1]:>9,.0f}   sign(leg)={'short' if shares[d-1]<0 else 'long' if shares[d-1]>0 else 'flat'}")
