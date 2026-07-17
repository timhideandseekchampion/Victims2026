"""Basket/triplet cointegration (Johansen) + optimal OU entry/exit bands.

Part A - do 3+-leg baskets cointegrate better than pairs? Johansen trace test
         on candidate triplets; trade the cointegrating vector's stationary combo.
Part B - fit Ornstein-Uhlenbeck to the pair spreads and search the entry/exit/
         stop band grid that maximizes the eval.py score.
All scored with the exact eval.py logic on the last 250 days.
"""
import warnings, itertools
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from common import (prices_array, log_returns, COMM_DEFAULT, COMM_INST0,
                    POSLIM_DEFAULT, POSLIM_INST0, N_TEST_DAYS, section, RESULTS)

P, df, tickers = prices_array()
N, T = P.shape
rets = log_returns(df)
commRate = np.full(N, COMM_DEFAULT); commRate[0] = COMM_INST0
dlrPosLimit = np.full(N, POSLIM_DEFAULT); dlrPosLimit[0] = POSLIM_INST0
IDX = {t: i for i, t in enumerate(tickers)}
FDR_PAIRS = [("AENO","NWIG"),("EORC","NGTE"),("HETT","ULXY"),("SMAH","ILVX"),
             ("HUXZ","ACAC"),("CTGI","EELT")]


def backtest(get_pos, start=T - N_TEST_DAYS, end=T):
    cash = 0.0; curPos = np.zeros(N); totDVol = 0.0; value = 0.0; comm = 0.0; pll = []
    for t in range(start, end + 1):
        hist = P[:, :t]; cur = hist[:, -1]
        if t < end:
            lim = (dlrPosLimit / cur).astype(int)
            newPos = np.clip(get_pos(hist), -lim, lim).astype(int)
        else:
            newPos = np.array(curPos)
        d = newPos - curPos; cash -= cur.dot(d) + comm
        dvol = cur * np.abs(d); comm = np.sum(dvol * commRate); totDVol += dvol.sum()
        curPos = np.array(newPos); pv = curPos.dot(cur)
        todayPL = cash + pv - value; value = cash + pv
        if t > start: pll.append(todayPL)
    pll = np.array(pll); mu, sd = pll.mean(), pll.std()
    sr = np.sqrt(250) * mu / sd if sd > 0 else 0
    score = mu * (sr**2 / (sr**2 + 1)) if (mu > 0 and sd > 1e-10) else mu
    return mu, sr, score, totDVol


# ============================ PART A: BASKETS ============================
section("22A. JOHANSEN BASKET/TRIPLET COINTEGRATION SEARCH")
print("Trace test on triplets. r>=1 with trace>95% crit => a stationary combo exists.")
print("Search: each FDR pair extended by every 3rd instrument (train on days 0..250).\n")
train = P[:, :T - N_TEST_DAYS]
baskets = []
seen = set()
for a, b in FDR_PAIRS:
    ia, ib = IDX[a], IDX[b]
    best = []
    for k in range(N):
        if k in (ia, ib): continue
        trip = tuple(sorted([ia, ib, k]))
        if trip in seen: continue
        seen.add(trip)
        Y = train[[ia, ib, k], :].T
        try:
            jo = coint_johansen(Y, det_order=0, k_ar_diff=1)
            trace0, crit0 = jo.lr1[0], jo.cvt[0, 1]     # r=0 vs r>=1
            trace1, crit1 = jo.lr1[1], jo.cvt[1, 1]     # r<=1 vs r>=2
            rank = int(trace0 > crit0) + int(trace1 > crit1)
            if rank >= 1:
                vec = jo.evec[:, 0]                      # top cointegrating vector
                spread = Y @ vec
                adfp = adfuller(spread, autolag="AIC")[1]
                best.append((tickers[k], trace0, rank, adfp, vec))
        except Exception:
            pass
    best.sort(key=lambda x: -x[1])
    if best:
        k_t, tr, rank, adfp, vec = best[0]
        baskets.append((ia, ib, IDX[k_t], vec))
        print(f"  {a}+{b}+{k_t:<5}: trace(r=0)={tr:5.1f} rank={rank} spread-ADF p={adfp:.4f}")
print(f"\nTriplets with >=1 cointegrating relation found: {len(baskets)}")

def basket_strat(hist, lookback=90, entry=0.7, dollars=6000):
    n, t = hist.shape; pos = np.zeros(n)
    if t < lookback + 2: return pos
    cur = hist[:, -1]
    for ia, ib, ic, vec in baskets:
        Y = hist[[ia, ib, ic], :].T
        spread = Y @ vec
        w = spread[-lookback:]; z = (spread[-1] - w.mean()) / (w.std() + 1e-9)
        if abs(z) > entry:
            # scale so the largest leg trades ~`dollars`
            v = vec / (np.abs(vec).max())
            pos[ia] += -np.sign(z) * v[0] * dollars / cur[ia]
            pos[ib] += -np.sign(z) * v[1] * dollars / cur[ib]
            pos[ic] += -np.sign(z) * v[2] * dollars / cur[ic]
    return pos.astype(int)

print("\nBacktest of the triplet-basket strategy (last 250 days):")
for lb in (60, 90):
    for ez in (0.5, 0.75, 1.0):
        mu, sr, sc, dv = backtest(lambda h, lb=lb, ez=ez: basket_strat(h, lb, ez))
        print(f"  lookback={lb} entry_z={ez}: Sharpe {sr:5.2f}  score {sc:7.2f}  mean ${mu:6.2f}")


# ============================ PART B: OU BANDS ============================
section("22B. ORNSTEIN-UHLENBECK OPTIMAL ENTRY/EXIT/STOP BANDS (pairs)")
print("Fit OU (AR(1)) to each pair spread -> theta, half-life. Then grid-search")
print("entry / exit / stop z-bands to maximize eval.py score.\n")

def fit_ou(spread):
    x = spread[:-1]; y = spread[1:]
    b1, b0 = np.polyfit(x, y, 1)
    theta = -np.log(max(b1, 1e-6)); hl = np.log(2) / theta if theta > 0 else np.inf
    return theta, hl

# report OU params per pair (full-sample, descriptive)
for a, b in FDR_PAIRS:
    ia, ib = IDX[a], IDX[b]
    beta = np.polyfit(P[ib], P[ia], 1)[0]; spread = P[ia] - beta * P[ib]
    theta, hl = fit_ou(spread)
    print(f"  {a}-{b}: OU theta={theta:.3f}  half-life={hl:.1f}d")

def ou_band_strat(hist, lookback=90, entry=1.0, exit_z=0.0, stop=99.0, dollars=8000):
    """position with hysteresis: open at |z|>entry, close when |z|<exit_z,
       kill (flat) when |z|>stop. Uses per-pair state via function attribute."""
    n, t = hist.shape; pos = np.zeros(n)
    if t < lookback + 2: return pos
    st = ou_band_strat.state
    cur = hist[:, -1]
    for a, b in FDR_PAIRS:
        ia, ib = IDX[a], IDX[b]
        beta = np.polyfit(hist[ib, -lookback:], hist[ia, -lookback:], 1)[0]
        spread = hist[ia, :] - beta * hist[ib, :]
        w = spread[-lookback:]; z = (spread[-1] - w.mean()) / (w.std() + 1e-9)
        key = (a, b); cur_state = st.get(key, 0)     # -1,0,+1 (sign of position on ia)
        if cur_state == 0:
            if abs(z) > entry and abs(z) < stop:
                cur_state = -int(np.sign(z))
        else:
            if abs(z) < exit_z or abs(z) > stop:
                cur_state = 0
        st[key] = cur_state
        if cur_state != 0:
            pos[ia] += cur_state * dollars / cur[ia]
            pos[ib] += -cur_state * beta * dollars / cur[ib]
    return pos.astype(int)

print("\nGrid search (entry / exit / stop) — score via eval.py:")
print(f"{'entry':>6}{'exit':>6}{'stop':>6}{'Sharpe':>8}{'Score':>9}")
best = (None, -1e9)
for entry in (0.75, 1.0, 1.5, 2.0):
    for exit_z in (0.0, 0.25, 0.5):
        for stop in (3.0, 4.0, 99.0):
            ou_band_strat.state = {}
            mu, sr, sc, dv = backtest(lambda h, e=entry, x=exit_z, s=stop:
                                      ou_band_strat(h, 90, e, x, s))
            if sc > best[1]:
                best = ((entry, exit_z, stop), sc, sr, mu)
            if stop == 99.0:  # print the no-stop slice
                print(f"{entry:>6.2f}{exit_z:>6.2f}{'none':>6}{sr:>8.2f}{sc:>9.2f}")
print(f"\nBEST OU bands: entry={best[0][0]} exit={best[0][1]} stop={best[0][2]}")
print(f"  -> Sharpe {best[2]:.2f}  score {best[1]:.2f}  mean ${best[3]:.2f}")

section("22C. VERDICT")
print("Baskets vs pairs: see 22A scores. OU hysteresis bands vs always-in |z|>entry:")
print(f"  best OU-band score {best[1]:.2f} (entry {best[0][0]}, exit {best[0][1]}, stop {best[0][2]})")
