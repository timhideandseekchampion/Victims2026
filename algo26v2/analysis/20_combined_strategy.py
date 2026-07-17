"""COMBINED competition strategy from all real edges, tuned for the score fn.

Edges combined (only the ones that actually backtest positive):
  E1  Rolling cointegration pairs  (primary; Sharpe ~4-6)     -> stat-arb spreads
  E2  Correlation-vs-ALGO residual reversion (secondary ~1.7) -> factor stat-arb
  (dropped: dist-fit, momentum, directional lead-lag = no/negative edge)

Both are market-neutral mean-reversion, so they diversify. ALGO (cheap comm,
10x limit) carries the net factor hedge. Score = mean*SR^2/(SR^2+1): with high
SR the fraction saturates, so we push mean PnL (more pairs, bigger size) while
keeping SR up.

Backtested with the exact eval.py logic; also a strict train/test OOS check.
"""
import warnings, itertools
warnings.filterwarnings("ignore")
import numpy as np
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm
from common import (prices_array, COMM_DEFAULT, COMM_INST0, POSLIM_DEFAULT,
                    POSLIM_INST0, N_TEST_DAYS, section)

P, df, tickers = prices_array()
N, T = P.shape
commRate = np.full(N, COMM_DEFAULT); commRate[0] = COMM_INST0
dlrPosLimit = np.full(N, POSLIM_DEFAULT); dlrPosLimit[0] = POSLIM_INST0
ALGO = 0

_pair_cache = {"day": -10**9, "pairs": []}


def select_pairs(hist, max_pairs, sel_window, pmax):
    n, t = hist.shape
    win = hist[:, -min(t, sel_window):]
    lw = np.log(win)
    # pre-rank candidate pairs by correlation to cut coint calls
    rr = np.diff(lw, axis=1)
    Cc = np.corrcoef(rr)
    cand = []
    for i in range(n):
        for j in range(i + 1, n):
            if abs(Cc[i, j]) > 0.4:
                cand.append((i, j))
    out = []
    for i, j in cand:
        try:
            p = coint(win[i], win[j])[1]
            if p < pmax:
                beta = np.polyfit(win[j], win[i], 1)[0]
                out.append((i, j, p, beta))
        except Exception:
            pass
    out.sort(key=lambda x: x[2])
    return out[:max_pairs]


def combined(hist, *, max_pairs=16, sel_window=250, reselect=25, pmax=0.05,
             pair_lb=90, pair_entry=0.7, pair_dollars=7000,
             use_overlay=True, ov_lb=90, ov_entry=0.9, ov_dollars=3500):
    n, t = hist.shape
    pos = np.zeros(n)
    if t < max(pair_lb, ov_lb) + 5:
        return pos.astype(int)
    cur = hist[:, -1]

    # ---- E1: rolling cointegration pairs ----
    if t - _pair_cache["day"] >= reselect or not _pair_cache["pairs"]:
        _pair_cache["pairs"] = select_pairs(hist, max_pairs, sel_window, pmax)
        _pair_cache["day"] = t
    for i, j, _, beta in _pair_cache["pairs"]:
        spread = hist[i, :] - beta * hist[j, :]
        w = spread[-pair_lb:]; z = (spread[-1] - w.mean()) / (w.std() + 1e-9)
        if abs(z) > pair_entry:
            pos[i] += -np.sign(z) * pair_dollars / cur[i]
            pos[j] += np.sign(z) * beta * pair_dollars / cur[j]

    # ---- E2: correlation-vs-ALGO residual reversion (factor stat-arb) ----
    if use_overlay:
        la = np.log(hist[ALGO, -ov_lb:]); algo_leg = 0.0
        for i in range(1, n):
            li = np.log(hist[i, -ov_lb:])
            beta = np.polyfit(la, li, 1)[0]
            resid = np.log(hist[i, :]) - beta * np.log(hist[ALGO, :])
            w = resid[-ov_lb:]; z = (resid[-1] - w.mean()) / (w.std() + 1e-9)
            if abs(z) > ov_entry:
                sh = -np.sign(z) * ov_dollars / cur[i]
                pos[i] += sh
                algo_leg += -sh * beta * cur[i] / cur[ALGO]
        pos[ALGO] += algo_leg
    return pos.astype(int)


def backtest(get_pos, start_day, end_day):
    cash = 0.0; curPos = np.zeros(N); totDVol = 0.0; value = 0.0; comm = 0.0; pll = []
    _pair_cache["day"] = -10**9; _pair_cache["pairs"] = []
    for t in range(start_day, end_day + 1):
        hist = P[:, :t]; cur = hist[:, -1]
        if t < end_day:
            lim = (dlrPosLimit / cur).astype(int)
            newPos = np.clip(get_pos(hist), -lim, lim).astype(int)
        else:
            newPos = np.array(curPos)
        d = newPos - curPos; cash -= cur.dot(d) + comm
        dvol = cur * np.abs(d); comm = np.sum(dvol * commRate); totDVol += dvol.sum()
        curPos = np.array(newPos); pv = curPos.dot(cur)
        todayPL = cash + pv - value; value = cash + pv
        if t > start_day: pll.append(todayPL)
    pll = np.array(pll); mu, sd = pll.mean(), pll.std()
    sharpe = np.sqrt(250) * mu / sd if sd > 0 else 0
    score = mu * (sharpe**2 / (sharpe**2 + 1)) if (mu > 0 and sd > 1e-10) else mu
    return dict(mean=mu, sharpe=sharpe, score=score, dvol=totDVol)


section("20A. COMPONENT vs COMBINED on last 250 days (eval.py scoring)")
full_start = T - N_TEST_DAYS
configs = {
    "pairs only":            dict(use_overlay=False),
    "overlay only":          dict(max_pairs=0),
    "COMBINED":              dict(),
}
print(f"{'config':<22}{'mean$':>9}{'Sharpe':>8}{'Score':>9}{'$vol':>12}")
for name, kw in configs.items():
    r = backtest(lambda h, kw=kw: combined(h, **kw), full_start, T)
    print(f"{name:<22}{r['mean']:>9.2f}{r['sharpe']:>8.2f}{r['score']:>9.2f}{r['dvol']:>12.0f}")

section("20B. TUNE for SCORE (grid; walk-forward on last 250 days)")
best = (None, -1e9)
print(f"{'pairs':>6}{'p_ent':>7}{'p_$':>7}{'ov_ent':>7}{'ov_$':>7}{'Sharpe':>8}{'Score':>8}")
for mp in (12, 20):
    for pe in (0.5, 0.7):
        for pd_ in (7000, 9000):
            for oe in (0.9, 1.2):
                for od in (3000, 5000):
                    kw = dict(max_pairs=mp, pair_entry=pe, pair_dollars=pd_,
                              ov_entry=oe, ov_dollars=od)
                    r = backtest(lambda h, kw=kw: combined(h, **kw), full_start, T)
                    if r["score"] > best[1]:
                        best = (kw, r["score"], r["sharpe"], r["mean"])
                    if pd_ == 9000 and od == 5000:  # print a slice
                        print(f"{mp:>6}{pe:>7.2f}{pd_:>7}{oe:>7.2f}{od:>7}{r['sharpe']:>8.2f}{r['score']:>8.2f}")
print(f"\nBEST config score={best[1]:.2f} Sharpe={best[2]:.2f} mean=${best[3]:.2f}")
print(f"  params: {best[0]}")

section("20C. STRICT OUT-OF-SAMPLE (tune on days 250-375, test 375-500)")
mid = T - 250; q3 = T - 125
# retune on the first test-half only
best_oos = (None, -1e9)
for mp in (12, 20):
    for pe in (0.5, 0.7):
        for od in (3000, 5000):
            kw = dict(max_pairs=mp, pair_entry=pe, ov_dollars=od, pair_dollars=9000)
            r = backtest(lambda h, kw=kw: combined(h, **kw), mid, q3)
            if r["score"] > best_oos[1]:
                best_oos = (kw, r["score"])
r_is = backtest(lambda h: combined(h, **best_oos[0]), mid, q3)
r_oos = backtest(lambda h: combined(h, **best_oos[0]), q3, T)
print(f"Tuned on 250-375: Sharpe {r_is['sharpe']:.2f} score {r_is['score']:.2f}")
print(f"Held-out 375-500: Sharpe {r_oos['sharpe']:.2f} score {r_oos['score']:.2f}  <- honest OOS")
print(f"  (config: {best_oos[0]})")

# save best full-sample config for the submission file
import json, os
from common import RESULTS
json.dump(best[0], open(os.path.join(RESULTS, "best_combined_config.json"), "w"))
print(f"\nSaved best config -> results/best_combined_config.json")
