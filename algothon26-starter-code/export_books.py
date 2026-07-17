#!/usr/bin/env python
"""Export per-day position matrices for ALL scorecard strategies into books.json,
so the dashboard can plot each one's entries/exits. Run:  python export_books.py
Then rebuild:  python dashboard.py --books books.json --runs runs.json
"""
import json, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import backtester as bt, strategy as st

df = pd.read_csv("prices.txt", sep=r"\s+"); names = list(df.columns)
prc = df.values.T; N, T = prc.shape
comm, lim = bt.make_grading_params(N)

# ---- shared helpers (copied from the research build so this file is standalone) ----
def _clip(pos, p):
    mx = (lim / p[:, -1]).astype(int); return np.clip(pos.astype(int), -mx, mx)
def realized_ic_xs(p, w, lb):
    t = p.shape[1]; ics = []
    for d in range(max(w + 1, t - lb), t - 1):
        s = -st.zscore(p[:, :d + 1], w); s = s[1:]; s = s - s.mean()
        fwd = p[1:, d + 1] / p[1:, d] - 1; fwd = fwd - fwd.mean()
        den = s.std() * fwd.std()
        if den > 1e-12: ics.append(float((s * fwd).mean() / den))
    return float(np.mean(ics)) if ics else 0.0
def realized_ic_index(p, w, lb):
    t = p.shape[1]; xs, ys = [], []
    for d in range(max(w + 1, t - lb), t - 1):
        xs.append(-st.zscore(p[:, :d + 1], w)[0]); ys.append(p[0, d + 1] / p[0, d] - 1)
    xs, ys = np.array(xs), np.array(ys)
    if len(xs) < 5 or xs.std() < 1e-12 or ys.std() < 1e-12: return 0.0
    return float(np.corrcoef(xs, ys)[0, 1])
def var_ratio(x, q):
    n = len(x); mu = x.mean()
    if n < q + 2: return 1.0
    v1 = np.sum((x - mu) ** 2) / (n - 1)
    qs = np.convolve(x, np.ones(q), "valid"); vq = np.sum((qs - q * mu) ** 2) / (len(qs) * q)
    return vq / v1 if v1 > 1e-12 else 1.0

# ---- the 7 books ----
base = st.make_two_leg(10, 5, 0.10, 1.0, 0.10, "fraction")

def adaptive_book(idio_w=10, algo_w=5, idio_scale=0.10, algo_scale=0.10,
                  ic_lb=15, ic_target=0.02, cap=1.2):
    def f(p):
        n, t = p.shape; pos = np.zeros(n, dtype=int)
        if t < idio_w + ic_lb + 3: return pos
        conf = float(np.clip(realized_ic_xs(p, idio_w, ic_lb) / ic_target, 0.0, cap))
        s = (-st.zscore(p, idio_w)).astype(float); s[0] = 0.0; s[1:] -= s[1:].mean()
        pos = (st.size_fraction_of_limit(s, p, idio_scale) * conf).astype(int)
        aconf = float(np.clip(realized_ic_index(p, algo_w, ic_lb) / ic_target, 0.0, cap))
        za = st.zscore(p, algo_w)[0]
        pos[0] = int(float(np.clip(-za / algo_scale, -1, 1) * aconf) * lim[0] / p[0, -1])
        return _clip(pos, p)
    return f

def macro_switch(rev_w=10, mom_w=60, vr_w=60, vr_q=20):
    def f(p):
        n, t = p.shape
        if t < max(mom_w, vr_w) + 2: return np.zeros(n, dtype=int)
        r0 = np.diff(np.log(p[0, -(vr_w + 1):]))
        s = st.momentum(p, mom_w) if var_ratio(r0, vr_q) > 1.0 else -st.zscore(p, rev_w)
        s = s.astype(float); s -= np.nanmean(s)
        return _clip(st.size_fraction_of_limit(s, p, 0.10), p)
    return f

def pername_switch(rev_w=10, mom_w=60, vr_w=60, vr_q=20):
    def f(p):
        n, t = p.shape
        if t < max(mom_w, vr_w) + 2: return np.zeros(n, dtype=int)
        rev = -st.zscore(p, rev_w); mom = st.momentum(p, mom_w)
        r = np.diff(np.log(p[:, -(vr_w + 1):]), axis=1)
        vr = np.array([var_ratio(r[k], vr_q) for k in range(n)])
        s = np.where(vr > 1.0, mom, rev).astype(float); s -= np.nanmean(s)
        return _clip(st.size_fraction_of_limit(s, p, 0.10), p)
    return f

def markov_switch(rev_w=10, mom_w=60, min_hist=80):
    def f(p):
        n, t = p.shape
        if t < min_hist: return np.zeros(n, dtype=int)
        from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
        y = np.diff(np.log(p[0])) * 100.0
        try:
            res = MarkovAutoregression(y, k_regimes=2, order=1, switching_ar=True).fit(disp=False)
            nm = list(res.model.param_names)
            ar = [(i, res.params[i]) for i, x in enumerate(nm) if "ar" in x.lower()]
            trend_reg = int(np.argmax([v for _, v in ar])) if ar else 0
            fp = np.asarray(res.filtered_marginal_probabilities)
            p_trend = float(fp[-1, trend_reg]) if fp.ndim == 2 else 0.5
        except Exception:
            p_trend = 0.5
        s = (p_trend * st.momentum(p, mom_w) + (1 - p_trend) * (-st.zscore(p, rev_w))).astype(float)
        s -= np.nanmean(s)
        return _clip(st.size_fraction_of_limit(s, p, 0.10), p)
    return f

def markov_trend(min_hist=80):
    """Switching-MEAN Markov on the index: go LONG all names when P(bull) is high,
    SHORT when low — 'detect shifts and buy into the trend' (directional market timing)."""
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    def f(p):
        n, t = p.shape; pos = np.zeros(n, dtype=int)
        if t < min_hist: return pos
        y = np.diff(np.log(p[0])) * 100.0
        try:
            r = MarkovRegression(y, k_regimes=2, trend="c", switching_variance=True).fit(disp=False)
            means = [r.params[i] for i, x in enumerate(r.model.param_names) if x.startswith("const")]
            bull = int(np.argmax(means))
            fm = np.asarray(r.filtered_marginal_probabilities)
            pb = float(fm[-1, bull]) if fm.ndim == 2 else float(fm[bull, -1])
        except Exception:
            pb = 0.5
        tilt = float(np.clip(2 * pb - 1, -1, 1))
        return _clip((tilt * lim / p[:, -1]).astype(int), p)
    return f

def full_by_sign(sig):
    def f(p):
        frac = np.sign(sig(p)); last = p[:, -1]
        return _clip((frac * lim / last).astype(int), p)
    return f
mom_book = lambda p: _clip(st.size_fraction_of_limit(
    (lambda z: z - np.nanmean(z))(st.momentum(p, 60)), p, 0.10), p)
macro_dir = st.hold_every(full_by_sign(lambda p: st.momentum(p, 120)), 60)

BOOKS = [
    ("base reversion", base),
    ("adaptive (A)", adaptive_book()),
    ("macro switch (B)", macro_switch()),
    ("per-name switch", pername_switch()),
    ("Markov switch", markov_switch()),
    ("markov trend (buy)", markov_trend()),
    ("pure momentum", mom_book),
    ("macro directional", macro_dir),
]

NTD = 486
out = []
for lab, gp in BOOKS:
    t0 = time.time()
    res = bt.run_backtest(prc, gp, num_test_days=NTD, comm_rate=comm,
                          dlr_pos_limit=lim, inst_names=names)
    byName = {names[i]: [int(round(float(x))) for x in res.positions[:, i]] for i in range(N)}
    out.append({"label": lab, "days": [int(d) for d in res.days], "byName": byName})
    print(f"exported {lab:<20} {time.time()-t0:6.1f}s  Score {res.score:7.1f}", flush=True)

json.dump(out, open("books.json", "w"), separators=(",", ":"))
print(f"wrote books.json with {len(out)} books", flush=True)
