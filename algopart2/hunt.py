"""
hunt.py — signal & test battery for algopart2/prices.txt, focused on days 400-750.

Context: prices.txt is the full 750-day, 51-instrument panel (inst 0 = ALGO = the
equal-weight index). The first 500 days were the original training file; days 500-750
are the "newly revealed" window the earlier research could only forecast. This script
re-hunts on the RECENT portion (days 400-750) to see which edges persist, whether the
regime shifted, and what score is actually achievable on the fresh data.

Self-contained: numpy + pandas only. Reproduces eval.py's scoring exactly
(Score = mean * SR^2/(SR^2+1), SR = sqrt(250)*mean/std, inst-0 = $100k limit / 0.2bp).

Run:  python hunt.py            # full report to stdout, writes results.json + FINDINGS numbers
"""
import json
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- data
PRICES = "prices.txt"
prc = pd.read_csv(PRICES, sep=r"\s+", header=0)
names = list(prc.columns)
P = prc.values.T                       # (nInst, nDays)
nInst, nDays = P.shape
logp = np.log(P)
ret = logp[:, 1:] - logp[:, :-1]       # (nInst, nDays-1) simple log returns, col t = ret into day t+1

# grading params (identical to eval.py)
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlrLim = np.full(nInst, 10_000); dlrLim[0] = 100_000

RNG = np.random.default_rng(12345)     # fixed seed -> reproducible permutation nulls


def zscore_rows(a):
    return (a - a.mean()) / (a.std() + 1e-12)


# --------------------------------------------------------------- eval-faithful score
def score_pll(pll):
    pll = np.asarray(pll)
    mu, sd = pll.mean(), pll.std()
    if mu <= 0 or sd < 1e-10:
        return mu, (np.sqrt(250) * mu / sd if sd > 0 else 0.0)
    sr = np.sqrt(250) * mu / sd
    return mu * sr**2 / (sr**2 + 1.0), sr


def run_backtest(getPos, startDay, endDay):
    """Exact eval.py loop over [startDay, endDay]. getPos(prcSoFar)->target shares.
    startDay is the first day positions are taken; endDay is the mark day (exclusive of trade)."""
    cash = 0.0; curPos = np.zeros(nInst); value = 0.0; comm = 0.0
    pll = []
    for t in range(startDay, endDay + 1):
        soFar = P[:, :t]
        cur = soFar[:, -1]
        if t < endDay:
            raw = getPos(soFar)
            lim = (dlrLim / cur).astype(int)
            newPos = np.clip(raw, -lim, lim).astype(int)
        else:
            newPos = curPos.copy()
        d = newPos - curPos
        cash -= cur.dot(d) + comm
        dv = cur * np.abs(d)
        comm = np.sum(dv * commRate)
        curPos = newPos.copy()
        pl = cash + curPos.dot(cur) - value
        value = cash + curPos.dot(cur)
        if t > startDay:
            pll.append(pl)
    pll = np.array(pll)
    sc, sr = score_pll(pll)
    return dict(score=sc, sharpe=sr, mean=pll.mean(), std=pll.std(),
                total=pll.sum(), maxdd=drawdown(pll), n=len(pll))


def drawdown(pll):
    eq = np.cumsum(pll); peak = np.maximum.accumulate(eq)
    return float((eq - peak).min())


# ------------------------------------------------------------------- signal builders
def sig_xs_rev(soFar, w):
    """cross-sectional reversion on the 50 idio names: -zscore of w-day return,
    demeaned (market neutral). Returns a 50-vector (ALGO excluded)."""
    lp = np.log(soFar[1:])
    r = lp[:, -1] - lp[:, -1 - w]
    r = r - r.mean()
    return -zscore_rows(r)


def sig_momentum(soFar, w):
    lp = np.log(soFar[1:])
    r = lp[:, -1] - lp[:, -1 - w]
    r = r - r.mean()
    return zscore_rows(r)


def algo_zrev(soFar, w=5, lookback=60):
    """ALGO index reversion z-score over a w-day move."""
    lpA = np.log(soFar[0])
    mv = lpA[w:] - lpA[:-w]
    z = (mv[-1] - mv[-lookback:].mean()) / (mv[-lookback:].std() + 1e-12)
    return -np.clip(z, -3, 3)


# EWLS peer lead-lag forecast (SIGNAL 1 from combinedv3), ridge-regularised
def ewls(X, Y, hl, a=0.1):
    n, p = X.shape
    lam = 0.5 ** (1.0 / hl)
    w = lam ** np.arange(n - 1, -1, -1); sw = w.sum()
    mx = (w[:, None] * X).sum(0) / sw; my = (w[:, None] * Y).sum(0) / sw
    Xc, Yc = X - mx, Y - my
    XtWX = Xc.T @ (w[:, None] * Xc); XtWY = Xc.T @ (w[:, None] * Yc)
    eps = 1e-8 * np.trace(XtWX) / p
    B = np.linalg.solve(XtWX + (eps + a) * np.eye(p), XtWY)
    return B, mx, my


def leadlag_forecast(soFar, hl=500):
    lp = np.log(soFar); r = lp[:, 1:] - lp[:, :-1]
    B, mx, my = ewls(r[:, :-1].T, r[1:, 1:].T, hl)
    pred = my + (r[:, -1] - mx) @ B
    return pred - pred.mean()


# ---------------------------------------------------------------------- IC machinery
def ic_series(signal_fn, S, E, horizon, warmup=95):
    """Daily cross-sectional Spearman-ish (Pearson on values) IC of signal vs forward
    `horizon`-day return of the 50 idio names, over days [S, E]."""
    ics = []
    for t in range(max(S, warmup), E - horizon):
        soFar = P[:, :t]
        sig = signal_fn(soFar)
        sig = sig[1:] if len(sig) == nInst else sig   # drop ALGO if signal is full-width
        fwd = np.log(P[1:, t + horizon - 1]) - np.log(P[1:, t - 1])
        fwd = fwd - fwd.mean()
        if sig.std() < 1e-12 or fwd.std() < 1e-12:
            continue
        ics.append(np.corrcoef(sig, fwd)[0, 1])
    ics = np.array(ics)
    if len(ics) < 3:
        return dict(ic=np.nan, t=np.nan, n=len(ics))
    t = ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics)))
    return dict(ic=float(ics.mean()), t=float(t), n=len(ics))


# ------------------------------------------------------------------- the two-leg book
def two_leg(idio_w=10, algo_w=5, idio_scale=0.10):
    """Shipped book: Leg A ALGO reversion @ $100k, Leg B -zscore(idio_w) demeaned @ $10k."""
    def gp(soFar):
        cur = soFar[:, -1]
        pos = np.zeros(nInst)
        if soFar.shape[1] < max(idio_w, algo_w, 65) + 2:
            return pos
        # Leg B: idio cross-sectional reversion
        s = sig_xs_rev(soFar, idio_w)
        pos[1:] = s * (dlrLim[1:] / cur[1:]) * (idio_scale / 0.10)   # scale ~ to $10k each at |z|~1
        pos[1:] = np.clip(pos[1:], -dlrLim[1:] / cur[1:], dlrLim[1:] / cur[1:])
        # Leg A: ALGO index reversion, sized to full $100k by conviction
        za = algo_zrev(soFar, algo_w)
        pos[0] = (za / 3.0) * (dlrLim[0] / cur[0])
        return pos
    return gp


def leg_only(which, idio_w=10, algo_w=5):
    def gp(soFar):
        cur = soFar[:, -1]; pos = np.zeros(nInst)
        if soFar.shape[1] < max(idio_w, algo_w, 65) + 2:
            return pos
        if which in ("idio", "both"):
            s = sig_xs_rev(soFar, idio_w)
            pos[1:] = np.clip(s * (dlrLim[1:] / cur[1:]), -dlrLim[1:] / cur[1:], dlrLim[1:] / cur[1:])
        if which in ("algo", "both"):
            za = algo_zrev(soFar, algo_w)
            pos[0] = (za / 3.0) * (dlrLim[0] / cur[0])
        return pos
    return gp


# ============================================================================ REPORT
out = {"meta": {"nInst": nInst, "nDays": nDays, "focus": "days 400-750"}}
print(f"# hunt.py — algopart2  ({nInst} instruments x {nDays} days)")
print(f"# focus window: days 400-750 (recent regime; days 500-750 = newly revealed)\n")

# ---- 1. structural fingerprints (recent window) --------------------------------
print("=" * 78)
print("1. STRUCTURE (returns over days 400-750)")
print("=" * 78)
r_win = ret[:, 399:749]                      # returns within the focus window
eqw = r_win[1:].mean(0)
corr_algo = np.corrcoef(r_win[0], eqw)[0, 1]
# PCA on idio names
Rc = (r_win[1:] - r_win[1:].mean(1, keepdims=True))
C = np.cov(Rc)
ev = np.sort(np.linalg.eigvalsh(C))[::-1]
pc = ev / ev.sum()
# beta of each name to ALGO
rA = r_win[0] - r_win[0].mean()
betas = (Rc @ rA) / (rA @ rA)
r2 = []
for i in range(50):
    yy = Rc[i]; pr = betas[i] * rA
    r2.append(1 - ((yy - pr) ** 2).sum() / ((yy) ** 2).sum())
r2 = np.array(r2)
print(f"corr(ALGO ret, equal-weight avg of 50)   = {corr_algo:.4f}   (index identity check)")
print(f"PC1 / PC2 / PC3 variance explained       = {pc[0]:.1%} / {pc[1]:.1%} / {pc[2]:.1%}")
print(f"mean beta to ALGO = {betas.mean():.3f}   frac>0 = {(betas>0).mean():.2f}   mean R2 = {r2.mean():.2f}")
out["structure"] = dict(corr_algo=float(corr_algo), pc1=float(pc[0]), pc2=float(pc[1]),
                        pc3=float(pc[2]), mean_beta=float(betas.mean()), mean_r2=float(r2.mean()))

# lag-1 autocorr and vol clustering, on idio (market-removed) returns
idio = Rc - np.outer(betas, rA)
ac1 = np.array([np.corrcoef(idio[i, :-1], idio[i, 1:])[0, 1] for i in range(50)])
vc1 = np.array([np.corrcoef(np.abs(idio[i, :-1]), np.abs(idio[i, 1:]))[0, 1] for i in range(50)])
print(f"idio lag-1 autocorr  mean = {ac1.mean():+.4f}  (t={ac1.mean()/(ac1.std(ddof=1)/np.sqrt(50)):+.2f})  -> per-name 1d predictability")
print(f"|idio| lag-1 autocorr mean = {vc1.mean():+.4f}  -> vol clustering / GARCH")
out["structure"].update(idio_ac1=float(ac1.mean()), vol_ac1=float(vc1.mean()))

# ---- 2. IC battery on the focus window -----------------------------------------
print("\n" + "=" * 78)
print("2. CROSS-SECTIONAL IC on days 400-750  (signal vs forward return, 50 idio names)")
print("=" * 78)
print(f"{'signal':<22}{'IC@1d':>9}{'t@1d':>7}{'IC@5d':>9}{'t@5d':>7}{'IC@10d':>9}{'t@10d':>7}")
ic_table = {}
sig_defs = [
    ("xs_rev5 (-z of 5d)",  lambda s: sig_xs_rev(s, 5)),
    ("xs_rev10",            lambda s: sig_xs_rev(s, 10)),
    ("xs_rev20",            lambda s: sig_xs_rev(s, 20)),
    ("xs_rev40",            lambda s: sig_xs_rev(s, 40)),
    ("leadlag_ewls(hl500)", lambda s: leadlag_forecast(s, 500)),
    ("momentum20 (control)",lambda s: sig_momentum(s, 20)),
    ("momentum60 (control)",lambda s: sig_momentum(s, 60)),
]
for nm, fn in sig_defs:
    row = {}
    cells = []
    for h in (1, 5, 10):
        d = ic_series(fn, 400, 749, h)
        row[f"ic{h}"] = d["ic"]; row[f"t{h}"] = d["t"]
        cells.append(f"{d['ic']:>9.4f}{d['t']:>7.2f}")
    ic_table[nm] = row
    print(f"{nm:<22}" + "".join(cells))
out["ic_table"] = ic_table

# ---- 3. permutation significance of the two edges (on the focus window) --------
print("\n" + "=" * 78)
print("3. PERMUTATION NULLS  (edge real, or random-walk artifact?)  window 500-750")
print("=" * 78)
NPERM = 300
def perm_pvalue(getPos_from_prc, S, E, nperm=NPERM):
    """Score with real prices vs prices rebuilt from SHUFFLED returns (kills time-structure)."""
    obs = run_backtest(getPos_from_prc(P), S, E)["score"]
    null = []
    base = np.log(P[:, :1])
    for _ in range(nperm):
        perm = ret.copy()
        idx = RNG.permutation(ret.shape[1])
        perm = perm[:, idx]
        Pp = np.exp(np.concatenate([base, base + np.cumsum(perm, axis=1)], axis=1))
        # rebind a backtest that reads Pp
        null.append(_score_on(Pp, getPos_from_prc(Pp), S, E))
    null = np.array(null)
    p = (null >= obs).mean()
    return obs, null.mean(), np.percentile(null, 95), p

def _score_on(Pp, getPos, S, E):
    cash=0.0; curPos=np.zeros(nInst); value=0.0; comm=0.0; pll=[]
    for t in range(S, E+1):
        soFar=Pp[:, :t]; cur=soFar[:, -1]
        if t < E:
            raw=getPos(soFar); lim=(dlrLim/cur).astype(int)
            newPos=np.clip(raw,-lim,lim).astype(int)
        else:
            newPos=curPos.copy()
        d=newPos-curPos; cash-=cur.dot(d)+comm; dv=cur*np.abs(d); comm=np.sum(dv*commRate)
        curPos=newPos.copy(); pl=cash+curPos.dot(cur)-value; value=cash+curPos.dot(cur)
        if t>S: pll.append(pl)
    return score_pll(np.array(pll))[0]

# NOTE: getPos closures capture P by name; for perm we need them to read the passed price array.
# Rebuild leg closures parameterised by the price matrix.
def make_idio(Pmat, w=10):
    def gp(soFar):
        cur=soFar[:,-1]; pos=np.zeros(nInst)
        if soFar.shape[1] < w+2: return pos
        s=sig_xs_rev(soFar,w)
        pos[1:]=np.clip(s*(dlrLim[1:]/cur[1:]), -dlrLim[1:]/cur[1:], dlrLim[1:]/cur[1:])
        return pos
    return gp
def make_algo(Pmat, w=5):
    def gp(soFar):
        cur=soFar[:,-1]; pos=np.zeros(nInst)
        if soFar.shape[1] < 65: return pos
        pos[0]=(algo_zrev(soFar,w)/3.0)*(dlrLim[0]/cur[0])
        return pos
    return gp

for label, mk in [("ALGO index leg (zrev5)", make_algo), ("idio leg (-z10)", make_idio)]:
    obs, nmean, n95, p = perm_pvalue(lambda Pm, mk=mk: mk(Pm), 500, 749)
    star = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))
    print(f"{label:<26} obs Score={obs:7.1f}   null mean={nmean:7.1f}  95%={n95:7.1f}  p={p:.3f} {star}")
    out.setdefault("perm", {})[label] = dict(obs=float(obs), null_mean=float(nmean), null_p95=float(n95), p=float(p))

# ---- 4. backtest scores across windows within 400-750 --------------------------
print("\n" + "=" * 78)
print("4. TWO-LEG BOOK — Score across windows (eval-faithful, full limits)")
print("=" * 78)
book = two_leg()
windows = [("full 400-750", 400, 749), ("old 400-500", 400, 500),
           ("NEW 500-750 (graded-like)", 500, 749), ("last 250 (500-750)", 500, 749)]
print(f"{'window':<28}{'Score':>8}{'Sharpe':>8}{'mean/d':>9}{'total':>10}{'maxDD':>9}{'days':>6}")
out["book_windows"] = {}
for label, S, E in windows:
    r = run_backtest(book, S, E)
    print(f"{label:<28}{r['score']:>8.1f}{r['sharpe']:>8.2f}{r['mean']:>9.1f}{r['total']:>10.0f}{r['maxdd']:>9.0f}{r['n']:>6}")
    out["book_windows"][label] = {k: float(v) for k, v in r.items()}

# leg attribution on the new window
print("\nLeg attribution on NEW window 500-750:")
for which in ("algo", "idio", "both"):
    r = run_backtest(leg_only(which), 500, 749)
    print(f"  {which:<6} Score={r['score']:7.1f}  Sharpe={r['sharpe']:5.2f}  total=${r['total']:,.0f}")
    out.setdefault("leg_attr", {})[which] = {k: float(v) for k, v in r.items()}

# ---- 5. rolling 100-day regime scan --------------------------------------------
print("\n" + "=" * 78)
print("5. ROLLING 100-day Score (regime map across 400-750)")
print("=" * 78)
roll = []
for s in range(300, 650, 50):
    r = run_backtest(book, s, s + 100)
    roll.append((s, s + 100, r["score"], r["sharpe"]))
    print(f"  days {s:>3}-{s+100:<3}  Score={r['score']:7.1f}  Sharpe={r['sharpe']:5.2f}")
out["rolling100"] = [dict(start=a, end=b, score=float(c), sharpe=float(d)) for a, b, c, d in roll]

# ---- 6. LEAD-LAG book — does the strongest IC actually trade? ------------------
print("\n" + "=" * 78)
print("6. LEAD-LAG EWLS book (SIGNAL 1) — the strongest IC. Does it MONETISE?")
print("=" * 78)
def leadlag_book(hl=500, conv=0.0):
    def gp(soFar):
        cur = soFar[:, -1]; pos = np.zeros(nInst)
        if soFar.shape[1] < 96:
            return pos
        w = leadlag_forecast(soFar, hl)
        wz = w / (w.std() + 1e-12)
        take = np.abs(wz) >= conv
        pos[1:] = np.where(take, np.sign(wz) * (dlrLim[1:] / cur[1:]), 0.0)
        return pos
    return gp

llb = leadlag_book()
print(f"{'window':<28}{'Score':>8}{'Sharpe':>8}{'total':>10}{'days':>6}")
out["leadlag_windows"] = {}
for label, S, E in [("full 400-750", 400, 749), ("old 400-500", 400, 500),
                    ("NEW 500-750", 500, 749)]:
    r = run_backtest(llb, S, E)
    print(f"{label:<28}{r['score']:>8.1f}{r['sharpe']:>8.2f}{r['total']:>10.0f}{r['n']:>6}")
    out["leadlag_windows"][label] = {k: float(v) for k, v in r.items()}

# lead-lag + ALGO overlay (add index leg back for the capital) and + reversion blend
def combined_book(hl=500, blend=0.3, revw=10, algo=True):
    def gp(soFar):
        cur = soFar[:, -1]; pos = np.zeros(nInst)
        if soFar.shape[1] < 96:
            return pos
        w = leadlag_forecast(soFar, hl); wz = w / (w.std() + 1e-12)
        if blend > 0:
            rv = sig_xs_rev(soFar, revw)
            wz = (1 - blend) * wz + blend * (rv / (rv.std() + 1e-12))
        pos[1:] = np.sign(wz) * (dlrLim[1:] / cur[1:])
        if algo:
            pos[0] = (algo_zrev(soFar, 5) / 3.0) * (dlrLim[0] / cur[0])
        return pos
    return gp
print("\ncombined (lead-lag + reversion blend + ALGO leg):")
for label, S, E in [("full 400-750", 400, 749), ("NEW 500-750", 500, 749)]:
    r = run_backtest(combined_book(), S, E)
    print(f"  {label:<20} Score={r['score']:7.1f}  Sharpe={r['sharpe']:5.2f}  total=${r['total']:,.0f}")
    out.setdefault("combined_windows", {})[label] = {k: float(v) for k, v in r.items()}

# permutation null for lead-lag on 500-750
def make_ll(Pmat, hl=500):
    def gp(soFar):
        cur = soFar[:, -1]; pos = np.zeros(nInst)
        if soFar.shape[1] < 96: return pos
        w = leadlag_forecast(soFar, hl); wz = w / (w.std() + 1e-12)
        pos[1:] = np.sign(wz) * (dlrLim[1:] / cur[1:])
        return pos
    return gp
obs, nmean, n95, p = perm_pvalue(lambda Pm: make_ll(Pm), 500, 749)
star = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))
print(f"\nlead-lag permutation null (500-750): obs={obs:.1f} null mean={nmean:.1f} 95%={n95:.1f} p={p:.3f} {star}")
out["perm"]["lead-lag (500-750)"] = dict(obs=float(obs), null_mean=float(nmean), null_p95=float(n95), p=float(p))

with open("results.json", "w") as f:
    json.dump(out, f, indent=2)
print("\n[written results.json]")
