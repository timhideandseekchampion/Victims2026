"""Do FUNDAMENTALLY DIFFERENT strategy families reach 700 on 500-750 where our ridge caps at 530?
Test: (A) cointegration pairs book, (B) single-name OU mean-reversion, (C) a NONLINEAR (gradient-
boosted) predictor's IC vs the ridge. Fit each to 500-750 (in-sample). If any clears ~600-700, it's
the recipe; if all cap near the ridge, the window genuinely has no more extractable edge."""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

prc = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nt = prc.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000); dlr[0] = 100_000
lp = np.log(prc); RET = lp[:, 1:] - lp[:, :-1]
S, E = nt - 250, nt


def score(pll):
    mu, sd = pll.mean(), pll.std()
    if mu <= 0 or sd < 1e-10: return mu
    sr = np.sqrt(250)*mu/sd; return mu*sr**2/(sr**2+1)


def run(gp, S, E):
    cash = 0; cp = np.zeros(nInst); val = 0; cm = 0; pll = []
    for t in range(S, E+1):
        p = prc[:, :t]; cur = p[:, -1]
        npos = np.clip(gp(p), -(dlr/cur).astype(int), (dlr/cur).astype(int)).astype(int) if t < E else cp.copy()
        d = npos-cp; cash -= cur.dot(d)+cm; dv = cur*np.abs(d); cm = (dv*commRate).sum(); cp = npos.copy()
        pl = cash+cp.dot(cur)-val; val = cash+cp.dot(cur)
        if t > S: pll.append(pl)
    return score(np.array(pll))


# (A) single-name OU mean-reversion: z = (price - MA)/std, trade -z, market-neutral
def ou_book(W=20):
    def gp(prc):
        ni, t = prc.shape; pos = np.zeros(ni)
        if t < W+2: return pos.astype(int)
        L = np.log(prc[:, -W:]); z = (L[:, -1] - L.mean(1)) / (L.std(1) + 1e-12)
        sig = -(z[1:] - z[1:].mean())
        pos[1:] = np.sign(sig) * (10000/prc[1:, -1]) * (np.abs(sig) >= 0.5)
        # hedge
        ret = np.log(prc[:, 1:]) - np.log(prc[:, :-1]); rA = ret[0]; rAc = rA-rA.mean(); den = rAc@rAc+1e-12
        betas = ((ret[1:]-ret[1:].mean(1, keepdims=True))@rAc)/den
        cap = 100000/prc[0, -1]; pos[0] = float(np.clip(-((pos[1:]*prc[1:, -1])@betas)/prc[0, -1], -cap, cap))
        return pos.astype(int)
    return gp


# (B) cointegration pairs: fast residual-reversion selection (from forward_mc), optimized dollars
def pairs_book(dollars=10000, k=24, win=90):
    def gp(prc):
        ni, t = prc.shape; pos = np.zeros(ni)
        if t < win+3: return pos.astype(int)
        Lg = np.log(prc); R = np.diff(Lg[:, -win:], axis=1)
        mkt = R[1:].mean(0); den = (mkt-mkt.mean())@(mkt-mkt.mean())+1e-12
        idio = R[1:] - np.outer(((R[1:]-R[1:].mean(1, keepdims=True))@(mkt-mkt.mean()))/den, mkt)
        C = np.corrcoef(idio); cur = prc[:, -1]; cand = []
        Lw = Lg[:, -win:]
        for a in range(idio.shape[0]):
            for b in range(a+1, idio.shape[0]):
                if abs(C[a, b]) < 0.3: continue
                i, j = a+1, b+1
                beta = np.polyfit(Lw[j], Lw[i], 1)[0]; spr = Lw[i]-beta*Lw[j]
                ds = np.diff(spr); lev = spr[:-1]-spr[:-1].mean(); rho = -(ds@lev)/((lev@lev)+1e-12)
                if rho > 0.05:
                    zz = (spr[-1]-spr.mean())/(spr.std()+1e-9); cand.append((rho, i, j, beta, zz))
        cand.sort(reverse=True); used = set()
        for rho, i, j, beta, zz in cand:
            if len(used) >= 2*k: break
            if i in used or j in used or abs(zz) <= 1.0: continue
            u = -np.sign(zz); pos[i] += u*dollars/cur[i]; pos[j] += -u*beta*dollars/cur[j]
            used.add(i); used.add(j)
        return pos.astype(int)
    return gp


# (C) nonlinear predictor IC on 500-750 vs ridge (does nonlinearity beat linear?)
def nonlinear_ic():
    def ewls(X, Y, hl=500, a=0.1):
        n, p = X.shape; lam = 0.5**(1.0/hl); w = lam**np.arange(n-1, -1, -1); sw = w.sum()
        mx = (w[:, None]*X).sum(0)/sw; my = (w[:, None]*Y).sum(0)/sw; Xc, Yc = X-mx, Y-my
        return np.linalg.solve(Xc.T@(w[:, None]*Xc)+(1e-8*np.trace(Xc.T@(w[:, None]*Xc))/p+a)*np.eye(p), Xc.T@(w[:, None]*Yc)), mx, my
    ics_lin, ics_nl = [], []
    for d in range(S, E-1, 3):                       # every 3rd day (GBM is slow)
        # pooled training on all history up to d: features = today's 51-vector, target = each name next ret
        Xtr, Ytr = [], []
        for tau in range(60, d-1):
            Xtr.append(RET[:, tau]); Ytr.append(RET[1:, tau+1])
        Xtr = np.array(Xtr); Ytr = np.array(Ytr)     # (T,51) -> predict (T,50)
        # linear
        B, mx, my = ewls(Xtr, Ytr)
        lin = my + (RET[:, d]-mx) @ B
        # nonlinear pooled (one GBM over stacked name-targets)
        Xs = np.repeat(Xtr, 50, axis=0); ys = Ytr.ravel()
        idn = np.tile(np.arange(50), len(Xtr))
        feat = np.hstack([Xs, idn[:, None]])
        gbm = HistGradientBoostingRegressor(max_iter=60, max_depth=3, learning_rate=0.1).fit(feat, ys)
        xd = np.hstack([np.repeat(RET[:, d][None, :], 50, axis=0), np.arange(50)[:, None]])
        nl = gbm.predict(xd)
        fwd = RET[1:, d+1]
        ics_lin.append(np.corrcoef(lin-lin.mean(), fwd)[0, 1])
        ics_nl.append(np.corrcoef(nl-nl.mean(), fwd)[0, 1])
    return np.mean(ics_lin), np.mean(ics_nl)


print("baseline ridge (idio) on 500-750 = 505 (ceiling 530). Testing other families:\n")
print(f"{'family':34} {'S@500-750':>10} {'S@250-500':>10}")
for name, gp in [("OU single-name reversion W=15", ou_book(15)),
                 ("OU single-name reversion W=30", ou_book(30)),
                 ("cointegration pairs $10k", pairs_book(10000)),
                 ("cointegration pairs $20k", pairs_book(20000))]:
    print(f"{name:34} {run(gp, S, E):10.0f} {run(gp, 250, 500):10.0f}")

lin_ic, nl_ic = nonlinear_ic()
print(f"\nNONLINEAR check on 500-750: linear ridge IC {lin_ic:+.4f}  vs  gradient-boosted IC {nl_ic:+.4f}")
print("(if GBM IC >> linear, nonlinearity is the missing edge; if <=, linear is optimal)")
