"""Model-uncertainty stress test — the honest way to evaluate on data we DON'T have.

The two research efforts disagree on the true mechanism:
  World A (algo26v1): dense lead-lag VAR(1) -> the ridge is optimal.
  World B (algo26v2): sparse cointegration PAIRS -> pair-trading is optimal.
On the one real sample they look alike; on UNSEEN futures they diverge. A strategy tuned
to World A can be fragile if the future is World B, and vice-versa. We generate many
synthetic futures of EACH world and score three strategies on each, valuing the WORST-CASE
across worlds (robustness to which model is right), not the best case on the known file.
"""
import numpy as np, pandas as pd
from dgp_simulator import DGP

real = pd.read_csv("prices.txt", sep=r"\s+").values.T
nInst = 51
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000); dlr[0] = 100_000

# ---------------- strategies ----------------
def _ridge_fit(X, Y, hl=2000, a=0.1):
    n, p = X.shape; lam = 0.5**(1.0/hl); w = lam**np.arange(n-1, -1, -1); sw = w.sum()
    mx = (w[:, None]*X).sum(0)/sw; my = (w[:, None]*Y).sum(0)/sw; Xc, Yc = X-mx, Y-my
    XtWX = Xc.T@(w[:, None]*Xc); XtWY = Xc.T@(w[:, None]*Yc); eps = 1e-8*np.trace(XtWX)/p
    return np.linalg.solve(XtWX+(eps+a)*np.eye(p), XtWY), mx, my

def _ridge_leg(prc, cache):
    ni, t = prc.shape; lp = np.log(prc); ret = lp[:, 1:]-lp[:, :-1]
    if cache["t"] != t: cache["m"] = _ridge_fit(ret[:, :-1].T, ret[1:, 1:].T); cache["t"] = t
    B, mx, my = cache["m"]; pred = my+(ret[:, -1]-mx)@B; return pred-pred.mean(), ret, lp

def _fast_pairs_leg(prc, lp, dollars, k=20, win=90):
    """Fast causal PROPER-cointegration pairs: select by residual mean-reversion strength.

    Raw correlation is dominated by the market factor, so it can't find cointegrated pairs.
    Instead: prefilter on IDIOSYNCRATIC (market-residualised) return correlation, then rank
    candidates by their spread's OU reversion coefficient rho = -corr(dSpread_t, spread_{t-1})
    (strong positive rho => the spread reliably pulls back). Trade the top-k by z-score.
    """
    ni, t = prc.shape; pos = np.zeros(ni)
    if t < win+3: return pos
    L = lp[:, -win:]; R = np.diff(L, axis=1)
    mkt = R[1:].mean(0); denom = (mkt-mkt.mean())@(mkt-mkt.mean())+1e-12
    idio = R[1:] - np.outer(((R[1:]-R[1:].mean(1, keepdims=True))@(mkt-mkt.mean()))/denom, mkt)  # market-residual
    C = np.corrcoef(idio); cur = prc[:, -1]; cand = []
    for a in range(idio.shape[0]):
        for b in range(a+1, idio.shape[0]):
            if abs(C[a, b]) < 0.3: continue
            i, j = a+1, b+1
            beta = np.polyfit(L[j], L[i], 1)[0]; spr = L[i]-beta*L[j]
            ds = np.diff(spr); lev = spr[:-1]-spr[:-1].mean()
            rho = -(ds@lev)/((lev@lev)+1e-12)                 # OU reversion coef (per day)
            if rho > 0.05:
                z = (spr[-1]-spr.mean())/(spr.std()+1e-9)
                cand.append((rho, i, j, beta, z))
    cand.sort(reverse=True); used = set()
    for rho, i, j, beta, z in cand:
        if len(used) >= 2*k: break
        if i in used or j in used or abs(z) <= 1.0: continue
        u = -np.sign(z)
        pos[i] += u*dollars/cur[i]; pos[j] += -u*beta*dollars/cur[j]
        used.add(i); used.add(j)
    return pos

def _algo_rev(prc):
    t = prc.shape[1]; cap = 100000/prc[0, -1]
    if t <= 92: return 0.0
    lpA = np.log(prc[0]); mv = lpA[30:]-lpA[:-30]; z = (mv[-1]-mv[-60:].mean())/(mv[-60:].std()+1e-12)
    return float(np.clip(-np.clip(z, -3, 3)*200000/prc[0, -1], -cap, cap))

def _hedge(pos, ret, prc, cap, rev):
    rA = ret[0]; rAc = rA-rA.mean(); den = rAc@rAc+1e-12
    betas = ((ret[1:]-ret[1:].mean(1, keepdims=True))@rAc)/den
    net = (pos[1:]*prc[1:, -1])@betas; room = max(cap-abs(rev), 0.0)
    pos[0] = rev+float(np.clip(-net/prc[0, -1], -room, room)); return pos

def make(kind, hl=2000):
    c = {"t": None, "m": None}
    def gp(prc):
        ni, t = prc.shape; pos = np.zeros(ni)
        if t < 95: return pos.astype(int)
        lp = np.log(prc); ret = lp[:, 1:]-lp[:, :-1]
        if c["t"] != t: c["m"] = _ridge_fit(ret[:, :-1].T, ret[1:, 1:].T, hl=hl); c["t"] = t
        B, mx, my = c["m"]; pred = my+(ret[:, -1]-mx)@B; w = pred-pred.mean(); cap = 100000/prc[0, -1]
        if kind in ("ridge", "ensemble"):
            s = np.sign(w)*(10000/prc[1:, -1]); pos[1:] += np.where(np.abs(w) >= 0.2*(np.std(w)+1e-12), s, 0.0)
        if kind == "pairs":
            pos += _fast_pairs_leg(prc, lp, 10000)
        if kind == "ensemble":
            pos += _fast_pairs_leg(prc, lp, 6000)          # diversifying pairs sleeve
        rev = _algo_rev(prc) if kind in ("ridge", "ensemble") else 0.0
        return _hedge(pos, ret, prc, cap, rev).astype(int)
    return gp

# ---------------- World B: cointegration-pairs DGP ----------------
def world_b(hist, n_future, seed):
    """Continue `hist` with a PAIRS world: market factor + 20 mean-reverting pair spreads."""
    rng = np.random.default_rng(seed)
    lp = np.log(hist); r0 = lp[1:, -1]-lp[1:, -2]; n = 50
    sig = np.std(np.diff(lp[1:], axis=1), axis=1).mean()
    fut = np.zeros((n, n_future))
    f = rng.standard_normal(n_future)*sig*0.6                 # market factor
    npair = 20; phi = 0.6; s = np.zeros(npair)
    idio = rng.standard_normal((n, n_future))*sig*0.5
    for tt in range(n_future):
        r = 0.9*f[tt] + idio[:, tt]                           # all load on market
        for k in range(npair):
            s[k] = phi*s[k] + rng.standard_normal()*sig*0.7   # OU spread
            r[2*k] += 0.5*s[k]; r[2*k+1] -= 0.5*s[k]          # pair (2k,2k+1) cointegrated
        fut[:, tt] = r
    cp = (hist[1:, -1])[:, None]*np.exp(np.cumsum(fut, axis=1))
    algo = hist[0, -1]*np.exp(np.cumsum(fut.mean(0)))
    return np.hstack([hist, np.vstack([algo[None, :], cp])])

# ---------------- World C: VAR world whose STRUCTURE SHIFTS in the future ----------------
def world_c(dgp, hist, n_future, seed):
    """Fitted VAR, but the future transition matrix is REWIRED (lead-lag links permuted +
    partially sign-flipped) so the structure the ridge trained on is stale. This is the real
    model risk: the edge doesn't vanish, it CHANGES to something not in the training sample."""
    rng = np.random.default_rng(seed)
    A2 = dgp.A.copy()
    perm = rng.permutation(A2.shape[0])
    A2 = 0.5*A2 + 0.5*(-A2[perm][:, perm])          # half old structure, half rewired+flipped
    from dgp_simulator import DGP as _D
    d2 = _D(A2, dgp.Sigma, dgp.p0, dgp.algo_p0, dgp.algo_w, signal_scale=dgp.signal_scale)
    r = d2._gen_returns(n_future, rng, r_init=np.log(hist[1:, -1])-np.log(hist[1:, -2]))
    fut = d2._panel(r, hist[1:, -1], float(hist[0, -1]))
    return np.hstack([hist, fut])

# ---------------- scoring ----------------
def score(pll):
    mu, sd = pll.mean(), pll.std()
    if mu <= 0 or sd < 1e-10: return mu
    sr = np.sqrt(250)*mu/sd; return mu*sr**2/(sr**2+1)

def fwd_score(gpf, panel, start):
    gp = gpf(); cash = 0; cp = np.zeros(nInst); val = 0; cm = 0; pll = []; nt = panel.shape[1]
    for t in range(start, nt+1):
        p = panel[:, :t]; cur = p[:, -1]
        npos = np.clip(gp(p), -(dlr/cur).astype(int), (dlr/cur).astype(int)).astype(int) if t < nt else cp.copy()
        d = npos-cp; cash -= cur.dot(d)+cm; dv = cur*np.abs(d); cm = (dv*commRate).sum(); cp = npos.copy()
        pl = cash+cp.dot(cur)-val; val = cash+cp.dot(cur)
        if t > start: pll.append(pl)
    return score(np.array(pll))

if __name__ == "__main__":
    NSEED, FUT = 16, 220
    dgpA = DGP.fit(real); dgpA.signal_scale = 0.3          # calibrated regime (see dgp_simulator)
    strat = {"ridge HL2000": lambda: make("ridge", hl=2000),
             "ridge HL500 ": lambda: make("ridge", hl=500),
             "ridge HL250 ": lambda: make("ridge", hl=250)}
    T = real.shape[1]
    print(f"Forward MC: {NSEED} unseen futures/world, score last-{FUT} synthetic days.\n")
    for world, gen in [("A: VAR/lead-lag (v1's model)", lambda s: dgpA.extend(real, FUT, s)),
                       ("B: cointegration pairs (v2's model)", lambda s: world_b(real, FUT, s)),
                       ("C: STRUCTURE SHIFTS on unseen data", lambda s: world_c(dgpA, real, FUT, s))]:
        panels = [gen(s) for s in range(NSEED)]
        start = T+1                                        # score only the synthetic future
        print(f"### World {world}")
        res = {}
        for name, gpf in strat.items():
            sc = np.array([fwd_score(gpf, p, start) for p in panels])
            res[name] = sc
            print(f"   {name:9} median {np.median(sc):7.0f}  mean {sc.mean():7.0f}  "
                  f"p10 {np.percentile(sc,10):7.0f}  min {sc.min():7.0f}")
        print()
