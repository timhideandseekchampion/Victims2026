"""validate_gate.py — robustness of the skew-gated lead-lag ALGO leg across ALL rolling
250-day windows (finalize.py-style), not just 3 hand-picked ones. Ridge computed ONCE per day
and cached, then baseline vs gated positions assembled cheaply and scored per window."""
import numpy as np, pandas as pd
prc = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nDays = prc.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
logp = np.log(prc)
ENS = [250, 500, 1000, 2000]
FRAC_SCALE, CONTRA_DOL = 0.09, 1_000_000

def daily(t):
    """compute ridge ONCE for day t; return idio positions + ALGO leg components."""
    lp = logp[:, :t]; r = lp[:, 1:] - lp[:, :-1]; cur = prc[:, t-1]
    X = r[:, :-1].T; Y = r[1:, 1:].T; xin = r[:, -1]; n = X.shape[0]
    fs = []
    for hl in ENS:
        lam = 0.5**(1/hl); w = lam**np.arange(n-1,-1,-1); sw = w.sum()
        mx = (w[:,None]*X).sum(0)/sw; my = (w[:,None]*Y).sum(0)/sw
        Xc = X-mx; Yc = Y-my
        eps = 1e-8*np.trace(Xc.T@(w[:,None]*Xc))/X.shape[1]
        B = np.linalg.solve(Xc.T@(w[:,None]*Xc)+(eps+0.1)*np.eye(nInst), Xc.T@(w[:,None]*Yc))
        f = my+(xin-mx)@B; d = f-f.mean(); fs.append(d/(d.std()+1e-12))
    wz = np.mean(fs, 0)
    rr = logp[1:, t-1]-logp[1:, t-1-10]; rr = rr-rr.mean(); rv = -rr/(rr.std()+1e-12)
    wz = 0.7*wz + 0.3*rv
    idio = np.sign(wz) * (dlr[1:]/cur[1:])
    cap = dlr[0]/cur[0]; notl = CONTRA_DOL/cur[0]
    frac = float(np.mean(np.sign(wz)))
    ll_av = np.clip(frac/FRAC_SCALE, -3, 3)/3.0*notl
    lpA = logp[0, :t]; mv = lpA[30:]-lpA[:-30]
    z = (mv[-1]-mv[-60:].mean())/(mv[-60:].std()+1e-12)
    rev_av = -np.clip(z,-3,3)/3.0*notl
    return idio, frac, ll_av, rev_av, cap, cur

print("precomputing daily signals (ridge once/day)...")
SIG = {t: daily(t) for t in range(120, nDays+1)}

def pos_for(t, gate):
    idio, frac, ll_av, rev_av, cap, cur = SIG[t]
    p = np.zeros(nInst); p[1:] = idio
    if gate is None:                       # baseline: reversion
        av = rev_av
    elif abs(frac) >= gate:                # gate ON: lead-lag
        av = ll_av
    else:                                  # gate OFF: reversion default
        av = rev_av
    p[0] = np.clip(av, -cap, cap)
    lim = (dlr/cur).astype(int)
    return np.clip(p, -lim, lim).astype(int)

def score(mu, sd):
    if mu <= 0 or sd < 1e-10: return mu
    sr = np.sqrt(250)*mu/sd; return mu*sr**2/(sr**2+1)

def window_score(gate, S, E):
    cash=0.0; cp=np.zeros(nInst); value=0.0; comm=0.0; pll=[]
    for t in range(S, E+1):
        cur = prc[:, t-1]
        newPos = pos_for(t, gate) if t < E else cp
        dP = newPos-cp; cash -= cur.dot(dP)+comm
        comm = np.sum(cur*np.abs(dP)*commRate); cp = newPos
        pl = cash+cp.dot(cur)-value; value = cash+cp.dot(cur)
        if t > S: pll.append(pl)
    pll = np.array(pll); return score(pll.mean(), pll.std())

# all rolling 250-day windows across the data
ENDS = list(range(380, nDays+1, 10))
print(f"\nscoring {len(ENDS)} rolling 250-day windows (test end days {ENDS[0]}..{ENDS[-1]}, step 10)\n")
print(f"{'config':<22}{'mean':>8}{'median':>8}{'worst':>8}{'best':>8}{'>=700':>8}{'beats base':>11}")
base = np.array([window_score(None, E-250, E) for E in ENDS])
for name, gate in [("BASELINE reversion", None), ("GATED |frac|>=0.12", 0.12), ("GATED |frac|>=0.16", 0.16)]:
    sc = np.array([window_score(gate, E-250, E) for E in ENDS])
    beats = int((sc > base + 1e-6).sum()) if gate is not None else len(sc)
    print(f"{name:<22}{sc.mean():>8.0f}{np.median(sc):>8.0f}{sc.min():>8.0f}{sc.max():>8.0f}"
          f"{int((sc>=700).sum()):>8}{beats:>8}/{len(sc)}")
