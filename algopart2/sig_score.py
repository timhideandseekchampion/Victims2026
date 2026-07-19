"""sig_score.py — the honest test of 'which signal is best': not IC, but SCORE when actually
traded (sign-sized idio book + the same net-$ ALGO gate as SAFE_lldollar). Compares each signal
head-to-head so we can see whether volsc's marginally-higher IC survives as PnL. (It should not:
score rewards full deployment; IC does not.)"""
import numpy as np, pandas as pd
import SAFE_rotate as R

prc = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nDays = prc.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
R._ensure_cache(prc)

def score(mu, sd):
    if mu <= 0 or sd < 1e-10: return mu
    sr = np.sqrt(250) * mu / sd; return mu * sr**2 / (sr**2 + 1)

def pos_for(t, name):
    """trade signal `name` exactly like SAFE_lldollar: sign-sized idio + net-$ ALGO gate."""
    cur = prc[:, t-1]; wz = R._SIG[t][name]; pos = np.zeros(nInst)
    pos[1:] = np.sign(wz) * (dlr[1:] / cur[1:])
    idio_lim = (dlr[1:] / cur[1:]).astype(int)
    idio_int = np.clip(pos[1:], -idio_lim, idio_lim).astype(int)
    net_dol = float((idio_int * cur[1:]).sum())
    cap = dlr[0] / cur[0]
    if abs(net_dol) >= R.ALGO_LL_DOLLAR:
        av = float(np.sign(net_dol) * cap)
    else:
        lpA = np.log(prc[0, :t]); mv = lpA[R.CONTRA_K:] - lpA[:-R.CONTRA_K]
        z = (mv[-1] - mv[-R.CONTRA_WZ:].mean()) / (mv[-R.CONTRA_WZ:].std() + 1e-12)
        av = float(np.clip(-np.clip(z,-3,3)/3.0*(R.CONTRA_DOL/cur[0]), -cap, cap))
    pos[0] = av
    lim = (dlr / cur).astype(int); return np.clip(pos, -lim, lim).astype(int)

def run(name, S, E):
    cash=0.0; cp=np.zeros(nInst); value=0.0; comm=0.0; pll=[]
    for t in range(S, E+1):
        cur=prc[:,t-1]; newPos = pos_for(t, name) if t<E else cp
        dP=newPos-cp; cash-=cur.dot(dP)+comm; comm=np.sum(cur*np.abs(dP)*commRate); cp=newPos
        pl=cash+cp.dot(cur)-value; value=cash+cp.dot(cur)
        if t>S: pll.append(pl)
    pll=np.array(pll); return pll.mean(), pll.std(), score(pll.mean(), pll.std())

names = ["champ", "momJT", "residMom", "momVS", "mom", "volsc", "ll2", "revL", "resid"]
print("500-750 (graded leg): signal traded sign-sized + net-$ ALGO gate")
print(f"{'signal':<8}{'meanIC':>8}{'mean$':>9}{'std$':>9}{'Sharpe':>8}{'SCORE':>8}")
for nm in names:
    ic = R._ic(nm, 500, 750).mean()
    mu, sd, sc = run(nm, 500, 750)
    sr = np.sqrt(250)*mu/sd
    star = "  <-- shipped (champion)" if nm=="champ" else ("  <-- highest IC" if nm=="volsc" else "")
    print(f"{nm:<8}{ic:>+8.4f}{mu:>9.1f}{sd:>9.1f}{sr:>8.2f}{sc:>8.0f}{star}")

print("\nrolling 250d windows (mean SCORE / floor):")
ends = list(range(400, nDays+1, 25))
print(f"{'signal':<8}{'meanSCORE':>11}{'floor':>8}")
for nm in names:
    ss = np.array([run(nm, e-250, e)[2] for e in ends if e-250 >= R.WARMUP])
    print(f"{nm:<8}{ss.mean():>11.0f}{ss.min():>8.0f}")
