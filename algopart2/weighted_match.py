"""weighted_match.py — size the ALGO leg to the book's true ALGO-EQUIVALENT dollar exposure,
weighting each stock's $ position by how much it drives the index (its beta to ALGO), instead of
a flat $50k or the equal-weight dollar sum net$.

  E_beta = sum_i ( stock_$_position_i * beta_i_to_ALGO )   # the market bet the book really carries
This is the rigorous "expected dollar position equivalent based on all the weights" (beta folds in
index weight, vol and correlation). Compare matching E_beta vs full $100k vs match net$."""
import numpy as np, pandas as pd
prc = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nDays = prc.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
logp = np.log(prc); r_all = logp[:, 1:]-logp[:, :-1]
ENS = [250, 500, 1000, 2000]; CONTRA_DOL = 1_000_000; GATE = 50_000; BW = 120

def daily(t):
    lp = logp[:, :t]; r = lp[:, 1:] - lp[:, :-1]; cur = prc[:, t-1]
    X = r[:, :-1].T; Y = r[1:, 1:].T; xin = r[:, -1]; n = X.shape[0]; fs = []
    for hl in ENS:
        lam = 0.5**(1/hl); w = lam**np.arange(n-1, -1, -1); sw = w.sum()
        mx = (w[:, None]*X).sum(0)/sw; my = (w[:, None]*Y).sum(0)/sw; Xc = X-mx; Yc = Y-my
        eps = 1e-8*np.trace(Xc.T@(w[:, None]*Xc))/X.shape[1]
        B = np.linalg.solve(Xc.T@(w[:, None]*Xc)+(eps+0.1)*np.eye(nInst), Xc.T@(w[:, None]*Yc))
        f = my+(xin-mx)@B; d = f-f.mean(); fs.append(d/(d.std()+1e-12))
    wz = np.mean(fs, 0)
    rr = logp[1:, t-1]-logp[1:, t-1-10]; rr = rr-rr.mean(); rv = -rr/(rr.std()+1e-12); wz = 0.7*wz+0.3*rv
    idio = np.sign(wz)*(dlr[1:]/cur[1:]); il = (dlr[1:]/cur[1:]).astype(int); idio = np.clip(idio, -il, il).astype(int)
    idio_dol = idio*cur[1:]; net = float(idio_dol.sum()); cap = dlr[0]/cur[0]
    # causal betas of each stock to ALGO over trailing BW days
    s = max(0, t-1-BW); rA = r_all[0, s:t-1]; RS = r_all[1:, s:t-1]
    rAc = rA - rA.mean(); den = (rAc@rAc) + 1e-12
    beta = (RS - RS.mean(1, keepdims=True)) @ rAc / den
    E_beta = float(idio_dol @ beta)                                # ALGO-equivalent $ exposure of the book
    lpA = logp[0, :t]; mv = lpA[30:]-lpA[:-30]; z = (mv[-1]-mv[-60:].mean())/(mv[-60:].std()+1e-12)
    rev = -np.clip(z, -3, 3)/3.0*CONTRA_DOL
    return idio, net, E_beta, float(beta.mean()), rev, cap, cur

print("precomputing ..."); SIG = {t: daily(t) for t in range(120, nDays+1)}

# diagnostic: do betas differ enough that E_beta != net$?
nets = np.array([SIG[t][1] for t in range(131, nDays+1)])
Ebs  = np.array([SIG[t][2] for t in range(131, nDays+1)])
bmn  = np.array([SIG[t][3] for t in range(131, nDays+1)])
print(f"\nmean beta across stocks ~ {bmn.mean():.2f}   corr(E_beta, net$) = {np.corrcoef(Ebs, nets)[0,1]:.3f}   "
      f"median |E_beta|/|net$| = {np.median(np.abs(Ebs)/(np.abs(nets)+1e-9)):.2f}\n")

def pos_for(t, mode):
    idio, net, E_beta, _, rev, cap, cur = SIG[t]; p = np.zeros(nInst); p[1:] = idio; c0 = cur[0]
    on = abs(net) >= GATE
    if not on:               dol = rev
    elif mode == "full":     dol = np.sign(net)*CONTRA_DOL
    elif mode == "match":    dol = net
    elif mode == "matchbeta":dol = E_beta                          # match ALGO-equivalent exposure
    elif mode == "beta_gate":                                       # gate & size both on E_beta
        dol = E_beta if abs(E_beta) >= GATE else rev
    p[0] = np.clip(dol/c0, -cap, cap); lim = (dlr/cur).astype(int)
    return np.clip(p, -lim, lim).astype(int)

def score(mu, sd):
    if mu <= 0 or sd < 1e-10: return mu
    sr = np.sqrt(250)*mu/sd; return mu*sr**2/(sr**2+1)
def metrics(mode, S, E):
    cash=0.0; cp=np.zeros(nInst); value=0.0; comm=0.0; pll=[]
    for t in range(S, E+1):
        cur = prc[:, t-1]; nP = pos_for(t, mode) if t < E else cp
        dP = nP-cp; cash -= cur.dot(dP)+comm; comm = np.sum(cur*np.abs(dP)*commRate); cp = nP
        pl = cash+cp.dot(cur)-value; value = cash+cp.dot(cur)
        if t > S: pll.append(pl)
    pll = np.array(pll); mu, sd = pll.mean(), pll.std(); return mu, sd, np.sqrt(250)*mu/sd, score(mu, sd)

ENDS = list(range(380, nDays+1, 10))
print(f"{'sizing':<30}{'mean':>8}{'std':>9}{'Sharpe':>8}{'score':>8}   [rolling score]")
for mode, lbl in [("full","full $100k (current)"),("match","match net$ (equal-wt)"),
                  ("matchbeta","match ALGO-equiv (beta-wt)"),("beta_gate","gate+size on ALGO-equiv")]:
    m = metrics(mode, 501, 750); r = np.array([metrics(mode, E-250, E) for E in ENDS]).mean(0)
    print(f"{lbl:<30}{m[0]:>8.1f}{m[1]:>9.1f}{m[2]:>8.2f}{m[3]:>8.1f}      {r[3]:>8.1f}")
