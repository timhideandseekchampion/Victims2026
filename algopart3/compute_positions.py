"""Compute per-asset daily positions for SAFE and SWING over all days (count read from
prices.txt), derive entry/exit events (sign flips), and export a compact JSON for the dashboard."""
import json, numpy as np, pandas as pd
import SAFE, SWING, QUAL, SAFE_llalgo, SAFE_lldollar, SAFE_llvol, SAFE_llvol_vo, SAFE_llboost, SAFE_llboost_v2, SAFE_llboost_v3, SAFE_llboost_v4, SAFE_llboost_v5, SAFE_llboost_v6, SAFE_llboost_v7

prc = pd.read_csv("prices.txt", sep=r"\s+", header=0)
names = list(prc.columns)
P = prc.values.T
nInst, nt = P.shape

commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5

def positions(mod):
    pos = np.zeros((nInst, nt), dtype=int)
    for t in range(131, nt + 1):
        pos[:, t - 1] = mod.getMyPosition(P[:, :t])
    return pos

def per_asset_pnl(pos):
    """cumulative PnL per asset: hold pos[:,d] from day d into d+1; fees on position changes."""
    pnl = np.zeros((nInst, nt))
    for d in range(1, nt):
        mkt = pos[:, d - 1] * (P[:, d] - P[:, d - 1])
        fee = commRate * np.abs(pos[:, d] - pos[:, d - 1]) * P[:, d]
        pnl[:, d] = pnl[:, d - 1] + mkt - fee
    return pnl

def algo_skew():
    """per-day ALGO lead-lag gate diagnostics: frac = mean(sign(wz)) (the skew that drives the
    gated ALGO leg) and the direction the reversion leg would take. Mirrors SAFE_llalgo exactly."""
    logp = np.log(P); ENS = [250, 500, 1000, 2000]
    frac = np.zeros(nt); rev_dir = np.zeros(nt)
    for t in range(131, nt + 1):
        lp = logp[:, :t]; r = lp[:, 1:] - lp[:, :-1]
        X = r[:, :-1].T; Y = r[1:, 1:].T; xin = r[:, -1]; n = X.shape[0]
        fs = []
        for hl in ENS:
            lam = 0.5**(1/hl); w = lam**np.arange(n-1, -1, -1); sw = w.sum()
            mx = (w[:, None]*X).sum(0)/sw; my = (w[:, None]*Y).sum(0)/sw
            Xc = X-mx; Yc = Y-my
            eps = 1e-8*np.trace(Xc.T@(w[:, None]*Xc))/X.shape[1]
            B = np.linalg.solve(Xc.T@(w[:, None]*Xc)+(eps+0.1)*np.eye(nInst), Xc.T@(w[:, None]*Yc))
            f = my+(xin-mx)@B; d = f-f.mean(); fs.append(d/(d.std()+1e-12))
        wz = np.mean(fs, 0)
        rr = logp[1:, t-1]-logp[1:, t-1-10]; rr = rr-rr.mean(); rv = -rr/(rr.std()+1e-12)
        wz = 0.7*wz + 0.3*rv
        frac[t-1] = float(np.mean(np.sign(wz)))
        lpA = logp[0, :t]; mv = lpA[30:]-lpA[:-30]
        z = (mv[-1]-mv[-60:].mean())/(mv[-60:].std()+1e-12)
        rev_dir[t-1] = float(-np.sign(np.clip(z, -3, 3)))
    return frac, rev_dir

print("computing ALGO skew-gate diagnostics ...")
_frac, _revdir = algo_skew()

out = {"names": names, "nt": nt,
       "prices": [[round(float(x), 2) for x in P[i]] for i in range(nInst)],
       "algo_idx": 0,
       "algo_skew": {"frac": [round(float(x), 3) for x in _frac],
                     "rev_dir": [int(x) for x in _revdir],
                     "gate": SAFE_llalgo.ALGO_LL_GATE, "nInst": nInst}}
for label, mod in [("SAFE", SAFE), ("SWING", SWING), ("QUAL", QUAL),
                   ("LLALGO", SAFE_llalgo), ("LLDOLLAR", SAFE_lldollar),
                   ("LLVOL", SAFE_llvol), ("LLVOL_VO", SAFE_llvol_vo),
                   ("LLBOOST", SAFE_llboost), ("LLBOOST_V2", SAFE_llboost_v2),
                   ("LLBOOST_V3", SAFE_llboost_v3), ("LLBOOST_V4", SAFE_llboost_v4),
                   ("LLBOOST_V5", SAFE_llboost_v5), ("LLBOOST_V6", SAFE_llboost_v6),
                   ("LLBOOST_V7", SAFE_llboost_v7)]:
    print("computing", label, "...")
    pos = positions(mod)
    sign = np.sign(pos).astype(int)                       # -1 short, 0 flat, +1 long
    # entry/exit events per asset: day where sign changes
    events = []
    flips_total = 0
    for i in range(nInst):
        evs = []
        prev = 0
        for d in range(nt):
            s = sign[i, d]
            if s != prev and s != 0:
                evs.append([d, int(s)])                   # [day, +1 long-entry / -1 short-entry]
                flips_total += 1
            prev = s
        events.append(evs)
    pnl = per_asset_pnl(pos)
    out[label] = {"sign": [[int(x) for x in sign[i]] for i in range(nInst)],
                  "events": events,
                  "algo_dollar": [round(float(pos[0, d] * P[0, d])) for d in range(nt)],  # row-0 $ notional
                  "pnl": [[round(float(x), 1) for x in pnl[i]] for i in range(nInst)],
                  "pnl_final": [round(float(pnl[i, -1]), 1) for i in range(nInst)]}
    print(f"  {label}: {flips_total} entries (~{flips_total/nInst:.1f}/asset); "
          f"total PnL ${pnl[:,-1].sum():,.0f}; winners {int((pnl[:,-1]>0).sum())}/{nInst}")

json.dump(out, open("positions_data.json", "w"))
print("wrote positions_data.json", f"({len(json.dumps(out))//1024} KB)")
