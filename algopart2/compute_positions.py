"""Compute per-asset daily positions for SAFE and SWING over all 750 days, derive entry/exit
events (sign flips), and export a compact JSON for the dashboard."""
import json, numpy as np, pandas as pd
import SAFE, SWING

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

out = {"names": names, "nt": nt,
       "prices": [[round(float(x), 2) for x in P[i]] for i in range(nInst)],
       "algo_idx": 0}
for label, mod in [("SAFE", SAFE), ("SWING", SWING)]:
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
                  "pnl": [[round(float(x), 1) for x in pnl[i]] for i in range(nInst)],
                  "pnl_final": [round(float(pnl[i, -1]), 1) for i in range(nInst)]}
    print(f"  {label}: {flips_total} entries (~{flips_total/nInst:.1f}/asset); "
          f"total PnL ${pnl[:,-1].sum():,.0f}; winners {int((pnl[:,-1]>0).sum())}/{nInst}")

json.dump(out, open("positions_data.json", "w"))
print("wrote positions_data.json", f"({len(json.dumps(out))//1024} KB)")
