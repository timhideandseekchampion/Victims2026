"""Compute the ALGO-signal investigation dataset for signals.html:
  * cross-sectional vol->next-return IC for all 51 instruments (is the vol edge a generator
    property or idiosyncratic to ALGO?),
  * the live trailing IC of the vol signal vs the lead-lag net-$ signal over time (what the
    adaptive gate 'sees'),
  * a head-to-head of combining LLVOL + LLMATCH.
Exports signals_data.json."""
import json, numpy as np, pandas as pd, math, SAFE
P = pd.read_csv("prices.txt", sep=r"\s+", header=0); names = list(P.columns); P = P.values.T.astype(float)
nInst, nt = P.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
IC_L = 250

def volz_ret(series):
    lp = np.log(series); r = np.diff(lp); T = len(lp)
    vol = np.full(T, np.nan)
    for t in range(20, T): vol[t] = r[t - 20:t].std()
    z = np.full(T, np.nan)
    for t in range(80, T):
        w = vol[t - 60:t]; z[t] = (vol[t] - w.mean()) / (w.std() + 1e-12)
    ret1 = np.full(T, np.nan); ret1[:T - 1] = lp[1:] - lp[:-1]
    return z, ret1

def ic(z, ret1, s, e):
    idx = np.arange(0, len(z) - 1); m = (idx >= s) & (idx <= e) & ~np.isnan(z[:len(z) - 1])
    x = z[:len(z) - 1][m]; y = ret1[:len(z) - 1][m]
    if len(x) < 40 or x.std() < 1e-12: return float("nan")
    return float(np.corrcoef(x, y)[0, 1])

# ---- A) cross-sectional vol->return IC + per-name half-sample persistence ----
icf = []; icn = []; ih1 = []; ih2 = []
for i in range(nInst):
    z, ret1 = volz_ret(P[i])
    icf.append(ic(z, ret1, 1, nt)); icn.append(ic(z, ret1, 751, nt))
    ih1.append(ic(z, ret1, 1, 500)); ih2.append(ic(z, ret1, 501, nt))
icf = np.array(icf); icn = np.array(icn); ih1 = np.array(ih1); ih2 = np.array(ih2)
persist = {"h1": [round(float(x), 4) for x in ih1], "h2": [round(float(x), 4) for x in ih2],
           "corr": round(float(np.corrcoef(ih1, ih2)[0, 1]), 3)}
def xs_summary(a):
    return {"mean": round(float(a.mean()), 4), "median": round(float(np.median(a)), 4),
            "pos": int((a > 0).sum()), "n": len(a),
            "tcross": round(float(a.mean() / (a.std(ddof=1) / math.sqrt(len(a)))), 2)}
algo_pctile = round(100 * float((icf < icf[0]).mean()))

# ---- ALGO-specific robustness: circular-shift surrogate p + block stability ----
zA, retA = volz_ret(P[0])
def shift_p(s, e, N=4000):
    idx = np.arange(0, len(zA) - 1); m = (idx >= s) & (idx <= e) & ~np.isnan(zA[:len(zA) - 1])
    x = zA[:len(zA) - 1][m]; y = retA[:len(zA) - 1][m]; n = len(x); obs = np.corrcoef(x, y)[0, 1]
    rng = np.random.RandomState(1); null = np.empty(N)
    for i in range(N):
        sh = rng.randint(20, n - 20); null[i] = np.corrcoef(x, np.roll(y, sh))[0, 1]
    return round(float(np.mean(np.abs(null) >= abs(obs))), 4)
shift_full, shift_new = shift_p(1, nt), shift_p(751, nt)
blocks = []
for a in range(100, nt - 1, 200):
    b = min(a + 200, nt - 1); idx = np.arange(0, nt - 1); m = (idx >= a) & (idx < b) & ~np.isnan(zA[:nt - 1])
    if m.sum() < 50: continue
    blocks.append({"a": a, "b": b, "ic": round(float(np.corrcoef(zA[:nt - 1][m], retA[:nt - 1][m])[0, 1]), 3)})

# ---- B) live trailing IC of vol vs lead-lag net-$ ----
cur0 = P[0]; logp = np.log(P[0]); r = np.diff(logp)
ret1 = np.full(nt, np.nan); ret1[:nt - 1] = logp[1:] - logp[:-1]
vol = np.full(nt, np.nan)
for t in range(20, nt): vol[t] = r[t - 20:t].std()
volz = np.full(nt, np.nan)
for t in range(80, nt):
    w = vol[t - 60:t]; volz[t] = (vol[t] - w.mean()) / (w.std() + 1e-12)
IDIO = np.zeros((nInst, nt)); NETDOL = np.zeros(nt)
for k in range(130, nt):
    cur = P[:, k]; lim = (dlr / cur).astype(int)
    p = np.clip(np.asarray(SAFE.getMyPosition(P[:, :k + 1])), -lim, lim).astype(int); p[0] = 0
    IDIO[:, k] = p; NETDOL[k] = float((p[1:] * cur[1:]).sum())
def trail_ic(f, t, L=IC_L):
    s0 = max(0, t - L); xs = f[s0:t]; ys = ret1[s0:t]; ok = ~np.isnan(xs) & ~np.isnan(ys)
    if ok.sum() < 60: return 0.0
    xs, ys = xs[ok], ys[ok]
    return 0.0 if xs.std() < 1e-12 else float(np.corrcoef(xs, ys)[0, 1])
days = list(range(IC_L + 20, nt))
ICV = [round(trail_ic(volz, t), 4) for t in days]
ICM = [round(trail_ic(NETDOL, t), 4) for t in days]

# ---- C) combine head-to-head ----
lim0 = (dlr[0] / cur0).astype(int); fhv = np.clip(volz, -3, 3) / 3.0; fhm = np.clip(NETDOL / 100000.0, -1, 1)
icv_t = np.array([trail_ic(volz, t) for t in range(nt)]); icm_t = np.array([trail_ic(NETDOL, t) for t in range(nt)])
def to_row(av): return np.clip(np.clip(av / cur0, -(dlr[0] / cur0), (dlr[0] / cur0)), -lim0, lim0).astype(int)
def score(mu, sd):
    if mu <= 0 or sd < 1e-10: return float(mu)
    sr = np.sqrt(250) * mu / sd; return float(mu * sr ** 2 / (sr ** 2 + 1))
def score_win(row0, S, E):
    Pk = IDIO.copy(); Pk[0, :] = row0; curPos = np.zeros(nInst); comm = np.zeros(nInst); prev = None; pl = []
    for tt in range(S, E + 1):
        cur = P[:, tt - 1]; newPos = Pk[:, tt - 1].copy() if tt < E else curPos.copy()
        if tt > S: pl.append(float((curPos * (cur - prev) - comm).sum()))
        dP = newPos - curPos; comm = commRate * np.abs(dP) * cur; prev = cur; curPos = newPos
    pl = np.array(pl); return score(pl.mean(), pl.std())
ED = list(range(400, nt + 1, 10))
volA = np.zeros(nt); matA = np.zeros(nt); naive = np.zeros(nt); icw = np.zeros(nt)
for t in range(130, nt):
    v = 15 * max(0, icv_t[t]) * np.nan_to_num(fhv[t]) * 100000.0; m = NETDOL[t]
    volA[t] = np.clip(v, -dlr[0], dlr[0]); matA[t] = np.clip(m, -dlr[0], dlr[0])
    naive[t] = np.clip(v + m, -dlr[0], dlr[0])
    icw[t] = np.clip(15 * (max(0, icv_t[t]) * np.nan_to_num(fhv[t]) + max(0, icm_t[t]) * fhm[t]) * 100000.0, -dlr[0], dlr[0])
combine = []
for nm, av in [("LLVOL only", volA), ("LLMATCH only", matA), ("naive sum", naive), ("IC-weighted", icw)]:
    row = to_row(av); v = [score_win(row, E - 250, E) for E in ED]
    combine.append({"nm": nm, "OLD": round(score_win(row, 500, 750), 1), "NEW": round(score_win(row, 750, 1000), 1),
                    "mean": round(float(np.mean(v)), 1), "floor": round(float(min(v)), 1)})

out = {"names": names, "nt": nt, "algo_idx": 0,
       "xsec": {"full": [round(float(x), 4) for x in icf], "new": [round(float(x), 4) for x in icn]},
       "xsec_summary": {"full": xs_summary(icf), "new": xs_summary(icn)},
       "algo": {"full_ic": round(float(icf[0]), 3), "new_ic": round(float(icn[0]), 3), "pctile": algo_pctile,
                "shift_p_full": shift_full, "shift_p_new": shift_new, "blocks": blocks},
       "persist": persist,
       "live": {"days": days, "icv": ICV, "icm": ICM},
       "windows": {"OLD": [501, 750], "NEW": [751, nt]}, "combine": combine}
json.dump(out, open("signals_data.json", "w"))
print("wrote signals_data.json")
print(f"cross-sectional vol IC: full mean {out['xsec_summary']['full']['mean']} "
      f"({out['xsec_summary']['full']['pos']}/51 >0, t={out['xsec_summary']['full']['tcross']})")
print(f"ALGO full IC {out['algo']['full_ic']} = {out['algo']['pctile']}th percentile of 51 names")
for c in combine: print(f"  {c['nm']:<14} OLD {c['OLD']} NEW {c['NEW']} mean {c['mean']} floor {c['floor']}")
