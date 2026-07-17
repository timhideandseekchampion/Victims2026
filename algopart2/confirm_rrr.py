"""confirm_rrr.py — is reduced-rank ridge (rank 5) a REAL improvement over plain ridge, or window-fit?
Test IC on 400-500, 500-750, 250-500 separately (step=1, proper t), then backtest SCORE with the
RRR forecast swapped into the book and compare to plain ridge across all 250-day legs."""
import numpy as np, pandas as pd
prc = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nt = prc.shape
lp = np.log(prc); RET = lp[:, 1:] - lp[:, :-1]
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0

def design(d, hl=1000):
    X = RET[:, :d - 1].T; Y = RET[1:, 1:d].T; xin = RET[:, d - 1]
    n = X.shape[0]; lam = 0.5 ** (1.0 / hl); w = lam ** np.arange(n - 1, -1, -1); sw = w.sum()
    mx = (w[:, None] * X).sum(0) / sw; my = (w[:, None] * Y).sum(0) / sw
    return (X - mx), (Y - my), w, mx, my, xin

_pl = {}; _rr = {}
def ridge_fc(d, a=0.3):
    if d in _pl: return _pl[d]
    Xc, Yc, w, mx, my, xin = design(d); p = Xc.shape[1]; Wm = w[:, None]
    B = np.linalg.solve(Xc.T @ (Wm * Xc) + a * np.eye(p), Xc.T @ (Wm * Yc))
    f = my + (xin - mx) @ B; _pl[d] = f - f.mean(); return _pl[d]
def rrr_fc(d, a=0.3, k=5):
    key = (d, k)
    if key in _rr: return _rr[key]
    Xc, Yc, w, mx, my, xin = design(d); p = Xc.shape[1]; Wm = w[:, None]
    B = np.linalg.solve(Xc.T @ (Wm * Xc) + a * np.eye(p), Xc.T @ (Wm * Yc))
    Yhat = Xc @ B; M = Yhat.T @ (Wm * Yhat)
    ev, V = np.linalg.eigh(M); Pk = V[:, -k:]
    Brr = B @ Pk @ Pk.T
    f = my + (xin - mx) @ Brr; _rr[key] = f - f.mean(); return _rr[key]

def ic(fn, S, E):
    xs = []
    for d in range(S, min(E, nt - 1)):
        if d < 96: continue
        s = fn(d); fwd = RET[1:, d]
        if s.std() > 1e-12 and fwd.std() > 1e-12: xs.append(np.corrcoef(s, fwd)[0, 1])
    xs = np.array(xs); return xs.mean(), xs.mean() / (xs.std(ddof=1) / np.sqrt(len(xs)))

print("IC by window (step=1, proper t):")
print(f"{'window':<12}{'ridge IC(t)':>18}{'RRR k5 IC(t)':>18}{'RRR k4':>10}{'RRR k6':>10}")
for lbl, S, E in [("250-500", 250, 500), ("400-500", 400, 500), ("500-750", 500, 750), ("400-750", 400, 749)]:
    ri, rt = ic(ridge_fc, S, E)
    a5, t5 = ic(lambda d: rrr_fc(d, k=5), S, E)
    a4, _ = ic(lambda d: rrr_fc(d, k=4), S, E)
    a6, _ = ic(lambda d: rrr_fc(d, k=6), S, E)
    print(f"{lbl:<12}{ri:8.4f}({rt:4.1f})   {a5:8.4f}({t5:4.1f})   {a4:8.4f}  {a6:8.4f}")

# ---- backtest SCORE: same book, ridge vs RRR forecast ----
def book(fc_fn, Sd, Ed, blend=0.3):
    cash = 0.0; cp = np.zeros(nInst); value = 0.0; comm = 0.0; pll = []
    for t in range(Sd, Ed + 1):
        cur = prc[:, t - 1]; pos = np.zeros(nInst)
        if t < Ed and t >= 96:
            f = fc_fn(t); wz = f / (f.std() + 1e-12)
            rr = lp[1:, t - 1] - lp[1:, t - 11]; rr = rr - rr.mean(); z = -rr / (rr.std() + 1e-12)
            wz = (1 - blend) * wz + blend * z
            pos[1:] = np.sign(wz) * (dlr[1:] / cur[1:])
            cap = dlr[0] / cur[0]
            lpA = np.log(prc[0, :t]); mv = lpA[30:] - lpA[:-30]
            zz = (mv[-1] - mv[-60:].mean()) / (mv[-60:].std() + 1e-12)
            av = float(np.clip(-np.clip(zz, -3, 3) / 3.0 * (1_000_000 / cur[0]), -cap, cap))
            r = RET[:, 1:t]; rA = r[0] - r[0].mean()
            bet = ((r[1:] - r[1:].mean(1, keepdims=True)) @ rA) / (rA @ rA + 1e-12)
            hs = -((pos[1:] * cur[1:]) @ bet) / cur[0]
            room = max(cap - abs(av), 0.0); pos[0] = av + float(np.clip(hs, -room, room))
            lim = (dlr / cur).astype(int); pos = np.clip(pos, -lim, lim).astype(int)
        else:
            pos = cp.copy()
        dp = pos - cp; cash -= cur.dot(dp) + comm
        comm = np.sum(cur * np.abs(dp) * commRate); cp = pos.copy()
        plt = cash + cp.dot(cur) - value; value = cash + cp.dot(cur)
        if t > Sd: pll.append(plt)
    pll = np.array(pll); mu, sd = pll.mean(), pll.std()
    if mu <= 0 or sd < 1e-10: return mu
    sr = np.sqrt(250) * mu / sd; return mu * sr**2 / (sr**2 + 1)

print("\nSCORE by 250-day leg (full book, sign sizing, hedge):")
print(f"{'leg':<12}{'ridge':>9}{'RRR k5':>9}{'delta':>8}")
legs = [(S, S + 250) for S in range(250, 501, 50)]
tot_r = tot_x = 0
for S, E in legs:
    sr_ = book(ridge_fc, S, E); sx_ = book(lambda d: rrr_fc(d, k=5), S, E)
    tot_r += sr_; tot_x += sx_
    print(f"{f'{S}-{E}':<12}{sr_:9.0f}{sx_:9.0f}{sx_-sr_:8.0f}")
print(f"{'mean':<12}{tot_r/len(legs):9.0f}{tot_x/len(legs):9.0f}{(tot_x-tot_r)/len(legs):8.0f}")
