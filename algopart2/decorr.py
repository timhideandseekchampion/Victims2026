"""
decorr.py — test the research's #1 idea: extend the ensemble along DECORRELATED axes
(ridge-penalty perturbation x predictor-set) instead of only half-lives, and see if it beats
the pure-HL SAFE ensemble on the FLOOR/consistency metric. Also report the mean pairwise error
correlation rho among members (Bates-Granger floor = sigma^2 * rho). Verified book engine
(matches eval.py: SAFE=611, part2=604). alpha and structure identical to the ships.
"""
import numpy as np, pandas as pd
prc = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nt = prc.shape
lp_all = np.log(prc)
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0

_rc = {}
def ridge_z(t, hl, a, predset):
    key = (t, hl, a, predset)
    if key in _rc: return _rc[key]
    lp = lp_all[:, :t]; r = lp[:, 1:] - lp[:, :-1]
    X = r[:, :-1].T; Y = r[1:, 1:].T; xin = r[:, -1]
    if predset == "noalgo": X = X[:, 1:]; xin = xin[1:]
    n = X.shape[0]; lam = 0.5 ** (1.0 / hl); w = lam ** np.arange(n - 1, -1, -1); sw = w.sum()
    mx = (w[:, None] * X).sum(0) / sw; my = (w[:, None] * Y).sum(0) / sw
    Xc = X - mx; Yc = Y - my; p = Xc.shape[1]
    B = np.linalg.solve(Xc.T @ (w[:, None] * Xc) + a * np.eye(p), Xc.T @ (w[:, None] * Yc))
    f = my + (xin - mx) @ B; f = f - f.mean()
    v = f / (f.std() + 1e-12); _rc[key] = v; return v
def revz(t, w=10):
    rr = lp_all[1:, t - 1] - lp_all[1:, t - 1 - w]; rr = rr - rr.mean()
    return -rr / (rr.std() + 1e-12)

HL_ONLY = [(hl, 0.1, "full") for hl in (250, 500, 1000, 2000)]                 # = SAFE
DIVERSE = [(hl, a, ps) for hl in (250, 500, 1000, 2000)
           for a in (0.05, 0.1, 0.2) for ps in ("full", "noalgo")]            # 24 members
def ens_forecast(t, members, blend=0.30):
    core = np.mean([ridge_z(t, hl, a, ps) for hl, a, ps in members], 0)
    return (1 - blend) * core + blend * revz(t)

def book(members, Sd, Ed, blend=0.30):
    cash = 0.0; cp = np.zeros(nInst); value = 0.0; comm = 0.0; pll = []
    for t in range(Sd, Ed + 1):
        soFar = prc[:, :t]; cur = soFar[:, -1]; pos = np.zeros(nInst)
        if t < Ed and t >= 130:
            wz = ens_forecast(t, members, blend)
            pos[1:] = np.sign(wz) * (dlr[1:] / cur[1:])
            cap = dlr[0] / cur[0]
            lpA = lp_all[0, :t]; mv = lpA[30:] - lpA[:-30]
            z = (mv[-1] - mv[-60:].mean()) / (mv[-60:].std() + 1e-12)
            av = float(np.clip(-np.clip(z, -3, 3) / 3.0 * (1_000_000 / cur[0]), -cap, cap))
            r = lp_all[:, 1:t] - lp_all[:, :t - 1]; rA = r[0] - r[0].mean()
            bet = ((r[1:] - r[1:].mean(1, keepdims=True)) @ rA) / (rA @ rA + 1e-12)
            hs = -((pos[1:] * cur[1:]) @ bet) / cur[0]
            room = max(cap - abs(av), 0.0); pos[0] = av + float(np.clip(hs, -room, room))
            lim = (dlr / cur).astype(int); pos = np.clip(pos, -lim, lim).astype(int)
        else:
            pos = cp.copy()
        dp = pos - cp; cash -= cur.dot(dp) + comm
        comm = np.sum(cur * np.abs(dp) * commRate); cp = pos.copy()
        pl = cash + cp.dot(cur) - value; value = cash + cp.dot(cur)
        if t > Sd: pll.append(pl)
    pll = np.array(pll); mu, sd = pll.mean(), pll.std()
    if mu <= 0 or sd < 1e-10: return mu
    sr = np.sqrt(250) * mu / sd; return mu * sr ** 2 / (sr ** 2 + 1)

# member error-correlation rho (Bates-Granger): how decorrelated are the diverse members?
def member_rho(members, S=400, E=740):
    errs = {m: [] for m in members}
    for t in range(S, min(E, nt - 1), 3):
        fwd = lp_all[1:, t] - lp_all[1:, t - 1]; fwd = fwd - fwd.mean()
        fwd = fwd / (fwd.std() + 1e-12)
        for m in members:
            f = ridge_z(t, *m); errs[m].append(np.mean((f - fwd) ** 2))
    M = np.array([errs[m] for m in members])
    C = np.corrcoef(M); iu = np.triu_indices(len(members), 1)
    return C[iu].mean()

legs = [(96, 346), (150, 400), (250, 500), (350, 600), (450, 700), (500, 750)]
print("Ensemble comparison — FLOOR (min leg) is the qualifying metric:\n")
print(f"{'ensemble':<26}{'cold96':>8}{'min':>7}{'mean':>7}{'std':>7}{'500-750':>9}")
for name, mem in [("SAFE (4 half-lives)", HL_ONLY), ("DIVERSE (24: hl x a x pset)", DIVERSE)]:
    scs = [book(mem, S, E) for S, E in legs]
    print(f"{name:<26}{scs[0]:8.0f}{min(scs):7.0f}{np.mean(scs):7.0f}{np.std(scs):7.0f}{scs[-1]:9.0f}")
print(f"\nmean pairwise member error-corr rho:  HL-only={member_rho(HL_ONLY):.3f}   diverse={member_rho(DIVERSE):.3f}")
print("(lower rho => more decorrelated => lower Bates-Granger floor. does DIVERSE actually decorrelate?)")
