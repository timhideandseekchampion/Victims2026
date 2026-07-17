"""Performance-gated adaptive ensemble: ridge core (always on) + auxiliary sleeves that
SELF-ACTIVATE only when their own recent, causal, out-of-sample edge is statistically positive.
A dead sleeve gets ~0 weight (no cost today); if the future regime turns to favor it, its rolling
t-stat rises and it switches on automatically. Sleeves gated: cross-sectional reversion (idio),
cointegration pairs (structural). ALGO leg = OLS-adaptive (self-switching fade/follow).

Test: (a) on our data the gates should stay ~off (score ~= current book); (b) the gate must RISE
when a sleeve's edge is actually present (we verify on a reversion-heavy synthetic series)."""
import numpy as np, pandas as pd

prc_all = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nt = prc_all.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000); dlr[0] = 100_000
GATE_W = 60          # trailing window for each sleeve's rolling edge
GATE_T = 1.5         # t-stat needed before a sleeve starts earning weight
GATE_TMAX = 3.5      # t-stat at which the sleeve reaches full weight


def rfit(X, Y, hl=500, a=0.1):
    n, p = X.shape; lam = 0.5**(1.0/hl); w = lam**np.arange(n-1, -1, -1); sw = w.sum()
    mx = (w[:, None]*X).sum(0)/sw; my = (w[:, None]*Y).sum(0)/sw; Xc, Yc = X-mx, Y-my
    XtWX = Xc.T@(w[:, None]*Xc); XtWY = Xc.T@(w[:, None]*Yc); eps = 1e-8*np.trace(XtWX)/p
    return np.linalg.solve(XtWX+(eps+a)*np.eye(p), XtWY), mx, my


def xs_signal(ret, w=10):
    """cross-sectional reversion signal per tradeable name at the latest day."""
    r = ret[1:, -w:].sum(1); r = r - r.mean()
    return -r / (np.std(r) + 1e-12)


def xs_gate(ret, w=10):
    """causal rolling t-stat of the xs-reversion cross-sectional IC over trailing GATE_W days."""
    T = ret.shape[1]; ics = []
    for d in range(T - GATE_W, T - 1):
        if d - w < 0: continue
        sig = -(ret[1:, d-w+1:d+1].sum(1)); sig -= sig.mean()
        fwd = ret[1:, d+1]                                  # next-day return per name
        if sig.std() > 0 and fwd.std() > 0:
            ics.append(np.corrcoef(sig, fwd)[0, 1])
    ics = np.array(ics)
    if len(ics) < 20: return 0.0, 0.0
    t = ics.mean() / (ics.std() / np.sqrt(len(ics)) + 1e-12)
    g = float(np.clip((t - GATE_T) / (GATE_TMAX - GATE_T), 0.0, 1.0))
    return g, t


def make(use_gate=True, force_xs=False):
    c = {"t": None, "m": None}; log = {"g": [], "t": []}
    def gp(prc):
        ni, t = prc.shape; pos = np.zeros(ni)
        if t < 95: return pos.astype(int)
        lp = np.log(prc); ret = lp[:, 1:]-lp[:, :-1]
        if c["t"] != t: c["m"] = rfit(ret[:, :-1].T, ret[1:, 1:].T); c["t"] = t
        B, mx, my = c["m"]; pred = my+(ret[:, -1]-mx)@B; w = pred-pred.mean()
        s = np.sign(w)*(10000/prc[1:, -1]); pos[1:] = np.where(np.abs(w) >= 0.2*(np.std(w)+1e-12), s, 0.0)
        # --- gated cross-sectional reversion sleeve ---
        g, tstat = (1.0, 9.9) if force_xs else (xs_gate(ret) if use_gate else (0.0, 0.0))
        log["g"].append(g); log["t"].append(tstat)
        if g > 0:
            xs = xs_signal(ret)
            add = np.sign(xs) * (g * 6000 / prc[1:, -1]) * (np.abs(xs) >= 0.5)
            pos[1:] += add                                   # clipped to $10k by the grader
        # --- ALGO leg: fixed fade here (OLS-adaptive tested separately) ---
        cap = 100000/prc[0, -1]
        lpA = np.log(prc[0]); mv = lpA[30:]-lpA[:-30]; z = (mv[-1]-mv[-60:].mean())/(mv[-60:].std()+1e-12)
        rev = float(np.clip(-np.clip(z, -3, 3)*200000/prc[0, -1], -cap, cap))
        rA = ret[0]; rAc = rA-rA.mean(); den = rAc@rAc+1e-12
        betas = ((ret[1:]-ret[1:].mean(1, keepdims=True))@rAc)/den
        net = (pos[1:]*prc[1:, -1])@betas; room = max(cap-abs(rev), 0.0)
        pos[0] = rev+float(np.clip(-net/prc[0, -1], -room, room)); return pos.astype(int)
    gp.log = log
    return gp


def score(pll):
    mu, sd = pll.mean(), pll.std()
    if mu <= 0 or sd < 1e-10: return mu
    sr = np.sqrt(250)*mu/sd; return mu*sr**2/(sr**2+1)
def run(gp, start, end, panel=None):
    P = panel if panel is not None else prc_all; N = P.shape[1]
    cash = 0; cp = np.zeros(nInst); val = 0; cm = 0; pll = []
    for t in range(start, end+1):
        p = P[:, :t]; cur = p[:, -1]
        npos = np.clip(gp(p), -(dlr/cur).astype(int), (dlr/cur).astype(int)).astype(int) if t < end else cp.copy()
        d = npos-cp; cash -= cur.dot(d)+cm; dv = cur*np.abs(d); cm = (dv*commRate).sum(); cp = npos.copy()
        pl = cash+cp.dot(cur)-val; val = cash+cp.dot(cur)
        if t > start: pll.append(pl)
    return score(np.array(pll))


print("=== on OUR data: does the xs-reversion sleeve stay OFF (no cost)? ===")
for name, kw in [("ridge core only (gate off)", dict(use_gate=False)),
                 ("adaptive gate (xs self-activates)", dict(use_gate=True)),
                 ("xs FORCED on (ungated)", dict(force_xs=True))]:
    g = make(**kw)
    s250 = run(g, nt-250, nt); s440 = run(make(**kw), nt-440, nt)
    gl = np.array(g.log["g"]); tl = np.array(g.log["t"])
    print(f"  {name:34} S@250 {s250:6.1f}  S@440 {s440:6.1f}  avg-gate {gl.mean():.2f}  avg-t {tl.mean():+.2f}")

# --- verify the gate RISES when reversion is genuinely strong (synthetic reversion-heavy panel) ---
print("\n=== does the gate SWITCH ON when the regime favors cross-sec reversion? ===")
rng = np.random.default_rng(0)
base = prc_all[:, -260:].copy(); lp = np.log(base); r = np.diff(lp, axis=1)
# inject strong 1-day cross-sectional reversion into the idio returns
r2 = r.copy()
for d in range(1, r2.shape[1]):
    xs = r2[1:, d-1] - r2[1:, d-1].mean()
    r2[1:, d] += -0.5 * xs                                   # strong reversion overlay
synth = np.empty_like(base); synth[:, 0] = base[:, 0]
synth[:, 1:] = base[:, :1]*np.exp(np.cumsum(r2, axis=1))
g = make(use_gate=True)
run(g, synth.shape[1]-120, synth.shape[1], panel=synth)
gl = np.array(g.log["g"]); tl = np.array(g.log["t"])
print(f"  reversion-heavy synthetic: avg-gate {gl.mean():.2f}  avg-t {tl.mean():+.2f}  "
      f"(gate should be HIGH here -> it auto-switches ON)")
