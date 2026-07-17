"""Algothon 2026 — COMBINED v3 ADAPTIVE (self-switching ensemble, robust to regime change).

Core (always on) = market-neutral peer-lead-lag ridge (HL=500, conviction gate, beta-hedge).
ALGO leg = OLS-ADAPTIVE fade/follow (data sets the coefficient; auto-follows only if the index
statistically trends). Plus a MENU of algo26v2-inspired AUXILIARY sleeves, each behind a
PERFORMANCE GATE: a sleeve only earns weight to the degree its OWN recent, causal, out-of-sample
edge is statistically positive (rolling cross-sectional IC t-stat over GATE_W days). A sleeve that
is dead on the current data gets ~0 weight (no cost); if the real/future data shifts to favor it,
its t-stat rises and it SWITCHES ON automatically.

Verified behavior:
  * Forcing a weak sleeve on (ungated) HURTS (761->685). The gate keeps it off on our data
    (avg gate ~0.1) so the ensemble scores ~= the core (~760-764), i.e. no cost today.
  * On a regime where the sleeve's edge is real (e.g. strong cross-sectional reversion), the gate
    goes to ~1.0 and the sleeve activates. So dead signals cost nothing but latent ones are ready.

Gated sleeves (from v2's engine): xs = cross-sectional reversion; corr = residual-vs-ALGO
reversion. (pairs/lead/mf can be added to GATED_SLEEVES the same way; kept off here for speed.)
Set every gate weight to 0 (raise GATE_T high) to reproduce the OLS-adaptive primary exactly.
"""
import numpy as np

HALF_LIFE = 500
ALPHA = 0.1
LIMIT = 10_000
ALGO_LIMIT = 100_000
CONV_Z = 0.2
HEDGE = True
CONTRA_DOLLARS = 200_000
CONTRA_K = 30
CONTRA_WZ = 60
ALGO_MODE = "ols"          # "ols" = data-driven fade/follow | "fade" = fixed reversion
OLS_WINDOW = 250

GATE_W = 60                # trailing days for each sleeve's rolling edge
GATE_T = 3.0               # t-stat before a sleeve starts earning weight (high bar: stay off on
                           # noise. At 3.0 the ensemble ~= the OLS-adaptive primary @250 while the
                           # sleeves still fully activate (gate->1.0) in a genuine reversion regime.
                           # Lower it to be more eager to switch sleeves on; raise it to be stricter.)
GATE_TMAX = 5.0            # t-stat at which a sleeve reaches full weight
SLEEVE_DOLLARS = 6_000     # per-name notional a fully-activated sleeve adds
GATED_SLEEVES = ("xs", "corr", "ar1")

_cache = {"fit_t": None, "model": None}


def _ewls_ridge_fit(X, Y):
    n, p = X.shape
    lam = 0.5 ** (1.0 / HALF_LIFE)
    w = lam ** np.arange(n - 1, -1, -1); sw = w.sum()
    mx = (w[:, None] * X).sum(0) / sw; my = (w[:, None] * Y).sum(0) / sw
    Xc, Yc = X - mx, Y - my
    XtWX = Xc.T @ (w[:, None] * Xc); XtWY = Xc.T @ (w[:, None] * Yc)
    eps = 1e-8 * np.trace(XtWX) / p
    return np.linalg.solve(XtWX + (eps + ALPHA) * np.eye(p), XtWY), mx, my


# ---------- OLS-adaptive ALGO leg ----------
def _algo_signal(lpA):
    L = len(lpA)
    mv = np.full(L, np.nan); mv[CONTRA_K:] = lpA[CONTRA_K:] - lpA[:-CONTRA_K]
    z = np.full(L, np.nan)
    for d in range(CONTRA_K + CONTRA_WZ, L):
        seg = mv[d - CONTRA_WZ + 1:d + 1]
        z[d] = (mv[d] - np.nanmean(seg)) / (np.nanstd(seg) + 1e-12)
    zt = z[-1]
    if not np.isfinite(zt):
        return 0.0
    if ALGO_MODE == "fade":
        return -float(np.clip(zt, -3, 3))
    rA = np.diff(lpA); ds = np.arange(CONTRA_K + CONTRA_WZ, L - 1)[-OLS_WINDOW:]
    x = z[ds]; y = rA[ds]; ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
    if len(x) < 30:
        return -float(np.clip(zt, -3, 3))
    xm = x - x.mean(); beta = (xm @ (y - y.mean())) / ((xm @ xm) + 1e-18)
    scale = np.std(beta * x) + 1e-12
    return float(np.clip(beta * zt / scale, -3, 3))


# ---------- auxiliary sleeve signals (v2-inspired) ----------
def _sleeve_signal(kind, ret, lp, d):
    """Signal vector over the 50 tradeable names using data up to column d (inclusive)."""
    if kind == "xs":                                   # cross-sectional reversion
        r = ret[1:, d - 9:d + 1].sum(1); r = r - r.mean()
        return -r / (np.std(r) + 1e-12)
    if kind == "corr":                                 # residual-vs-ALGO reversion
        W = 60
        a = lp[0, d - W + 1:d + 1]; ac = a - a.mean(); den = ac @ ac + 1e-12
        X = lp[1:, d - W + 1:d + 1]; Xc = X - X.mean(1, keepdims=True)
        beta = (Xc @ ac) / den
        resid = (lp[1:, d] - lp[1:, d - W + 1]) - beta * (lp[0, d] - lp[0, d - W + 1])
        resid = resid - resid.mean()
        return -resid / (np.std(resid) + 1e-12)
    if kind == "ar1":                                  # idiosyncratic AR(1) momentum/reversion
        W = 40                                         # armed by the white-noise detector: follows a
        R = ret[1:, d - W + 1:d + 1]                   # name's own move to the degree it is auto-
        Rc = R - R.mean(1, keepdims=True)              # correlated. ~0 while returns are white noise.
        num = (Rc[:, :-1] * Rc[:, 1:]).sum(1)
        den = (Rc * Rc).sum(1) + 1e-12
        acn = num / den                                # per-name lag-1 autocorr
        sig = acn * ret[1:, d]                         # AR(1) forecast: ac>0 follow, ac<0 fade
        sig = sig - sig.mean()                         # market-neutral (idio); ALGO leg covers index
        return sig / (np.std(sig) + 1e-12)
    return np.zeros(ret.shape[0] - 1)


def _gate(kind, ret, lp):
    """Rolling cross-sectional IC t-stat of the sleeve -> weight in [0,1] (causal)."""
    T = ret.shape[1]; ics = []
    for d in range(T - GATE_W, T - 1):
        if d - 10 < 0 or d - 60 < 0:
            continue
        sig = _sleeve_signal(kind, ret, lp, d)
        fwd = ret[1:, d + 1]
        if sig.std() > 0 and fwd.std() > 0:
            ics.append(np.corrcoef(sig, fwd)[0, 1])
    ics = np.array(ics)
    if len(ics) < 20:
        return 0.0
    t = ics.mean() / (ics.std() / np.sqrt(len(ics)) + 1e-12)
    return float(np.clip((t - GATE_T) / (GATE_TMAX - GATE_T), 0.0, 1.0))


def getMyPosition(prcSoFar):
    nInst, t = prcSoFar.shape
    pos = np.zeros(nInst)
    if t < 95:
        return pos.astype(int)
    lp = np.log(prcSoFar)
    ret = lp[:, 1:] - lp[:, :-1]
    if _cache["fit_t"] != t:
        _cache["model"] = _ewls_ridge_fit(ret[:, :-1].T, ret[1:, 1:].T)
        _cache["fit_t"] = t
    B, mx, my = _cache["model"]
    pred = my + (ret[:, -1] - mx) @ B
    w = pred - pred.mean()
    sized = np.sign(w) * (LIMIT / prcSoFar[1:, -1])
    if CONV_Z > 0:
        keep = np.abs(w) >= CONV_Z * (np.std(w) + 1e-12)
        sized = np.where(keep, sized, 0.0)
    pos[1:] = sized

    # gated auxiliary sleeves — self-activate only when their recent edge is significant
    for kind in GATED_SLEEVES:
        g = _gate(kind, ret, lp)
        if g > 0:
            sig = _sleeve_signal(kind, ret, lp, ret.shape[1] - 1)
            pos[1:] += np.sign(sig) * (g * SLEEVE_DOLLARS / prcSoFar[1:, -1]) * (np.abs(sig) >= 0.5)

    # ALGO index leg (OLS-adaptive)
    cap_sh = ALGO_LIMIT / prcSoFar[0, -1]
    rev_sh = 0.0
    if CONTRA_DOLLARS > 0 and t > CONTRA_K + CONTRA_WZ + 2:
        rev_sh = float(np.clip(_algo_signal(lp[0]) * CONTRA_DOLLARS / prcSoFar[0, -1], -cap_sh, cap_sh))

    hedge_sh = 0.0
    if HEDGE:
        rA0 = ret[0]; rAc = rA0 - rA0.mean(); denom = rAc @ rAc + 1e-12
        betas = ((ret[1:] - ret[1:].mean(1, keepdims=True)) @ rAc) / denom
        net_beta = (pos[1:] * prcSoFar[1:, -1]) @ betas
        hedge_sh = -net_beta / prcSoFar[0, -1]
    room = max(cap_sh - abs(rev_sh), 0.0)
    pos[0] = rev_sh + float(np.clip(hedge_sh, -room, room))
    return pos.astype(int)
