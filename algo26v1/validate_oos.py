"""Out-of-sample validation harness for Algothon strategies.

The competition reveals data progressively (we have days 1-500; 501-750 then -1000
arrive later) and every scoring window is 250 days. This harness does genuine
rolling-origin out-of-sample evaluation: fit only on days [1..t], score the NEXT
`test_len` days with the EXACT eval.py accounting, for several origins t. It is the
tool to run the moment real data lands — it turns "one noisy 500-day sample" into a
real train/validate split, so we can choose HALF_LIFE / ALPHA / CONV_Z / overlay
sizing on data the model never saw, instead of guessing.

Config is a plain dict so any candidate (v2, v3, v4, future) can be compared on the
same footing without importing the submission modules. Usage:

    python validate_oos.py                      # dry-run on the current 500 days
    # or import build_getpos(cfg) / score_window(...) from your own script.

Exact eval.py mechanics are replicated (integer shares, per-instrument caps &
commissions, mark-to-market daily PnL, Score = mean * SR^2/(SR^2+1)). Fresh model
state per run so the module cache never leaks a future fit across origins.
"""
import sys
import numpy as np
import pandas as pd
from extra_signals import drift_tilt_forecast, index_spread_positions, coint_pairs_positions  # speculative overlays (quarantined)

PRICES = "./prices.txt"
NINST_ALGO = 0                 # column 0 is the ALGO index (10x cap, 0.2bp commission)

DEFAULT_CFG = dict(
    half_life=2000, alpha=0.1, limit=10_000, algo_limit=100_000,
    conv_z=0.2, drop_intercept=False, hedge=True,
    contra_dollars=200_000, contra_k=30, contra_wz=60, contra_clip=3.0,   # contra_clip = the +-z cap on the reversion
    alpha_adaptive=False,          # if True: shrinkage decays as alpha*min(1, 500/n) — less regularization with more data
    view_scale=False,              # if True: scale book gross by today's forecast dispersion vs its trailing median (per-day conviction gate)
    view_floor=0.3, view_win=60,   # view_scale params
    # --- borderline STANDALONE signals from the hunts, wired as pre-registered OOS candidates (2026-07-14) ---
    drift_tilt=0.0, drift_win=60,  # tilt sizing toward cross-sec-DEMEANED trailing drift (idiosyncratic drift-continuation; in-sample EV~0 since idio drift=0, +EV iff forward drift appears). Market-neutral by construction (not beta).
    index_spread=0.0,              # $: long ALGO / short equal-weight basket (index-vs-constituents spread, in-sample t=3.47 real but capture cannibalizes book — OOS re-tests if still Score-negative)
    coint_pairs=0.0,               # $ per leg: Engle-Granger cointegration pairs overlay (reselect every 25d from past data only; in-sample Sharpe 2.58 but phase-sensitive)
)


def load_prices(fn=PRICES):
    return pd.read_csv(fn, sep=r"\s+").values.T          # (nInst, nDays)


def _ewls_fit(X, Y, half_life, alpha):
    n, p = X.shape
    lam = 0.5 ** (1.0 / half_life)
    w = lam ** np.arange(n - 1, -1, -1)
    sw = w.sum()
    mx = (w[:, None] * X).sum(0) / sw
    my = (w[:, None] * Y).sum(0) / sw
    Xc, Yc = X - mx, Y - my
    XtWX = Xc.T @ (w[:, None] * Xc)
    XtWY = Xc.T @ (w[:, None] * Yc)
    eps = 1e-8 * np.trace(XtWX) / p
    B = np.linalg.solve(XtWX + (eps + alpha) * np.eye(p), XtWY)
    return B, mx, my


def build_getpos(cfg):
    """Return a getMyPosition(prcSoFar) closure for a config dict (fresh cache each call)."""
    c = {**DEFAULT_CFG, **cfg}
    cache = {"model": None, "fit_t": None}

    def getMyPosition(prcSoFar):
        nInst, t = prcSoFar.shape
        pos = np.zeros(nInst)
        if t < 60:
            return pos
        lp = np.log(prcSoFar)
        ret = lp[:, 1:] - lp[:, :-1]
        if cache["fit_t"] != t:
            X = ret[:, :-1].T
            Y = ret[1:, 1:].T
            n_eff = X.shape[0]
            alpha_use = c["alpha"] * min(1.0, 500.0 / n_eff) if c["alpha_adaptive"] else c["alpha"]
            cache["model"] = _ewls_fit(X, Y, c["half_life"], alpha_use)
            cache["fit_t"] = t
        B, mx, my = cache["model"]
        pred = (0.0 if c["drop_intercept"] else my) + (ret[:, -1] - mx) @ B
        w = pred - pred.mean()
        if c["drift_tilt"] != 0:                          # speculative: idiosyncratic drift-continuation tilt
            w = drift_tilt_forecast(w, ret, c["drift_tilt"], c["drift_win"])
        sized = np.sign(w) * (c["limit"] / prcSoFar[1:, -1])
        if c["conv_z"] > 0:
            keep = np.abs(w) >= c["conv_z"] * (np.std(w) + 1e-12)
            sized = np.where(keep, sized, 0.0)
        pos[1:] = sized
        if c["view_scale"]:                              # per-day conviction gate: deploy less when the model's view is weak
            disp = float(np.std(w))
            buf = cache.setdefault("dispbuf", [])
            if len(buf) >= 10:                           # causal: trailing median of prior days' dispersion
                med = float(np.median(buf[-c["view_win"]:]))
                pos[1:] *= np.clip(disp / (med + 1e-12), c["view_floor"], 1.0)
            buf.append(disp)
        cap_sh = c["algo_limit"] / prcSoFar[0, -1]
        rev_sh = 0.0
        if c["contra_dollars"] > 0 and t > c["contra_k"] + c["contra_wz"] + 2:
            lpA = np.log(prcSoFar[0])
            move = lpA[c["contra_k"]:] - lpA[:-c["contra_k"]]
            z = (move[-1] - move[-c["contra_wz"]:].mean()) / (move[-c["contra_wz"]:].std() + 1e-12)
            rev_sh = -float(np.clip(z, -c["contra_clip"], c["contra_clip"])) * c["contra_dollars"] / prcSoFar[0, -1]
        rev_sh = float(np.clip(rev_sh, -cap_sh, cap_sh))
        hedge_sh = 0.0
        if c["hedge"]:
            rA = ret[0]; rAc = rA - rA.mean(); denom = rAc @ rAc + 1e-12
            betas = ((ret[1:] - ret[1:].mean(1, keepdims=True)) @ rAc) / denom
            net_beta = (pos[1:] * prcSoFar[1:, -1]) @ betas
            hedge_sh = -net_beta / prcSoFar[0, -1]
        room = max(cap_sh - abs(rev_sh), 0.0)
        pos[0] = rev_sh + float(np.clip(hedge_sh, -room, room))

        # speculative standalone overlays (see extra_signals.py) — off unless the cfg flag is set
        index_spread_positions(pos, prcSoFar, c["index_spread"])
        coint_pairs_positions(pos, lp, prcSoFar, cache, t, c["coint_pairs"])

        return pos.astype(int)

    return getMyPosition


def score_window(prc, getpos, start_day, test_len):
    """Exact eval.py accounting: trade days [start_day, start_day+test_len), mark next day.

    Only prices up to each day are passed to getpos, so the fit can only see the past.
    """
    nInst, nDays = prc.shape
    comm = np.full(nInst, 1e-4); comm[NINST_ALGO] = 2e-5
    dlim = np.full(nInst, 10_000); dlim[NINST_ALGO] = 100_000
    cash = 0.0; cur = np.zeros(nInst); val = 0.0; fee = 0.0; pl = []
    tot_dvol = 0.0
    end = start_day + test_len
    for t in range(start_day, end + 1):
        p = prc[:, t - 1]                           # last observed close (day t-1, 0-indexed)
        if t < end:
            raw = getpos(prc[:, :t])                # history up to and including day t-1
            cap = (dlim / p).astype(int)
            npos = np.clip(raw, -cap, cap).astype(int)
        else:
            npos = cur.copy()
        d = npos - cur
        cash -= p.dot(d) + fee
        dv = p * np.abs(d); fee = np.sum(dv * comm)
        cur = npos.copy(); pv = cur.dot(p)
        today = cash + pv - val; val = cash + pv
        if t > start_day:
            pl.append(today); tot_dvol += float(dv.sum())
    pl = np.array(pl)
    mu, sd = pl.mean(), pl.std()
    sr = np.sqrt(250) * mu / sd if sd > 0 else 0.0
    score = mu * sr ** 2 / (sr ** 2 + 1) if mu > 0 else mu
    winrate = float(np.mean(pl > 0)) if len(pl) else 0.0
    return dict(mean=mu, std=sd, sharpe=sr, score=score, n=len(pl), pl=pl,
                turnover=tot_dvol / max(len(pl), 1), winrate=winrate)


def rolling_oos(prc, cfg, origins, test_len=250):
    """Fit-on-[1..origin], score-next-`test_len`, for each origin. Returns per-window + average."""
    rows = []
    for o in origins:
        if o + test_len > prc.shape[1]:
            continue
        r = score_window(prc, build_getpos(cfg), o, test_len)
        r["origin"] = o
        rows.append(r)
    avg = {k: float(np.mean([r[k] for r in rows])) for k in ("mean", "sharpe", "score")} if rows else {}
    return rows, avg


# --- PRE-REGISTERED candidate battery (fixed 2026-07-14, BEFORE seeing real OOS data) ---
# Discipline: this list is frozen now so we cannot post-hoc cherry-pick after seeing 501-750.
# When real data lands, run rolling_oos(prc, cfg, origins=[250,500,750], test_len=250) for each and
# ship ONLY the config with the best AVERAGE OOS score that also beats v4 in BOTH later windows.
# Every item has an a-priori mechanism; none is a blind grid point.
CANDIDATE_BATTERY = {
    "v2 (HL500)":          dict(half_life=500),                       # incumbent-ish
    "v4 (HL2000)":         dict(half_life=2000),                      # current standing entry
    "expanding (HLinf)":   dict(half_life=10**9),                     # pure ML estimate; best iff stationarity holds far out
    "alpha 0.05":          dict(half_life=2000, alpha=0.05),          # more data -> less shrinkage (LW hinted ~0.011)
    "alpha 0.03":          dict(half_life=2000, alpha=0.03),
    "lean (no overlay)":   dict(half_life=2000, contra_dollars=0),    # does the speculative ALGO leg survive OOS?
    "contra 300k":         dict(half_life=2000, contra_dollars=300_000),  # recent half hinted higher; check OOS
    "conv_z 0.15":         dict(half_life=2000, conv_z=0.15),         # book conviction bar, flat plateau — confirm
    "drop intercept":      dict(half_life=2000, drop_intercept=True), # v3 idea: does the IC gain convert with more data?
    # --- extension (2026-07-14): more-data-regime & one new mechanism ---
    "HL1000":              dict(half_life=1000),                      # pin the memory curve between 500 and expanding
    "HL1500":              dict(half_life=1500),
    "alpha 0.01":          dict(half_life=2000, alpha=0.01),          # LW-implied shrinkage; may win ONLY with more data
    # "alpha adaptive" REMOVED 2026-07-15: DGP-simulator test showed optimal absolute alpha is FLAT/slightly-RISING
    # with data (not ~1/n), so alpha*500/n is backwards & mildly harmful. Alpha is a non-lever (IC flat +-1% over 0.02-0.2).
    "alpha0.05 + lean":    dict(half_life=2000, alpha=0.05, contra_dollars=0),   # stack two independent robustness moves
    "day-gate view-scale": dict(half_life=2000, view_scale=True),     # NEW mechanism; WEAK prior (dispersion ~0 edge-predictive) — close it on OOS
    "v5 (WZ20 + noHedge)": dict(half_life=2000, contra_wz=20, hedge=False),  # user's last-250-tuned config — OOS will confirm/refute
    # --- borderline STANDALONE signals from the hunts (2026-07-14): "a little correlation & p-stat" -> adjudicate OOS ---
    "drift-tilt 0.5":      dict(half_life=2000, drift_tilt=0.5),      # idiosyncratic drift-continuation; the KEY one — in-sample ~0-EV (idio drift=0), +EV iff drift appears forward
    "drift-tilt 1.0":      dict(half_life=2000, drift_tilt=1.0),
    "index-spread 8k":     dict(half_life=2000, index_spread=8_000),  # long ALGO/short basket; in-sample t=3.47 signal but Score-negative (cannibalizes) — does OOS overturn?
    "coint-pairs 8k":      dict(half_life=2000, coint_pairs=8_000),   # Engle-Granger overlay; in-sample Sharpe 2.58 but phase-sensitive
}

# --- comprehensive one-at-a-time sensitivity sweep (every real variable), pre-registered 2026-07-14 ---
# Run: python validate_oos.py sweep   (holds all other knobs at the v4 baseline)
SWEEP_GRID = {
    "half_life":      [60, 120, 250, 500, 1000, 2000, 10**9],  # 60-250 = NON-STATIONARITY DETECTOR: beat v4 OOS only if the process drifts
    "alpha":          [0.01, 0.03, 0.05, 0.1, 0.2, 0.5],
    "conv_z":         [0.0, 0.1, 0.15, 0.2, 0.25, 0.3],
    "contra_dollars": [0, 100_000, 200_000, 300_000, 400_000],
    "contra_k":       [15, 20, 30, 40, 50],
    "contra_wz":      [20, 40, 60, 80, 100],   # 20 added (user's v5 pick — earn it on real OOS, not last-250)
    "contra_clip":    [1.5, 2.0, 3.0, 4.0, 5.0],
    "drop_intercept": [False, True],
    "hedge":          [True, False],
    "drift_tilt":     [0.0, 0.25, 0.5, 1.0, 2.0],   # idiosyncratic drift-continuation: beats v4 OOS only if forward drift is nonzero
}


def sensitivity_sweep(prc, origins, test_len=250):
    base_pl = np.concatenate([score_window(prc, build_getpos({}), o, test_len)["pl"] for o in origins])
    print("\n=== ONE-AT-A-TIME sensitivity sweep — every variable vs the v4 baseline ===")
    print("short half-lives (60-250) are the NON-STATIONARITY test: they beat v4 OOS only if the process drifts.")
    print("DISCIPLINE: with 1 window this is a proxy; on real data require |t|>=2 AND consistency across origins,")
    print("and remember we are testing ~35 settings — expect ~2 false 't>2' hits by chance (Bonferroni).\n")
    for var, grid in SWEEP_GRID.items():
        print(f"{var}  (v4 = {DEFAULT_CFG.get(var)}):")
        for v in grid:
            rows, avg = rolling_oos(prc, {var: v}, origins, test_len)
            pl = np.concatenate([r["pl"] for r in rows]) if rows else base_pl
            d = pl - base_pl
            t = d.mean() / d.std() * np.sqrt(len(d)) if d.std() > 0 else 0.0
            star = "  <= v4" if DEFAULT_CFG.get(var) == v else ""
            print(f"   {str(v):>11}: score {avg.get('score', 0):7.1f}   d$/day {d.mean():+7.1f}  t {t:+5.2f}{star}")


if __name__ == "__main__":
    prc = load_prices()
    nDays = prc.shape[1]
    print(f"Loaded {prc.shape[0]} instruments x {nDays} days\n")

    # Use every genuine OOS origin the data supports: fit 1..o, score o+1..o+250.
    origins = [o for o in (250, 500, 750, 1000, 1250) if o + 250 <= nDays]
    if not origins:
        origins = [max(60, nDays - 250)]                 # <500 days: single within-sample split (proxy only)
    real_oos = nDays > 500
    banner = "GENUINE out-of-sample" if real_oos else "within-sample proxy (real OOS unlocks at >500 days)"
    print(f"Rolling-origin validation — {banner}. origins={origins}\n")

    BASE = "v4 (HL2000)"                              # paired significance is measured against this
    results = {}
    for name, cfg in CANDIDATE_BATTERY.items():
        rows, avg = rolling_oos(prc, cfg, origins, test_len=250)
        pl = np.concatenate([r["pl"] for r in rows]) if rows else np.array([])
        turn = float(np.mean([r["turnover"] for r in rows])) if rows else 0.0
        winr = float(np.mean([r["winrate"] for r in rows])) if rows else 0.0
        results[name] = dict(avg=avg.get("score", 0), rows=rows, pl=pl, turnover=turn, winrate=winr)
    base_pl = results[BASE]["pl"]

    def paired(pl):                                  # paired day-by-day diff vs BASE: mean/day, t, both-halves
        if len(pl) != len(base_pl) or len(pl) < 4:
            return None
        d = pl - base_pl
        if d.std() == 0:
            return dict(dm=0.0, t=0.0, h1=0.0, h2=0.0)
        t = d.mean() / d.std() * np.sqrt(len(d))
        h = len(d) // 2
        return dict(dm=d.mean(), t=t, h1=d[:h].mean(), h2=d[h:].mean())

    hdr = f"  {'candidate':22s} {'avgScore':>8} {'win%':>5} {'turnover$':>10} | {'vs v4 d$/d':>10} {'t':>6} {'halves':>7}"
    print(hdr)
    for name in sorted(results, key=lambda n: -results[n]["avg"]):
        r = results[name]; p = paired(r["pl"])
        if name == BASE:
            tag = "  (baseline)"
        elif p is None:
            tag = "  n/a"
        else:
            bh = "yes" if (p["h1"] > 0 and p["h2"] > 0) else ("no" if (p["h1"] < 0 and p["h2"] < 0) else "mixed")
            tag = f"  {p['dm']:+8.1f} {p['t']:+6.2f} {bh:>7}"
        print(f"  {name:22s} {r['avg']:8.1f} {r['winrate']*100:4.0f}% {r['turnover']:10.0f} |{tag}")
    print("\nSHIP RULE (statistical): ship a candidate over v4 ONLY if paired d$/day > 0 with |t| >= 2")
    print("AND both-halves positive AND it holds in the LATEST origin. Score rank alone is not enough.")
    print("On <500 days this is a within-sample proxy; treat as a dry-run until real OOS data lands.")

    if "sweep" in sys.argv:
        sensitivity_sweep(prc, origins)
