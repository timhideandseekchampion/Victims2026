"""Algothon 2026 — strategy scaffold + baselines.

=============================================================================
 WHERE TO WORK:  section §4 ("YOUR STRATEGY"). Everything else is a reusable
 toolkit and the plumbing that turns your idea into a *legal* integer share
 vector (the contract eval.py expects). A fresh copy of this file already runs
 a fair baseline — you improve on it, and measure against it with bench.py.
=============================================================================

The build loop:
    1. edit `alpha()` in §4 (or compose the §1/§2 helpers into a new idea)
    2. python backtester.py --strategy strategy --stats --walk-forward 5
    3. python bench.py                 # did you beat the baselines?
    4. python backtester.py --strategy strategy --export-positions pos.csv
       python dashboard.py --positions pos.csv     # eyeball entries/exits

Single-file rule: the competition submits ONE file (+ optional requirements.txt),
so helpers, alphas and plumbing all live here. When ready to submit / run eval:
    cp strategy.py teamName.py         # eval.py imports getMyPosition from teamName

Nothing here is tuned. The defaults are round, honest yardsticks — not optimised
picks. (We saw earlier that chasing the peak in-sample Score overfits; use
--walk-forward and bench.py to judge robustness before trusting a number.)
"""
import numpy as np

# --- position limits, mirrored from eval.py (do NOT change: the grader uses these) ---
DEFAULT_DLR_LIMIT = 10_000
INST0_DLR_LIMIT = 100_000
ANN = 250  # trading days/year, to match the scoring


# ============================================================================
# §1  INDICATOR HELPERS  — building blocks for an alpha.
#     Each takes prices shaped (nInst, t) and returns one value per instrument
#     (a length-nInst vector), computed as of the most recent day.
# ============================================================================
def returns(prices, w=1):
    """Simple return over the last w days, per instrument."""
    if prices.shape[1] <= w:
        return np.zeros(prices.shape[0])
    return prices[:, -1] / prices[:, -1 - w] - 1.0


def momentum(prices, w=60):
    """Log return over the last w days (trend strength), per instrument."""
    if prices.shape[1] <= w:
        return np.zeros(prices.shape[0])
    return np.log(prices[:, -1] / prices[:, -1 - w])


def zscore(prices, w=20):
    """(price - rolling mean) / rolling std over the last w days, per instrument.
    Positive = expensive vs its own recent average; negative = cheap."""
    window = min(w, prices.shape[1])
    recent = prices[:, -window:]
    mu = recent.mean(axis=1)
    sd = recent.std(axis=1)
    return np.divide(prices[:, -1] - mu, sd, out=np.zeros_like(mu), where=sd > 1e-9)


def rank(scores):
    """Cross-sectional rank of a score vector, rescaled to ~[-1, 1] (NaN-safe).
    Turns any raw signal into a comparable spread across the universe."""
    s = np.where(np.isfinite(scores), scores, np.nan)
    order = np.argsort(np.argsort(np.nan_to_num(s, nan=-np.inf)))
    n = len(scores)
    r = order / (n - 1) if n > 1 else np.zeros(n)  # 0..1
    return 2.0 * r - 1.0                            # -1..1


def realised_vol(prices, w=20):
    """Annualised volatility of daily log returns over the last w days, per inst."""
    if prices.shape[1] < 2:
        return np.zeros(prices.shape[0])
    logret = np.diff(np.log(prices[:, -(w + 1):]), axis=1)
    return logret.std(axis=1) * np.sqrt(ANN)


def ema(prices, span):
    """Exponential moving average along time; returns the latest EMA vector."""
    a = 2.0 / (span + 1.0)
    e = prices[:, 0].astype(float).copy()
    for t in range(1, prices.shape[1]):
        e = a * prices[:, t] + (1 - a) * e
    return e


def ewma_z(prices, span):
    """Adaptive z-score: (price - EWMA mean) / EWMA std, exponentially weighted so
    it recalibrates faster to drift than the flat-window zscore(). One incremental
    pass; returns the latest z vector. Stateless."""
    a = 2.0 / (span + 1.0)
    mean = prices[:, 0].astype(float).copy()
    var = np.zeros(prices.shape[0])
    for t in range(1, prices.shape[1]):
        delta = prices[:, t] - mean
        mean = mean + a * delta
        var = (1 - a) * (var + a * delta * delta)
    sd = np.sqrt(var)
    return np.divide(prices[:, -1] - mean, sd, out=np.zeros_like(mean), where=sd > 1e-9)


# ============================================================================
# §2  SIZING HELPERS  — turn a score vector into a *legal* integer share vector.
#     These handle every chore that used to be error-prone boilerplate:
#     dollar-neutralising, scaling to the $ limits, ÷price, integer rounding,
#     and clipping so you never exceed a position limit.
# ============================================================================
def dollar_limits(n):
    lim = np.full(n, float(DEFAULT_DLR_LIMIT))
    lim[0] = INST0_DLR_LIMIT
    return lim


def neutralize(scores):
    """Subtract the cross-sectional mean so longs ≈ shorts (roughly $-neutral)."""
    return scores - np.nanmean(scores)


def to_shares(target_dollars, prices, limits):
    """$ targets -> integer shares, clipped to ±limit (as the grader would).
    prices: full (nInst, t) history; the latest column is used for $→shares."""
    last = prices[:, -1]
    shares = (target_dollars / last).astype(int)
    max_shares = (limits / last).astype(int)
    return np.clip(shares, -max_shares, max_shares)


def size_fraction_of_limit(scores, prices, scale=2.0):
    """Each instrument gets a fraction of its own $ limit, proportional to its
    score (clipped to ±1 full size). `scale` = score magnitude that maps to a
    full-size position. Simple, robust, always within limits.
    prices: full (nInst, t) history."""
    limits = dollar_limits(len(scores))
    frac = np.clip(scores / scale, -1.0, 1.0)
    return to_shares(frac * limits, prices, limits)


def size_inverse_vol(scores, prices, scale=2.0, vol_w=20):
    """Risk-parity flavour: weight by score / volatility so quiet names get more
    capital and noisy names less (each contributes similar risk). Rescaled so the
    largest target sits at its $ limit, then clipped."""
    limits = dollar_limits(len(scores))
    vol = realised_vol(prices, vol_w)
    w = np.divide(scores, vol, out=np.zeros_like(scores), where=vol > 1e-9)
    peak = np.max(np.abs(w))
    if peak < 1e-12:
        return np.zeros(len(scores), dtype=int)
    frac = np.clip((w / peak) * (2.0 / scale), -1.0, 1.0)  # scale tunes aggression
    return to_shares(frac * limits, prices, limits)


def size_kelly(scores, prices, scale=2.0, vol_w=20, kelly_frac=0.5):
    """Fractional-Kelly / inverse-VARIANCE sizing: weight by score / variance
    (Kelly f* ∝ edge/variance). `kelly_frac` scales it down (full Kelly overbets a
    noisy edge). An EXPERIMENT — on this data it tends to track inverse-vol; the
    Score rewards Sharpe, not log-growth, so don't expect a Kelly win."""
    limits = dollar_limits(len(scores))
    vol = realised_vol(prices, vol_w)
    var = np.where(vol < 1e-9, np.inf, vol * vol)
    w = scores / var
    peak = np.max(np.abs(w))
    if peak < 1e-12:
        return np.zeros(len(scores), dtype=int)
    frac = np.clip((w / peak) * kelly_frac * (2.0 / scale), -1.0, 1.0)
    return to_shares(frac * limits, prices, limits)


# ============================================================================
# §2b TURNOVER CONTROL  — fees are the constraint on this data, and the signal's
#     edge peaks at ~5-day horizon, so trading LESS keeps more of the edge net of
#     fees. All three are STATELESS (they recompute, never store a global), so
#     they're legal under the getMyPosition(prcSoFar) contract.
# ============================================================================
def smooth_signal(signal_fn, span):
    """Wrap a signal so its score is an EMA over the last `span` as-of days.
    Slower-moving score -> fewer position flips -> lower turnover."""
    if span <= 1:
        return signal_fn

    def f(prices):
        t = prices.shape[1]
        s = min(span, t)
        a = 2.0 / (s + 1.0)
        e = None
        for j in range(t - s, t):
            score = signal_fn(prices[:, :j + 1])
            e = score if e is None else a * score + (1 - a) * e
        return e
    return f


def hold_every(get_pos, k):
    """Wrap a position function to only re-decide every k days (hold in between).
    Stateless: recomputes the book as of the most recent rebalance day."""
    if k <= 1:
        return get_pos

    def f(prices):
        t = prices.shape[1]
        cut = max(1, t - (t % k))
        return get_pos(prices[:, :cut])
    return f


def no_trade_band(get_pos, band):
    """Wrap a position function to ignore small moves: keep yesterday's target
    unless the new target differs by more than `band` × the instrument's max
    shares. Cuts churn from tiny signal wiggles. Stateless (recomputes prior
    target from prcSoFar[:, :-1]); an approximation of a true position band."""
    if band <= 0:
        return get_pos

    def f(prices):
        target = get_pos(prices)
        if prices.shape[1] < 2:
            return target
        prev = get_pos(prices[:, :-1])
        max_sh = np.maximum((dollar_limits(len(target)) / prices[:, -1]).astype(int), 1)
        keep = np.abs(target - prev) < band * max_sh
        return np.where(keep, prev, target).astype(int)
    return f


# ============================================================================
# §3  BASELINE ALPHAS  — swappable references. Un-tuned, round defaults. These
#     are the numbers to beat (see bench.py). Higher score = want more long.
# ============================================================================
def alpha_reversion(prices):
    """Mean reversion: cheap-vs-own-recent-average names score positive.
    This universe favours reversion (see the dashboard portfolio view)."""
    return -zscore(prices, 20)


def alpha_momentum(prices):
    """Trend following: recent winners score positive."""
    return momentum(prices, 60)


def alpha_xs_rank(prices):
    """Cross-sectional reversion: rank by recent 5-day return, fade the leaders."""
    return -rank(returns(prices, 5))


BASELINES = {
    "reversion": alpha_reversion,
    "momentum": alpha_momentum,
    "xs_rank": alpha_xs_rank,
}


# --- extra reversion signals for the research catalog (research.py) ----------
def zrev(w):
    """Factory: contrarian z-score reversion with lookback w."""
    return lambda p: -zscore(p, w)


def alpha_rev_ema(prices):
    """Reversion vs an EMA baseline (smoother trend estimate) instead of SMA."""
    e = ema(prices, 20)
    sd = prices[:, -min(20, prices.shape[1]):].std(axis=1)
    return -np.divide(prices[:, -1] - e, sd, out=np.zeros_like(e), where=sd > 1e-9)


def alpha_rev_rankz(prices):
    """Outlier-robust reversion: rank of the z-score (bounded, tail-insensitive)."""
    return -rank(zscore(prices, 20))


def alpha_catchup(prices, w=5, lb=60):
    """Lead-lag catch-up: long stocks that LAGGED the index's recent move (expect
    them to catch up), short those that ran ahead. gap = beta·index_move − stock_move
    over the last w days (beta from a `lb`-day regression). Tested via the BACKTESTER
    (clean) — its raw IC is inflated by shared-endpoint bias, so trust the Score."""
    n, t = prices.shape
    if t < w + 2:
        return np.zeros(n)
    L = min(lb, t - 1)
    r = np.diff(np.log(prices[:, -(L + 1):]), axis=1)
    r0 = r[0]; v0 = float(r0 @ r0)
    beta = (r @ r0) / v0 if v0 > 1e-12 else np.zeros(n)
    idx_mv = np.log(prices[0, -1] / prices[0, -1 - w])
    stk_mv = np.log(prices[:, -1] / prices[:, -1 - w])
    return beta * idx_mv - stk_mv               # >0 = lagged the move → long


def alpha_boll(prices, w=20, k=1.0):
    """Bollinger-style reversion: z-score reversion that only ACTS outside the ±k
    band (dead-zone near the mean) — trades extremes, flat otherwise. Bollinger
    bands ARE z-scores (band = mean ± k·std), so this is the 'only at the band'
    flavour of the same edge; its selectivity is why it looks steadier."""
    z = zscore(prices, w)
    return np.where(np.abs(z) > k, -z, 0.0)


def alpha_rev_blend(prices):
    """Multi-window reversion: average the 10/20/40-day z-score reversion signals.
    Diversifying across horizons beat any single window in research.py
    (net Score ~69 vs ~57 for w=20, and a better walk-forward profile)."""
    return (zrev(10)(prices) + zrev(20)(prices) + zrev(40)(prices)) / 3.0


# --- adaptive (EWMA) reversion: self-recalibrating normalization -------------
def zrev_ewma(span):
    """Factory: contrarian reversion off an EWMA z-score (adaptive normalization)."""
    return lambda p: -ewma_z(p, span)


def alpha_rev_eblend(prices):
    """Adaptive twin of alpha_rev_blend using EWMA z at spans 10/20/40. Only worth
    shipping over the static blend if it wins out-of-sample (see SIGNALS.md)."""
    return (zrev_ewma(10)(prices) + zrev_ewma(20)(prices) + zrev_ewma(40)(prices)) / 3.0


# Full signal catalog for research.py / the dashboard Signals tab. Higher = long.
SIGNALS = {
    "rev_z10": zrev(10), "rev_z20": zrev(20), "rev_z30": zrev(30),
    "rev_z40": zrev(40), "rev_z60": zrev(60),
    "rev_ema20": alpha_rev_ema, "rev_rankz": alpha_rev_rankz,
    "rev_blend": alpha_rev_blend,
    "rev_ez10": zrev_ewma(10), "rev_ez20": zrev_ewma(20), "rev_ez40": zrev_ewma(40),
    "rev_eblend": alpha_rev_eblend,
    "catchup": alpha_catchup, "boll": alpha_boll,
    "xs_rev5": alpha_xs_rank, "momentum60": alpha_momentum,
}


# ============================================================================
# §3b  THE SUBMISSION — TWO-LEG FULL-LIMIT REVERSION
#      Reverse-engineering (analyze.py / DGP.md) says the graded universe is a
#      ONE-FACTOR market: ALGO (inst 0) IS the index; the other 50 are ~80%
#      idiosyncratic. TWO short-horizon reversions pay, and the score comes from
#      trading BOTH at (near) the full dollar limits:
#        Leg B  — idiosyncratic: cross-sectional z-reversion on the 50 names,
#                 demeaned so it's ~market-neutral, each sized toward its $10k limit.
#        Leg A  — index: ALGO mean-reverts over ~5 days (k-day return autocorr
#                 ≈ −0.1, stable across windows). Traded at its $100k limit (10×)
#                 with the 0.2bp fee (5× lower), it is the single biggest score
#                 contributor — the designers' fingerprint, on its own faster window.
#      SIZING is the dominant lever: Score scales ~linearly with deployed capital
#      up to the limit clip (Sharpe is scale-invariant), so both legs run hot.
#      Validated across held-out windows + synthetic re-draws (see DGP.md/SIGNALS.md),
#      not one sample. This replaced the timid SCALE=2 single-signal book (~105 → ~300).
# ============================================================================
def two_leg_positions(prc, idio_w, algo_w, idio_scale, algo_frac, algo_scale,
                      idio_sizing):
    """Two independent legs summed into one legal integer-share book. Stateless:
    recomputes entirely from `prc` (nInst, t), no globals. Leg A overwrites the
    ALGO slot that Leg B leaves flat, so the index gets its own window + sizing."""
    n, t = prc.shape
    pos = np.zeros(n, dtype=int)
    if t < max(idio_w, algo_w) + 2:
        return pos
    limits = dollar_limits(n)
    # --- Leg B: idiosyncratic cross-sectional reversion on the 50 non-index names ---
    s = (-zscore(prc, idio_w)).astype(float)
    s[0] = 0.0                                   # exclude the index from this leg
    s[1:] = s[1:] - np.nanmean(s[1:])            # neutralise among the 50 (≈$-neutral)
    if idio_sizing == "inverse_vol":
        pos = size_inverse_vol(s, prc, idio_scale)
    else:
        pos = size_fraction_of_limit(s, prc, idio_scale)
    # --- Leg A: ALGO index reversion at (a fraction of) its $100k limit ---
    za = zscore(prc, algo_w)[0]
    fa = float(np.clip(-za / algo_scale, -1.0, 1.0) * algo_frac)
    pos[0] = int(fa * limits[0] / prc[0, -1])
    return pos


def make_two_leg(idio_w=10, algo_w=5, idio_scale=0.10, algo_frac=1.0,
                 algo_scale=0.10, idio_sizing="fraction"):
    """Factory (research.py / simulate.py / sweeps): bind a two-leg config as a
    get_position(prc) callable without editing the module knobs."""
    return lambda prc: two_leg_positions(prc, idio_w, algo_w, idio_scale,
                                         algo_frac, algo_scale, idio_sizing)


# ============================================================================
# §4  YOUR STRATEGY  — EDIT THIS SECTION.
#     STRATEGY="two_leg" ships the §3b book (the submission). STRATEGY="single"
#     falls back to the single-signal alpha() path below (kept for research).
#     Lower *_SCALE = more aggressive (larger positions, up to the $ limit).
# ============================================================================
STRATEGY = "two_leg"     # "two_leg" (submission) | "single" (alpha() path)

# --- two-leg knobs (the submission; tuned on held-out windows + synthetic panels) ---
IDIO_WINDOW = 10         # reversion lookback for the 50 idiosyncratic names
ALGO_WINDOW = 5          # reversion lookback for the ALGO index leg (reverts faster)
IDIO_SCALE = 0.10        # idio aggression: z / IDIO_SCALE clipped to ±1 (small = hot)
ALGO_SCALE = 0.10        # index aggression: z / ALGO_SCALE clipped to ±1
ALGO_FRAC = 1.0          # fraction of ALGO's $100k limit to deploy (dial down to cap tail)
IDIO_SIZING = "fraction" # idio leg sizing: "fraction" | "inverse_vol"

# --- single-signal path knobs (only used when STRATEGY="single") ---
ACTIVE = "reversion"     # which baseline alpha() delegates to by default
SIZING = "fraction"      # "fraction" | "inverse_vol" | "kelly"
SCALE = 0.10             # score magnitude that maps to a full-size position
MIN_HISTORY = 5          # stay flat until we have at least this many days
SMOOTH = 1               # EMA span applied to the signal (1 = off)
HOLD_DAYS = 1            # re-decide the book every N days (1 = every day)
BAND = 0.0               # no-trade band, fraction of max shares (0 = off)


def alpha(prices):
    """Single-signal idea (used when STRATEGY="single"). prices: (nInst, t) →
    one raw score per instrument (higher = want to be more long).

    This is the idiosyncratic reversion signal on its own; the shipped book
    (§3b two_leg) also trades the ALGO index leg, which roughly doubles the score.
    """
    return zrev(IDIO_WINDOW)(prices)


# ============================================================================
# §5  PLUMBING  — the eval.py contract. Stateless (no globals) so nothing leaks
#     between backtest runs. You normally don't need to touch this.
# ============================================================================
def _raw_positions(prc, signal_fn, sizing, scale):
    """signal -> neutralize -> size -> legal integer shares (no turnover control)."""
    n, t = prc.shape
    if t < MIN_HISTORY:
        return np.zeros(n, dtype=int)
    scores = neutralize(signal_fn(prc))
    if sizing == "inverse_vol":
        return size_inverse_vol(scores, prc, scale)
    if sizing == "kelly":
        return size_kelly(scores, prc, scale)
    return size_fraction_of_limit(scores, prc, scale)


def _positions(prc, signal_fn, sizing, scale, smooth, hold, band):
    """Full pipeline: (optional signal smoothing) -> sizing -> (optional hold /
    no-trade band). With smooth=1, hold=1, band=0 this is exactly _raw_positions."""
    sig = smooth_signal(signal_fn, smooth)
    get_pos = lambda p: _raw_positions(p, sig, sizing, scale)
    get_pos = hold_every(get_pos, hold)
    get_pos = no_trade_band(get_pos, band)
    return get_pos(prc)


def make_get_position(signal_fn=None, active=None, sizing=None, scale=None,
                      smooth=None, hold=None, band=None):
    """Factory for research/bench: bind a config without editing the module.
    Pass either a signal_fn directly, an `active` baseline name, or neither
    (uses your §4 alpha()). Turnover knobs default to the module settings."""
    sig = signal_fn if signal_fn is not None else (BASELINES[active] if active else alpha)
    s = sizing if sizing is not None else SIZING
    k = scale if scale is not None else SCALE
    sm = smooth if smooth is not None else SMOOTH
    h = hold if hold is not None else HOLD_DAYS
    b = band if band is not None else BAND
    return lambda prc: _positions(prc, sig, s, k, sm, h, b)


def getMyPosition(prcSoFar):
    """Entry point eval.py / backtester.py call. STRATEGY selects the §3b two-leg
    book (the submission) or the single-signal alpha() path. Stateless."""
    if STRATEGY == "two_leg":
        return two_leg_positions(prcSoFar, IDIO_WINDOW, ALGO_WINDOW, IDIO_SCALE,
                                 ALGO_FRAC, ALGO_SCALE, IDIO_SIZING)
    return _positions(prcSoFar, alpha, SIZING, SCALE, SMOOTH, HOLD_DAYS, BAND)


# ============================================================================
# §6  COMBINE + EXPERIMENTS  — compose strategies, and skeptical ideas the data
#     argues against but that we measure anyway (see research.py / SIGNALS.md).
#     Kept out of the default path; use via research.py or by editing §4.
# ============================================================================
def combine_signals(specs):
    """specs: [(signal_fn, weight), ...]. Rank-normalise each signal (→[-1,1], so
    disparate signals are comparable) then weighted-sum. Returns a signal fn."""
    def f(prices):
        total = None
        for fn, w in specs:
            s = rank(fn(prices)) * w
            total = s if total is None else total + s
        return total
    return f


def combine_positions(specs):
    """specs: [(get_position_fn, weight), ...]. Weighted sum of share vectors, then
    re-clip to the $ limits. Returns a get_position fn (book-level combination)."""
    def f(prices):
        n = prices.shape[0]
        total = np.zeros(n)
        for gp, w in specs:
            total = total + w * np.asarray(gp(prices), dtype=float)
        limits = dollar_limits(n)
        max_sh = (limits / prices[:, -1]).astype(int)
        return np.clip(total.astype(int), -max_sh, max_sh)
    return f


def regime_gate(get_pos, turb_scale=0.5, min_hist=40):
    """EXPERIMENT: de-risk in the market's high-variance regime. Fits a 2-state
    Gaussian mixture on ALGO (index) returns and scales the book down when today's
    state is the turbulent one. Data shows ~no vol clustering, so expect ~no gain."""
    def f(prices):
        pos = get_pos(prices)
        if prices.shape[1] < min_hist:
            return pos
        from sklearn.mixture import GaussianMixture
        X = np.diff(np.log(prices[0])).reshape(-1, 1)
        gm = GaussianMixture(2, covariance_type="full", random_state=0).fit(X)
        turb = int(np.argmax(gm.covariances_.ravel()))
        cur = int(gm.predict(X[-1:])[0])
        scale = turb_scale if cur == turb else 1.0
        return (pos * scale).astype(int)
    return f


def beta_neutralize(get_pos, index_col=0, lookback=60):
    """Hedge the book's exposure to the ALGO market factor (inst 0). Estimate each
    name's beta to ALGO over `lookback` days, then project the beta direction out
    of the position vector so sum(beta_i·pos_i) ≈ 0. This is the 'cracking the
    data' insight applied: ALGO is the index, so neutralising to it isolates the
    idiosyncratic reversion. Re-clips to limits. Stateless."""
    def f(prices):
        pos = np.asarray(get_pos(prices), dtype=float)
        n, t = prices.shape
        if t < lookback + 2:
            return pos.astype(int)
        r = np.diff(np.log(prices[:, -(lookback + 1):]), axis=1)
        r0 = r[index_col]
        v0 = float(r0 @ r0)
        if v0 < 1e-12:
            return pos.astype(int)
        beta = (r @ r0) / v0                       # per-name beta to ALGO
        denom = float(beta @ beta)
        if denom > 1e-9:
            pos = pos - (float(beta @ pos) / denom) * beta   # remove net-beta component
        max_sh = (dollar_limits(n) / prices[:, -1]).astype(int)
        return np.clip(pos.astype(int), -max_sh, max_sh)
    return f


def alpha_trend_revert(prices, rev_w=20, mom_w=60, class_w=60):
    """EXPERIMENT: per-stock, follow momentum for 'trenders' and reversion for
    'reverters', classified by trailing return autocorrelation. Data shows per-stock
    trend is a coin-flip, so expect this to underperform pure reversion."""
    n, t = prices.shape
    z = zscore(prices, rev_w)
    mo = momentum(prices, min(mom_w, t - 1))
    r = np.diff(np.log(prices), axis=1)
    if r.shape[1] < class_w + 1:
        return -z
    ac = np.array([np.corrcoef(r[k, -class_w:], r[k, -class_w - 1:-1])[0, 1]
                   if r.shape[1] > class_w else 0.0 for k in range(n)])
    ac = np.nan_to_num(ac)
    return np.where(ac > 0, mo, -z)     # trenders -> momentum, reverters -> -z


# --- EV / confidence / cost-gate overlays ------------------------------------
def cost_gate(signal_fn, keep=0.6):
    """'Is it worth trading?' — keep only the highest-conviction names each day
    (top `keep` fraction by |signal|), zero the rest. Marginal names' expected edge
    doesn't clear the ~1bp fee, so dropping them cuts fee drag. Signal-level."""
    def f(prices):
        s = np.array(signal_fn(prices), dtype=float)
        n = len(s)
        k = max(1, int(round(keep * n)))
        thr = np.sort(np.abs(s))[::-1][k - 1]     # k-th largest |signal|
        return np.where(np.abs(s) >= thr, s, 0.0)
    return f


def confidence_scale(get_pos, signal_fn, lookback=30, target_ic=0.02, floor=0.3, cap=1.3):
    """Scale the whole book by the signal's RECENT realised IC (rolling cross-
    sectional corr of past signal vs next-day return). Trade bigger when the edge
    has been working lately, smaller when not. Stateless (recomputes)."""
    def f(prices):
        pos = np.asarray(get_pos(prices), dtype=float)
        n, t = prices.shape
        if t < lookback + 3:
            return pos.astype(int)
        ics = []
        for d in range(t - lookback, t - 1):
            s = signal_fn(prices[:, :d + 1]); s = s - np.nanmean(s)
            fwd = prices[:, d + 1] / prices[:, d] - 1; fwd = fwd - fwd.mean()
            den = np.std(s) * np.std(fwd)
            if den > 1e-12:
                ics.append(float(np.mean(s * fwd) / den))
        conf = np.clip((np.mean(ics) / target_ic) if ics else 1.0, floor, cap)
        max_sh = (dollar_limits(n) / prices[:, -1]).astype(int)
        return np.clip((pos * conf).astype(int), -max_sh, max_sh)
    return f


def markov_gate(get_pos, turb_scale=0.4, min_hist=60):
    """Regime de-risking via a proper 2-state Markov-SWITCHING model (statsmodels)
    on the ALGO factor. Unlike the per-day GMM (regime_gate/regime_scale), a
    Markov-switching model has TRANSITION probabilities — it models regime
    persistence. Scales the book down by the filtered probability of the
    high-variance regime. SLOW (MLE fit per day) — research/robustness only; likely
    too slow for the submission."""
    def f(prices):
        pos = np.asarray(get_pos(prices), dtype=float)
        n, t = prices.shape
        if t < min_hist:
            return pos.astype(int)
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
        y = np.diff(np.log(prices[0])) * 100.0            # ALGO returns, scaled for stability
        try:
            res = MarkovRegression(y, k_regimes=2, switching_variance=True).fit(disp=False)
            nm = list(res.model.param_names)
            sig2 = [float(res.params[i]) for i, x in enumerate(nm) if "sigma2" in x]
            turb = int(np.argmax(sig2))                # which regime is high-variance
            fp = np.asarray(res.filtered_marginal_probabilities)
            p_turb = float(fp[-1, turb]) if fp.shape[-1] == 2 else float(fp[turb, -1])
        except Exception:
            return pos.astype(int)
        scale = 1.0 - (1.0 - turb_scale) * p_turb
        max_sh = (dollar_limits(n) / prices[:, -1]).astype(int)
        return np.clip((pos * scale).astype(int), -max_sh, max_sh)
    return f


def regime_scale(get_pos, turb_scale=0.4, min_hist=40):
    """Continuous version of regime_gate: scale the book by the GMM posterior
    probability of the calm state (smooth de-risking) rather than a hard on/off."""
    def f(prices):
        pos = np.asarray(get_pos(prices), dtype=float)
        if prices.shape[1] < min_hist:
            return pos.astype(int)
        from sklearn.mixture import GaussianMixture
        X = np.diff(np.log(prices[0])).reshape(-1, 1)
        gm = GaussianMixture(2, covariance_type="full", random_state=0).fit(X)
        turb = int(np.argmax(gm.covariances_.ravel()))
        p_turb = float(gm.predict_proba(X[-1:])[0, turb])
        scale = 1.0 - (1.0 - turb_scale) * p_turb    # calm→1.0, turbulent→turb_scale
        max_sh = (dollar_limits(len(pos)) / prices[:, -1]).astype(int)
        return np.clip((pos * scale).astype(int), -max_sh, max_sh)
    return f
