"""
##########################################################################################
### SAFE Live + structural-pair confirmation + inversion protection · submission build ###
##########################################################################################
Self-contained tournament build derived from SAFE Live.  Six structural pairs selected using
only days 1-500 supply a small confidence adjustment; their later days 501-750 are therefore
a genuinely forward validation period.  The pair sleeve and all safety gates are causal.

  1. LEAN FORECASTS: computes only the signals the gates read (champ + mom/momJT/residMom).
     The six diagnostic signals (revL/ll2/resid/btiao/volsc/momVS) are research-only and
     never traded — dropped here. ll2's double-size ridge was the single largest cost.
  2. WINDOWED CACHE: forecasts/ICs are only ever needed over the trailing
     ROT_W + max(KILL_P, ROT_P) days, so the cache fills (and prunes) a sliding window
     instead of all history. First call is seconds, not minutes, at any history length.

Protection stack (all verified in the SAFE_rotate research build):
  * champion = lead-lag ensemble + reversion blend (== SAFE_lldollar, the main edge)
  * momentum challengers mom/momJT/residMom — IC-gated rotation (Balanced: W=40,P=7,t=2.5)
  * xsac validator — direct cross-sectional autocorr; relaxes the gate in a momentum regime
  * kill switch — flattens the idio book if the traded edge's IC INVERTS (sustained t<-3)
  * inversion response — after a faster, still-persistent failure test, reverses the
    stock book and beta-hedges it with ALGO; completely inactive on visible days 1-750
  * structural pairs — six independently selected cointegrating relationships influence
    only 10% of affected stock capacity, subject to live stationarity and payoff gates
  * ALGO leg — net-$ gate; else IC-gated FADE (default) vs TREND (index TSMOM)

------------------------------------------------------------------------------------------
READING GUIDE (naming conventions used throughout this file)
  * Instrument 0 is the ALGO index; instruments 1..49 are the 50 tradable stocks.
    Slicing `[1:]` therefore means "the stock book" and `[0]` means "the ALGO leg".
  * A "day" is an integer count of price columns seen so far.  A function that takes
    `day`/`as_of_day` answers "what would I have known using the first `day` columns?".
  * A "signal"/"forecast" is a demeaned 50-vector of desired per-stock exposures; we
    trade its SIGN (long the positives, short the negatives) at the dollar limit.
  * IC = daily information coefficient = cross-sectional corr(forecast, realized return).
  * "book return" = the realized PnL proxy of trading a signal's sign for one day.
  * Public surface consumed by the evaluator: `getMyPosition` and `BOOK`.  Everything
    prefixed with `_` is internal.  ALL-CAPS names are fixed tuning knobs.
==========================================================================================
"""
import numpy as np

BOOK = "SAFE · STRUCTURAL PAIRS · PROTECTED"

# ---- trading knobs (identical to SAFE_lldollar) -----------------------------
HALF_LIVES  = (250, 500, 1000, 2000)  # ridge memory half-lives (days) for the lead-lag ensemble
RIDGE_A     = 0.1        # ridge (L2) penalty in the exponentially-weighted regression
BLEND       = 0.3        # champion = (1-BLEND)*lead-lag + BLEND*short-reversion
REV_W       = 10         # look-back (days) for the short-horizon reversion term
CONTRA_DOL  = 1_000_000  # notional scale for the ALGO index leg's z-score sizing
CONTRA_K    = 30         # horizon (days) of the ALGO index momentum move used for its z-score
CONTRA_WZ   = 60         # window (days) used to standardise that move into a z-score
WARMUP      = 96         # min history before we trade anything at all
ALGO_LL_DOLLAR = 50_000  # if |net stock $| exceeds this, ALGO leg just hedges that net exposure

# ---- rotation knobs (ADOPTED pnl-W60 gate; verify-pnl-gate workflow) ----------
GATE_MODE  = "pnl"   # how a challenger must beat the champion: "ic" | "pnl" (adopted) | "sharpe"
ROT_W      = 60      # trailing window (days) over which the rotation gate is measured
ROT_TCRIT  = 2.5     # t-stat bar a challenger must clear   (ic mode only)
ROT_MARGIN = 0.0     # extra IC edge a challenger must show  (ic mode only)
PNL_MARGIN = 0.0     # (pnl/sharpe) required trailing book-return edge over champion
ROT_P      = 5       # a challenger must win this many consecutive as-of days to be adopted
ROT_BONF   = True    # Bonferroni-inflate the t-bar for testing multiple challengers

# ---- momentum challengers ------------------------------------------------------
MOMJT_L    = 120     # momJT long look-back (days)
MOMJT_S    = 20      # momJT skip-recent window (days), to avoid short-term reversal
RESIDM_L   = 120     # residMom regression/look-back window (days)
RESIDM_S   = 20      # residMom skip-recent window (days)

# ---- ALGO index leg: IC-gated FADE vs TREND ------------------------------------
ALGO_ROT_W = 120     # window (days) over which we test whether the ALGO trend "works"
ALGO_ROT_H = 5       # step / forward-horizon (days) between the samples in that test
ALGO_TCRIT = 3.0     # t-stat bar for the ALGO trend to be judged significant
ALGO_P     = 10      # ALGO must pass on this many consecutive days to switch FADE->TREND
                     # (accelerant tested + reverted: inert — see SAFE_rotate.py note / verify_accel2.py)

# ---- kill switch -----------------------------------------------------------------
KILL_ON    = True    # master enable for the kill switch
KILL_TCRIT = 3.0     # the traded signal's IC must be negative with t < -KILL_TCRIT ...
KILL_P     = 10      # ... on every one of the last KILL_P days to flatten the stock book

# ---- hidden-regime inversion response ------------------------------------------
# If the selected stock signal's realized sign-book payoff is significantly
# negative over 20 days on three consecutive as-of dates, treat that as evidence
# that the relationship has reversed.  These values were chosen from causal
# stress tests, not by optimizing the visible-period score: this branch never
# activates anywhere in days 1-750.
INV_W      = 20      # window (days) of book-return over which the failure test is run
INV_TCRIT  = 1.5     # book-return must be negative with t < -INV_TCRIT ...
INV_P      = 3       # ... on this many consecutive as-of days to confirm an inversion
INV_BETA_W = 120     # window (days) for the market-beta estimate used by the ALGO hedge

# ---- structural-pair confirmation ---------------------------------------------
# These six disjoint identities survived BH-FDR selection using days 1-500 and
# then remained mean-reverting on the untouched days 501-750.  Their column
# names are AENO-NWIG, EORC-NGTE, HETT-ULXY, SMAH-ILVX, HUXZ-ACAC, CTGI-EELT.
PAIR_IDENTITIES = ((1, 20), (13, 45), (7, 40), (10, 46), (8, 27), (25, 37))  # (leg_i, leg_j) column pairs
PAIR_BETA_W      = 250     # window (days) for the pair's hedge-ratio (beta) regression
PAIR_SPREAD_W    = 60      # window (days) for standardising the spread into a z-score
PAIR_ENTRY_Z     = 0.5     # |z| at which we open a pair (fade the spread)
PAIR_EXIT_Z      = 0.25    # |z| at which we close a pair
PAIR_DOLLARS     = 10_000.0  # notional per pair leg
PAIR_WEIGHT      = 0.10    # pairs may steer only 10% of an affected stock's dollar capacity
PAIR_LIVE_T_MAX  = -2.0    # spread must still be mean-reverting (error-correction t below this)
PAIR_GATE_W      = 60      # window (days) of realised pair-vs-SAFE payoff used to enable the sleeve

# ---- xsac validator ----------------------------------------------------------------
XSAC_W     = 40      # window (days) for the average cross-sectional lag-1 autocorrelation
XSAC_TH    = 0.07    # autocorr above this on every recent day => "momentum regime"
XSAC_P     = 5       # how many recent days must all exceed XSAC_TH to declare that regime
ROT_P_FAST = 3       # in a momentum regime, require only this many winning days (vs ROT_P)

# windowed cache: everything the gates read lies within this trailing span
#   pnl gate: ROT_P + ROT_W - 1 = 64   _kill: KILL_P + ROT_W - 1 = 69 (deepest)
#   _in_momentum_regime: XSAC_P + XSAC_W = 45  (RET-only)
LOOKBACK   = ROT_W + max(KILL_P, ROT_P) + 6          # = 76, covers all of the above (deepest 69, margin 7)
PRUNE_PAD  = 10                                      # keep a small margin beyond LOOKBACK

# ---- module-level caches (all keyed by day; pruned to the sliding window) ------------
_DOLLAR_LIMITS      = None   # per-instrument dollar position limits (ALGO = 100k, stocks = 10k)
_FORECAST_CACHE     = {}     # day -> {signal_name: 50-vec forecast for that day}
_IDIO_RETURN_CACHE  = {}     # day -> realized demeaned stock return over that day
_IC_CACHE           = {}     # (signal_name, day) -> realized daily IC (small floats; kept)
_ALGO_Z_CACHE       = {}     # history_length -> ALGO index trend z-score from the first N columns
_XSAC_CACHE         = {}     # day -> cross-sectional lag-1 autocorr corr(returns[day-1], returns[day])
_BOOK_RETURN_CACHE  = {}     # (signal_name, day) -> daily as-if-traded book return: sign(forecast).return
_PAIR_STATE         = {}     # (i, j) -> current pair position state: -1 short spread / 0 flat / +1 long
_PAIR_PAYOFFS       = []     # rolling realised "pair minus SAFE" counterfactual PnL, one per day
_PAIR_PREV_DELTA    = None   # yesterday's pair-vs-SAFE share deltas (to score the counterfactual)
_PAIR_PREV_PRICES   = None   # yesterday's prices (to turn the delta into a dollar payoff)
_PAIR_PREV_T        = None   # yesterday's day index (so we only score consecutive calls)


def _dollar_position_limits(n_inst):
    """Cache and return the per-instrument dollar caps: ALGO (0) = 100k, each stock = 10k."""
    global _DOLLAR_LIMITS
    if _DOLLAR_LIMITS is None or len(_DOLLAR_LIMITS) != n_inst:
        _DOLLAR_LIMITS = np.full(n_inst, 10_000.0)
        _DOLLAR_LIMITS[0] = 100_000.0
    return _DOLLAR_LIMITS


def _exp_weighted_ridge(features, targets, half_life, ridge):
    """Exponentially-weighted, ridge-regularised least squares.

    Recent rows count more (weight decays with the given half-life).  Returns the
    coefficient matrix plus the weighted means of the features and targets, so the
    caller can form a centred prediction.
    """
    n_obs, n_features = features.shape
    decay = 0.5 ** (1.0 / half_life)
    weights = decay ** np.arange(n_obs - 1, -1, -1)
    weight_sum = weights.sum()
    mean_x = (weights[:, None] * features).sum(0) / weight_sum
    mean_y = (weights[:, None] * targets).sum(0) / weight_sum
    x_centered, y_centered = features - mean_x, targets - mean_y
    xtwx = x_centered.T @ (weights[:, None] * x_centered)
    xtwy = x_centered.T @ (weights[:, None] * y_centered)
    jitter = 1e-8 * np.trace(xtwx) / n_features
    coefs = np.linalg.solve(xtwx + (jitter + ridge) * np.eye(n_features), xtwy)
    return coefs, mean_x, mean_y


def _compute_forecasts(prices):
    """Compute the traded signals (each a demeaned 50-vector). Identical formulas to SAFE_rotate."""
    log_prices = np.log(prices)
    returns = log_prices[:, 1:] - log_prices[:, :-1]   # daily log returns, shape (n_inst, n_days)
    n_days = returns.shape[1]
    forecasts = {}

    # champion: lead-lag ensemble blended with short reversion.
    # For each half-life, regress each stock's next return on all instruments' prior
    # returns, predict the next return, then z-score across stocks; average the ensemble.
    leadlag_components = []
    for half_life in HALF_LIVES:
        coefs, mean_x, mean_y = _exp_weighted_ridge(returns[:, :-1].T, returns[1:, 1:].T, half_life, RIDGE_A)
        pred = mean_y + (returns[:, -1] - mean_x) @ coefs
        component = pred - pred.mean()
        leadlag_components.append(component / (component.std() + 1e-12))
    leadlag_z = np.mean(leadlag_components, 0)
    # short-horizon reversion: recent winners are expected to give back, so we negate.
    recent_move = log_prices[1:, -1] - log_prices[1:, -1 - REV_W]
    recent_move = recent_move - recent_move.mean()
    reversion_z = -recent_move / (recent_move.std() + 1e-12)
    forecasts["champ"] = (1 - BLEND) * leadlag_z + BLEND * reversion_z

    # mom: short cross-sectional momentum (the raw recent move — sign-flip of the champion's reversion term)
    forecasts["mom"] = recent_move.copy()

    # momJT: Jegadeesh-Titman cross-sectional momentum (long lookback, skip recent reversal)
    if log_prices.shape[1] >= MOMJT_L + 1:
        momentum_gap = log_prices[1:, -1 - MOMJT_S] - log_prices[1:, -1 - MOMJT_L]
        momentum_gap = momentum_gap - momentum_gap.mean()
        forecasts["momJT"] = momentum_gap / (momentum_gap.std() + 1e-12)
    else:
        forecasts["momJT"] = forecasts["champ"].copy()

    # residMom: Blitz-Huij-Martens residual (factor-neutral) momentum.
    # Strip each stock's exposure to the ALGO index, then accumulate the residual return.
    if n_days >= RESIDM_L + 1:
        stock_returns_window = returns[1:, -RESIDM_L:]
        algo_returns_window = returns[0, -RESIDM_L:]
        algo_returns_centered = algo_returns_window - algo_returns_window.mean()
        betas = (stock_returns_window @ algo_returns_centered) / (algo_returns_centered @ algo_returns_centered + 1e-12)
        residuals = stock_returns_window - betas[:, None] * algo_returns_window[None, :]
        cum_resid = (residuals[:, :RESIDM_L - RESIDM_S] if RESIDM_S > 0 else residuals).sum(1)
        cum_resid = cum_resid - cum_resid.mean()
        forecasts["residMom"] = cum_resid / (cum_resid.std() + 1e-12)
    else:
        forecasts["residMom"] = forecasts["champ"].copy()

    return forecasts


CHALLENGERS = ("mom", "momJT", "residMom")


def _ensure_cache(prices):
    """Fill the forecast/return caches over the trailing LOOKBACK window only, then prune.
    Gate/kill/xsac lookbacks all lie inside the window (see LOOKBACK); ICs for older days
    are already memoized in _IC_CACHE when they were inside the window."""
    n_days = prices.shape[1]
    window_start = max(WARMUP, n_days - LOOKBACK)
    for day in range(window_start, n_days + 1):
        if day not in _FORECAST_CACHE:
            _FORECAST_CACHE[day] = _compute_forecasts(prices[:, :day])
        if day not in _IDIO_RETURN_CACHE and day < n_days:
            stock_return = np.log(prices[1:, day]) - np.log(prices[1:, day - 1])
            _IDIO_RETURN_CACHE[day] = stock_return - stock_return.mean()
    prune_before = window_start - PRUNE_PAD
    for cache in (_FORECAST_CACHE, _IDIO_RETURN_CACHE, _XSAC_CACHE):
        for key in [k for k in cache if k < prune_before]:
            del cache[key]
    for stale_day in [d for (name, d) in _BOOK_RETURN_CACHE if d < prune_before]:  # prune (name, day) keys
        for name in ("champ",) + CHALLENGERS:
            _BOOK_RETURN_CACHE.pop((name, stale_day), None)


def _daily_ic(name, day):
    """Cross-sectional corr between a signal's forecast and the realized return on `day`."""
    key = (name, day)
    value = _IC_CACHE.get(key)
    if value is None:
        forecast = _FORECAST_CACHE[day][name]
        realized = _IDIO_RETURN_CACHE[day]
        forecast_c = forecast - forecast.mean()
        realized_c = realized - realized.mean()
        denom = np.sqrt((forecast_c @ forecast_c) * (realized_c @ realized_c))
        value = float(forecast_c @ realized_c / denom) if denom > 1e-18 else 0.0
        _IC_CACHE[key] = value
    return value


def _ic_series(name, start_day, end_day):
    """Vector of daily ICs for a signal over [start_day, end_day)."""
    return np.array([_daily_ic(name, day) for day in range(start_day, end_day)])


def _daily_book_return(name, day):
    """Realized one-day PnL proxy of trading a signal's sign: sum(sign(forecast) * return)."""
    key = (name, day)
    value = _BOOK_RETURN_CACHE.get(key)
    if value is None:
        value = float((np.sign(_FORECAST_CACHE[day][name]) * _IDIO_RETURN_CACHE[day]).sum())
        _BOOK_RETURN_CACHE[key] = value
    return value


def _book_return_series(name, start_day, end_day):
    """Vector of daily book returns for a signal over [start_day, end_day)."""
    return np.array([_daily_book_return(name, day) for day in range(start_day, end_day)])


def _rotation_t_threshold():
    """The champion-beating t-bar, Bonferroni-inflated for the number of challengers tested."""
    if ROT_BONF and len(CHALLENGERS) > 1:
        return float(np.sqrt(ROT_TCRIT ** 2 + 2.0 * np.log(len(CHALLENGERS))))
    return ROT_TCRIT


def _xsectional_autocorr(day):
    """Lag-1 cross-sectional autocorrelation: corr(yesterday's returns, today's returns)."""
    value = _XSAC_CACHE.get(day)
    if value is None:
        prev = _IDIO_RETURN_CACHE.get(day - 1)
        curr = _IDIO_RETURN_CACHE.get(day)
        if prev is None or curr is None:
            return None
        denom = np.sqrt((prev @ prev) * (curr @ curr))
        value = float(prev @ curr / denom) if denom > 1e-18 else 0.0
        _XSAC_CACHE[day] = value
    return value


def _mean_xsectional_autocorr(as_of_day):
    """Average of the last XSAC_W daily cross-sectional autocorrelations (None if too sparse)."""
    values = [_xsectional_autocorr(day) for day in range(as_of_day - XSAC_W + 1, as_of_day + 1)]
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if len(values) >= XSAC_W // 2 else None


def _in_momentum_regime(day):
    """True when the mean autocorrelation stayed above XSAC_TH on every one of the last XSAC_P days."""
    for as_of_day in range(day - XSAC_P, day):
        value = _mean_xsectional_autocorr(as_of_day)
        if value is None or value <= XSAC_TH:
            return False
    return True


def _best_challenger_at(as_of_day, t_threshold=None):
    """Pick the challenger that beats the champion at `as_of_day`, or None if none qualifies.
    Selection metric depends on GATE_MODE: information coefficient, raw book PnL, or Sharpe."""
    window_start = as_of_day - ROT_W + 1
    if window_start < WARMUP:
        return None
    if GATE_MODE == "ic":
        if t_threshold is None:
            t_threshold = _rotation_t_threshold()
        champ_ic = _ic_series("champ", window_start, as_of_day + 1)
        best_name = None
        best_metric = -1e18
        for name in CHALLENGERS:
            chal_ic = _ic_series(name, window_start, as_of_day + 1)
            ic_diff = chal_ic - champ_ic
            t_diff = ic_diff.mean() / (ic_diff.std() / np.sqrt(len(ic_diff)) + 1e-18)
            t_ic = chal_ic.mean() / (chal_ic.std() / np.sqrt(len(chal_ic)) + 1e-18)
            if ic_diff.mean() >= ROT_MARGIN and t_diff > t_threshold and chal_ic.mean() > 0 and t_ic > t_threshold and chal_ic.mean() > best_metric:
                best_metric = chal_ic.mean()
                best_name = name
        return best_name
    # profitability-based gate (adopted): beat champion on trailing realized book-return
    champ_book_ret = _book_return_series("champ", window_start, as_of_day + 1)
    best_name = None
    best_metric = -1e18
    for name in CHALLENGERS:
        chal_book_ret = _book_return_series(name, window_start, as_of_day + 1)
        if GATE_MODE == "sharpe":
            beats_champ = chal_book_ret.mean() / (chal_book_ret.std() + 1e-9) > champ_book_ret.mean() / (champ_book_ret.std() + 1e-9) + PNL_MARGIN and chal_book_ret.mean() > 0
            metric = chal_book_ret.mean() / (chal_book_ret.std() + 1e-9)
        else:  # "pnl"
            beats_champ = (chal_book_ret - champ_book_ret).mean() > PNL_MARGIN
            metric = chal_book_ret.mean()
        if beats_champ and metric > best_metric:
            best_metric = metric
            best_name = name
    return best_name


def _choose_signal(day):
    """Adopt a challenger only if the SAME one wins on every recent day; else stay with champion.
    In a momentum regime the persistence requirement and t-bar are relaxed."""
    momentum_regime = _in_momentum_regime(day)
    persistence = ROT_P_FAST if momentum_regime else ROT_P
    t_threshold = ROT_TCRIT if momentum_regime else _rotation_t_threshold()
    picks = [_best_challenger_at(as_of_day, t_threshold) for as_of_day in range(day - persistence, day)]
    if picks and picks[0] is not None and all(p == picks[0] for p in picks):
        return picks[0]
    return "champ"


def _algo_trend_z(log_algo):
    """z-score of the ALGO index's CONTRA_K-day momentum move, standardised over CONTRA_WZ days."""
    moves = log_algo[CONTRA_K:] - log_algo[:-CONTRA_K]
    if len(moves) < CONTRA_WZ:
        return None
    return float((moves[-1] - moves[-CONTRA_WZ:].mean()) / (moves[-CONTRA_WZ:].std() + 1e-12))


def _algo_trend_z_cached(log_algo, length):
    """_algo_trend_z evaluated on the first `length` columns, memoized by that length."""
    value = _ALGO_Z_CACHE.get(length)
    if value is None:
        value = _algo_trend_z(log_algo[:length])
        _ALGO_Z_CACHE[length] = value
    return value


def _algo_trend_significant_at(log_algo, day):
    """Does the ALGO trend z-score positively predict the next ALGO_ROT_H-day move (t > ALGO_TCRIT)?"""
    min_length = CONTRA_K + CONTRA_WZ + 1
    z_scores = []
    forward_moves = []
    for length in range(day - ALGO_ROT_H, day - ALGO_ROT_H - ALGO_ROT_W, -ALGO_ROT_H):
        if length < min_length:
            break
        z = _algo_trend_z_cached(log_algo, length)
        if z is None:
            continue
        z_scores.append(z)
        forward_moves.append(log_algo[length - 1 + ALGO_ROT_H] - log_algo[length - 1])
    if len(z_scores) < 8:
        return False
    z_scores = np.asarray(z_scores)
    forward_moves = np.asarray(forward_moves)
    if z_scores.std() < 1e-12 or forward_moves.std() < 1e-12:
        return False
    corr = float(np.corrcoef(z_scores, forward_moves)[0, 1])
    n = len(z_scores)
    if corr <= 0:
        return False
    t_corr = corr * np.sqrt((n - 2) / (1.0 - corr ** 2 + 1e-12))
    return t_corr > ALGO_TCRIT


def _algo_leg_mode(log_algo, day):
    """'trend' only if the ALGO trend is significant on every one of the last ALGO_P days; else 'fade'."""
    for as_of_day in range(day - ALGO_P + 1, day + 1):
        if not _algo_trend_significant_at(log_algo, as_of_day):
            return "fade"
    return "trend"


def _kill_switch_triggered(day, chosen_signal):
    """True when the traded signal's IC has been significantly NEGATIVE on every one of the last KILL_P days."""
    if not KILL_ON:
        return False
    for as_of_day in range(day - KILL_P, day):
        window_start = as_of_day - ROT_W + 1
        if window_start < WARMUP:
            return False
        ic = _ic_series(chosen_signal, window_start, as_of_day + 1)
        t_ic = ic.mean() / (ic.std() / np.sqrt(len(ic)) + 1e-18)
        if not (ic.mean() < 0 and t_ic < -KILL_TCRIT):
            return False
    return True


def _signal_selected_at(day):
    """The signal that would have been selected using only information available at `day`."""
    ready = day >= WARMUP + ROT_W + ROT_P
    return _choose_signal(day) if ready else "champ"


def _signal_inverted_at(day):
    """Causal test for a statistically meaningful negative signal payoff over the last INV_W days."""
    window_start = day - INV_W
    if window_start < WARMUP:
        return False
    book_returns = _book_return_series(_signal_selected_at(day), window_start, day)
    mean = float(book_returns.mean())
    t_stat = mean / (float(book_returns.std(ddof=1)) / np.sqrt(len(book_returns)) + 1e-18)
    return mean < 0 and t_stat < -INV_TCRIT


def _inversion_confirmed(day):
    """Require the failure condition to persist (INV_P consecutive days) before changing the book."""
    return all(_signal_inverted_at(as_of_day) for as_of_day in range(day - INV_P + 1, day + 1))


def _beta_hedge_algo_shares(prices, positions):
    """ALGO share count that offsets the stock book's rolling market beta to the index."""
    returns = np.diff(np.log(prices), axis=1)
    returns = returns[:, -min(INV_BETA_W, returns.shape[1]):]
    algo_returns = returns[0] - returns[0].mean()
    stock_returns = returns[1:] - returns[1:].mean(axis=1, keepdims=True)
    betas = stock_returns @ algo_returns / (algo_returns @ algo_returns + 1e-12)
    current_prices = prices[:, -1]
    stock_beta_dollars = float((positions[1:] * current_prices[1:]) @ betas)
    return -stock_beta_dollars / current_prices[0]


def _pair_ols_spread(y, x):
    """OLS-hedged spread y - (intercept + beta*x), returned with the hedge ratio beta."""
    x_centered = x - x.mean()
    beta = float(x_centered @ (y - y.mean()) / (x_centered @ x_centered + 1e-12))
    intercept = float(y.mean() - beta * x.mean())
    return y - intercept - beta * x, beta


def _pair_error_correction_t(spread):
    """t-stat of the error-correction coefficient: how fast the spread reverts to its mean.
    Strongly negative => stationary/mean-reverting (an Engle-Granger style stationarity check)."""
    lag = spread[:-1] - spread[:-1].mean()
    change = np.diff(spread)
    denominator = float(lag @ lag)
    if denominator < 1e-16:
        return 0.0
    rho = float(lag @ change / denominator)
    residual = change - rho * lag
    standard_error = np.sqrt(
        float(residual @ residual) / max(1, len(change) - 1) / denominator + 1e-18
    )
    return rho / standard_error


def _pair_positions(prices):
    """Current fixed-pair book, with a live rolling stationarity check.
    For each identity: only trade while the spread is still mean-reverting; enter when the
    spread is stretched (|z| >= entry), exit when it snaps back (|z| <= exit)."""
    positions = np.zeros(prices.shape[0])
    if prices.shape[1] < PAIR_BETA_W:
        return positions
    log_prices = np.log(prices)
    window = log_prices[:, -PAIR_BETA_W:]
    current_prices = prices[:, -1]
    for i, j in PAIR_IDENTITIES:
        spread, beta = _pair_ols_spread(window[i], window[j])
        if _pair_error_correction_t(spread) >= PAIR_LIVE_T_MAX:   # not reverting enough -> stand aside
            _PAIR_STATE[(i, j)] = 0
            continue
        recent = spread[-PAIR_SPREAD_W:]
        z = float((spread[-1] - recent.mean()) / (recent.std() + 1e-12))
        state = _PAIR_STATE.get((i, j), 0)
        if state == 0 and abs(z) >= PAIR_ENTRY_Z:
            state = -int(np.sign(z))   # fade: short the spread when it is high, long when low
        elif state != 0 and abs(z) <= PAIR_EXIT_Z:
            state = 0
        _PAIR_STATE[(i, j)] = state
        if state:
            scale = PAIR_DOLLARS / max(1.0, abs(beta))
            positions[i] += state * scale / current_prices[i]
            positions[j] += -state * beta * scale / current_prices[j]
    return positions


def _recalculate_algo(prices, positions):
    """Re-apply the ALGO leg policy after the stock book has been adjusted by the pair sleeve.
    Either hedge a large net stock exposure, or run the index z-score in TREND/FADE mode."""
    current_prices = prices[:, -1]
    dollar_limits = _dollar_position_limits(len(current_prices))
    stock_limits = (dollar_limits[1:] / current_prices[1:]).astype(int)
    stock_shares = np.clip(positions[1:], -stock_limits, stock_limits).astype(int)
    net_dollars = float((stock_shares * current_prices[1:]).sum())
    algo_share_cap = dollar_limits[0] / current_prices[0]
    if ALGO_LL_DOLLAR > 0 and abs(net_dollars) >= ALGO_LL_DOLLAR:
        positions[0] = float(np.sign(net_dollars) * algo_share_cap)
        return positions
    log_algo = np.log(prices[0])
    moves = log_algo[CONTRA_K:] - log_algo[:-CONTRA_K]
    z = (moves[-1] - moves[-CONTRA_WZ:].mean()) / (moves[-CONTRA_WZ:].std() + 1e-12)
    algo_z_shares = np.clip(z, -3, 3) / 3.0 * (CONTRA_DOL / current_prices[0])
    day = prices.shape[1]
    trend = day >= WARMUP + ALGO_ROT_W + ALGO_P and _algo_leg_mode(log_algo, day) == "trend"
    positions[0] = float(np.clip(algo_z_shares if trend else -algo_z_shares, -algo_share_cap, algo_share_cap))
    return positions


def getMyPosition(prices_so_far):
    """Entry point: given all prices seen so far (shape [n_inst, day]), return integer share positions.
    Instrument 0 is the ALGO index; 1..49 are the stocks."""
    global _PAIR_PREV_DELTA, _PAIR_PREV_PRICES, _PAIR_PREV_T
    prices_so_far = np.asarray(prices_so_far, dtype=float)
    n_inst, day = prices_so_far.shape
    dollar_limits = _dollar_position_limits(n_inst)
    current_prices = prices_so_far[:, -1]
    positions = np.zeros(n_inst)
    if day < WARMUP:
        return positions.astype(int)

    _ensure_cache(prices_so_far)
    ready = day >= WARMUP + ROT_W + ROT_P
    chosen_signal = _choose_signal(day) if ready else "champ"
    forecast = _FORECAST_CACHE[day][chosen_signal]

    log_prices = np.log(prices_so_far)

    # --- stock book: trade the chosen signal's sign at full dollar limit, unless killed ---
    killed = ready and _kill_switch_triggered(day, chosen_signal)
    if not killed:
        positions[1:] = np.sign(forecast) * (dollar_limits[1:] / current_prices[1:])

    # net stock dollar exposure (from integer-clipped shares), used by the ALGO net-$ gate
    stock_share_limits = (dollar_limits[1:] / current_prices[1:]).astype(int)
    stock_shares = np.clip(positions[1:], -stock_share_limits, stock_share_limits).astype(int)
    net_stock_dollars = float((stock_shares * current_prices[1:]).sum())

    # --- ALGO leg: hedge a large net stock exposure, else run the index z-score TREND/FADE ---
    algo_share_cap = dollar_limits[0] / current_prices[0]
    if ALGO_LL_DOLLAR > 0 and abs(net_stock_dollars) >= ALGO_LL_DOLLAR:
        algo_shares = float(np.sign(net_stock_dollars) * algo_share_cap)
    else:
        log_algo = log_prices[0]
        moves = log_algo[CONTRA_K:] - log_algo[:-CONTRA_K]
        z = (moves[-1] - moves[-CONTRA_WZ:].mean()) / (moves[-CONTRA_WZ:].std() + 1e-12)
        algo_z_shares = np.clip(z, -3, 3) / 3.0 * (CONTRA_DOL / current_prices[0])
        trend = (day >= WARMUP + ALGO_ROT_W + ALGO_P) and (_algo_leg_mode(log_algo, day) == "trend")
        algo_shares = float(np.clip(algo_z_shares if trend else -algo_z_shares, -algo_share_cap, algo_share_cap))

    positions[0] = algo_shares

    # The ordinary strategy is left untouched until a causal, persistent
    # inversion is observed.  When confirmed, reverse the existing stock book
    # at full size and replace the directional ALGO leg with a beta hedge.
    if ready and _inversion_confirmed(day):
        positions[1:] *= -1.0
        positions[0] = _beta_hedge_algo_shares(prices_so_far, positions)

    # Match the original SAFE submission boundary before applying the separate
    # pair sleeve: the base strategy returns clipped integer shares.
    share_limits = (dollar_limits / current_prices).astype(int)
    positions = np.clip(positions, -share_limits, share_limits).astype(int).astype(float)

    # Realise yesterday's pair-minus-SAFE counterfactual payoff.  We update it
    # even while the sleeve is disabled so a recovered edge can reactivate.
    if _PAIR_PREV_T is not None and day == _PAIR_PREV_T + 1:
        _PAIR_PAYOFFS.append(float(_PAIR_PREV_DELTA @ (current_prices - _PAIR_PREV_PRICES)))
        if len(_PAIR_PAYOFFS) > 4 * PAIR_GATE_W:
            del _PAIR_PAYOFFS[:-4 * PAIR_GATE_W]

    # Compute the pair book and record how it would differ from the base book, so the
    # counterfactual payoff above can be scored tomorrow even if the sleeve is off today.
    pair_target = _pair_positions(prices_so_far)
    active_pairs = pair_target[1:] != 0
    pair_delta = np.zeros(n_inst)
    pair_delta_stocks = pair_delta[1:]                          # view into pair_delta
    pair_delta_stocks[active_pairs] = pair_target[1:][active_pairs] - positions[1:][active_pairs]
    _PAIR_PREV_DELTA = pair_delta
    _PAIR_PREV_PRICES = current_prices.copy()
    _PAIR_PREV_T = day

    # Enable the pair sleeve until it has a track record, then only while its recent payoff is positive.
    pair_enabled = (
        len(_PAIR_PAYOFFS) < PAIR_GATE_W
        or np.mean(_PAIR_PAYOFFS[-PAIR_GATE_W:]) > 0.0
    )
    if pair_enabled:
        # Blend the pair target into the affected stocks at PAIR_WEIGHT of their capacity.
        stock_dollars = positions[1:] * current_prices[1:]
        pair_dollars = pair_target[1:] * current_prices[1:]
        stock_dollars[active_pairs] = (
            (1 - PAIR_WEIGHT) * stock_dollars[active_pairs]
            + PAIR_WEIGHT * pair_dollars[active_pairs]
        )
        positions[1:] = stock_dollars / current_prices[1:]
        positions = _recalculate_algo(prices_so_far, positions)

    final_limits = (dollar_limits / current_prices).astype(int)
    return np.clip(positions, -final_limits, final_limits).astype(int)
