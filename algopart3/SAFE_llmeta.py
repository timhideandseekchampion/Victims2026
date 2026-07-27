"""
================================================================================
###  SAFE_llmeta.py  ·  DEAD END, KEPT AS A DOCUMENTED OVERFITTING EXAMPLE     ###
================================================================================
  >>> DO NOT SHIP THIS. Confirmed overfit, not a structural improvement -- see  <<<
  >>> the verdict below. Kept in the repo so the negative result isn't lost.   <<<

WHAT IT DOES: idio book identical to SAFE.py. The ALGO leg picks WHICHEVER of
SAFE_llvol's or SAFE_lldollar's own ALGO-leg mechanism had the better trailing
META_L-day realized PnL, and runs that ONE mechanism at full size.

THE INITIAL CASE (why this looked promising): neither ALGO mechanism is stable
across the whole file -- SAFE_lldollar's fixed-direction lead-lag skew gate wins
days ~150-750 (694 OLD) then decays hard (452 NEW); SAFE_llvol's adaptive
vol/momentum switch is flat/choppy early, strengthens from ~day 500 on (684 ->
761). A PERFECT HINDSIGHT day-by-day switch is worse than shipping SAFE_llvol
alone (rolling mean 751.6 vs 759.4, floor 553.7 vs 564.9 -- day-by-day is
noise-dominated and pays heavy commission whipsawing between the two differently-
sized position schemes). Sweeping a SLOW trailing-mean switch instead (lookback
10-250 days) found what looked like a real result at META_L=30-36:
              OLD     NEW   roll_mean   roll_floor
  SAFE_llvol  683.9   761.1   759.4       564.9
  META_L=33   671.1   710.3   757.1       638.0     (+73 floor, -2 mean)

THE VERDICT, AFTER CHECKING WHERE THAT "+73 floor" ACTUALLY COMES FROM: this is
overfitting, confirmed rather than just suspected.
  - LLVOL's rolling floor traces to ONE window (days 190-440), nowhere near OLD
    or NEW -- the early stretch before its own vol-continuation edge had kicked
    in (the IC-block analysis shows day 100-300 was actually negative).
  - SAFE_lldollar was ITSELF originally discovered by hunting on days 400-750
    (see SAFE.py's own docstring) -- the "fix" for LLVOL's weak window is a
    DIFFERENT mechanism independently fitted to cover nearly that same stretch.
    META_L just tunes how fast to lean on it. That's not regime detection, it's
    borrowing a pre-fitted patch.
  - META_L=33 scores WORSE than shipped LLVOL on BOTH OLD and NEW individually
    (671.1 vs 683.9, 710.3 vs 761.1). Every parameter in this codebase that's
    actually validated (e.g. COMBINE_GAIN) earns it by improving BOTH disjoint
    sub-periods at once -- this clears neither.
  - The "improved" floor didn't fix the weak window, it moved it: days 180-430
    are still barely-worst (638.8), and a NEW worst window appears at days
    430-680 (638.0) that was never a problem for plain LLVOL. Checked the
    switch's actual choices during days 190-440: it leans LLDOLLAR 70% of those
    days (vs 50% LLVOL) -- confirms the mechanism, and that it's a patch, not a
    fix.

Three compounding layers of in-sample fitting (LLDOLLAR's own params + LLVOL's
own params + META_L on top), validated only against metrics from the one file
all three were fit to, with the "win" traced to patching one specific known-bad
window -- do not ship this. Idio book (instruments 1..49) is still byte-identical
to SAFE.py, only the ALGO-leg experiment here is retracted.

COST: recomputes each mechanism's ALGO-leg decision for every one of the trailing
META_L days from scratch on every call (no cached state, so it stays as stateless
and obviously-correct as the rest of this family). LLVOL's replay is cheap (its
ALGO leg only needs instrument 0's own price history -- reuses its private
_algo_vol_shares directly, no ridge fit). LLDOLLAR's replay is NOT cheap: its ALGO
leg reads the idio book's net-$ skew, so each of the META_L historical days needs
a full idio ridge-ensemble fit. That's roughly META_L extra ridge fits per call --
materially slower than every other strategy in this family. Fine for offline
backtesting; if this were latency-constrained, memoizing the per-day idio fit
(keyed off a validated price-history snapshot, not just the day index) would cut
this ~30x, but that isn't done here to keep the file simple and unambiguously
correct rather than fast.
================================================================================
"""
import numpy as np
import SAFE_llvol
import SAFE_lldollar

BOOK = "SAFE · LL-META (trailing-performance switch: LLVOL <-> LLDOLLAR ALGO legs)"

META_L = 33          # trailing days of realized ALGO-leg PnL used to pick the mechanism (see sweep above)
COMM0  = 2e-5        # instrument-0 commission rate (mirrors eval.py's inst0CommRate), used only for
                     # the internal trailing-PnL comparison signal, not the actual traded accounting
WARMUP = SAFE_llvol.WARMUP + META_L + 2

_DLR = None


def _limits(nInst):
    global _DLR
    if _DLR is None or len(_DLR) != nInst:
        _DLR = np.full(nInst, 10_000.0); _DLR[0] = 100_000.0
    return _DLR


def _llvol_algo_trailing_pnl_mean(prcSoFar, t, L):
    """Causal trailing mean realized PnL/day of LLVOL's OWN historical ALGO-leg decisions, for the
    L days ending just before day index t. Uses LLVOL's private _algo_vol_shares directly (cheap:
    only needs instrument 0's own price history, no idio ridge fit)."""
    lpA = np.log(prcSoFar[0])
    start = max(0, t - L - 1)
    prev_pos = 0.0; prev_comm = 0.0; pnl = []
    dlr0 = 100_000.0
    for k in range(start, t):
        cur0 = prcSoFar[0, k]
        if k > start:
            pnl.append(prev_pos * (cur0 - prcSoFar[0, k - 1]) - prev_comm)
        newPos = float(SAFE_llvol._algo_vol_shares(lpA[:k + 1], cur0, dlr0))
        prev_comm = COMM0 * abs(newPos - prev_pos) * cur0
        prev_pos = newPos
    return float(np.mean(pnl)) if pnl else 0.0


def _lldollar_algo_trailing_pnl_mean(prcSoFar, t, L):
    """Causal trailing mean realized PnL/day of LLDOLLAR's OWN historical ALGO-leg decisions, for
    the L days ending just before day index t. LLDOLLAR's ALGO leg reads the idio book's net-$ skew,
    so this replays its full getMyPosition (idio ridge fit + ALGO gate) at each historical day."""
    start = max(0, t - L - 1)
    prev_pos = 0.0; prev_comm = 0.0; pnl = []
    for k in range(start, t):
        cur0 = prcSoFar[0, k]
        if k > start:
            pnl.append(prev_pos * (cur0 - prcSoFar[0, k - 1]) - prev_comm)
        newPos = float(np.asarray(SAFE_lldollar.getMyPosition(prcSoFar[:, :k + 1]))[0])
        prev_comm = COMM0 * abs(newPos - prev_pos) * cur0
        prev_pos = newPos
    return float(np.mean(pnl)) if pnl else 0.0


def getMyPosition(prcSoFar):
    prcSoFar = np.asarray(prcSoFar, dtype=float)
    nInst, t = prcSoFar.shape
    dlr = _limits(nInst)
    cur = prcSoFar[:, -1]
    pos = np.zeros(nInst)
    if t < WARMUP:
        return pos.astype(int)

    # idio book: identical to SAFE.py across the whole family -- get it (and LLVOL's ALGO leg for
    # today) from one call, reused unless the switch below picks LLDOLLAR instead.
    full_llvol = np.asarray(SAFE_llvol.getMyPosition(prcSoFar), dtype=float)
    pos[1:] = full_llvol[1:]

    v_mean = _llvol_algo_trailing_pnl_mean(prcSoFar, t - 1, META_L)
    d_mean = _lldollar_algo_trailing_pnl_mean(prcSoFar, t - 1, META_L)

    if d_mean > v_mean:
        pos[0] = float(np.asarray(SAFE_lldollar.getMyPosition(prcSoFar))[0])
    else:
        pos[0] = full_llvol[0]

    lim = (dlr / cur).astype(int)
    return np.clip(pos, -lim, lim).astype(int)
