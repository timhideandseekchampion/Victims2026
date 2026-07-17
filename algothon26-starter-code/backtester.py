#!/usr/bin/env python
"""Algothon 2026 backtester.

A drop-in, richer alternative to eval.py. The core PnL/fee/position-limit
mechanics are a faithful copy of eval.py (including its one-day commission
lag), so the headline Score matches the official grader exactly. On top of
that it adds:

  * arbitrary test windows and walk-forward / rolling out-of-sample folds
  * extra metrics: max drawdown, turnover, hit-rate, best/worst day
  * per-instrument PnL attribution
  * equity-curve + drawdown plots (matplotlib)
  * per-day CSV export
  * benchmarking one strategy against another (or against flat/no-trade)

Everything is importable, so you can also call run_backtest(...) from a
notebook or a parameter-sweep script.

Usage examples:
    python backtester.py                         # score last 250 days of teamName
    python backtester.py --strategy myAlgo       # score myAlgo.py
    python backtester.py --days 100              # score last 100 days only
    python backtester.py --walk-forward 5        # 5 sequential out-of-sample folds
    python backtester.py --rolling 100           # rolling 100-day windows across history
    python backtester.py --plot equity.png       # save equity/drawdown chart
    python backtester.py --attribution           # per-instrument PnL breakdown
    python backtester.py --compare flat          # A/B against a no-trade baseline
    python backtester.py --csv days.csv --quiet  # dump per-day series, no chatter
"""
from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Grading constants (must match eval.py / the sandbox exactly)
# ----------------------------------------------------------------------------
SCORE_DEFAULT_PARAM = 1.0

DEFAULT_COMM_RATE = 0.0001      # 1 bp
INST0_COMM_RATE = 0.00002       # special rate for instrument 0

DEFAULT_DLR_POS_LIMIT = 10_000  # dollars
INST0_DLR_POS_LIMIT = 100_000   # special limit for instrument 0

ANNUALISATION = 250             # trading days per year used by the score


def load_prices(fn):
    """Load prices (one instrument per column) -> array shape (nInst, nDays)."""
    df = pd.read_csv(fn, sep=r"\s+", header=0, index_col=None)
    names = list(df.columns)
    return df.values.T, names


def make_grading_params(n_inst):
    """Per-instrument commission rates and dollar position limits."""
    comm_rate = np.full(n_inst, DEFAULT_COMM_RATE)
    comm_rate[0] = INST0_COMM_RATE
    dlr_pos_limit = np.full(n_inst, DEFAULT_DLR_POS_LIMIT)
    dlr_pos_limit[0] = INST0_DLR_POS_LIMIT
    return comm_rate, dlr_pos_limit


def score(mu, sigma, param=SCORE_DEFAULT_PARAM):
    """Official score from daily-PnL mean & std (identical to eval.py)."""
    if mu <= 0 or sigma < 1e-10:
        return mu
    sr = np.sqrt(ANNUALISATION) * mu / sigma
    frac = sr ** 2 / (sr ** 2 + param ** 2)
    return mu * frac


# ----------------------------------------------------------------------------
# Result container
# ----------------------------------------------------------------------------
@dataclass
class BacktestResult:
    # per-scored-day series (length = num_test_days)
    days: np.ndarray                # absolute day index of each scored day
    pnl: np.ndarray                 # daily PnL ($)
    value: np.ndarray               # portfolio value ($) after each day
    dvolume: np.ndarray             # dollar volume traded each day
    positions: np.ndarray           # shape (num_test_days, nInst) held positions
    inst_pnl: np.ndarray            # shape (num_test_days, nInst) per-instrument PnL
    # scalars
    mean_pl: float = 0.0
    std_pl: float = 0.0
    ann_sharpe: float = 0.0
    score: float = 0.0
    tot_dvolume: float = 0.0
    final_return: float = 0.0
    inst_names: list = field(default_factory=list)
    gross: np.ndarray = field(default_factory=lambda: np.array([]))  # gross $ exposure/day
    net: np.ndarray = field(default_factory=lambda: np.array([]))    # net $ exposure/day (long-short)

    # -- derived analytics -------------------------------------------------
    @property
    def cum_pnl(self):
        return np.cumsum(self.pnl)

    @property
    def avg_gross(self):
        """Average gross dollar exposure (Σ|position·price|) — how much capital is deployed."""
        return float(np.mean(self.gross)) if self.gross.size else 0.0

    @property
    def avg_net(self):
        """Average net dollar exposure (Σ position·price) — ~0 means market-neutral."""
        return float(np.mean(self.net)) if self.net.size else 0.0

    @property
    def max_drawdown(self):
        """Largest peak-to-trough drop of the cumulative-PnL curve ($)."""
        curve = self.cum_pnl
        if curve.size == 0:
            return 0.0
        running_peak = np.maximum.accumulate(curve)
        return float(np.max(running_peak - curve))

    @property
    def hit_rate(self):
        return float(np.mean(self.pnl > 0)) if self.pnl.size else 0.0

    @property
    def avg_daily_turnover(self):
        return float(np.mean(self.dvolume)) if self.dvolume.size else 0.0

    @property
    def best_day(self):
        return (int(self.days[np.argmax(self.pnl)]), float(np.max(self.pnl))) if self.pnl.size else (0, 0.0)

    @property
    def worst_day(self):
        return (int(self.days[np.argmin(self.pnl)]), float(np.min(self.pnl))) if self.pnl.size else (0, 0.0)

    # -- risk / reward metrics --------------------------------------------
    # All are computed on the daily-PnL series (dollars): the competition has
    # no fixed capital base, so a "% return" is ill-defined — the grader's own
    # Sharpe is on dollar PnL, and these stay consistent with it.
    @property
    def ann_return(self):
        """Expected annualised PnL ($): mean daily PnL x 250 trading days."""
        return float(self.mean_pl * ANNUALISATION)

    @property
    def downside_dev(self):
        """Downside deviation of daily PnL vs a 0 target (annualised)."""
        if self.pnl.size == 0:
            return 0.0
        neg = np.minimum(self.pnl, 0.0)
        return float(np.sqrt(np.mean(neg ** 2)))

    @property
    def sortino(self):
        dd = self.downside_dev
        return float(np.sqrt(ANNUALISATION) * self.mean_pl / dd) if dd > 1e-12 else 0.0

    @property
    def calmar(self):
        """Annualised PnL over max drawdown — return per unit of worst pain."""
        mdd = self.max_drawdown
        return float(self.ann_return / mdd) if mdd > 1e-12 else 0.0

    @property
    def var95(self):
        """Daily PnL 5th-percentile (a bad-but-not-worst day), a loss (< 0)."""
        return float(np.percentile(self.pnl, 5)) if self.pnl.size else 0.0

    @property
    def cvar95(self):
        """Expected PnL on the worst 5% of days (conditional VaR)."""
        if self.pnl.size == 0:
            return 0.0
        tail = self.pnl[self.pnl <= self.var95]
        return float(np.mean(tail)) if tail.size else 0.0

    @property
    def profit_factor(self):
        """Gross gains / gross losses. >1 makes money; >1.5 is healthy."""
        gains = self.pnl[self.pnl > 0].sum()
        losses = -self.pnl[self.pnl < 0].sum()
        return float(gains / losses) if losses > 1e-12 else float("inf")

    @property
    def skew(self):
        """Skew of daily PnL (>0 = right tail / occasional big wins)."""
        if self.pnl.size < 2:
            return 0.0
        x = self.pnl - self.pnl.mean()
        s = x.std()
        return float(np.mean((x / s) ** 3)) if s > 1e-12 else 0.0

    @property
    def kurtosis(self):
        """Excess kurtosis of daily PnL (>0 = fat tails vs normal)."""
        if self.pnl.size < 2:
            return 0.0
        x = self.pnl - self.pnl.mean()
        s = x.std()
        return float(np.mean((x / s) ** 4) - 3) if s > 1e-12 else 0.0

    def pnl_percentiles(self, ps=(5, 25, 50, 75, 95)):
        """Daily-PnL distribution percentiles ($)."""
        return {p: float(np.percentile(self.pnl, p)) for p in ps} if self.pnl.size else {p: 0.0 for p in ps}

    def full_metrics(self):
        """Ordered dict of every headline metric, for --stats and tables."""
        bd, bv = self.best_day
        wd, wv = self.worst_day
        return {
            "Score (official)": round(self.score, 3),
            "Mean daily PnL ($)": round(self.mean_pl, 2),
            "Expected annual PnL ($)": round(self.ann_return, 0),
            "Daily PnL std ($)": round(self.std_pl, 2),
            "Annualised Sharpe": round(self.ann_sharpe, 3),
            "Sortino": round(self.sortino, 3),
            "Calmar": round(self.calmar, 3),
            "Max drawdown ($)": round(self.max_drawdown, 2),
            "Hit rate": f"{self.hit_rate*100:.1f}%",
            "Profit factor": round(self.profit_factor, 3),
            "VaR 95% (daily $)": round(self.var95, 2),
            "CVaR 95% (daily $)": round(self.cvar95, 2),
            "Best day ($)": f"{bv:.2f} (day {bd})",
            "Worst day ($)": f"{wv:.2f} (day {wd})",
            "Daily PnL skew": round(self.skew, 2),
            "Daily PnL kurtosis (excess)": round(self.kurtosis, 2),
            "Avg daily turnover ($)": round(self.avg_daily_turnover, 0),
            "Avg gross exposure ($)": round(self.avg_gross, 0),
            "Avg net exposure ($)": round(self.avg_net, 0),
            "Total $ volume": round(self.tot_dvolume, 0),
        }

    def instrument_attribution(self):
        """Return a DataFrame of per-instrument total PnL / turnover, sorted."""
        total = self.inst_pnl.sum(axis=0)
        turnover = np.abs(np.diff(self.positions, axis=0, prepend=0)).sum(axis=0)
        df = pd.DataFrame({
            "instrument": self.inst_names,
            "pnl": total,
            "share_days_held": np.mean(self.positions != 0, axis=0),
            "trades": turnover,
        })
        return df.sort_values("pnl", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------
# Core engine (faithful to eval.py, generalised over window)
# ----------------------------------------------------------------------------
def run_backtest(prc_hist, get_position, num_test_days=250, last_day=None,
                 comm_rate=None, dlr_pos_limit=None, inst_names=None):
    """Run the exact eval.py PnL loop over a chosen window.

    num_test_days scored days end on `last_day` (default: end of data). The
    day before the first scored day is a warm-up day used only to establish
    the starting position, exactly as eval.py does with startDay.
    """
    n_inst, nt = prc_hist.shape
    if last_day is None:
        last_day = nt
    if comm_rate is None or dlr_pos_limit is None:
        comm_rate, dlr_pos_limit = make_grading_params(n_inst)
    if inst_names is None:
        inst_names = [f"inst{i}" for i in range(n_inst)]

    start_day = last_day - num_test_days
    if start_day < 1:
        raise ValueError(
            f"Not enough history: need >{num_test_days} days before day {last_day}, "
            f"have {nt}."
        )

    cash = 0.0
    cur_pos = np.zeros(n_inst)
    tot_dvolume = 0.0
    value = 0.0
    comm = 0.0                       # scalar commission from yesterday's trade (one-day lag)
    prev_comm_vec = np.zeros(n_inst) # same, per instrument, for exact attribution

    days, pnl_l, value_l, dvol_l, pos_l, inst_pnl_l = [], [], [], [], [], []
    gross_l, net_l = [], []
    ret = 0.0

    for t in range(start_day, last_day + 1):
        prc_so_far = prc_hist[:, :t]
        cur_prices = prc_so_far[:, -1]
        prev_prices = prc_so_far[:, -2] if t >= 2 else cur_prices
        prev_pos = cur_pos.copy()

        if t < last_day:
            new_pos_orig = np.asarray(get_position(prc_so_far))
            pos_limits = (dlr_pos_limit / cur_prices).astype(int)
            new_pos = np.clip(new_pos_orig, -pos_limits, pos_limits).astype(int)
        else:
            # final day only marks PnL; no new trade
            new_pos = np.array(cur_pos)

        delta_pos = new_pos - cur_pos
        comm_prev = comm  # commission charged today is from yesterday's trade

        cash -= cur_prices.dot(delta_pos) + comm_prev

        dvolumes = cur_prices * np.abs(delta_pos)
        dvolume = np.sum(dvolumes)
        tot_dvolume += dvolume
        comm = float(np.sum(dvolumes * comm_rate))  # this trade's commission (paid tomorrow)

        cur_pos = np.array(new_pos)
        pos_value = cur_pos.dot(cur_prices)
        today_pl = cash + pos_value - value
        value = cash + pos_value

        # exact per-instrument attribution: today_pl == sum_i inst_pnl_i where
        #   inst_pnl_i = prev_pos_i*(price_t - price_{t-1}) - comm_prev_i
        inst_pnl = prev_pos * (cur_prices - prev_prices) - prev_comm_vec
        prev_comm_vec = dvolumes * comm_rate  # charged tomorrow

        if tot_dvolume > 0:
            ret = value / tot_dvolume

        if t > start_day:  # skip warm-up day, exactly like eval.py
            days.append(t)
            pnl_l.append(today_pl)
            value_l.append(value)
            dvol_l.append(dvolume)
            pos_l.append(cur_pos.copy())
            inst_pnl_l.append(inst_pnl.copy())
            net_l.append(float(pos_value))                       # Σ position·price
            gross_l.append(float(np.abs(cur_pos).dot(cur_prices)))  # Σ |position·price|

    pnl = np.array(pnl_l)
    plmu, plstd = (float(np.mean(pnl)), float(np.std(pnl))) if pnl.size else (0.0, 0.0)
    ann_sharpe = np.sqrt(ANNUALISATION) * plmu / plstd if plstd > 0 else 0.0

    return BacktestResult(
        days=np.array(days),
        pnl=pnl,
        value=np.array(value_l),
        dvolume=np.array(dvol_l),
        positions=np.array(pos_l) if pos_l else np.zeros((0, n_inst)),
        inst_pnl=np.array(inst_pnl_l) if inst_pnl_l else np.zeros((0, n_inst)),
        mean_pl=plmu,
        std_pl=plstd,
        ann_sharpe=float(ann_sharpe),
        score=float(score(plmu, plstd)),
        tot_dvolume=float(tot_dvolume),
        final_return=float(ret),
        inst_names=list(inst_names),
        gross=np.array(gross_l),
        net=np.array(net_l),
    )


# ----------------------------------------------------------------------------
# Strategy loading
# ----------------------------------------------------------------------------
def load_strategy(name):
    """Return a FRESH getMyPosition callable from a module name, or a baseline.

    The module is reloaded so any module-level state (e.g. the starter's
    accumulating `currentPos` global) is reset. This keeps each independent
    backtest run (fold, rolling window, comparison) starting from a clean
    state, exactly as the official grader does on a single run.
    """
    if name == "flat":  # no-trade baseline
        return lambda prc: np.zeros(prc.shape[0])
    if name == "random":
        rng = np.random.default_rng(0)
        return lambda prc: rng.integers(-100, 100, size=prc.shape[0])
    mod = importlib.import_module(name)
    importlib.reload(mod)  # reset module-level globals between independent runs
    if not hasattr(mod, "getMyPosition"):
        sys.exit(f"error: module '{name}' has no getMyPosition(prcSoFar)")
    return mod.getMyPosition


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
def print_summary(res, label=""):
    head = f"===== {label} =====" if label else "====="
    print(head)
    print(f"mean(PL):        {res.mean_pl:.2f}   (expected annual PnL ${res.ann_return:,.0f})")
    print(f"StdDev(PL):      {res.std_pl:.2f}")
    print(f"annSharpe(PL):   {res.ann_sharpe:.2f}")
    print(f"Sortino:         {res.sortino:.2f}")
    print(f"Calmar:          {res.calmar:.2f}")
    print(f"return:          {res.final_return:.5f}")
    print(f"totDvolume:      {res.tot_dvolume:.0f}")
    print(f"avg turnover:    {res.avg_daily_turnover:.0f}/day")
    print(f"avg gross expo:  {res.avg_gross:,.0f}   avg net expo: {res.avg_net:,.0f}  (net~0 = market-neutral)")
    print(f"maxDrawdown:     {res.max_drawdown:.2f}")
    print(f"hit rate:        {res.hit_rate*100:.1f}%")
    bd, bv = res.best_day
    wd, wv = res.worst_day
    print(f"best day:        day {bd}  ${bv:.2f}")
    print(f"worst day:       day {wd}  ${wv:.2f}")
    print(f"Score:           {res.score:.2f}")


def print_stats(res, label=""):
    """Full metrics table (--stats)."""
    print(f"\n--- full metrics{' · ' + label if label else ''} ---")
    for k, v in res.full_metrics().items():
        print(f"  {k:<26} {v}")


# ----------------------------------------------------------------------------
# Auto-notes: plain-language reading of a result. Same thresholds power the
# dashboard's "Notes" boxes, so both surfaces say the same thing.
# ----------------------------------------------------------------------------
INSIGHT_RULES = {
    "turnover_high": 60_000,   # $/day above which fee drag is material here
    "sharpe_weak": 0.5,
    "sharpe_strong": 1.5,
    "score_weak": 5.0,
    "profit_factor_thin": 1.1,
    "calmar_weak": 0.5,
}


def insights(res):
    """Return a list of short flagged observations about a BacktestResult."""
    R = INSIGHT_RULES
    out = []
    if res.score <= 0:
        out.append("✗ not profitable (Score 0) — mean PnL ≤ 0; nothing to size.")
    elif res.score < R["score_weak"]:
        out.append(f"△ thin Score ({res.score:.1f}) — barely above the flat line.")
    if res.ann_sharpe < R["sharpe_weak"]:
        out.append(f"△ weak Sharpe ({res.ann_sharpe:.2f}) — noisy; Score punishes low Sharpe hard.")
    elif res.ann_sharpe >= R["sharpe_strong"]:
        out.append(f"✓ strong Sharpe ({res.ann_sharpe:.2f}).")
    if res.avg_daily_turnover > R["turnover_high"]:
        out.append(f"⚠ high turnover ({res.avg_daily_turnover:,.0f}/day) — fee drag is material; "
                   f"check whether a slower/vol-scaled version keeps more net Score.")
    if 0 < res.profit_factor < R["profit_factor_thin"]:
        out.append(f"△ thin profit factor ({res.profit_factor:.2f}) — gains barely exceed losses.")
    if res.score > 0 and 0 < res.calmar < R["calmar_weak"]:
        out.append(f"⚠ low Calmar ({res.calmar:.2f}) — drawdown large vs return; consider gentler sizing.")
    # Score mechanics reminder when PnL is decent but Sharpe caps the Score
    if res.mean_pl > 0 and res.ann_sharpe < 1.0:
        out.append("• Score = mean·SR²/(SR²+1): raising Sharpe converts more PnL into Score.")
    if not out:
        out.append("✓ nothing alarming — profitable, reasonable Sharpe and turnover.")
    return out


def print_insights(res, label=""):
    print(f"\n--- notes{' · ' + label if label else ''} ---")
    for line in insights(res):
        print(f"  {line}")


# ----------------------------------------------------------------------------
# Monte Carlo — how much of the result is skill vs luck. Bootstrap-resample the
# daily PnL and look at the spread of Score / Sharpe / total PnL / max drawdown.
# ----------------------------------------------------------------------------
def montecarlo(res, n=2000, seed=0):
    pnl = res.pnl
    if pnl.size == 0:
        return {}
    rng = np.random.default_rng(seed)
    T = len(pnl)
    sc = np.empty(n); sh = np.empty(n); tot = np.empty(n); dd = np.empty(n)
    for i in range(n):
        s = rng.choice(pnl, size=T, replace=True)
        mu, sd = s.mean(), s.std()
        sc[i] = score(mu, sd)
        sh[i] = np.sqrt(ANNUALISATION) * mu / sd if sd > 1e-12 else 0.0
        tot[i] = s.sum()
        curve = np.cumsum(s); dd[i] = np.max(np.maximum.accumulate(curve) - curve)
    band = lambda a: (float(np.percentile(a, 5)), float(np.median(a)), float(np.percentile(a, 95)))
    return {"n": n, "score": band(sc), "sharpe": band(sh), "total_pnl": band(tot),
            "max_dd": band(dd), "prob_profit": float(np.mean(tot > 0))}


def print_montecarlo(res, n=2000):
    mc = montecarlo(res, n)
    if not mc:
        print("\n--- monte carlo: no PnL ---")
        return
    print(f"\n--- monte carlo ({mc['n']} bootstrap resamples of daily PnL) ---")
    print(f"{'metric':<14}{'5%':>12}{'median':>12}{'95%':>12}{'  (actual)':>14}")
    rows = [("Score", "score", res.score), ("Sharpe", "sharpe", res.ann_sharpe),
            ("Total PnL", "total_pnl", res.pnl.sum()), ("Max drawdown", "max_dd", res.max_drawdown)]
    for lab, key, act in rows:
        lo, md, hi = mc[key]
        print(f"{lab:<14}{lo:>12.2f}{md:>12.2f}{hi:>12.2f}{act:>14.2f}")
    print(f"P(profitable resample): {mc['prob_profit']*100:.1f}%   "
          f"→ {'robust edge' if mc['prob_profit'] > 0.9 else 'fragile — much of the result could be luck' if mc['prob_profit'] < 0.75 else 'moderate confidence'}")


def print_attribution(res, top=10):
    df = res.instrument_attribution()
    print("\n--- per-instrument PnL attribution ---")
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print("top winners:")
        print(df.head(top).to_string(index=False))
        print("\ntop losers:")
        print(df.tail(top).iloc[::-1].to_string(index=False))
    print(f"\nsum of instrument PnL: {df['pnl'].sum():.2f}  "
          f"(total PL: {res.pnl.sum():.2f})")


def walk_forward(prc_hist, strategy, k, num_test_days, comm_rate, dlr_pos_limit, names):
    """Split the last num_test_days into k sequential out-of-sample folds.

    Each fold reloads the strategy so folds are independent (fresh state),
    mirroring a clean grader run on that window.
    """
    n_inst, nt = prc_hist.shape
    fold = num_test_days // k
    print(f"\n--- walk-forward: {k} folds of ~{fold} days ---")
    rows = []
    for i in range(k):
        last = nt - (k - 1 - i) * fold
        # a built callable is stateless → reuse; a module name is reloaded per fold
        gp = strategy if callable(strategy) else load_strategy(strategy)
        res = run_backtest(prc_hist, gp, num_test_days=fold, last_day=last,
                           comm_rate=comm_rate, dlr_pos_limit=dlr_pos_limit, inst_names=names)
        rows.append({
            "fold": i + 1, "days": f"{last-fold+1}-{last}",
            "mean_pl": res.mean_pl, "sharpe": res.ann_sharpe, "score": res.score,
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print(f"mean score across folds: {df['score'].mean():.2f}   "
          f"(std {df['score'].std():.2f})")
    return df


def rolling(prc_hist, strategy, window, num_test_days, comm_rate, dlr_pos_limit, names, step=None):
    """Slide a `window`-day test across all available history (fresh state each)."""
    n_inst, nt = prc_hist.shape
    step = step or max(window // 4, 1)
    print(f"\n--- rolling {window}-day windows (step {step}) ---")
    rows = []
    last = nt
    while last - window >= 1:
        gp = strategy if callable(strategy) else load_strategy(strategy)
        res = run_backtest(prc_hist, gp, num_test_days=window, last_day=last,
                           comm_rate=comm_rate, dlr_pos_limit=dlr_pos_limit, inst_names=names)
        rows.append({"end_day": last, "mean_pl": res.mean_pl,
                     "sharpe": res.ann_sharpe, "score": res.score})
        last -= step
    df = pd.DataFrame(rows[::-1])
    print(df.to_string(index=False))
    print(f"score  mean {df['score'].mean():.2f}  min {df['score'].min():.2f}  "
          f"max {df['score'].max():.2f}  %positive {100*np.mean(df['score']>0):.0f}%")
    return df


def make_plot(results, out_path):
    """results: list of (label, BacktestResult). Saves equity + drawdown + exposure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 9), sharex=True,
                                        gridspec_kw={"height_ratios": [3, 1, 1.4]})
    for label, res in results:
        ax1.plot(res.days, res.cum_pnl, label=f"{label} (score {res.score:.2f})")
        curve = res.cum_pnl
        dd = curve - np.maximum.accumulate(curve)
        ax2.fill_between(res.days, dd, 0, alpha=0.3)
        if res.gross.size:
            ax3.plot(res.days, res.gross, label=f"{label} gross")
            ax3.plot(res.days, res.net, lw=0.9, alpha=0.8, label=f"{label} net")
    ax1.set_title("Cumulative PnL"); ax1.set_ylabel("cum PnL ($)"); ax1.axhline(0, color="k", lw=0.7); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.set_title("Drawdown"); ax2.set_ylabel("drawdown ($)"); ax2.grid(alpha=0.3)
    ax3.set_title("Dollar exposure — gross (capital deployed) & net (long−short)")
    ax3.set_ylabel("$ exposure"); ax3.set_xlabel("day"); ax3.axhline(0, color="k", lw=0.7); ax3.legend(fontsize=8); ax3.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"\nsaved plot -> {out_path}")


def dump_csv(res, out_path):
    df = pd.DataFrame({
        "day": res.days, "pnl": res.pnl, "cum_pnl": res.cum_pnl,
        "value": res.value, "dvolume": res.dvolume,
        "gross_exposure": res.gross, "net_exposure": res.net,
    })
    df.to_csv(out_path, index=False)
    print(f"saved per-day series -> {out_path}")


def plot_entries(res, prc_hist, inst, names, out_path):
    """Plot one instrument's price over the test window with entry/exit markers
    (position sign-changes): ▲ go long, ▼ go short, · go flat. Also its position size."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    idx = names.index(inst) if (isinstance(inst, str) and inst in names) else int(inst)
    days = res.days
    price = np.array([prc_hist[idx, d - 1] for d in days])
    pos = res.positions[:, idx]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(days, price, color="#2a78d6", lw=1.4, label=names[idx])
    for i in range(1, len(pos)):
        a, b = np.sign(pos[i - 1]), np.sign(pos[i])
        if b == a:
            continue
        if b > 0:
            ax1.scatter(days[i], price[i], marker="^", color="#0ca30c", s=45, zorder=3)
        elif b < 0:
            ax1.scatter(days[i], price[i], marker="v", color="#d03b3b", s=45, zorder=3)
        else:
            ax1.scatter(days[i], price[i], marker="o", color="#898781", s=18, zorder=3)
    ax1.set_title(f"Entries (▲ long) / exits (▼ short, · flat) — {names[idx]}")
    ax1.set_ylabel("price"); ax1.grid(alpha=0.3); ax1.legend()
    ax2.fill_between(days, pos * price, 0, color="#2a78d6", alpha=0.25)
    ax2.axhline(0, color="k", lw=0.7)
    ax2.set_title("Position ($)"); ax2.set_ylabel("$"); ax2.set_xlabel("day"); ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=120)
    print(f"saved entry/exit plot -> {out_path}  ({names[idx]})")


def export_runs(prc_hist, comm_rate, dlr_pos_limit, names, out_path, num_test_days=250):
    """Run a curated set of strategy configs through the real (fee-aware) engine
    and dump each one's equity curve + metrics to JSON, for the dashboard's
    Compare tab. This is the honest way to visualise configs — the dashboard's
    own portfolio panel is only a gross equal-weight approximation."""
    import json
    import strategy as st
    blend = st.alpha_rev_blend
    mk = st.make_get_position
    configs = [
        ("baseline rev_z20", mk(signal_fn=st.zrev(20))),
        ("rev_blend", mk(signal_fn=blend)),
        ("blend + inverse_vol", mk(signal_fn=blend, sizing="inverse_vol")),
        ("blend + regime-gate", st.regime_gate(mk(signal_fn=blend))),
        ("blend + invvol + regime", st.regime_gate(mk(signal_fn=blend, sizing="inverse_vol"))),
        ("blend + beta-neutral", st.beta_neutralize(mk(signal_fn=blend))),
        ("flat (no trades)", lambda p: np.zeros(p.shape[0])),
    ]
    runs = []
    for label, gp in configs:
        r = run_backtest(prc_hist, gp, num_test_days=num_test_days,
                         comm_rate=comm_rate, dlr_pos_limit=dlr_pos_limit, inst_names=names)
        runs.append({
            "label": label,
            "days": [int(d) for d in r.days],
            "cum_pnl": [round(float(x), 2) for x in r.cum_pnl],
            "metrics": {"Score": round(r.score, 2), "Sharpe": round(r.ann_sharpe, 2),
                        "Sortino": round(r.sortino, 2), "Calmar": round(r.calmar, 2),
                        "maxDD": round(r.max_drawdown, 0), "mean_pl": round(r.mean_pl, 1),
                        "turnover": round(r.avg_daily_turnover, 0)},
        })
        print(f"  ran: {label:<26} Score {r.score:.1f}")
    with open(out_path, "w") as f:
        json.dump({"runs": runs}, f, separators=(",", ":"))
    print(f"saved {len(runs)} runs -> {out_path}  "
          f"(feed to: python dashboard.py --runs {out_path})")


def export_positions(res, out_path):
    """Per-instrument position matrix (day + one column per instrument).

    This is what dashboard.py --positions consumes to draw real entries/exits.
    """
    df = pd.DataFrame(res.positions, columns=res.inst_names)
    df.insert(0, "day", res.days)
    df.to_csv(out_path, index=False)
    print(f"saved position matrix -> {out_path}  "
          f"(feed to: python dashboard.py --positions {out_path})")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description="Algothon 2026 backtester")
    p.add_argument("--strategy", default="teamName",
                   help="module with getMyPosition (default: teamName); "
                        "also 'flat' or 'random' baselines")
    p.add_argument("--prices", default="./prices.txt", help="price file")
    p.add_argument("--days", type=int, default=None,
                   help="test days to score (default: all available minus a 50-day "
                        "warm-up, e.g. 450 of 500). NOTE eval.py grades on a fixed 250.")
    p.add_argument("--walk-forward", type=int, metavar="K",
                   help="split test window into K sequential out-of-sample folds")
    p.add_argument("--rolling", type=int, metavar="WINDOW",
                   help="slide a WINDOW-day test across all history")
    p.add_argument("--compare", metavar="STRATEGY",
                   help="also run this strategy (module/flat/random) side by side")
    p.add_argument("--combine", metavar="MODS",
                   help="comma-separated strategy modules to combine into ONE book "
                        "(position-level weighted sum, then clipped to limits)")
    p.add_argument("--weights", metavar="W",
                   help="comma-separated weights for --combine (default: equal)")
    # --- build a strategy from parts (no file editing needed) ---
    p.add_argument("--signal", metavar="NAME",
                   help="use a named signal from strategy.SIGNALS (e.g. rev_blend, rev_z20, xs_rev5)")
    p.add_argument("--sizing", choices=["fraction", "inverse_vol", "kelly"],
                   help="position sizing (default: strategy's SIZING)")
    p.add_argument("--scale", type=float, help="aggression (lower = larger positions)")
    p.add_argument("--smooth", type=int, help="EMA span for signal smoothing")
    p.add_argument("--hold", type=int, help="rebalance every N days")
    p.add_argument("--band", type=float, help="no-trade band (fraction of max shares)")
    # --- the two-leg book (the submission): ALGO index reversion + idio reversion ---
    p.add_argument("--two-leg", action="store_true",
                   help="run the two-leg book: ALGO index reversion ($100k) + 50-name "
                        "cross-sectional reversion, both near the dollar limits")
    p.add_argument("--idio-w", type=int, help="two-leg: idio reversion window (default 10)")
    p.add_argument("--algo-w", type=int, help="two-leg: ALGO index reversion window (default 5)")
    p.add_argument("--idio-scale", type=float, help="two-leg: idio aggression, lower=bigger (default 0.10)")
    p.add_argument("--algo-scale", type=float, help="two-leg: index aggression, lower=bigger (default 0.10)")
    p.add_argument("--algo-frac", type=float,
                   help="two-leg: fraction of ALGO's $100k limit to deploy (default 1.0)")
    p.add_argument("--idio-sizing", choices=["fraction", "inverse_vol"],
                   help="two-leg: idio leg sizing (default fraction)")
    p.add_argument("--regime-gate", action="store_true",
                   help="de-risk in ALGO's high-variance regime (GMM)")
    p.add_argument("--beta-neutral", action="store_true",
                   help="hedge the book's net beta exposure to ALGO (the index)")
    p.add_argument("--ev-gate", nargs="?", type=float, const=0.6, metavar="KEEP",
                   help="only trade the top KEEP fraction by conviction (default 0.6) — skip fee-losing marginal names")
    p.add_argument("--confidence", action="store_true",
                   help="scale the book by the signal's recent realised IC (size up when it's working)")
    p.add_argument("--regime-scale", action="store_true",
                   help="continuous regime de-risking (scale by GMM calm-probability)")
    p.add_argument("--markov-gate", action="store_true",
                   help="regime de-risking via a Markov-switching model (statsmodels; slow)")
    p.add_argument("--attribution", action="store_true",
                   help="show per-instrument PnL breakdown")
    p.add_argument("--stats", action="store_true",
                   help="print the full risk/reward metrics table")
    p.add_argument("--montecarlo", nargs="?", type=int, const=2000, metavar="N",
                   help="bootstrap-resample daily PnL N times (default 2000) for "
                        "Score/Sharpe/drawdown confidence bands (skill vs luck)")
    p.add_argument("--plot", nargs="?", const="equity.png", metavar="FILE",
                   help="save equity/drawdown/exposure chart (default equity.png)")
    p.add_argument("--plot-entries", nargs="?", const="0", metavar="INST",
                   help="save an entry/exit + position chart for one instrument (name or index, default 0=ALGO)")
    p.add_argument("--csv", metavar="FILE", help="dump per-day series to CSV")
    p.add_argument("--export-positions", metavar="FILE",
                   help="dump per-instrument position matrix for dashboard.py")
    p.add_argument("--export-runs", metavar="FILE",
                   help="run a set of configs and dump their equity curves + metrics "
                        "for the dashboard Compare tab (JSON)")
    p.add_argument("--quiet", action="store_true", help="suppress per-day lines")
    args = p.parse_args(argv)

    prc_hist, names = load_prices(args.prices)
    n_inst, nt = prc_hist.shape
    comm_rate, dlr_pos_limit = make_grading_params(n_inst)
    if args.days is None:                    # default: score all available minus a 50-day warm-up
        args.days = nt - 50
    print(f"Loaded {n_inst} instruments for {nt} days from {args.prices}  (scoring {args.days} days)")

    build_parts = any(x is not None for x in (args.signal, args.sizing, args.scale,
                      args.smooth, args.hold, args.band, args.ev_gate)) \
        or args.regime_gate or args.beta_neutral or args.confidence or args.regime_scale or args.markov_gate
    wf_target = args.strategy  # what walk-forward/rolling run: module name or callable

    if args.combine:
        mods = [m.strip() for m in args.combine.split(",")]
        ws = ([float(x) for x in args.weights.split(",")] if args.weights
              else [1.0 / len(mods)] * len(mods))
        if len(ws) != len(mods):
            sys.exit("error: --weights count must match --combine module count")
        fns = [load_strategy(m) for m in mods]

        def get_position(prc, fns=fns, ws=ws):
            total = None
            for fn, w in zip(fns, ws):
                p = w * np.asarray(fn(prc), dtype=float)
                total = p if total is None else total + p
            return total  # run_backtest clips to limits + casts int
        args.strategy = "combine(" + "+".join(f"{w:g}*{m}" for m, w in zip(mods, ws)) + ")"
        wf_target = get_position
    elif args.two_leg:
        import strategy as st
        kw = {}
        for name, v in [("idio_w", args.idio_w), ("algo_w", args.algo_w),
                        ("idio_scale", args.idio_scale), ("algo_scale", args.algo_scale),
                        ("algo_frac", args.algo_frac), ("idio_sizing", args.idio_sizing)]:
            if v is not None:
                kw[name] = v
        gp = st.make_two_leg(**kw)
        if args.beta_neutral:
            gp = st.beta_neutralize(gp)
        if args.regime_gate:
            gp = st.regime_gate(gp)
        elif args.regime_scale:
            gp = st.regime_scale(gp)
        elif args.markov_gate:
            gp = st.markov_gate(gp)
        get_position = gp
        wf_target = gp
        lbl = "two_leg(" + ",".join(f"{k}={v}" for k, v in kw.items()) + ")" if kw else "two_leg"
        for flag, tag in [(args.beta_neutral, "beta-neutral"), (args.regime_gate, "regime-gate"),
                          (args.regime_scale, "regime-scale"), (args.markov_gate, "markov-gate")]:
            if flag:
                lbl += f" + {tag}"
        args.strategy = lbl
    elif build_parts:
        import strategy as st
        if args.signal and args.signal not in st.SIGNALS:
            sys.exit(f"error: --signal '{args.signal}' not in strategy.SIGNALS: {list(st.SIGNALS)}")
        base_sig = st.SIGNALS[args.signal] if args.signal else st.alpha
        sig = st.cost_gate(base_sig, args.ev_gate) if args.ev_gate is not None else base_sig
        gp = st.make_get_position(signal_fn=sig, sizing=args.sizing, scale=args.scale,
                                  smooth=args.smooth, hold=args.hold, band=args.band)
        if args.beta_neutral:
            gp = st.beta_neutralize(gp)
        if args.confidence:
            gp = st.confidence_scale(gp, base_sig)
        if args.regime_gate:
            gp = st.regime_gate(gp)
        elif args.regime_scale:
            gp = st.regime_scale(gp)
        elif args.markov_gate:
            gp = st.markov_gate(gp)
        get_position = gp
        wf_target = gp
        parts = [args.signal or "alpha"]
        for lbl, v in [("sizing", args.sizing), ("scale", args.scale), ("smooth", args.smooth),
                       ("hold", args.hold), ("band", args.band), ("ev-gate", args.ev_gate)]:
            if v is not None:
                parts.append(f"{lbl}={v}")
        if args.beta_neutral:
            parts.append("beta-neutral")
        if args.confidence:
            parts.append("confidence")
        if args.regime_gate:
            parts.append("regime-gate")
        if args.regime_scale:
            parts.append("regime-scale")
        if args.markov_gate:
            parts.append("markov-gate")
        args.strategy = " + ".join(parts)
    else:
        get_position = load_strategy(args.strategy)

    try:
        res = run_backtest(prc_hist, get_position, num_test_days=args.days,
                           comm_rate=comm_rate, dlr_pos_limit=dlr_pos_limit, inst_names=names)
    except ValueError as e:
        sys.exit(f"error: {e}")

    if not args.quiet:
        for d, v, pl, dv in zip(res.days, res.value, res.pnl, res.dvolume):
            print(f"Day {d} value: {v:.2f} todayPL: ${pl:.2f} $-traded: {dv:.0f}")

    print_summary(res, label=args.strategy)
    print_insights(res, label=args.strategy)

    if args.stats:
        print_stats(res, label=args.strategy)

    if args.montecarlo:
        print_montecarlo(res, args.montecarlo)

    if args.attribution:
        print_attribution(res)

    plot_series = [(args.strategy, res)]

    if args.compare:
        get_cmp = load_strategy(args.compare)
        res_cmp = run_backtest(prc_hist, get_cmp, num_test_days=args.days,
                               comm_rate=comm_rate, dlr_pos_limit=dlr_pos_limit, inst_names=names)
        print_summary(res_cmp, label=args.compare)
        plot_series.append((args.compare, res_cmp))
        print(f"\ndelta score ({args.strategy} - {args.compare}): "
              f"{res.score - res_cmp.score:+.2f}")

    if args.walk_forward:
        walk_forward(prc_hist, wf_target, args.walk_forward, args.days,
                     comm_rate, dlr_pos_limit, names)

    if args.rolling:
        rolling(prc_hist, wf_target, args.rolling, args.days,
                comm_rate, dlr_pos_limit, names)

    if args.csv:
        dump_csv(res, args.csv)

    if args.export_positions:
        export_positions(res, args.export_positions)

    if args.export_runs:
        export_runs(prc_hist, comm_rate, dlr_pos_limit, names, args.export_runs, args.days)

    if args.plot:
        make_plot(plot_series, args.plot)

    if args.plot_entries is not None:
        plot_entries(res, prc_hist, args.plot_entries, names, "entries.png")

    return res


if __name__ == "__main__":
    main()
