"""Tournament simulator — how much directional risk maximizes P(top-10 advance)?

Turns "optimal risk within reason" into a number. Built on the calibrated DGP
(dgp_simulator.py) so the forward windows and the achievable edge are faithful.

MODEL (grounded in the reverse-engineered facts):
  * Everyone shares the SAME commoditized book edge (peer lead-lag, IC~0.05). On a given
    forward window all competitors' book PnL is ~the same series -> the book does NOT
    differentiate rank. What differentiates rank is (a) the net DIRECTIONAL beta each
    team layers on and (b) how the market happens to move that window.
  * Position caps ($10k/name, $100k ALGO) mean the only real variance lever is net beta
    (dropping market-neutrality). Drift=0 => beta is ~0-EV: pure symmetric variance.
  * Score = mean_daily_PnL * SR^2/(SR^2+1)  (exact eval.py scorer; depends only on
    per-window daily-PnL mean & std, so we work analytically from per-window moments).

Per forward window w we extract 5 numbers from a real book run:
    book_mean, book_var, algo_mean (window market drift), algo_var, cov(book, algo).
A competitor with book-quality q_i and net beta b_i has:
    mean = q_i*book_mean + b_i*algo_mean
    var  = q_i^2*book_var + b_i^2*algo_var + 2*q_i*b_i*cov + idio_var
    Score = scorer(mean, sqrt(var))
We pick OUR beta; the field draws (q_i, b_i). Rank us among N+1; tally P(top-10).

The field's beta appetite (sigma_field) is the ONE unknown we can't yet observe -- it is
exactly what the practice-board score DISPERSION will reveal. So we run scenarios and
show how the optimal shifts; recalibrate sigma_field the moment real data lands.
"""
import numpy as np, pandas as pd
from dgp_simulator import DGP
from validate_oos import build_getpos, score_window

# ----------------- config -----------------
M_WINDOWS = 60           # synthetic forward windows drawn from the calibrated DGP
FIELD_DRAWS = 300        # field resamples per window
N_FIELD = 40             # rival teams (we are ~41st on the real board)
TOP_K = 10               # advance threshold
BOOK_CV = 0.15           # implementation dispersion in rivals' book quality q_i ~ N(1, cv)
IDIO_FRAC = 0.30         # rivals' extra idiosyncratic PnL sd, as a fraction of book sd
OUR_BETAS = [0, 50_000, 100_000, 200_000, 300_000, 500_000]     # |net beta| dollars we sweep
FIELD_SIGMAS = {"neutral field": 50_000, "moderate field": 200_000, "aggressive field": 400_000}


def scorer(mu, sd):
    if mu <= 0 or sd < 1e-10:
        return mu
    sr = np.sqrt(250) * mu / sd
    return mu * sr ** 2 / (sr ** 2 + 1)


def window_moments(real, seed):
    """One synthetic 250-day forward window: return (book_mean, book_var, a_mean, a_var, cov)."""
    panel = DGP_.extend(real, 250, seed)                     # (51, 750): real 1-500 + synth 501-750
    lean = dict(half_life=2000, contra_dollars=0)            # market-neutral common edge (the book)
    r = score_window(panel, build_getpos(lean), start_day=500, test_len=250)
    book_pl = r["pl"]                                        # (250,) daily book PnL
    algo = panel[0, 500:]                                    # ALGO price over the scored window
    algo_ret = np.diff(np.log(algo))                         # daily market return ~ PnL per $1 beta
    n = min(len(book_pl), len(algo_ret))
    book_pl, algo_ret = book_pl[:n], algo_ret[:n]
    cov = np.cov(book_pl, algo_ret)[0, 1]
    return book_pl.mean(), book_pl.var(), algo_ret.mean(), algo_ret.var(), cov


def our_score(m, beta):
    bm, bv, am, av, cov = m
    mean = bm + beta * am
    var = bv + beta ** 2 * av + 2 * beta * cov
    return scorer(mean, np.sqrt(max(var, 1e-12)))


def field_scores(m, sigma_field, rng):
    """N_FIELD rival scores on this window: shared book (quality q_i) + their own beta."""
    bm, bv, am, av, cov = m
    q = rng.normal(1.0, BOOK_CV, N_FIELD)
    b = rng.normal(0.0, sigma_field, N_FIELD)               # random-signed directional bets
    idio = (IDIO_FRAC ** 2) * bv * rng.chisquare(1, N_FIELD)  # extra idiosyncratic variance
    mean = q * bm + b * am
    var = q ** 2 * bv + b ** 2 * av + 2 * q * b * cov + idio
    sr = np.sqrt(250) * mean / np.sqrt(np.maximum(var, 1e-12))
    sc = mean * sr ** 2 / (sr ** 2 + 1)
    return np.where(mean > 0, sc, mean)


if __name__ == "__main__":
    real = pd.read_csv("prices.txt", sep=r"\s+").values.T
    DGP_ = DGP.fit(real); DGP_.calibrate(target_oos_ic=0.051)
    print(f"calibrated signal_scale={DGP_.signal_scale};  drawing {M_WINDOWS} forward windows...\n")
    moms = [window_moments(real, seed=1000 + i) for i in range(M_WINDOWS)]

    # sanity: distribution of window market drift (should straddle 0 -> beta is a coin flip)
    ann = np.array([m[2] for m in moms]) * 252 * 100
    print(f"forward-window market drift: mean {ann.mean():+.1f}%/yr  sd {ann.std():.1f}  "
          f"(P(up)={np.mean(ann>0):.0%}) -> beta ~0-EV symmetric\n")

    rng = np.random.default_rng(0)
    for fname, sigma in FIELD_SIGMAS.items():
        print(f"=== {fname}  (rivals' beta sd = ${sigma:,}) ===")
        print(f"  {'our |beta|':>12} {'P(top10)':>9} {'mean rank':>10} {'ourScore p50':>13} {'[p5, p95]':>16}")
        best = (None, -1)
        for beta in OUR_BETAS:
            top = 0; ranks = []; scores = []
            for m in moms:
                osc = our_score(m, beta)
                for _ in range(FIELD_DRAWS):
                    fs = field_scores(m, sigma, rng)
                    rank = 1 + int((fs > osc).sum())          # our rank among N+1
                    ranks.append(rank); top += (rank <= TOP_K)
                scores.append(osc)
            n = M_WINDOWS * FIELD_DRAWS
            ptop = top / n
            if ptop > best[1]:
                best = (beta, ptop)
            sc = np.array(scores)
            print(f"  {beta:>12,} {ptop:>8.1%} {np.mean(ranks):>10.1f} {np.median(sc):>13.0f} "
                  f"[{np.percentile(sc,5):>6.0f},{np.percentile(sc,95):>6.0f}]")
        print(f"  -> optimal |beta| = ${best[0]:,}  (P(top10) {best[1]:.1%})\n")

    print("READ: if optimal|beta|=0, don't add directional risk. If it rises with the field's")
    print("sigma, we must MATCH the field's variance to stay in the lottery -- and the practice")
    print("board's score DISPERSION is what pins sigma_field. Recalibrate then.")
