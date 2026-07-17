"""Calibrated generator for the Algothon 2026 DGP — sample faithful synthetic futures.

Reverse-engineered model (see dgp_reverse.py): a stationary linear-Gaussian VAR(1) on
daily log-returns with zero drift and a one-factor-ish Gaussian innovation covariance,
plus ALGO as an equal-weight price index of the 50 constituents.

    r_t = signal_scale * A @ r_{t-1} + eps_t,   eps_t ~ N(0, Sigma),   drift = 0
    price_t = price_{t-1} * exp(r_t)
    ALGO_t  = base * mean_i(price_{i,t})           # equal-weight arithmetic index

`A` and `Sigma` are fit from the real 500 days. Because a finite-sample ridge estimate
of a dense 2500-parameter matrix is SHRUNK, simulating straight from it produces data
that is slightly LESS predictable than reality; `signal_scale` (calibrated by
`.calibrate()`) rescales `A` so a walk-forward strategy recovers the REAL out-of-sample
IC (~0.051). Only after calibration is the simulator faithful.

Two entry points:
    DGP.fit(prices).simulate(n_days, seed)        # fresh synthetic history (realistic levels)
    DGP.fit(prices).extend(prices, n_future, seed) # continue REAL history into the future
                                                    # -> the honest forward-window engine

Uses numpy's seeded RNG (reproducible). No competition data leaks into a simulation:
fitting uses only days already observed.
"""
import numpy as np


def _fit_var1(ret, alpha):
    """Ridge VAR(1): returns (A, Sigma, resid). ret is (nAssets, T)."""
    X = ret[:, :-1].T                                   # (T-1, n) predictors r_{t-1}
    Y = ret[:, 1:].T                                    # (T-1, n) targets   r_t
    mx = X.mean(0); Xc = X - mx; Yc = Y - Y.mean(0)     # demean (drift handled separately = 0)
    n = X.shape[1]
    lam = alpha * np.trace(Xc.T @ Xc) / n
    A = np.linalg.solve(Xc.T @ Xc + lam * np.eye(n), Xc.T @ Yc).T
    resid = Yc - Xc @ A.T
    Sigma = np.cov(resid.T)
    return A, Sigma, resid


class DGP:
    def __init__(self, A, Sigma, p0, algo_p0, algo_w, signal_scale=1.0, zero_drift=False):
        self.A = A                          # (50,50) transition among constituents
        self.Sigma = Sigma                  # (50,50) innovation covariance
        self.p0 = p0                        # (50,) constituent starting prices (real day-0)
        self.algo_p0 = algo_p0              # scalar: real ALGO starting price
        self.algo_w = algo_w                # (50,) fitted return-weights -> ALGO is a weighted-return index (zero drift)
        self.signal_scale = signal_scale
        self.zero_drift = zero_drift        # DGP drift PARAMETER is 0; default False = natural sampling scatter
                                            # (real: chi2(49) p=0.61 => zero-mean scatter, NOT sample-demeaned)
        # Cholesky of Sigma (jitter for PSD safety) for MVN sampling
        self._L = np.linalg.cholesky(Sigma + 1e-12 * np.eye(len(Sigma)) * np.trace(Sigma) / len(Sigma))

    # ---------- fitting ----------
    @classmethod
    def fit(cls, prices, alpha=0.1, signal_scale=1.0):
        """prices: (51, T) real panel incl. ALGO at row 0."""
        lp = np.log(prices)
        ret = lp[:, 1:] - lp[:, :-1]
        A, Sigma, _ = _fit_var1(ret[1:], alpha)          # constituents only (drop ALGO)
        # recover ALGO as a weighted-return index of the constituents (R^2~0.995, drift~0)
        G = np.linalg.lstsq(np.c_[np.ones(ret.shape[1]), ret[1:].T], ret[0], rcond=None)[0]
        return cls(A, Sigma, prices[1:, 0].copy(), float(prices[0, 0]), G[1:], signal_scale)

    # ---------- core generation ----------
    def _gen_returns(self, n_days, rng, r_init=None):
        n = len(self.A)
        eps = (self._L @ rng.standard_normal((n, n_days))).T   # (n_days, n) ~ N(0, Sigma)
        r = np.zeros((n_days, n))
        prev = np.zeros(n) if r_init is None else r_init
        sA = self.signal_scale * self.A
        for t in range(n_days):
            r[t] = sA @ prev + eps[t]
            prev = r[t]
        return r                                          # (n_days, n) log-returns

    def _panel(self, r, p0_const, algo_start):
        """Build a (51, len) price panel from constituent returns r (n_days, 50).

        Constituents cumulate from p0_const; ALGO is the fitted weighted-return index
        (zero drift by construction), cumulating from algo_start.
        """
        if self.zero_drift:
            r = r - r.mean(0)                              # force exact per-asset zero drift
        cp = p0_const[:, None] * np.exp(np.cumsum(r.T, axis=1))          # (50, n_days)
        algo_ret = self.algo_w @ r.T                                     # (n_days,)
        ap = algo_start * np.exp(np.cumsum(algo_ret))                    # (n_days,)
        return np.vstack([ap[None, :], cp])                             # (51, n_days)

    def simulate(self, n_days, seed):
        """Fresh synthetic history of length n_days (realistic price levels)."""
        rng = np.random.default_rng(seed)
        burn = 200                                         # draw initial state from ~stationary dist
        r = self._gen_returns(n_days + burn, rng)[burn:]
        return self._panel(r, self.p0, self.algo_p0)

    def extend(self, prices, n_future, seed):
        """Continue the REAL panel `prices` (51,T) with n_future synthetic days.

        Seeds the VAR state from the last real return so the join is seamless. Returns
        (51, T + n_future). This is the forward-window engine: fit on 1..T, and the
        appended days are a faithful draw of what T+1..T+n_future could look like.
        """
        rng = np.random.default_rng(seed)
        lp = np.log(prices)
        r = self._gen_returns(n_future, rng, r_init=lp[1:, -1] - lp[1:, -2])
        fut = self._panel(r, prices[1:, -1], float(prices[0, -1]))
        return np.hstack([prices, fut])

    # ---------- calibration ----------
    def calibrate(self, target_oos_ic=0.051,
                  scales=(0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0), seed=0, n=800):
        """Pick signal_scale so a plain ridge walk-forward on SIM data hits the real OOS IC.

        Returns (best_scale, table). Mutates self.signal_scale to the best.
        """
        table = []
        for sc in scales:
            self.signal_scale = sc
            ic = self._sim_oos_ic(seed=seed, n=n)
            table.append((sc, ic))
        best = min(table, key=lambda x: abs(x[1] - target_oos_ic))
        self.signal_scale = best[0]
        return best[0], table

    def _sim_oos_ic(self, seed, n):
        """Fit ridge on first half of a length-n sim, measure cross-sec IC on second half."""
        panel = self.simulate(n, seed)
        lp = np.log(panel); ret = lp[1:, 1:] - lp[1:, :-1]     # constituents (50, n)
        h = ret.shape[1] // 2
        A, _, _ = _fit_var1(ret[:, :h], alpha=0.1)
        Xte = ret[:, h:-1].T; Yte = ret[:, h+1:].T
        Xte = Xte - Xte.mean(0)
        pred = Xte @ A.T
        ics = [np.corrcoef(pred[k], Yte[k])[0, 1] for k in range(len(Yte)) if Yte[k].std() > 0]
        return float(np.mean(ics))


# ============================ self-test / faithfulness check ============================
if __name__ == "__main__":
    import pandas as pd
    real = pd.read_csv("prices.txt", sep=r"\s+").values.T
    print(f"Fitting DGP on real panel {real.shape} ...")
    dgp = DGP.fit(real)

    # 1) calibrate signal_scale to the real OOS IC
    best, table = dgp.calibrate(target_oos_ic=0.051)
    print("\n[calibrate] signal_scale -> sim OOS IC:")
    for sc, ic in table:
        print(f"    scale {sc:>4}:  sim OOS IC {ic:+.4f}{'   <-- chosen' if sc==best else ''}")
    print(f"  chosen signal_scale = {best}  (target real OOS IC 0.051)")

    # 2) faithfulness: do synthetic panels reproduce the real stylized facts?
    def facts(panel):
        lp = np.log(panel); ret = lp[1:, 1:] - lp[1:, :-1]
        T = ret.shape[1]
        mu_t = np.abs(ret.mean(1) / (ret.std(1) / np.sqrt(T)))       # per-asset drift |t|
        Sig = np.cov(ret.T); ev = np.linalg.eigvalsh(Sig)[::-1]
        disp = ret.std(0).mean()                                     # daily cross-sec dispersion
        algo_dd = (np.log(panel[0, 1:]) - np.log(panel[0, :-1]))
        return dict(drift_maxt=mu_t.max(), drift_hi=(mu_t > 2).sum(),
                    fac1=ev[0] / ev.sum(), disp=disp,
                    algo_yr=algo_dd.mean() * 252 * 100)
    fr = facts(real)
    sims = [dgp.simulate(499, seed=s) for s in range(6)]
    fss = [facts(p) for p in sims]
    print("\n[faithfulness] real vs synthetic (mean of 6 sims):")
    print(f"  {'stat':22} {'REAL':>10} {'SIM':>10}")
    for k, lab in [("drift_hi", "# assets |t_drift|>2"), ("drift_maxt", "max |t_drift|"),
                   ("fac1", "factor-1 var share"), ("disp", "cross-sec dispersion"),
                   ("algo_yr", "ALGO drift %/yr")]:
        s = np.mean([f[k] for f in fss])
        print(f"  {lab:22} {fr[k]:>10.4f} {s:>10.4f}")

    # 3) demo the forward engine
    fut = dgp.extend(real, n_future=500, seed=42)
    print(f"\n[extend] real {real.shape} -> with 500 synthetic future days -> {fut.shape}")
    print("  (fit strategies on days 1-500, score on the synthetic 501-1000 for forward MC)")
