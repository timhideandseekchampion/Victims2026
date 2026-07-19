"""consecutive_analysis.py — is 'P consecutive qualifying days' a p^P lottery?
The p^7 argument needs the P daily events to be INDEPENDENT Bernoulli(p). Two reasons it isn't:
  (i)  the gate does NOT ask 'did momentum win TODAY' (a noisy ~1-day event, the ~80% you mean).
       It asks 'does momentum's 40-DAY-average IC significantly beat the champion's' — a smooth stat.
  (ii) consecutive gates share 39/40 of their window, so the daily gate decision is heavily
       AUTOCORRELATED; passes come in long runs, not independent flips.
This builds a MARGINAL regime (champion keeps part of its edge, momentum only somewhat better) and
sweeps the gap, reporting BOTH the signal's 1-day win-rate and the gate's daily pass-rate, the
day-to-day autocorrelation of the gate decision, p^7, and the ACTUAL switch behaviour. xsac stays
OFF (serially-random RET) so this is the STRICT 7-day lane — the hardest case.
"""
import numpy as np
import SAFE_rotate as R
N = 50

def world(champ_s, mom_s, DAYS=800, D1=200, D2=760, seed=0):
    rng = np.random.default_rng(seed); SIG = {}; RET = {}
    for n in range(DAYS):
        truth = rng.standard_normal(N); truth -= truth.mean(); RET[n] = truth
        on = D1 <= n < D2
        champ_f = (truth * champ_s if on else truth * 0.9) + rng.standard_normal(N)   # champ keeps edge in-regime
        mom_f   = (truth * mom_s   if on else 0.0)        + rng.standard_normal(N)     # momentum a bit better in-regime
        d = {"champ": champ_f, "momJT": mom_f, "mom": rng.standard_normal(N), "residMom": rng.standard_normal(N)}
        SIG[n] = {k: (v - v.mean()) for k, v in d.items()}
    return SIG, RET, D1, D2

R.CHALLENGERS = ("mom", "momJT", "residMom")
print(f"STRICT lane: P={R.ROT_P} consecutive days, bar={R._tcrit():.2f}, ROT_W={R.ROT_W}. champ keeps edge in-regime.")
print(f"{'gap':>5}{'momIC':>7}{'champIC':>8}{'SIGNAL win/day':>15}{'GATE p/day':>12}{'p^7':>7}{'gate autocorr':>14}{'ACTUAL sw%':>11}")
CH = 0.55
for mom_s in (0.55, 0.62, 0.70, 0.80, 0.95, 1.20):
    SIG, RET, D1, D2 = world(CH, mom_s)
    R._SIG, R._RET, R._ICD, R._XC = SIG, RET, {}, {}
    reg = range(D1 + R.ROT_W, D2)
    icm = np.array([R._ic1("momJT", n) for n in reg]); icc = np.array([R._ic1("champ", n) for n in reg])
    sig_win = float((icm > icc).mean())                                   # the noisy 1-day win rate (~your 80%)
    g = np.array([1 if R._gate_at(a, R._tcrit()) == "momJT" else 0 for a in reg])
    p = g.mean()
    ac = float(np.corrcoef(g[:-1], g[1:])[0, 1]) if g.std() > 0 else 1.0  # constant -> perfect persistence
    ch = np.array([1 if R._choose(t) == "momJT" else 0 for t in reg])
    print(f"{mom_s-CH:>5.2f}{icm.mean():>7.3f}{icc.mean():>8.3f}{sig_win:>15.2f}{p:>12.2f}{p**7:>7.2f}{ac:>14.2f}{ch.mean():>11.2f}")
print("\nSIGNAL win/day = fraction of single days momentum's IC beats champion's  (this is your ~0.8).")
print("GATE p/day     = fraction of days the 40-day t-test clears the bar        (what actually gates).")
print("p^7 = what switching would be IF gate days were independent; ACTUAL sw% = what really happens.")
print("gate autocorr ~1 => passes run in blocks, so ACTUAL sw% >> p^7. The 40-day averaging also makes")
print("GATE p/day jump ~0->1 over a narrow gap: a marginal edge is (correctly) blocked; a real one sails.")
