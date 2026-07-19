"""rotate_test.py — validate SAFE_rotate.py:
 (1) SAFETY on real data: never rotates (bar not met) => identical positions/score to SAFE_lldollar.
 (2) FUNCTION + SWITCH-BACK: inject a regime that turns ON then OFF; controller must go
     champ -> challenger -> champ (reclaim the strong LL+reversion when the edge dies).
 (3) FALSE-POSITIVE STRESS: a bench of pure-noise challengers must (almost) never trigger a
     rotation. Measure single-day spurious qualifications vs actual rotations (persistence),
     with and without the Bonferroni bump, as the bench grows.
"""
import numpy as np, pandas as pd
import SAFE_rotate as R
import SAFE_lldollar as BASE

prc = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nDays = prc.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
N = 50

def score(mu, sd):
    if mu <= 0 or sd < 1e-10: return mu
    sr = np.sqrt(250) * mu / sd; return mu * sr**2 / (sr**2 + 1)

def evalpos(getpos, S, E):
    cash=0.0; cp=np.zeros(nInst); value=0.0; comm=0.0; pll=[]; pos_by_t={}
    for t in range(S, E+1):
        cur=prc[:,t-1]; newPos=getpos(prc[:,:t]) if t<E else cp
        lim=(dlr/cur).astype(int); newPos=np.clip(newPos,-lim,lim).astype(int); pos_by_t[t]=newPos
        dP=newPos-cp; cash-=cur.dot(dP)+comm; comm=np.sum(cur*np.abs(dP)*commRate); cp=newPos
        pl=cash+cp.dot(cur)-value; value=cash+cp.dot(cur)
        if t>S: pll.append(pl)
    pll=np.array(pll); return pll.mean(), pll.std(), score(pll.mean(), pll.std()), pos_by_t

# ================= (1) SAFETY ON REAL DATA =================
print("="*72); print(f"(1) SAFETY — real data ({len(R.CHALLENGERS)} challengers, Bonferroni on)"); print("="*72)
R._ensure_cache(prc)
print(f"significance bar: base TCRIT={R.ROT_TCRIT}  ->  effective (Bonf, C={len(R.CHALLENGERS)}) = {R._tcrit():.2f}")
T0 = R.WARMUP + R.ROT_W + R.ROT_P
picks = {t: R._choose(t) for t in range(T0, nDays+1)}
from collections import Counter
print(f"rotation decisions over days {T0}-{nDays}:  {dict(Counter(picks.values()))}")

lo, hi = 500, 750
icc = R._ic("champ", lo, hi)
print(f"\nrealized IC over {lo}-{hi}  (bar to rotate: beat champ AND t>{R._tcrit():.2f}):")
print(f"  {'champ':<7} mean={icc.mean():+.4f}  t={icc.mean()/(icc.std()/np.sqrt(len(icc))):+.2f}")
for name in R.CHALLENGERS:
    ic = R._ic(name, lo, hi); d = ic - icc
    t_i = ic.mean()/(ic.std()/np.sqrt(len(ic))+1e-18); t_d = d.mean()/(d.std()/np.sqrt(len(d))+1e-18)
    ok = "ROTATE" if (d.mean()>=R.ROT_MARGIN and t_d>R._tcrit() and ic.mean()>0 and t_i>R._tcrit()) else "stay"
    print(f"  {name:<7} mean={ic.mean():+.4f}  t={t_i:+.2f}   vs champ dIC={d.mean():+.4f} t_d={t_d:+.2f}  -> {ok}")

mr = evalpos(R.getMyPosition, lo, hi); mb = evalpos(BASE.getMyPosition, lo, hi)
diffs = max(int(np.abs(mr[3][t]-mb[3][t]).max()) for t in mr[3])
print(f"\nmax |SAFE_rotate - SAFE_lldollar| position over {lo}-{hi}: {diffs} shares (0=identical)")
print(f"score:  rotate={mr[2]:.1f}   lldollar={mb[2]:.1f}")

# ================= (2) FUNCTION + SWITCH-BACK =================
print("\n"+"="*72); print("(2) FUNCTION + SWITCH-BACK — regime ON [D1,D2) then OFF"); print("="*72)
rng = np.random.default_rng(0); DAYS=650; D1=200; D2=420
SIG={}; RET={}
TRUTH = "momJT"                                            # the momentum challenger under test
BENCH = tuple(R.CHALLENGERS)                               # the traded bench (pruned to momentum family)
for n in range(DAYS):
    truth = rng.standard_normal(N); truth -= truth.mean(); RET[n]=truth
    on = (D1 <= n < D2)                                     # TRUTH is the true edge only inside the regime
    champ_f = (rng.standard_normal(N) if on else truth*0.9+rng.standard_normal(N))
    mom_f   = (truth*0.9+rng.standard_normal(N) if on else rng.standard_normal(N))
    d = {"champ":champ_f, TRUTH:mom_f}
    for k in [nm for nm in BENCH if nm != TRUTH]: d[k]=rng.standard_normal(N)
    SIG[n]={k:(v-v.mean()) for k,v in d.items()}
R._SIG, R._RET = SIG, RET; R._ICD.clear(); R._XC.clear()
R.CHALLENGERS = BENCH
tl = {t: R._choose(t) for t in range(R.WARMUP+R.ROT_W+R.ROT_P, DAYS)}
first_mom  = next((t for t,p in tl.items() if p==TRUTH), None)
reclaim    = next((t for t,p in tl.items() if t>D2 and p=="champ"), None)
print(f"regime ON at D1={D1}, OFF at D2={D2}   (W={R.ROT_W}, P={R.ROT_P}, bar={R._tcrit():.2f})")
print(f"  before D1  (150-199): {set(tl.get(t) for t in range(150,200))}")
print(f"  rotates to '{TRUTH}' at day {first_mom}   (expected ~ D1+W+P = {D1+R.ROT_W+R.ROT_P})")
print(f"  mid-regime (330-380): {set(tl.get(t) for t in range(330,380))}")
print(f"  RECLAIMS champ at day {reclaim}   (after regime ends at D2={D2})")
print(f"  after settle(560-620): {set(tl.get(t) for t in range(560,620))}")

# ================= (3) FALSE-POSITIVE STRESS =================
print("\n"+"="*72); print("(3) FALSE-POSITIVE STRESS — pure-noise bench (champ weak, IC~0.07)"); print("="*72)
def noise_run(n_noise, bonf, days=450, seed=7):
    R.ROT_BONF = bonf
    rng = np.random.default_rng(seed)
    names = tuple(f"z{i}" for i in range(n_noise))
    SIG={}; RET={}
    for n in range(days):
        truth = rng.standard_normal(N); truth -= truth.mean(); RET[n]=truth
        cf = truth*0.07 + rng.standard_normal(N)           # champion ~ realistic weak IC
        d = {"champ": cf-cf.mean()}
        for nm in names:
            z = rng.standard_normal(N); d[nm]=z-z.mean()
        SIG[n]=d
    R._SIG, R._RET, R.CHALLENGERS = SIG, RET, names; R._ICD.clear(); R._XC.clear()
    T0 = R.WARMUP + R.ROT_W + R.ROT_P
    quals = sum(1 for a in range(R.WARMUP+R.ROT_W, days) if R._gate_at(a) is not None)
    rots  = sum(1 for t in range(T0, days) if R._choose(t) != "champ")
    ndays = days - T0
    return quals, rots, ndays

print(f"{'bench':<10}{'Bonf':<7}{'bar':>6}{'1-day quals':>13}{'rotations':>11}   (of ~284 decision-days)")
for n_noise in (6, 20, 40):
    for bonf in (False, True):
        R.CHALLENGERS = tuple(f"z{i}" for i in range(n_noise)); R.ROT_BONF = bonf
        bar = R._tcrit()
        q,r,nd = noise_run(n_noise, bonf)
        print(f"{n_noise:<10}{str(bonf):<7}{bar:>6.2f}{q:>13}{r:>11}")
R.ROT_BONF = True
