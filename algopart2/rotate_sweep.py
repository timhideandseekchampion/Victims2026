"""rotate_sweep.py — the aggressiveness dial for the protection layer. The champion (lead-lag +
reversion) is the main edge; rotation is protection. Faster settings capture a regime switch sooner
but whipsaw more in a choppy regime. Quantify the trade-off across presets so we can pick a point:

  COST     real-data 500-750 SCORE + #days off champion   (should stay ~694 — no momentum to fire on)
  CAPTURE  switch-lag in a SUSTAINED injected momentum regime (days from onset to switch)
  WHIPSAW  #signal-flips + wrong-signal days in a CHOPPY regime (edge alternates every `period` days)
"""
import numpy as np, pandas as pd
import SAFE_rotate as R
BENCH = tuple(R.CHALLENGERS)   # traded bench (pruned)

prc = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nDays = prc.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0

def set_params(W, P, TC, BONF, KILL=True):
    R.ROT_W, R.ROT_P, R.ROT_TCRIT, R.ROT_BONF, R.KILL_ON = W, P, TC, BONF, KILL

# ---------- COST: real-data score + days off champion ----------
R._ensure_cache(prc)                                   # warm once (params don't affect the cache)
REAL_SIG = {k: v for k, v in R._SIG.items()}           # snapshot the real forecast cache
REAL_RET = {k: v for k, v in R._RET.items()}           # (synthetic-world tests clobber R._SIG later)
def real_cost(W, P, TC, BONF):
    set_params(W, P, TC, BONF)
    R._SIG = {k: v for k, v in REAL_SIG.items()}       # restore real cache (undo any synthetic pollution)
    R._RET = {k: v for k, v in REAL_RET.items()}; R._ICD.clear(); R._XC.clear()
    cash=0.0; cp=np.zeros(nInst); value=0.0; comm=0.0; pll=[]; off=0
    for t in range(500, 751):
        cur=prc[:,t-1]
        if t<750:
            newPos=R.getMyPosition(prc[:,:t])
            ch = R._choose(t) if t>=R.WARMUP+R.ROT_W+R.ROT_P else "champ"
            if ch!="champ" or (newPos[1:]==0).all(): off+=1     # rotated OR killed (flat)
        else: newPos=cp
        dP=newPos-cp; cash-=cur.dot(dP)+comm; comm=np.sum(cur*np.abs(dP)*commRate); cp=newPos
        pl=cash+cp.dot(cur)-value; value=cash+cp.dot(cur)
        if t>500: pll.append(pl)
    pll=np.array(pll); mu,sd=pll.mean(),pll.std()
    sc=mu*(np.sqrt(250)*mu/sd)**2/((np.sqrt(250)*mu/sd)**2+1) if mu>0 else mu
    return sc, off

# ---------- synthetic IC worlds (monkeypatch the controller) ----------
N=50
def build_world(kind, DAYS=650, D1=200, D2=420, period=30, seed=0):
    rng=np.random.default_rng(seed); SIG={}; RET={}
    for n in range(DAYS):
        truth=rng.standard_normal(N); truth-=truth.mean(); RET[n]=truth
        if kind=="sustained": on = (D1<=n<D2)
        else:                  on = ((n//period)%2==1)          # choppy: alternate every `period`
        champ_f=(rng.standard_normal(N) if on else truth*0.9+rng.standard_normal(N))
        mom_f  =(truth*0.9+rng.standard_normal(N) if on else rng.standard_normal(N))
        d={"champ":champ_f,"momJT":mom_f}
        for k in [nm for nm in BENCH if nm != "momJT"]: d[k]=rng.standard_normal(N)
        SIG[n]={k:(v-v.mean()) for k,v in d.items()}
    return SIG, RET

def capture_lag(W,P,TC,BONF, D1=200,D2=420):
    set_params(W,P,TC,BONF); SIG,RET=build_world("sustained",D1=D1,D2=D2)
    R._SIG,R._RET=SIG,RET; R._ICD.clear(); R._XC.clear(); R.CHALLENGERS=BENCH
    picks={t:R._choose(t) for t in range(R.WARMUP+W+P, 650)}
    first=next((t for t,p in picks.items() if p=="momJT"), None)
    reclaim=next((t for t,p in picks.items() if t>D2 and p=="champ"), None)
    return (first-D1 if first else None), (reclaim-D2 if reclaim else None)

def whipsaw(W,P,TC,BONF, period=30):
    set_params(W,P,TC,BONF); SIG,RET=build_world("choppy",period=period)
    R._SIG,R._RET=SIG,RET; R._ICD.clear(); R._XC.clear(); R.CHALLENGERS=BENCH
    picks=[R._choose(t) for t in range(R.WARMUP+W+P, 650)]
    flips=sum(1 for i in range(1,len(picks)) if picks[i]!=picks[i-1])
    nonchamp=sum(1 for p in picks if p!="champ")
    return flips, nonchamp, len(picks)

presets = {
    "Conservative (current)": (60,10,3.0,True),
    "Balanced":               (40, 7,2.5,True),
    "Aggressive":             (30, 5,2.0,False),
    "Very aggressive":        (20, 5,2.0,False),
}
print(f"{'preset':<24}{'realScore':>10}{'offChamp':>9}   {'captureLag':>11}{'reclaimLag':>11}   {'chopFlips':>10}{'chopWrong%':>11}")
for name,(W,P,TC,BONF) in presets.items():
    sc,off = real_cost(W,P,TC,BONF)
    lag,rec = capture_lag(W,P,TC,BONF)
    fl,nc,tot = whipsaw(W,P,TC,BONF)
    print(f"{name:<24}{sc:>10.0f}{off:>9}   {str(lag):>11}{str(rec):>11}   {fl:>10}{100*nc/tot:>10.0f}%")
print("\ncaptureLag = days after regime onset to switch (lower=faster protection)")
print("chopFlips/chopWrong% = whipsaw in a choppy regime that flips every 30d (lower=steadier; the COST of speed)")
# restore defaults
set_params(40,7,2.5,True); R.CHALLENGERS=BENCH   # restore Balanced defaults
