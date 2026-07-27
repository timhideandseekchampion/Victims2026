"""Build trailing_ic.html — show WHY the ALGO vol switch works and the cross-sectional switch doesn't.
Panel 1: trailing cross-sectional reversion IC (stationary noise around a negative mean — the 'momentum
zones' are just excursions inside the ±2SE noise band). Panel 2: trailing ALGO vol IC (a real, drifting,
persistently-positive regime). Data computed here and inlined."""
import json, numpy as np, pandas as pd
P = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P.shape; logp = np.log(P)
R = np.full((nInst, nt), np.nan); R[:, 1:] = logp[:, 1:] - logp[:, :-1]
NR = np.full((nInst, nt), np.nan); NR[:, :-1] = logp[:, 1:] - logp[:, :-1]

# --- cross-sectional 10-day momentum feature (demeaned, z-scored each day) ---
MOM = np.full((nInst, nt), np.nan)
for t in range(10, nt):
    m = logp[1:, t] - logp[1:, t - 10]; m = m - m.mean(); MOM[1:, t] = m / (m.std() + 1e-12)
# daily cross-sectional IC (for the noise band) and trailing 120d pooled IC
daily = []
for s in range(11, nt - 1):
    x = MOM[1:, s]; y = NR[1:, s]; mm = ~np.isnan(x) & ~np.isnan(y)
    if mm.sum() > 20: daily.append(np.corrcoef(x[mm], y[mm])[0, 1])
se = np.std(daily) / np.sqrt(120)                         # SE of the 120-day pooled IC
xs_days, xsic = [], []
for t in range(200, nt):
    xs, ys = [], []
    for s in range(t - 120, t):
        mm = ~np.isnan(MOM[1:, s]) & ~np.isnan(NR[1:, s]); xs.append(MOM[1:, s][mm]); ys.append(NR[1:, s][mm])
    x, y = np.concatenate(xs), np.concatenate(ys); xs_days.append(t); xsic.append(float(np.corrcoef(x, y)[0, 1]))
xs_mean = float(np.mean(xsic))

# --- ALGO vol trailing 90d IC ---
r0 = np.diff(logp[0]); vol = np.full(nt, np.nan)
for t in range(20, nt): vol[t] = r0[t - 20:t].std()
volz = np.full(nt, np.nan)
for t in range(80, nt):
    w = vol[t - 60:t]; volz[t] = (vol[t] - w.mean()) / (w.std() + 1e-12)
ret1 = np.full(nt, np.nan); ret1[:nt - 1] = logp[0, 1:] - logp[0, :-1]
vol_days, volic = [], []
for t in range(150, nt):
    a = max(0, t - 90); xs = volz[a:t]; ys = ret1[a:t]; ok = ~np.isnan(xs) & ~np.isnan(ys)
    if ok.sum() > 60 and xs[ok].std() > 1e-12:
        vol_days.append(t); volic.append(float(np.corrcoef(xs[ok], ys[ok])[0, 1]))
vol_se = 1.0 / np.sqrt(90)

D = {"xs": {"days": xs_days, "ic": [round(v, 4) for v in xsic], "mean": round(xs_mean, 4), "se": round(float(se), 4)},
     "vol": {"days": vol_days, "ic": [round(v, 4) for v in volic], "se": round(float(vol_se), 4)}}

HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Trailing IC — real regime vs noise</title>
<style>
:root{--surface:#fcfcfb;--plane:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;
 --border:rgba(11,11,11,.10);--blue:#2a78d6;--aqua:#1baf7a;--red:#e34948;--violet:#4a3aa7;--noiseband:rgba(137,135,129,.14);--posband:rgba(27,175,122,.12);color-scheme:light;}
@media (prefers-color-scheme:dark){:root{--surface:#1a1a19;--plane:#0d0d0d;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;
 --border:rgba(255,255,255,.10);--blue:#3987e5;--aqua:#199e70;--red:#e66767;--violet:#9085e9;--noiseband:rgba(137,135,129,.16);--posband:rgba(25,158,112,.15);color-scheme:dark;}}
:root[data-theme=dark]{--surface:#1a1a19;--plane:#0d0d0d;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--blue:#3987e5;--aqua:#199e70;--red:#e66767;--violet:#9085e9;--noiseband:rgba(137,135,129,.16);--posband:rgba(25,158,112,.15);color-scheme:dark;}
:root[data-theme=light]{--surface:#fcfcfb;--plane:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);--blue:#2a78d6;--aqua:#1baf7a;--red:#e34948;--violet:#4a3aa7;--noiseband:rgba(137,135,129,.14);--posband:rgba(27,175,122,.12);color-scheme:light;}
*{box-sizing:border-box}body{margin:0;background:var(--plane);color:var(--ink);font:14px/1.5 system-ui,-apple-system,Arial,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:26px 20px 70px}h1{font-size:21px;margin:0 0 4px}
.sub{color:var(--ink2);font-size:14px;margin:0 0 20px;max-width:820px}.sub b{color:var(--ink)}
.panel{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:16px 18px 12px;margin:0 0 18px}
.panel h2{font-size:15.5px;margin:0 0 2px}.panel .note{color:var(--ink2);font-size:12.5px;margin:0 0 10px;max-width:840px}.panel .note b{color:var(--ink)}
.legend{display:flex;gap:16px;flex-wrap:wrap;color:var(--ink2);font-size:12.5px;margin:2px 0 8px}.legend span{display:inline-flex;align-items:center;gap:6px}
.sw{width:13px;height:11px;border-radius:2px;display:inline-block}.ln{width:16px;border-top-width:2.5px;border-top-style:solid;display:inline-block}
svg{display:block;width:100%;height:auto;overflow:visible}svg text{fill:var(--muted);font:11px system-ui,sans-serif}svg text.lab{fill:var(--ink);font-weight:600}
.tip{position:fixed;pointer-events:none;background:var(--ink);color:var(--surface);padding:6px 9px;border-radius:6px;font-size:12px;opacity:0;transition:opacity .08s;z-index:20;white-space:nowrap;font-variant-numeric:tabular-nums}
.foot{color:var(--muted);font-size:11.5px;margin-top:16px;line-height:1.6}
</style></head><body><div class="wrap">
<h1>Trailing IC — a real regime vs. noise dressed up as one</h1>
<p class="sub">Both panels show a signal's <b>recent effectiveness</b> (trailing IC of the signal vs next-day return).
The dynamic-switch idea flips the signal whenever this line crosses zero. It only works if the crossings are
<b>real regime shifts</b>. Grey band = <b>±2 SE</b> (statistically indistinguishable from zero).</p>

<div class="panel">
  <h2>1 · Cross-sectional reversion — flat noise around a negative mean (switch FAILS)</h2>
  <p class="note">The line hugs its mean (<b id="xm"></b>) and stays <b>inside the ±2SE noise band</b> — its excursions
  above zero (the "momentum zones", shaded green) are indistinguishable from chance. There's <b>no trend, no regime</b> —
  just a stable reversion measured with noise. Flipping to momentum on those crossings trades the wrong way + pays turnover
  → the book craters (651→553).</p>
  <div class="legend"><span><span class="ln" style="border-color:var(--violet)"></span>trailing IC</span>
    <span><span class="sw" style="background:var(--noiseband)"></span>±2 SE (≈ zero)</span>
    <span><span class="sw" style="background:var(--posband)"></span>"momentum zone" (IC&gt;0)</span></div>
  <svg id="xs"></svg>
</div>

<div class="panel">
  <h2>2 · ALGO vol — a genuine upward drift, persistently positive (switch WORKS)</h2>
  <p class="note">This line <b>trends upward</b> and stays positive — a real, sustained regime shift (the vol→return
  effect strengthened over time). Not per-window noise: it's consistently on one side, which is why sizing/holding the
  side it indicates <b>adds</b> edge. Same rule, opposite outcome — because this signal genuinely moves.</p>
  <div class="legend"><span><span class="ln" style="border-color:var(--aqua)"></span>trailing IC</span>
    <span><span class="sw" style="background:var(--noiseband)"></span>±2 SE</span></div>
  <svg id="vol"></svg>
</div>
<p class="foot">Cross-sectional IC = 120-day pooled correlation of the demeaned 10-day return with next-day return across
the 49 stocks; SE from the daily cross-sectional IC dispersion. Vol IC = 90-day trailing correlation of z-scored
realized vol with ALGO's next-day return. Built from prices.txt · build_trailingic.py.</p>
<div class="tip" id="tip"></div></div>
<script>
const D=__DATA__, SVGNS="http://www.w3.org/2000/svg";
const cssv=k=>getComputedStyle(document.documentElement).getPropertyValue(k).trim();
const tip=document.getElementById('tip');
function el(n,a){const e=document.createElementNS(SVGNS,n);for(const k in a)e.setAttribute(k,a[k]);return e;}
function clear(id){const s=document.getElementById(id);while(s.firstChild)s.removeChild(s.firstChild);return s;}
function chart(id,days,ic,se,center,col,poszone){
  const s=clear(id),W=s.clientWidth||1000,padL=52,padR=16,padT=12,padB=28,H=280;
  s.setAttribute('viewBox',`0 0 ${W} ${H}`);s.setAttribute('height',H);
  let lo=Math.min(-2*se+center,...ic),hi=Math.max(2*se+center,...ic);const pd=(hi-lo)*.1;lo-=pd;hi+=pd;
  const X=v=>padL+(v-days[0])/(days[days.length-1]-days[0])*(W-padL-padR),Y=v=>padT+(hi-v)/(hi-lo)*(H-padT-padB);
  // noise band around center (0 for vol, mean is drawn separately)
  s.appendChild(el('rect',{x:padL,y:Y(center+2*se),width:W-padL-padR,height:Y(center-2*se)-Y(center+2*se),fill:cssv('--noiseband')}));
  // positive (momentum) zone shading for cross-sectional
  if(poszone){let d=0;while(d<ic.length){if(ic[d]>0){let e=d;while(e+1<ic.length&&ic[e+1]>0)e++;
    s.appendChild(el('rect',{x:X(days[d]),y:padT,width:Math.max(1,X(days[e])-X(days[d])),height:H-padT-padB,fill:cssv('--posband')}));d=e+1;}else d++;}}
  // grid + y ticks
  const ys=[];for(let g=Math.ceil(lo*20)/20;g<=hi;g+=0.025)ys.push(g);
  ys.forEach(g=>{const y=Y(g);s.appendChild(el('line',{x1:padL,y1:y,x2:W-padR,y2:y,stroke:cssv('--grid'),'stroke-width':1}));
    const t=el('text',{x:padL-6,y:y+3,'text-anchor':'end'});t.textContent=g.toFixed(3);s.appendChild(t);});
  // zero line (bold)
  s.appendChild(el('line',{x1:padL,y1:Y(0),x2:W-padR,y2:Y(0),stroke:cssv('--axis'),'stroke-width':1.6}));
  const z=el('text',{x:W-padR,y:Y(0)-4,'text-anchor':'end',class:'lab'});z.textContent='0';s.appendChild(z);
  // mean line (dashed)
  if(poszone){s.appendChild(el('line',{x1:padL,y1:Y(center),x2:W-padR,y2:Y(center),stroke:cssv('--red'),'stroke-width':1.3,'stroke-dasharray':'5 4'}));
    const t=el('text',{x:padL+4,y:Y(center)-4,class:'lab'});t.textContent='mean '+center.toFixed(3);t.setAttribute('fill',cssv('--red'));s.appendChild(t);}
  // x ticks
  for(let dd=Math.ceil(days[0]/100)*100;dd<=days[days.length-1];dd+=100){const t=el('text',{x:X(dd),y:H-8,'text-anchor':'middle'});t.textContent=dd;s.appendChild(t);}
  s.appendChild(Object.assign(el('text',{x:(padL+W-padR)/2,y:H-0,'text-anchor':'middle'}),{textContent:'day'}));
  // the IC line
  const pts=days.map((d,i)=>X(d)+','+Y(ic[i])).join(' ');
  s.appendChild(el('polyline',{points:pts,fill:'none',stroke:cssv(col),'stroke-width':2,'stroke-linejoin':'round'}));
  // hover
  const cr=el('line',{x1:0,y1:padT,x2:0,y2:H-padB,stroke:cssv('--axis'),'stroke-width':1,opacity:0});s.appendChild(cr);
  const dot=el('circle',{r:4,fill:cssv(col),stroke:cssv('--surface'),'stroke-width':1.5,opacity:0});s.appendChild(dot);
  const hit=el('rect',{x:padL,y:padT,width:W-padL-padR,height:H-padT-padB,fill:'transparent'});s.appendChild(hit);
  hit.addEventListener('mousemove',e=>{const r=s.getBoundingClientRect(),sc=W/r.width,mx=(e.clientX-r.left)*sc;
    const dv=days[0]+(mx-padL)/(W-padL-padR)*(days[days.length-1]-days[0]);let idx=0,bd=1e9;days.forEach((d,i)=>{const q=Math.abs(d-dv);if(q<bd){bd=q;idx=i;}});
    cr.setAttribute('x1',X(days[idx]));cr.setAttribute('x2',X(days[idx]));cr.setAttribute('opacity',.6);
    dot.setAttribute('cx',X(days[idx]));dot.setAttribute('cy',Y(ic[idx]));dot.setAttribute('opacity',1);
    tip.innerHTML=`day ${days[idx]}<br>trailing IC <b>${ic[idx]>=0?'+':''}${ic[idx].toFixed(3)}</b>`;
    tip.style.left=(e.clientX+14)+'px';tip.style.top=(e.clientY+14)+'px';tip.style.opacity=1;});
  hit.addEventListener('mouseleave',()=>{cr.setAttribute('opacity',0);dot.setAttribute('opacity',0);tip.style.opacity=0;});
}
function draw(){document.getElementById('xm').textContent=D.xs.mean.toFixed(3);
  chart('xs',D.xs.days,D.xs.ic,D.xs.se,D.xs.mean,'--violet',true);
  chart('vol',D.vol.days,D.vol.ic,D.vol.se,0,'--aqua',false);}
draw();window.addEventListener('resize',draw);
new MutationObserver(draw).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
</script></body></html>"""
open("trailing_ic.html", "w").write(HTML.replace("__DATA__", json.dumps(D, separators=(",", ":"))))
print("wrote trailing_ic.html")
print(f"cross-sectional IC: mean {D['xs']['mean']}, SE {D['xs']['se']}, band +-{round(2*D['xs']['se'],3)}, "
      f"% days >0 = {round(100*np.mean(np.array(xsic)>0))}%")
print(f"vol IC: range [{round(min(volic),3)},{round(max(volic),3)}], starts {round(volic[0],3)} ends {round(volic[-1],3)}")
