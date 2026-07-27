"""Build signals.html — the ALGO vol-signal investigation, data inlined. Reads signals_data.json.
Panels: (1) cross-sectional vol->return IC across all 51 names, (2) per-name persistence scatter
(H1 vs H2), (3) live trailing IC of vol vs lead-lag over time, (4) combine head-to-head."""
import json
DATA = json.load(open("signals_data.json"))

HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ALGO signals — is the vol edge real?</title>
<style>
:root{
  --surface:#fcfcfb; --plane:#f9f9f7; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,.10);
  --blue:#2a78d6; --aqua:#1baf7a; --red:#e34948; --amber:#eda100; --violet:#4a3aa7;
  --posband:rgba(27,175,122,.13); --negband:rgba(227,73,72,.12); --vioband:rgba(74,58,167,.12);
  color-scheme:light;
}
@media (prefers-color-scheme:dark){:root{
  --surface:#1a1a19; --plane:#0d0d0d; --ink:#fff; --ink2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,.10);
  --blue:#3987e5; --aqua:#199e70; --red:#e66767; --amber:#c98500; --violet:#9085e9;
  --posband:rgba(25,158,112,.16); --negband:rgba(230,103,103,.15); --vioband:rgba(144,133,233,.16);
  color-scheme:dark;
}}
:root[data-theme=dark]{
  --surface:#1a1a19; --plane:#0d0d0d; --ink:#fff; --ink2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,.10);
  --blue:#3987e5; --aqua:#199e70; --red:#e66767; --amber:#c98500; --violet:#9085e9;
  --posband:rgba(25,158,112,.16); --negband:rgba(230,103,103,.15); --vioband:rgba(144,133,233,.16);
  color-scheme:dark;
}
:root[data-theme=light]{
  --surface:#fcfcfb; --plane:#f9f9f7; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,.10);
  --blue:#2a78d6; --aqua:#1baf7a; --red:#e34948; --amber:#eda100; --violet:#4a3aa7;
  --posband:rgba(27,175,122,.13); --negband:rgba(227,73,72,.12); --vioband:rgba(74,58,167,.12);
  color-scheme:light;
}
*{box-sizing:border-box}
body{margin:0;background:var(--plane);color:var(--ink);font:14px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:26px 20px 80px}
h1{font-size:22px;margin:0 0 4px;letter-spacing:-.01em}
.sub{color:var(--ink2);font-size:14px;margin:0 0 22px;max-width:840px} .sub b{color:var(--ink)}
.kpis{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:0 0 26px}
.kpi{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:15px 17px}
.kpi .lab{color:var(--ink2);font-size:12.5px;font-weight:600;margin-bottom:8px}
.kpi .big{font-size:28px;font-weight:700;font-variant-numeric:tabular-nums;letter-spacing:-.02em}
.kpi .sub2{color:var(--ink2);font-size:12px;margin-top:6px}
.pos{color:var(--aqua)} .neg{color:var(--red)} .vio{color:var(--violet)}
.panel{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:16px 18px 12px;margin:0 0 18px}
.panel h2{font-size:15.5px;margin:0 0 2px}
.panel .note{color:var(--ink2);font-size:12.5px;margin:0 0 12px;max-width:840px} .panel .note b{color:var(--ink)}
.legend{display:flex;gap:16px;flex-wrap:wrap;color:var(--ink2);font-size:12.5px;margin:2px 0 10px}
.legend span{display:inline-flex;align-items:center;gap:6px}
.sw{width:11px;height:11px;border-radius:3px;display:inline-block}
.ln{width:16px;height:0;border-top-width:2.5px;border-top-style:solid;display:inline-block}
svg{display:block;width:100%;height:auto;overflow:visible}
svg text{fill:var(--muted);font:11px system-ui,sans-serif}
svg text.lab{fill:var(--ink);font-weight:600} svg text.val{fill:var(--ink);font-variant-numeric:tabular-nums}
.tip{position:fixed;pointer-events:none;background:var(--ink);color:var(--surface);padding:7px 10px;border-radius:7px;
  font-size:12px;opacity:0;transition:opacity .08s;z-index:20;white-space:nowrap;font-variant-numeric:tabular-nums;box-shadow:0 4px 16px rgba(0,0,0,.22)}
.foot{color:var(--muted);font-size:11.5px;margin-top:18px;line-height:1.6}
@media (max-width:760px){.kpis{grid-template-columns:1fr}}
</style></head><body><div class="wrap">

<h1>ALGO index signals — is the vol edge real, or overfit?</h1>
<p class="sub">The realized-vol leg (<b>LLVOL</b>) works on ALGO. Two questions decide whether to trust it:
is the vol→next-return effect a <b>universal generator property</b> (should appear in all 51 names) or
<b>idiosyncratic to ALGO</b>? And does per-name significance <b>persist</b> out-of-sample (or is selecting
"significant" names just noise)? Verdict: vol is <b>not</b> universal and does <b>not</b> persist per-name —
but on ALGO it is stable, strengthening, and survives a shift-surrogate test. Use it on ALGO only.</p>

<div class="kpis" id="kpis"></div>

<div class="panel">
  <h2>1 · vol→next-return IC across all 51 instruments — is it universal?</h2>
  <p class="note">Each bar is one instrument's full-sample vol→return IC. If vol were a property of the
  generator, most bars would be positive and tall. Instead they scatter around zero (mean +0.004, 27/51
  positive) — <b>except ALGO</b> (violet), the clear standout. Signals here are instrument-specific: the
  <b>lead-lag</b> edge lives in the stocks and is absent on ALGO; the <b>vol</b> edge is the mirror image.</p>
  <div class="legend">
    <span><span class="sw" style="background:var(--violet)"></span>ALGO (index)</span>
    <span><span class="sw" style="background:var(--aqua)"></span>positive IC</span>
    <span><span class="sw" style="background:var(--red)"></span>negative IC</span>
  </div>
  <svg id="xsec"></svg>
</div>

<div class="panel">
  <h2>2 · Does per-name significance persist? (first half vs second half)</h2>
  <p class="note">x = each name's vol IC over days 1–500, y = over 501–1000. Real, name-specific structure
  would line up on the diagonal. Instead it's a <b>shapeless cloud (corr = <span id="pcorr"></span>)</b> — the
  "significant" names are random each half, so <b>selecting them = trading noise</b>. Only <b>ALGO</b> (violet,
  ringed) sits firmly in the win-both quadrant and persists.</p>
  <svg id="scatter"></svg>
</div>

<div class="panel">
  <h2>3 · Live trailing IC — what the adaptive gate sees</h2>
  <p class="note">250-day trailing IC (causal) of the <b>vol</b> signal vs the <b>lead-lag net-$</b> signal on ALGO,
  by day. The vol IC (violet) climbs and stays positive through the graded windows (shaded); lead-lag (blue)
  is weaker and noisier. LLVOL sizes the leg by the violet line, so it scales up exactly when the edge is live.</p>
  <div class="legend">
    <span><span class="ln" style="border-color:var(--violet)"></span>vol signal — live IC</span>
    <span><span class="ln" style="border-color:var(--blue)"></span>lead-lag net-$ — live IC</span>
  </div>
  <svg id="live"></svg>
</div>

<div class="panel">
  <h2>4 · Combining LLVOL + LLMATCH doesn't help</h2>
  <p class="note">Rolling-mean and rolling-floor score of the two legs alone vs combined (naive sum, and
  IC-weighted to prioritise the stronger). Adding lead-lag <b>drags LLVOL down</b> — vol alone is best on both.</p>
  <div class="legend">
    <span><span class="sw" style="background:var(--violet)"></span>rolling mean</span>
    <span><span class="sw" style="background:var(--aqua)"></span>rolling floor</span>
  </div>
  <svg id="combine"></svg>
</div>

<p class="foot">IC = Pearson correlation of the causal signal with next-day return. Shift-surrogate p from 4,000
circular shifts (preserves each series' autocorrelation, breaks their alignment). Scores are exact eval.py.
Built from signals_data.json · compute_signals.py.</p>

<div class="tip" id="tip"></div>
</div>
<script>
const DATA=__DATA__, SVGNS="http://www.w3.org/2000/svg";
const cssv=k=>getComputedStyle(document.documentElement).getPropertyValue(k).trim();
const tip=document.getElementById('tip');
function showTip(e,h){tip.innerHTML=h;tip.style.left=(e.clientX+14)+'px';tip.style.top=(e.clientY+14)+'px';tip.style.opacity=1;}
function hideTip(){tip.style.opacity=0;}
function el(n,a){const e=document.createElementNS(SVGNS,n);for(const k in a)e.setAttribute(k,a[k]);return e;}
function clear(id){const s=document.getElementById(id);while(s.firstChild)s.removeChild(s.firstChild);return s;}
const AI=DATA.algo_idx;

function renderKpis(){
  const a=DATA.algo, xf=DATA.xsec_summary.full;
  const tiles=[
    {lab:'ALGO vol→return IC (new window)', big:a.new_ic.toFixed(3), cls:'vio',
     sub:`shift-surrogate p ${a.shift_p_new<0.001?'< 0.001':'= '+a.shift_p_new} · persists every sub-period`},
    {lab:'Same effect across all 51 names', big:'+'+xf.mean.toFixed(3), cls:'',
     sub:`${xf.pos}/51 positive · cross-sectional t=${xf.tcross} — not universal`},
    {lab:'Per-name persistence (H1 vs H2)', big:DATA.persist.corr.toFixed(2), cls:'neg',
     sub:'≈0 ⇒ selecting "significant" stocks = trading noise'}];
  const box=document.getElementById('kpis');box.innerHTML='';
  for(const t of tiles){const d=document.createElement('div');d.className='kpi';
    d.innerHTML=`<div class="lab">${t.lab}</div><div class="big ${t.cls}">${t.big}</div><div class="sub2">${t.sub}</div>`;
    box.appendChild(d);}
}

// ---- Panel 1: cross-sectional bar ----
function renderXsec(){
  const s=clear('xsec'), ic=DATA.xsec.full, N=ic.length;
  const order=[...ic.keys()].sort((a,b)=>ic[b]-ic[a]);
  const W=s.clientWidth||1100, padL=40,padR=14,padT=12,padB=40, H=300;
  s.setAttribute('viewBox',`0 0 ${W} ${H}`); s.setAttribute('height',H);
  let lim=0; ic.forEach(v=>lim=Math.max(lim,Math.abs(v))); lim*=1.1;
  const bw=(W-padL-padR)/N, Y=v=>padT+(lim-v)/(2*lim)*(H-padT-padB), z=Y(0);
  for(const g of [-0.1,-0.05,0,0.05,0.1]){if(Math.abs(g)>lim)continue;const y=Y(g);
    s.appendChild(el('line',{x1:padL,y1:y,x2:W-padR,y2:y,stroke:g===0?cssv('--axis'):cssv('--grid'),'stroke-width':g===0?1.3:1}));
    const t=el('text',{x:padL-6,y:y+3,'text-anchor':'end'});t.textContent=g.toFixed(2);s.appendChild(t);}
  order.forEach((i,k)=>{const v=ic[i], x=padL+k*bw, isA=(i===AI);
    const col=isA?'--violet':(v>=0?'--aqua':'--red');
    const rect=el('rect',{x:x+bw*0.12,y:Math.min(z,Y(v)),width:Math.max(1,bw*0.76),height:Math.abs(Y(v)-z),
      fill:cssv(col),rx:1}); rect.style.cursor='pointer';
    rect.addEventListener('mousemove',e=>showTip(e,`<b>${DATA.names[i]}</b>${isA?' (ALGO index)':''}<br>vol IC = ${v>=0?'+':''}${v.toFixed(3)}`));
    rect.addEventListener('mouseleave',hideTip); s.appendChild(rect);
    if(isA){const t=el('text',{x:x+bw/2,y:Y(v)-6,'text-anchor':'middle',class:'val'});t.textContent='ALGO';t.setAttribute('fill',cssv('--violet'));s.appendChild(t);}});
  const xl=el('text',{x:(padL+W-padR)/2,y:H-6,'text-anchor':'middle'});xl.textContent='51 instruments, sorted by vol→return IC';s.appendChild(xl);
}

// ---- Panel 2: persistence scatter ----
function renderScatter(){
  const s=clear('scatter'), h1=DATA.persist.h1, h2=DATA.persist.h2, N=h1.length;
  const W=s.clientWidth||1100, pad=46, H=420;
  s.setAttribute('viewBox',`0 0 ${W} ${H}`); s.setAttribute('height',H);
  let lim=0.02; for(let i=0;i<N;i++)lim=Math.max(lim,Math.abs(h1[i]),Math.abs(h2[i])); lim*=1.1;
  const X=v=>pad+(v+lim)/(2*lim)*(W-2*pad), Y=v=>pad+(lim-v)/(2*lim)*(H-2*pad);
  // quadrant tints
  s.appendChild(el('rect',{x:X(0),y:Y(lim),width:X(lim)-X(0),height:Y(0)-Y(lim),fill:cssv('--posband')}));
  s.appendChild(el('rect',{x:X(-lim),y:Y(0),width:X(0)-X(-lim),height:Y(-lim)-Y(0),fill:cssv('--negband')}));
  // axes + diagonal
  s.appendChild(el('line',{x1:X(0),y1:pad,x2:X(0),y2:H-pad,stroke:cssv('--axis'),'stroke-width':1}));
  s.appendChild(el('line',{x1:pad,y1:Y(0),x2:W-pad,y2:Y(0),stroke:cssv('--axis'),'stroke-width':1}));
  const d=el('line',{x1:X(-lim),y1:Y(-lim),x2:X(lim),y2:Y(lim),stroke:cssv('--muted'),'stroke-width':1,'stroke-dasharray':'4 4'});s.appendChild(d);
  for(const g of [-0.1,0,0.1]){if(Math.abs(g)>lim)continue;
    let t=el('text',{x:X(g),y:H-pad+16,'text-anchor':'middle'});t.textContent=g.toFixed(2);s.appendChild(t);
    let u=el('text',{x:pad-8,y:Y(g)+3,'text-anchor':'end'});u.textContent=g.toFixed(2);s.appendChild(u);}
  s.appendChild(Object.assign(el('text',{x:W/2,y:H-8,'text-anchor':'middle'}),{textContent:'vol IC · days 1–500  →'}));
  const yl=el('text',{x:14,y:H/2,'text-anchor':'middle'});yl.setAttribute('transform',`rotate(-90 14 ${H/2})`);yl.textContent='vol IC · days 501–1000  →';s.appendChild(yl);
  s.appendChild(Object.assign(el('text',{x:X(lim)-6,y:Y(lim)+14,'text-anchor':'end',class:'lab'}),{textContent:'persist ✓'}));
  for(let i=0;i<N;i++){const isA=(i===AI);
    const c=el('circle',{cx:X(h1[i]),cy:Y(h2[i]),r:isA?7:4.5,fill:cssv(isA?'--violet':'--muted'),
      stroke:cssv('--surface'),'stroke-width':isA?2:1,opacity:isA?1:.7}); c.style.cursor='pointer';
    c.addEventListener('mousemove',e=>showTip(e,`<b>${DATA.names[i]}</b>${isA?' (ALGO)':''}<br>H1 ${h1[i]>=0?'+':''}${h1[i].toFixed(3)} · H2 ${h2[i]>=0?'+':''}${h2[i].toFixed(3)}`));
    c.addEventListener('mouseleave',hideTip); s.appendChild(c);
    if(isA){const t=el('text',{x:X(h1[i])+11,y:Y(h2[i])-6,class:'val'});t.textContent='ALGO';t.setAttribute('fill',cssv('--violet'));s.appendChild(t);}}
  document.getElementById('pcorr').textContent=DATA.persist.corr.toFixed(2);
}

// ---- Panel 3: live IC line chart ----
function renderLive(){
  const s=clear('live'), L=DATA.live, ed=L.days, W=s.clientWidth||1100, padL=54,padR=52,padT=14,padB=30, H=300;
  s.setAttribute('viewBox',`0 0 ${W} ${H}`); s.setAttribute('height',H);
  let lo=Infinity,hi=-Infinity;[L.icv,L.icm].forEach(a=>a.forEach(v=>{lo=Math.min(lo,v);hi=Math.max(hi,v);}));
  lo=Math.min(lo,-0.02); hi=Math.max(hi,0.02); const pd=(hi-lo)*.08; lo-=pd; hi+=pd;
  const X=v=>padL+(v-ed[0])/(ed[ed.length-1]-ed[0])*(W-padL-padR), Y=v=>padT+(hi-v)/(hi-lo)*(H-padT-padB);
  for(const b of [DATA.windows.OLD,DATA.windows.NEW]){s.appendChild(el('rect',{x:X(b[0]),y:padT,
    width:Math.max(1,X(b[1])-X(b[0])),height:H-padT-padB,fill:cssv('--vioband')}));}
  for(let g=Math.ceil(lo*20)/20; g<=hi; g+=0.05){const y=Y(g);
    s.appendChild(el('line',{x1:padL,y1:y,x2:W-padR,y2:y,stroke:Math.abs(g)<1e-9?cssv('--axis'):cssv('--grid'),'stroke-width':Math.abs(g)<1e-9?1.3:1}));
    const t=el('text',{x:padL-8,y:y+3,'text-anchor':'end'});t.textContent=g.toFixed(2);s.appendChild(t);}
  for(let d=Math.ceil(ed[0]/100)*100; d<=ed[ed.length-1]; d+=100){const t=el('text',{x:X(d),y:H-padB+16,'text-anchor':'middle'});t.textContent=d;s.appendChild(t);}
  s.appendChild(Object.assign(el('text',{x:(padL+W-padR)/2,y:H-2,'text-anchor':'middle'}),{textContent:'day (end of 250-day trailing IC window)'}));
  const series=[{y:L.icm,c:'--blue',nm:'lead-lag',lab:'lead-lag'},{y:L.icv,c:'--violet',nm:'vol',lab:'vol'}];
  for(const se of series){const pts=ed.map((x,i)=>X(x)+','+Y(se.y[i])).join(' ');
    s.appendChild(el('polyline',{points:pts,fill:'none',stroke:cssv(se.c),'stroke-width':se.nm==='vol'?2.6:2,'stroke-linejoin':'round'}));
    const t=el('text',{x:X(ed[ed.length-1])+6,y:Y(se.y[se.y.length-1])+3,class:'val'});t.textContent=se.lab;t.setAttribute('fill',cssv(se.c));s.appendChild(t);}
  // crosshair
  const cr=el('line',{x1:0,y1:padT,x2:0,y2:H-padB,stroke:cssv('--axis'),'stroke-width':1,opacity:0});s.appendChild(cr);
  const dv=series.map(se=>{const c=el('circle',{r:4,fill:cssv(se.c),stroke:cssv('--surface'),'stroke-width':1.5,opacity:0});s.appendChild(c);return c;});
  const hit=el('rect',{x:padL,y:padT,width:W-padL-padR,height:H-padT-padB,fill:'transparent'});s.appendChild(hit);
  hit.addEventListener('mousemove',e=>{const r=s.getBoundingClientRect(),sc=W/r.width,mx=(e.clientX-r.left)*sc;
    const xv=ed[0]+(mx-padL)/(W-padL-padR)*(ed[ed.length-1]-ed[0]);
    let idx=0,bd=1e18;ed.forEach((x,i)=>{const dd=Math.abs(x-xv);if(dd<bd){bd=dd;idx=i;}});
    cr.setAttribute('x1',X(ed[idx]));cr.setAttribute('x2',X(ed[idx]));cr.setAttribute('opacity',.6);
    let h=`<b>day ${ed[idx]}</b>`;series.forEach((se,k)=>{dv[k].setAttribute('cx',X(ed[idx]));dv[k].setAttribute('cy',Y(se.y[idx]));dv[k].setAttribute('opacity',1);
      h+=`<br><span style="color:${cssv(se.c)}">■</span> ${se.nm} IC ${se.y[idx]>=0?'+':''}${se.y[idx].toFixed(3)}`;});showTip(e,h);});
  hit.addEventListener('mouseleave',()=>{cr.setAttribute('opacity',0);dv.forEach(d=>d.setAttribute('opacity',0));hideTip();});
}

// ---- Panel 4: combine grouped bars ----
function renderCombine(){
  const s=clear('combine'), C=DATA.combine, W=s.clientWidth||1100, padL=130,padR=60,padT=10,padB=26;
  const rowH=52, H=padT+padB+C.length*rowH; s.setAttribute('viewBox',`0 0 ${W} ${H}`); s.setAttribute('height',H);
  let lo=1e9,hi=-1e9; C.forEach(c=>{lo=Math.min(lo,c.floor);hi=Math.max(hi,c.mean);});
  lo=Math.floor((lo-30)/50)*50; hi=Math.ceil((hi+20)/50)*50;
  const X=v=>padL+(v-lo)/(hi-lo)*(W-padL-padR);
  for(let g=lo;g<=hi;g+=50){const x=X(g);s.appendChild(el('line',{x1:x,y1:padT,x2:x,y2:H-padB,stroke:cssv('--grid'),'stroke-width':1}));
    const t=el('text',{x,y:H-padB+16,'text-anchor':'middle'});t.textContent=g;s.appendChild(t);}
  C.forEach((c,i)=>{const y=padT+i*rowH+6, isV=(c.nm==='LLVOL only');
    const lab=el('text',{x:padL-12,y:y+rowH/2-2,'text-anchor':'end',class:'lab'});lab.textContent=c.nm;
    if(isV)lab.setAttribute('fill',cssv('--violet'));s.appendChild(lab);
    [['mean','--violet',c.mean],['floor','--aqua',c.floor]].forEach((b,j)=>{
      const yy=y+j*15, w=Math.max(1,X(b[2])-padL);
      const rect=el('rect',{x:padL,y:yy,width:w,height:11,rx:3,fill:cssv(b[1]),opacity:isV?1:.62});rect.style.cursor='pointer';
      rect.addEventListener('mousemove',e=>showTip(e,`<b>${c.nm}</b><br>rolling ${b[0]} = ${b[2].toFixed(0)} · OLD ${c.OLD} · NEW ${c.NEW}`));
      rect.addEventListener('mouseleave',hideTip);s.appendChild(rect);
      const t=el('text',{x:X(b[2])+6,y:yy+9,class:'val'});t.textContent=b[2].toFixed(0);s.appendChild(t);});
  });
}

function renderAll(){renderKpis();renderXsec();renderScatter();renderLive();renderCombine();}
renderAll();
let rt;window.addEventListener('resize',()=>{clearTimeout(rt);rt=setTimeout(renderAll,120);});
new MutationObserver(renderAll).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
</script></body></html>"""

open("signals.html", "w").write(HTML.replace("__DATA__", json.dumps(DATA, separators=(",", ":"))))
print("wrote signals.html", f"({len(HTML)//1024} KB base)")
