"""Build a self-contained diagnostics dashboard (data inlined) that explains what happened
when prices.txt grew from 750 -> 1000 days: the graded window moved to a fresh draw, the idio
book held, and the ALGO index leg flipped from tailwind to headwind. Reads diag_data.json."""
import json
DATA = json.load(open("diag_data.json"))

HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Arbitrage Victims — what changed at 1000 days</title>
<style>
:root{
  --surface:#fcfcfb; --plane:#f9f9f7; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,.10);
  --blue:#2a78d6; --aqua:#1baf7a; --red:#e34948; --amber:#eda100; --violet:#4a3aa7; --green:#2f9e44; --orange:#e0692c; --teal:#0e8f8f; --magenta:#b0348a; --lime:#6a9e1f; --brown:#8a5a3c; --rose:#b0203f; --steel:#3d6d99; --coral:#c9573a; --gold:#a67c00; --cobalt:#3355c0; --umber:#b0501f; --jade:#1f8f6a; --purple:#7a2fb0; --berry:#9a2f55; --indigo:#5a3f9a; --cerulean:#1a7fab; --olive:#8a7a1f;
  --posband:rgba(27,175,122,.13); --negband:rgba(227,73,72,.12); --blueband:rgba(42,120,214,.10);
  color-scheme:light;
}
@media (prefers-color-scheme:dark){:root{
  --surface:#1a1a19; --plane:#0d0d0d; --ink:#fff; --ink2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,.10);
  --blue:#3987e5; --aqua:#199e70; --red:#e66767; --amber:#c98500; --violet:#9085e9; --green:#5cbf63; --orange:#e8813f; --teal:#2bb3b3; --magenta:#d456ac; --lime:#8ec93f; --brown:#b07a58; --rose:#e0507a; --steel:#6fa3d6; --coral:#e8896a; --gold:#d9ad33; --cobalt:#6f85e0; --umber:#cf6a28; --jade:#22bf8a; --purple:#a855e0; --berry:#d94f7a; --indigo:#8f70e0; --cerulean:#3fb0d9; --olive:#c9ad3f;
  --posband:rgba(25,158,112,.16); --negband:rgba(230,103,103,.15); --blueband:rgba(57,135,229,.14);
  color-scheme:dark;
}}
:root[data-theme=dark]{
  --surface:#1a1a19; --plane:#0d0d0d; --ink:#fff; --ink2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,.10);
  --blue:#3987e5; --aqua:#199e70; --red:#e66767; --amber:#c98500; --violet:#9085e9; --green:#5cbf63; --orange:#e8813f; --teal:#2bb3b3; --magenta:#d456ac; --lime:#8ec93f; --brown:#b07a58; --rose:#e0507a; --steel:#6fa3d6; --coral:#e8896a; --gold:#d9ad33; --cobalt:#6f85e0; --umber:#cf6a28; --jade:#22bf8a; --purple:#a855e0; --berry:#d94f7a; --indigo:#8f70e0; --cerulean:#3fb0d9; --olive:#c9ad3f;
  --posband:rgba(25,158,112,.16); --negband:rgba(230,103,103,.15); --blueband:rgba(57,135,229,.14);
  color-scheme:dark;
}
:root[data-theme=light]{
  --surface:#fcfcfb; --plane:#f9f9f7; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,.10);
  --blue:#2a78d6; --aqua:#1baf7a; --red:#e34948; --amber:#eda100; --violet:#4a3aa7; --green:#2f9e44; --orange:#e0692c; --teal:#0e8f8f; --magenta:#b0348a; --lime:#6a9e1f; --brown:#8a5a3c; --rose:#b0203f; --steel:#3d6d99; --coral:#c9573a; --gold:#a67c00; --cobalt:#3355c0; --umber:#b0501f; --jade:#1f8f6a; --purple:#7a2fb0; --berry:#9a2f55; --indigo:#5a3f9a; --cerulean:#1a7fab; --olive:#8a7a1f;
  --posband:rgba(27,175,122,.13); --negband:rgba(227,73,72,.12); --blueband:rgba(42,120,214,.10);
  color-scheme:light;
}
*{box-sizing:border-box}
body{margin:0;background:var(--plane);color:var(--ink);
  font:14px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.viz-root{max-width:1180px;margin:0 auto;padding:26px 20px 80px}
h1{font-size:22px;margin:0 0 4px;letter-spacing:-.01em}
.sub{color:var(--ink2);font-size:14px;margin:0 0 22px;max-width:800px}
.sub b{color:var(--ink)}
.kpis{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:0 0 26px}
.kpi{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:15px 17px}
.kpi .lab{color:var(--ink2);font-size:12.5px;font-weight:600;margin-bottom:9px}
.kpi .big{font-size:15px;font-weight:600;display:flex;align-items:baseline;gap:8px;flex-wrap:wrap}
.kpi .from{color:var(--muted);font-variant-numeric:tabular-nums}
.kpi .arrow{color:var(--muted);font-size:14px}
.kpi .to{font-size:27px;font-weight:700;font-variant-numeric:tabular-nums;letter-spacing:-.02em}
.kpi .delta{font-size:12.5px;font-weight:600;margin-top:7px;font-variant-numeric:tabular-nums}
.pos{color:var(--aqua)} .neg{color:var(--red)}
.panel{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:16px 18px 12px;margin:0 0 18px}
.panel h2{font-size:15.5px;margin:0 0 2px}
.panel .note{color:var(--ink2);font-size:12.5px;margin:0 0 12px;max-width:820px}
.panel .note b{color:var(--ink)}
.legend{display:flex;gap:16px;flex-wrap:wrap;color:var(--ink2);font-size:12.5px;margin:2px 0 10px}
.legend span{display:inline-flex;align-items:center;gap:6px}
.dot{width:11px;height:11px;border-radius:50%;display:inline-block}
.ln{width:16px;height:0;border-top-width:2.5px;border-top-style:solid;display:inline-block}
.toolbar{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:0 0 8px}
.seg{display:inline-flex;flex-wrap:wrap;border:1px solid var(--border);border-radius:8px;overflow:hidden}
.seg button{border:0;background:transparent;color:var(--ink2);padding:6px 13px;font-weight:600;cursor:pointer;font-size:12.5px}
.seg button.on{background:var(--ink);color:var(--surface)}
svg{display:block;width:100%;height:auto;overflow:visible}
svg text{fill:var(--muted);font:11px system-ui,sans-serif}
svg text.lab{fill:var(--ink);font-weight:600}
svg text.val{fill:var(--ink);font-variant-numeric:tabular-nums}
.tip{position:fixed;pointer-events:none;background:var(--ink);color:var(--surface);
  padding:7px 10px;border-radius:7px;font-size:12px;opacity:0;transition:opacity .08s;z-index:20;
  white-space:nowrap;font-variant-numeric:tabular-nums;box-shadow:0 4px 16px rgba(0,0,0,.22)}
.foot{color:var(--muted);font-size:11.5px;margin-top:18px;line-height:1.6}
@media (max-width:760px){.kpis{grid-template-columns:1fr}}
</style></head><body><div class="viz-root">

<h1>Arbitrage Victims — what changed when prices grew to __NT__ days</h1>
<p class="sub">The grader scores the <b>last 250 days</b>, so the graded window moved from
days <b>501–750</b> to a fresh draw at days <b>751–1000</b>. Nothing broke: the 49-name idio book
is unchanged. <b>The entire score drop is the ALGO index leg (instrument 0)</b>, which flipped from a
tailwind to a headwind on the new draw. Every number below is the exact <code>eval.py</code> score.</p>

<div class="kpis" id="kpis"></div>

<div class="panel">
  <h2>1 · Every book fell on the fresh draw — except the one with no ALGO leg</h2>
  <p class="note">Each row is one strategy's exact graded score. Left dot = <b>old</b> window (501–750),
  right dot = <b>new</b> window (751–1000). The <b>idio book with the ALGO leg switched off</b> barely moves
  (585 → 586) — proof the drop is the overlay, not the edge.</p>
  <div class="legend">
    <span><span class="dot" style="background:var(--muted)"></span>old window 501–750</span>
    <span><span class="dot" style="background:var(--blue)"></span>new window 751–1000</span>
    <span><span class="ln" style="border-color:var(--red)"></span>score fell</span>
    <span><span class="ln" style="border-color:var(--aqua)"></span>score held / rose</span>
  </div>
  <svg id="dumbbell"></svg>
</div>

<div class="panel">
  <h2>2 · The idio book held; the ALGO leg flipped sign</h2>
  <p class="note">Total PnL attributed to each leg over each window (shipped LLALGO book). The idio
  book earned <b>~+$160k in both</b> windows — a stable edge. The ALGO index leg swung from
  <b class="pos">+$28k</b> to <b class="neg">−$29k</b>: a ~$58k reversal that also inflated daily
  volatility, hitting the Sharpe term of the score twice.</p>
  <svg id="legbars"></svg>
</div>

<div class="panel">
  <h2>3 · Rolling 250-day score — the graded window is just one draw</h2>
  <p class="note">Exact score of a 250-day window ending on each day. <b>LLVOL</b> (adaptive realized-vol leg)
  sits highest across the board — best rolling mean <b>and</b> floor — but leans on a vol→return effect that is
  likely specific to the synthetic generator (a possible risk-premium; its causal gate self-disables if the
  effect is absent). The <b>shipped binary gate</b> (LLALGO) has the best history but its <b>worst</b> draw is the
  one now graded; <b>LLMATCH</b> (volume-matched lead-lag) and <b>idio-only</b> are the no-assumptions baselines.
  Shaded = the two graded windows.</p>
  <div class="legend">
    <span><span class="ln" style="border-color:var(--violet)"></span>LLVOL — adaptive vol leg</span>
    <span><span class="ln" style="border-color:var(--blue)"></span>LLALGO — shipped binary gate</span>
    <span><span class="ln" style="border-color:var(--amber)"></span>LLMATCH k=__MK__ — volume-matched</span>
    <span><span class="ln" style="border-color:var(--aqua)"></span>idio only — leg off</span>
  </div>
  <svg id="rolling"></svg>
</div>

<div class="panel">
  <h2>4 · Cumulative PnL by leg over the graded window</h2>
  <p class="note">Where the money is made and lost, day by day. On the <b>new</b> window the ALGO line
  drifts steadily <b class="neg">down</b> while the idio line climbs — the leg is a pure drag on this draw.</p>
  <div class="toolbar">
    <div class="seg" id="cumwin"><button data-w="NEW" class="on">new 751–1000</button><button data-w="OLD">old 501–750</button></div>
    <div class="seg" id="cumstrat"><button data-s="LLALGO" class="on">LLALGO (lead-lag)</button><button data-s="SAFE">SAFE (reversion)</button><button data-s="LLBOOST">LLBOOST (pairwise boost)</button><button data-s="LLBOOST_V2">LLBOOST v2 (adaptive mom)</button><button data-s="LLBOOST_V3">LLBOOST v3 (candidate pool)</button><button data-s="LLBOOST_V4">LLBOOST v4 (v2+v3)</button><button data-s="LLBOOST_V5">LLBOOST v5 (v3 refined)</button><button data-s="LLBOOST_V6">LLBOOST v6 (v2+v5)</button><button data-s="LLBOOST_V7">LLBOOST v7 (retuned COMBINE_GAIN)</button><button data-s="LLBOOST_V8">LLBOOST v8 (ALGO deadband)</button><button data-s="LLBOOST_V9">LLBOOST v9 (beta-demean)</button><button data-s="LLBOOST_V10">LLBOOST v10 (rank-stability)</button><button data-s="LLBOOST_V11">LLBOOST v11 (idio kill switch)</button><button data-s="LLBOOST_V12">LLBOOST v12 (v11 + post-jump fade, shipped)</button><button data-s="LLBOOST_V13">LLBOOST v13 (v11 + gated boost fallback, experimental)</button><button data-s="LLBOOST_V14">LLBOOST v14 (v11+v12+v13 + momentum/xsac insurance, experimental)</button><button data-s="LLBOOST_V15">LLBOOST v15 (v12 + insurance, no v13, recommended)</button><button data-s="LLBOOST_V16">LLBOOST v16 (v15 + IC/t-stat gate)</button><button data-s="LLBOOST_V17">LLBOOST v17 (v15 + dual EW IC gate)</button><button data-s="LLBOOST_V18">LLBOOST v18 (v17 + Bonferroni-corrected IC gate)</button></div>
  </div>
  <div class="legend">
    <span><span class="ln" style="border-color:var(--ink)"></span>total</span>
    <span><span class="ln" style="border-color:var(--aqua)"></span>idio book (49 names)</span>
    <span><span class="ln" style="border-color:var(--red)"></span>ALGO index leg</span>
  </div>
  <svg id="cum"></svg>
</div>

<div class="panel">
  <h2>5 · MATCH_K sweep — how big should the volume-matched index leg be?</h2>
  <p class="note">Score vs the size multiplier <b>k</b> (index&nbsp;$ = k × the book's net-$ tilt). <b>k=0 = leg off.</b>
  Raising k trades <b>old-window</b> score for <b>new-window</b> score and lowers the rolling <b>floor</b>
  (worst of 61 draws) — the robustness metric. The floor peaks at k≈0 and the rolling mean at k≈1, so
  <b>k=1 (the 1:1 match, shipped in SAFE_llmatch)</b> is the robust pick; k=2 chases the current new draw.</p>
  <div class="legend">
    <span><span class="ln" style="border-color:var(--blue)"></span>new 751–1000</span>
    <span><span class="ln" style="border-color:var(--amber)"></span>old 501–750</span>
    <span><span class="ln" style="border-color:var(--aqua)"></span>rolling floor (worst of 61)</span>
  </div>
  <svg id="ksweep"></svg>
</div>

<p class="foot">Scores use the official accounting: 250-day test window, per-day commission lagged one day,
integer share clip to the dollar limits, score = mean · SR²/(SR²+1) with SR annualised. Leg attribution
splits each day's PnL into instrument 0 (ALGO) vs instruments 1–49 (idio) and sums exactly to the total.
Built from diag_data.json · compute_diagnostics.py.</p>

<div class="tip" id="tip"></div>
</div>
<script>
const DATA = __DATA__;
const MK = DATA.ksweep.match_k;               // SAFE_llmatch.MATCH_K (labels track it)
const SVGNS="http://www.w3.org/2000/svg";
const cssv=k=>getComputedStyle(document.documentElement).getPropertyValue(k).trim();
const tip=document.getElementById('tip');
function showTip(e,html){tip.innerHTML=html;tip.style.left=(e.clientX+14)+'px';tip.style.top=(e.clientY+14)+'px';tip.style.opacity=1;}
function hideTip(){tip.style.opacity=0;}
function el(n,a){const e=document.createElementNS(SVGNS,n);for(const k in a)e.setAttribute(k,a[k]);return e;}
function fmtK(v){const s=v<0?'-':'';v=Math.abs(v);return s+'$'+(v>=1000?(v/1000).toFixed(v>=10000?0:1)+'k':v.toFixed(0));}
function clear(id){const s=document.getElementById(id);while(s.firstChild)s.removeChild(s.firstChild);return s;}

// ---------- KPI tiles ----------
function renderKpis(){
  const H=DATA.headline, IO=DATA.idio_only, ship=H.LLALGO;
  const tiles=[
    {lab:'Shipped book score (LLALGO)', from:ship.OLD.score, to:ship.NEW.score, fmt:x=>x.toFixed(0), inv:false},
    {lab:'Idio book only — ALGO leg off', from:IO.OLD.score, to:IO.NEW.score, fmt:x=>x.toFixed(0), inv:false},
    {lab:'ALGO index leg — total PnL', from:ship.OLD.algo, to:ship.NEW.algo, fmt:fmtK, inv:false}];
  const box=document.getElementById('kpis');box.innerHTML='';
  for(const t of tiles){
    const d=t.to-t.from, good=d>=0;
    const div=document.createElement('div');div.className='kpi';
    div.innerHTML=`<div class="lab">${t.lab}</div>
      <div class="big"><span class="from">${t.fmt(t.from)}</span><span class="arrow">→</span>
      <span class="to ${good?'pos':'neg'}">${t.fmt(t.to)}</span></div>
      <div class="delta ${good?'pos':'neg'}">${good?'▲':'▼'} ${t.fmt(Math.abs(d)).replace('$','Δ$').replace(/^(\d)/,'Δ$1')} on the new draw</div>`;
    box.appendChild(div);
  }
}

// ---------- Panel 1: dumbbell ----------
function renderDumbbell(){
  const s=clear('dumbbell');
  const rows=[...DATA.strategies.map(k=>({nm:k==='LLMATCH'?'LLMATCH k='+MK:k,o:DATA.headline[k].OLD.score,n:DATA.headline[k].NEW.score,hi:(k==='LLMATCH'||k==='LLVOL'||k==='LLBOOST'||k==='LLBOOST_V2'||k==='LLBOOST_V3'||k==='LLBOOST_V4'||k==='LLBOOST_V5'||k==='LLBOOST_V6'||k==='LLBOOST_V7'||k==='LLBOOST_V8'||k==='LLBOOST_V9'||k==='LLBOOST_V10'||k==='LLBOOST_V11'||k==='LLBOOST_V12'||k==='LLBOOST_V13'||k==='LLBOOST_V14'||k==='LLBOOST_V15'||k==='LLBOOST_V16'||k==='LLBOOST_V17'||k==='LLBOOST_V18')})),
              {nm:'idio only',o:DATA.idio_only.OLD.score,n:DATA.idio_only.NEW.score,hi:true}];
  const W=s.clientWidth||1100, rowH=42, padT=16, padB=28, padL=118, padR=54;
  const H=padT+padB+rows.length*rowH; s.setAttribute('viewBox',`0 0 ${W} ${H}`); s.setAttribute('height',H);
  let lo=Infinity,hi=-Infinity; for(const r of rows){lo=Math.min(lo,r.o,r.n);hi=Math.max(hi,r.o,r.n);}
  lo=Math.floor((lo-30)/50)*50; hi=Math.ceil((hi+30)/50)*50;
  const X=v=>padL+(v-lo)/(hi-lo)*(W-padL-padR);
  for(let g=lo;g<=hi;g+=50){const x=X(g);
    s.appendChild(el('line',{x1:x,y1:padT-6,x2:x,y2:H-padB,stroke:cssv('--grid'),'stroke-width':1}));
    const t=el('text',{x,y:H-padB+16,'text-anchor':'middle'});t.textContent=g;s.appendChild(t);}
  rows.forEach((r,i)=>{
    const y=padT+i*rowH+rowH/2, xo=X(r.o), xn=X(r.n), fell=r.n<r.o;
    if(r.hi)s.appendChild(el('rect',{x:2,y:padT+i*rowH+3,width:W-4,height:rowH-6,rx:7,fill:cssv('--blueband')}));
    const lab=el('text',{x:padL-14,y:y+4,'text-anchor':'end',class:'lab'});lab.textContent=r.nm;s.appendChild(lab);
    s.appendChild(el('line',{x1:xo,y1:y,x2:xn,y2:y,stroke:cssv(fell?'--red':'--aqua'),'stroke-width':2.5,'stroke-linecap':'round'}));
    const mk=(x,col,val,side)=>{const c=el('circle',{cx:x,cy:y,r:6.5,fill:cssv(col),stroke:cssv('--surface'),'stroke-width':2});
      c.style.cursor='pointer';
      c.addEventListener('mousemove',e=>showTip(e,`<b>${r.nm}</b> · ${side}<br>score <b>${val.toFixed(1)}</b>`));
      c.addEventListener('mouseleave',hideTip);s.appendChild(c);};
    mk(xo,'--muted',r.o,'old 501–750'); mk(xn,'--blue',r.n,'new 751–1000');
    const dv=el('text',{x:xn+(xn>=xo?12:-12),y:y+4,'text-anchor':xn>=xo?'start':'end',class:'val'});
    dv.textContent=r.n.toFixed(0); dv.setAttribute('fill',cssv(fell?'--red':'--aqua')); s.appendChild(dv);
  });
}

// ---------- Panel 2: leg attribution diverging bars ----------
function renderLegbars(){
  const s=clear('legbars');
  const sh=DATA.headline.LLALGO;
  const items=[{nm:'idio book · old 501–750',v:sh.OLD.idio},{nm:'idio book · new 751–1000',v:sh.NEW.idio},
               {nm:'ALGO leg · old 501–750',v:sh.OLD.algo},{nm:'ALGO leg · new 751–1000',v:sh.NEW.algo}];
  const W=s.clientWidth||1100, rowH=46, padT=10, padB=28, padL=168, padR=70;
  const H=padT+padB+items.length*rowH; s.setAttribute('viewBox',`0 0 ${W} ${H}`); s.setAttribute('height',H);
  let mx=0; for(const it of items)mx=Math.max(mx,Math.abs(it.v)); mx=Math.ceil(mx/20000)*20000;
  const X=v=>padL+(v+mx)/(2*mx)*(W-padL-padR), zero=X(0);
  for(let g=-mx;g<=mx;g+=40000){const x=X(g);
    s.appendChild(el('line',{x1:x,y1:padT,x2:x,y2:H-padB,stroke:cssv('--grid'),'stroke-width':1}));
    const t=el('text',{x,y:H-padB+16,'text-anchor':'middle'});t.textContent=fmtK(g);s.appendChild(t);}
  s.appendChild(el('line',{x1:zero,y1:padT,x2:zero,y2:H-padB,stroke:cssv('--axis'),'stroke-width':1.5}));
  items.forEach((it,i)=>{
    const y=padT+i*rowH+8, h=rowH-20, pos=it.v>=0, x0=pos?zero:X(it.v), w=Math.abs(X(it.v)-zero);
    const rect=el('rect',{x:x0,y,width:Math.max(1,w),height:h,rx:4,fill:cssv(pos?'--aqua':'--red')});
    rect.style.cursor='pointer';
    rect.addEventListener('mousemove',e=>showTip(e,`<b>${it.nm}</b><br>${it.v>=0?'+':''}${fmtK(it.v)}`));
    rect.addEventListener('mouseleave',hideTip); s.appendChild(rect);
    const lab=el('text',{x:padL-14,y:y+h/2+4,'text-anchor':'end',class:'lab'});lab.textContent=it.nm;s.appendChild(lab);
    const vx=pos?X(it.v)+9:X(it.v)-9;
    const dv=el('text',{x:vx,y:y+h/2+4,'text-anchor':pos?'start':'end',class:'val'});
    dv.textContent=(it.v>=0?'+':'')+fmtK(it.v); dv.setAttribute('fill',cssv(pos?'--aqua':'--red')); s.appendChild(dv);
  });
}

// ---------- generic line-chart plumbing ----------
function lineChart(id, series, xdom, ydom, xlab, opts){
  opts=opts||{};
  const s=clear(id), W=s.clientWidth||1100, H=opts.H||300;
  const padL=58,padR=18,padT=14,padB=30;
  s.setAttribute('viewBox',`0 0 ${W} ${H}`); s.setAttribute('height',H);
  const X=v=>padL+(v-xdom[0])/(xdom[1]-xdom[0])*(W-padL-padR);
  const Y=v=>padT+(ydom[1]-v)/(ydom[1]-ydom[0])*(H-padT-padB);
  // shaded bands (graded windows)
  (opts.bands||[]).forEach(b=>{s.appendChild(el('rect',{x:X(b[0]),y:padT,width:Math.max(1,X(b[1])-X(b[0])),
    height:H-padT-padB,fill:cssv(b[2])}));});
  // y grid
  const yt=opts.yticks||5;
  for(let i=0;i<=yt;i++){const v=ydom[0]+(ydom[1]-ydom[0])*i/yt,y=Y(v);
    s.appendChild(el('line',{x1:padL,y1:y,x2:W-padR,y2:y,stroke:cssv('--grid'),'stroke-width':1}));
    const t=el('text',{x:padL-8,y:y+3,'text-anchor':'end'});t.textContent=opts.yfmt?opts.yfmt(v):v.toFixed(0);s.appendChild(t);}
  // zero line emphasis
  if(ydom[0]<0&&ydom[1]>0){const y=Y(0);s.appendChild(el('line',{x1:padL,y1:y,x2:W-padR,y2:y,stroke:cssv('--axis'),'stroke-width':1.3}));}
  // x ticks
  const xt=opts.xticks||xtSpan(xdom);
  xt.forEach(v=>{const x=X(v);const t=el('text',{x,y:H-padB+16,'text-anchor':'middle'});t.textContent=v;s.appendChild(t);});
  const xl=el('text',{x:(padL+W-padR)/2,y:H-2,'text-anchor':'middle'});xl.textContent=xlab;s.appendChild(xl);
  // series polylines
  const labelQueue=[];
  for(const se of series){
    const pts=se.x.map((xv,i)=>X(xv)+','+Y(se.y[i])).join(' ');
    s.appendChild(el('polyline',{points:pts,fill:'none',stroke:cssv(se.col),'stroke-width':se.w||2.2,
      'stroke-linejoin':'round','stroke-linecap':'round'}));
    // direct label at last point (position finalized below, after decluttering)
    if(se.label){labelQueue.push({lx:X(se.x[se.x.length-1]),ly:Y(se.y[se.y.length-1]),label:se.label,col:se.col});}
  }
  // declutter: when several lines converge, push overlapping end-labels apart vertically
  labelQueue.sort((a,b)=>a.ly-b.ly);
  const minGap=13;
  for(let i=1;i<labelQueue.length;i++){
    if(labelQueue[i].ly-labelQueue[i-1].ly<minGap) labelQueue[i].ly=labelQueue[i-1].ly+minGap;
  }
  labelQueue.forEach(lb=>{
    const t=el('text',{x:lb.lx+6,y:lb.ly+3,class:'val'});t.textContent=lb.label;t.setAttribute('fill',cssv(lb.col));s.appendChild(t);
  });
  // vertical markers (e.g. the chosen MATCH_K)
  (opts.vlines||[]).forEach(vl=>{const x=X(vl.x);
    s.appendChild(el('line',{x1:x,y1:padT,x2:x,y2:H-padB,stroke:cssv('--ink2'),'stroke-width':1.3,'stroke-dasharray':'4 3',opacity:.65}));
    const t=el('text',{x:x+5,y:padT+11,class:'val'});t.textContent=vl.label;t.setAttribute('fill',cssv('--ink2'));s.appendChild(t);});
  // crosshair hover
  const cross=el('line',{x1:0,y1:padT,x2:0,y2:H-padB,stroke:cssv('--axis'),'stroke-width':1,opacity:0});s.appendChild(cross);
  const dots=series.map(se=>{const c=el('circle',{r:4,fill:cssv(se.col),stroke:cssv('--surface'),'stroke-width':1.5,opacity:0});s.appendChild(c);return c;});
  const hit=el('rect',{x:padL,y:padT,width:W-padL-padR,height:H-padT-padB,fill:'transparent'});s.appendChild(hit);
  hit.addEventListener('mousemove',e=>{
    const r=s.getBoundingClientRect(),scale=(W)/r.width,mx=(e.clientX-r.left)*scale;
    const xv=xdom[0]+(mx-padL)/(W-padL-padR)*(xdom[1]-xdom[0]);
    let idx=0,bd=1e18; series[0].x.forEach((xx,i)=>{const d=Math.abs(xx-xv);if(d<bd){bd=d;idx=i;}});
    const xx=series[0].x[idx];cross.setAttribute('x1',X(xx));cross.setAttribute('x2',X(xx));cross.setAttribute('opacity',.6);
    let rowsH=`<b>${xlab.split(' ')[0]} ${xx}</b>`;
    series.forEach((se,k)=>{dots[k].setAttribute('cx',X(xx));dots[k].setAttribute('cy',Y(se.y[idx]));dots[k].setAttribute('opacity',1);
      rowsH+=`<br><span style="color:${cssv(se.col)}">■</span> ${se.nm}: <b>${opts.tipfmt?opts.tipfmt(se.y[idx]):se.y[idx]}</b>`;});
    showTip(e,rowsH);
  });
  hit.addEventListener('mouseleave',()=>{cross.setAttribute('opacity',0);dots.forEach(d=>d.setAttribute('opacity',0));hideTip();});
}
function xtSpan(d){const span=d[1]-d[0],step=span<=120?20:span<=350?50:100,t=[];
  for(let v=Math.ceil(d[0]/step)*step;v<=d[1];v+=step)t.push(v);return t;}

// ---------- Panel 3: rolling ----------
function renderRolling(){
  const R=DATA.rolling, ed=R.end_days;
  let lo=Infinity,hi=-Infinity;[R.LLALGO,R.LLMATCH,R.LLVOL,R.LLVOL_VO,R.LLBOOST,R.LLBOOST_V2,R.LLBOOST_V3,R.LLBOOST_V4,R.LLBOOST_V5,R.LLBOOST_V6,R.LLBOOST_V7,R.LLBOOST_V8,R.LLBOOST_V9,R.LLBOOST_V10,R.LLBOOST_V11,R.LLBOOST_V12,R.LLBOOST_V13,R.LLBOOST_V14,R.LLBOOST_V15,R.LLBOOST_V16,R.LLBOOST_V17,R.LLBOOST_V18,R.IDIO].forEach(a=>a.forEach(v=>{lo=Math.min(lo,v);hi=Math.max(hi,v);}));
  lo=Math.floor((lo-20)/50)*50; hi=Math.ceil((hi+20)/50)*50;
  lineChart('rolling',[
    {x:ed,y:R.LLBOOST_V18,col:'--olive',nm:'LLBOOST v18 (v17 + Bonferroni-corrected dual-EW IC gate)',label:'LLBOOST v18',w:5.2},
    {x:ed,y:R.LLBOOST_V17,col:'--cerulean',nm:'LLBOOST v17 (v15 + dual exponentially-weighted IC gate, HL=15/60)',label:'LLBOOST v17',w:5.0},
    {x:ed,y:R.LLBOOST_V16,col:'--indigo',nm:'LLBOOST v16 (v15 + flat 60d IC/t-stat significance gate)',label:'LLBOOST v16',w:4.8},
    {x:ed,y:R.LLBOOST_V15,col:'--berry',nm:'LLBOOST v15 (v12 + momentum/xsac insurance, no v13, recommended)',label:'LLBOOST v15',w:4.6},
    {x:ed,y:R.LLBOOST_V14,col:'--purple',nm:'LLBOOST v14 (v11+v12+v13 merged + momentum/xsac insurance, experimental)',label:'LLBOOST v14',w:4.4},
    {x:ed,y:R.LLBOOST_V13,col:'--jade',nm:'LLBOOST v13 (v11 + gated decayed-selection boost fallback, experimental)',label:'LLBOOST v13',w:4.2},
    {x:ed,y:R.LLBOOST_V12,col:'--umber',nm:'LLBOOST v12 (v11 + post-jump fixed-size fade, shipped)',label:'LLBOOST v12',w:4.0},
    {x:ed,y:R.LLBOOST_V11,col:'--cobalt',nm:'LLBOOST v11 (v10 + idio kill switch)',label:'LLBOOST v11',w:3.9},
    {x:ed,y:R.LLBOOST_V10,col:'--gold',nm:'LLBOOST v10 (v9 + rank-stability blend)',label:'LLBOOST v10',w:3.8},
    {x:ed,y:R.LLBOOST_V9,col:'--coral',nm:'LLBOOST v9 (v8 + beta-adjusted idio ridge target)',label:'LLBOOST v9',w:3.4},
    {x:ed,y:R.LLBOOST_V8,col:'--steel',nm:'LLBOOST v8 (ALGO min-conviction HOLD deadband)',label:'LLBOOST v8',w:3.2},
    {x:ed,y:R.LLBOOST_V7,col:'--rose',nm:'LLBOOST v7 (v6 + retuned COMBINE_GAIN)',label:'LLBOOST v7',w:2.8},
    {x:ed,y:R.LLBOOST_V6,col:'--brown',nm:'LLBOOST v6 (v5 refined boost + v2 adaptive momentum)',label:'LLBOOST v6',w:2.4},
    {x:ed,y:R.LLBOOST_V5,col:'--lime',nm:'LLBOOST v5 (v3 boost pool, re-tuned IC_L/MIN_DAY)',label:'LLBOOST v5',w:2.6},
    {x:ed,y:R.LLBOOST_V4,col:'--magenta',nm:'LLBOOST v4 (v3 candidate pool + v2 adaptive momentum)',label:'LLBOOST v4',w:3.0},
    {x:ed,y:R.LLBOOST_V3,col:'--teal',nm:'LLBOOST v3 (volatility-restricted candidate pool)',label:'LLBOOST v3',w:2.4},
    {x:ed,y:R.LLBOOST_V2,col:'--orange',nm:'LLBOOST v2 (adaptive momentum, experimental)',label:'LLBOOST v2',w:2.8},
    {x:ed,y:R.LLBOOST,col:'--green',nm:'LLBOOST (llvol + pairwise boost)',label:'LLBOOST',w:2.8},
    {x:ed,y:R.LLVOL,col:'--violet',nm:'LLVOL vol+momentum',label:'LLVOL v+m',w:2.6},
    {x:ed,y:R.LLVOL_VO,col:'--red',nm:'LLVOL·VO vol-only',label:'LLVOL·VO',w:2.2},
    {x:ed,y:R.LLALGO,col:'--blue',nm:'LLALGO (binary gate)',label:'LLALGO'},
    {x:ed,y:R.LLMATCH,col:'--amber',nm:'LLMATCH k='+MK,label:'LLMATCH'},
    {x:ed,y:R.IDIO,col:'--aqua',nm:'idio only',label:'idio'}],
    [ed[0],ed[ed.length-1]],[lo,hi],'day (250-day window ends here)',
    {H:300,bands:[[DATA.windows.OLD[0],DATA.windows.OLD[1],'--blueband'],
                  [DATA.windows.NEW[0],DATA.windows.NEW[1],'--posband']],
     tipfmt:v=>v.toFixed(0)});
}

// ---------- Panel 4: cumulative by leg ----------
let cumWin='NEW', cumStrat='LLALGO';
function renderCum(){
  const c=DATA.cum[cumStrat][cumWin], io=DATA.cum.IDIO_ONLY[cumWin];
  let lo=Infinity,hi=-Infinity;[c.tot,c.idio,c.algo].forEach(a=>a.forEach(v=>{lo=Math.min(lo,v);hi=Math.max(hi,v);}));
  const pad=(hi-lo)*.06||1; lo-=pad; hi+=pad;
  lineChart('cum',[
    {x:c.days,y:c.tot,col:'--ink',nm:'total',label:'total',w:2.4},
    {x:c.days,y:c.idio,col:'--aqua',nm:'idio',label:'idio'},
    {x:c.days,y:c.algo,col:'--red',nm:'ALGO',label:'ALGO'}],
    [c.days[0],c.days[c.days.length-1]],[lo,hi],'day',
    {H:320,yfmt:fmtK,tipfmt:fmtK});
}
document.getElementById('cumwin').addEventListener('click',e=>{if(!e.target.dataset.w)return;
  cumWin=e.target.dataset.w;[...e.currentTarget.children].forEach(b=>b.classList.toggle('on',b.dataset.w===cumWin));renderCum();});
document.getElementById('cumstrat').addEventListener('click',e=>{if(!e.target.dataset.s)return;
  cumStrat=e.target.dataset.s;[...e.currentTarget.children].forEach(b=>b.classList.toggle('on',b.dataset.s===cumStrat));renderCum();});

// ---------- Panel 5: MATCH_K sweep ----------
function renderKsweep(){
  const K=DATA.ksweep, kx=K.k;
  let lo=Infinity,hi=-Infinity;[K.new,K.old,K.roll_floor].forEach(a=>a.forEach(v=>{lo=Math.min(lo,v);hi=Math.max(hi,v);}));
  lo=Math.floor((lo-15)/25)*25; hi=Math.ceil((hi+15)/25)*25;
  lineChart('ksweep',[
    {x:kx,y:K.new,col:'--blue',nm:'new 751–1000',label:'new'},
    {x:kx,y:K.old,col:'--amber',nm:'old 501–750',label:'old'},
    {x:kx,y:K.roll_floor,col:'--aqua',nm:'rolling floor',label:'floor'}],
    [kx[0],kx[kx.length-1]],[lo,hi],'MATCH_K (index size ×)',
    {H:300,xticks:[0,0.5,1,1.5,2,2.5,3],tipfmt:v=>v.toFixed(0),
     vlines:[{x:MK,label:'k='+MK+' (SAFE_llmatch)'}]});
}
function renderAll(){renderKpis();renderDumbbell();renderLegbars();renderRolling();renderCum();renderKsweep();}
renderAll();
let rt;window.addEventListener('resize',()=>{clearTimeout(rt);rt=setTimeout(renderAll,120);});
new MutationObserver(renderAll).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
</script></body></html>"""

html = (HTML.replace("__DATA__", json.dumps(DATA, separators=(",", ":")))
            .replace("__NT__", str(DATA["nt"]))
            .replace("__MK__", "%g" % DATA["ksweep"]["match_k"]))
open("diagnostics.html", "w").write(html)
print("wrote diagnostics.html", f"({len(html)//1024} KB)")
