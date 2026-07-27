"""Build research.html — visualizes the research findings (data inlined). Reads research_data.json.
Panels: (1) ML overfitting IS vs OOS, (2) idio edge stability across windows, (3) lead-lag mechanism
(cross-name share + lagged-market persistence), (4) the signal scoreboard (mechanism + OOS verdicts)."""
import json
DATA = json.load(open("research_data.json"))

HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Research findings — what's real vs fitted</title>
<style>
:root{
  --surface:#fcfcfb;--plane:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;
  --border:rgba(11,11,11,.10);--blue:#2a78d6;--aqua:#1baf7a;--red:#e34948;--amber:#eda100;--violet:#4a3aa7;
  --posband:rgba(27,175,122,.13);--negband:rgba(227,73,72,.12);color-scheme:light;}
@media (prefers-color-scheme:dark){:root{
  --surface:#1a1a19;--plane:#0d0d0d;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;
  --border:rgba(255,255,255,.10);--blue:#3987e5;--aqua:#199e70;--red:#e66767;--amber:#c98500;--violet:#9085e9;
  --posband:rgba(25,158,112,.16);--negband:rgba(230,103,103,.15);color-scheme:dark;}}
:root[data-theme=dark]{--surface:#1a1a19;--plane:#0d0d0d;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;
  --axis:#383835;--border:rgba(255,255,255,.10);--blue:#3987e5;--aqua:#199e70;--red:#e66767;--amber:#c98500;
  --violet:#9085e9;--posband:rgba(25,158,112,.16);--negband:rgba(230,103,103,.15);color-scheme:dark;}
:root[data-theme=light]{--surface:#fcfcfb;--plane:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;
  --axis:#c3c2b7;--border:rgba(11,11,11,.10);--blue:#2a78d6;--aqua:#1baf7a;--red:#e34948;--amber:#eda100;
  --violet:#4a3aa7;--posband:rgba(27,175,122,.13);--negband:rgba(227,73,72,.12);color-scheme:light;}
*{box-sizing:border-box}
body{margin:0;background:var(--plane);color:var(--ink);font:14px/1.5 system-ui,-apple-system,"Segoe UI",Arial,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:26px 20px 80px}
h1{font-size:22px;margin:0 0 4px;letter-spacing:-.01em}
.sub{color:var(--ink2);font-size:14px;margin:0 0 22px;max-width:860px}.sub b{color:var(--ink)}
.panel{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:16px 18px 14px;margin:0 0 18px}
.panel h2{font-size:15.5px;margin:0 0 2px}
.panel .note{color:var(--ink2);font-size:12.5px;margin:0 0 12px;max-width:860px}.panel .note b{color:var(--ink)}
.legend{display:flex;gap:16px;flex-wrap:wrap;color:var(--ink2);font-size:12.5px;margin:2px 0 10px}
.legend span{display:inline-flex;align-items:center;gap:6px}.sw{width:11px;height:11px;border-radius:3px;display:inline-block}
svg{display:block;width:100%;height:auto;overflow:visible}
svg text{fill:var(--muted);font:11px system-ui,sans-serif}
svg text.lab{fill:var(--ink);font-weight:600}svg text.val{fill:var(--ink);font-variant-numeric:tabular-nums}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--border)}
th{color:var(--ink2);font-weight:600}
td.sig{font-weight:600}
.badge{display:inline-block;padding:2px 9px;border-radius:20px;font-size:11.5px;font-weight:700}
.b-ok{background:rgba(27,175,122,.16);color:var(--aqua)} .b-mid{background:rgba(237,161,0,.16);color:var(--amber)}
.b-no{background:rgba(227,73,72,.15);color:var(--red)}
.tip{position:fixed;pointer-events:none;background:var(--ink);color:var(--surface);padding:7px 10px;border-radius:7px;
  font-size:12px;opacity:0;transition:opacity .08s;z-index:20;white-space:nowrap;font-variant-numeric:tabular-nums}
.foot{color:var(--muted);font-size:11.5px;margin-top:18px;line-height:1.6}
</style></head><body><div class="wrap">
<h1>Research findings — what's a real edge vs a fitted backtest</h1>
<p class="sub">Every signal was put through the same filter: <b>is there a mechanism that must produce it, and does it
hold on unseen data</b> — not just "is the backtest number big." A big number can be pure hindsight (the ML panel
proves it). The durable edge is the <b>idio lead-lag</b>; almost everything on the ALGO leg was noise.</p>

<div class="panel">
  <h2>1 · The proof a big backtest number can be a mirage — ML in-sample vs out-of-sample</h2>
  <p class="note">Each model's IC fitting the idio next-day return. <b>Gradient boosting</b> gets a gorgeous
  <b>in-sample</b> IC (it memorized where training returns went — "buy here, sell there") and generalizes to
  <b>nothing</b> out-of-sample. The <b>linear ridge</b> has no gap (OOS ≥ IS) — it can't overfit a weak linear edge.</p>
  <div class="legend"><span><span class="sw" style="background:var(--amber)"></span>in-sample (H1, fitted)</span>
    <span><span class="sw" style="background:var(--blue)"></span>out-of-sample (H2, unseen)</span></div>
  <svg id="ml"></svg>
</div>

<div class="panel">
  <h2>2 · The idio edge is stable across every window (durable alpha)</h2>
  <p class="note">Pooled forecast IC across the 49 names, by window. The lead-lag IC is <b>~0.06 everywhere</b>
  (H1, H2, OLD, NEW) — it doesn't decay or regime-flip. Reversion is tiny alone but <b>orthogonal</b> (corr −0.04),
  so the blend is a logical diversification, not a kitchen sink.</p>
  <div class="legend"><span><span class="sw" style="background:var(--blue)"></span>lead-lag</span>
    <span><span class="sw" style="background:var(--aqua)"></span>blend .7/.3</span>
    <span><span class="sw" style="background:var(--muted)"></span>reversion</span></div>
  <svg id="idio"></svg>
</div>

<div class="panel">
  <h2>3 · The mechanism — it's genuine cross-name lead-lag, and the network persists</h2>
  <p class="note"><b id="csh"></b> of the ridge's predictive weight is <b>cross-name</b> (each name predicted from
  <i>other</i> names' moves, not its own) — real lead-lag, not single-series curve-fitting. And <i>which</i> names
  lag the market is stable: per-name lagged-market correlation, first half vs second, lines up on the diagonal
  (<b id="pc"></b>). A structural network you can't fake — the reason the edge generalizes.</p>
  <svg id="mech"></svg>
</div>

<div class="panel">
  <h2>4 · Signal scoreboard — mechanism × out-of-sample</h2>
  <p class="note">Everything tested this session. A signal is kept only if it has a <b>mechanism</b> AND <b>holds OOS</b>.
  Green = real, amber = real-but-fragile, red = noise/hindsight.</p>
  <table id="board"><thead><tr><th>signal</th><th>mechanism</th><th>out-of-sample</th><th>verdict</th></tr></thead><tbody></tbody></table>
</div>

<p class="foot">IC = correlation of the causal forecast with next-day return, pooled across the 49 idio names.
H1 = days 131–500, H2 = 501–1000. ML trained on H1, tested on H2 (HistGradientBoosting, early-stopped).
Built from research_data.json · compute_research.py.</p>
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

function renderML(){
  const s=clear('ml'),M=DATA.ml,W=s.clientWidth||1100,padL=180,padR=60,padT=8,padB=26,rowH=54;
  const H=padT+padB+M.length*rowH; s.setAttribute('viewBox',`0 0 ${W} ${H}`); s.setAttribute('height',H);
  const hi=0.26,X=v=>padL+v/hi*(W-padL-padR);
  for(let g=0;g<=0.25;g+=0.05){const x=X(g);s.appendChild(el('line',{x1:x,y1:padT,x2:x,y2:H-padB,stroke:cssv('--grid'),'stroke-width':1}));
    const t=el('text',{x,y:H-padB+16,'text-anchor':'middle'});t.textContent=g.toFixed(2);s.appendChild(t);}
  M.forEach((m,i)=>{const y=padT+i*rowH+6,isGB=m.model.indexOf('GBM')>=0;
    const lab=el('text',{x:padL-12,y:y+rowH/2-2,'text-anchor':'end',class:'lab'});lab.textContent=m.model;
    if(isGB)lab.setAttribute('fill',cssv('--red'));s.appendChild(lab);
    [['is','--amber',m.is],['oos','--blue',m.oos]].forEach((b,j)=>{const yy=y+j*16,w=Math.max(1,X(b[2])-padL);
      const rect=el('rect',{x:padL,y:yy,width:w,height:12,rx:3,fill:cssv(b[1])});rect.style.cursor='pointer';
      rect.addEventListener('mousemove',e=>showTip(e,`<b>${m.model}</b><br>${b[0]==='is'?'in-sample':'out-of-sample'} IC = ${b[2].toFixed(3)}`));
      rect.addEventListener('mouseleave',hideTip);s.appendChild(rect);
      const t=el('text',{x:X(b[2])+6,y:yy+10,class:'val'});t.textContent=b[2].toFixed(3);s.appendChild(t);});
    if(isGB){const t=el('text',{x:X(m.is)+6,y:y+10,class:'val'});t.textContent='← memorized noise';t.setAttribute('fill',cssv('--red'));}
  });
}

function renderIdio(){
  const s=clear('idio'),I=DATA.idio_ic,wins=['full','H1','H2','OLD','NEW'],W=s.clientWidth||1100;
  const padL=44,padR=14,padT=12,padB=30,H=280;s.setAttribute('viewBox',`0 0 ${W} ${H}`);s.setAttribute('height',H);
  const hi=0.08,Y=v=>padT+(hi-v)/hi*(H-padT-padB),z=Y(0);
  for(let g=0;g<=0.08;g+=0.02){const y=Y(g);s.appendChild(el('line',{x1:padL,y1:y,x2:W-padR,y2:y,stroke:cssv('--grid'),'stroke-width':1}));
    const t=el('text',{x:padL-6,y:y+3,'text-anchor':'end'});t.textContent=g.toFixed(2);s.appendChild(t);}
  const gw=(W-padL-padR)/wins.length, series=[['leadlag','--blue'],['blend','--aqua'],['reversion','--muted']];
  wins.forEach((wn,i)=>{const x0=padL+i*gw, bw=gw*0.8/series.length;
    series.forEach((se,j)=>{const v=I[se[0]][wn],x=x0+gw*0.1+j*bw;
      const rect=el('rect',{x:x+1,y:Y(Math.max(v,0)),width:bw-2,height:Math.abs(Y(v)-z),rx:2,fill:cssv(se[1])});
      rect.style.cursor='pointer';rect.addEventListener('mousemove',e=>showTip(e,`<b>${se[0]}</b> · ${wn}<br>IC = ${v.toFixed(3)}`));
      rect.addEventListener('mouseleave',hideTip);s.appendChild(rect);});
    const t=el('text',{x:x0+gw/2,y:H-padB+16,'text-anchor':'middle',class:'lab'});t.textContent=wn;s.appendChild(t);});
}

function renderMech(){
  const s=clear('mech'),h1=DATA.mech.h1,h2=DATA.mech.h2,N=h1.length,W=s.clientWidth||1100,pad=46,H=380;
  s.setAttribute('viewBox',`0 0 ${W} ${H}`);s.setAttribute('height',H);
  let lim=0.05;for(let i=0;i<N;i++){if(!isNaN(h1[i]))lim=Math.max(lim,Math.abs(h1[i]),Math.abs(h2[i]));}lim*=1.1;
  const X=v=>pad+(v+lim)/(2*lim)*(W-2*pad),Y=v=>pad+(lim-v)/(2*lim)*(H-2*pad);
  s.appendChild(el('rect',{x:X(0),y:Y(lim),width:X(lim)-X(0),height:Y(0)-Y(lim),fill:cssv('--posband')}));
  s.appendChild(el('line',{x1:X(0),y1:pad,x2:X(0),y2:H-pad,stroke:cssv('--axis'),'stroke-width':1}));
  s.appendChild(el('line',{x1:pad,y1:Y(0),x2:W-pad,y2:Y(0),stroke:cssv('--axis'),'stroke-width':1}));
  s.appendChild(el('line',{x1:X(-lim),y1:Y(-lim),x2:X(lim),y2:Y(lim),stroke:cssv('--muted'),'stroke-width':1,'stroke-dasharray':'4 4'}));
  for(const g of [-0.1,0,0.1]){if(Math.abs(g)>lim)continue;
    let t=el('text',{x:X(g),y:H-pad+16,'text-anchor':'middle'});t.textContent=g.toFixed(2);s.appendChild(t);
    let u=el('text',{x:pad-8,y:Y(g)+3,'text-anchor':'end'});u.textContent=g.toFixed(2);s.appendChild(u);}
  s.appendChild(Object.assign(el('text',{x:W/2,y:H-6,'text-anchor':'middle'}),{textContent:'lagged-market corr · days 1–480  →'}));
  const yl=el('text',{x:14,y:H/2,'text-anchor':'middle'});yl.setAttribute('transform',`rotate(-90 14 ${H/2})`);
  yl.textContent='lagged-market corr · days 480–1000  →';s.appendChild(yl);
  s.appendChild(Object.assign(el('text',{x:X(lim)-6,y:Y(lim)+14,'text-anchor':'end',class:'lab'}),{textContent:'names follow market with a lag ✓'}));
  for(let i=0;i<N;i++){if(isNaN(h1[i])||isNaN(h2[i]))continue;
    const c=el('circle',{cx:X(h1[i]),cy:Y(h2[i]),r:4.5,fill:cssv('--aqua'),stroke:cssv('--surface'),'stroke-width':1,opacity:.72});
    c.style.cursor='pointer';c.addEventListener('mousemove',e=>showTip(e,`<b>${DATA.names[i+1]}</b><br>H1 ${h1[i].toFixed(3)} · H2 ${h2[i].toFixed(3)}`));
    c.addEventListener('mouseleave',hideTip);s.appendChild(c);}
  document.getElementById('csh').textContent=DATA.mech.cross_share+'%';
  document.getElementById('pc').textContent='corr H1↔H2 = '+DATA.mech.persist_corr;
}

function renderBoard(){
  const tb=document.querySelector('#board tbody');tb.innerHTML='';
  const cls={0:'b-no',1:'b-ok',2:'b-mid'},txt={0:'noise',1:'real',2:'fragile'};
  for(const r of DATA.scoreboard){const tr=document.createElement('tr');
    tr.innerHTML=`<td class="sig">${r.sig}</td><td>${r.mech}</td><td>${r.oos}</td><td><span class="badge ${cls[r.ok]}">${txt[r.ok]}</span></td>`;
    tb.appendChild(tr);}
}
function renderAll(){renderML();renderIdio();renderMech();renderBoard();}
renderAll();
let rt;window.addEventListener('resize',()=>{clearTimeout(rt);rt=setTimeout(renderAll,120);});
new MutationObserver(renderAll).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
</script></body></html>"""
open("research.html","w").write(HTML.replace("__DATA__", json.dumps(DATA, separators=(",", ":"))))
print("wrote research.html", f"({len(HTML)//1024} KB base)")
