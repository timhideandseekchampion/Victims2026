"""Build a self-contained interactive dashboard (data inlined) for reviewing SAFE/SWING
entries & exits + per-asset PnL across all 750 days."""
import json
DATA = json.load(open("positions_data.json"))

HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SAFE vs SWING — entries & exits review</title>
<style>
:root{
  --bg:#fbfbfa; --surface:#fff; --ink:#1a1a1a; --ink2:#5a5f66; --muted:#9aa0a6;
  --grid:#ececec; --border:#e3e3e0;
  --long:#0e9e73; --short:#e0455e; --pnl:#3b6fd4;
  --longband:rgba(14,158,115,.12); --shortband:rgba(224,69,94,.11);
}
@media (prefers-color-scheme:dark){:root{
  --bg:#15171a; --surface:#1d2024; --ink:#e9eaec; --ink2:#a6adb5; --muted:#6b7178;
  --grid:#2a2e33; --border:#31363c;
  --long:#2fd39a; --short:#ff6b83; --pnl:#6fa0f5;
  --longband:rgba(47,211,154,.15); --shortband:rgba(255,107,131,.13);
}}
:root[data-theme=dark]{--bg:#15171a;--surface:#1d2024;--ink:#e9eaec;--ink2:#a6adb5;--muted:#6b7178;--grid:#2a2e33;--border:#31363c;--long:#2fd39a;--short:#ff6b83;--pnl:#6fa0f5;--longband:rgba(47,211,154,.15);--shortband:rgba(255,107,131,.13);}
:root[data-theme=light]{--bg:#fbfbfa;--surface:#fff;--ink:#1a1a1a;--ink2:#5a5f66;--muted:#9aa0a6;--grid:#ececec;--border:#e3e3e0;--long:#0e9e73;--short:#e0455e;--longband:rgba(14,158,115,.12);--shortband:rgba(224,69,94,.11);}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1280px;margin:0 auto;padding:20px 20px 60px}
h1{font-size:19px;margin:0 0 2px} .sub{color:var(--ink2);font-size:13px;margin:0 0 16px}
.bar{display:flex;flex-wrap:wrap;gap:14px;align-items:center;margin:14px 0;padding:12px 14px;background:var(--surface);border:1px solid var(--border);border-radius:10px}
.seg{display:inline-flex;border:1px solid var(--border);border-radius:8px;overflow:hidden}
.seg button{border:0;background:transparent;color:var(--ink2);padding:7px 16px;font-weight:600;cursor:pointer;font-size:13px}
.seg button.on{background:var(--ink);color:var(--bg)}
.legend{display:flex;gap:16px;align-items:center;flex-wrap:wrap;color:var(--ink2);font-size:12.5px}
.legend span{display:inline-flex;align-items:center;gap:6px}
.sw{width:14px;height:14px;border-radius:3px;display:inline-block}
.tri{font-size:13px;line-height:1}
.detailhead{display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:8px;margin:4px 2px 8px}
.detailhead .nm{font-size:16px;font-weight:700}
.stat{color:var(--ink2);font-size:12.5px} .stat b{color:var(--ink)}
.pos{color:var(--long)} .neg{color:var(--short)}
.card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:12px}
canvas{display:block;width:100%}
.hint{color:var(--muted);font-size:12px;margin:6px 2px 0}
.gridhead{margin:22px 2px 8px;font-weight:700;font-size:14px;display:flex;justify-content:space-between;align-items:center}
.gridwrap{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:9px}
.mini{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:7px 8px 5px;cursor:pointer;transition:border-color .12s}
.mini:hover{border-color:var(--ink2)} .mini.sel{border-color:var(--pnl);box-shadow:0 0 0 1px var(--pnl)}
.mini .mh{display:flex;justify-content:space-between;font-size:11.5px;margin-bottom:2px}
.mini .tk{font-weight:700} .mini .pl{font-variant-numeric:tabular-nums}
.tip{position:fixed;pointer-events:none;background:var(--ink);color:var(--bg);padding:6px 9px;border-radius:6px;font-size:12px;opacity:0;transition:opacity .08s;z-index:9;white-space:nowrap;font-variant-numeric:tabular-nums}
select{background:var(--surface);color:var(--ink);border:1px solid var(--border);border-radius:7px;padding:6px 9px;font-size:13px}
.sortbtn{background:var(--surface);border:1px solid var(--border);color:var(--ink2);border-radius:7px;padding:5px 10px;font-size:12px;cursor:pointer}
</style></head><body><div class="wrap">
<h1>SAFE vs SWING — entries &amp; exits, all 51 assets · 750 days</h1>
<p class="sub">Green band = book is <b>long</b> that asset · red band = <b>short</b>. Triangles mark each entry (position flip). Positions re-decide daily and flip often (~every 2 days), so zoom in (drag on the price chart) to see individual ▲/▼. Bottom strip = cumulative PnL for that asset.</p>

<div class="bar">
  <div class="seg" id="stratseg">
    <button data-s="SAFE" class="on">SAFE</button><button data-s="SWING">SWING</button>
  </div>
  <label>Asset <select id="assetsel"></select></label>
  <button class="sortbtn" id="resetzoom">Reset zoom</button>
  <div class="legend">
    <span><span class="sw" style="background:var(--longband);border:1px solid var(--long)"></span>long</span>
    <span><span class="sw" style="background:var(--shortband);border:1px solid var(--short)"></span>short</span>
    <span><span class="tri" style="color:var(--long)">▲</span> long entry</span>
    <span><span class="tri" style="color:var(--short)">▼</span> short entry</span>
    <span><span class="sw" style="background:var(--pnl)"></span> cumulative PnL</span>
  </div>
</div>

<div class="detailhead">
  <span class="nm" id="dnm"></span>
  <span class="stat" id="dstat"></span>
</div>
<div class="card">
  <canvas id="price" height="300"></canvas>
  <canvas id="pnl" height="120"></canvas>
</div>
<p class="hint">Drag left→right on the price chart to zoom a date range · click a mini-chart below to switch asset.</p>

<div class="detailhead" style="margin-top:22px">
  <span class="nm">Do winners &amp; losers persist? — 1st-half vs 2nd-half PnL per asset</span>
  <span class="stat" id="pcap"></span>
</div>
<div class="card"><canvas id="persist" height="380"></canvas></div>
<p class="hint">Each dot is one asset: x = PnL over the first half (days 130–440), y = second half (440–749).
Persistent skill would line up on the diagonal; <b>scatter with no diagonal = luck</b>. The 5 you flagged are ringed &amp; labelled — watch them jump quadrants between halves. Bottom-left = lost both halves (a true persistent loser); notice how few sit there.</p>

<div class="gridhead"><span id="ghead"></span>
  <button class="sortbtn" id="sortbtn">sort: PnL ▼</button></div>
<div class="gridwrap" id="grid"></div>

<div class="tip" id="tip"></div>
</div>
<script>
const DATA = __DATA__;
const N = DATA.names.length, NT = DATA.nt;
let strat="SAFE", asset=0, x0=0, x1=NT-1, sortMode="pnl";
const css=k=>getComputedStyle(document.body).getPropertyValue(k).trim();

function niceDayTicks(a,b){const span=b-a,step=span<=120?20:span<=300?50:100,t=[];for(let d=Math.ceil(a/step)*step;d<=b;d+=step)t.push(d);return t;}
function dpi(cv,h){const r=devicePixelRatio||1,w=cv.clientWidth;cv.width=w*r;cv.height=h*r;const c=cv.getContext("2d");c.setTransform(r,0,0,r,0,0);return[c,w,h];}

function drawBands(c,W,H,pad,sign,xs){ // colored long/short background over [x0,x1]
  let d=x0;
  while(d<=x1){let s=sign[d];if(s===0){d++;continue;}let e=d;while(e+1<=x1&&sign[e+1]===s)e++;
    c.fillStyle=s>0?css('--longband'):css('--shortband');
    const xa=xs(d),xb=xs(e+1>x1?x1:e+1);c.fillRect(xa,pad.t,Math.max(1,xb-xa),H-pad.t-pad.b);d=e+1;}
}
function renderPrice(){
  const cv=document.getElementById('price'),[c,W,H]=dpi(cv,300);
  const pad={l:52,r:14,t:12,b:22},prices=DATA.prices[asset],S=DATA[strat];
  let lo=Infinity,hi=-Infinity;for(let d=x0;d<=x1;d++){lo=Math.min(lo,prices[d]);hi=Math.max(hi,prices[d]);}
  const pd=(hi-lo)||1;lo-=pd*.06;hi+=pd*.06;
  const xs=d=>pad.l+(d-x0)/((x1-x0)||1)*(W-pad.l-pad.r), ys=v=>pad.t+(hi-v)/(hi-lo)*(H-pad.t-pad.b);
  c.clearRect(0,0,W,H);
  drawBands(c,W,H,pad,S.sign[asset],xs);
  // grid + y labels
  c.strokeStyle=css('--grid');c.fillStyle=css('--muted');c.font="11px sans-serif";c.lineWidth=1;c.textAlign="right";
  for(let i=0;i<=4;i++){const v=lo+(hi-lo)*i/4,y=ys(v);c.beginPath();c.moveTo(pad.l,y);c.lineTo(W-pad.r,y);c.stroke();c.fillText(v.toFixed(1),pad.l-6,y+3);}
  c.textAlign="center";for(const d of niceDayTicks(x0,x1)){c.fillText(d,xs(d),H-6);}
  // price line
  c.strokeStyle=css('--ink');c.lineWidth=1.6;c.beginPath();
  for(let d=x0;d<=x1;d++){const X=xs(d),Y=ys(prices[d]);d===x0?c.moveTo(X,Y):c.lineTo(X,Y);}c.stroke();
  // entry triangles (only those in range)
  const evs=S.events[asset];const span=x1-x0;const showTri=span<=400;  // hide when too dense to read
  if(showTri){for(const[d,dir]of evs){if(d<x0||d>x1)continue;const X=xs(d),Y=ys(prices[d]);
    c.fillStyle=dir>0?css('--long'):css('--short');c.beginPath();
    if(dir>0){c.moveTo(X,Y-11);c.lineTo(X-5,Y-3);c.lineTo(X+5,Y-3);}else{c.moveTo(X,Y+11);c.lineTo(X-5,Y+3);c.lineTo(X+5,Y+3);}
    c.closePath();c.fill();}}
  cv._xs=xs;cv._pad=pad;cv._prices=prices;
}
function renderPnl(){
  const cv=document.getElementById('pnl'),[c,W,H]=dpi(cv,120);
  const pad={l:52,r:14,t:10,b:18},pnl=DATA[strat].pnl[asset];
  let lo=0,hi=0;for(let d=x0;d<=x1;d++){lo=Math.min(lo,pnl[d]);hi=Math.max(hi,pnl[d]);}
  if(hi===lo)hi+=1;const xs=d=>pad.l+(d-x0)/((x1-x0)||1)*(W-pad.l-pad.r),ys=v=>pad.t+(hi-v)/(hi-lo)*(H-pad.t-pad.b);
  c.clearRect(0,0,W,H);
  c.strokeStyle=css('--grid');c.fillStyle=css('--muted');c.font="11px sans-serif";c.textAlign="right";
  const z=ys(0);c.beginPath();c.moveTo(pad.l,z);c.lineTo(W-pad.r,z);c.stroke();
  c.fillText('$'+hi.toFixed(0),pad.l-6,ys(hi)+3);c.fillText('$'+lo.toFixed(0),pad.l-6,ys(lo)+0);
  c.textAlign="left";c.fillText('cumulative PnL',pad.l+2,pad.t+2);
  // area
  c.beginPath();c.moveTo(xs(x0),z);for(let d=x0;d<=x1;d++)c.lineTo(xs(d),ys(pnl[d]));c.lineTo(xs(x1),z);c.closePath();
  c.fillStyle=css('--pnl')+'22';c.fill();
  c.strokeStyle=css('--pnl');c.lineWidth=1.6;c.beginPath();
  for(let d=x0;d<=x1;d++){const X=xs(d),Y=ys(pnl[d]);d===x0?c.moveTo(X,Y):c.lineTo(X,Y);}c.stroke();
  cv._xs=xs;cv._pad=pad;cv._pnl=pnl;
}
function renderHead(){
  const S=DATA[strat],evs=S.events[asset],pnl=S.pnl_final[asset];
  const longs=S.sign[asset].filter(x=>x>0).length,shorts=S.sign[asset].filter(x=>x<0).length;
  document.getElementById('dnm').textContent=DATA.names[asset]+(asset===0?"  (ALGO = index)":"");
  document.getElementById('dstat').innerHTML=
    `${strat} · <b>${evs.length}</b> entries · days long <b>${longs}</b> / short <b>${shorts}</b> · final PnL `+
    `<b class="${pnl>=0?'pos':'neg'}">$${pnl.toLocaleString()}</b>`;
}
function renderDetail(){renderHead();renderPrice();renderPnl();markSel();}

function miniCanvas(i){
  const cv=document.createElement('canvas');cv.height=42;const box=document.createElement('div');
  box.className='mini'+(i===asset?' sel':'');box.dataset.i=i;
  const pl=DATA[strat].pnl_final[i];
  box.innerHTML=`<div class="mh"><span class="tk">${DATA.names[i]}</span><span class="pl ${pl>=0?'pos':'neg'}">$${(pl/1000).toFixed(1)}k</span></div>`;
  box.appendChild(cv);box.addEventListener('click',()=>{asset=i;x0=0;x1=NT-1;document.getElementById('assetsel').value=i;renderDetail();markSel();});
  requestAnimationFrame(()=>{const r=devicePixelRatio||1,W=cv.clientWidth,H=42;cv.width=W*r;cv.height=H*r;const c=cv.getContext('2d');c.setTransform(r,0,0,r,0,0);
    const pr=DATA.prices[i],sg=DATA[strat].sign[i];let lo=Infinity,hi=-Infinity;for(let d=0;d<NT;d++){lo=Math.min(lo,pr[d]);hi=Math.max(hi,pr[d]);}const pd=(hi-lo)||1;
    const xs=d=>d/(NT-1)*W,ys=v=>2+(hi-v)/(hi-lo+1e-9)*(H-4);
    let d=0;while(d<NT){const s=sg[d];if(s===0){d++;continue;}let e=d;while(e+1<NT&&sg[e+1]===s)e++;c.fillStyle=s>0?css('--longband'):css('--shortband');c.fillRect(xs(d),0,Math.max(1,xs(e+1)-xs(d)),H);d=e+1;}
    c.strokeStyle=css('--ink');c.lineWidth=1;c.beginPath();for(let d=0;d<NT;d++){const X=xs(d),Y=ys(pr[d]);d===0?c.moveTo(X,Y):c.lineTo(X,Y);}c.stroke();});
  return box;
}
function renderGrid(){
  const g=document.getElementById('grid');g.innerHTML='';
  let order=[...Array(N).keys()];
  if(sortMode==='pnl')order.sort((a,b)=>DATA[strat].pnl_final[b]-DATA[strat].pnl_final[a]);
  const S=DATA[strat];const tot=S.pnl_final.reduce((a,b)=>a+b,0),win=S.pnl_final.filter(x=>x>0).length;
  document.getElementById('ghead').innerHTML=`All 51 assets — ${strat} · total PnL <b>$${tot.toLocaleString()}</b> · <b>${win}/51</b> winners (click to inspect)`;
  for(const i of order)g.appendChild(miniCanvas(i));
}
function markSel(){document.querySelectorAll('.mini').forEach(m=>m.classList.toggle('sel',+m.dataset.i===asset));}

const FLAG=["ILVX","SMAH","FWWG","ELLT","ACIX"];
const MID=440;
function renderPersist(){
  const cv=document.getElementById('persist'),[c,W,H]=dpi(cv,380);
  const pad={l:60,r:16,t:14,b:34},pnl=DATA[strat].pnl;
  const pts=[];for(let i=1;i<N;i++){pts.push({i,x:pnl[i][MID]-pnl[i][130],y:pnl[i][NT-1]-pnl[i][MID],nm:DATA.names[i]});}
  const xs_=pts.map(p=>p.x),ys_=pts.map(p=>p.y);
  const lim=Math.max(...xs_.map(Math.abs),...ys_.map(Math.abs))*1.08;
  const X=v=>pad.l+(v+lim)/(2*lim)*(W-pad.l-pad.r), Y=v=>pad.t+(lim-v)/(2*lim)*(H-pad.t-pad.b);
  c.clearRect(0,0,W,H);
  // quadrant tints (lose-both bottom-left / win-both top-right)
  c.fillStyle=css('--shortband');c.fillRect(X(-lim),Y(0),X(0)-X(-lim),Y(-lim)-Y(0));
  c.fillStyle=css('--longband');c.fillRect(X(0),Y(lim),X(lim)-X(0),Y(0)-Y(lim));
  // axes at 0 + diagonal
  c.strokeStyle=css('--border');c.lineWidth=1;c.beginPath();c.moveTo(X(0),pad.t);c.lineTo(X(0),H-pad.b);c.moveTo(pad.l,Y(0));c.lineTo(W-pad.r,Y(0));c.stroke();
  c.strokeStyle=css('--muted');c.setLineDash([4,4]);c.beginPath();c.moveTo(X(-lim),Y(-lim));c.lineTo(X(lim),Y(lim));c.stroke();c.setLineDash([]);
  c.fillStyle=css('--muted');c.font="11px sans-serif";c.textAlign="center";
  c.fillText('1st-half PnL  →',W/2,H-8);c.save();c.translate(14,H/2);c.rotate(-Math.PI/2);c.fillText('2nd-half PnL  →',0,0);c.restore();
  c.textAlign="left";c.fillStyle=css('--ink2');c.fillText('lose both',X(-lim)+6,Y(-lim)-8);c.textAlign="right";c.fillText('win both',X(lim)-6,Y(lim)+14);
  // dots
  const flagIdx=new Set(FLAG.map(f=>DATA.names.indexOf(f)));
  for(const p of pts){const isF=flagIdx.has(p.i);
    c.beginPath();c.arc(X(p.x),Y(p.y),isF?5.5:4,0,7);
    c.fillStyle=(p.x<0&&p.y<0)?css('--short'):(p.x>0&&p.y>0)?css('--long'):css('--muted');
    c.globalAlpha=isF?1:.72;c.fill();c.globalAlpha=1;
    if(isF){c.lineWidth=2;c.strokeStyle=css('--ink');c.stroke();
      c.fillStyle=css('--ink');c.font="bold 10.5px sans-serif";c.textAlign="left";c.fillText(p.nm,X(p.x)+8,Y(p.y)-6);}
    p.px=X(p.x);p.py=Y(p.y);}
  cv._pts=pts;
  // correlation
  const mx=xs_.reduce((a,b)=>a+b)/pts.length,my=ys_.reduce((a,b)=>a+b)/pts.length;
  let sxy=0,sx=0,sy=0;for(const p of pts){sxy+=(p.x-mx)*(p.y-my);sx+=(p.x-mx)**2;sy+=(p.y-my)**2;}
  const r=sxy/Math.sqrt(sx*sy);const both=pts.filter(p=>p.x<0&&p.y<0).length;
  document.getElementById('pcap').innerHTML=`${strat} · corr(1st,2nd) = <b>${r.toFixed(2)}</b> (≈0 ⇒ no persistence) · lost <b>both</b> halves: <b>${both}/50</b>`;
}

// hover tooltip on price chart
const tip=document.getElementById('tip');
function hover(cv,kind){cv.addEventListener('mousemove',e=>{const r=cv.getBoundingClientRect(),xs=cv._xs,pad=cv._pad;if(!xs)return;
  const mx=e.clientX-r.left;let d=Math.round(x0+(mx-pad.l)/((cv.clientWidth-pad.l-pad.r))*(x1-x0));d=Math.max(x0,Math.min(x1,d));
  const S=DATA[strat],s=S.sign[asset][d];const pos=s>0?'LONG':s<0?'SHORT':'flat';
  tip.innerHTML=`day ${d} · ${DATA.names[asset]}<br>price $${DATA.prices[asset][d].toFixed(2)} · <b style="color:${s>0?css('--long'):css('--short')}">${pos}</b><br>PnL $${S.pnl[asset][d].toLocaleString()}`;
  tip.style.left=(e.clientX+14)+'px';tip.style.top=(e.clientY+14)+'px';tip.style.opacity=1;});
  cv.addEventListener('mouseleave',()=>tip.style.opacity=0);}
hover(document.getElementById('price'));hover(document.getElementById('pnl'));
// scatter hover (nearest point)
(function(){const cv=document.getElementById('persist');cv.addEventListener('mousemove',e=>{const pts=cv._pts;if(!pts)return;
  const r=cv.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;let best=null,bd=1e9;
  for(const p of pts){const d=(p.px-mx)**2+(p.py-my)**2;if(d<bd){bd=d;best=p;}}
  if(best&&bd<400){tip.innerHTML=`<b>${best.nm}</b><br>1st half $${best.x.toLocaleString(undefined,{maximumFractionDigits:0})}<br>2nd half $${best.y.toLocaleString(undefined,{maximumFractionDigits:0})}`;
    tip.style.left=(e.clientX+14)+'px';tip.style.top=(e.clientY+14)+'px';tip.style.opacity=1;}else tip.style.opacity=0;});
  cv.addEventListener('mouseleave',()=>tip.style.opacity=0);})();

// drag to zoom on price chart
(function(){const cv=document.getElementById('price');let dragging=false,sx=0;
 cv.addEventListener('mousedown',e=>{dragging=true;sx=e.clientX-cv.getBoundingClientRect().left;});
 window.addEventListener('mouseup',e=>{if(!dragging)return;dragging=false;const pad=cv._pad,xs=cv._xs;
   const ex=e.clientX-cv.getBoundingClientRect().left;const toD=px=>Math.round(x0+(px-pad.l)/((cv.clientWidth-pad.l-pad.r))*(x1-x0));
   let a=toD(Math.min(sx,ex)),b=toD(Math.max(sx,ex));if(b-a>=8){x0=Math.max(0,a);x1=Math.min(NT-1,b);renderDetail();}});
})();

// controls
document.getElementById('stratseg').addEventListener('click',e=>{if(!e.target.dataset.s)return;
  strat=e.target.dataset.s;[...e.currentTarget.children].forEach(b=>b.classList.toggle('on',b.dataset.s===strat));renderDetail();renderPersist();renderGrid();});
const sel=document.getElementById('assetsel');DATA.names.forEach((n,i)=>{const o=document.createElement('option');o.value=i;o.textContent=n+(i===0?' (ALGO/index)':'');sel.appendChild(o);});
sel.addEventListener('change',()=>{asset=+sel.value;x0=0;x1=NT-1;renderDetail();markSel();});
document.getElementById('resetzoom').addEventListener('click',()=>{x0=0;x1=NT-1;renderDetail();});
document.getElementById('sortbtn').addEventListener('click',e=>{sortMode=sortMode==='pnl'?'idx':'pnl';e.target.textContent=sortMode==='pnl'?'sort: PnL ▼':'sort: ticker';renderGrid();});
window.addEventListener('resize',()=>{renderDetail();renderPersist();renderGrid();});
renderDetail();renderPersist();renderGrid();
</script></body></html>"""

html = HTML.replace("__DATA__", json.dumps(DATA, separators=(",", ":")))
open("dashboard.html", "w").write(html)
print("wrote dashboard.html", f"({len(html)//1024} KB)")
