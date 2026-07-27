"""Build behavior.html — a position-behavior explorer (data inlined from positions_data.json):
a long/short timeline heatmap for all 51 instruments (green=long, red=short, faint=flat) across all
days, per strategy; click any row for the classic price + long/short-band + entry-marker detail."""
import json
DATA = json.load(open("positions_data.json"))

HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Position behaviour — long/short timeline</title>
<style>
:root{--bg:#fbfbfa;--surface:#fff;--ink:#1a1a1a;--ink2:#5a5f66;--muted:#9aa0a6;--grid:#ececec;--border:#e3e3e0;
  --long:#0e9e73;--short:#e0455e;--pnl:#3b6fd4;--vo:#d98a2b;--longband:rgba(14,158,115,.14);--shortband:rgba(224,69,94,.13);--flat:rgba(150,150,150,.06);}
@media (prefers-color-scheme:dark){:root{--bg:#15171a;--surface:#1d2024;--ink:#e9eaec;--ink2:#a6adb5;--muted:#6b7178;
  --grid:#2a2e33;--border:#31363c;--long:#2fd39a;--short:#ff6b83;--pnl:#6fa0f5;--vo:#e0a24a;--longband:rgba(47,211,154,.16);--shortband:rgba(255,107,131,.15);--flat:rgba(150,150,150,.05);}}
:root[data-theme=dark]{--bg:#15171a;--surface:#1d2024;--ink:#e9eaec;--ink2:#a6adb5;--muted:#6b7178;--grid:#2a2e33;--border:#31363c;--long:#2fd39a;--short:#ff6b83;--pnl:#6fa0f5;--vo:#e0a24a;--longband:rgba(47,211,154,.16);--shortband:rgba(255,107,131,.15);--flat:rgba(150,150,150,.05);}
:root[data-theme=light]{--bg:#fbfbfa;--surface:#fff;--ink:#1a1a1a;--ink2:#5a5f66;--muted:#9aa0a6;--grid:#ececec;--border:#e3e3e0;--long:#0e9e73;--short:#e0455e;--pnl:#3b6fd4;--vo:#d98a2b;--longband:rgba(14,158,115,.14);--shortband:rgba(224,69,94,.13);--flat:rgba(150,150,150,.06);}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif}
.wrap{max-width:1280px;margin:0 auto;padding:20px 20px 60px}
h1{font-size:19px;margin:0 0 2px}.sub{color:var(--ink2);font-size:13px;margin:0 0 14px}
.bar{display:flex;flex-wrap:wrap;gap:14px;align-items:center;margin:12px 0;padding:11px 14px;background:var(--surface);border:1px solid var(--border);border-radius:10px}
.seg{display:inline-flex;border:1px solid var(--border);border-radius:8px;overflow:hidden}
.seg button{border:0;background:transparent;color:var(--ink2);padding:7px 15px;font-weight:600;cursor:pointer;font-size:13px}
.seg button.on{background:var(--ink);color:var(--bg)}
.legend{display:flex;gap:16px;align-items:center;flex-wrap:wrap;color:var(--ink2);font-size:12.5px}
.legend span{display:inline-flex;align-items:center;gap:6px}.sw{width:14px;height:12px;border-radius:2px;display:inline-block}
.stat{color:var(--ink2);font-size:12.5px}.stat b{color:var(--ink)}
.card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:10px 12px;margin-top:12px}
canvas{display:block;width:100%}
.hint{color:var(--muted);font-size:12px;margin:6px 2px 0}
.detailhead{display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:8px;margin:2px 2px 8px}
.detailhead .nm{font-size:15px;font-weight:700}.pos{color:var(--long)}.neg{color:var(--short)}
.tip{position:fixed;pointer-events:none;background:var(--ink);color:var(--bg);padding:6px 9px;border-radius:6px;font-size:12px;opacity:0;transition:opacity .08s;z-index:9;white-space:nowrap;font-variant-numeric:tabular-nums}
</style></head><body><div class="wrap">
<h1>Position behaviour — long / short timeline · __NT__ days</h1>
<p class="sub">Each row is one instrument; each column a day. <b style="color:var(--long)">Green = long</b>,
<b style="color:var(--short)">red = short</b>, faint = flat. Click a row for its price + long/short bands + entry markers.</p>

<div class="bar">
  <div class="seg" id="stratseg">
    <button data-s="SAFE">SAFE</button><button data-s="QUAL">QUAL</button><button data-s="SWING">SWING</button><button data-s="LLALGO">LLALGO</button><button data-s="LLDOLLAR">LLDOLLAR</button><button data-s="LLVOL" class="on">LLVOL&nbsp;(v+m)</button><button data-s="LLVOL_VO">LLVOL&middot;VO</button>
  </div>
  <div class="legend">
    <span><span class="sw" style="background:var(--long)"></span>long</span>
    <span><span class="sw" style="background:var(--short)"></span>short</span>
    <span><span class="sw" style="background:var(--flat);border:1px solid var(--border)"></span>flat</span>
  </div>
  <span class="stat" id="hmstat"></span>
</div>

<div class="card"><canvas id="heat" height="760"></canvas></div>
<p class="hint">Row 0 = ALGO index (the leg we tune). Hover for the day &amp; position · click a row to inspect it below.</p>

<div class="detailhead" style="margin-top:18px"><span class="nm">ALGO index leg &mdash; vol+momentum vs vol-only ($ position)</span><span class="stat" id="algostat"></span></div>
<div class="card"><canvas id="algoleg" height="150"></canvas></div>
<p class="hint">The ALGO index (row 0) is the <b>only</b> leg where the two books differ &mdash; instruments 1&ndash;49 are identical.
<b style="color:var(--pnl)">Blue = LLVOL (v+m)</b>, <b style="color:var(--vo)">amber = LLVOL&middot;VO (vol-only)</b>. Above 0 = long ALGO, below = short.
Where blue runs <i>bigger</i> than amber, the momentum leg is confirming the vol side (conviction up); where they sit on opposite sides of 0, momentum flipped the position. Hover for both values.</p>

<div class="detailhead" style="margin-top:20px"><span class="nm" id="dnm">ALGO — click a row above</span><span class="stat" id="dstat"></span></div>
<div class="card">
  <canvas id="price" height="260"></canvas>
  <canvas id="posline" height="90"></canvas>
</div>
<p class="hint">Top: price with long (green) / short (red) shading + ▲/▼ entries. Bottom: the signed position over time (above 0 = long, below = short). Drag on the price chart to zoom.</p>

<div class="tip" id="tip"></div>
</div>
<script>
const DATA=__DATA__, N=DATA.names.length, NT=DATA.nt;
let strat="LLVOL", asset=0, x0=0, x1=NT-1;
const css=k=>getComputedStyle(document.body).getPropertyValue(k).trim();
const tip=document.getElementById('tip');
function dpi(cv,h){const r=devicePixelRatio||1,w=cv.clientWidth;cv.width=w*r;cv.height=h*r;const c=cv.getContext('2d');c.setTransform(r,0,0,r,0,0);return[c,w,h];}
function niceTicks(a,b){const span=b-a,step=span<=120?20:span<=300?50:100,t=[];for(let d=Math.ceil(a/step)*step;d<=b;d+=step)t.push(d);return t;}

// ---------- heatmap ----------
function renderHeat(){
  const cv=document.getElementById('heat'),H=760,[c,W]=dpi(cv,H);
  const padL=62,padR=8,padT=6,padB=22,rowH=(H-padT-padB)/N,plotW=W-padL-padR;
  const S=DATA[strat].sign;
  c.clearRect(0,0,W,H);
  c.font="10px sans-serif";c.textAlign="right";
  for(let i=0;i<N;i++){
    const y=padT+i*rowH, sg=S[i];
    // draw runs of same sign as filled segments
    let d=0;
    while(d<NT){const s=sg[d];let e=d;while(e+1<NT&&sg[e+1]===s)e++;
      const xa=padL+d/NT*plotW,xb=padL+(e+1)/NT*plotW;
      c.fillStyle=s>0?css('--long'):s<0?css('--short'):css('--flat');
      c.fillRect(xa,y,Math.max(0.5,xb-xa),Math.max(1,rowH-0.6));d=e+1;}
    c.fillStyle=(i===asset)?css('--pnl'):css('--ink2');
    c.fillText(DATA.names[i]+(i===0?'*':''),padL-5,y+rowH-2);
    if(i===asset){c.strokeStyle=css('--pnl');c.lineWidth=1.5;c.strokeRect(padL,y,plotW,rowH);}
  }
  c.strokeStyle=css('--muted');c.fillStyle=css('--muted');c.textAlign="center";c.font="10px sans-serif";
  for(const d of niceTicks(0,NT-1)){const x=padL+d/NT*plotW;c.fillText(d,x,H-6);}
  cv._geo={padL,padT,rowH,plotW};
  // stats
  const sg=DATA[strat].sign; let lng=0,sht=0,cells=0;
  for(let i=0;i<N;i++)for(let d=130;d<NT;d++){const s=sg[i][d];if(s>0)lng++;else if(s<0)sht++;cells++;}
  document.getElementById('hmstat').innerHTML=`${strat} · <b class="pos">${(100*lng/cells).toFixed(0)}%</b> long / <b class="neg">${(100*sht/cells).toFixed(0)}%</b> short cells`;
}
(function(){const cv=document.getElementById('heat');
  cv.addEventListener('mousemove',e=>{const g=cv._geo;if(!g)return;const r=cv.getBoundingClientRect();
    const mx=e.clientX-r.left,my=e.clientY-r.top;const i=Math.floor((my-g.padT)/g.rowH),d=Math.floor((mx-g.padL)/g.plotW*NT);
    if(i<0||i>=N||d<0||d>=NT){tip.style.opacity=0;return;}
    const s=DATA[strat].sign[i][d];const st=s>0?'LONG':s<0?'SHORT':'flat';
    tip.innerHTML=`<b>${DATA.names[i]}</b> · day ${d}<br><b style="color:${s>0?css('--long'):s<0?css('--short'):css('--muted')}">${st}</b> · PnL $${DATA[strat].pnl[i][d].toLocaleString()}`;
    tip.style.left=(e.clientX+14)+'px';tip.style.top=(e.clientY+14)+'px';tip.style.opacity=1;});
  cv.addEventListener('mouseleave',()=>tip.style.opacity=0);
  cv.addEventListener('click',e=>{const g=cv._geo;if(!g)return;const r=cv.getBoundingClientRect();
    const i=Math.floor((e.clientY-r.top-g.padT)/g.rowH);if(i>=0&&i<N){asset=i;x0=0;x1=NT-1;renderHeat();renderDetail();}});
})();

// ---------- detail: price + bands + entries, and the signed-position line ----------
function drawBands(c,W,H,pad,sign,xs){let d=x0;while(d<=x1){let s=sign[d];if(s===0){d++;continue;}let e=d;while(e+1<=x1&&sign[e+1]===s)e++;
  c.fillStyle=s>0?css('--longband'):css('--shortband');const xa=xs(d),xb=xs(e+1>x1?x1:e+1);c.fillRect(xa,pad.t,Math.max(1,xb-xa),H-pad.t-pad.b);d=e+1;}}
function renderPrice(){
  const cv=document.getElementById('price'),[c,W,H]=dpi(cv,260),pad={l:52,r:12,t:10,b:20};
  const pr=DATA.prices[asset],S=DATA[strat];let lo=Infinity,hi=-Infinity;for(let d=x0;d<=x1;d++){lo=Math.min(lo,pr[d]);hi=Math.max(hi,pr[d]);}
  const pd=(hi-lo)||1;lo-=pd*.06;hi+=pd*.06;
  const xs=d=>pad.l+(d-x0)/((x1-x0)||1)*(W-pad.l-pad.r),ys=v=>pad.t+(hi-v)/(hi-lo)*(H-pad.t-pad.b);
  c.clearRect(0,0,W,H);drawBands(c,W,H,pad,S.sign[asset],xs);
  c.strokeStyle=css('--grid');c.fillStyle=css('--muted');c.font="11px sans-serif";c.textAlign="right";
  for(let i=0;i<=4;i++){const v=lo+(hi-lo)*i/4,y=ys(v);c.beginPath();c.moveTo(pad.l,y);c.lineTo(W-pad.r,y);c.stroke();c.fillText(v.toFixed(1),pad.l-6,y+3);}
  c.textAlign="center";for(const d of niceTicks(x0,x1))c.fillText(d,xs(d),H-5);
  c.strokeStyle=css('--ink');c.lineWidth=1.5;c.beginPath();for(let d=x0;d<=x1;d++){const X=xs(d),Y=ys(pr[d]);d===x0?c.moveTo(X,Y):c.lineTo(X,Y);}c.stroke();
  const evs=S.events[asset],span=x1-x0;if(span<=420)for(const[d,dir]of evs){if(d<x0||d>x1)continue;const X=xs(d),Y=ys(pr[d]);
    c.fillStyle=dir>0?css('--long'):css('--short');c.beginPath();
    if(dir>0){c.moveTo(X,Y-10);c.lineTo(X-5,Y-3);c.lineTo(X+5,Y-3);}else{c.moveTo(X,Y+10);c.lineTo(X-5,Y+3);c.lineTo(X+5,Y+3);}c.closePath();c.fill();}
  cv._xs=xs;cv._pad=pad;
}
function renderPosLine(){
  const cv=document.getElementById('posline'),[c,W,H]=dpi(cv,90),pad={l:52,r:12,t:8,b:16};
  const sg=DATA[strat].sign[asset];
  const xs=d=>pad.l+(d-x0)/((x1-x0)||1)*(W-pad.l-pad.r),ys=v=>pad.t+(1-v)/2*(H-pad.t-pad.b);
  c.clearRect(0,0,W,H);const z=ys(0);
  c.strokeStyle=css('--grid');c.beginPath();c.moveTo(pad.l,z);c.lineTo(W-pad.r,z);c.stroke();
  c.fillStyle=css('--muted');c.font="10px sans-serif";c.textAlign="right";c.fillText('long',pad.l-6,ys(1)+8);c.fillText('short',pad.l-6,ys(-1)+2);
  // step line of position sign, colored by side
  for(let d=x0;d<x1;d++){const s=sg[d];if(s===0)continue;const X=xs(d),X2=xs(d+1),Y=ys(s);
    c.strokeStyle=s>0?css('--long'):css('--short');c.lineWidth=2;c.beginPath();c.moveTo(X,Y);c.lineTo(X2,Y);c.stroke();}
}
function renderDetail(){
  const S=DATA[strat],pnl=S.pnl_final[asset],evs=S.events[asset];
  const longs=S.sign[asset].filter(x=>x>0).length,shorts=S.sign[asset].filter(x=>x<0).length;
  document.getElementById('dnm').textContent=DATA.names[asset]+(asset===0?"  (ALGO = index)":"");
  document.getElementById('dstat').innerHTML=`${strat} · <b>${evs.length}</b> entries · days long <b>${longs}</b> / short <b>${shorts}</b> · final PnL <b class="${pnl>=0?'pos':'neg'}">$${pnl.toLocaleString()}</b>`;
  renderPrice();renderPosLine();
}
// hover + zoom on price
(function(){const cv=document.getElementById('price');cv.addEventListener('mousemove',e=>{const xs=cv._xs,pad=cv._pad;if(!xs)return;
  const mx=e.clientX-cv.getBoundingClientRect().left;let d=Math.round(x0+(mx-pad.l)/((cv.clientWidth-pad.l-pad.r))*(x1-x0));d=Math.max(x0,Math.min(x1,d));
  const s=DATA[strat].sign[asset][d],st=s>0?'LONG':s<0?'SHORT':'flat';
  tip.innerHTML=`day ${d} · ${DATA.names[asset]}<br>price $${DATA.prices[asset][d].toFixed(2)} · <b style="color:${s>0?css('--long'):css('--short')}">${st}</b>`;
  tip.style.left=(e.clientX+14)+'px';tip.style.top=(e.clientY+14)+'px';tip.style.opacity=1;});
  cv.addEventListener('mouseleave',()=>tip.style.opacity=0);
  let drag=false,sx=0;cv.addEventListener('mousedown',e=>{drag=true;sx=e.clientX-cv.getBoundingClientRect().left;});
  window.addEventListener('mouseup',e=>{if(!drag)return;drag=false;const pad=cv._pad;const ex=e.clientX-cv.getBoundingClientRect().left;
    const toD=px=>Math.round(x0+(px-pad.l)/((cv.clientWidth-pad.l-pad.r))*(x1-x0));let a=toD(Math.min(sx,ex)),b=toD(Math.max(sx,ex));
    if(b-a>=8){x0=Math.max(0,a);x1=Math.min(NT-1,b);renderDetail();}});
})();

document.getElementById('stratseg').addEventListener('click',e=>{if(!e.target.dataset.s)return;strat=e.target.dataset.s;
  [...e.currentTarget.children].forEach(b=>b.classList.toggle('on',b.dataset.s===strat));renderHeat();renderDetail();});

// ---------- ALGO leg overlay: v+m vs vol-only $ position ----------
function renderAlgoLeg(){
  const cv=document.getElementById('algoleg'),[c,W,H]=dpi(cv,150),pad={l:60,r:12,t:10,b:20};
  const A=DATA.LLVOL.algo_dollar,B=DATA.LLVOL_VO.algo_dollar;
  let m=1;for(let d=0;d<NT;d++)m=Math.max(m,Math.abs(A[d]),Math.abs(B[d]));m=Math.ceil(m/25000)*25000;
  const xs=d=>pad.l+d/(NT-1)*(W-pad.l-pad.r),ys=v=>pad.t+(m-v)/(2*m)*(H-pad.t-pad.b);
  c.clearRect(0,0,W,H);
  c.strokeStyle=css('--grid');c.font="10px sans-serif";c.textAlign="right";
  for(const v of [m,m/2,0,-m/2,-m]){const y=ys(v);c.beginPath();c.moveTo(pad.l,y);c.lineTo(W-pad.r,y);c.stroke();
    c.fillStyle=css('--muted');c.fillText((v>0?'+':'')+'$'+(v/1000)+'k',pad.l-6,y+3);}
  c.strokeStyle=css('--ink2');c.lineWidth=1.1;c.beginPath();c.moveTo(pad.l,ys(0));c.lineTo(W-pad.r,ys(0));c.stroke();
  c.fillStyle=css('--muted');c.textAlign="center";for(const d of niceTicks(0,NT-1))c.fillText(d,xs(d),H-5);
  const line=(arr,col,w)=>{c.strokeStyle=css(col);c.lineWidth=w;c.beginPath();
    for(let d=0;d<NT;d++){const X=xs(d),Y=ys(arr[d]);d===0?c.moveTo(X,Y):c.lineTo(X,Y);}c.stroke();};
  line(B,'--vo',1.4);line(A,'--pnl',1.8);
  let sa=0,sb=0,n=0;for(let d=130;d<NT;d++){sa+=Math.abs(A[d]);sb+=Math.abs(B[d]);n++;}
  document.getElementById('algostat').innerHTML=`mean |$|: <b style="color:var(--pnl)">$${Math.round(sa/n/1000)}k</b> v+m &middot; <b style="color:var(--vo)">$${Math.round(sb/n/1000)}k</b> vol-only`;
  cv._geoA={pad,xs,A,B};
}
(function(){const cv=document.getElementById('algoleg');
  cv.addEventListener('mousemove',e=>{const g=cv._geoA;if(!g)return;const r=cv.getBoundingClientRect();
    let d=Math.round((e.clientX-r.left-g.pad.l)/((cv.clientWidth-g.pad.l-g.pad.r))*(NT-1));d=Math.max(0,Math.min(NT-1,d));
    tip.innerHTML=`day ${d}<br><b style="color:${css('--pnl')}">v+m $${g.A[d].toLocaleString()}</b><br><b style="color:${css('--vo')}">vol-only $${g.B[d].toLocaleString()}</b>`;
    tip.style.left=(e.clientX+14)+'px';tip.style.top=(e.clientY+14)+'px';tip.style.opacity=1;});
  cv.addEventListener('mouseleave',()=>tip.style.opacity=0);
})();

function renderAll(){renderHeat();renderDetail();renderAlgoLeg();}
renderAll();
window.addEventListener('resize',renderAll);
new MutationObserver(renderAll).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
</script></body></html>"""
open("behavior.html","w").write(HTML.replace("__DATA__", json.dumps(DATA, separators=(",", ":"))).replace("__NT__", str(DATA["nt"])))
print("wrote behavior.html", f"({len(HTML)//1024} KB base)")
