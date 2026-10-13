/* Home-tile teaser for the tool-scheduling lab: one control (one tool per model call, or batches by dependency) over a small
   made-up task of five tool calls (not the lab's nine, so the tile gives away nothing of the lab's prediction). A small timeline:
   the model's row on top (dark blocks), one row per tool call (gray bars). Times are assumed: 1.5 s per model call, as in the lab. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const {tx,esc}=A;
const M=1.5;
const CALLS=[{n:{en:'Weather',zh:'查天气'},t:0.6,needs:[]},{n:{en:'Flights',zh:'查航班'},t:2.5,needs:[]},{n:{en:'Hotels',zh:'查酒店'},t:2.0,needs:[]},
  {n:{en:'Book flight',zh:'订机票'},t:3.0,needs:[1]},{n:{en:'Book hotel',zh:'订酒店'},t:2.0,needs:[2]}];
const r1=v=>Math.round(v*1000)/1000, f1=v=>v.toFixed(1);
/* each turn: a model call, then its tools all at once; one more model call answers at the end */
function plan(dep){const lv=[]; CALLS.forEach((c,i)=>{lv[i]=1+Math.max(0,...c.needs.map(j=>lv[j]));});
  const turns=dep?[...new Set(lv)].sort().map(l=>CALLS.map((c,i)=>i).filter(i=>lv[i]===l)):CALLS.map((c,i)=>[i]);
  let t=0; const mc=[], bars=[]; turns.forEach(tn=>{mc.push([t,t+M]); t=r1(t+M); let e=t; tn.forEach(i=>{bars[i]=[t,r1(t+CALLS[i].t)]; e=Math.max(e,r1(t+CALLS[i].t));}); t=e;});
  mc.push([t,r1(t+M)]); return {mc,bars,total:r1(t+M),n:mc.length};}
let si=0;
const T={render(el){
  const P=plan(si===1), Q=plan(si!==1), W=300, LW=62, X0=LW+4, X1=W-6, RH=17, H=RH*6+16, TM=20, X=t=>X0+(X1-X0)*t/TM;
  let h=''; for(let s=0;s<=TM;s+=5) h+=`<line x1="${X(s)}" y1="0" x2="${X(s)}" y2="${RH*6}" class="tz-tls-g"/><text x="${X(s)}" y="${RH*6+12}" class="tz-tls-ax" text-anchor="middle">${s}</text>`;
  h+=`<text x="${LW}" y="${RH/2+4}" class="tz-tls-l b" text-anchor="end">${esc(tx('Model','模型'))}</text>`;
  P.mc.forEach(([a,b])=>{h+=`<rect x="${X(a).toFixed(1)}" y="3" width="${(X(b)-X(a)).toFixed(1)}" height="${RH-6}" rx="2" class="tz-tls-mc"/>`;});
  CALLS.forEach((c,i)=>{const y=RH*(i+1); h+=`<text x="${LW}" y="${y+RH/2+4}" class="tz-tls-l" text-anchor="end">${esc(A.L(c.n))}</text>`;
    const [a,b]=P.bars[i]; h+=`<rect x="${X(a).toFixed(1)}" y="${y+3}" width="${(X(b)-X(a)).toFixed(1)}" height="${RH-6}" rx="2" class="tz-tls-b"/>`;});
  const read=si===1?tx(`By dependency: ${P.n} model calls and ${f1(P.total)} s instead of ${f1(Q.total)} s.`,`按依赖分批：${P.n} 次模型调用，${f1(P.total)} 秒；一个一个来要 ${f1(Q.total)} 秒，订到的一样。`)
                   :tx(`One tool per model call: ${P.n} model calls and ${f1(P.total)} s for five tool calls.`,`每次模型调用只要一个工具：5 个工具调用，${P.n} 次模型调用，${f1(P.total)} 秒。`);
  el.innerHTML=`<div class="tz">`+A.ui.seg({id:'tz-tls-s',key:'tzs',label:tx('How the calls go out','怎么发工具调用'),options:[['0',tx('One at a time','一个一个来')],['1',tx('By dependency','按依赖分批')]],value:String(si)})+
    `<div class="tz-vis"><svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}" role="img" aria-label="${esc(read)}">${h}</svg></div>`+
    `<div class="tz-leg"><span><i class="sw tz-tls-swm"></i>${tx('model call','模型调用')}</span><span><i class="sw tz-tls-swb"></i>${tx('tool call','工具调用')}</span></div>`+
    `<p class="tz-read">${esc(read)}</p>`+
    `<p class="tz-note">${esc(tx('Computed from assumed times, not a recording: each model call 1.5 s, and each tool takes as long as its bar (the axis is in seconds). Booking a flight needs the flight search’s result and booking a hotel the hotel search’s, so they wait for the next model call. Tools asked for in one reply run at the same time; your code sends their results back together, so the next model call waits for the slowest. One more model call writes the answer at the end.',
      '按假设的耗时算出来的，不是录像：每次模型调用 1.5 秒，每个工具的耗时就是它那根条的长度（横轴单位是秒）。订机票要用查航班的结果，订酒店要用查酒店的结果，所以要等下一次模型调用。一次回复里要的工具同时执行；你的代码把结果一起发回去，所以下一次模型调用等最慢的那个。最后还有一次模型调用给出答案。'))}</p></div>`;
  el.querySelectorAll('.seg-ctl[data-key="tzs"] button').forEach(b=>b.addEventListener('click',()=>{si=+b.dataset.v; T.render(el);}));}};
A.registerTeaser('tool-scheduling',T);
})();
