/* Home-tile teaser for the product capstone: one control adds the design choices one at a time (send when done → stream → cache the
   shared system prompt → route easy messages to the small model) and two bars show the time to the first visible word and the GPU cost
   per day for 1M daily users, against the lab's budgets. The numbers come from the lab's own engine (window.__pdz), so tile and lab agree. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const {tx,esc}=A;
const OPTS=[{v:'0',t:()=>tx('Plain','最简单'),p:{stream:0}},{v:'1',t:()=>tx('+ stream','+ 流式'),p:{stream:1}},{v:'2',t:()=>tx('+ cache','+ 缓存'),p:{stream:1,cache:1}},{v:'3',t:()=>tx('+ route','+ 路由'),p:{stream:1,cache:1,share:0.7}}];
let si=1;
const T={fid:'toy',render(el){
  const E=window.__pdz; if(!E){el.innerHTML=''; return;}
  const C=E.consts, r=E.compute(OPTS[si].p), vis=r.vis, cost=r.costDay, VM=C.BRIEF.vis*2, CM=50000;
  const f=(v,d)=>A.fmt.num(v,d), usd=v=>'$'+A.fmt.int(Math.round(v)), W=300, x0=4, x1=W-4, sx=(v,m)=>x0+(x1-x0)*Math.min(v,m)/m;
  const bar=(y,v,m,goal,bad,label,val)=>`<text class="tz-pdz-l" x="${x0}" y="${y-4}">${esc(label)}</text><text class="tz-pdz-v${bad?' bad':''}" x="${x1}" y="${y-4}" text-anchor="end">${esc(val)}</text>`+
    `<rect class="tz-pdz-tr" x="${x0}" y="${y}" width="${x1-x0}" height="12" rx="3"/><rect class="tz-pdz-b${bad?' bad':''}" x="${x0}" y="${y}" width="${Math.max(2,sx(v,m)-x0).toFixed(1)}" height="12" rx="3"/>`+
    `<line class="tz-pdz-g" x1="${sx(goal,m)}" x2="${sx(goal,m)}" y1="${y-2}" y2="${y+14}"/>`;
  const vb=vis>C.BRIEF.vis+1e-9, cb=cost>C.BRIEF.budget+1e-9;
  const svg=bar(18,vis,VM,C.BRIEF.vis,vb,tx('first word on screen','首字可见时间'),vis>VM?tx(`${f(vis,1)} s (off scale)`,`${f(vis,1)} 秒（超出刻度）`):tx(`${f(vis,2)} s`,`${f(vis,2)} 秒`))+
    bar(62,cost,CM,C.BRIEF.budget,cb,tx('GPU cost per day','每天的 GPU 花费'),usd(cost));
  const step=OPTS[si].t();
  el.innerHTML=`<div class="tz">`+A.ui.seg({id:'tz-pdz-s',key:'tzs',label:tx('Design so far','设计到哪一步'),options:OPTS.map(o=>[o.v,o.t()]),value:String(si)})+
    `<div class="tz-vis"><svg viewBox="0 0 ${W} 84" width="100%" height="84" role="img" aria-label="${esc(tx(`First word after ${f(vis,2)} s; ${usd(cost)} a day.`,`首字 ${f(vis,2)} 秒；每天 ${usd(cost)}。`))}">${svg}</svg></div>`+
    `<div class="tz-leg"><span><i class="sw tz-pdz-swb"></i>${tx('this design','这个设计')}</span><span><i class="sw tz-pdz-swbad"></i>${tx('over budget','超出预算')}</span><span><i class="sw tz-pdz-swg"></i>${tx('budget','预算')}</span></div>`+
    `<p class="tz-read">${si===0?tx(`Sending each answer when it is done, 1M daily users wait <b>${f(vis,1)} s</b> for the first word, on <b>${usd(cost)}</b> a day of GPUs.`,`每个回答写完再发，每天 1M 用户要等 <b>${f(vis,1)} 秒</b>才看到第一个字，GPU 每天 <b>${usd(cost)}</b>。`)
      :tx(`With ${['','streaming','streaming and caching','streaming, caching and routing'][si]}, the first word appears after <b>${f(vis,2)} s</b>, on <b>${usd(cost)}</b> a day of GPUs.`,`加上${['','流式输出','流式输出和缓存','流式输出、缓存和路由'][si]}，第一个字 <b>${f(vis,2)} 秒</b>后出现，GPU 每天 <b>${usd(cost)}</b>。`)}</p>`+
    `<p class="tz-note">${tx(`Setup: a chat app for 1M daily users, 10 messages each, peak 3 times the average, every answer checked every 8 tokens. Latencies of the components are the lab’s assumptions; GPUs come from the serving capstone’s engine (Llama-3.1-70B on 4 H100s, Llama-3.1-8B on 1, sized so 90% of first tokens come within 1 s), at an assumed $3 per GPU hour. Budgets: first word ${f(C.BRIEF.vis,1)} s, ${usd(C.BRIEF.budget)} a day.`,`设定：每天 1M 用户的聊天应用，每人 10 条消息，高峰是平均的 3 倍，每个回答每 8 个 token 查一次。各部件的耗时是这个 Lab 的假设；卡数来自 Serving 综合挑战的引擎（Llama-3.1-70B 用 4 张 H100，Llama-3.1-8B 用 1 张，按 90% 的首 token 在 1 秒内配卡），每 GPU 小时按假设的 3 美元算。预算：首字 ${f(C.BRIEF.vis,1)} 秒，每天 ${usd(C.BRIEF.budget)}。`)}</p></div>`;
  el.querySelectorAll('.seg-ctl[data-key="tzs"] button').forEach(b=>b.addEventListener('click',()=>{si=+b.dataset.v; T.render(el);}));}};
A.registerTeaser('product-design',T);
})();
