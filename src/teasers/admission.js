/* teaser for admission: a day of chat traffic against the capacity of 4 or 6 H100s, and 4 H100s with a gateway's rules.
   Numbers from the lab's own engine (window.ADM_ENGINE); the heavy runs are computed after the first paint. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
let v='4', T0=0;
const T={paid:40,free:85,script:25};
const SET={'4':{n:4,q:0,rl:0,prio:false,th:0.6},'6':{n:6,q:0,rl:0,prio:false,th:0.6},'4r':{n:4,q:200,rl:60,prio:true,th:0.6}};
const tx=(en,zh)=>A.lang()==='zh'?zh:en;
const fP=x=>`${x>=1?100:x<=0?0:Math.min(99,Math.max(1,Math.round(x*100)))}%`;
function chart(E,s){
  const d=E.day(T,s), W=300, H=110, LW=4, top=8, bot=16, pw=W-LW-4, ph=H-top-bot, bw=pw/24, ymax=Math.max(d.cap,...d.hours.map(x=>x.c))*1.12, Y=y=>top+ph*(1-y/ymax), out=[];
  d.hours.forEach(x=>{const bx=LW+x.h*bw+bw*0.12, w=bw*0.76, sc=s.rl?Math.min(T.script,s.rl/60):T.script, pa=E.PROF[x.h]*T.paid, fr=E.PROF[x.h]*T.free; let y0=0;
    for(const [val,c] of [[pa,'tz-adm-pa'],[fr,'tz-adm-fr'],[sc,'tz-adm-sc']]){out.push(`<rect class="${c}" x="${bx.toFixed(1)}" y="${Y(y0+val).toFixed(1)}" width="${w.toFixed(1)}" height="${(Y(y0)-Y(y0+val)).toFixed(1)}"/>`); y0+=val;}
    if(x.c>d.cap) out.push(`<rect class="tz-adm-ov" x="${bx.toFixed(1)}" y="${Y(x.c).toFixed(1)}" width="${w.toFixed(1)}" height="${(Y(d.cap)-Y(x.c)).toFixed(1)}"/>`);});
  out.push(`<line class="tz-adm-cap" x1="${LW}" x2="${LW+pw}" y1="${Y(d.cap).toFixed(1)}" y2="${Y(d.cap).toFixed(1)}"/>`);
  for(const h of [0,6,12,18]) out.push(`<text x="${(LW+h*bw+bw/2).toFixed(1)}" y="${H-3}" text-anchor="${h===0?'start':'middle'}">${h}:00</text>`);
  return `<svg class="tz-adm-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="${tx(`Chat requests per second each hour against the capacity of ${s.n} GPUs`,`每个小时每秒的聊天请求，和 ${s.n} 张卡的容量`)}">${out.join('')}</svg>`;}
function read(E){const s=SET[v], m=E.peakRun(T,s).m;
  if(v==='4') return tx(`Sized for the average, 4 GPUs fall behind every afternoon: at the peak only ${fP(m.paid.frac)} of paid chats are answered on time.`,`按平均配 4 张卡，每天下午都接不住：高峰时只有 ${fP(m.paid.frac)} 的付费聊天按时拿到回答。`);
  if(v==='6') return tx(`Sized for the peak, 6 GPUs answer ${fP(m.paid.frac)} of paid chats on time at the peak, and much of the fleet sits idle at night.`,`按高峰配 6 张卡，高峰时 ${fP(m.paid.frac)} 的付费聊天按时拿到回答，夜里卡有不少闲着。`);
  return tx(`The same GPUs with rules: paid chats go first and the extra free requests get 429 at once, so ${fP(m.paid.frac)} of paid chats are on time.`,`同样的卡加上规则：付费聊天先走，多出来的免费请求马上拿到 429，${fP(m.paid.frac)} 的付费聊天按时拿到回答。`);}
function render(el){
  const E=window.ADM_ENGINE; if(!E){el.innerHTML=''; return;}
  const ready=!!el._ready;
  el.innerHTML=`<div class="tz tz-adm">`+
   A.ui.seg({id:'tz-adm-seg',key:'tzadm',label:tx('The fleet at the 14:00 peak','下午 2 点高峰时的配置'),options:[['4',tx('4 GPUs','4 张卡')],['6',tx('6 GPUs','6 张卡')],['4r',tx('4 GPUs + rules','4 张卡 + 规则')]],value:v})+
   `<div class="tz-vis tz-adm-vis">${ready?chart(E,SET[v]):''}</div>`+
   `<div class="tz-leg"><span><i class="tz-adm-sw tz-adm-pa"></i>${tx('paid','付费')}</span><span><i class="tz-adm-sw tz-adm-fr"></i>${tx('free','免费')}</span><span><i class="tz-adm-sw tz-adm-sc"></i>${tx('script','脚本')}</span><span><i class="tz-adm-sw tz-adm-ov"></i>${tx('above capacity','超过容量')}</span><span><i class="tz-adm-sw tz-adm-cl"></i>${tx('capacity','容量')}</span></div>`+
   `<p class="tz-read">${ready?read(E):tx('Simulating the busiest minute…','正在模拟最忙的那一分钟……')}</p>`+
   `<p class="tz-note">${tx('Toy run: Llama-3.1-8B on H100s, this lab’s own day of traffic (at 14:00, 40 paid and 85 free chats per second plus a script sending 25), priced by the lab’s step model. Rules: paid first, at most 200 free requests waiting (more get 429), 60 requests a minute per key. Not a recording.','玩具实跑：H100 上的 Llama-3.1-8B，这个 Lab 自己定的一天流量（下午 2 点每秒 40 个付费、85 个免费聊天请求，加上一个每秒发 25 个的脚本），按 Lab 的步模型计时。规则：付费优先，免费请求最多排 200 个（再来的回 429），每个 key 每分钟 60 个。不是真实录制。')}</p></div>`;
  el.querySelector('.seg-ctl').addEventListener('click',e=>{const b=e.target.closest('button[data-v]'); if(!b) return; v=b.dataset.v; el._ready=false; render(el); later(el);});
  if(!ready) later(el);}
function later(el){clearTimeout(T0); T0=setTimeout(()=>{const E=window.ADM_ENGINE; if(!E||!el.isConnected) return; E.goodput(); E.peakRun(T,SET[v]); el._ready=true; render(el);},60);}
A.registerTeaser('admission',{fid:'toy',render});
})();
