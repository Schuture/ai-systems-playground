/* The parallelism lab's home tile answers the lab's question, “How do you train a model that does not fit on one GPU?”, with one
   control: the number of GPUs a 13B model (13 × 10^9 parameters, a round size the lab itself does not use, R3.8) is sharded over
   with ZeRO-3. Seven bars, one per GPU count, show the model states each GPU keeps, 18 bytes per parameter ÷ GPUs (the lab's
   accounting: BF16 weights 2, FP32 gradients 4, FP32 copy of the weights 4, Adam's m 4 and v 4; Megatron-LM dist_optimizer.md, Hugging Face
   model_memory_anatomy.md), against the 80 GB line of one H100; what is above the line is red (it does not fit).
   Activations, logits and the layers in flight are left out and the note says so. State lives in module scope.
   window.__tzZr is a read-only hook for the tests. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const esc=A.esc;
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh):en;
const P=13e9, BPP=18, CAP=80e9, NS=[1,2,4,8,16,32,64], H=112, TOP=240e9;
let tzN=4;   // the tile opens on the answer: 4 GPUs, under the line
const per=n=>BPP*P/n;
const fGB=b=>{const v=b/1e9; return (v>=100?v.toFixed(0):v.toFixed(1))+' GB';};
const NB=' ';
function readHTML(n){const v=per(n), fits=v<=CAP;
  if(n===1) return tx(`On 1${NB}GPU there is nothing to split: all <b>${fGB(v)}</b> of 13B’s model states, far over 80${NB}GB.`,
                      `只有 1${NB}张卡，没法分：13B 的模型状态 <b>${fGB(v)}</b> 全在这张卡上，远超 80${NB}GB。`);
  return tx(`ZeRO-3 over ${n}${NB}GPUs: each keeps 1/${n} of 13B’s model states, <b>${fGB(v)}</b>; an 80${NB}GB GPU ${fits?'holds it':'cannot'}.`,
            `ZeRO-3 分到 ${n}${NB}张卡上：每张卡存 13B 模型状态的 1/${n}，<b>${fGB(v)}</b>，80${NB}GB 的卡${fits?'放得下':'放不下'}。`);}
function notes(){return [
  tx('Model states at 18 bytes per parameter: BF16 weights 2, FP32 gradients 4, FP32 copy of the weights 4, Adam’s m and v 4 each, as Megatron-LM and Hugging Face count them.',
     '模型状态按每个参数 18 字节算：BF16 权重 2、FP32 梯度 4、FP32 权重副本 4、Adam 的 m 和 v 各 4，和 Megatron-LM、Hugging Face 的算法一样。'),
  tx('13B means 13 × 10⁹ parameters, a round size. ZeRO-3 splits all the model states evenly over the GPUs: twice the GPUs, half as much on each.',
     '13B 指 13 × 10⁹ 个参数，是个取整的规模。ZeRO-3 把模型状态全部平均分到各张卡上，卡数多一倍，每张卡存的就少一半。'),
  tx('Computed from a formula. Activations, logits and the weights of the layers in use come on top; the lab adds them. GB = 10⁹ bytes.',
     '由公式算出。激活、logits 和正在用的几层权重还要另算，Lab 里会加上。GB = 10⁹ 字节。')];}
function legendHTML(){
  return `<span><i class="sw tz-zr-sw" aria-hidden="true"></i>${esc(tx('model states per GPU','每张卡的模型状态'))}</span>`+
    `<span><i class="sw over" aria-hidden="true"></i>${esc(tx('over 80 GB','超出 80 GB'))}</span>`+
    `<span><i class="sw lim" aria-hidden="true"></i>${esc(tx('80 GB: one H100','80 GB：一张 H100'))}</span>`;}
const helpText=n=>tx(`One bar per number of GPUs: its height is the model states each GPU keeps under ZeRO-3. The bar with its value on top is ${n} GPUs. The dashed line is one H100’s 80 GB, and red is what does not fit.`,
  `每根柱子是一种卡数，柱高是用 ZeRO-3 时每张卡存的模型状态。标着数值的那根是 ${n} 张卡。虚线是一张 H100 的 80 GB，红色是放不下的部分。`);
const f1=v=>(Math.round(v*10)/10).toFixed(1);
function svgOf(W,n){
  const top=14, bot=H-16, Y=b=>bot-Math.min(b,TOP)/TOP*(bot-top), cw=W/NS.length, bw=Math.min(30,cw*0.56);
  let h='', lab='';
  NS.forEach((k,i)=>{const v=per(k), x=i*cw+(cw-bw)/2, on=k===n, yv=Y(v), yc=Y(CAP);
    h+=`<rect x="${f1(x)}" y="${f1(yv)}" width="${f1(bw)}" height="${f1(bot-yv)}" class="tz-zr-bar${on?' on':''}"/>`;
    if(v>CAP) h+=`<rect x="${f1(x)}" y="${f1(yv)}" width="${f1(bw)}" height="${f1(yc-yv)}" class="tz-zr-over"/>`;
    h+=`<text x="${f1(x+bw/2)}" y="${H-3}" class="tz-zr-n${on?' on':''}" text-anchor="middle">${k}</text>`;
    /* the value sits above its bar; when that would put it on the 80 GB line, it moves just above the line */
    if(on){const ya=Math.max(top-2,yv-4), hit=ya>yc-2&&ya-10<yc+2;
      lab=`<text x="${f1(x+bw/2)}" y="${f1(hit?yc-4:ya)}" class="tz-zr-v" text-anchor="middle">${esc(fGB(v))}</text>`;}});
  h+=`<line x1="0" x2="${W}" y1="${f1(Y(CAP))}" y2="${f1(Y(CAP))}" class="tz-zr-cap"/>`;
  h+=`<line x1="0" x2="${W}" y1="${f1(bot)}" y2="${f1(bot)}" class="tz-zr-base"/>`;
  return h+lab;   /* the value last, over the 80 GB line, with a halo (css) so the line never strikes through it */
}
A.registerTeaser('parallelism',{render(el){
  el.innerHTML=`<div class="tz tz-zr">${A.ui.seg({key:'tz-zr',label:tx('GPUs, ZeRO-3, 13B model','卡数（ZeRO-3，13B 模型）'),options:NS.map(k=>[String(k),String(k)]),value:String(tzN),inline:false})}`+
    `<svg class="tz-vis tz-zr-vis" role="img" tabindex="0" xmlns="http://www.w3.org/2000/svg"></svg>`+
    `<div class="tz-leg">${legendHTML()}</div>`+
    /* the longest sentence sits invisibly in the same grid cell, so the tile keeps its height */
    `<div class="tz-zr-rd"><p class="tz-read"></p><p class="tz-zr-twin" aria-hidden="true"></p></div>`+
    `${notes().map(()=>'<p class="tz-note"></p>').join('')}</div>`;
  const q=s=>el.querySelector(s), seg=q('.seg-ctl'), vis=q('svg.tz-zr-vis'), rd=q('.tz-read'), twin=q('.tz-zr-twin');
  el.querySelectorAll('.tz-note').forEach((p,i)=>{p.textContent=notes()[i]||'';});
  const longest=NS.map(readHTML).sort((a,b)=>b.replace(/<[^>]+>/g,'').length-a.replace(/<[^>]+>/g,'').length)[0];
  const upd=user=>{A.ui.setPressed(seg,String(tzN));
    const W=Math.max(240,Math.round(el.clientWidth||300)), help=helpText(tzN);
    vis.setAttribute('viewBox',`0 0 ${W} ${H}`); vis.setAttribute('width',W); vis.setAttribute('height',H); vis.innerHTML=svgOf(W,tzN);
    vis.setAttribute('aria-label',help); vis.setAttribute('data-tipx',help);
    twin.innerHTML=longest;
    const h=readHTML(tzN); if(rd._h!==h){rd._h=h; rd.innerHTML=h; if(user) A.announce(rd.textContent);}};
  seg.addEventListener('click',e=>{const b=e.target.closest('button[data-v]'); if(!b||!seg.contains(b)) return; const v=+b.dataset.v; if(!NS.includes(v)) return; tzN=v; upd(true);});
  upd(false);
}});
window.__tzZr=Object.freeze({NS:NS.slice(),P,BPP,CAP,per:n=>per(n),read:n=>readHTML(n),notes:()=>notes(),get n(){return tzN;}});
})();
