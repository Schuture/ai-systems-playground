/* Teaser for the agent-anatomy lab's home tile: one slider steps through the 5 model calls of a small made-up agent task (not the
   lab's bug fix, so the tile does not answer part 1's prediction, R3.8). Each row is one call: everything that call sends to the
   model, stacked by who wrote it (your code outlined, a tool hatched, the model solid). Calls not made yet are dashed outlines.
   Tokens: the run is illustrative; each call sends every block so far, so call k's row is call k − 1's row plus the new reply and
   tool result. State lives in module scope, so a re-render (domain or language switch) keeps the learner's position. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const esc=A.esc;
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh):en;
/* the run: system prompt and tools (3,000 tokens) and the task (30), then per call the model's reply and the tool's result */
const START={you:3030};
const CALLS=[{tool:'read_file',use:25,res:1500},{tool:'search',use:20,res:300},{tool:'edit_file',use:60,res:100},{tool:'run_tests',use:20,res:400},{tool:null,use:40,res:0}];
let k=0;
const fN=n=>n.toLocaleString('en-US');
/* what call i (0-based) sends: your code's blocks, the tool results and the model's replies before it */
function sent(i){let tool=0, model=0; for(let j=0;j<i;j++){tool+=CALLS[j].res; model+=CALLS[j].use;} return {you:START.you,tool,model,all:START.you+tool+model};}
const total=()=>CALLS.reduce((a,c,i)=>a+sent(i).all,0);
function readHTML(i){const s=sent(i), c=CALLS[i], n=fN(s.all);
  if(i===0) return tx(`Call 1: your code sends the system prompt, the tools and the task, <b>${n} tokens</b>. The model asks to read a file.`,
                      `第 1 次调用：你的代码发出 system prompt、工具说明和任务，共 <b>${n} 个 token</b>。模型要求读一个文件。`);
  if(!c.tool) return tx(`Call 5 sends <b>${n} tokens</b>, and the model answers: done. The 5 calls read <b>${fN(total())} tokens</b> in all.`,
                        `第 5 次调用发出 <b>${n} 个 token</b>，模型回答：做完了。5 次调用一共读了 <b>${fN(total())} 个 token</b>。`);
  return tx(`Call ${i+1} resends everything so far, <b>${n} tokens</b>. The model asks for ${c.tool}; your code runs it.`,
            `第 ${i+1} 次调用把之前的全部重发一遍，共 <b>${n} 个 token</b>。模型要求调用 ${c.tool}，由你的代码执行。`);}
function notes(){
  return [tx('Illustrative: a made-up task in the message format of a tool-use API. The model only asks for tools; your code runs them and sends the results back.',
             '示意：一个虚构的任务，用的是工具调用 API 的消息格式。模型只提出要调用哪个工具，执行工具、把结果发回去的是你的代码。'),
    tx('Tokens are counted at about 4 characters each. The lab follows one bug fix, call by call.',
       'token 按大约 4 个字符算 1 个。Lab 里一步一步看一个修 bug 的完整过程。')];}
function visHTML(i){const mx=sent(CALLS.length-1).all;
  return `<div class="tz-ana-rows">`+CALLS.map((c,j)=>{const s=sent(j), on=j===i, done=j<=i, W=v=>(v/mx*100).toFixed(2);
    const bar=done?`<i class="tz-ana-you" style="width:${W(s.you)}%"></i><i class="tz-ana-tool" style="width:${W(s.tool)}%"></i><i class="tz-ana-model" style="width:${W(s.model)}%"></i>`
                  :`<i class="tz-ana-todo" style="width:${W(s.all)}%"></i>`;
    return `<div class="tz-ana-row${on?' on':''}${done?'':' todo'}"><span class="tz-ana-lab">${esc(tx(`call ${j+1}`,`第 ${j+1} 次`))}</span><span class="tz-ana-trk">${bar}</span><span class="tz-ana-v">${done?fN(s.all):''}</span></div>`;}).join('')+`</div>`;}
A.registerTeaser('agent-anatomy',{fid:'illus',render(el){
  const lab=tx('Model call','第几次模型调用');
  el.innerHTML=`<div class="tz tz-ana">`+
    `<label class="tz-lab" for="tz-ana-in"><span>${esc(lab)}</span><output id="tz-ana-out"></output></label>`+
    `<input type="range" id="tz-ana-in" min="0" max="${CALLS.length-1}" step="1" value="${k}" aria-label="${esc(lab)}">`+
    `<div class="tz-vis tz-ana-vis" role="img" tabindex="0"></div>`+
    `<div class="tz-leg"><span><i class="sw tz-ana-swyou" aria-hidden="true"></i>${esc(tx('your code','你的代码'))}</span><span><i class="sw tz-ana-swtool" aria-hidden="true"></i>${esc(tx('tool results','工具结果'))}</span><span><i class="sw tz-ana-swmodel" aria-hidden="true"></i>${esc(tx('the model','模型'))}</span><span>${esc(tx('row = tokens one call sends','一行 = 一次调用发出的 token'))}</span></div>`+
    `<p class="tz-read"></p><p class="tz-note"></p><p class="tz-note"></p></div>`;
  const inp=el.querySelector('#tz-ana-in'), out=el.querySelector('#tz-ana-out'), vis=el.querySelector('.tz-ana-vis'), rd=el.querySelector('.tz-read');
  el.querySelectorAll('.tz-note').forEach((p,i)=>{p.textContent=notes()[i]||'';});
  const upd=user=>{const s=sent(k), o=tx(`call ${k+1} of ${CALLS.length}`,`第 ${k+1} / ${CALLS.length} 次`); out.textContent=o; inp.value=k; inp.setAttribute('aria-valuetext',o); if(A.ui.setRange) A.ui.setRange(inp);
    vis.innerHTML=visHTML(k);
    const help=tx(`Call ${k+1} sends ${fN(s.all)} tokens: ${fN(s.you)} written by your code, ${fN(s.tool)} of tool results, ${fN(s.model)} written by the model. Each row is one call; later calls are dashed.`,
                  `第 ${k+1} 次调用发出 ${fN(s.all)} 个 token：你的代码写的 ${fN(s.you)} 个，工具结果 ${fN(s.tool)} 个，模型写的 ${fN(s.model)} 个。每一行是一次调用，还没发生的画成虚线。`);
    vis.setAttribute('aria-label',help); vis.setAttribute('data-tipx',help);
    const h=readHTML(k); if(rd._h!==h){rd._h=h; rd.innerHTML=h; if(user) A.announce(rd.textContent);}};
  inp.addEventListener('input',()=>{const v=Math.max(0,Math.min(CALLS.length-1,Math.round(+inp.value))); if(v!==k){k=v; upd(true);}});
  upd(false);
}});
/* read-only hook for tests/agent-anatomy_numbers.py */
window.__tzAna=Object.freeze({calls:()=>CALLS.map(c=>({...c})),start:()=>({...START}),sent:i=>({...sent(i)}),total:()=>total(),read:i=>readHTML(i)});
})();
