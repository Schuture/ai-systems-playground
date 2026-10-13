/* Teaser for the prompt-injection lab's home tile (illustrative, no numbers): one switch, "Can the agent send email?", on the
   same poisoned page. The visual shows what the agent reads (the page hatched as untrusted, the hidden line described, never
   quoted as an instruction), the tools it has, and its next action. Yes: the hijacked send runs and the files leave (red, ✗). No: the
   same hijack has no way out. The injected instruction is red with ⚠, as in the lab's replay (R5.11). One control, one small visual,
   one sentence in .tz-read, a one-line legend, plain-text notes. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const esc=A.esc;
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh):en;
let send=1;
function readHTML(v){
  return v?tx('When the agent can send email, a hidden line can make it <b>mail the user’s files to drop@example.com</b>.',
              'agent 能发邮件时，藏着的一句话就能让它<b>把用户的文件发到 drop@example.com</b>。')
          :tx('When the agent cannot send email, <b>the files cannot leave</b>, even if it acts on the hidden line.',
              'agent 不能发邮件时，就算照着那句话做，<b>文件也发不出去</b>。');
}
function notes(){
  return [tx('Illustrative, not a recorded run: whether a model acts on such a line depends on the model and the wording. The hidden line is described here, not quoted. example.com is reserved for examples (RFC 2606).',
             '示意，不是真实的运行记录：模型会不会照做，取决于模型和措辞。藏着的那句话这里只描述，不照抄。example.com 是专供举例的域名（RFC 2606）。'),
          tx('Indirect prompt injection: instructions planted in content an agent reads, such as a web page or an email (Greshake et al., arXiv 2302.12173). Without a way out, the files cannot leave, whatever the model does.',
             '间接提示注入：把指令藏进 agent 会读的内容里，比如网页或邮件（Greshake 等，arXiv 2302.12173）。拿掉往外发的路，不管模型怎么做，文件都出不去。')];
}
function visHTML(v){
  const tool=(n,on)=>`<code class="tz-pin-tool${on?'':' off'}">${n}</code>`;
  return `<div class="tz-pin-page"><span class="tz-pin-k">${esc(tx('page','网页'))}</span><span>${esc(tx('…heats fast, a little loud…','……加热快，有点吵……'))} <mark class="tz-pin-hid">⚠ ${esc(tx('[a hidden line asks the agent to send the user’s files to drop@example.com]','［藏着一句话：要 agent 把用户的文件发到 drop@example.com］'))}</mark></span></div>`+
    `<div class="tz-pin-row"><span class="tz-pin-k">${esc(tx('tools','工具'))}</span><span>${tool('read_page',1)} ${tool('read_files',1)} ${tool('send_email',v)}</span></div>`+
    (v?`<div class="tz-pin-row tz-pin-bad"><span class="tz-pin-k">${esc(tx('next','下一步'))}</span><span>✗ <code>send_email</code> → drop@example.com${esc(tx(': the files leave','：文件漏出去了'))}</span></div>`
      :`<div class="tz-pin-row"><span class="tz-pin-k">${esc(tx('next','下一步'))}</span><span>⊘ ${esc(tx('no send_email: the files stay','没有 send_email：文件留在原处'))}</span></div>`);
}
A.registerTeaser('prompt-injection',{fid:'illus',render(el){
  const lab=tx('Can the agent send email?','agent 能发邮件吗？');
  el.innerHTML=`<div class="tz tz-pin">`+A.ui.seg({id:'tz-pin-send',key:'tzpin',label:lab,options:[[1,tx('Yes','能')],[0,tx('No','不能')]],value:send,inline:false})+
    `<div class="tz-vis tz-pin-vis" role="img" tabindex="0"></div>`+
    `<div class="tz-leg"><span><i class="sw tz-pin-sw-un" aria-hidden="true"></i>${esc(tx('untrusted page','不可信的网页'))}</span><span><i class="sw tz-pin-sw-hid" aria-hidden="true"></i>${esc(tx('⚠ injected instruction','⚠ 注入的指令'))}</span><span><i class="sw tz-pin-sw-bad" aria-hidden="true"></i>${esc(tx('✗ data left','✗ 数据漏出去了'))}</span></div>`+
    `<p class="tz-read"></p><p class="tz-note"></p><p class="tz-note"></p></div>`;
  const grp=el.querySelector('.seg-ctl'), vis=el.querySelector('.tz-pin-vis'), rd=el.querySelector('.tz-read');
  el.querySelectorAll('.tz-note').forEach((p,i)=>{p.textContent=notes()[i]||'';});
  const upd=user=>{if(grp) A.ui.setPressed(grp,String(send)); vis.innerHTML=visHTML(send);
    const help=send?tx('The agent reads the page with the hidden line, has a way to send, and sends the user’s files out.','agent 读到藏着话的网页，又有往外发的工具，于是把用户的文件发了出去。')
                   :tx('The agent reads the same page, but has no tool to send anything out.','agent 读到同一个网页，但没有往外发东西的工具。');
    vis.setAttribute('aria-label',help); vis.setAttribute('data-tipx',help);
    const h=readHTML(send); if(rd._h!==h){rd._h=h; rd.innerHTML=h; if(user) A.announce(rd.textContent);}};
  if(grp) grp.addEventListener('click',e=>{const b=e.target.closest('button[data-v]'); if(!b||!grp.contains(b)) return; const v=+b.dataset.v?1:0; if(v!==send){send=v; upd(true);}});
  upd(false);
}});
/* read-only hook for tests/prompt-injection_numbers.py */
window.__tzPin=Object.freeze({read:v=>readHTML(v),vis:v=>visHTML(v),notes});
})();
