"""The capability layer (Codex review 1 #5, Batch 2) in every lab: EN and ZH at 1440 x 900 and 390 x 844.
- Under the title, “You will be able to: …” / “学完你能：…”.
- In the challenge only, after the lab's own task, the card with three tabs:
  1 Explain it: at least 2 questions, 3 or more explanations each, one right; a wrong pick shows what it gets wrong, then why the right one
    is right; the right one is not always the first shown (over all the labs).
  2 A new case: at least 2 questions answered with a click; a “go” button, when there is one, opens the case on the stage.
  3 Go further: links to the product owner's interview notes on GitHub (README.md in EN, README.zh.md in ZH), opening in a new tab;
    “frontier lab” and no company name on the page.
- The verdict is on screen after each pick, and a pick never moves the page on a phone; answers survive a mode switch.
- Writing rules on everything the card shows (a typeset formula counts as one token): no “·” between clauses, no two colons in one Chinese sentence, a space between Chinese and
  Latin letters or digits.
Run: python3 tests/exits_flow.py [lab ids]"""
import pathlib, re, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
LABS = sys.argv[1:] or ['architecture', 'transformer', 'mha', 'kv-basics', 'gqa-mla', 'kv-cache', 'prefill-decode', 'paged-attention', 'continuous-batching', 'chunked-prefill', 'prefix-caching', 'dpo']
# with lab ids given, build only those labs (the KV cache lab is always in the page), into a file of their own, so runs for different labs can overlap
only = [x for x in LABS if x != 'kv-cache']
stem = 'exits' + ('-' + '-'.join(sys.argv[1:]) if sys.argv[1:] else '')
subprocess.run([sys.executable, str(P / 'build.py'), f'out/{stem}.html'] + (['--only', ','.join(only)] if sys.argv[1:] and only else []), check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / f'wrapped-{stem}.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / f'{stem}.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / f'wrapped-{stem}.html')
fails, n, right_pos = [], 0, []
COMPANY = re.compile(r'OpenAI|Anthropic|DeepMind|Google|Meta\b|Microsoft|xAI', re.I)
HREF = re.compile(r'^https://github\.com/Schuture/(OpenAI|Anthropic|Google-DeepMind)-Interview-Notes/blob/main/[a-z-]+/[a-z0-9-]+/README(\.zh)?\.md$')


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(f'{name} :: {str(info)[:300]}'); print('FAIL', name, '::', str(info)[:300])


RULES = r"""(el)=>{const out=[]; const tw=document.createTreeWalker(el,NodeFilter.SHOW_TEXT); let x;
 // a typeset formula (MathML, R6.29) counts as one token “F”: its “·” is a multiplication dot, not a clause joiner (2026-10-09). The text is
 // read like innerText (hidden parts left out) but walks the nodes, since a MathML element has no innerText of its own
 const txt=n=>{if(n.nodeType===3) return n.textContent; if(n.nodeType!==1) return ''; if(n.localName==='math') return ' F ';
   const cs=getComputedStyle(n); if(cs.display==='none'||cs.visibility==='hidden') return ''; let s=''; for(const c of n.childNodes) s+=txt(c);
   return /^(block|list-item|flex|grid|table)/.test(cs.display)?' '+s+' ':s;};
 el.querySelectorAll('p,li > a,li > span,button,.eyebrow').forEach(e=>{const t=txt(e).replace(/\s+/g,' ').trim();
   if(/ · /.test(t)) out.push('“·” joining parts: '+t.slice(0,90));
   for(const s of t.split(/(?<=[。？！])/)){if(/[\u4e00-\u9fff]/.test(s)&&(s.match(/：/g)||[]).length>=2) out.push('two colons: '+s.trim().slice(0,120));}});
 while((x=tw.nextNode())){const t=x.textContent; const m=t.match(/[\u4e00-\u9fff][A-Za-z0-9<√]|[A-Za-z0-9>%][\u4e00-\u9fff]/); if(m) out.push('no space between Chinese and Latin: '+t.trim().slice(0,80));}
 return out;}"""


def page(b, w, h, lang, lab):
    pg = b.new_page(viewport={'width': w, 'height': h})
    pg.set_default_timeout(8000)
    errs = []
    pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto(URL); pg.wait_for_timeout(250)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    pg.evaluate(f"()=>{{location.hash='{lab}'}}"); pg.wait_for_timeout(500)
    return pg, errs


def settle(pg, ms=2500):
    last, t = None, 0
    while t < ms:
        y = pg.evaluate('()=>scrollY')
        if y == last: return
        last = y; pg.wait_for_timeout(120); t += 120


CARD = "()=>[...document.querySelectorAll('[data-exits]')].find(e=>e.offsetParent!==null)||null"


def card_js(expr):
    return "()=>{const c=(" + CARD + ")(); if(!c) return null; " + expr + "}"


def mode(pg, m):
    pg.evaluate(f"()=>{{const b=[...document.querySelectorAll('.modes button[data-mode=\"{m}\"]')].find(e=>e.offsetParent!==null); b.click();}}"); pg.wait_for_timeout(400); settle(pg)


def quiz(pg, tag, t, lang, w, h):
    """answers every question of tab t (0 explain, 1 new case): a wrong pick first, then the right one"""
    pg.evaluate(card_js(f"c.querySelector('[data-ex=\"tab\"][data-i=\"{t}\"]').click(); return 1")); pg.wait_for_timeout(150)
    nq = pg.evaluate(card_js("const d=c.querySelectorAll('.ex-qs .pb-dot'); return d.length||1"))
    check(f'{tag} tab {t + 1}: at least 2 questions', nq >= 2, nq)
    for j in range(nq):
        if j:
            pg.evaluate(card_js(f"c.querySelector('.ex-qs .pb-dot[data-j=\"{j}\"]').click(); return 1")); pg.wait_for_timeout(150)
        opts = pg.evaluate(card_js("return [...c.querySelectorAll('.ex-opts .choice')].map(b=>({v:+b.dataset.v, t:b.textContent}))"))
        check(f'{tag} tab {t + 1} q{j + 1}: 3 or more options with radio dots', len(opts) >= 3 and pg.evaluate(card_js("return getComputedStyle(c.querySelector('.ex-opts .choice'),'::before').content!=='none'")), len(opts))
        ok_at, wrong_seen = None, False
        for k, o in enumerate(opts):
            # like a learner: the option on screen (in the middle, or for the first pick of the first question at the bottom edge), then a real click
            where = 'bottom' if (j == 0 and k == 0) else 'center'
            v = o['v']
            sel = f'[data-exits] .ex-opts .choice[data-v="{v}"]'
            if where == 'center':
                pg.evaluate(f"()=>document.querySelector('{sel}').scrollIntoView({{block:'center'}})")
            else:
                pg.evaluate(f"()=>{{const e=document.querySelector('{sel}'), r=e.getBoundingClientRect(); window.scrollBy(0, r.bottom-(innerHeight-30));}}")
            pg.wait_for_timeout(150)
            bb = pg.locator(sel).bounding_box()
            pg.mouse.click(bb['x'] + 20, bb['y'] + bb['height'] / 2); pg.wait_for_timeout(250); settle(pg)
            r = pg.evaluate(card_js(f"""const e=c.querySelector('.ex-res'); if(!e) return null; const b=e.getBoundingClientRect(), bar=document.querySelector('.topbar').getBoundingClientRect(), dk=document.getElementById('dock');
                const ob=c.querySelector('.ex-opts .choice[data-v="{v}"]').getBoundingClientRect(), fl=e.querySelector('p').getBoundingClientRect();
                return {{ok:e.classList.contains('ok'), ps:e.querySelectorAll('p').length, txt:e.innerText, top:Math.round(b.top), first:Math.round(fl.bottom), opt:Math.round(ob.top),
                  safe:Math.round(bar.bottom+(dk&&!dk.hidden?dk.getBoundingClientRect().height:0)), vh:innerHeight}}"""))
            if r is None:
                check(f'{tag} tab {t + 1} q{j + 1}: a pick shows a verdict', False); break
            check(f'{tag} tab {t + 1} q{j + 1} ({where}): after a pick, the verdict’s first line is on screen and the option picked stays below the bars',
                  r['top'] >= r['safe'] - 2 and r['first'] <= r['vh'] and r['opt'] >= r['safe'] - 2, r)
            if r['ok']:
                ok_at = k; break
            wrong_seen = True
            check(f'{tag} tab {t + 1} q{j + 1}: a wrong pick says “Not quite” and why', r['txt'].startswith('✗') and len(r['txt']) > 30, r['txt'][:80])
            if t == 0:
                check(f'{tag} explain q{j + 1}: a wrong pick says what it gets wrong, then why the right one is right (2 paragraphs)', r['ps'] >= 2, r['txt'][:160])
        check(f'{tag} tab {t + 1} q{j + 1}: one option is right', ok_at is not None, opts)
        if t == 0 and ok_at is not None:
            right_pos.append(ok_at)
        dots = pg.evaluate(card_js("return [...c.querySelectorAll('.ex-qs .pb-dot')].map(d=>d.textContent)"))
        if dots:
            check(f'{tag} tab {t + 1} q{j + 1}: its dot shows ✓', dots[j] == '✓', dots)
        for hit in pg.evaluate("el=>(" + RULES + ")(el)", pg.evaluate_handle(CARD)):
            if lang == 'zh' or hit.startswith('“·”'):
                check(f'{tag} tab {t + 1} q{j + 1}: writing rules', False, hit)
    return nq


with sync_playwright() as p:
    b = p.chromium.launch()
    for lab in LABS:
        for w, h in ((1440, 900), (390, 844)):
            for lang in ('en', 'zh'):
                tag = f'{lab} {w} {lang}'
                pg, errs = page(b, w, h, lang, lab)
                # since the 2026-10-22 refit the outcome and the minutes tag share one line, .lab-sub, right under the title
                oc = pg.evaluate("""()=>{const t=[...document.querySelectorAll('.labtitle')].find(e=>e.offsetParent!==null); if(!t) return null; let o=t.nextElementSibling;
                   if(o&&o.classList.contains('lab-sub')) o=o.firstElementChild;
                   return o&&o.classList.contains('lab-outcome')?o.innerText:''}""")
                check(f'{tag}: “You will be able to” sits right under the title', bool(oc) and oc.startswith('You will be able to:' if lang == 'en' else '学完你能：') and len(oc) > 30, oc)
                check(f'{tag}: no exits card in the guided tour', pg.evaluate(CARD) is None)
                mode(pg, 'challenge')
                has = pg.evaluate(card_js("return c.dataset.exits"))
                check(f'{tag}: the challenge has the card “Can you use it on a new case?”', has is not None, has)
                if has is None:
                    pg.close(); continue
                tabs = pg.evaluate(card_js("return [...c.querySelectorAll('[data-ex=\"tab\"]')].map(b=>b.textContent)"))
                check(f'{tag}: three tabs', tabs == (['1. Explain it', '2. A new case', '3. Go further'] if lang == 'en' else ['1. 讲清楚', '2. 新情况', '3. 延伸练习']), tabs)
                # no "To finish this lab" line (product owner, review of the Batch 3 preview: right or wrong, the learner moves on anyway)
                check(f'{tag}: no done line in the card', pg.evaluate(card_js("return c.querySelector('.ex-done')?1:0")) == 0)
                quiz(pg, tag, 0, lang, w, h)
                quiz(pg, tag, 1, lang, w, h)
                # go further
                pg.evaluate(card_js("c.querySelector('[data-ex=\"tab\"][data-i=\"2\"]').click(); return 1")); pg.wait_for_timeout(150)
                links = pg.evaluate(card_js("return [...c.querySelectorAll('.ex-links a')].map(a=>({h:a.getAttribute('href'), t:a.getAttribute('target'), r:a.getAttribute('rel'), x:a.innerText}))"))
                check(f'{tag}: go further lists 1 to 3 practice problems', 1 <= len(links) <= 3, links)
                for a in links:
                    check(f'{tag}: link {a["x"][:40]} goes to the notes, {"README.zh.md" if lang == "zh" else "README.md"}, in a new tab',
                          bool(HREF.match(a['h'] or '')) and (a['h'].endswith('README.zh.md') if lang == 'zh' else not a['h'].endswith('.zh.md')) and a['t'] == '_blank' and 'noopener' in (a['r'] or ''), a)
                txt = pg.evaluate(card_js("return c.innerText"))
                check(f'{tag}: “frontier lab” and no company name on the page', 'frontier-lab' in txt.lower() or 'frontier lab' in txt.lower(), txt[:200])
                check(f'{tag}: no company name in the card', not COMPANY.search(txt), (COMPANY.search(txt) or [''])[0])
                dr = pg.evaluate(card_js("const b=c.querySelector('[data-ex=\"drill\"]'); return b?b.textContent:null"))
                check(f'{tag}: go further points to the lab’s own interview drill at the bottom of the page', dr is not None, dr)
                if dr:
                    pg.evaluate(card_js("c.querySelector('[data-ex=\"drill\"]').click(); return 1")); pg.wait_for_timeout(700); settle(pg)
                    dt = pg.evaluate("()=>{const h=[...document.querySelectorAll('.c-card>h3')].find(x=>x.offsetParent!==null&&/Interview drill|面试题/.test(x.textContent)); if(!h) return null; const r=h.closest('.c-card').getBoundingClientRect(); return [Math.round(r.top),Math.round(r.bottom),innerHeight,h.textContent]}")
                    check(f'{tag}: its link brings the drill (marked optional) on screen', bool(dt) and 0 <= dt[0] < dt[2] and ('optional' in dt[3] or '可选' in dt[3]), dt)
                    pg.evaluate("()=>window.scrollTo(0,0)"); pg.wait_for_timeout(200)
                for hit in pg.evaluate("el=>(" + RULES + ")(el)", pg.evaluate_handle(CARD)):
                    if lang == 'zh' or hit.startswith('“·”'):
                        check(f'{tag} go further: writing rules', False, hit)
                # the answers survive a mode switch
                mode(pg, 'sandbox'); mode(pg, 'challenge')
                pg.evaluate(card_js("c.querySelector('[data-ex=\"tab\"][data-i=\"0\"]').click(); return 1")); pg.wait_for_timeout(150)
                dots = pg.evaluate(card_js("return [...c.querySelectorAll('.ex-qs .pb-dot')].map(d=>d.textContent)"))
                check(f'{tag}: after Scenarios and back, the explain answers are kept', bool(dots) and all(d == '✓' for d in dots), dots)
                # a “go” button opens the case on the stage
                pg.evaluate(card_js("c.querySelector('[data-ex=\"tab\"][data-i=\"1\"]').click(); return 1")); pg.wait_for_timeout(150)
                go = pg.evaluate(card_js("const g=c.querySelector('[data-ex=\"go\"]'); return g?g.textContent:null"))
                if go:
                    pg.evaluate(card_js("c.querySelector('[data-ex=\"go\"]').click(); return 1")); pg.wait_for_timeout(500); settle(pg)
                    m = pg.evaluate("()=>{const b=[...document.querySelectorAll('.modes button[aria-pressed=\"true\"]')].find(e=>e.offsetParent!==null); return b?b.dataset.mode:null}")
                    check(f'{tag}: “{go[:30]}” opens the case in Scenarios', m == 'sandbox', m)
                # a language switch re-mounts the lab: the card stays, in the other language
                mode(pg, 'challenge')
                other = 'zh' if lang == 'en' else 'en'
                pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(400)
                tb = pg.evaluate(card_js("return c.querySelector('[data-ex=\"tab\"]').textContent"))
                check(f'{tag}: after switching to {other.upper()} in the challenge, the card is still there, in {other.upper()}', tb == ('1. Explain it' if other == 'en' else '1. 讲清楚'), tb)
                check(f'{tag}: no page errors', not errs, errs[:2])
                pg.close()
    b.close()
check('over all the labs: the right explanation is not always shown first', len(set(right_pos)) > 1, right_pos)
print(f'{n} checks, {len(fails)} failed')
print('FAILED:', fails[:40] if fails else 'none')
sys.exit(1 if fails else 0)
