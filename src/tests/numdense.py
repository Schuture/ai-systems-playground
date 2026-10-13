"""Find explanation lines that pile up numbers (LAB_UX_GUIDE R6.20): reads the copy decks (out/<lab>-copy/<lang>.md, written by
tests/<lab>_copy_deck.py) and prints, per lab, the lines with too many numbers.
A number is a digit group (1,494 / 11.74 / 32K / 128 KiB / 50%); model, GPU and format names (Llama-3.1-8B, H100, BF16, FP8 …) do not
count. A line is flagged when it has 4 or more numbers, or 3 with 60 characters or more.
Tooltip lines ("- tip: …", or the list under a "_… tooltips:_" line) are counted apart: a tooltip that reads out the values of the hovered item (one pair, one bar) may carry
several numbers; a tooltip that explains may not. --no-tips hides the tooltip lines.
Run: python3 tests/numdense.py [zh|en] [--no-tips] [lab …]   (default zh, every deck found)"""
import pathlib, re, sys

P = pathlib.Path(__file__).resolve().parent.parent
DECKS = {'architecture': 'architecture-copy', 'transformer': 'transformer-copy', 'mha': 'mha-copy', 'kv-basics': 'kv-basics-copy',
         'gqa-mla': 'gqa-mla-copy', 'kv-cache': 'kv-copy', 'prefill-decode': 'prefill-decode-copy', 'paged-attention': 'pa-copy',
         'continuous-batching': 'cb-copy', 'dpo': 'dpo-copy', 'chunked-prefill': 'chunked-prefill-copy', 'prefix-caching': 'prefix-caching-copy',
         'serving-capstone': 'serving-capstone-copy',
         'sampling': 'sampling-copy', 'tokenization-cost': 'tokenization-cost-copy', 'rope': 'rope-copy', 'moe': 'moe-copy',
         # batch 6
         'tensors': 'tensors-copy', 'backprop': 'backprop-copy', 'softmax-numerics': 'softmax-numerics-copy',
         'number-formats': 'number-formats-copy', 'optimizers': 'optimizers-copy', 'train-debug': 'train-debug-copy',
         'roofline': 'roofline-copy', 'flash-attention': 'flash-attention-copy',
         # batch 10
         'pd-disagg': 'pd-disagg-copy',
         'multi-gpu-serving': 'multi-gpu-serving-copy',
         'quantization': 'quantization-copy',
         'embeddings': 'embeddings-copy',
         'tool-scheduling': 'tool-scheduling-copy',
         'rag-failure': 'rag-failure-copy',
         'vector-index': 'vector-index-copy',
         'admission': 'admission-copy',
         'contamination': 'contamination-copy',
         'product-design': 'product-design-copy',
         'chunk-rerank': 'chunk-rerank-copy',
         'eval-capstone': 'eval-capstone-copy'}
NAMES = re.compile(r'(?:Llama|Mixtral|Mistral|DeepSeek|Qwen|GPT|OPT|T5|Gemma|Phi|Falcon|Claude)[\w.\-‑‐ ]*?\d[\w.\-‑‐]*|'
                   r'\b(?:H100|H200|A100|B200|L40S|BF16|FP16|FP8|FP4|FP32|INT8|INT4|NVFP4|MXFP4|GB200|GQA-\d+|MQA|R\d+\.\d+|top-\d+|p\d\d)\b', re.I)
NUM = re.compile(r'(?<![\w.])[-−+]?\d[\d,]*(?:\.\d+)?(?:\s*(?:K|M|B|%|×|x|ms|s|GiB|GB|KiB|MiB|TiB|TB/s|TFLOP/s|GFLOP|TFLOP|FLOPs?|倍|个百分点|秒|毫秒))?(?![\w])')
EXCL = re.compile(r'tooltip|show the math|展开公式|popover|header|stage|settings|controls|connect|drill|interview|mini|legend|axis|chart|fidelity|assumptions|tips|modes|hero|metric|code', re.I)
SKIP = re.compile(r'^(#|\d+$|[-*] (guided|sandbox|challenge):)')


def count(line):
    s = NAMES.sub(' ', line)
    return len(NUM.findall(s))


def scan(lab, lang):
    f = P / 'out' / DECKS[lab] / f'{lang}.md'
    if not f.exists():
        return None
    seen, out, skip_sec, code, tipblk = set(), [], False, False, False
    for ln in f.read_text().splitlines():
        t = ln.strip()
        if re.match(r'^_.*tooltips?:_$', t):
            tipblk = True; continue
        if tipblk and t and not ln.startswith(' '):
            tipblk = False
        if t.startswith('```'):
            code = not code; continue
        if code:
            continue
        if t.startswith('#') and not t.startswith('#!') and (t.startswith('# ') or t.startswith('##')):
            skip_sec = bool(EXCL.search(t)); continue
        if skip_sec or not t or SKIP.match(t) or t in seen or len(t) < 40:
            continue
        if '⟶' in t or ' | ' in t or t.count('=') >= 2 or t.startswith('Mini result bar') or t.startswith('- ') and '：' not in t[:40] and ':' not in t[:40]:
            continue
        seen.add(t)
        n = count(t)
        if n >= 4 or (n >= 3 and len(t) >= 60):
            out.append((n, ('- tip: ' + t[2:]) if tipblk and t.startswith('- ') and not t.startswith('- tip:') else t))
    return out


if __name__ == '__main__':
    args = sys.argv[1:]
    notips = '--no-tips' in args
    if notips: args.remove('--no-tips')
    lang = args.pop(0) if args and args[0] in ('zh', 'en') else 'zh'
    labs = args or list(DECKS)
    total = 0
    for lab in labs:
        r = scan(lab, lang)
        if r is None:
            print(f'== {lab}: no deck'); continue
        tips = [x for x in r if x[1].startswith('- tip:')]
        if notips: r = [x for x in r if not x[1].startswith('- tip:')]
        total += len(r)
        print(f'== {lab}: {len(r)} lines' + ('' if notips else f' ({len(tips)} of them tooltips)'))
        for n, t in r:
            print(f'  [{n}] {t[:260]}')
    print('TOTAL', total)
