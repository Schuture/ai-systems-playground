"""Links a reader can click open formatted pages (the owner's review of batch 8, 2026-10-07: a Hugging Face link “显示的是人类无法
直接阅读的纯文本”): no raw.githubusercontent.com, gist raw, github.com/<owner>/<repo>/raw/, huggingface.co/<repo>/raw/ or /resolve/ URL in the
page's code. Use
github.com/<owner>/<repo>/blob/<ref>/<path> and huggingface.co/<repo>/blob/<ref>/<path>. Sources kept for the team (sources/) may
still record raw URLs; they are not on the page.
Run: python3 tests/link_format_check.py"""
import pathlib, re, sys

P = pathlib.Path(__file__).resolve().parent.parent
FILES = [P / 'app.js', P / 'main.js', P / 'body.html', P / 'head.html'] + sorted((P / 'labs').glob('*.js')) + sorted((P / 'teasers').glob('*.js'))
BAD = re.compile(r'raw\.githubusercontent\.com|gist\.githubusercontent\.com|github\.com/[^/\s\'"`<>]+/[^/\s\'"`<>]+/raw/|huggingface\.co/[^\s\'"`<>]*?/(raw|resolve)/')
hits = []
for f in FILES:
    if not f.exists():
        continue
    for i, line in enumerate(f.read_text().splitlines(), 1):
        for m in BAD.finditer(line):
            hits.append(f'{f.relative_to(P)}:{i}: {line[max(0, m.start() - 40):m.end() + 60].strip()}')
print('\n'.join(hits) if hits else 'no raw links')
print(f'{len(FILES)} files checked, {len(hits)} raw links')
sys.exit(1 if hits else 0)
