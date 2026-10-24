"""Assemble the single-file page from sources.

  python3 build.py                                  -> ai-systems-playground.html with every lab and teaser
  python3 build.py out/site --split                 -> out/site/index.html + app.css + app.js (for the Artifact)
  python3 build.py out/pa.html --only paged-attention,pa-teaser
                                                    -> a private build with only the named labs/teasers
                                                       (names are file stems in labs/ and teasers/)
  --standalone (with either form)                   -> a complete HTML document for any web server or a double click: doctype,
                                                       <meta charset="utf-8"> and the phone viewport in <head> (Codex D01, round 3).
                                                       Without it the page is a fragment for the Claude Artifact, which adds that shell
                                                       itself; served as-is by a plain server, the fragment loses its UTF-8 and the
                                                       script stops at its first Chinese regular expression.
  python3 build.py out/web --split --standalone     -> the website: out/web/index.html + app.css + app.js

Order: head.html, <style> styles.css + labs/*.css + teasers/*.css </style>, body.html,
<script> app.js (core, KV lab, home) + labs/*.js + teasers/*.js + main.js </script>.
Each lab or teaser file registers itself with window.ASP; main.js starts the app.

static/ (the fonts, the site icon, the share picture) is copied next to every page built outside src/, so the page finds
static/fonts/*.woff2 and the icons at the same relative paths in every build: the website, the Artifact, the test pages in out/ and
ai-systems-playground.html in src/ itself. The website (--standalone)
also preloads the two fonts the first screen needs and carries the share card (Open Graph) with the picture's address on the site."""
import os, pathlib, shutil, sys

P = pathlib.Path(__file__).parent
args = sys.argv[1:]
only = None
split = '--split' in args
if split: args.remove('--split')
standalone = '--standalone' in args
if standalone: args.remove('--standalone')
if '--only' in args:
    i = args.index('--only'); only = set(x.strip() for x in args[i + 1].split(',') if x.strip()); del args[i:i + 2]
dest = P / (args[0] if args else 'ai-systems-playground.html')
dest.parent.mkdir(parents=True, exist_ok=True)

def read(p): return (P / p).read_text().rstrip() + '\n'
def pick(folder, ext):
    return [f for f in sorted((P / folder).glob('*' + ext)) if not f.stem.startswith('_') and (only is None or f.stem in only)]

css = [read('styles.css')] + [read(f'labs/{f.name}') for f in pick('labs', '.css')] + [read(f'teasers/{f.name}') for f in pick('teasers', '.css')]
js = [read('app.js')] + [read(f'labs/{f.name}') for f in pick('labs', '.js')] + [read(f'teasers/{f.name}') for f in pick('teasers', '.js')] + [read('main.js')]
# the shell a browser needs when no Artifact wraps the page (the two rules are the ones the Artifact's own shell adds)
SHELL = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width,initial-scale=1">\n'
         '<style>body{margin:0}[hidden]{display:none!important}</style>\n')
# head.html starts with <meta charset="utf-8"> so that a fragment opened from disk is read as UTF-8 too; the shell has its own
headhtml = read('head.html').replace('<meta charset="utf-8">\n', '', 1) if standalone else read('head.html')
SITE = 'https://schuture.github.io/ai-systems-playground/'
if standalone:   # a fetch the browser starts before app.css arrives (over http only: a page opened from disk loads the fonts without it)
    headhtml += ''.join(f'<link rel="preload" href="static/fonts/{f}.woff2" as="font" type="font/woff2" crossorigin>\n'
                        for f in ('figtree-latin', 'bricolage-grotesque-latin'))
    headhtml += (f'<meta property="og:type" content="website">\n<meta property="og:url" content="{SITE}">\n'
                 f'<meta property="og:image" content="{SITE}static/og-image.png">\n<meta property="og:image:width" content="1200">\n'
                 '<meta property="og:image:height" content="630">\n<meta name="twitter:card" content="summary_large_image">\n')

def copy_static(to):
    """static/ next to the page; a file is replaced through a temporary copy, so a test reading it meanwhile never sees half a file"""
    for f in sorted((P / 'static').rglob('*')):
        if not f.is_file() or f.suffix not in ('.woff2', '.txt', '.svg', '.png'): continue
        d = to / 'static' / f.relative_to(P / 'static')
        if d.exists() and d.stat().st_size == f.stat().st_size and d.stat().st_mtime >= f.stat().st_mtime: continue
        d.parent.mkdir(parents=True, exist_ok=True); tmp = d.with_name(f'.{d.name}.{os.getpid()}')
        shutil.copy2(f, tmp); os.replace(tmp, d)

if split:   # a folder for the Artifact: a small page plus app.css and app.js, so the page itself stays small to read and update
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'app.css').write_text(('@charset "utf-8";\n' if standalone else '') + '\n'.join(css)); (dest / 'app.js').write_text('\n'.join(js))
    head = headhtml + '<link rel="stylesheet" href="app.css">\n'
    body = read('body.html') + '\n<script src="app.js" charset="utf-8"></script>\n'
    dest = dest / 'index.html'
else:
    head = headhtml + '<style>\n' + '\n'.join(css) + '</style>\n'
    body = read('body.html') + '\n<script>\n' + '\n'.join(js) + '</script>\n'
out = SHELL + head + '</head>\n<body>\n' + body + '</body>\n</html>\n' if standalone else head + '\n' + body
dest.write_text(out)
if dest.parent.resolve() != P.resolve(): copy_static(dest.parent)
print('wrote', dest.relative_to(P) if dest.is_relative_to(P) else dest, len(out), 'bytes; labs:', [f.stem for f in pick('labs', '.js')], 'teasers:', [f.stem for f in pick('teasers', '.js')])
