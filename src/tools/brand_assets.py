"""The site icon as PNG (32 px for browsers without SVG icons, 180 px for a phone's home screen) and the share picture (1200 x 630, the
Open Graph card a chat app shows for a link to the site), drawn with the site's own fonts and colours and written into static/.
Run after changing static/favicon.svg or the words below:   python3 tools/brand_assets.py
Needs Playwright's Chromium. The Chinese line uses Noto Sans SC from Google Fonts when the network allows it, else the system's font."""
import os, pathlib, tempfile
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
S = P / 'static'
FONTS = ''.join(f'@font-face{{font-family:"{f}";src:url({(S / "fonts" / n).as_uri()}) format("woff2");font-weight:{w}}}'
                for f, n, w in [('Figtree', 'figtree-latin.woff2', '300 900'), ('Bricolage Grotesque', 'bricolage-grotesque-latin.woff2', '400 800')])
# the logo of the top bar (body.html), in the light theme's colours
LOGO = ('<path stroke="#6A5341" stroke-width="{t}" d="M12 22V11"/><path stroke="#B03F74" stroke-width="{l}" d="M12 15.5C9 13.6 6.6 11.2 6.1 7.4"/>'
        '<path stroke="#0D7A86" stroke-width="{l}" d="M12 13.4c3-1.6 5.3-3.6 5.9-7.2"/><path stroke="#3D4A5C" stroke-width="{e}" d="M12 11V5"/>'
        '<circle fill="#B03F74" cx="6.1" cy="6.3" r="{r}"/><circle fill="#0D7A86" cx="17.9" cy="5.2" r="{r}"/><circle fill="#3D4A5C" cx="12" cy="3.6" r="{r2}"/>')


def icon(size, pad):
    """a square tile in the leaf grey with the tree centred; pad is the tree's margin as a share of the tile"""
    inner = size * (1 - 2 * pad)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}"><rect width="{size}" height="{size}" fill="#EEF3EC"/>'
            f'<g transform="translate({size * pad} {size * pad}) scale({inner / 24})" fill="none" stroke-linecap="round">'
            + LOGO.format(t=3.0, l=2.6, e=2.4, r=2.5, r2=2.2) + '</g></svg>')


# the tree's lab boxes along the limbs of the big logo: x, y (in the logo's 24-unit grid), colour
BOXES = [(3.2, 10.6, '#B03F74'), (2.4, 13.2, '#B03F74'), (18.6, 9.2, '#0D7A86'), (19.4, 11.8, '#0D7A86'), (20.2, 14.4, '#0D7A86'),
         (13.6, 6.8, '#3D4A5C'), (13.4, 18.6, '#6A5341')]
CARD = f"""<!doctype html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@500;700&display=swap">
<style>{FONTS}
html,body{{margin:0}} body{{width:1200px;height:630px;overflow:hidden;background:#FFFFFF;font-family:Figtree,"Noto Sans SC",sans-serif;color:#16201B}}
.l{{position:absolute;left:76px;top:70px;width:640px}}
.k{{display:flex;align-items:center;gap:14px;font:650 25px/1 Figtree,"Noto Sans SC",sans-serif;color:#47534C}}
.k i{{display:inline-flex;gap:6px}} .k b{{width:13px;height:13px;border-radius:50%}}
h1{{font:800 96px/0.98 "Bricolage Grotesque",sans-serif;letter-spacing:-.03em;margin:34px 0 30px}}
.en{{font:500 34px/1.3 Figtree,sans-serif;color:#47534C;margin:0}}
.zh{{font:700 32px/1.4 "Noto Sans SC",sans-serif;color:#47534C;margin:10px 0 0}}
.u{{position:absolute;left:76px;bottom:62px;font:600 24px/1 Figtree,sans-serif;color:#66726B}}
.r{{position:absolute;right:56px;top:56px;width:400px;height:518px;border-radius:40px;background:#F3F6F2}}
.r svg{{position:absolute;left:28px;top:40px}}
</style></head><body>
<div class="l"><div class="k"><i><b style="background:#6A5341"></b><b style="background:#B03F74"></b><b style="background:#0D7A86"></b><b style="background:#3D4A5C"></b></i>60 hands-on labs · English / 中文</div>
<h1>AI Systems<br>Playground</h1>
<p class="en">Learn AI systems by operating them.</p>
<p class="zh">亲手操作，看懂 AI 系统</p></div>
<div class="u">schuture.github.io/ai-systems-playground</div>
<div class="r"><svg width="344" height="440" viewBox="0 0 24 24" preserveAspectRatio="xMidYMid meet" fill="none" stroke-linecap="round">
{''.join(f'<rect x="{x}" y="{y}" width="3.4" height="1.5" rx=".5" fill="{c}" opacity=".9"/>' for x, y, c in BOXES)}
{LOGO.format(t=2.2, l=1.8, e=1.6, r=1.8, r2=1.6)}</svg></div>
</body></html>"""

with sync_playwright() as p, tempfile.TemporaryDirectory() as tmp:
    px = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')
    b = p.chromium.launch(proxy={'server': px} if px else None)
    for name, size, pad in [('favicon-32.png', 32, .1), ('apple-touch-icon.png', 180, .16)]:
        pg = b.new_page(viewport={'width': size, 'height': size})
        pg.set_content(f'<html><body style="margin:0">{icon(size, pad)}</body></html>')
        pg.screenshot(path=str(S / name), clip={'x': 0, 'y': 0, 'width': size, 'height': size}); pg.close()
    pg = b.new_page(viewport={'width': 1200, 'height': 630})
    card = pathlib.Path(tmp) / 'card.html'; card.write_text(CARD)   # a file, so the page may load the fonts from static/
    pg.goto(card.as_uri()); pg.wait_for_timeout(300)
    pg.evaluate('document.fonts.ready'); pg.wait_for_timeout(500)
    pg.screenshot(path=str(S / 'og-image.png')); pg.close()
    b.close()
print('wrote', ', '.join(n for n in ('favicon-32.png', 'apple-touch-icon.png', 'og-image.png')), 'in', S)
