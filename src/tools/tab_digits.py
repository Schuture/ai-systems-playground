"""Figtree's ten digits in their even-width (tabular) form, as a font of their own: static/fonts/figtree-tab-digits.woff2, family
"Figtree Tab". --font-mono puts it before Figtree, so labels and numbers that used IBM Plex Mono keep figures of one width (a number that
counts up does not shift the words after it) while running text keeps Figtree's own figures. Figtree's licence (OFL-Figtree.txt, no
reserved name) allows the modified copy.
Run after replacing static/fonts/figtree-latin.woff2:   python3 tools/tab_digits.py      (needs fonttools and brotli)"""
import pathlib
from fontTools.ttLib import TTFont
from fontTools import subset

F = pathlib.Path(__file__).resolve().parent.parent / 'static' / 'fonts'
font = TTFont(F / 'figtree-latin.woff2')
gsub = font['GSUB'].table
tnum = {}
for rec in gsub.FeatureList.FeatureRecord:
    if rec.FeatureTag != 'tnum': continue
    for i in rec.Feature.LookupListIndex:
        lookup = gsub.LookupList.Lookup[i]
        for st in lookup.SubTable:
            st = st.ExtSubTable if lookup.LookupType == 7 else st
            tnum.update(getattr(st, 'mapping', {}))
DIGITS = range(0x30, 0x3A)
for table in font['cmap'].tables:
    for cp in DIGITS:
        if table.cmap.get(cp) in tnum: table.cmap[cp] = tnum[table.cmap[cp]]
opts = subset.Options(); opts.flavor = 'woff2'; opts.layout_features = []; opts.name_IDs = ['*']
sub = subset.Subsetter(opts); sub.populate(unicodes=DIGITS); sub.subset(font)
for rec in font['name'].names:
    if rec.nameID in (1, 3, 4, 6, 16):
        rec.string = rec.toUnicode().replace('Figtree', 'Figtree Tab' if rec.nameID != 6 else 'FigtreeTab')
font.flavor = 'woff2'
font.save(F / 'figtree-tab-digits.woff2')
print('wrote', F / 'figtree-tab-digits.woff2', (F / 'figtree-tab-digits.woff2').stat().st_size, 'bytes')
