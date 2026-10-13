"""Shared by the lab tests since Batch 1 (Codex review 1): on phones the labs use the m2 template, where Settings is a drawer.
Tests that drive controls in Settings open every drawer as soon as it appears; tests/site_nav.py checks the drawer itself."""
OPEN_DRAWERS = ("new MutationObserver(()=>document.querySelectorAll('.m2 .controls:not(.open)').forEach(c=>c.classList.add('open')))"
                ".observe(document,{subtree:true,childList:true})")


def open_drawers(pg):
    pg.add_init_script(OPEN_DRAWERS)
    return pg
