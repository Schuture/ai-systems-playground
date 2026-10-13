# AI Systems Playground

An interactive, in-browser course on how modern AI systems work: 60 hands-on labs covering tensors and training, Transformers and attention, KV cache and serving, post-training (SFT, LoRA, PPO, DPO, GRPO), evaluation, RAG and agents. Everything runs in the browser; there is no server. Available in Chinese and English.

**Open the site:** https://YOUR-GITHUB-NAME.github.io/ai-systems-playground/

## How the site is built

The site is three files (`index.html`, `app.js`, `app.css`) assembled from `src/`:

```
cd src
python3 build.py out/web --split --standalone
```

Open `src/out/web/index.html` in a browser to check it locally. Every push to `main` runs the same command in GitHub Actions (`.github/workflows/pages.yml`) and publishes the result to GitHub Pages.

## Layout

| Path | What it holds |
|---|---|
| `src/app.js` | the shared kit, the home page and the knowledge tree, the KV cache memory lab |
| `src/labs/` | one `.js` and one `.css` per lab |
| `src/teasers/` | the small demos on the home page tiles |
| `src/head.html`, `src/body.html`, `src/styles.css`, `src/main.js` | the page shell and shared styles |
| `src/build.py` | assembles the page |
| `src/tests/` | Playwright tests (numbers, flow, layout) |
| `src/capstone/` | Python reference engine for the serving capstone |
| `src/CONTRACT.md`, `src/LAB_UX_GUIDE.md` | the kit's API and the rules every lab follows |
