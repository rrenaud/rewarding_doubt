#!/usr/bin/env bash
# Build index.md, then render it, the README, docs and run reports to standalone HTML
# with KaTeX math, next to each .md.
set -euo pipefail
cd "$(dirname "$0")/.."
command -v node >/dev/null || source "$HOME/.nvm/nvm.sh"
[ -d node_modules/katex ] || npm install --no-fund --no-audit
htmls=()
python3 scripts/make_index.py
for md in index.md README.md docs/objectives.md runs/*/report.md; do
  html="${md%.md}.html"
  title=$(grep -m1 '^# ' "$md" | sed 's/^# //')
  pandoc "$md" --from gfm+tex_math_dollars --to html5 --standalone --katex --wrap=none \
    --lua-filter scripts/md_links_to_html.lua --metadata pagetitle="$title" \
    --variable maxwidth=56em --output "$html"
  htmls+=("$html")
done
# Typeset math to static HTML (no client-side JavaScript or CDN needed).
node scripts/katex_prerender.mjs "${htmls[@]}"
