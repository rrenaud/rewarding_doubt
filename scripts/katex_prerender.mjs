// Replace pandoc's raw-TeX math spans with KaTeX HTML at build time, so pages need no
// JavaScript. Usage: node scripts/katex_prerender.mjs page.html [...]
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import katex from "katex";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const css = path.join(root, "node_modules/katex/dist/katex.min.css");
const unescape = (s) => s.replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&quot;/g, '"')
  .replace(/&#39;/g, "'").replace(/&amp;/g, "&");

let failed = false;
for (const file of process.argv.slice(2)) {
  let html = fs.readFileSync(file, "utf8");
  // Drop pandoc's client-side KaTeX loader; it points at a path that may not exist.
  html = html.replace(/<script[^>]*katex[\s\S]*?<\/script>\s*/g, "")
    .replace(/<script>document\.addEventListener\("DOMContentLoaded"[\s\S]*?<\/script>\s*/g, "")
    .replace(/<link[^>]*katex[^>]*>\s*/g, "");
  let count = 0;
  html = html.replace(/<span\s+class="math (inline|display)">([\s\S]*?)<\/span>/g, (_, kind, tex) => {
    count++;
    try {
      return katex.renderToString(unescape(tex), { displayMode: kind === "display", throwOnError: true });
    } catch (err) {
      failed = true;
      console.error(`${file}: ${err.message}`);
      return katex.renderToString(unescape(tex), { displayMode: kind === "display", throwOnError: false });
    }
  });
  const href = path.relative(path.dirname(path.resolve(file)), css).split(path.sep).join("/");
  html = html.replace("</head>", `  <link rel="stylesheet" href="${href}" />\n</head>`);
  fs.writeFileSync(file, html);
  console.log(`${file}: ${count} formulas`);
}
process.exit(failed ? 1 : 0);
