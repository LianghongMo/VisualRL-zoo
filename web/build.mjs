// Build one portable HTML lesson book and the identical static Site output.
import * as esbuild from "esbuild";
import { mkdirSync, writeFileSync, readFileSync, existsSync, copyFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, "..");
const staticDir = join(root, "dist");
const serve = process.argv.includes("--serve");
// A development rebuild must never overwrite a validated production artifact.
const outDir = serve ? join(here, "dist", "preview") : join(root, "docs");
const metaPath = join(here, "site.json");
const metadata = existsSync(metaPath) ? JSON.parse(readFileSync(metaPath, "utf8")) : {};
const title = metadata.title || "Visual RL · 看见强化学习";
const description = metadata.description || "从一张地图开始，用物理图像、公式和可操作的例子理解强化学习。";
const escapeHtml = (text) => String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
// This URL is copied from the Sites connector, never from browser or forwarded headers.
const siteUrl = metadata.url && new URL(metadata.url).protocol === "https:" ? metadata.url : "";
const imagePath = join(here, "public", "og.png");
const imageUrl = siteUrl && existsSync(imagePath) ? new URL("og.png", `${siteUrl.replace(/\/$/, "")}/`).href : "";
const head = `<title>${escapeHtml(title)}</title>
<meta name="description" content="${escapeHtml(description)}">
<meta property="og:type" content="website">
<meta property="og:title" content="${escapeHtml(title)}">
<meta property="og:description" content="${escapeHtml(description)}">
<meta name="twitter:card" content="${imageUrl ? "summary_large_image" : "summary"}">
<meta name="twitter:title" content="${escapeHtml(title)}">
<meta name="twitter:description" content="${escapeHtml(description)}">
${siteUrl ? `<meta property="og:url" content="${escapeHtml(siteUrl)}">` : ""}
${imageUrl ? `<meta property="og:image" content="${escapeHtml(imageUrl)}">\n<meta name="twitter:image" content="${escapeHtml(imageUrl)}">\n<meta property="og:image:alt" content="Visual RL — See how a robot learns">` : ""}`;
const body = (js) => `<div id="app"></div><script type="module">\n${js}</script>`;
const fragment = (js, css) => `${head}<style>${css}</style>${body(js)}`;
const page = (js, css) => `<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
${head}<style>${css}</style></head><body>${body(js)}</body></html>`;
const writeHtml = {
  name: "write-html",
  setup(build) {
    build.onEnd((result) => {
      if (result.errors.length) return;
      const js = result.outputFiles.find((f) => f.path.endsWith(".js")).text;
      const css = result.outputFiles.find((f) => f.path.endsWith(".css")).text;
      if (/<\/script/i.test(js)) throw new Error("bundle contains a script closing tag and cannot be inlined");
      for (const dir of serve ? [outDir] : [outDir, staticDir]) {
        mkdirSync(dir, { recursive: true });
        writeFileSync(join(dir, "index.html"), page(js, css));
        if (existsSync(imagePath)) copyFileSync(imagePath, join(dir, "og.png"));
      }
      mkdirSync(join(here, "dist"), { recursive: true });
      writeFileSync(join(here, "dist", "fragment.html"), fragment(js, css));
      console.log(`wrote ${serve ? "web/dist/preview/index.html" : "docs/index.html and dist/index.html"} (${Math.round((js.length + css.length) / 1024)} KB)`);
    });
  },
};
const options = { absWorkingDir: here, entryPoints: ["src/main.js"], bundle: true, format: "esm", target: "es2022", minify: !serve, legalComments: "none", loader: { ".py": "text" }, outdir: join(outDir, "assets"), write: false, plugins: [writeHtml] };
if (serve) {
  const ctx = await esbuild.context(options);
  await ctx.watch();
  const { port } = await ctx.serve({ servedir: outDir, port: 8000 });
  console.log(`serving http://localhost:${port}`);
} else { await esbuild.build(options); }
