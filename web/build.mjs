// Bundles the lessons into one self-contained HTML file (all JS and CSS inline).
//
//   node build.mjs           → ../docs/index.html (open it directly, or serve it with GitHub Pages)
//   node build.mjs --serve   → rebuild on change and serve on http://localhost:8000
import * as esbuild from "esbuild";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const outDir = join(here, "..", "docs");
const serve = process.argv.includes("--serve");

const FONTS =
  '<link rel="preconnect" href="https://fonts.googleapis.com">\n' +
  '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n' +
  '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Sans+Math&family=Recursive:wght,CASL,MONO@300..800,0..1,0..1&display=swap">';

// Body of the page: also usable as a fragment by hosts that add their own <html> skeleton.
const fragment = (js, css) => `<title>Visual RL</title>
${FONTS}
<style>
${css}</style>
<div id="app"></div>
<script type="module">
${js}</script>
`;

const page = (js, css) => `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="description" content="Learn reinforcement learning by seeing every decision, target, and update.">
${fragment(js, css).replace('<div id="app"></div>', "</head>\n<body>\n<div id=\"app\"></div>")}</body>
</html>
`;

const writeHtml = {
  name: "write-html",
  setup(build) {
    build.onEnd((result) => {
      if (result.errors.length) return;
      const js = result.outputFiles.find((f) => f.path.endsWith(".js")).text;
      const css = result.outputFiles.find((f) => f.path.endsWith(".css")).text;
      if (js.includes("</script")) throw new Error("bundle contains </script and cannot be inlined");
      mkdirSync(outDir, { recursive: true });
      writeFileSync(join(outDir, "index.html"), page(js, css));
      mkdirSync(join(here, "dist"), { recursive: true });
      writeFileSync(join(here, "dist", "fragment.html"), fragment(js, css));
      console.log(`wrote docs/index.html (${Math.round((js.length + css.length) / 1024)} KB)`);
    });
  },
};

const options = {
  absWorkingDir: here,
  entryPoints: ["src/main.js"],
  bundle: true,
  format: "esm",
  target: "es2022",
  minify: !serve,
  legalComments: "none",
  loader: { ".py": "text" },
  outdir: "dist/assets",
  write: false,
  plugins: [writeHtml],
};

if (serve) {
  const ctx = await esbuild.context(options);
  await ctx.watch();
  const { port } = await ctx.serve({ servedir: outDir, port: 8000 });
  console.log(`serving http://localhost:${port}`);
} else {
  await esbuild.build(options);
}
