import assert from "node:assert/strict";
import fs from "node:fs";

const runtime = fs.readFileSync(new URL("./report-interactive-runtime.js", import.meta.url), "utf8");
const figures = fs.readFileSync(new URL("./report-interactive-figures.js", import.meta.url), "utf8");
const exporter = fs.readFileSync(new URL("./report-interactive-export.js", import.meta.url), "utf8");

assert.match(runtime, /plotly\.min\.js\?raw/);
assert.match(runtime, /malchan-figure-modal/);
assert.match(runtime, /malchan-x-min/);
assert.match(runtime, /malchan-y-max/);
assert.match(runtime, /malchan-font-size/);
assert.match(runtime, /malchan-plot-height/);
assert.match(runtime, /Plotly\.relayout/);
assert.match(runtime, /Plotly\.downloadImage/);
assert.match(figures, /collectInteractiveFigures/);
assert.match(figures, /legacyPdFigure/);
assert.match(exporter, /downloadInteractiveHtmlReport/);
assert.match(exporter, /data-open-report-figure/);

console.log("interactive report source checks passed");
