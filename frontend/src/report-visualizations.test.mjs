import assert from "node:assert/strict";
import fs from "node:fs";

const source = fs.readFileSync(new URL("./report-visualizations.js", import.meta.url), "utf8");
const interactiveExport = fs.readFileSync(new URL("./report-interactive-export.js", import.meta.url), "utf8");
const interactiveFigures = fs.readFileSync(new URL("./report-interactive-figures.js", import.meta.url), "utf8");
const interactiveRuntime = fs.readFileSync(new URL("./report-interactive-runtime.js", import.meta.url), "utf8");
const reportPage = fs.readFileSync(new URL("./pages/ReportPage.jsx", import.meta.url), "utf8");

assert.match(source, /visualizationYy\(modelId, target, \{ cv: true, residual: false \}\)/);
assert.match(source, /visualizationYy\(modelId, target, \{ cv, residual: true \}\)/);
assert.match(source, /IMPORTANCE_METHODS/);
assert.match(source, /Baseline \+ SHAP/);
assert.match(source, /visualizationBeeswarm/);
assert.match(source, /visualizationPdp2d/);
assert.match(source, /figureToPng/);
assert.match(source, /plotly\.toImage/);

assert.match(interactiveFigures, /collectInteractiveFigures/);
assert.match(interactiveFigures, /combinedImportanceFigure/);
assert.match(interactiveFigures, /legacyPdFigure/);
assert.match(interactiveFigures, /visualizationPdp2d/);
assert.match(interactiveExport, /downloadInteractiveHtmlReport/);
assert.match(interactiveExport, /attachInteractiveRegistry/);
assert.match(interactiveExport, /removeReportSection\(reportHtml, "diagnostics"\)/);
assert.match(interactiveExport, /removeReportNavItem\(reportHtml, "diagnostics"\)/);
assert.match(interactiveExport, /<span>05<\/span>/);
assert.match(interactiveExport, /data-open-report-figure/);
assert.match(interactiveRuntime, /plotly\.min\.js\?raw/);
assert.match(interactiveRuntime, /malchan-figure-modal/);
assert.match(interactiveRuntime, /malchan-x-min/);
assert.match(interactiveRuntime, /malchan-font-size/);
assert.match(interactiveRuntime, /Plotly\.relayout/);
assert.match(interactiveRuntime, /Plotly\.downloadImage/);
assert.match(reportPage, /downloadInteractiveHtmlReport/);
assert.match(reportPage, /軸レンジ、文字サイズ、表示高さ/);

console.log("report visualization source checks passed");
