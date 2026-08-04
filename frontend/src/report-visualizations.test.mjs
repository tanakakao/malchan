import assert from "node:assert/strict";
import fs from "node:fs";

const source = fs.readFileSync(new URL("./report-visualizations.js", import.meta.url), "utf8");
const reportPage = fs.readFileSync(new URL("./pages/ReportPage.jsx", import.meta.url), "utf8");

assert.match(source, /visualizationYy\(modelId, target, \{ cv: true, residual: false \}\)/);
assert.match(source, /visualizationYy\(modelId, target, \{ cv, residual: true \}\)/);
assert.match(source, /IMPORTANCE_METHODS/);
assert.match(source, /method: "model"/);
assert.match(source, /method: "shap"/);
assert.match(source, /method: "pfi"/);
assert.match(source, /Baseline \+ SHAP/);
assert.match(source, /Observed target/);
assert.match(source, /Partial dependence/);
assert.match(source, /visualizationBeeswarm/);
assert.match(source, /visualizationPdp2d/);
assert.match(source, /figureToPng/);
assert.match(source, /data:image/);
assert.match(reportPage, /downloadDetailedHtmlReport/);
assert.match(reportPage, /Y–Y、重要度、PD、SHAPを収集しています/);

console.log("report visualization source checks passed");
