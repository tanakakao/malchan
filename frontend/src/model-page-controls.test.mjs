import assert from "node:assert/strict";
import fs from "node:fs";

const control = fs.readFileSync(
  new URL("./components/ModelBundleControl.jsx", import.meta.url),
  "utf8",
);
const dataPage = fs.readFileSync(new URL("./pages/DataPage.jsx", import.meta.url), "utf8");
const comparisonTable = fs.readFileSync(
  new URL("./components/ComparisonTable.jsx", import.meta.url),
  "utf8",
);
const modelVisualization = fs.readFileSync(
  new URL("./components/ModelResultVisualizationControl.jsx", import.meta.url),
  "utf8",
);
const bundles = fs.readFileSync(new URL("./modelBundles.js", import.meta.url), "utf8");

assert.match(control, /model-run-action-host/);
assert.match(control, /model-header-actions/);
assert.match(control, /model-save-control/);
assert.match(control, /aria-label="モデル保存名"/);
assert.match(control, /モデルを保存/);
assert.match(control, /sourceRunButtonRef\.current\?\.click\(\)/);
assert.match(control, /bestModelNames\(comparison\)/);
assert.doesNotMatch(control, /モデルファイルを読み込む/);
assert.doesNotMatch(control, /model-bundle-panel/);

assert.match(dataPage, /data-source-grid/);
assert.match(dataPage, /保存モデルを読み込む/);
assert.match(dataPage, /accept="\.malchan,application\/vnd\.malchan\.model"/);
assert.match(dataPage, /loadModelBundle/);

assert.doesNotMatch(comparisonTable, /BestModelEvaluation/);
assert.doesNotMatch(comparisonTable, /ベストモデル精度評価/);
assert.match(modelVisualization, /bochan-evaluation-table/);
assert.match(modelVisualization, /<th>OOF<\/th>/);
assert.match(modelVisualization, /step === "explain"/);
assert.doesNotMatch(
  modelVisualization,
  /<EvaluationSummary evaluation=\{evaluation\} target=\{targetDiagram\?\.target\} \/>/,
);

assert.match(bundles, /export function modelBundleFilename/);
assert.match(bundles, /replace\(\/\[\\\\\/:\*\?"<>\|\\u0000-\\u001f\]\+\/g, "_"\)/);
assert.match(bundles, /saveName\.trim\(\)/);
assert.match(bundles, /endsWith\("\.malchan"\)/);

console.log("model page control checks passed");
