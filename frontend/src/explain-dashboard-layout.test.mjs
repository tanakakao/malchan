import assert from "node:assert/strict";
import fs from "node:fs";

const control = fs.readFileSync(
  new URL("./components/ModelResultVisualizationControl.jsx", import.meta.url),
  "utf8",
);
const css = fs.readFileSync(new URL("./xai.css", import.meta.url), "utf8");

assert.match(control, /step === "explain"/);
assert.match(control, /xai-evaluation-host/);
assert.match(control, /xai-yy-panel/);
assert.match(control, /xai-importance-panel/);
assert.match(control, /xai-relationship-panel/);
assert.match(control, /精度評価の結果/);
assert.match(control, /modelVisualization\(modelInfo\.model_id\)/);
assert.match(control, /event\.target\.closest\("\.xai-overview"\)/);

assert.match(css, /grid-template-areas:\s*\n\s*"yy evaluation"\s*\n\s*"importance relationship"/);
assert.match(css, /\.xai-evaluation-host/);
assert.match(css, /\.xai-importance-panel/);
assert.match(css, /\.xai-relationship-panel/);
assert.match(css, /@media \(max-width: 1120px\)/);

console.log("Explain dashboard layout source checks passed");
