import assert from "node:assert/strict";
import fs from "node:fs";

const apiSource = fs.readFileSync(new URL("./api.js", import.meta.url), "utf8");
const overlaySource = fs.readFileSync(
  new URL("./components/ApiProcessingOverlay.jsx", import.meta.url),
  "utf8",
);
const mainSource = fs.readFileSync(new URL("./main.jsx", import.meta.url), "utf8");
const cssSource = fs.readFileSync(new URL("./processing-progress.css", import.meta.url), "utf8");

assert.match(apiSource, /malchan:api-progress/);
assert.match(apiSource, /phase: "start"/);
assert.match(apiSource, /phase: "complete"/);
assert.match(apiSource, /durationMs/);
assert.match(apiSource, /モデル学習・パラメータ探索/);
assert.match(apiSource, /候補モデルを交差検証・比較/);
assert.match(apiSource, /学習済みモデルを精度検証/);
assert.match(apiSource, /逆解析・候補探索/);
assert.match(apiSource, /foreground: true/);
assert.match(apiSource, /\[malchan api timing\]/);

assert.match(mainSource, /ApiProcessingOverlay/);
assert.match(mainSource, /<ApiProcessingOverlay \/>/);

assert.match(overlaySource, /CLOSE_DELAY_MS = 700/);
assert.match(overlaySource, /formatProcessingDuration/);
assert.match(overlaySource, /operation\.completed/);
assert.match(overlaySource, /currentStage\.startedAt/);
assert.match(overlaySource, /完了率を推測せず/);
assert.match(overlaySource, /最長:/);
assert.doesNotMatch(overlaySource, /progressPercent|percentComplete|estimatedPercent/);

assert.match(cssSource, /font-size: 12px/);
assert.match(cssSource, /text-overflow: ellipsis/);
assert.match(cssSource, /white-space: nowrap/);
assert.match(cssSource, /@media \(max-width: 620px\)/);

console.log("processing progress and timing checks passed");
