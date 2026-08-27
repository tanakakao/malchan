import assert from "node:assert/strict";
import fs from "node:fs";

const hook = fs.readFileSync(
  new URL("./context/useWorkspacePageState.js", import.meta.url),
  "utf8",
);
const predictionPage = fs.readFileSync(
  new URL("./pages/PredictionPage.jsx", import.meta.url),
  "utf8",
);

assert.match(hook, /const workspacePageStateStore = new Map\(\)/);
assert.match(hook, /Object\.is\(cached\.resetKey, resetKey\)/);
assert.match(hook, /Object\.is\(resetKeyRef\.current, resetKey\)/);
assert.match(hook, /export function clearWorkspacePageState/);
assert.doesNotMatch(hook, /localStorage|sessionStorage/);

assert.match(predictionPage, /useWorkspacePageState/);
assert.match(predictionPage, /"prediction",\s*createPredictionWorkspace,\s*modelInfo,/s);
assert.match(predictionPage, /mode: "custom"/);
assert.match(predictionPage, /customPrediction: null/);
assert.match(predictionPage, /customShap: null/);
assert.match(predictionPage, /filePredictions: \[\]/);
assert.match(predictionPage, /selectedRows: new Set\(\)/);
assert.match(predictionPage, /fileShapLabels: \[\]/);
assert.doesNotMatch(predictionPage, /const \[mode, setMode\] = useState/);
assert.doesNotMatch(predictionPage, /const \[fileRows, setFileRows\] = useState/);
assert.match(predictionPage, /const \[running, setRunning\] = useState/);
assert.match(predictionPage, /const \[error, setError\] = useState/);

console.log("workspace page state persistence checks passed");
