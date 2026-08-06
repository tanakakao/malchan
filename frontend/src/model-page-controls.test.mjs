import assert from "node:assert/strict";
import fs from "node:fs";

const control = fs.readFileSync(
  new URL("./components/ModelBundleControl.jsx", import.meta.url),
  "utf8",
);
const bundles = fs.readFileSync(new URL("./modelBundles.js", import.meta.url), "utf8");

assert.match(control, /settings\.insertAdjacentElement\("beforebegin", nextHost\)/);
assert.match(control, /model-run-action-host/);
assert.match(control, /sourceRunButtonRef\.current\?\.click\(\)/);
assert.match(control, /evaluationPanel\.hidden = true/);
assert.match(control, /<h3>モデルの保存・読み込み<\/h3>/);
assert.match(control, /保存名/);
assert.match(control, /bestModelNames\(comparison\)/);
assert.match(control, /model_names_by_target: targetNames/);
assert.doesNotMatch(control, /サーバーへ永続保存せず/);
assert.doesNotMatch(control, /bochanと同様の信頼済みファイル方式/);
assert.doesNotMatch(control, /署名用の秘密値は不要/);

assert.match(bundles, /export function modelBundleFilename/);
assert.match(bundles, /replace\(\/\[\\\\\/:\*\?"<>\|\\u0000-\\u001f\]\+\/g, "_"\)/);
assert.match(bundles, /saveName\.trim\(\)/);
assert.match(bundles, /endsWith\("\.malchan"\)/);

console.log("model page control checks passed");
