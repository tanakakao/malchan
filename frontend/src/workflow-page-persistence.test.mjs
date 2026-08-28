import assert from "node:assert/strict";
import fs from "node:fs";

const app = fs.readFileSync(new URL("./App.jsx", import.meta.url), "utf8");
const host = fs.readFileSync(
  new URL("./components/PersistentWorkflowPage.jsx", import.meta.url),
  "utf8",
);
const preparePage = fs.readFileSync(new URL("./pages/PreparePage.jsx", import.meta.url), "utf8");
const modelPage = fs.readFileSync(new URL("./pages/ModelPage.jsx", import.meta.url), "utf8");
const optimizePage = fs.readFileSync(new URL("./pages/OptimizePage.jsx", import.meta.url), "utf8");

assert.match(app, /PERSISTENT_PAGE_IDS = new Set\(\["prepare", "model", "optimize"\]\)/);
assert.match(app, /PersistentWorkflowPage/);
assert.match(app, /const \[persistentPageIds, setPersistentPageIds\] = React\.useState\(\[\]\)/);
assert.match(app, /pageId === "optimize"[\s\S]*modelInfo\?\.model_id \|\| "unregistered"[\s\S]*:\s*rows;/);
assert.match(app, /mode === "simple" && step === "model"/);
assert.match(app, /!currentPageUsesCache && <Page \/>/);

assert.match(host, /const objectIdentityTokens = new WeakMap\(\)/);
assert.match(host, /const staleWhileHidden = !active && mountedToken !== token/);
assert.match(host, /if \(active && mountedToken !== token\)/);
assert.match(host, /hidden=\{!active\}/);
assert.match(host, /<Page key=\{token\} \/>/);

// These are the page-local settings that must survive navigation while the
// corresponding page remains cached.
assert.match(preparePage, /const \[featureTypeOverrides, setFeatureTypeOverrides\] = useState\(\{\}\)/);
assert.match(modelPage, /const \[preprocessing, setPreprocessing\] = useState\(DEFAULT_PREPROCESSING\)/);
assert.match(modelPage, /const \[parameterMode, setParameterMode\] = useState\("tuning"\)/);
assert.match(modelPage, /const \[accuracyValidation, setAccuracyValidation\] = useState\(true\)/);
assert.match(modelPage, /const \[cvMethod, setCvMethod\] = useState\("kfold"\)/);
assert.match(optimizePage, /const \[selectedTargets, setSelectedTargets\] = useState/);
assert.match(optimizePage, /const \[variableSettings, setVariableSettings\] = useState/);
assert.match(optimizePage, /const \[sumConstraint, setSumConstraint\] = useState/);

console.log("workflow page persistence checks passed");
