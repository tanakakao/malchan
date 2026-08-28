import assert from "node:assert/strict";
import fs from "node:fs";

const store = fs.readFileSync(
  new URL("./context/useAnalysisColumnSelection.js", import.meta.url),
  "utf8",
);
const context = fs.readFileSync(
  new URL("./context/WorkbenchContext.jsx", import.meta.url),
  "utf8",
);
const dataPage = fs.readFileSync(new URL("./pages/DataPage.jsx", import.meta.url), "utf8");
const explorePage = fs.readFileSync(new URL("./pages/ExplorePage.jsx", import.meta.url), "utf8");
const preparePage = fs.readFileSync(new URL("./pages/PreparePage.jsx", import.meta.url), "utf8");

assert.match(store, /useSyncExternalStore/);
assert.match(store, /datasetKey/);
assert.match(store, /enabledColumns: normalizedColumns/);
assert.match(store, /loading a different dataset resets all columns to enabled/i);
assert.doesNotMatch(store, /localStorage|sessionStorage/);

assert.match(dataPage, /ANALYSIS COLUMNS/);
assert.match(dataPage, /setEnabledColumns\(nextEnabled\)/);
assert.match(dataPage, /changeTargets\(targets\.filter\(\(target\) => nextSet\.has\(target\)\)\)/);
assert.match(dataPage, /setNumFeatures\(numFeatures\.filter\(\(column\) => nextSet\.has\(column\)\)\)/);
assert.match(dataPage, /setCatFeatures\(catFeatures\.filter\(\(column\) => nextSet\.has\(column\)\)\)/);
assert.match(dataPage, /<DataTable rows=\{rows\} columns=\{enabledColumns\} \/>/);
assert.match(dataPage, /columnSelectionLocked/);

// Workbench training is explicitly driven by numFeatures / catFeatures and target(s).
// DataPage prunes those selections before Model is reached, so an OFF column is not
// part of the model input even though the untouched raw row object remains in memory.
assert.match(context, /num_cols: numFeatures\.filter\(\(column\) => features\.includes\(column\)\)/);
assert.match(context, /cat_cols: catFeatures\.filter\(\(column\) => features\.includes\(column\)\)/);
assert.match(context, /target_col: target/);
assert.match(context, /target_cols: targets/);

assert.match(explorePage, /useAnalysisColumnSelection/);
assert.match(explorePage, /rawNumeric\.filter\(\(column\) => enabledSet\.has\(column\)\)/);

assert.match(preparePage, /useAnalysisColumnSelection/);
assert.match(preparePage, /columns: rawColumns/);
assert.match(preparePage, /rawNumeric\.filter\(\(column\) => enabledSet\.has\(column\)\)/);
assert.match(preparePage, /rawCategorical\.filter\(\(column\) => enabledSet\.has\(column\)\)/);
assert.match(preparePage, /\(\) => columns\.filter\(\(column\) => !targetSet\.has\(column\)\)/);

console.log("analysis column selection checks passed");
