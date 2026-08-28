import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { getWorkflowCompletion } from "./workflowCompletion.js";

const modelInfo = { model_id: "model-1", xai_status: "not_requested" };
const base = {
  rows: [{ x: 1, y: 2 }, { x: 2, y: 3 }],
  columns: ["x", "y"],
  ready: true,
  modelInfo,
};

const customPrediction = getWorkflowCompletion({
  ...base,
  predictionWorkspace: { customPrediction: { y: 2.5 }, filePredictions: [] },
});
assert.equal(customPrediction.predict.complete, true);

const filePrediction = getWorkflowCompletion({
  ...base,
  predictionWorkspace: { customPrediction: null, filePredictions: [{ y: 2.5 }] },
});
assert.equal(filePrediction.predict.complete, true);

const staleInverse = getWorkflowCompletion({
  ...base,
  inverseResult: { model_id: "model-1", candidates: [{}] },
  inverseCurrent: false,
});
assert.equal(staleInverse.optimize.complete, false, "same model_id can still represent a stale tuned predictor");

const staleReport = getWorkflowCompletion({
  ...base,
  report: "old report",
  reportCurrent: false,
});
assert.equal(staleReport.report.complete, false);

const here = path.dirname(fileURLToPath(import.meta.url));
const app = fs.readFileSync(path.join(here, "App.jsx"), "utf8");
const workspaceState = fs.readFileSync(path.join(here, "context", "useWorkspacePageState.js"), "utf8");
const reportPage = fs.readFileSync(path.join(here, "pages", "ReportPage.jsx"), "utf8");

assert.match(workspaceState, /useWorkspacePageStateSnapshot/);
assert.match(workspaceState, /useSyncExternalStore/);
assert.match(workspaceState, /emitWorkspacePageState\(scope\)/);
assert.match(app, /useWorkspacePageStateSnapshot\("prediction", modelInfo\)/);
assert.match(app, /useArtifactFreshness\(inverseResult, modelInfo\)/);
assert.match(app, /useArtifactFreshness\(report, modelInfo\)/);
assert.match(app, /modelInfo \|\| "unregistered"/);
assert.doesNotMatch(app, /modelInfo\?\.model_id \|\| "unregistered"/);
assert.match(app, /if \(report\) setReport\(""\)/);
assert.match(reportPage, /useAnalysisColumnSelection\(rawColumns, rows\)/);
assert.match(reportPage, /rawStats\.filter\(\(item\) => enabledSet\.has\(item\.column\)\)/);
assert.match(reportPage, /stats\.reduce\(\(sum, item\) => sum \+ item\.missing, 0\)/);

console.log("final workflow regression checks passed");
