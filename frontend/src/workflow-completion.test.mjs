import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { getWorkflowCompletion, workflowStatusText } from "./workflowCompletion.js";

const empty = getWorkflowCompletion();
assert.equal(empty.data.complete, false);
assert.equal(empty.prepare.complete, false);
assert.equal(empty.model.complete, false);
assert.equal(empty.explore.optional, true);
assert.equal(empty.explore.complete, false);
assert.equal(workflowStatusText(empty.model), "未完了");

const prepared = getWorkflowCompletion({
  rows: [{ x: 1, y: 2 }, { x: 2, y: 3 }],
  columns: ["x", "y"],
  ready: true,
});
assert.equal(prepared.data.complete, true);
assert.equal(prepared.prepare.complete, true);
assert.equal(prepared.model.complete, false);
assert.equal(prepared.model.available, true);
assert.equal(prepared.explore.complete, false, "visitable Explore must not look completed");

const trained = getWorkflowCompletion({
  rows: [{ x: 1, y: 2 }, { x: 2, y: 3 }],
  columns: ["x", "y"],
  ready: true,
  modelInfo: { model_id: "model-1", xai_status: "ready" },
});
assert.equal(trained.model.complete, true);
assert.equal(trained.explain.complete, true);
assert.equal(trained.predict.complete, false);
assert.equal(trained.optimize.complete, false);

const analyzed = getWorkflowCompletion({
  rows: [{ x: 1, y: 2 }],
  columns: ["x", "y"],
  modelInfo: { model_id: "model-1", xai_status: "not_requested" },
  comparison: { targets: {} },
  prediction: { y: 3.2 },
  inverseResult: { model_id: "model-1", candidates: [] },
  report: "report body",
});
assert.equal(analyzed.explain.complete, true);
assert.equal(analyzed.predict.complete, true);
assert.equal(analyzed.optimize.complete, true);
assert.equal(analyzed.report.complete, true);
assert.equal(workflowStatusText(analyzed.optimize), "探索済み");

const staleInverse = getWorkflowCompletion({
  modelInfo: { model_id: "model-2", xai_status: "not_requested" },
  columns: ["x", "y"],
  inverseResult: { model_id: "model-1", candidates: [] },
});
assert.equal(staleInverse.optimize.complete, false, "inverse results from another model must not look complete");

const imported = getWorkflowCompletion({
  rows: [],
  columns: ["x", "y"],
  ready: false,
  modelInfo: { model_id: "imported-model", xai_status: "not_requested" },
});
assert.equal(imported.data.complete, true, "imported model should establish a data/schema context");
assert.equal(imported.prepare.complete, false, "importing a model must not pretend Prepare was performed in this workspace");
assert.equal(imported.model.complete, true);

const here = path.dirname(fileURLToPath(import.meta.url));
const appSource = fs.readFileSync(path.join(here, "App.jsx"), "utf8");
const cssSource = fs.readFileSync(path.join(here, "workflow-completion.css"), "utf8");

assert.match(appSource, /getWorkflowCompletion\(\{/);
assert.doesNotMatch(appSource, /stepIndex\s*<\s*index/, "navigation order must not imply completion");
assert.match(appSource, /stepStatus\.complete \? "✓"/);
assert.match(appSource, /data-workflow-status=/);
assert.match(appSource, /tab-status/);
assert.match(cssSource, /\.tab\.complete::before/);
assert.match(cssSource, /\.workflow-strip > i\.complete/);
assert.match(cssSource, /\.workflow-step\.complete span/);

console.log("workflow completion regression checks passed");
