import assert from "node:assert/strict";
import fs from "node:fs";
import { evaluationMetricNames, metricValue } from "./evaluationMetrics.js";

const result = {
  train: [{ RMSE: 0.0471, MAE: 0.0336, MAPE: 0.2177, R2: 0.9686 }],
  test: [{ RMSE: 0.0704, MAE: 0.06, MAPE: 0.4174, R2: 0.5767 }],
  oof: {
    mae: 0.06,
    mse: 0.0054,
    rmse: 0.0733,
    r2: 0.929,
    mape: 0.4174,
  },
};

assert.deepEqual(
  evaluationMetricNames(result).map((metric) => metric.toUpperCase()),
  ["RMSE", "MAE", "MAPE", "R2", "MSE"],
);
assert.equal(metricValue(result.oof, "RMSE"), 0.0733);
assert.equal(metricValue(result.train[0], "rmse"), 0.0471);

const css = fs.readFileSync(new URL("./explain-evaluation.css", import.meta.url), "utf8");
assert.match(css, /\.bochan-evaluation-table\s*\{[^}]*width:\s*100%/s);
assert.match(css, /\.bochan-evaluation-table table\s*\{[^}]*width:\s*100%/s);
assert.match(css, /table-layout:\s*fixed/);
assert.match(css, /max-height:\s*none/);

console.log("evaluation metric table checks passed");
