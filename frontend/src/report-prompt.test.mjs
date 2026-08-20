import assert from "node:assert/strict";

import {
  buildChatGptReportPrompt,
  collectReportAnalysisSummary,
  summarizeFeatureDistribution,
  summarizePdpResponse,
  summarizeShapResponse,
} from "./report-prompt.js";

const shap = summarizeShapResponse({
  feature: "temperature",
  value_columns: ["shap"],
  records: [
    { temperature: 600, shap: -1.0 },
    { temperature: 650, shap: -0.5 },
    { temperature: 700, shap: 0.2 },
    { temperature: 750, shap: 0.8 },
    { temperature: 800, shap: 1.2 },
  ],
});
assert.equal(shap.available, true);
assert.equal(shap.direction, "positive");
assert.match(shap.text, /押し上げ/);

const plateau = summarizePdpResponse({
  x_values: [0, 1, 2, 3, 4, 5],
  series: [{ name: "y", pd_values: [0, 2, 4, 5.8, 6.0, 6.02] }],
});
assert.equal(plateau.available, true);
assert.equal(plateau.series[0].kind, "increase_then_plateau");
assert.match(plateau.text, /飽和/);

const uShape = summarizePdpResponse({
  x_values: [0, 1, 2, 3, 4],
  series: [{ name: "y", pd_values: [4, 2, 1, 2.2, 4.5] }],
});
assert.equal(uShape.series[0].kind, "u_shape");

const distribution = summarizeFeatureDistribution([
  { temperature: 600 },
  { temperature: 650 },
  { temperature: 700 },
  { temperature: 750 },
  { temperature: 800 },
], "temperature");
assert.equal(distribution.type, "numeric");
assert.equal(distribution.median, 700);
assert.match(distribution.text, /中央50%/);

const fakeApi = {
  async xaiSummary() {
    return {
      status: "ready",
      targets: {
        strength: {
          status: "ready",
          importance_methods: ["shap", "pfi"],
          shap_features: ["temperature"],
          pdp_features: ["temperature"],
        },
      },
    };
  },
  async xaiImportance() {
    return { items: [{ feature: "temperature", value: 0.72 }] };
  },
  async xaiShap() {
    return {
      feature: "temperature",
      value_columns: ["shap"],
      records: [
        { temperature: 600, shap: -1.0 },
        { temperature: 650, shap: -0.4 },
        { temperature: 700, shap: 0.1 },
        { temperature: 750, shap: 0.7 },
        { temperature: 800, shap: 1.1 },
      ],
    };
  },
  async xaiPdp() {
    return {
      feature: "temperature",
      x_values: [600, 650, 700, 750, 800, 850],
      series: [{ name: "strength", pd_values: [10, 12, 15, 18, 18.2, 18.22] }],
    };
  },
};

const summary = await collectReportAnalysisSummary({
  apiClient: fakeApi,
  reportProblem: "強度を高める条件を把握したい",
  fileName: "trial.csv",
  rows: [
    { temperature: 600, strength: 10 },
    { temperature: 650, strength: 12 },
    { temperature: 700, strength: 15 },
    { temperature: 750, strength: 18 },
    { temperature: 800, strength: 18.1 },
  ],
  features: ["temperature"],
  targets: ["strength"],
  tasks: { strength: "regression" },
  modelInfo: { model_id: "model-1", xai_status: "ready", model_names: ["LightGBM"] },
  comparison: {
    targets: {
      strength: {
        best_model_name: "LightGBM",
        metric: "RMSE",
        best_is_tuned: true,
        best_cv_scores: {
          test: [{ RMSE: 1.23, R2: 0.87 }],
          train: [{ RMSE: 0.62, R2: 0.95 }],
        },
      },
    },
  },
  objectives: { strength: { mode: "direction", value: "max" } },
  bounds: { temperature: { min: 600, max: 850 } },
});

assert.equal(summary.xai.targets.strength.features.length, 1);
assert.equal(summary.xai.targets.strength.features[0].shap.direction, "positive");
assert.equal(summary.xai.targets.strength.features[0].pdp.series[0].kind, "increase_then_plateau");

const prompt = buildChatGptReportPrompt(summary);
assert.match(prompt, /強度を高める条件/);
assert.match(prompt, /temperature/);
assert.match(prompt, /重要度/);
assert.match(prompt, /SHAP/);
assert.match(prompt, /1D PD/);
assert.match(prompt, /RMSE=1\.23/);
assert.doesNotMatch(prompt, /2D PD|2D Partial Dependence|2次元/);

console.log("report-prompt tests passed");
