import assert from "node:assert/strict";

import {
  analyzeModelQuality,
  analyzeNumericDistribution,
  analyzeTargetRelation,
  buildDeterministicReportAnalysis,
  buildFeatureConclusion,
  renderDeterministicAnalysisSection,
} from "./report-local-analysis.js";

const rows = Array.from({ length: 40 }, (_, index) => {
  const temperature = 600 + index * 5;
  const pressure = 1 + index * 0.02;
  return {
    temperature,
    temperature_copy: temperature * 1.01,
    pressure,
    batch: index % 2 ? "B" : "A",
    property: 10 + (temperature - 600) * 0.04 - pressure * 0.3,
  };
});
rows.push({ ...rows[0] });

const distribution = analyzeNumericDistribution(rows, "temperature");
assert.equal(distribution.count, 41);
assert.equal(distribution.missing, 0);
assert.ok(distribution.histogram.length >= 4);

const relation = analyzeTargetRelation({
  rows,
  feature: "temperature",
  target: "property",
  featureNumeric: true,
  targetTask: "regression",
});
assert.ok(relation.spearman > 0.95);
assert.equal(relation.direction, "positive");

const quality = analyzeModelQuality("regression", {
  validationMetrics: { R2: 0.82, RMSE: 0.4 },
  trainMetrics: { R2: 0.9 },
}, { std: 2 });
assert.equal(quality.grade, "high");

const featureConclusion = buildFeatureConclusion({
  feature: {
    feature: "temperature",
    rank: 1,
    importance: 0.7,
    shap: { available: true, direction: "positive", text: "高値側ほど予測値を押し上げる傾向" },
    pdp: { available: true, direction: "positive", text: "property: 増加後に飽和。飽和開始の目安は約 720" },
    consistency: "SHAPと1D PDの方向が整合（増加側）",
  },
  target: "property",
  relation,
  distribution,
  modelQuality: quality,
  correlatedFeatures: [],
  objectiveText: "最大化",
});
assert.equal(featureConclusion.confidence, "high");
assert.match(featureConclusion.text, /高値側を優先/);
assert.match(featureConclusion.text, /飽和/);

const apiClient = {
  xaiSummary: async () => ({
    status: "ready",
    targets: {
      property: {
        status: "ready",
        importance_methods: ["shap", "pfi"],
        shap_features: ["temperature", "pressure"],
        pdp_features: ["temperature", "pressure"],
      },
    },
  }),
  xaiImportance: async () => ({
    items: [
      { feature: "temperature", value: 0.7 },
      { feature: "pressure", value: 0.2 },
    ],
  }),
  xaiShap: async (_modelId, _target, feature) => ({
    feature,
    value_columns: ["shap"],
    records: rows.slice(0, 20).map((row) => ({
      [feature]: row[feature],
      shap: feature === "temperature"
        ? (row.temperature - 700) / 100
        : -(row.pressure - 1.3),
    })),
  }),
  xaiPdp: async (_modelId, _target, feature) => feature === "temperature"
    ? {
        x_values: [600, 650, 700, 720, 740, 760],
        series: [{ name: "property", pd_values: [10, 12, 15, 16, 16.1, 16.15] }],
      }
    : {
        x_values: [1.0, 1.2, 1.4, 1.6],
        series: [{ name: "property", pd_values: [16, 15, 14, 13] }],
      },
};

const comparison = {
  targets: {
    property: {
      best_model_name: "LightGBM",
      metric: "RMSE",
      best_is_tuned: true,
      best_cv_scores: {
        test: [{ R2: 0.82, RMSE: 0.4 }],
        train: [{ R2: 0.9, RMSE: 0.25 }],
      },
    },
  },
};

const analysis = await buildDeterministicReportAnalysis({
  apiClient,
  reportProblem: "propertyを高める条件を調べる",
  fileName: "sample.csv",
  rows,
  features: ["temperature", "temperature_copy", "pressure", "batch"],
  targets: ["property"],
  tasks: { property: "regression" },
  numericColumns: ["temperature", "temperature_copy", "pressure", "property"],
  categoricalColumns: ["batch"],
  missing: 0,
  modelInfo: {
    model_id: "model-1",
    model_names_by_target: { property: ["LightGBM"] },
    xai_status: "ready",
  },
  comparison,
  objectives: { property: { mode: "direction", value: "max" } },
  inverseResult: {
    candidates: [{ temperature: 720, pressure: 1.1 }],
    pareto_size: 1,
  },
});

assert.equal(analysis.dataQuality.duplicateCount, 1);
assert.ok(analysis.correlation.strongPairs.some((item) => item.left === "temperature" && item.right === "temperature_copy"));
assert.equal(analysis.modelQuality.property.grade, "high");
assert.equal(analysis.conclusions[0].items[0].feature, "temperature");
assert.match(analysis.conclusions[0].items[0].text, /SHAP/);
assert.ok(analysis.nextActions.some((item) => item.includes("強相関")));
assert.ok(analysis.nextActions.some((item) => item.includes("追加実験点")));

const html = renderDeterministicAnalysisSection(analysis);
assert.match(html, /id="local-analysis"/);
assert.match(html, /データ品質・分布/);
assert.match(html, /相関・単変量関係/);
assert.match(html, /重要度 → SHAP → Partial Dependence → 統合結論/);
assert.match(html, /結論信頼度/);

console.log("deterministic local analysis report checks passed");
