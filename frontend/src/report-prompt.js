const DEFAULT_TOP_FEATURES = 6;
const MAX_INVERSE_CANDIDATES = 5;

function finiteNumber(value) {
  if (value === null || value === undefined || value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function mean(values) {
  const clean = values.filter(Number.isFinite);
  if (!clean.length) return null;
  return clean.reduce((sum, value) => sum + value, 0) / clean.length;
}

function quantile(values, q) {
  const clean = values.filter(Number.isFinite).sort((left, right) => left - right);
  if (!clean.length) return null;
  if (clean.length === 1) return clean[0];
  const position = (clean.length - 1) * q;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return clean[lower];
  const weight = position - lower;
  return clean[lower] * (1 - weight) + clean[upper] * weight;
}

function correlation(xs, ys) {
  if (xs.length !== ys.length || xs.length < 3) return null;
  const xMean = mean(xs);
  const yMean = mean(ys);
  if (xMean === null || yMean === null) return null;
  let numerator = 0;
  let xSquare = 0;
  let ySquare = 0;
  xs.forEach((x, index) => {
    const dx = x - xMean;
    const dy = ys[index] - yMean;
    numerator += dx * dy;
    xSquare += dx * dx;
    ySquare += dy * dy;
  });
  const denominator = Math.sqrt(xSquare * ySquare);
  return denominator > 0 ? numerator / denominator : null;
}

export function formatPromptNumber(value, digits = 4) {
  const number = finiteNumber(value);
  if (number === null) return String(value ?? "—");
  const abs = Math.abs(number);
  if ((abs > 0 && abs < 0.0001) || abs >= 100000) return number.toExponential(3);
  return new Intl.NumberFormat("ja-JP", { maximumFractionDigits: digits }).format(number);
}

export function summarizeFeatureDistribution(rows = [], feature) {
  const values = rows
    .map((row) => row?.[feature])
    .filter((value) => value !== null && value !== undefined && value !== "");
  if (!values.length) {
    return { type: "unavailable", text: "入力データ上の分布情報なし" };
  }

  const numericValues = values.map(finiteNumber).filter((value) => value !== null);
  if (numericValues.length >= Math.max(3, Math.ceil(values.length * 0.8))) {
    const minimum = Math.min(...numericValues);
    const maximum = Math.max(...numericValues);
    const q25 = quantile(numericValues, 0.25);
    const median = quantile(numericValues, 0.5);
    const q75 = quantile(numericValues, 0.75);
    return {
      type: "numeric",
      count: numericValues.length,
      min: minimum,
      q25,
      median,
      q75,
      max: maximum,
      text: `観測範囲 ${formatPromptNumber(minimum)}〜${formatPromptNumber(maximum)}、中央50% ${formatPromptNumber(q25)}〜${formatPromptNumber(q75)}、中央値 ${formatPromptNumber(median)}`,
    };
  }

  const counts = new Map();
  values.forEach((value) => {
    const key = String(value);
    counts.set(key, (counts.get(key) || 0) + 1);
  });
  const top = [...counts.entries()]
    .sort((left, right) => right[1] - left[1])
    .slice(0, 5)
    .map(([value, count]) => ({ value, count }));
  return {
    type: "categorical",
    count: values.length,
    unique: counts.size,
    top,
    text: `カテゴリ数 ${counts.size}、主なカテゴリ ${top.map((item) => `${item.value}(${item.count})`).join(", ")}`,
  };
}

function summarizeNumericShap(records, feature, valueColumn) {
  const pairs = records
    .map((record) => ({
      x: finiteNumber(record?.[feature]),
      shap: finiteNumber(record?.[valueColumn]),
    }))
    .filter((item) => item.x !== null && item.shap !== null)
    .sort((left, right) => left.x - right.x);
  if (pairs.length < 3) return null;

  const groupSize = Math.max(1, Math.floor(pairs.length / 3));
  const lowMean = mean(pairs.slice(0, groupSize).map((item) => item.shap));
  const highMean = mean(pairs.slice(-groupSize).map((item) => item.shap));
  const meanAbs = mean(pairs.map((item) => Math.abs(item.shap))) || 0;
  const delta = (highMean ?? 0) - (lowMean ?? 0);
  const threshold = Math.max(1e-12, meanAbs * 0.2);
  const direction = delta > threshold ? "positive" : delta < -threshold ? "negative" : "mixed";
  const corr = correlation(
    pairs.map((item) => item.x),
    pairs.map((item) => item.shap),
  );
  const directionText = direction === "positive"
    ? "高値側ほど予測値を押し上げる傾向"
    : direction === "negative"
      ? "高値側ほど予測値を押し下げる傾向"
      : "高値・低値で一方向の差が明瞭ではない";
  const corrText = corr === null ? "" : `（feature-SHAP相関 ${formatPromptNumber(corr, 3)}）`;
  return {
    name: valueColumn,
    direction,
    correlation: corr,
    lowMean,
    highMean,
    text: `${valueColumn}: ${directionText}${corrText}`,
  };
}

function summarizeCategoricalShap(records, feature, valueColumn) {
  const groups = new Map();
  records.forEach((record) => {
    const category = record?.[feature];
    const shap = finiteNumber(record?.[valueColumn]);
    if (category === null || category === undefined || shap === null) return;
    const key = String(category);
    const values = groups.get(key) || [];
    values.push(shap);
    groups.set(key, values);
  });
  const means = [...groups.entries()]
    .map(([category, values]) => ({ category, mean: mean(values) }))
    .filter((item) => item.mean !== null)
    .sort((left, right) => right.mean - left.mean);
  if (means.length < 2) return null;
  const highest = means[0];
  const lowest = means.at(-1);
  return {
    name: valueColumn,
    direction: "categorical",
    highest,
    lowest,
    text: `${valueColumn}: ${highest.category} が相対的に高いSHAP、${lowest.category} が相対的に低いSHAP（各カテゴリ平均）`,
  };
}

export function summarizeShapResponse(response) {
  const records = Array.isArray(response?.records) ? response.records : [];
  const feature = response?.feature;
  const valueColumns = Array.isArray(response?.value_columns) ? response.value_columns : [];
  if (!feature || !records.length || !valueColumns.length) {
    return { available: false, series: [], direction: null, text: "SHAP傾向を取得できませんでした" };
  }

  const numericCount = records
    .map((record) => finiteNumber(record?.[feature]))
    .filter((value) => value !== null).length;
  const numericFeature = numericCount >= Math.max(3, Math.ceil(records.length * 0.8));
  const series = valueColumns.slice(0, 4)
    .map((column) => numericFeature
      ? summarizeNumericShap(records, feature, column)
      : summarizeCategoricalShap(records, feature, column))
    .filter(Boolean);
  if (!series.length) {
    return { available: false, series: [], direction: null, text: "SHAP傾向を安定して要約できませんでした" };
  }
  return {
    available: true,
    numericFeature,
    series,
    direction: series.length === 1 ? series[0].direction : null,
    text: series.map((item) => item.text).join(" / "),
  };
}

function numericPdpShape(xValues, pdValues) {
  const pairs = xValues
    .map((x, index) => ({ x: finiteNumber(x), y: finiteNumber(pdValues[index]) }))
    .filter((item) => item.x !== null && item.y !== null)
    .sort((left, right) => left.x - right.x);
  if (pairs.length < 3) return null;

  const xs = pairs.map((item) => item.x);
  const ys = pairs.map((item) => item.y);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const amplitude = maxY - minY;
  const scale = Math.max(1, Math.max(...ys.map(Math.abs)));
  if (amplitude <= scale * 1e-8) {
    return { kind: "flat", direction: "flat", text: "ほぼ平坦" };
  }

  const diffs = ys.slice(1).map((value, index) => value - ys[index]);
  const tolerance = amplitude * 0.03;
  const positive = diffs.filter((value) => value > tolerance).length;
  const negative = diffs.filter((value) => value < -tolerance).length;
  const active = positive + negative;
  const increasing = active > 0 && positive / active >= 0.8;
  const decreasing = active > 0 && negative / active >= 0.8;
  const minIndex = ys.indexOf(minY);
  const maxIndex = ys.indexOf(maxY);
  const interior = (index) => index > 0 && index < ys.length - 1;

  if (interior(minIndex)
      && ys[0] - minY > amplitude * 0.25
      && ys.at(-1) - minY > amplitude * 0.25) {
    return {
      kind: "u_shape",
      direction: "nonlinear",
      point: xs[minIndex],
      text: `U字型。最小付近は約 ${formatPromptNumber(xs[minIndex])}`,
    };
  }
  if (interior(maxIndex)
      && maxY - ys[0] > amplitude * 0.25
      && maxY - ys.at(-1) > amplitude * 0.25) {
    return {
      kind: "inverse_u_shape",
      direction: "nonlinear",
      point: xs[maxIndex],
      text: `逆U字型。最大付近は約 ${formatPromptNumber(xs[maxIndex])}`,
    };
  }

  const tailSize = Math.max(1, Math.floor(diffs.length / 3));
  const tailChange = mean(diffs.slice(-tailSize).map(Math.abs)) || 0;
  const headChange = mean(diffs.slice(0, Math.max(1, diffs.length - tailSize)).map(Math.abs)) || 0;
  const net = ys.at(-1) - ys[0];
  if (Math.abs(net) > amplitude * 0.6 && headChange > 0 && tailChange < headChange * 0.3) {
    const pointIndex = Math.max(1, ys.length - tailSize - 1);
    return net > 0
      ? {
          kind: "increase_then_plateau",
          direction: "positive",
          point: xs[pointIndex],
          text: `増加後に飽和。飽和開始の目安は約 ${formatPromptNumber(xs[pointIndex])}`,
        }
      : {
          kind: "decrease_then_plateau",
          direction: "negative",
          point: xs[pointIndex],
          text: `減少後に飽和。飽和開始の目安は約 ${formatPromptNumber(xs[pointIndex])}`,
        };
  }

  const absDiffs = diffs.map(Math.abs);
  const averageSlope = mean(absDiffs) || 0;
  const maxSlope = Math.max(...absDiffs);
  const maxSlopeIndex = absDiffs.indexOf(maxSlope);
  if (averageSlope > 0
      && maxSlope > averageSlope * 2.5
      && maxSlopeIndex > 0
      && maxSlopeIndex < diffs.length - 1) {
    const point = (xs[maxSlopeIndex] + xs[maxSlopeIndex + 1]) / 2;
    return diffs[maxSlopeIndex] > 0
      ? {
          kind: "threshold_increase",
          direction: "positive",
          point,
          text: `閾値的な上昇。変化点の目安は約 ${formatPromptNumber(point)}`,
        }
      : {
          kind: "threshold_decrease",
          direction: "negative",
          point,
          text: `閾値的な低下。変化点の目安は約 ${formatPromptNumber(point)}`,
        };
  }

  if (increasing) return { kind: "monotonic_increase", direction: "positive", text: "概ね単調増加" };
  if (decreasing) return { kind: "monotonic_decrease", direction: "negative", text: "概ね単調減少" };
  return { kind: "non_monotonic", direction: "nonlinear", text: "非単調で複雑な変化" };
}

export function summarizePdpResponse(response) {
  const xValues = Array.isArray(response?.x_values) ? response.x_values : [];
  const series = Array.isArray(response?.series) ? response.series : [];
  if (!xValues.length || !series.length) {
    return { available: false, series: [], direction: null, text: "1D PD傾向を取得できませんでした" };
  }

  const numericCount = xValues.map(finiteNumber).filter((value) => value !== null).length;
  const numericFeature = numericCount >= Math.max(3, Math.ceil(xValues.length * 0.8));
  if (!numericFeature) {
    return {
      available: true,
      numericFeature: false,
      series: series.slice(0, 4).map((item) => ({
        name: item.name,
        kind: "categorical",
        direction: "categorical",
        text: `${item.name}: カテゴリ間で予測値が変化`,
      })),
      direction: null,
      text: "カテゴリ水準によってPDが変化",
    };
  }

  const summarized = series.slice(0, 4)
    .map((item) => {
      const shape = numericPdpShape(xValues, Array.isArray(item.pd_values) ? item.pd_values : []);
      return shape ? { name: item.name, ...shape, text: `${item.name}: ${shape.text}` } : null;
    })
    .filter(Boolean);
  if (!summarized.length) {
    return { available: false, series: [], direction: null, text: "1D PD傾向を安定して要約できませんでした" };
  }
  return {
    available: true,
    numericFeature: true,
    series: summarized,
    direction: summarized.length === 1 ? summarized[0].direction : null,
    text: summarized.map((item) => item.text).join(" / "),
  };
}

function consistencyText(shap, pdp) {
  if (!shap?.available || !pdp?.available) return "一方または両方の結果がなく、整合性は判定しない";
  const shapDirection = shap.direction;
  const pdpDirection = pdp.direction;
  if (!["positive", "negative"].includes(shapDirection)
      || !["positive", "negative"].includes(pdpDirection)) {
    return "単純な増減方向だけでは比較できないため、非線形性・相互作用を含めて解釈する";
  }
  if (shapDirection === pdpDirection) {
    return `SHAPと1D PDの方向が整合（${shapDirection === "positive" ? "増加側" : "減少側"}）`;
  }
  return "SHAPと1D PDの方向が一致しないため、相互作用・特徴量相関・データ分布の影響に注意";
}

function chooseImportanceMethod(meta) {
  const methods = Array.isArray(meta?.importance_methods) ? meta.importance_methods : [];
  if (methods.includes("shap")) return "shap";
  if (methods.includes("pfi")) return "pfi";
  if (methods.includes("model")) return "model";
  return null;
}

function targetPerformance(comparison, target) {
  const result = comparison?.targets?.[target];
  if (!result) return null;
  return {
    modelName: result.best_model_name || null,
    selectionMetric: result.metric || null,
    tuned: Boolean(result.best_is_tuned),
    validationMetrics: result.best_cv_scores?.test?.[0] || {},
    trainMetrics: result.best_cv_scores?.train?.[0] || {},
  };
}

function activeModelName(modelInfo, target) {
  return modelInfo?.best_model_names?.[target]
    || modelInfo?.model_names_by_target?.[target]?.[0]
    || modelInfo?.model_names?.[0]
    || null;
}

function objectiveText(objectives, target) {
  const objective = objectives?.[target];
  if (!objective) return "未設定";
  if (objective.mode === "target") return `目標値/クラス ${objective.value}`;
  if (objective.value === "min") return "最小化";
  return "最大化";
}

async function collectTargetXai({ apiClient, modelId, target, meta, rows, features, topFeatures }) {
  const importanceMethod = chooseImportanceMethod(meta);
  if (!importanceMethod) {
    return {
      target,
      status: meta?.status || "unavailable",
      importanceMethod: null,
      features: [],
      warning: "利用可能な重要度がありません",
    };
  }

  const importance = await apiClient.xaiImportance(modelId, target, {
    method: importanceMethod,
    combined: true,
    top_n: Math.max(topFeatures, 1),
  });
  const allowed = new Set(features || []);
  const items = (importance?.items || [])
    .filter((item) => !allowed.size || allowed.has(item.feature))
    .slice(0, topFeatures);
  const shapFeatures = new Set(meta?.shap_features || []);
  const pdpFeatures = new Set(meta?.pdp_features || []);

  const summarizedFeatures = [];
  for (const [index, item] of items.entries()) {
    const feature = item.feature;
    let shap = { available: false, direction: null, text: "利用不可" };
    let pdp = { available: false, direction: null, text: "利用不可" };
    let shapError = null;
    let pdpError = null;

    if (shapFeatures.has(feature)) {
      try {
        shap = summarizeShapResponse(await apiClient.xaiShap(modelId, target, feature));
      } catch (error) {
        shapError = error?.message || String(error);
      }
    }
    if (pdpFeatures.has(feature)) {
      try {
        pdp = summarizePdpResponse(await apiClient.xaiPdp(modelId, target, feature));
      } catch (error) {
        pdpError = error?.message || String(error);
      }
    }

    summarizedFeatures.push({
      feature,
      rank: index + 1,
      importance: finiteNumber(item.value),
      distribution: summarizeFeatureDistribution(rows, feature),
      shap,
      pdp,
      consistency: consistencyText(shap, pdp),
      ...(shapError ? { shapError } : {}),
      ...(pdpError ? { pdpError } : {}),
    });
  }
  return {
    target,
    status: meta?.status || "ready",
    importanceMethod,
    features: summarizedFeatures,
  };
}

export async function collectReportAnalysisSummary({
  apiClient,
  reportProblem = "",
  fileName = "",
  rows = [],
  features = [],
  targets = [],
  tasks = {},
  missing = 0,
  modelInfo = null,
  comparison = null,
  inverseResult = null,
  objectives = {},
  bounds = {},
  topFeatures = DEFAULT_TOP_FEATURES,
  onProgress,
} = {}) {
  const warnings = [];
  const summary = {
    problem: reportProblem?.trim() || "未入力",
    data: {
      fileName: fileName || "未指定",
      rowCount: rows.length,
      featureCount: features.length,
      features: [...features],
      targets: targets.map((target) => ({ target, task: tasks[target] || "unknown" })),
      missingCount: Number(missing || 0),
    },
    model: {
      modelId: modelInfo?.model_id || null,
      xaiStatus: modelInfo?.xai_status || null,
      targets: Object.fromEntries(targets.map((target) => [target, {
        modelName: targetPerformance(comparison, target)?.modelName || activeModelName(modelInfo, target),
        performance: targetPerformance(comparison, target),
      }])),
    },
    xai: { status: "unavailable", targets: {} },
    optimization: {
      objectives: Object.fromEntries(targets.map((target) => [target, objectiveText(objectives, target)])),
      bounds,
      candidates: (inverseResult?.candidates || []).slice(0, MAX_INVERSE_CANDIDATES),
      candidateCount: inverseResult?.candidates?.length || 0,
      paretoSize: inverseResult?.pareto_size ?? null,
    },
    warnings,
  };

  if (!modelInfo?.model_id || !apiClient) {
    warnings.push("学習済みモデルがないため、重要度・SHAP・1D PDの要約は含まれていません。");
    return summary;
  }

  onProgress?.("XAIの利用可能状態を確認しています...");
  try {
    const xaiSummary = await apiClient.xaiSummary(modelInfo.model_id);
    summary.xai.status = xaiSummary?.status || "unknown";
    for (const target of targets) {
      const meta = xaiSummary?.targets?.[target];
      if (!meta || !["ready", "partial"].includes(meta.status)) {
        summary.xai.targets[target] = {
          target,
          status: meta?.status || "unavailable",
          importanceMethod: null,
          features: [],
          warning: meta?.error || "XAIが利用できません",
        };
        continue;
      }
      onProgress?.(`${target}: 重要度・SHAP・1D PDの傾向を要約しています...`);
      try {
        summary.xai.targets[target] = await collectTargetXai({
          apiClient,
          modelId: modelInfo.model_id,
          target,
          meta,
          rows,
          features,
          topFeatures,
        });
      } catch (error) {
        const message = error?.message || String(error);
        summary.xai.targets[target] = {
          target,
          status: "partial",
          importanceMethod: null,
          features: [],
          warning: message,
        };
        warnings.push(`${target} のXAI要約の一部を取得できませんでした: ${message}`);
      }
    }
  } catch (error) {
    const message = error?.message || String(error);
    warnings.push(`XAI要約を取得できませんでした: ${message}`);
  }
  return summary;
}

function metricsText(metrics) {
  const entries = Object.entries(metrics || {});
  if (!entries.length) return "利用可能な交差検証指標なし";
  return entries.map(([name, value]) => `${name}=${formatPromptNumber(value)}`).join(", ");
}

function featurePromptLines(feature) {
  return [
    `#### ${feature.rank}. ${feature.feature}`,
    `- 重要度: ${formatPromptNumber(feature.importance)}（順位 ${feature.rank}）`,
    `- データ分布: ${feature.distribution?.text || "不明"}`,
    `- SHAP: ${feature.shap?.text || "利用不可"}`,
    `- 1D PD: ${feature.pdp?.text || "利用不可"}`,
    `- SHAP/PD整合性: ${feature.consistency}`,
  ].join("\n");
}

export function buildChatGptReportPrompt(summary) {
  const targetSections = summary.data.targets.map(({ target, task }) => {
    const model = summary.model.targets?.[target] || {};
    const performance = model.performance;
    const xai = summary.xai.targets?.[target];
    const featureItems = xai?.features || [];
    return [
      `### ${target}（${task}）`,
      `- 採用モデル: ${model.modelName || "不明"}`,
      performance
        ? `- モデル比較: 選定指標=${performance.selectionMetric || "不明"}, tuned=${performance.tuned ? "yes" : "no"}`
        : "- モデル比較: 未実施または利用可能な結果なし",
      performance ? `- Validation指標: ${metricsText(performance.validationMetrics)}` : "",
      performance ? `- Train指標: ${metricsText(performance.trainMetrics)}` : "",
      `- 逆解析目的: ${summary.optimization.objectives?.[target] || "未設定"}`,
      "",
      "#### 重要度・SHAP・1D PD",
      xai?.importanceMethod ? `重要度方式: ${xai.importanceMethod}` : "重要度方式: 利用不可",
      featureItems.length
        ? featureItems.map(featurePromptLines).join("\n\n")
        : `- ${xai?.warning || "XAI結果なし"}`,
    ].filter(Boolean).join("\n");
  }).join("\n\n");

  const inverseCandidates = summary.optimization.candidates?.length
    ? summary.optimization.candidates
      .map((candidate, index) => `${index + 1}. ${JSON.stringify(candidate)}`)
      .join("\n")
    : "未実施または候補なし";
  const warnings = summary.warnings?.length
    ? summary.warnings.map((warning) => `- ${warning}`).join("\n")
    : "- 特記事項なし";

  return `あなたは材料・製造データ分析の専門家です。以下の「分析で得られた事実」だけを根拠として、日本語の技術レポートを作成してください。

# 分析上のルール
- 相関関係、特徴量重要度、SHAP、Partial Dependenceを因果関係として断定しないでください。
- 特徴量重要度は「モデルがその特徴をどの程度利用したか」を示すもので、影響方向そのものではない点を明記してください。
- SHAPはモデル予測への局所的寄与、1D PDは他変数を平均化した主効果の近似として扱ってください。
- SHAPと1D PDの方向が整合する場合は整合的なモデル傾向として述べ、一致しない場合は相互作用、特徴量相関、データ分布の影響候補を示してください。
- 観測データの範囲や中央50%から外れる領域、データが疎な領域では強い断定を避けてください。
- 「未実施」「利用不可」と記載された結果を推測で補完しないでください。
- 数値的な閾値、飽和点、極値はモデル上の目安であり、追加実験による確認が必要であることを示してください。

# 分析課題
${summary.problem}

# 分析で得られた事実

## データ概要
- ファイル: ${summary.data.fileName}
- サンプル数: ${summary.data.rowCount}
- 説明変数数: ${summary.data.featureCount}
- 説明変数: ${summary.data.features.join(", ") || "未設定"}
- 目的変数: ${summary.data.targets.map((item) => `${item.target}(${item.task})`).join(", ") || "未設定"}
- 欠損セル数: ${summary.data.missingCount}

## モデル・特徴量傾向
${targetSections || "モデル結果なし"}

## 逆解析
- 候補数: ${summary.optimization.candidateCount}
- Pareto候補数: ${summary.optimization.paretoSize ?? "—"}
- 数値探索範囲: ${Object.keys(summary.optimization.bounds || {}).length ? JSON.stringify(summary.optimization.bounds) : "未設定"}
- 上位候補:
${inverseCandidates}

## 利用上の注意・不足情報
${warnings}

# 作成してほしいレポート
以下の順でまとめてください。
1. 分析目的と背景
2. データ概要と分析方法
3. モデル精度・モデル選定結果
4. 主要因子のランキング
5. 各主要因子の影響傾向（重要度・SHAP・1D PDを統合して説明）
6. 閾値・飽和・非線形な傾向など、条件設計に有用なポイント
7. 逆解析候補から得られる条件設計の示唆
8. 結果の限界と因果解釈上の注意
9. 次に確認すべき実験・追加解析
10. 結論

専門家向けの内容を維持しつつ、会議資料にも転用しやすい明瞭な日本語で記述してください。`;
}

export async function generateChatGptReportPrompt(options = {}) {
  const summary = await collectReportAnalysisSummary(options);
  return {
    summary,
    prompt: buildChatGptReportPrompt(summary),
    summarizedFeatureCount: Object.values(summary.xai.targets || {})
      .reduce((count, target) => count + (target.features?.length || 0), 0),
  };
}
