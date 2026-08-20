import { collectReportAnalysisSummary, formatPromptNumber } from "./report-prompt.js";

const DEFAULT_MAX_HISTOGRAMS = 8;
const DEFAULT_MAX_RELATIONS = 12;
const DEFAULT_MAX_CORRELATION_COLUMNS = 9;
const STRONG_FEATURE_CORRELATION = 0.85;

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
  const clean = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!clean.length) return null;
  if (clean.length === 1) return clean[0];
  const position = (clean.length - 1) * q;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return clean[lower];
  const weight = position - lower;
  return clean[lower] * (1 - weight) + clean[upper] * weight;
}

function standardDeviation(values) {
  const clean = values.filter(Number.isFinite);
  if (clean.length < 2) return null;
  const average = mean(clean);
  const variance = clean.reduce((sum, value) => sum + (value - average) ** 2, 0) / (clean.length - 1);
  return Math.sqrt(variance);
}

function pearson(xs, ys) {
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

function ranks(values) {
  const sorted = values
    .map((value, index) => ({ value, index }))
    .sort((a, b) => a.value - b.value);
  const result = Array(values.length).fill(0);
  let cursor = 0;
  while (cursor < sorted.length) {
    let end = cursor + 1;
    while (end < sorted.length && sorted[end].value === sorted[cursor].value) end += 1;
    const averageRank = (cursor + 1 + end) / 2;
    for (let index = cursor; index < end; index += 1) result[sorted[index].index] = averageRank;
    cursor = end;
  }
  return result;
}

function spearman(xs, ys) {
  if (xs.length !== ys.length || xs.length < 3) return null;
  return pearson(ranks(xs), ranks(ys));
}

function numericPairs(rows, xColumn, yColumn) {
  return (rows || [])
    .map((row) => ({ x: finiteNumber(row?.[xColumn]), y: finiteNumber(row?.[yColumn]) }))
    .filter((item) => item.x !== null && item.y !== null);
}

function skewness(values) {
  const clean = values.filter(Number.isFinite);
  if (clean.length < 3) return null;
  const average = mean(clean);
  const sd = standardDeviation(clean);
  if (!sd) return 0;
  const n = clean.length;
  const thirdMoment = clean.reduce((sum, value) => sum + ((value - average) / sd) ** 3, 0);
  return (n / ((n - 1) * (n - 2))) * thirdMoment;
}

function histogram(values, binCount = 12) {
  const clean = values.filter(Number.isFinite);
  if (!clean.length) return [];
  const minimum = Math.min(...clean);
  const maximum = Math.max(...clean);
  if (minimum === maximum) return [{ start: minimum, end: maximum, count: clean.length }];
  const bins = Math.max(4, Math.min(binCount, Math.ceil(Math.sqrt(clean.length))));
  const width = (maximum - minimum) / bins;
  const result = Array.from({ length: bins }, (_, index) => ({
    start: minimum + width * index,
    end: index === bins - 1 ? maximum : minimum + width * (index + 1),
    count: 0,
  }));
  clean.forEach((value) => {
    const index = Math.min(bins - 1, Math.floor((value - minimum) / width));
    result[index].count += 1;
  });
  return result;
}

export function analyzeNumericDistribution(rows = [], column) {
  const values = rows.map((row) => finiteNumber(row?.[column]));
  const clean = values.filter((value) => value !== null);
  const total = rows.length;
  if (!clean.length) {
    return { column, type: "numeric", count: 0, missing: total, missingRate: total ? 1 : 0, available: false };
  }
  const q25 = quantile(clean, 0.25);
  const q75 = quantile(clean, 0.75);
  const iqr = q75 - q25;
  const lower = q25 - 1.5 * iqr;
  const upper = q75 + 1.5 * iqr;
  const outlierCount = iqr > 0 ? clean.filter((value) => value < lower || value > upper).length : 0;
  const unique = new Set(clean).size;
  return {
    column,
    type: "numeric",
    available: true,
    count: clean.length,
    missing: total - clean.length,
    missingRate: total ? (total - clean.length) / total : 0,
    unique,
    min: Math.min(...clean),
    max: Math.max(...clean),
    mean: mean(clean),
    std: standardDeviation(clean),
    q25,
    median: quantile(clean, 0.5),
    q75,
    skewness: skewness(clean),
    outlierCount,
    outlierRate: clean.length ? outlierCount / clean.length : 0,
    constant: unique <= 1,
    lowSpread: iqr === 0,
    histogram: histogram(clean),
  };
}

export function analyzeCategoricalDistribution(rows = [], column) {
  const values = rows
    .map((row) => row?.[column])
    .filter((value) => value !== null && value !== undefined && value !== "");
  const counts = new Map();
  values.forEach((value) => {
    const key = String(value);
    counts.set(key, (counts.get(key) || 0) + 1);
  });
  const categories = [...counts.entries()]
    .sort((a, b) => b[1] - a[1])
    .map(([value, count]) => ({ value, count }));
  return {
    column,
    type: "categorical",
    available: Boolean(values.length),
    count: values.length,
    missing: rows.length - values.length,
    missingRate: rows.length ? (rows.length - values.length) / rows.length : 0,
    unique: counts.size,
    constant: counts.size <= 1,
    categories,
  };
}

function duplicateCount(rows, columns) {
  const seen = new Set();
  let duplicates = 0;
  (rows || []).forEach((row) => {
    const key = JSON.stringify((columns || []).map((column) => row?.[column] ?? null));
    if (seen.has(key)) duplicates += 1;
    else seen.add(key);
  });
  return duplicates;
}

function etaSquared(groups, values) {
  if (groups.length !== values.length || values.length < 3) return null;
  const average = mean(values);
  if (average === null) return null;
  const grouped = new Map();
  groups.forEach((group, index) => {
    const key = String(group);
    if (!grouped.has(key)) grouped.set(key, []);
    grouped.get(key).push(values[index]);
  });
  if (grouped.size < 2) return null;
  const between = [...grouped.values()].reduce((sum, groupValues) => {
    const groupMean = mean(groupValues);
    return sum + groupValues.length * (groupMean - average) ** 2;
  }, 0);
  const total = values.reduce((sum, value) => sum + (value - average) ** 2, 0);
  return total > 0 ? between / total : null;
}

function cramersV(xs, ys) {
  if (xs.length !== ys.length || xs.length < 3) return null;
  const xLevels = [...new Set(xs.map(String))];
  const yLevels = [...new Set(ys.map(String))];
  if (xLevels.length < 2 || yLevels.length < 2) return null;
  const xIndex = new Map(xLevels.map((value, index) => [value, index]));
  const yIndex = new Map(yLevels.map((value, index) => [value, index]));
  const table = Array.from({ length: xLevels.length }, () => Array(yLevels.length).fill(0));
  xs.forEach((value, index) => {
    table[xIndex.get(String(value))][yIndex.get(String(ys[index]))] += 1;
  });
  const rowTotals = table.map((row) => row.reduce((sum, value) => sum + value, 0));
  const columnTotals = yLevels.map((_, column) => table.reduce((sum, row) => sum + row[column], 0));
  const n = xs.length;
  let chiSquare = 0;
  table.forEach((row, rowIndex) => row.forEach((observed, columnIndex) => {
    const expected = (rowTotals[rowIndex] * columnTotals[columnIndex]) / n;
    if (expected > 0) chiSquare += ((observed - expected) ** 2) / expected;
  }));
  const denominator = n * Math.min(xLevels.length - 1, yLevels.length - 1);
  return denominator > 0 ? Math.sqrt(chiSquare / denominator) : null;
}

function relationStrength(value, metric) {
  const abs = Math.abs(value ?? 0);
  if (metric === "eta2") {
    if (abs >= 0.14) return "強い";
    if (abs >= 0.06) return "中程度";
    if (abs >= 0.01) return "弱い";
    return "ごく弱い";
  }
  if (metric === "cramers_v") {
    if (abs >= 0.5) return "強い";
    if (abs >= 0.3) return "中程度";
    if (abs >= 0.1) return "弱い";
    return "ごく弱い";
  }
  if (abs >= 0.7) return "強い";
  if (abs >= 0.4) return "中程度";
  if (abs >= 0.2) return "弱い";
  return "ごく弱い";
}

export function analyzeTargetRelation({ rows = [], feature, target, featureNumeric, targetTask }) {
  if (targetTask === "regression") {
    if (featureNumeric) {
      const pairs = numericPairs(rows, feature, target);
      if (pairs.length < 3) return null;
      const xs = pairs.map((item) => item.x);
      const ys = pairs.map((item) => item.y);
      const pearsonValue = pearson(xs, ys);
      const spearmanValue = spearman(xs, ys);
      const rankingValue = Math.abs(spearmanValue ?? pearsonValue ?? 0);
      return {
        feature,
        target,
        kind: "numeric_regression",
        metric: "spearman",
        value: spearmanValue,
        pearson: pearsonValue,
        spearman: spearmanValue,
        effect: rankingValue,
        direction: (spearmanValue ?? pearsonValue ?? 0) > 0 ? "positive" : (spearmanValue ?? pearsonValue ?? 0) < 0 ? "negative" : "flat",
        strength: relationStrength(spearmanValue ?? pearsonValue, "correlation"),
        sampleCount: pairs.length,
      };
    }
    const pairs = rows
      .map((row) => ({ group: row?.[feature], value: finiteNumber(row?.[target]) }))
      .filter((item) => item.group !== null && item.group !== undefined && item.group !== "" && item.value !== null);
    if (pairs.length < 3) return null;
    const value = etaSquared(pairs.map((item) => item.group), pairs.map((item) => item.value));
    return value === null ? null : {
      feature,
      target,
      kind: "categorical_regression",
      metric: "eta2",
      value,
      effect: value,
      direction: null,
      strength: relationStrength(value, "eta2"),
      sampleCount: pairs.length,
    };
  }

  if (featureNumeric) {
    const pairs = rows
      .map((row) => ({ group: row?.[target], value: finiteNumber(row?.[feature]) }))
      .filter((item) => item.group !== null && item.group !== undefined && item.group !== "" && item.value !== null);
    if (pairs.length < 3) return null;
    const value = etaSquared(pairs.map((item) => item.group), pairs.map((item) => item.value));
    return value === null ? null : {
      feature,
      target,
      kind: "numeric_classification",
      metric: "eta2",
      value,
      effect: value,
      direction: null,
      strength: relationStrength(value, "eta2"),
      sampleCount: pairs.length,
    };
  }

  const pairs = rows
    .map((row) => ({ x: row?.[feature], y: row?.[target] }))
    .filter((item) => item.x !== null && item.x !== undefined && item.x !== "" && item.y !== null && item.y !== undefined && item.y !== "");
  if (pairs.length < 3) return null;
  const value = cramersV(pairs.map((item) => item.x), pairs.map((item) => item.y));
  return value === null ? null : {
    feature,
    target,
    kind: "categorical_classification",
    metric: "cramers_v",
    value,
    effect: value,
    direction: null,
    strength: relationStrength(value, "cramers_v"),
    sampleCount: pairs.length,
  };
}

function metricLookup(metrics, names) {
  const entries = Object.entries(metrics || {});
  const normalized = new Map(entries.map(([key, value]) => [String(key).toLowerCase().replace(/[^a-z0-9]/g, ""), finiteNumber(value)]));
  for (const name of names) {
    const value = normalized.get(name.toLowerCase().replace(/[^a-z0-9]/g, ""));
    if (value !== undefined && value !== null) return value;
  }
  return null;
}

export function analyzeModelQuality(task, performance, targetDistribution) {
  if (!performance) {
    return { grade: "unassessed", label: "未評価", text: "交差検証結果がないため、モデル妥当性は自動判定していません。", warnings: [] };
  }
  const validation = performance.validationMetrics || {};
  const train = performance.trainMetrics || {};
  const warnings = [];
  let score = null;
  let metric = null;
  let grade = "unassessed";

  if (task === "regression") {
    const r2 = metricLookup(validation, ["r2", "r2score"]);
    if (r2 !== null) {
      score = r2;
      metric = "R²";
      grade = r2 >= 0.8 ? "high" : r2 >= 0.5 ? "medium" : "low";
      const trainR2 = metricLookup(train, ["r2", "r2score"]);
      if (trainR2 !== null && trainR2 - r2 >= 0.15) warnings.push(`TrainとValidationのR²差が${formatPromptNumber(trainR2 - r2)}あり、過学習の可能性があります。`);
    } else {
      const rmse = metricLookup(validation, ["rmse"]);
      const sd = targetDistribution?.std;
      if (rmse !== null && Number.isFinite(sd) && sd > 0) {
        const normalized = rmse / sd;
        score = normalized;
        metric = "RMSE/標準偏差";
        grade = normalized <= 0.5 ? "high" : normalized <= 0.9 ? "medium" : "low";
      }
    }
  } else {
    const f1 = metricLookup(validation, ["f1", "f1score"]);
    const auc = metricLookup(validation, ["rocauc", "auc"]);
    const accuracy = metricLookup(validation, ["accuracy", "acc"]);
    const candidates = [["F1", f1], ["ROC-AUC", auc], ["Accuracy", accuracy]].filter((item) => item[1] !== null);
    if (candidates.length) {
      [metric, score] = candidates[0];
      grade = score >= 0.85 ? "high" : score >= 0.7 ? "medium" : "low";
      const trainScore = metricLookup(train, metric === "F1" ? ["f1", "f1score"] : metric === "ROC-AUC" ? ["rocauc", "auc"] : ["accuracy", "acc"]);
      if (trainScore !== null && trainScore - score >= 0.12) warnings.push(`TrainとValidationの${metric}差が${formatPromptNumber(trainScore - score)}あり、過学習の可能性があります。`);
    }
  }

  if (grade === "unassessed") {
    return { grade, label: "要確認", score, metric, text: "利用可能な指標だけでは予測性能を一意に評価できません。CV指標を確認してください。", warnings };
  }
  const label = grade === "high" ? "良好" : grade === "medium" ? "中程度" : "要注意";
  return {
    grade,
    label,
    score,
    metric,
    text: `${metric}=${formatPromptNumber(score)}。モデルの予測性能は「${label}」と判定しました。`,
    warnings,
  };
}

function relationText(relation) {
  if (!relation) return "単変量解析では十分なデータがありません。";
  if (relation.metric === "spearman") {
    const direction = relation.direction === "positive" ? "正" : relation.direction === "negative" ? "負" : "ほぼなし";
    return `単変量ではSpearman ρ=${formatPromptNumber(relation.spearman)}（${relation.strength}な${direction}の関係）、Pearson r=${formatPromptNumber(relation.pearson)}です。`;
  }
  if (relation.metric === "eta2") return `群間差の効果量 η²=${formatPromptNumber(relation.value)}（${relation.strength}）です。`;
  return `カテゴリ関連度 Cramér's V=${formatPromptNumber(relation.value)}（${relation.strength}）です。`;
}

function objectiveDirection(objectiveText) {
  if (objectiveText === "最大化") return "max";
  if (objectiveText === "最小化") return "min";
  return null;
}

function modelEffectDirection(feature) {
  const shapDirection = feature?.shap?.direction;
  const pdpDirection = feature?.pdp?.direction;
  if (["positive", "negative"].includes(shapDirection) && shapDirection === pdpDirection) return shapDirection;
  if (["positive", "negative"].includes(pdpDirection)) return pdpDirection;
  if (["positive", "negative"].includes(shapDirection)) return shapDirection;
  return null;
}

function designGuidance(feature, objectiveTextValue) {
  const objective = objectiveDirection(objectiveTextValue);
  const effect = modelEffectDirection(feature);
  if (!objective) return "目標値指定または目的条件未設定のため、単純な増減方向は提案しません。";
  if (!effect) return "非線形または方向不一致のため、PDが示す閾値・極値周辺を候補として追加検証してください。";
  const raise = (objective === "max" && effect === "positive") || (objective === "min" && effect === "negative");
  return raise
    ? "条件設計では高値側を優先する方向が候補です。ただし観測範囲内で検証してください。"
    : "条件設計では低値側を優先する方向が候補です。ただし観測範囲内で検証してください。";
}

function confidenceForFeature({ modelQuality, feature, relation, distribution, correlatedFeatures }) {
  let score = 0;
  if (modelQuality?.grade === "high") score += 2;
  else if (modelQuality?.grade === "medium") score += 1;
  else if (modelQuality?.grade === "low") score -= 1;
  if (feature?.shap?.available && feature?.pdp?.available) score += 1;
  if (String(feature?.consistency || "").includes("方向が整合")) score += 2;
  if (String(feature?.consistency || "").includes("一致しない")) score -= 1;
  const pdpDirection = feature?.pdp?.direction;
  if (relation?.direction && ["positive", "negative"].includes(pdpDirection)) {
    score += relation.direction === pdpDirection ? 1 : -1;
  }
  if ((distribution?.missingRate || 0) >= 0.2) score -= 1;
  if ((distribution?.outlierRate || 0) >= 0.1) score -= 1;
  if (correlatedFeatures?.length) score -= 1;
  if ((distribution?.count || 0) >= 30) score += 1;
  return score >= 4 ? "high" : score >= 2 ? "medium" : "low";
}

export function buildFeatureConclusion({
  feature,
  target,
  relation,
  distribution,
  modelQuality,
  correlatedFeatures = [],
  objectiveText: targetObjective = "未設定",
}) {
  const confidence = confidenceForFeature({ modelQuality, feature, relation, distribution, correlatedFeatures });
  const importanceLabel = feature.rank <= 2 ? "最重要クラス" : feature.rank <= 4 ? "主要" : "上位";
  const notes = [];
  if (correlatedFeatures.length) notes.push(`他特徴量（${correlatedFeatures.join(", ")}）と強い相関があり、個別効果の分離には注意が必要です。`);
  if ((distribution?.missingRate || 0) >= 0.2) notes.push(`欠損率が${formatPromptNumber(distribution.missingRate * 100)}%と高めです。`);
  if ((distribution?.outlierRate || 0) >= 0.1) notes.push(`IQR基準の外れ値率が${formatPromptNumber(distribution.outlierRate * 100)}%です。`);
  const shapText = feature?.shap?.available ? `SHAPでは${feature.shap.text}。` : "SHAPは利用できません。";
  const pdpText = feature?.pdp?.available ? `1D PDでは${feature.pdp.text}。` : "1D PDは利用できません。";
  return {
    target,
    feature: feature.feature,
    rank: feature.rank,
    importance: feature.importance,
    confidence,
    confidenceLabel: confidence === "high" ? "高" : confidence === "medium" ? "中" : "低",
    relation,
    guidance: designGuidance(feature, targetObjective),
    notes,
    text: [
      `${feature.feature}は重要度${feature.rank}位の${importanceLabel}因子です。`,
      relationText(relation),
      shapText,
      pdpText,
      `SHAP/PD整合性: ${feature.consistency || "未判定"}。`,
      designGuidance(feature, targetObjective),
      ...notes,
    ].join(" "),
  };
}

function buildCorrelationMatrix(rows, columns) {
  return columns.map((rowColumn) => columns.map((columnColumn) => {
    if (rowColumn === columnColumn) return 1;
    const pairs = numericPairs(rows, rowColumn, columnColumn);
    return pairs.length >= 3 ? pearson(pairs.map((item) => item.x), pairs.map((item) => item.y)) : null;
  }));
}

function strongFeatureCorrelations(rows, features, numericSet) {
  const numericFeatures = features.filter((feature) => numericSet.has(feature));
  const pairs = [];
  for (let left = 0; left < numericFeatures.length; left += 1) {
    for (let right = left + 1; right < numericFeatures.length; right += 1) {
      const values = numericPairs(rows, numericFeatures[left], numericFeatures[right]);
      if (values.length < 3) continue;
      const value = pearson(values.map((item) => item.x), values.map((item) => item.y));
      if (value !== null && Math.abs(value) >= STRONG_FEATURE_CORRELATION) {
        pairs.push({ left: numericFeatures[left], right: numericFeatures[right], value });
      }
    }
  }
  return pairs.sort((a, b) => Math.abs(b.value) - Math.abs(a.value)).slice(0, 20);
}

function selectCorrelationColumns({ targets, xaiSummary, numericColumns, maxColumns }) {
  const numericSet = new Set(numericColumns);
  const candidates = [];
  targets.filter((target) => numericSet.has(target)).forEach((target) => candidates.push(target));
  Object.values(xaiSummary?.targets || {}).forEach((target) => {
    (target.features || []).forEach((feature) => {
      if (numericSet.has(feature.feature)) candidates.push(feature.feature);
    });
  });
  numericColumns.forEach((column) => candidates.push(column));
  return [...new Set(candidates)].slice(0, maxColumns);
}

function analysisWarnings({ rows, distributions, strongCorrelations, modelQuality, baseSummary }) {
  const warnings = [...(baseSummary.warnings || [])];
  if (rows.length && rows.length < 30) warnings.push(`サンプル数が${rows.length}件と少なく、統計量・モデル解釈の安定性に注意が必要です。`);
  distributions.forEach((item) => {
    if (item.missingRate >= 0.2) warnings.push(`${item.column}: 欠損率 ${formatPromptNumber(item.missingRate * 100)}%。`);
    if (item.constant) warnings.push(`${item.column}: 値が実質的に一定です。`);
    if (item.type === "numeric" && Math.abs(item.skewness || 0) >= 2) warnings.push(`${item.column}: 分布の歪みが大きいです（skew=${formatPromptNumber(item.skewness)}）。`);
    if (item.type === "numeric" && item.outlierRate >= 0.1) warnings.push(`${item.column}: IQR基準の外れ値が ${formatPromptNumber(item.outlierRate * 100)}%。`);
  });
  strongCorrelations.slice(0, 5).forEach((item) => warnings.push(`${item.left} と ${item.right} は強く相関しています（r=${formatPromptNumber(item.value)}）。重要度の分配やPD解釈に注意してください。`));
  Object.entries(modelQuality).forEach(([target, quality]) => {
    if (quality.grade === "low") warnings.push(`${target}: モデル性能が要注意のため、重要度・PDの結論を強く断定しないでください。`);
    quality.warnings.forEach((warning) => warnings.push(`${target}: ${warning}`));
  });
  return [...new Set(warnings)];
}

function nextActions({ warnings, conclusions, optimization }) {
  const actions = [];
  if (warnings.some((warning) => warning.includes("欠損率") || warning.includes("外れ値"))) actions.push("欠損・外れ値処理を変えた感度分析を行い、重要因子ランキングが維持されるか確認する。");
  if (warnings.some((warning) => warning.includes("強く相関"))) actions.push("強相関の説明変数は片方を除いた再学習、または相関した変数群として解釈し、個別効果の断定を避ける。");
  if (warnings.some((warning) => warning.includes("モデル性能") || warning.includes("過学習"))) actions.push("データ追加、特徴量見直し、CV条件の確認を行い、予測性能を改善してからXAI結論を再評価する。");
  const nonlinear = conclusions.flatMap((item) => item.items || []).filter((item) => /閾値|飽和|U字|非線形|極値/.test(item.text));
  if (nonlinear.length) actions.push("PDで閾値・飽和・極値が示された領域の前後に追加実験点を置き、局所的な再現性を確認する。");
  if (conclusions.flatMap((item) => item.items || []).some((item) => item.confidence === "low")) actions.push("信頼度「低」の因子は追加データ、相互作用解析、条件を分けた再解析で確認する。");
  if ((optimization?.candidateCount || 0) > 0) actions.push("逆解析の上位候補を実験検証し、予測値と実測値のずれを次回モデル更新へ反映する。");
  if (!actions.length) actions.push("主要因子のPDが示す有望領域で確認実験を行い、モデル上の傾向が実験で再現するか検証する。");
  return actions;
}

export async function buildDeterministicReportAnalysis({
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
  numericColumns = [],
  categoricalColumns = [],
  topFeatures = 8,
  maxHistograms = DEFAULT_MAX_HISTOGRAMS,
  maxRelations = DEFAULT_MAX_RELATIONS,
  maxCorrelationColumns = DEFAULT_MAX_CORRELATION_COLUMNS,
  onProgress,
} = {}) {
  onProgress?.("データ品質・分布・相関を解析しています...");
  const baseSummary = await collectReportAnalysisSummary({
    apiClient,
    reportProblem,
    fileName,
    rows,
    features,
    targets,
    tasks,
    missing,
    modelInfo,
    comparison,
    inverseResult,
    objectives,
    bounds,
    topFeatures,
    onProgress,
  });
  const numericSet = new Set(numericColumns);
  const allColumns = [...new Set([...features, ...targets])];
  const distributions = allColumns.map((column) => numericSet.has(column)
    ? analyzeNumericDistribution(rows, column)
    : analyzeCategoricalDistribution(rows, column));
  const distributionByColumn = Object.fromEntries(distributions.map((item) => [item.column, item]));
  const duplicates = duplicateCount(rows, allColumns);
  const relationsByTarget = {};
  targets.forEach((target) => {
    relationsByTarget[target] = features
      .map((feature) => analyzeTargetRelation({
        rows,
        feature,
        target,
        featureNumeric: numericSet.has(feature),
        targetTask: tasks[target] || "unknown",
      }))
      .filter(Boolean)
      .sort((a, b) => (b.effect || 0) - (a.effect || 0));
  });
  const strongCorrelations = strongFeatureCorrelations(rows, features, numericSet);
  const correlationColumns = selectCorrelationColumns({
    targets,
    xaiSummary: baseSummary.xai,
    numericColumns: numericColumns.filter((column) => allColumns.includes(column)),
    maxColumns: maxCorrelationColumns,
  });
  const modelQuality = Object.fromEntries(targets.map((target) => [
    target,
    analyzeModelQuality(
      tasks[target],
      baseSummary.model.targets?.[target]?.performance,
      distributionByColumn[target],
    ),
  ]));

  const conclusions = targets.map((target) => {
    const xaiFeatures = baseSummary.xai.targets?.[target]?.features || [];
    const relationMap = new Map((relationsByTarget[target] || []).map((item) => [item.feature, item]));
    return {
      target,
      task: tasks[target] || "unknown",
      modelQuality: modelQuality[target],
      items: xaiFeatures.map((feature) => {
        const correlatedFeatures = strongCorrelations
          .filter((pair) => pair.left === feature.feature || pair.right === feature.feature)
          .map((pair) => pair.left === feature.feature ? pair.right : pair.left);
        return buildFeatureConclusion({
          feature,
          target,
          relation: relationMap.get(feature.feature) || null,
          distribution: distributionByColumn[feature.feature],
          modelQuality: modelQuality[target],
          correlatedFeatures,
          objectiveText: baseSummary.optimization.objectives?.[target] || "未設定",
        });
      }),
    };
  });

  const prioritizedHistogramColumns = [...new Set([
    ...targets,
    ...conclusions.flatMap((target) => target.items.map((item) => item.feature)),
    ...features,
  ])].filter((column) => distributionByColumn[column]?.available).slice(0, maxHistograms);
  const warnings = analysisWarnings({ rows, distributions, strongCorrelations, modelQuality, baseSummary });
  const actions = nextActions({ warnings, conclusions, optimization: baseSummary.optimization });
  const overall = conclusions.map((target) => {
    const top = target.items.slice(0, 3);
    if (!top.length) return `${target.target}: XAI結果がないため重要因子の統合結論は作成していません。`;
    return `${target.target}: ${top.map((item) => `${item.feature}（重要度${item.rank}位・信頼度${item.confidenceLabel}）`).join("、")}を主要因子として抽出しました。`;
  });

  return {
    baseSummary,
    dataQuality: {
      rowCount: rows.length,
      columnCount: allColumns.length,
      duplicateCount: duplicates,
      duplicateRate: rows.length ? duplicates / rows.length : 0,
      missingCount: Number(missing || 0),
      distributions,
    },
    histograms: prioritizedHistogramColumns.map((column) => distributionByColumn[column]),
    targetRelations: Object.fromEntries(Object.entries(relationsByTarget).map(([target, items]) => [target, items.slice(0, maxRelations)])),
    correlation: {
      columns: correlationColumns,
      matrix: buildCorrelationMatrix(rows, correlationColumns),
      strongPairs: strongCorrelations,
    },
    modelQuality,
    conclusions,
    overall,
    warnings,
    nextActions: actions,
  };
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function percent(value) {
  return Number.isFinite(value) ? `${formatPromptNumber(value * 100, 2)}%` : "—";
}

function metricCard(label, value, note = "") {
  return `<div class="metric-card"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong>${note ? `<small>${escapeHtml(note)}</small>` : ""}</div>`;
}

function histogramSvg(distribution) {
  if (distribution.type === "categorical") {
    const items = distribution.categories.slice(0, 10);
    if (!items.length) return '<p class="empty">分布データなし</p>';
    const max = Math.max(...items.map((item) => item.count), 1);
    return `<div class="local-histogram categorical">${items.map((item) => `<div class="local-category-row"><span>${escapeHtml(item.value)}</span><div><i style="width:${(item.count / max) * 100}%"></i></div><strong>${item.count}</strong></div>`).join("")}</div>`;
  }
  const bins = distribution.histogram || [];
  if (!bins.length) return '<p class="empty">分布データなし</p>';
  const max = Math.max(...bins.map((bin) => bin.count), 1);
  const width = 520;
  const height = 150;
  const padding = { left: 34, right: 12, top: 12, bottom: 30 };
  const innerWidth = width - padding.left - padding.right;
  const innerHeight = height - padding.top - padding.bottom;
  const barWidth = innerWidth / bins.length;
  const bars = bins.map((bin, index) => {
    const barHeight = (bin.count / max) * innerHeight;
    return `<rect x="${padding.left + index * barWidth + 1}" y="${padding.top + innerHeight - barHeight}" width="${Math.max(1, barWidth - 2)}" height="${barHeight}" rx="2"/>`;
  }).join("");
  return `<svg class="local-histogram-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(distribution.column)} histogram">${bars}<line x1="${padding.left}" y1="${padding.top + innerHeight}" x2="${padding.left + innerWidth}" y2="${padding.top + innerHeight}"/><text x="${padding.left}" y="${height - 8}">${escapeHtml(formatPromptNumber(distribution.min, 3))}</text><text x="${padding.left + innerWidth}" y="${height - 8}" text-anchor="end">${escapeHtml(formatPromptNumber(distribution.max, 3))}</text></svg>`;
}

function distributionNote(item) {
  if (item.type === "categorical") return `カテゴリ ${item.unique}、欠損 ${percent(item.missingRate)}`;
  const skew = item.skewness === null ? "—" : formatPromptNumber(item.skewness, 2);
  return `中央値 ${formatPromptNumber(item.median, 3)} / skew ${skew} / 外れ値 ${percent(item.outlierRate)} / 欠損 ${percent(item.missingRate)}`;
}

function renderHistograms(analysis) {
  if (!analysis.histograms.length) return '<p class="empty">表示できる分布がありません。</p>';
  return `<div class="local-histogram-grid">${analysis.histograms.map((item) => `<article class="local-chart-card"><h4>${escapeHtml(item.column)}</h4><p>${escapeHtml(distributionNote(item))}</p>${histogramSvg(item)}</article>`).join("")}</div>`;
}

function correlationCell(value) {
  if (!Number.isFinite(value)) return '<td class="local-corr-na">—</td>';
  const magnitude = Math.min(1, Math.abs(value));
  const alpha = 0.08 + magnitude * 0.28;
  const background = value >= 0 ? `rgba(185,79,87,${alpha})` : `rgba(50,95,160,${alpha})`;
  return `<td style="background:${background}">${escapeHtml(formatPromptNumber(value, 2))}</td>`;
}

function renderCorrelationMatrix(analysis) {
  const { columns, matrix } = analysis.correlation;
  if (!columns.length) return '<p class="empty">相関行列を作成できる数値列がありません。</p>';
  return `<div class="table-scroll local-correlation-table"><table><thead><tr><th></th>${columns.map((column) => `<th>${escapeHtml(column)}</th>`).join("")}</tr></thead><tbody>${columns.map((column, rowIndex) => `<tr><th>${escapeHtml(column)}</th>${matrix[rowIndex].map(correlationCell).join("")}</tr>`).join("")}</tbody></table></div>`;
}

function relationValue(item) {
  if (item.metric === "spearman") return `ρ=${formatPromptNumber(item.spearman, 3)} / r=${formatPromptNumber(item.pearson, 3)}`;
  if (item.metric === "eta2") return `η²=${formatPromptNumber(item.value, 3)}`;
  return `V=${formatPromptNumber(item.value, 3)}`;
}

function renderRelations(analysis) {
  return Object.entries(analysis.targetRelations).map(([target, items]) => `<article class="subcard local-relation-card"><h4>${escapeHtml(target)}との単変量関係</h4>${items.length ? `<div class="table-scroll"><table><thead><tr><th>特徴量</th><th>指標</th><th>強さ</th><th>N</th></tr></thead><tbody>${items.map((item) => `<tr><td>${escapeHtml(item.feature)}</td><td>${escapeHtml(relationValue(item))}</td><td>${escapeHtml(item.strength)}</td><td>${item.sampleCount}</td></tr>`).join("")}</tbody></table></div>` : '<p class="empty">単変量関連度を計算できませんでした。</p>'}</article>`).join("");
}

function qualityClass(grade) {
  if (grade === "high") return "success";
  if (grade === "low") return "warning";
  return "";
}

function renderModelQuality(analysis) {
  return `<div class="local-quality-grid">${Object.entries(analysis.modelQuality).map(([target, quality]) => `<article class="subcard local-quality-card"><div class="card-heading"><div><span>MODEL VALIDITY</span><h4>${escapeHtml(target)}</h4></div><span class="badge ${qualityClass(quality.grade)}">${escapeHtml(quality.label)}</span></div><p>${escapeHtml(quality.text)}</p>${quality.warnings.length ? `<ul>${quality.warnings.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>` : ""}</article>`).join("")}</div>`;
}

function renderConclusions(analysis) {
  const targets = analysis.conclusions.map((target) => `<article class="subcard local-conclusion-target"><div class="card-heading"><div><span>INTEGRATED INTERPRETATION</span><h3>${escapeHtml(target.target)}</h3></div><span class="badge ${qualityClass(target.modelQuality.grade)}">モデル ${escapeHtml(target.modelQuality.label)}</span></div>${target.items.length ? `<div class="local-conclusion-list">${target.items.map((item) => `<article class="local-conclusion-item"><div class="local-conclusion-heading"><div><strong>${item.rank}. ${escapeHtml(item.feature)}</strong><span>重要度 ${escapeHtml(formatPromptNumber(item.importance, 4))}</span></div><span class="local-confidence ${item.confidence}">結論信頼度 ${escapeHtml(item.confidenceLabel)}</span></div><p>${escapeHtml(item.text)}</p></article>`).join("")}</div>` : '<p class="empty">重要度・SHAP・PDを統合できるXAI結果がありません。</p>'}</article>`).join("");
  return targets || '<p class="empty">統合結論を作成できませんでした。</p>';
}

export const LOCAL_ANALYSIS_CSS = `
    .local-analysis-summary{padding:16px;border:1px solid #efd0d3;border-radius:14px;background:#fff7f7;margin-bottom:14px}.local-analysis-summary h3{margin:0 0 8px}.local-analysis-summary ul{margin:0;padding-left:20px}.local-analysis-block{margin-top:22px}.local-analysis-block>h3{margin:0 0 4px;font-size:16px}.local-analysis-block>.local-lead{margin:0 0 12px;color:var(--muted);font-size:11px}.local-histogram-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.local-chart-card{padding:14px;border:1px solid var(--line);border-radius:13px;background:#fff}.local-chart-card h4{margin:0}.local-chart-card>p{margin:2px 0 8px;color:var(--muted);font-size:10px}.local-histogram-svg{display:block;width:100%;min-height:130px}.local-histogram-svg rect{fill:var(--primary)}.local-histogram-svg line{stroke:#b9c3d0}.local-histogram-svg text{font-size:9px;fill:var(--muted)}.local-histogram.categorical{display:grid;gap:5px}.local-category-row{display:grid;grid-template-columns:minmax(70px,130px) 1fr 35px;gap:7px;align-items:center;font-size:10px}.local-category-row>div{height:9px;background:#edf1f6;border-radius:999px;overflow:hidden}.local-category-row i{display:block;height:100%;background:var(--primary);border-radius:999px}.local-correlation-table td{text-align:center;font-variant-numeric:tabular-nums}.local-corr-na{background:#f6f7f9;color:var(--muted)}.local-quality-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.local-quality-card p{margin:8px 0 0;color:#475569}.local-quality-card ul{margin:8px 0 0;padding-left:20px;color:#7f373d}.badge.warning{background:#fff1d6;color:#9c650d}.local-conclusion-list{display:grid;gap:10px;margin-top:12px}.local-conclusion-item{padding:14px;border:1px solid var(--line);border-radius:12px;background:#fbfcfe}.local-conclusion-heading{display:flex;justify-content:space-between;gap:12px;align-items:flex-start}.local-conclusion-heading>div{display:grid}.local-conclusion-heading span{color:var(--muted);font-size:9px}.local-conclusion-item p{margin:8px 0 0;color:#475569}.local-confidence{padding:4px 7px;border-radius:999px;font-size:9px;font-weight:800;white-space:nowrap}.local-confidence.high{background:var(--success-soft);color:var(--success)}.local-confidence.medium{background:#fff4df;color:#a4680d}.local-confidence.low{background:#fff0f0;color:#a13d45}.local-warning-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.local-warning-grid ul{margin:0;padding-left:20px}.local-warning-grid li+li{margin-top:5px}
    @media(max-width:850px){.local-histogram-grid,.local-quality-grid,.local-warning-grid{grid-template-columns:1fr}.local-conclusion-heading{display:block}.local-confidence{display:inline-block;margin-top:6px}}
`;

export function renderDeterministicAnalysisSection(analysis) {
  const warnings = analysis.warnings.length
    ? analysis.warnings.map((item) => `<li>${escapeHtml(item)}</li>`).join("")
    : "<li>主要な自動警告はありません。</li>";
  const actions = analysis.nextActions.map((item) => `<li>${escapeHtml(item)}</li>`).join("");
  return `
    <section class="report-section" id="local-analysis">
      <header class="section-heading"><span>05A</span><div><h2>自動分析所見</h2><p>LLMを使わず、統計・相関・モデル評価・重要度・SHAP・1D PDをルールベースで統合した所見です。</p></div></header>
      <article class="local-analysis-summary">
        <h3>重要因子に関する総括</h3>
        <ul>${analysis.overall.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
      </article>
      <div class="metric-grid">
        ${metricCard("Samples", analysis.dataQuality.rowCount)}
        ${metricCard("Duplicate rows", analysis.dataQuality.duplicateCount, percent(analysis.dataQuality.duplicateRate))}
        ${metricCard("Missing cells", analysis.dataQuality.missingCount)}
        ${metricCard("Strong correlations", analysis.correlation.strongPairs.length, `|r| ≥ ${STRONG_FEATURE_CORRELATION}`)}
      </div>
      <section class="local-analysis-block">
        <h3>1. データ品質・分布</h3>
        <p class="local-lead">欠損・重複・外れ値・歪度を確認し、主要列のヒストグラムまたはカテゴリ頻度を表示します。</p>
        ${renderHistograms(analysis)}
      </section>
      <section class="local-analysis-block">
        <h3>2. 相関・単変量関係</h3>
        <p class="local-lead">数値列はPearson/Spearman、カテゴリを含む関係はη²またはCramér's Vで補完します。相関は因果を意味しません。</p>
        ${renderCorrelationMatrix(analysis)}
        ${renderRelations(analysis)}
      </section>
      <section class="local-analysis-block">
        <h3>3. 機械学習モデルの妥当性</h3>
        <p class="local-lead">交差検証結果を基準に、XAIをどの程度強く解釈できるかを判定します。</p>
        ${renderModelQuality(analysis)}
      </section>
      <section class="local-analysis-block">
        <h3>4. 重要度 → SHAP → Partial Dependence → 統合結論</h3>
        <p class="local-lead">重要度上位因子について単変量傾向、SHAP方向、1D PD形状を突き合わせ、特徴量ごとに結論信頼度を付けます。</p>
        ${renderConclusions(analysis)}
      </section>
      <section class="local-analysis-block">
        <h3>5. 注意事項・次のアクション</h3>
        <div class="local-warning-grid">
          <article class="subcard note-card"><h4>解釈上の注意</h4><ul>${warnings}</ul></article>
          <article class="subcard"><h4>推奨する確認</h4><ul>${actions}</ul></article>
        </div>
      </section>
    </section>`;
}
