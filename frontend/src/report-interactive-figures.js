import { api } from "./api";

const IMPORTANCE_METHODS = [
  { method: "model", label: "モデル固有重要度" },
  { method: "shap", label: "SHAP重要度" },
  { method: "pfi", label: "Permutation Importance" },
];

function clone(value) {
  return value === null || value === undefined
    ? value
    : JSON.parse(JSON.stringify(value));
}

function uniqueStrings(values) {
  return [...new Set((values || [])
    .filter((value) => value !== null && value !== undefined && value !== "")
    .map(String))];
}

async function yyFigures(modelId, target, task) {
  let cv = true;
  let response = null;
  try {
    response = await api.visualizationYy(modelId, target, { cv: true, residual: false });
  } catch {
    cv = false;
    try {
      response = await api.visualizationYy(modelId, target, { cv: false, residual: false });
    } catch {
      return [];
    }
  }
  const result = [clone(response.figure)];
  if (task === "regression") {
    try {
      const residual = await api.visualizationYy(modelId, target, { cv, residual: true });
      result.push(clone(residual.figure));
    } catch {
      result.push(null);
    }
  }
  return result;
}

function normalizedImportanceItems(response) {
  const parsed = (Array.isArray(response?.items) ? response.items : [])
    .map((item) => ({ feature: String(item?.feature ?? ""), value: Number(item?.value) }))
    .filter((item) => item.feature && Number.isFinite(item.value));
  const maximum = Math.max(0, ...parsed.map((item) => Math.abs(item.value)));
  return Object.fromEntries(parsed.map((item) => [
    item.feature,
    maximum > 0 ? item.value / maximum : 0,
  ]));
}

function combinedImportanceFigure(responses, target) {
  const valuesByMethod = Object.fromEntries(IMPORTANCE_METHODS.map((item) => [
    item.method,
    normalizedImportanceItems(responses[item.method]),
  ]));
  const features = uniqueStrings(Object.values(valuesByMethod)
    .flatMap((values) => Object.keys(values)));
  const selected = features
    .map((feature) => ({
      feature,
      score: IMPORTANCE_METHODS.reduce(
        (sum, item) => sum + Math.abs(valuesByMethod[item.method][feature] || 0),
        0,
      ) / IMPORTANCE_METHODS.length,
    }))
    .sort((left, right) => left.score - right.score)
    .slice(-10)
    .map((item) => item.feature);
  if (!selected.length) return null;
  return {
    data: IMPORTANCE_METHODS.map((item) => ({
      type: "bar",
      orientation: "h",
      name: item.label,
      y: selected,
      x: selected.map((feature) => valuesByMethod[item.method][feature] || 0),
      hovertemplate: "%{y}<br>%{x:.4f}<extra>%{fullData.name}</extra>",
    })),
    layout: {
      title: `Feature importance: ${target}`,
      barmode: "group",
      xaxis: { title: "Normalized importance", zeroline: true },
      yaxis: { title: "Feature", automargin: true },
      legend: { orientation: "h", y: 1.12 },
    },
  };
}

async function importanceFigures(modelId, target) {
  const responses = {};
  for (const item of IMPORTANCE_METHODS) {
    try {
      responses[item.method] = await api.xaiImportance(modelId, target, {
        method: item.method,
        combined: true,
        top_n: 1000,
      });
    } catch {
      responses[item.method] = { items: [] };
    }
  }
  return [combinedImportanceFigure(responses, target)];
}

async function shapFigures(modelId, target, staticItems) {
  const result = [];
  for (const item of staticItems) {
    const output = item?.metadata?.selected_output || "";
    try {
      const response = await api.visualizationBeeswarm(modelId, target, {
        ...(output ? { output } : {}),
        top_n: 10,
      });
      result.push(clone(response.figure));
    } catch {
      result.push(null);
    }
  }
  return result;
}

function finiteMean(values, fallback = 0) {
  const numeric = values.map(Number).filter(Number.isFinite);
  return numeric.length
    ? numeric.reduce((sum, value) => sum + value, 0) / numeric.length
    : fallback;
}

function shapValueColumns(response) {
  const configured = Array.isArray(response?.value_columns)
    ? response.value_columns.map(String)
    : [];
  if (configured.length) return configured;
  const first = response?.records?.[0] || {};
  return Object.keys(first).filter((key) => key === "shap" || key.startsWith("shap_"));
}

function resolveShapColumn(response, output, outputIndex) {
  const columns = shapValueColumns(response);
  const requested = String(output ?? "");
  const candidates = [
    requested,
    requested ? `shap_${requested}` : "",
    outputIndex === 0 ? "shap" : "",
    columns[outputIndex],
    columns.length === 1 ? columns[0] : "",
  ].filter(Boolean);
  return candidates.find((candidate) => columns.includes(candidate)) || "";
}

function baselineForOutput(rows, target, task, output, shapResponse) {
  const direct = shapResponse?.base_value ?? shapResponse?.expected_value;
  if (Number.isFinite(Number(direct))) return Number(direct);
  if (task === "classification") {
    return rows.length
      ? rows.filter((row) => String(row?.[target]) === String(output)).length / rows.length
      : 0;
  }
  return finiteMean(rows.map((row) => row?.[target]), 0);
}

function legacyPdFigure({ target, task, feature, rows, pdResponse, shapResponse, output }) {
  const seriesItems = (pdResponse?.series || []).filter((item) => (
    item && item.name !== null && item.name !== undefined && Array.isArray(item.pd_values)
  ));
  let outputIndex = seriesItems.findIndex((item) => String(item.name) === String(output));
  if (outputIndex < 0) outputIndex = 0;
  const series = seriesItems[outputIndex];
  if (!series) return null;
  const outputName = String(series.name ?? output ?? target);
  const shapColumn = resolveShapColumn(shapResponse, outputName, outputIndex);
  const baseline = baselineForOutput(rows, target, task, outputName, shapResponse);
  const rawRows = rows.filter((row) => row?.[feature] !== null && row?.[feature] !== undefined);
  const data = [{
    type: "scatter",
    mode: "lines",
    name: "Partial dependence",
    x: pdResponse?.x_values || [],
    y: series.pd_values,
    line: { color: "#d97706", width: 3 },
  }];
  if (shapColumn) {
    const records = (shapResponse?.records || []).filter((record) => (
      record?.[feature] !== null
      && record?.[feature] !== undefined
      && Number.isFinite(Number(record?.[shapColumn]))
    ));
    data.push({
      type: "scatter",
      mode: "markers",
      name: "Baseline + SHAP",
      x: records.map((record) => record[feature]),
      y: records.map((record) => baseline + Number(record[shapColumn])),
      marker: { color: "#3b82f6", size: 7, opacity: 0.62 },
    });
  }
  data.push({
    type: "scatter",
    mode: "markers",
    name: task === "classification" ? "Observed class" : "Observed target",
    x: rawRows.map((row) => row[feature]),
    y: rawRows.map((row) => (
      task === "classification"
        ? Number(String(row[target]) === outputName)
        : row[target]
    )),
    marker: { color: "#c94f58", size: 7, opacity: 0.48 },
  });
  return {
    data,
    layout: {
      title: task === "classification"
        ? `PD / SHAP / observed: ${target} (${outputName}) / ${feature}`
        : `PD / SHAP / observed: ${target} / ${feature}`,
      xaxis: { title: feature, automargin: true },
      yaxis: {
        title: task === "classification" ? `P(${outputName}) / observed` : target,
        automargin: true,
      },
      legend: { orientation: "h", y: 1.15 },
      shapes: [{
        type: "line",
        xref: "paper",
        x0: 0,
        x1: 1,
        y0: baseline,
        y1: baseline,
        line: { color: "#302929", width: 1, dash: "dash" },
      }],
    },
  };
}

async function pdFigures(modelId, target, task, staticItems, rows) {
  const cache = new Map();
  const result = [];
  for (const item of staticItems) {
    const feature = item?.metadata?.feature;
    const output = item?.metadata?.output || "";
    if (!feature) {
      result.push(null);
      continue;
    }
    if (!cache.has(feature)) {
      cache.set(feature, Promise.all([
        api.xaiPdp(modelId, target, feature, { include_ice: true, max_ice: 30 }),
        api.xaiShap(modelId, target, feature),
      ]).catch(() => null));
    }
    const payload = await cache.get(feature);
    result.push(payload
      ? legacyPdFigure({
          target,
          task,
          feature,
          rows,
          pdResponse: payload[0],
          shapResponse: payload[1],
          output,
        })
      : null);
  }
  return result;
}

async function pd2dFigures(modelId, target, staticItems) {
  const result = [];
  for (const item of staticItems) {
    const metadata = item?.metadata || {};
    const featureX = metadata.feature_x;
    const featureY = metadata.feature_y;
    const output = metadata.selected_output || "";
    if (!featureX || !featureY) {
      result.push(null);
      continue;
    }
    try {
      const response = await api.visualizationPdp2d(modelId, target, {
        feature_x: featureX,
        feature_y: featureY,
        ...(output ? { output } : {}),
      });
      result.push(clone(response.figure));
    } catch {
      result.push(null);
    }
  }
  return result;
}

export async function collectInteractiveFigures({ modelId, visualizations, rows, onProgress }) {
  const result = { targets: {} };
  const entries = Object.entries(visualizations?.targets || {});
  for (let index = 0; index < entries.length; index += 1) {
    const [target, staticTarget] = entries[index];
    const task = staticTarget.task || "regression";
    onProgress?.(`編集用Plotly図を収集しています: ${target} (${index + 1}/${entries.length})`);
    result.targets[target] = {
      yy: await yyFigures(modelId, target, task),
      importance: await importanceFigures(modelId, target),
      shap: await shapFigures(modelId, target, staticTarget.shap || []),
      pd: await pdFigures(modelId, target, task, staticTarget.pd || [], rows || []),
      pd2d: await pd2dFigures(modelId, target, staticTarget.pd2d || []),
    };
  }
  return result;
}
