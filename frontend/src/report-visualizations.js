import { api } from "./api";
import { buildHtmlReport, reportFileName } from "./report";

const IMPORTANCE_METHODS = [
  { method: "model", label: "モデル固有重要度" },
  { method: "shap", label: "SHAP重要度" },
  { method: "pfi", label: "Permutation Importance" },
];

const FIGURE_SIZES = {
  yy: { width: 1080, height: 560 },
  importance: { width: 820, height: 560 },
  shap: { width: 920, height: 640 },
  pd: { width: 820, height: 560 },
  pd2d: { width: 760, height: 680 },
};

let plotlyPromise = null;

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function uniqueStrings(values) {
  return [...new Set((values || []).filter((value) => value !== null && value !== undefined).map(String))];
}

function taskLabel(task) {
  return task === "classification" ? "分類" : task === "regression" ? "回帰" : task || "不明";
}

function targetSummary(summary, target) {
  return summary?.targets?.[target] || {};
}

function pdpFeatures(summary, target, fallbackFeatures) {
  const detail = targetSummary(summary, target);
  return uniqueStrings([
    ...(detail.pdp_features || []),
    ...(detail.features || []),
  ]).filter((feature) => fallbackFeatures.includes(feature));
}

function outputNames(response) {
  return uniqueStrings(response?.metadata?.outputs || []);
}

function selectedOutput(response) {
  const value = response?.metadata?.selected_output;
  return value === null || value === undefined || value === "" ? "" : String(value);
}

async function getPlotly() {
  if (!plotlyPromise) {
    plotlyPromise = import("plotly.js-dist-min").then((module) => module.default || module);
  }
  return plotlyPromise;
}

async function figureToPng(figure, size) {
  if (!figure) throw new Error("Plotly figure is empty.");
  const plotly = await getPlotly();
  const container = document.createElement("div");
  container.style.position = "fixed";
  container.style.left = "-100000px";
  container.style.top = "0";
  container.style.width = `${size.width}px`;
  container.style.height = `${size.height}px`;
  container.style.background = "#ffffff";
  container.setAttribute("aria-hidden", "true");
  document.body.appendChild(container);

  try {
    const layout = {
      ...(figure.layout || {}),
      autosize: false,
      width: size.width,
      height: size.height,
      paper_bgcolor: "#ffffff",
      plot_bgcolor: "#ffffff",
      font: {
        color: "#302929",
        family: 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", "Yu Gothic UI", Meiryo, sans-serif',
        ...(figure.layout?.font || {}),
      },
      margin: {
        l: 72,
        r: 32,
        t: 76,
        b: 68,
        ...(figure.layout?.margin || {}),
      },
    };
    await plotly.newPlot(container, figure.data || [], layout, {
      staticPlot: true,
      displayModeBar: false,
      responsive: false,
    });
    return await plotly.toImage(container, {
      format: "png",
      width: size.width,
      height: size.height,
      scale: 2,
    });
  } finally {
    plotly.purge(container);
    container.remove();
  }
}

async function captureFigure({ title, kind, request, note = "" }) {
  try {
    const response = await request();
    const image = await figureToPng(response?.figure, FIGURE_SIZES[kind]);
    return {
      title,
      note,
      image,
      metadata: response?.metadata || {},
      error: "",
    };
  } catch (error) {
    return {
      title,
      note,
      image: "",
      metadata: {},
      error: error?.message || String(error),
    };
  }
}

async function captureOutputFigures({
  firstResponse,
  baseTitle,
  kind,
  requestForOutput,
  noteForOutput = () => "",
}) {
  const outputs = outputNames(firstResponse);
  const selected = selectedOutput(firstResponse);
  const results = [];

  if (!outputs.length || outputs.length === 1) {
    const image = await figureToPng(firstResponse.figure, FIGURE_SIZES[kind]);
    results.push({
      title: outputs[0] ? `${baseTitle}（${outputs[0]}）` : baseTitle,
      note: noteForOutput(outputs[0] || selected),
      image,
      metadata: firstResponse.metadata || {},
      error: "",
    });
    return results;
  }

  for (const output of outputs) {
    if (output === selected) {
      try {
        const image = await figureToPng(firstResponse.figure, FIGURE_SIZES[kind]);
        results.push({
          title: `${baseTitle}（${output}）`,
          note: noteForOutput(output),
          image,
          metadata: firstResponse.metadata || {},
          error: "",
        });
      } catch (error) {
        results.push({
          title: `${baseTitle}（${output}）`,
          note: noteForOutput(output),
          image: "",
          metadata: firstResponse.metadata || {},
          error: error?.message || String(error),
        });
      }
      continue;
    }
    results.push(await captureFigure({
      title: `${baseTitle}（${output}）`,
      kind,
      note: noteForOutput(output),
      request: () => requestForOutput(output),
    }));
  }
  return results;
}

async function fetchOutputFigures({ baseTitle, kind, initialRequest, requestForOutput, noteForOutput }) {
  try {
    const firstResponse = await initialRequest();
    return await captureOutputFigures({
      firstResponse,
      baseTitle,
      kind,
      requestForOutput,
      noteForOutput,
    });
  } catch (error) {
    return [{
      title: baseTitle,
      note: "",
      image: "",
      metadata: {},
      error: error?.message || String(error),
    }];
  }
}

async function captureYyFigures(modelId, target, task) {
  let cv = true;
  let yyResponse = null;
  let yyError = "";
  try {
    yyResponse = await api.visualizationYy(modelId, target, { cv: true, residual: false });
  } catch (_cvError) {
    cv = false;
    try {
      yyResponse = await api.visualizationYy(modelId, target, { cv: false, residual: false });
    } catch (error) {
      yyError = error?.message || String(error);
    }
  }

  if (!yyResponse) {
    return [{
      title: task === "classification" ? "混同行列" : "実測値–予測値",
      note: "",
      image: "",
      metadata: {},
      error: yyError || "診断図を生成できませんでした。",
    }];
  }

  const figures = [];
  try {
    figures.push({
      title: task === "classification" ? "混同行列" : "実測値–予測値",
      note: cv ? "Train / Validation（交差検証）" : "学習済みモデルによる予測",
      image: await figureToPng(yyResponse.figure, FIGURE_SIZES.yy),
      metadata: yyResponse.metadata || {},
      error: "",
    });
  } catch (error) {
    figures.push({
      title: task === "classification" ? "混同行列" : "実測値–予測値",
      note: "",
      image: "",
      metadata: yyResponse.metadata || {},
      error: error?.message || String(error),
    });
  }

  if (task === "regression") {
    figures.push(await captureFigure({
      title: "実測値–残差",
      kind: "yy",
      note: cv ? "Train / Validation（交差検証）" : "学習済みモデルによる予測",
      request: () => api.visualizationYy(modelId, target, { cv, residual: true }),
    }));
  }
  return figures;
}

function normalizedImportanceItems(response) {
  const items = Array.isArray(response?.items) ? response.items : [];
  const parsed = items
    .map((item) => ({
      feature: String(item?.feature ?? ""),
      value: Number(item?.value),
    }))
    .filter((item) => item.feature && Number.isFinite(item.value));
  const maximum = Math.max(0, ...parsed.map((item) => Math.abs(item.value)));
  return Object.fromEntries(parsed.map((item) => [
    item.feature,
    maximum > 0 ? item.value / maximum : 0,
  ]));
}

function combinedImportanceFigure(responses, target) {
  const valuesByMethod = Object.fromEntries(
    IMPORTANCE_METHODS.map((item) => [
      item.method,
      normalizedImportanceItems(responses[item.method]),
    ]),
  );
  const features = uniqueStrings(
    Object.values(valuesByMethod).flatMap((values) => Object.keys(values)),
  );
  const ranked = features
    .map((feature) => ({
      feature,
      score: IMPORTANCE_METHODS.reduce(
        (sum, item) => sum + Math.abs(valuesByMethod[item.method][feature] || 0),
        0,
      ) / IMPORTANCE_METHODS.length,
    }))
    .sort((left, right) => left.score - right.score)
    .slice(-10);
  const selected = ranked.map((item) => item.feature);

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

async function captureImportanceFigures(modelId, target) {
  const responses = {};
  const failures = [];
  for (const item of IMPORTANCE_METHODS) {
    try {
      responses[item.method] = await api.xaiImportance(modelId, target, {
        method: item.method,
        combined: true,
        top_n: 1000,
      });
    } catch (error) {
      failures.push(`${item.label}: ${error?.message || String(error)}`);
      responses[item.method] = { items: [] };
    }
  }
  const available = Object.values(responses).some((response) => response?.items?.length);
  if (!available) {
    return [{
      title: "特徴量重要度（Model / SHAP / PFI）",
      note: "",
      image: "",
      metadata: {},
      error: failures.join(" / ") || "重要度データがありません。",
    }];
  }
  try {
    const image = await figureToPng(
      combinedImportanceFigure(responses, target),
      FIGURE_SIZES.importance,
    );
    return [{
      title: "特徴量重要度（Model / SHAP / PFI）",
      note: "各手法を最大絶対値で正規化し、平均重要度上位10特徴量を比較します。",
      image,
      metadata: { failures },
      error: "",
    }];
  } catch (error) {
    return [{
      title: "特徴量重要度（Model / SHAP / PFI）",
      note: "",
      image: "",
      metadata: { failures },
      error: error?.message || String(error),
    }];
  }
}

async function captureShapFigures(modelId, target) {
  return fetchOutputFigures({
    baseTitle: "SHAP Beeswarm",
    kind: "shap",
    initialRequest: () => api.visualizationBeeswarm(modelId, target, { top_n: 10 }),
    requestForOutput: (output) => api.visualizationBeeswarm(modelId, target, {
      output,
      top_n: 10,
    }),
    noteForOutput: () => "平均絶対SHAP値の高い順に上位10特徴量を表示します。",
  });
}

function finiteMean(values, fallback = 0) {
  const numeric = values.map(Number).filter(Number.isFinite);
  return numeric.length
    ? numeric.reduce((sum, value) => sum + value, 0) / numeric.length
    : fallback;
}

function pdSeries(response) {
  return (response?.series || []).filter((item) => (
    item && item.name !== null && item.name !== undefined && Array.isArray(item.pd_values)
  ));
}

function shapValueColumns(response) {
  const configured = Array.isArray(response?.value_columns) ? response.value_columns.map(String) : [];
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
    if (!rows.length) return 0;
    return rows.filter((row) => String(row?.[target]) === String(output)).length / rows.length;
  }
  return finiteMean(rows.map((row) => row?.[target]), 0);
}

function legacyPdFigure({
  target,
  task,
  feature,
  rows,
  pdResponse,
  shapResponse,
  series,
  outputIndex,
}) {
  const output = String(series.name ?? target);
  const xValues = pdResponse?.x_values || [];
  const shapColumn = resolveShapColumn(shapResponse, output, outputIndex);
  const shapRecords = Array.isArray(shapResponse?.records) ? shapResponse.records : [];
  const baseline = baselineForOutput(rows, target, task, output, shapResponse);
  const rawRows = rows.filter((row) => row?.[feature] !== null && row?.[feature] !== undefined);

  const data = [
    {
      type: "scatter",
      mode: "lines",
      name: "Partial dependence",
      x: xValues,
      y: series.pd_values,
      line: { color: "#d97706", width: 3 },
      hovertemplate: `${feature}=%{x}<br>PD=%{y:.5g}<extra></extra>`,
    },
  ];

  if (shapColumn) {
    const filtered = shapRecords.filter((record) => (
      record?.[feature] !== null
      && record?.[feature] !== undefined
      && Number.isFinite(Number(record?.[shapColumn]))
    ));
    data.push({
      type: "scatter",
      mode: "markers",
      name: "Baseline + SHAP",
      x: filtered.map((record) => record[feature]),
      y: filtered.map((record) => baseline + Number(record[shapColumn])),
      marker: { color: "#3b82f6", size: 7, opacity: 0.62 },
      hovertemplate: `${feature}=%{x}<br>Base + SHAP=%{y:.5g}<extra></extra>`,
    });
  }

  data.push({
    type: "scatter",
    mode: "markers",
    name: task === "classification" ? "Observed class" : "Observed target",
    x: rawRows.map((row) => row[feature]),
    y: rawRows.map((row) => (
      task === "classification"
        ? Number(String(row[target]) === output)
        : row[target]
    )),
    marker: { color: "#c94f58", size: 7, opacity: 0.48 },
    hovertemplate: `${feature}=%{x}<br>Observed=%{y}<extra></extra>`,
  });

  return {
    data,
    layout: {
      title: task === "classification"
        ? `PD / SHAP / observed: ${target} (${output}) / ${feature}`
        : `PD / SHAP / observed: ${target} / ${feature}`,
      xaxis: { title: feature, automargin: true },
      yaxis: { title: task === "classification" ? `P(${output}) / observed` : target, automargin: true },
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
      annotations: [{
        xref: "paper",
        x: 1,
        y: baseline,
        text: `baseline=${baseline.toPrecision(4)}`,
        showarrow: false,
        xanchor: "right",
        yanchor: "bottom",
        font: { size: 10, color: "#716565" },
      }],
    },
  };
}

async function capturePdFigures(modelId, target, task, features, rows, onProgress) {
  const figures = [];
  for (let index = 0; index < features.length; index += 1) {
    const feature = features[index];
    onProgress?.(`PD図を作成しています: ${target} / ${feature} (${index + 1}/${features.length})`);
    try {
      const [pdResponse, shapResponse] = await Promise.all([
        api.xaiPdp(modelId, target, feature, { include_ice: true, max_ice: 30 }),
        api.xaiShap(modelId, target, feature),
      ]);
      const seriesItems = pdSeries(pdResponse);
      if (!seriesItems.length) throw new Error("PD系列がありません。");
      for (let outputIndex = 0; outputIndex < seriesItems.length; outputIndex += 1) {
        const series = seriesItems[outputIndex];
        const output = String(series.name ?? "");
        try {
          const image = await figureToPng(legacyPdFigure({
            target,
            task,
            feature,
            rows,
            pdResponse,
            shapResponse,
            series,
            outputIndex,
          }), FIGURE_SIZES.pd);
          figures.push({
            title: task === "classification"
              ? `Partial Dependence: ${feature}（${output}）`
              : `Partial Dependence: ${feature}`,
            note: "橙=PD、青=ベース値+SHAP、赤=実測値、破線=ベース値",
            image,
            metadata: { feature, output },
            error: "",
          });
        } catch (error) {
          figures.push({
            title: `Partial Dependence: ${feature}${output ? `（${output}）` : ""}`,
            note: "",
            image: "",
            metadata: { feature, output },
            error: error?.message || String(error),
          });
        }
      }
    } catch (error) {
      figures.push({
        title: `Partial Dependence: ${feature}`,
        note: "",
        image: "",
        metadata: { feature },
        error: error?.message || String(error),
      });
    }
  }
  return figures;
}

function featurePairs(features) {
  const pairs = [];
  for (let right = 1; right < features.length; right += 1) {
    for (let left = 0; left < right; left += 1) {
      pairs.push([features[left], features[right]]);
    }
  }
  return pairs;
}

async function capturePd2dFigures(modelId, target, numericFeatures, onProgress) {
  const pairs = featurePairs(numericFeatures);
  const figures = [];
  for (let index = 0; index < pairs.length; index += 1) {
    const [featureX, featureY] = pairs[index];
    onProgress?.(`2D PD図を作成しています: ${target} / ${featureX} × ${featureY} (${index + 1}/${pairs.length})`);
    const pairFigures = await fetchOutputFigures({
      baseTitle: `2D Partial Dependence: ${featureX} × ${featureY}`,
      kind: "pd2d",
      initialRequest: () => api.visualizationPdp2d(modelId, target, {
        feature_x: featureX,
        feature_y: featureY,
      }),
      requestForOutput: (output) => api.visualizationPdp2d(modelId, target, {
        feature_x: featureX,
        feature_y: featureY,
        output,
      }),
      noteForOutput: () => "PDヒートマップと学習データ位置を表示します。",
    });
    figures.push(...pairFigures);
  }
  return figures;
}

export async function collectReportVisualizations({
  modelId,
  targets,
  tasks,
  features,
  numericFeatures,
  rows,
  onProgress,
}) {
  if (!modelId) return { targets: {}, errors: ["モデルが登録されていません。"] };

  let summary = null;
  const errors = [];
  try {
    onProgress?.("XAI情報を確認しています...");
    summary = await api.xaiSummary(modelId);
  } catch (error) {
    errors.push(`XAI概要: ${error?.message || String(error)}`);
  }

  const result = { targets: {}, errors };
  for (let targetIndex = 0; targetIndex < targets.length; targetIndex += 1) {
    const target = targets[targetIndex];
    const task = tasks[target] || "regression";
    onProgress?.(`診断図を作成しています: ${target} (${targetIndex + 1}/${targets.length})`);

    const availablePdFeatures = pdpFeatures(summary, target, features);
    const numericPdFeatures = availablePdFeatures.filter((feature) => numericFeatures.includes(feature));
    const xaiReady = targetSummary(summary, target).status === "ready";

    const targetResult = {
      task,
      xaiReady,
      yy: await captureYyFigures(modelId, target, task),
      importance: [],
      shap: [],
      pd: [],
      pd2d: [],
    };

    if (xaiReady) {
      onProgress?.(`重要度を作成しています: ${target}`);
      targetResult.importance = await captureImportanceFigures(modelId, target);
      onProgress?.(`SHAPを作成しています: ${target}`);
      targetResult.shap = await captureShapFigures(modelId, target);
      targetResult.pd = await capturePdFigures(
        modelId,
        target,
        task,
        availablePdFeatures,
        rows,
        onProgress,
      );
    } else {
      const reason = "XAIが未計算のため表示できません。Explain画面の「XAIを再計算」を実行してください。";
      targetResult.importance = IMPORTANCE_METHODS.map((item) => ({
        title: item.label,
        note: "",
        image: "",
        metadata: {},
        error: reason,
      }));
      targetResult.shap = [{ title: "SHAP Beeswarm", note: "", image: "", metadata: {}, error: reason }];
      targetResult.pd = [{ title: "Partial Dependence", note: "", image: "", metadata: {}, error: reason }];
    }

    if (numericPdFeatures.length >= 2) {
      targetResult.pd2d = await capturePd2dFigures(
        modelId,
        target,
        numericPdFeatures,
        onProgress,
      );
    }
    result.targets[target] = targetResult;
  }
  return result;
}

function renderFigureCard(figure) {
  if (figure.error) {
    return `<article class="export-figure-card export-figure-error"><h5>${escapeHtml(figure.title)}</h5><p>${escapeHtml(figure.error)}</p></article>`;
  }
  return `
    <article class="export-figure-card">
      <header><h5>${escapeHtml(figure.title)}</h5>${figure.note ? `<p>${escapeHtml(figure.note)}</p>` : ""}</header>
      <a href="${figure.image}" target="_blank" rel="noreferrer" title="クリックして原寸表示">
        <img src="${figure.image}" alt="${escapeHtml(figure.title)}">
      </a>
    </article>`;
}

function renderFigureGroup(title, description, figures, collapsible = false) {
  if (!figures?.length) return "";
  const body = `<div class="export-figure-grid">${figures.map(renderFigureCard).join("")}</div>`;
  if (collapsible) {
    return `<details class="export-figure-details" open><summary><strong>${escapeHtml(title)}</strong><span>${figures.length} figures</span></summary>${description ? `<p class="export-group-description">${escapeHtml(description)}</p>` : ""}${body}</details>`;
  }
  return `<section class="export-figure-group"><div class="export-group-heading"><h4>${escapeHtml(title)}</h4>${description ? `<p>${escapeHtml(description)}</p>` : ""}</div>${body}</section>`;
}

function renderVisualizationSection(visualizations) {
  const entries = Object.entries(visualizations?.targets || {});
  if (!entries.length) {
    return '<p class="empty">学習済みモデルがないため、モデル可視化は収録されていません。</p>';
  }
  const targets = entries.map(([target, result]) => `
    <article class="subcard export-target-card">
      <div class="card-heading"><div><span>TARGET VISUALIZATION</span><h3>${escapeHtml(target)}</h3><p>${escapeHtml(taskLabel(result.task))}</p></div><span class="badge ${result.xaiReady ? "success" : ""}">${result.xaiReady ? "XAI ready" : "XAI unavailable"}</span></div>
      ${renderFigureGroup("Y–Y・残差", "回帰は実測値–予測値と実測値–残差、分類は混同行列を表示します。", result.yy)}
      ${renderFigureGroup("特徴量重要度", "モデル固有重要度、SHAP重要度、Permutation Importanceを正規化して同一図で比較します。", result.importance)}
      ${renderFigureGroup("SHAP", "特徴量値の大小と予測への寄与方向をBeeswarmで表示します。", result.shap)}
      ${renderFigureGroup("Partial Dependence", "各説明変数についてPD、ベース値+SHAP、実測値、ベースラインを表示します。", result.pd, true)}
      ${renderFigureGroup("2D Partial Dependence", "数値説明変数の全組合せについてPDヒートマップを表示します。", result.pd2d, true)}
    </article>`).join("");
  const globalErrors = (visualizations.errors || []).map((error) => `<li>${escapeHtml(error)}</li>`).join("");
  return `${globalErrors ? `<div class="export-global-errors"><strong>可視化時の注意</strong><ul>${globalErrors}</ul></div>` : ""}${targets}`;
}

const VISUALIZATION_CSS = `
    :root{--bg:#fcfbfb;--surface:#fff;--surface-2:#faf6f6;--line:rgba(53,43,43,.10);--text:#302929;--muted:#716565;--primary:#b94f57;--primary-soft:#fbecee;--success:#9f4b51;--success-soft:#f8e9ea}
    .sidebar{background:#201a1a;color:#f7eded}.brand-mark{background:linear-gradient(145deg,#6f3035,#c8666e)}.sidebar a:hover{background:#3c3232}.cover{background:linear-gradient(135deg,#302828,#b94f57)}.cover-kicker{color:#f5d9dc}.section-heading>span{background:var(--primary-soft);color:var(--primary)}
    .export-target-card{padding:20px}.export-target-card>.card-heading p{margin:3px 0 0;color:var(--muted);font-size:11px}.export-figure-group{margin-top:22px}.export-group-heading{margin-bottom:10px}.export-group-heading h4{margin:0;font-size:16px}.export-group-heading p,.export-group-description{margin:3px 0 10px;color:var(--muted);font-size:11px}.export-figure-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.export-figure-card{min-width:0;overflow:hidden;border:1px solid var(--line);border-radius:14px;background:#fff}.export-figure-card header{padding:11px 13px;border-bottom:1px solid var(--line);background:#faf6f6}.export-figure-card h5{margin:0;font-size:13px}.export-figure-card header p{margin:3px 0 0;color:var(--muted);font-size:9px}.export-figure-card a{display:block;background:#fff}.export-figure-card img{display:block;width:100%;height:auto}.export-figure-error{padding:14px;background:#fff7f7;border-style:dashed}.export-figure-error h5{margin:0 0 5px;color:#b53e47}.export-figure-error p{margin:0;color:var(--muted);font-size:10px}.export-figure-details{margin-top:20px}.export-figure-details summary{display:flex;justify-content:space-between;gap:12px;background:#faf6f6}.export-figure-details>p,.export-figure-details>.export-figure-grid{margin:12px}.export-global-errors{padding:12px;border:1px solid #e8c5c8;border-radius:11px;background:#fff5f5;color:#7f373d}.export-global-errors ul{margin:6px 0 0;padding-left:20px}
    @media(max-width:850px){.export-figure-grid{grid-template-columns:1fr}}
    @media print{.export-figure-grid{grid-template-columns:1fr 1fr}.export-figure-card{break-inside:avoid}.export-figure-details{break-inside:auto}.export-figure-details>summary{display:none}.export-figure-details:not([open])>*:not(summary){display:block}}
`;

function injectVisualizations(baseHtml, visualizations) {
  const navLink = '<a href="#model-figures">モデル可視化</a>';
  const sectionHtml = `
    <section class="report-section" id="model-figures">
      <header class="section-heading"><span>06</span><div><h2>モデル可視化</h2><p>旧Excel exportで出力していたY–Y、重要度、PD、SHAPをHTMLへ収録</p></div></header>
      ${renderVisualizationSection(visualizations)}
    </section>`;

  return baseHtml
    .replace("</style>", `${VISUALIZATION_CSS}\n  </style>`)
    .replace('<a href="#optimization">予測・逆解析</a>', `${navLink}<a href="#optimization">予測・逆解析</a>`)
    .replace('<section class="report-section" id="optimization">', `${sectionHtml}<section class="report-section" id="optimization">`);
}

function triggerDownload(html, fileName) {
  const blob = new Blob(["\uFEFF", html], { type: "text/html;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = fileName;
  anchor.style.display = "none";
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
}

export async function downloadDetailedHtmlReport(snapshot, {
  modelId,
  targets,
  tasks,
  features,
  numericFeatures,
  rows,
  onProgress,
} = {}) {
  const visualizations = modelId
    ? await collectReportVisualizations({
        modelId,
        targets,
        tasks,
        features,
        numericFeatures,
        rows,
        onProgress,
      })
    : { targets: {}, errors: [] };
  onProgress?.("HTMLレポートを組み立てています...");
  const html = injectVisualizations(buildHtmlReport(snapshot), visualizations);
  const fileName = reportFileName(snapshot);
  triggerDownload(html, fileName);
  return {
    fileName,
    visualizations,
  };
}
