import { api } from "./api";
import { buildHtmlReport, reportFileName } from "./report";
import { collectReportVisualizations } from "./report-visualizations";
import { collectInteractiveFigures } from "./report-interactive-figures";
import {
  buildDeterministicReportAnalysis,
  LOCAL_ANALYSIS_CSS,
  renderDeterministicAnalysisSection,
} from "./report-local-analysis";
import { embeddedScriptTag } from "./report-script-embedding";
import { removeReportNavItem, removeReportSection } from "./report-html-sections";
import {
  REPORT_TARGET_TABS_CSS,
  reportTargetTabsRuntimeScript,
} from "./report-target-tabs";
import {
  INTERACTIVE_REPORT_CSS,
  interactiveModalHtml,
  interactiveRuntimeScript,
  loadPlotlySource,
  safeInlineScript,
  safeScriptJson,
} from "./report-interactive-runtime";

const GROUP_KEYS = ["yy", "importance", "shap", "pd"];

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function taskLabel(task) {
  if (task === "classification") return "分類";
  if (task === "regression") return "回帰";
  return task || "不明";
}

function attachInteractiveRegistry(visualizations, interactiveFigures) {
  const registry = {};
  let sequence = 0;
  Object.entries(visualizations?.targets || {}).forEach(([target, staticTarget]) => {
    const interactiveTarget = interactiveFigures?.targets?.[target] || {};
    GROUP_KEYS.forEach((groupKey) => {
      const staticItems = staticTarget?.[groupKey] || [];
      const figures = interactiveTarget?.[groupKey] || [];
      staticItems.forEach((item, index) => {
        const figure = figures[index];
        if (!figure || item.error) return;
        const id = `malchan-report-figure-${sequence}`;
        sequence += 1;
        item.interactiveId = id;
        registry[id] = {
          title: item.title,
          image: item.image,
          figure,
        };
      });
    });
  });
  return registry;
}

function renderFigureCard(figure) {
  if (figure.error) {
    return `<article class="export-figure-card export-figure-error"><h5>${escapeHtml(figure.title)}</h5><p>${escapeHtml(figure.error)}</p></article>`;
  }
  const interactive = Boolean(figure.interactiveId);
  const body = interactive
    ? `<button type="button" class="export-figure-open" data-open-report-figure="${escapeHtml(figure.interactiveId)}" aria-label="${escapeHtml(figure.title)}を拡大して編集"><img src="${figure.image}" alt="${escapeHtml(figure.title)}"><span>クリックして拡大・編集</span></button>`
    : `<img class="export-figure-static" src="${figure.image}" alt="${escapeHtml(figure.title)}">`;
  return `
    <article class="export-figure-card">
      <header>
        <div><h5>${escapeHtml(figure.title)}</h5>${figure.note ? `<p>${escapeHtml(figure.note)}</p>` : ""}</div>
        ${interactive ? `<button type="button" class="export-edit-button" data-open-report-figure="${escapeHtml(figure.interactiveId)}">拡大・編集</button>` : ""}
      </header>
      ${body}
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
    </article>`).join("");
  const globalErrors = (visualizations.errors || [])
    .map((error) => `<li>${escapeHtml(error)}</li>`)
    .join("");
  return `${globalErrors ? `<div class="export-global-errors"><strong>可視化時の注意</strong><ul>${globalErrors}</ul></div>` : ""}${targets}`;
}

const STATIC_VISUALIZATION_CSS = `
    :root{--bg:#fcfbfb;--surface:#fff;--surface-2:#faf6f6;--line:rgba(53,43,43,.10);--text:#302929;--muted:#716565;--primary:#b94f57;--primary-soft:#fbecee;--success:#9f4b51;--success-soft:#f8e9ea}
    .sidebar{background:#201a1a;color:#f7eded}.brand-mark{background:linear-gradient(145deg,#6f3035,#c8666e)}.sidebar a:hover{background:#3c3232}.cover{background:linear-gradient(135deg,#302828,#b94f57)}.cover-kicker{color:#f5d9dc}.section-heading>span{background:var(--primary-soft);color:var(--primary)}
    .export-target-card{padding:20px}.export-target-card>.card-heading p{margin:3px 0 0;color:var(--muted);font-size:11px}.export-figure-group{margin-top:22px}.export-group-heading{margin-bottom:10px}.export-group-heading h4{margin:0;font-size:16px}.export-group-heading p,.export-group-description{margin:3px 0 10px;color:var(--muted);font-size:11px}.export-figure-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.export-figure-card{min-width:0;overflow:hidden;border:1px solid var(--line);border-radius:14px;background:#fff}.export-figure-card header{padding:11px 13px;border-bottom:1px solid var(--line);background:#faf6f6}.export-figure-card h5{margin:0;font-size:13px}.export-figure-card header p{margin:3px 0 0;color:var(--muted);font-size:9px}.export-figure-error{padding:14px;background:#fff7f7;border-style:dashed}.export-figure-error h5{margin:0 0 5px;color:#b53e47}.export-figure-error p{margin:0;color:var(--muted);font-size:10px}.export-figure-details{margin-top:20px}.export-figure-details summary{display:flex;justify-content:space-between;gap:12px;background:#faf6f6}.export-figure-details>p,.export-figure-details>.export-figure-grid{margin:12px}.export-global-errors{padding:12px;border:1px solid #e8c5c8;border-radius:11px;background:#fff5f5;color:#7f373d}.export-global-errors ul{margin:6px 0 0;padding-left:20px}
    @media(max-width:850px){.export-figure-grid{grid-template-columns:1fr}}
    @media print{.export-figure-grid{grid-template-columns:1fr 1fr}.export-figure-card{break-inside:avoid}.export-figure-details{break-inside:auto}.export-figure-details>summary{display:none}.export-figure-details:not([open])>*:not(summary){display:block}}
`;

export function injectReportContent(baseHtml, visualizations, registry, plotlySource, localAnalysis = null) {
  const hasModelVisualizations = Object.keys(visualizations?.targets || {}).length > 0;
  let reportHtml = baseHtml;
  if (hasModelVisualizations) {
    reportHtml = removeReportSection(reportHtml, "diagnostics");
    reportHtml = removeReportNavItem(reportHtml, "diagnostics");
  }

  const modelSectionHtml = hasModelVisualizations ? `
    <section class="report-section" id="model-figures">
      <header class="section-heading"><span>05</span><div><h2>モデル可視化</h2><p>精度診断とモデル挙動を重複なくまとめ、図はクリックして拡大・編集できます。</p></div></header>
      ${renderVisualizationSection(visualizations)}
    </section>` : "";
  const analysisSectionHtml = localAnalysis
    ? renderDeterministicAnalysisSection(localAnalysis)
    : "";
  const runtime = `
    ${interactiveModalHtml()}
    <script type="application/json" id="malchan-interactive-figures">${safeScriptJson(registry)}</script>
    ${embeddedScriptTag(plotlySource)}
    <script>${safeInlineScript(interactiveRuntimeScript())}</script>
    <script>${safeInlineScript(reportTargetTabsRuntimeScript())}</script>`;
  const navPrefix = `${hasModelVisualizations ? '<a href="#model-figures">モデル可視化</a>' : ""}${localAnalysis ? '<a href="#local-analysis">自動分析所見</a>' : ""}`;
  return reportHtml
    .replace("</style>", `${STATIC_VISUALIZATION_CSS}${LOCAL_ANALYSIS_CSS}${REPORT_TARGET_TABS_CSS}${INTERACTIVE_REPORT_CSS}\n  </style>`)
    .replace('<a href="#optimization">予測・逆解析</a>', `${navPrefix}<a href="#optimization">予測・逆解析</a>`)
    .replace('<section class="report-section" id="optimization">', `${modelSectionHtml}${analysisSectionHtml}<section class="report-section" id="optimization">`)
    .replace("</body>", `${runtime}\n</body>`);
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

export async function downloadInteractiveHtmlReport(snapshot, {
  modelId,
  targets,
  tasks,
  features,
  rows,
  onProgress,
} = {}) {
  let localAnalysis = null;
  try {
    localAnalysis = await buildDeterministicReportAnalysis({
      apiClient: api,
      reportProblem: snapshot?.problem || "",
      fileName: snapshot?.data?.fileName || "",
      rows,
      features,
      targets,
      tasks,
      missing: snapshot?.data?.missingCount || 0,
      modelInfo: snapshot?.model || null,
      comparison: snapshot?.comparison || null,
      inverseResult: snapshot?.optimization?.result || null,
      objectives: snapshot?.optimization?.objectives || {},
      bounds: snapshot?.optimization?.bounds || {},
      numericColumns: snapshot?.data?.numericColumns || [],
      categoricalColumns: snapshot?.data?.categoricalColumns || [],
      onProgress,
    });
  } catch (error) {
    onProgress?.(`自動分析所見を生成できませんでした。従来のレポート生成を続行します: ${error?.message || String(error)}`);
  }

  const visualizations = modelId
    ? await collectReportVisualizations({
        modelId,
        targets,
        tasks,
        features,
        // 2D PD is intentionally excluded from reports because its all-pairs
        // evaluation dominates report-generation time for wide datasets.
        numericFeatures: [],
        rows,
        onProgress,
      })
    : { targets: {}, errors: [] };

  const interactiveFigures = modelId
    ? await collectInteractiveFigures({ modelId, visualizations, rows, onProgress })
    : { targets: {} };
  const registry = attachInteractiveRegistry(visualizations, interactiveFigures);
  onProgress?.("拡大・編集機能と自動分析所見をHTMLへ組み込んでいます...");
  const plotlySource = Object.keys(registry).length ? await loadPlotlySource() : "";
  const html = injectReportContent(
    buildHtmlReport(snapshot),
    visualizations,
    registry,
    plotlySource,
    localAnalysis,
  );
  const fileName = reportFileName(snapshot);
  triggerDownload(html, fileName);
  return {
    fileName,
    visualizations,
    localAnalysis,
    interactiveFigureCount: Object.keys(registry).length,
    interactiveRuntimeEmbedded: Boolean(plotlySource),
  };
}
