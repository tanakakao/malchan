const REPORT_SCHEMA_VERSION = "1.0";
const MAX_DIAGNOSTIC_ROWS = 500;
const MAX_DIAGNOSTIC_TABLE_ROWS = 60;
const MAX_SCATTER_POINTS = 350;

const MODEL_INFO_KEYS = [
  "model_id",
  "target_cols",
  "tasks",
  "model_names",
  "model_names_by_target",
  "best_model_names",
  "num_cols",
  "cat_cols",
  "smiles_cols",
  "comp_cols",
  "ensemble",
  "ens_type",
  "base_model",
  "tuning",
  "impute",
  "num_impute_type",
  "num_scale_type",
  "cat_impute",
  "poly",
  "poly_degree",
  "poly_interaction_only",
  "decomposition",
  "decomposition_method",
  "dec_n_components",
  "sampling_method",
  "xai_status",
  "created_at",
  "version",
];

function present(value) {
  return value !== null && value !== undefined && value !== "";
}

function compactModelInfo(modelInfo) {
  if (!modelInfo || typeof modelInfo !== "object") return null;
  const compact = {};
  MODEL_INFO_KEYS.forEach((key) => {
    if (present(modelInfo[key])) compact[key] = modelInfo[key];
  });
  return compact;
}

function compactRows(rows, limit) {
  if (!Array.isArray(rows)) return [];
  return rows.slice(0, limit);
}

export function createReportSnapshot({
  reportProblem = "",
  reportText = "",
  fileName = "",
  rows = [],
  columns = [],
  numeric = [],
  categorical = [],
  targets = [],
  tasks = {},
  features = [],
  stats = [],
  missing = 0,
  modelInfo = null,
  comparison = null,
  diagnostics = [],
  prediction = null,
  inverseResult = null,
  objectives = {},
  bounds = {},
  sampler = "",
  inverseTrials = 0,
  topK = 0,
} = {}) {
  return {
    schemaVersion: REPORT_SCHEMA_VERSION,
    generatedAt: new Date().toISOString(),
    problem: reportProblem,
    reportText,
    data: {
      fileName,
      rowCount: Array.isArray(rows) ? rows.length : 0,
      columnCount: Array.isArray(columns) ? columns.length : 0,
      missingCount: Number(missing || 0),
      columns: [...columns],
      numericColumns: [...numeric],
      categoricalColumns: [...categorical],
      features: [...features],
      targets: targets.map((target) => ({
        name: target,
        task: tasks[target] || "unknown",
      })),
      statistics: Array.isArray(stats) ? stats : [],
      rawRowsEmbedded: false,
    },
    model: compactModelInfo(modelInfo),
    comparison,
    diagnostics: {
      totalRows: Array.isArray(diagnostics) ? diagnostics.length : 0,
      rows: compactRows(diagnostics, MAX_DIAGNOSTIC_ROWS),
    },
    prediction,
    optimization: {
      objectives,
      bounds,
      sampler,
      trials: Number(inverseTrials || 0),
      topK: Number(topK || 0),
      result: inverseResult,
    },
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

function safeJsonText(value) {
  return JSON.stringify(value, null, 2)
    .replaceAll("&", "\\u0026")
    .replaceAll("<", "\\u003c")
    .replaceAll(">", "\\u003e");
}

function formatNumber(value, digits = 4) {
  if (typeof value !== "number" || !Number.isFinite(value)) return escapeHtml(value ?? "—");
  if (Math.abs(value) >= 100000 || (Math.abs(value) > 0 && Math.abs(value) < 0.0001)) {
    return value.toExponential(4);
  }
  return new Intl.NumberFormat("ja-JP", { maximumFractionDigits: digits }).format(value);
}

function formatValue(value) {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "number") return formatNumber(value);
  if (typeof value === "boolean") return value ? "有効" : "無効";
  if (Array.isArray(value)) return escapeHtml(value.join(", "));
  if (typeof value === "object") return `<code>${escapeHtml(JSON.stringify(value))}</code>`;
  return escapeHtml(value);
}

function formatDate(isoString) {
  const date = new Date(isoString);
  if (Number.isNaN(date.getTime())) return escapeHtml(isoString);
  return new Intl.DateTimeFormat("ja-JP", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(date);
}

function section(id, number, title, description, body) {
  return `
    <section class="report-section" id="${escapeHtml(id)}">
      <header class="section-heading">
        <span>${escapeHtml(number)}</span>
        <div>
          <h2>${escapeHtml(title)}</h2>
          ${description ? `<p>${escapeHtml(description)}</p>` : ""}
        </div>
      </header>
      ${body}
    </section>`;
}

function metricCard(label, value, note = "") {
  return `<div class="metric-card"><span>${escapeHtml(label)}</span><strong>${formatValue(value)}</strong>${note ? `<small>${escapeHtml(note)}</small>` : ""}</div>`;
}

function targetLabel(task) {
  if (task === "regression") return "回帰";
  if (task === "classification") return "分類";
  return task || "不明";
}

function renderDataOverview(snapshot) {
  const data = snapshot.data;
  const targetText = data.targets.length
    ? data.targets.map((target) => `${target.name}（${targetLabel(target.task)}）`).join(" / ")
    : "未設定";
  const statistics = data.statistics || [];
  const rows = statistics.map((item) => `
    <tr>
      <td><strong>${escapeHtml(item.column)}</strong></td>
      <td>${formatNumber(item.count)}</td>
      <td>${formatNumber(item.missing)}</td>
      <td>${formatNumber(item.unique)}</td>
      <td>${formatNumber(item.min)}</td>
      <td>${formatNumber(item.max)}</td>
      <td>${formatNumber(item.mean)}</td>
    </tr>`).join("");

  return `
    <div class="metric-grid">
      ${metricCard("入力ファイル", data.fileName || "未指定")}
      ${metricCard("データサイズ", `${data.rowCount} 行 × ${data.columnCount} 列`)}
      ${metricCard("欠損セル", data.missingCount)}
      ${metricCard("目的変数", targetText)}
    </div>
    <div class="two-column">
      <article class="subcard">
        <h3>変数構成</h3>
        <dl class="definition-list">
          <div><dt>説明変数</dt><dd>${data.features.length ? escapeHtml(data.features.join(", ")) : "未設定"}</dd></div>
          <div><dt>数値列</dt><dd>${data.numericColumns.length ? escapeHtml(data.numericColumns.join(", ")) : "なし"}</dd></div>
          <div><dt>カテゴリ列</dt><dd>${data.categoricalColumns.length ? escapeHtml(data.categoricalColumns.join(", ")) : "なし"}</dd></div>
        </dl>
      </article>
      <article class="subcard note-card">
        <h3>共有時の注意</h3>
        <p>このHTMLには入力データの全行を埋め込まず、統計量、モデル結果、診断結果の要約のみを収録しています。</p>
      </article>
    </div>
    <article class="subcard">
      <h3>列ごとの基本統計</h3>
      ${rows ? `<div class="table-scroll"><table><thead><tr><th>列</th><th>有効数</th><th>欠損</th><th>ユニーク</th><th>最小</th><th>最大</th><th>平均</th></tr></thead><tbody>${rows}</tbody></table></div>` : '<p class="empty">統計情報はありません。</p>'}
    </article>`;
}

function renderProblem(snapshot) {
  const text = snapshot.problem?.trim();
  return `<article class="subcard prose"><p>${text ? escapeHtml(text).replaceAll("\n", "<br>") : "課題は入力されていません。"}</p></article>`;
}

function renderModel(snapshot) {
  if (!snapshot.model) return '<p class="empty">モデルはまだ学習または読み込みされていません。</p>';
  const model = snapshot.model;
  const rows = Object.entries(model)
    .filter(([key]) => key !== "model_id")
    .map(([key, value]) => `<tr><th>${escapeHtml(key)}</th><td>${formatValue(value)}</td></tr>`)
    .join("");
  return `
    <div class="metric-grid model-metrics">
      ${metricCard("モデルID", model.model_id || "—")}
      ${metricCard("XAI状態", model.xai_status || "不明")}
      ${metricCard("アンサンブル", model.ensemble ? "有効" : "無効")}
      ${metricCard("チューニング", model.tuning ? "有効" : "無効")}
    </div>
    <article class="subcard">
      <h3>モデル・前処理設定</h3>
      <div class="table-scroll"><table class="key-value-table"><tbody>${rows}</tbody></table></div>
    </article>`;
}

function objectColumns(rows) {
  const columns = [];
  const seen = new Set();
  rows.forEach((row) => {
    if (!row || typeof row !== "object" || Array.isArray(row)) return;
    Object.keys(row).forEach((key) => {
      if (!seen.has(key)) {
        seen.add(key);
        columns.push(key);
      }
    });
  });
  return columns;
}

function renderObjectTable(rows, maxRows = 100) {
  if (!Array.isArray(rows) || !rows.length) return '<p class="empty">表示できる表データはありません。</p>';
  const displayed = rows.slice(0, maxRows);
  const columns = objectColumns(displayed);
  if (!columns.length) return '<p class="empty">表示できる表データはありません。</p>';
  const head = columns.map((column) => `<th>${escapeHtml(column)}</th>`).join("");
  const body = displayed.map((row) => `<tr>${columns.map((column) => `<td>${formatValue(row?.[column])}</td>`).join("")}</tr>`).join("");
  const omitted = rows.length - displayed.length;
  return `<div class="table-scroll"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>${omitted > 0 ? `<p class="table-note">残り ${omitted} 行は省略しています。</p>` : ""}`;
}

function renderCvScores(result) {
  const train = result?.best_cv_scores?.train?.[0] || {};
  const validation = result?.best_cv_scores?.test?.[0] || {};
  const metrics = [...new Set([...Object.keys(train), ...Object.keys(validation)])];
  if (!metrics.length) return "";
  return `
    <h4>ベストモデルの交差検証</h4>
    <div class="table-scroll compact-table"><table>
      <thead><tr><th>指標</th><th>Train</th><th>Validation</th></tr></thead>
      <tbody>${metrics.map((metric) => `<tr><td>${escapeHtml(metric)}</td><td>${formatNumber(train[metric])}</td><td>${formatNumber(validation[metric])}</td></tr>`).join("")}</tbody>
    </table></div>`;
}

function renderComparison(snapshot) {
  const targets = snapshot.comparison?.targets;
  if (!targets || !Object.keys(targets).length) {
    return '<p class="empty">モデル比較はまだ実施されていません。</p>';
  }
  return Object.entries(targets).map(([target, result]) => {
    const failures = Object.entries(result.failures || {}).map(([model, reason]) => ({ model, reason }));
    return `
      <article class="subcard comparison-card">
        <div class="card-heading">
          <div><span>TARGET</span><h3>${escapeHtml(target)}</h3></div>
          <span class="badge ${result.best_is_tuned ? "success" : ""}">${result.best_is_tuned ? "Tuned" : "Compared"}</span>
        </div>
        <div class="metric-grid compact-metrics">
          ${metricCard("採用モデル", result.best_model_name || "—")}
          ${metricCard("選定指標", result.metric || "—", result.higher_is_better ? "大きいほど良い" : "小さいほど良い")}
          ${metricCard("候補数", result.ranking?.length || 0)}
          ${metricCard("失敗候補", failures.length)}
        </div>
        ${renderCvScores(result)}
        <h4>候補モデルランキング</h4>
        ${renderObjectTable(result.ranking || [], 100)}
        ${result.best_params ? `<details><summary>採用パラメータ</summary><pre>${escapeHtml(JSON.stringify(result.best_params, null, 2))}</pre></details>` : ""}
        ${failures.length ? `<details><summary>失敗した候補</summary>${renderObjectTable(failures, 100)}</details>` : ""}
      </article>`;
  }).join("");
}

function diagnosticPairs(snapshot, target) {
  return (snapshot.diagnostics?.rows || []).map((item) => ({
    actual: item?.actual?.[target],
    predicted: item?.predicted?.[target],
  })).filter((item) => present(item.actual) && present(item.predicted));
}

function regressionMetrics(pairs) {
  const numeric = pairs.filter((item) => Number.isFinite(Number(item.actual)) && Number.isFinite(Number(item.predicted)))
    .map((item) => ({ actual: Number(item.actual), predicted: Number(item.predicted) }));
  if (!numeric.length) return { rows: [], rmse: null, mae: null, r2: null };
  const mean = numeric.reduce((sum, item) => sum + item.actual, 0) / numeric.length;
  const squaredErrors = numeric.map((item) => (item.predicted - item.actual) ** 2);
  const absoluteErrors = numeric.map((item) => Math.abs(item.predicted - item.actual));
  const residualSum = squaredErrors.reduce((sum, value) => sum + value, 0);
  const totalSum = numeric.reduce((sum, item) => sum + (item.actual - mean) ** 2, 0);
  return {
    rows: numeric,
    rmse: Math.sqrt(residualSum / numeric.length),
    mae: absoluteErrors.reduce((sum, value) => sum + value, 0) / numeric.length,
    r2: totalSum > 0 ? 1 - residualSum / totalSum : null,
  };
}

function evenlySample(rows, limit) {
  if (rows.length <= limit) return rows;
  const step = rows.length / limit;
  return Array.from({ length: limit }, (_, index) => rows[Math.floor(index * step)]);
}

function scatterSvg(rows, target) {
  if (!rows.length) return '<p class="empty">数値の診断データがありません。</p>';
  const points = evenlySample(rows, MAX_SCATTER_POINTS);
  const values = points.flatMap((item) => [item.actual, item.predicted]);
  let min = Math.min(...values);
  let max = Math.max(...values);
  if (min === max) {
    min -= 0.5;
    max += 0.5;
  }
  const padding = (max - min) * 0.06;
  min -= padding;
  max += padding;
  const width = 720;
  const height = 360;
  const margin = { left: 64, right: 24, top: 22, bottom: 55 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;
  const x = (value) => margin.left + ((value - min) / (max - min)) * innerWidth;
  const y = (value) => margin.top + innerHeight - ((value - min) / (max - min)) * innerHeight;
  const ticks = Array.from({ length: 6 }, (_, index) => min + ((max - min) * index) / 5);
  return `
    <div class="chart-frame">
      <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(target)}の実測値と予測値">
        <rect x="0" y="0" width="${width}" height="${height}" rx="16" fill="#ffffff" />
        ${ticks.map((tick) => `<line x1="${x(tick)}" y1="${margin.top}" x2="${x(tick)}" y2="${margin.top + innerHeight}" stroke="#e5e9f0"/><line x1="${margin.left}" y1="${y(tick)}" x2="${margin.left + innerWidth}" y2="${y(tick)}" stroke="#e5e9f0"/><text x="${x(tick)}" y="${height - 29}" text-anchor="middle" font-size="11" fill="#697386">${escapeHtml(formatNumber(tick, 3))}</text><text x="${margin.left - 10}" y="${y(tick) + 4}" text-anchor="end" font-size="11" fill="#697386">${escapeHtml(formatNumber(tick, 3))}</text>`).join("")}
        <line x1="${x(min)}" y1="${y(min)}" x2="${x(max)}" y2="${y(max)}" stroke="#315efb" stroke-width="2" stroke-dasharray="7 6"/>
        ${points.map((item) => `<circle cx="${x(item.actual)}" cy="${y(item.predicted)}" r="3.5" fill="#315efb" fill-opacity="0.58"/>`).join("")}
        <line x1="${margin.left}" y1="${margin.top + innerHeight}" x2="${margin.left + innerWidth}" y2="${margin.top + innerHeight}" stroke="#94a3b8"/>
        <line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${margin.top + innerHeight}" stroke="#94a3b8"/>
        <text x="${margin.left + innerWidth / 2}" y="${height - 5}" text-anchor="middle" font-size="13" font-weight="700" fill="#172033">実測値</text>
        <text x="17" y="${margin.top + innerHeight / 2}" text-anchor="middle" transform="rotate(-90 17 ${margin.top + innerHeight / 2})" font-size="13" font-weight="700" fill="#172033">予測値</text>
      </svg>
    </div>`;
}

function confusionMatrix(pairs) {
  const labels = [...new Set(pairs.flatMap((item) => [String(item.actual), String(item.predicted)]))];
  const counts = Object.fromEntries(labels.map((actual) => [actual, Object.fromEntries(labels.map((predicted) => [predicted, 0]))]));
  pairs.forEach((item) => {
    counts[String(item.actual)][String(item.predicted)] += 1;
  });
  return { labels, counts };
}

function renderClassificationDiagnostics(pairs) {
  if (!pairs.length) return '<p class="empty">分類の診断データがありません。</p>';
  const { labels, counts } = confusionMatrix(pairs);
  const correct = pairs.filter((item) => String(item.actual) === String(item.predicted)).length;
  const head = labels.map((label) => `<th>${escapeHtml(label)}</th>`).join("");
  const body = labels.map((actual) => `<tr><th>${escapeHtml(actual)}</th>${labels.map((predicted) => `<td class="${actual === predicted ? "matrix-diagonal" : ""}">${counts[actual][predicted]}</td>`).join("")}</tr>`).join("");
  return `
    <div class="metric-grid compact-metrics">
      ${metricCard("診断件数", pairs.length)}
      ${metricCard("正解数", correct)}
      ${metricCard("Accuracy", correct / pairs.length)}
      ${metricCard("クラス数", labels.length)}
    </div>
    <div class="table-scroll"><table class="confusion-table"><thead><tr><th>Actual \\ Predicted</th>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
}

function renderDiagnostics(snapshot) {
  if (!snapshot.diagnostics?.totalRows) {
    return '<p class="empty">診断用予測はまだ計算されていません。Explain画面で診断結果を表示してから出力すると、実測値と予測値を収録できます。</p>';
  }
  return snapshot.data.targets.map(({ name, task }) => {
    const pairs = diagnosticPairs(snapshot, name);
    if (task === "classification") {
      return `<article class="subcard"><div class="card-heading"><div><span>CLASSIFICATION</span><h3>${escapeHtml(name)}</h3></div><span class="badge">${pairs.length} rows</span></div>${renderClassificationDiagnostics(pairs)}</article>`;
    }
    const metrics = regressionMetrics(pairs);
    const tableRows = metrics.rows.slice(0, MAX_DIAGNOSTIC_TABLE_ROWS).map((item, index) => ({
      index: index + 1,
      actual: item.actual,
      predicted: item.predicted,
      residual: item.actual - item.predicted,
    }));
    return `
      <article class="subcard">
        <div class="card-heading"><div><span>REGRESSION</span><h3>${escapeHtml(name)}</h3></div><span class="badge">${metrics.rows.length} rows</span></div>
        <div class="metric-grid compact-metrics">
          ${metricCard("RMSE", metrics.rmse)}
          ${metricCard("MAE", metrics.mae)}
          ${metricCard("R²", metrics.r2)}
          ${metricCard("診断件数", metrics.rows.length)}
        </div>
        ${scatterSvg(metrics.rows, name)}
        <details><summary>診断データ</summary>${renderObjectTable(tableRows, MAX_DIAGNOSTIC_TABLE_ROWS)}</details>
      </article>`;
  }).join("");
}

function renderOptimization(snapshot) {
  const optimization = snapshot.optimization || {};
  const result = optimization.result;
  const objectiveRows = Object.entries(optimization.objectives || {}).map(([target, value]) => ({ target, ...value }));
  const boundRows = Object.entries(optimization.bounds || {}).map(([feature, value]) => ({ feature, ...value }));
  const candidates = result?.candidates || [];
  return `
    <div class="metric-grid">
      ${metricCard("探索手法", optimization.sampler || "未設定")}
      ${metricCard("Trials", optimization.trials || 0)}
      ${metricCard("出力候補数", candidates.length)}
      ${metricCard("Paretoサイズ", result?.pareto_size ?? "—")}
    </div>
    <div class="two-column">
      <article class="subcard"><h3>目的条件</h3>${renderObjectTable(objectiveRows, 100)}</article>
      <article class="subcard"><h3>探索範囲</h3>${renderObjectTable(boundRows, 100)}</article>
    </div>
    <article class="subcard">
      <h3>逆解析候補</h3>
      ${candidates.length ? renderObjectTable(candidates, 200) : '<p class="empty">逆解析はまだ実施されていません。</p>'}
    </article>`;
}

function renderPrediction(snapshot) {
  if (!snapshot.prediction) return "";
  const rows = typeof snapshot.prediction === "object" ? [snapshot.prediction] : [{ prediction: snapshot.prediction }];
  return `<article class="subcard"><h3>直近の単一予測</h3>${renderObjectTable(rows, 10)}</article>`;
}

function renderReportText(snapshot) {
  const text = snapshot.reportText?.trim();
  return `
    ${renderPrediction(snapshot)}
    <article class="subcard prose report-text">
      <h3>レポート用テキスト</h3>
      <p>${text ? escapeHtml(text).replaceAll("\n", "<br>") : "レポート用テキストはまだ作成されていません。"}</p>
    </article>`;
}

function reportTitle(snapshot) {
  const base = snapshot.data.fileName?.replace(/\.[^.]+$/, "") || "malchan_analysis";
  return `${base} 分析レポート`;
}

export function buildHtmlReport(snapshot) {
  const title = reportTitle(snapshot);
  const generatedAt = formatDate(snapshot.generatedAt);
  const comparisonCount = Object.keys(snapshot.comparison?.targets || {}).length;
  const inverseCount = snapshot.optimization?.result?.candidates?.length || 0;
  const navItems = [
    ["problem", "分析課題"],
    ["data", "データ概要"],
    ["model", "モデル設定"],
    ["comparison", "モデル比較"],
    ["diagnostics", "精度診断"],
    ["optimization", "予測・逆解析"],
    ["report-text", "レポート用テキスト"],
  ];
  return `<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>${escapeHtml(title)}</title>
  <style>
    :root{--bg:#eef2f7;--surface:#fff;--surface-2:#f7f9fc;--line:#dce3ed;--text:#172033;--muted:#66758a;--primary:#315efb;--primary-soft:#edf2ff;--success:#16835d;--success-soft:#eaf8f2;--warning:#c88213;--shadow:0 12px 30px rgba(25,39,68,.10)}
    *{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--text);font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI","Yu Gothic UI",Meiryo,sans-serif;font-size:14px;line-height:1.6}button{font:inherit}.report-shell{display:grid;grid-template-columns:240px minmax(0,1fr);min-height:100vh}.sidebar{position:sticky;top:0;height:100vh;padding:28px 18px;background:#111827;color:#dbe5f4;overflow:auto}.brand{display:flex;align-items:center;gap:10px;margin-bottom:28px}.brand-mark{width:38px;height:38px;border-radius:12px;display:grid;place-items:center;background:linear-gradient(145deg,#2748be,#5e7aff);font-size:22px;font-weight:900}.brand strong{display:block;letter-spacing:.08em}.brand small{display:block;color:#91a3bf;font-size:10px}.sidebar nav{display:grid;gap:5px}.sidebar a{display:block;padding:9px 10px;border-radius:9px;color:#b9c5d8;text-decoration:none;font-size:12px}.sidebar a:hover{background:#202a3b;color:#fff}.sidebar-note{margin-top:24px;padding:12px;border:1px solid #344055;border-radius:11px;color:#9fb0c8;font-size:10px}.main{min-width:0;padding:38px 42px 70px}.cover{position:relative;overflow:hidden;padding:34px;margin-bottom:22px;border-radius:24px;background:linear-gradient(135deg,#14213d,#315efb);color:#fff;box-shadow:var(--shadow)}.cover:after{content:"";position:absolute;width:300px;height:300px;right:-100px;top:-150px;border-radius:50%;background:rgba(255,255,255,.10)}.cover-kicker{font-size:10px;letter-spacing:.18em;font-weight:900;color:#bcd0ff}.cover h1{max-width:760px;margin:8px 0 10px;font-size:32px;line-height:1.25}.cover p{max-width:780px;margin:0;color:#dce6ff}.cover-meta{display:flex;gap:9px;flex-wrap:wrap;margin-top:22px}.cover-meta span{padding:6px 9px;border:1px solid rgba(255,255,255,.2);border-radius:999px;background:rgba(255,255,255,.08);font-size:10px}.toolbar{display:flex;justify-content:flex-end;margin:0 0 16px}.toolbar button{border:0;border-radius:10px;padding:9px 13px;background:#172033;color:#fff;font-weight:750;cursor:pointer}.report-section{padding:24px;margin-bottom:20px;border:1px solid var(--line);border-radius:18px;background:var(--surface);box-shadow:0 4px 16px rgba(25,39,68,.05)}.section-heading{display:flex;gap:12px;align-items:flex-start;margin-bottom:17px}.section-heading>span{width:31px;height:31px;display:grid;place-items:center;flex:0 0 auto;border-radius:9px;background:var(--primary-soft);color:var(--primary);font-size:11px;font-weight:900}.section-heading h2{margin:0;font-size:20px}.section-heading p{margin:3px 0 0;color:var(--muted);font-size:11px}.metric-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-bottom:14px}.metric-card{min-width:0;padding:14px;border:1px solid var(--line);border-radius:13px;background:var(--surface-2)}.metric-card>span{display:block;color:var(--muted);font-size:9px;letter-spacing:.06em;text-transform:uppercase}.metric-card>strong{display:block;margin-top:4px;font-size:16px;overflow-wrap:anywhere}.metric-card small{display:block;margin-top:2px;color:var(--muted);font-size:9px}.compact-metrics{margin-top:12px}.two-column{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px}.subcard{min-width:0;padding:16px;margin-top:12px;border:1px solid var(--line);border-radius:14px;background:var(--surface)}.subcard h3{margin:0 0 10px;font-size:15px}.subcard h4{margin:16px 0 8px}.note-card{background:#fff9ec;border-color:#f2d49e}.note-card p,.prose p{margin:0;color:#475569}.definition-list{margin:0}.definition-list div{display:grid;grid-template-columns:90px 1fr;gap:12px;padding:8px 0;border-bottom:1px solid var(--line)}.definition-list div:last-child{border-bottom:0}.definition-list dt{color:var(--muted);font-size:10px}.definition-list dd{margin:0;overflow-wrap:anywhere}.table-scroll{overflow:auto;border:1px solid var(--line);border-radius:11px}.compact-table{max-width:620px}table{width:100%;border-collapse:separate;border-spacing:0;font-size:11px}th,td{padding:8px 10px;border-right:1px solid var(--line);border-bottom:1px solid var(--line);white-space:nowrap;text-align:left}tr:last-child td,tr:last-child th{border-bottom:0}th:last-child,td:last-child{border-right:0}thead th{background:#edf2f8;color:#526176}tbody tr:nth-child(even){background:#fafbfc}.key-value-table th{width:210px;background:#f4f7fb;color:#526176}.table-note{margin:7px 0 0;color:var(--muted);font-size:10px}.card-heading{display:flex;align-items:flex-start;justify-content:space-between;gap:12px}.card-heading span:first-child{color:var(--primary);font-size:8px;letter-spacing:.14em;font-weight:900}.card-heading h3{margin:2px 0}.badge{padding:5px 8px;border-radius:999px;background:#edf2f8;color:#526176;font-size:9px;font-weight:800}.badge.success{background:var(--success-soft);color:var(--success)}details{margin-top:12px;border:1px solid var(--line);border-radius:11px;overflow:hidden}summary{padding:10px 12px;background:#f7f9fc;font-weight:750;cursor:pointer}pre{margin:0;padding:13px;overflow:auto;background:#111827;color:#dbe5f4;font-size:10px;line-height:1.55}.chart-frame{overflow:auto;margin-top:10px;border:1px solid var(--line);border-radius:14px;background:#fff}.chart-frame svg{display:block;width:100%;min-width:620px}.confusion-table td{text-align:center}.matrix-diagonal{background:var(--success-soft);color:var(--success);font-weight:800}.empty{padding:18px;border:1px dashed #c8d2df;border-radius:11px;background:#f7f9fc;color:var(--muted);text-align:center}.report-text{background:#f8faff}.footer{padding:12px 4px;color:var(--muted);font-size:10px;text-align:center}code{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:10px;white-space:normal;overflow-wrap:anywhere}
    @media(max-width:1000px){.report-shell{display:block}.sidebar{position:static;height:auto}.sidebar nav{grid-template-columns:repeat(3,1fr)}.sidebar-note{display:none}.main{padding:24px 18px 50px}.metric-grid{grid-template-columns:repeat(2,1fr)}}
    @media(max-width:620px){.main{padding:14px 10px 35px}.cover{padding:24px 20px;border-radius:16px}.cover h1{font-size:25px}.sidebar nav{grid-template-columns:1fr 1fr}.metric-grid,.two-column{grid-template-columns:1fr}.report-section{padding:16px}.definition-list div{grid-template-columns:1fr}.key-value-table th{width:auto}}
    @media print{body{background:#fff}.report-shell{display:block}.sidebar,.toolbar{display:none}.main{padding:0}.cover{box-shadow:none;break-after:avoid}.report-section{box-shadow:none;break-inside:avoid;border-color:#cbd5e1}.subcard{break-inside:avoid}details{break-inside:avoid}details>summary{display:none}details:not([open])>*:not(summary){display:block}.table-scroll{overflow:visible}.footer{margin-top:18px}}
  </style>
</head>
<body>
  <div class="report-shell">
    <aside class="sidebar">
      <div class="brand"><div class="brand-mark">m</div><div><strong>MALCHAN</strong><small>ANALYSIS REPORT</small></div></div>
      <nav>${navItems.map(([id, label]) => `<a href="#${id}">${escapeHtml(label)}</a>`).join("")}</nav>
      <div class="sidebar-note">スキーマ ${escapeHtml(snapshot.schemaVersion)}<br>${escapeHtml(generatedAt)}</div>
    </aside>
    <main class="main">
      <div class="toolbar"><button type="button" onclick="window.print()">印刷 / PDF保存</button></div>
      <header class="cover">
        <span class="cover-kicker">MATERIALS &amp; MANUFACTURING MACHINE LEARNING</span>
        <h1>${escapeHtml(title)}</h1>
        <p>データ概要、モデル比較、精度診断、逆解析結果をmalchanの分析状態から統合したHTMLレポートです。</p>
        <div class="cover-meta"><span>作成: ${escapeHtml(generatedAt)}</span><span>目的変数: ${snapshot.data.targets.length}</span><span>モデル比較: ${comparisonCount}</span><span>逆解析候補: ${inverseCount}</span></div>
      </header>
      ${section("problem", "01", "分析課題", "分析の背景と解決したい課題", renderProblem(snapshot))}
      ${section("data", "02", "データ概要", "入力データの構成と基本統計", renderDataOverview(snapshot))}
      ${section("model", "03", "モデル設定", "学習済みモデルと前処理の設定", renderModel(snapshot))}
      ${section("comparison", "04", "モデル比較", "同一条件で評価した候補モデルと採用根拠", renderComparison(snapshot))}
      ${section("diagnostics", "05", "精度診断", "実測値と予測値に基づく診断", renderDiagnostics(snapshot))}
      ${section("optimization", "06", "予測・逆解析", "目的条件、探索範囲、提案候補", renderOptimization(snapshot))}
      ${section("report-text", "07", "レポート用テキスト", "生成AI向けプロンプトまたは編集済みの分析所見", renderReportText(snapshot))}
      <footer class="footer">Generated by malchan · Raw input rows are not embedded in this report.</footer>
    </main>
  </div>
  <script type="application/json" id="malchan-report-data">${safeJsonText(snapshot)}</script>
</body>
</html>`;
}

function sanitizedFileStem(fileName) {
  const base = (fileName || "malchan_analysis").replace(/\.[^.]+$/, "");
  const sanitized = base
    .normalize("NFKC")
    .replace(/[\\/:*?"<>|]+/g, "_")
    .replace(/\s+/g, "_")
    .replace(/^_+|_+$/g, "");
  return sanitized || "malchan_analysis";
}

export function reportFileName(snapshot) {
  const date = String(snapshot.generatedAt || new Date().toISOString()).slice(0, 10).replaceAll("-", "");
  return `${sanitizedFileStem(snapshot.data?.fileName)}_report_${date}.html`;
}

export function downloadHtmlReport(snapshot) {
  const html = buildHtmlReport(snapshot);
  const fileName = reportFileName(snapshot);
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
  return fileName;
}
