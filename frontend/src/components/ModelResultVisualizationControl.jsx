import React, { useEffect, useMemo, useState } from "react";
import { createPortal } from "react-dom";
import { api } from "../api";
import { useWorkbench } from "../context/WorkbenchContext";

const NODE_KIND_LABELS = {
  pipeline: "処理フロー",
  branch: "列・特徴量分岐",
  ensemble: "アンサンブル",
  transformer: "変換",
  estimator: "予測器",
  passthrough: "そのまま使用",
  dropped: "除外",
  reference: "省略",
};

const NODE_NAME_LABELS = {
  model: "学習パイプライン",
  preprocess: "前処理",
  predictor: "予測モデル",
  column_preprocess: "列別前処理",
  num_cat_prerprocess: "数値・カテゴリ列の前処理",
  num_cat_common: "数値・カテゴリ共通変換",
  common_preprocess: "全特徴量の共通変換",
  num_cat: "数値・カテゴリ特徴量",
  num: "数値列",
  cat: "カテゴリ列",
  imputer: "欠損値補完",
  scaler: "スケーリング",
  identity: "変換なし",
  "one-hot": "One-Hotエンコード",
  ordinal: "順序エンコード",
};

function formatMetric(value) {
  if (!Number.isFinite(value)) return "—";
  const magnitude = Math.abs(value);
  if (magnitude >= 1000 || (magnitude > 0 && magnitude < 0.001)) {
    return value.toExponential(4);
  }
  return value.toFixed(4);
}

function summarizeMetric(records, metric) {
  const values = (records || [])
    .map((record) => Number(record?.[metric]))
    .filter(Number.isFinite);
  if (!values.length) return null;
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
  return { mean, std: Math.sqrt(variance), count: values.length };
}

function formatSummary(summary) {
  if (!summary) return "—";
  const mean = formatMetric(summary.mean);
  return summary.count > 1 ? `${mean} ± ${formatMetric(summary.std)}` : mean;
}

function friendlyNodeName(name) {
  if (NODE_NAME_LABELS[name]) return NODE_NAME_LABELS[name];
  if (/^smiles_\d+$/.test(name)) return `SMILES特徴量 ${Number(name.split("_")[1]) + 1}`;
  if (/^comp_\d+$/.test(name)) return `組成式特徴量 ${Number(name.split("_")[1]) + 1}`;
  if (/^model_\d+$/.test(name)) return `構成モデル ${Number(name.split("_")[1])}`;
  return name;
}

function StructureCard({ node }) {
  const columns = node?.columns || [];
  const parameters = Object.entries(node?.parameters || {});
  const kind = node?.kind || "transformer";

  return (
    <article className={`model-structure-card kind-${kind}`}>
      <div className="model-structure-card-head">
        <span className="model-structure-kind">{NODE_KIND_LABELS[kind] || kind}</span>
        {columns.length > 0 && <span className="model-structure-count">{columns.length}列</span>}
      </div>
      <strong>{friendlyNodeName(node?.name || "step")}</strong>
      <code>{node?.class_name || "Estimator"}</code>

      {columns.length > 0 && (
        <div className="model-structure-columns" aria-label="対象列">
          {columns.map((column, index) => (
            <span key={`${column}-${index}`} title={column}>{column}</span>
          ))}
        </div>
      )}

      {parameters.length > 0 && (
        <details className="model-structure-parameters">
          <summary>主要設定</summary>
          <dl>
            {parameters.map(([name, value]) => (
              <div key={name}>
                <dt>{name}</dt>
                <dd title={value}>{value}</dd>
              </div>
            ))}
          </dl>
        </details>
      )}
    </article>
  );
}

function StructureNode({ node, path = "root" }) {
  if (!node) return null;
  const children = node.children || [];
  const layout = node.kind === "pipeline"
    ? "sequence"
    : ["branch", "ensemble"].includes(node.kind)
      ? "branches"
      : "nested";

  return (
    <div className={`model-structure-node node-${node.kind || "transformer"}`}>
      <StructureCard node={node} />
      {children.length > 0 && (
        <div className={`model-structure-children layout-${layout}`}>
          {children.map((child, index) => (
            <div className="model-structure-child" key={`${path}-${child.name}-${index}`}>
              <StructureNode node={child} path={`${path}-${index}`} />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function ModelStructureDiagram({ structure }) {
  if (!structure) {
    return <p className="settings-note">モデル構造を取得できませんでした。</p>;
  }
  return (
    <div className="model-structure-canvas">
      <div className="model-structure-entry"><span>入力データ</span><i aria-hidden="true">↓</i></div>
      <StructureNode node={structure} />
      <div className="model-structure-output"><i aria-hidden="true">↓</i><span>予測結果</span></div>
    </div>
  );
}

function ModelTargetTabs({ targets, activeTarget, onChange }) {
  if (targets.length <= 1) return null;
  return (
    <div className="model-result-target-tabs" role="tablist" aria-label="登録モデルの目的変数">
      {targets.map((item) => (
        <button
          key={item.target}
          type="button"
          role="tab"
          aria-selected={item.target === activeTarget}
          className={item.target === activeTarget ? "active" : ""}
          onClick={() => onChange(item.target)}
        >
          <strong>{item.target}</strong>
          <span>{item.task === "classification" ? "分類" : "回帰"}</span>
        </button>
      ))}
    </div>
  );
}

function EvaluationSummary({ evaluation, target }) {
  const result = evaluation?.targets?.[target];
  const metrics = useMemo(() => {
    if (!result) return [];
    return [...new Set([
      ...(result.train || []).flatMap((record) => Object.keys(record || {})),
      ...(result.test || []).flatMap((record) => Object.keys(record || {})),
    ])];
  }, [result]);

  if (!evaluation || !result) {
    return (
      <section className="model-result-evaluation empty">
        <div className="model-result-section-head">
          <div><span>VALIDATION</span><strong>精度検証</strong></div>
          <span className="status-chip">未実施</span>
        </div>
        <p>精度検証を有効にして学習すると、TrainとValidationの指標をここに表示します。</p>
      </section>
    );
  }

  const methodLabel = evaluation.method === "loo" ? "Leave-One-Out" : `${evaluation.n_splits}-fold`;
  return (
    <section className="model-result-evaluation">
      <div className="model-result-section-head">
        <div><span>VALIDATION</span><strong>精度検証</strong></div>
        <span className="status-chip success">{methodLabel}</span>
      </div>
      <div className="model-result-metric-head" aria-hidden="true">
        <span>指標</span><span>Train</span><span>Validation</span>
      </div>
      <div className="model-result-metrics">
        {metrics.map((metric) => (
          <div className="model-result-metric-row" key={metric}>
            <strong>{metric}</strong>
            <span>{formatSummary(summarizeMetric(result.train, metric))}</span>
            <span>{formatSummary(summarizeMetric(result.test, metric))}</span>
          </div>
        ))}
      </div>
      <p className="model-result-metric-note">複数foldの場合は平均 ± 標準偏差を表示しています。</p>
    </section>
  );
}

export default function ModelResultVisualizationControl() {
  const { step, modelInfo } = useWorkbench();
  const [host, setHost] = useState(null);
  const [result, setResult] = useState(null);
  const [liveEvaluation, setLiveEvaluation] = useState(null);
  const [activeTarget, setActiveTarget] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (step !== "model" || !modelInfo?.model_id) {
      setHost((current) => {
        if (current?.isConnected) current.remove();
        return null;
      });
      return undefined;
    }

    const contentRoot = document.querySelector(".content-inner") || document.body;
    let frameId = null;
    let disposed = false;
    const resolveHost = () => {
      if (disposed) return;
      const panel = contentRoot.querySelector(".model-registration-panel");
      if (!panel) {
        setHost(null);
        return;
      }
      const panelTitle = panel.querySelector(":scope > .panel-title");
      let nextHost = panel.querySelector(":scope > .model-result-visualization-host");
      if (!nextHost) {
        nextHost = document.createElement("div");
        nextHost.className = "model-result-visualization-host";
        nextHost.dataset.location = "registered-model";
        nextHost.setAttribute("aria-label", "登録モデルの構成と精度検証");
      }
      if (panelTitle) {
        if (panelTitle.nextElementSibling !== nextHost) panelTitle.insertAdjacentElement("afterend", nextHost);
      } else if (panel.firstElementChild !== nextHost) {
        panel.prepend(nextHost);
      }
      setHost(nextHost);
    };
    const scheduleResolve = () => {
      if (frameId !== null) window.cancelAnimationFrame(frameId);
      frameId = window.requestAnimationFrame(() => {
        frameId = null;
        resolveHost();
      });
    };
    scheduleResolve();
    contentRoot.addEventListener("click", scheduleResolve);
    return () => {
      disposed = true;
      contentRoot.removeEventListener("click", scheduleResolve);
      if (frameId !== null) window.cancelAnimationFrame(frameId);
      setHost((current) => {
        if (current?.isConnected) current.remove();
        return null;
      });
    };
  }, [step, modelInfo?.model_id]);

  useEffect(() => {
    setLiveEvaluation(null);
    if (!modelInfo?.model_id) {
      setResult(null);
      return undefined;
    }
    let cancelled = false;
    setLoading(true);
    setError("");
    api.modelVisualization(modelInfo.model_id)
      .then((payload) => { if (!cancelled) setResult(payload); })
      .catch((requestError) => { if (!cancelled) setError(requestError.message || String(requestError)); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [modelInfo?.model_id]);

  useEffect(() => {
    const updateEvaluation = (event) => {
      const evaluation = event.detail;
      if (evaluation?.model_id === modelInfo?.model_id) setLiveEvaluation(evaluation);
    };
    window.addEventListener("malchan:model-evaluated", updateEvaluation);
    return () => window.removeEventListener("malchan:model-evaluated", updateEvaluation);
  }, [modelInfo?.model_id]);

  const targets = result?.targets || [];
  useEffect(() => {
    setActiveTarget((current) => (
      targets.some((item) => item.target === current) ? current : targets[0]?.target || ""
    ));
  }, [targets.map((item) => item.target).join("\u0001")]);

  if (!host) return null;
  const targetDiagram = targets.find((item) => item.target === activeTarget) || targets[0];
  const evaluation = liveEvaluation || result?.evaluation;
  const createdAt = modelInfo?.created_at ? new Date(modelInfo.created_at).toLocaleString("ja-JP") : "—";

  return createPortal(
    <div className="model-result-visualization">
      <div className="model-result-summary">
        <div><span>MODEL ID</span><strong title={modelInfo?.model_id}>{modelInfo?.model_id}</strong></div>
        <div><span>目的変数</span><strong>{modelInfo?.target_cols?.length || 0}</strong></div>
        <div><span>入力特徴量</span><strong>{modelInfo?.feature_columns?.length || 0}</strong></div>
        <div><span>登録日時</span><strong>{createdAt}</strong></div>
      </div>

      <ModelTargetTabs targets={targets} activeTarget={activeTarget} onChange={setActiveTarget} />

      <section className="model-result-diagram-section">
        <div className="model-result-section-head">
          <div><span>MODEL STRUCTURE</span><strong>{targetDiagram?.target || "学習済みモデル"}の処理構成</strong></div>
          <div className="model-result-model-tags">
            {(targetDiagram?.model_names || []).map((name) => <span key={name}>{name}</span>)}
          </div>
        </div>
        {loading && <p className="settings-note">学習済みモデルの構成を取得しています...</p>}
        {error && <p className="xai-error">{error}</p>}
        {!loading && targetDiagram && <ModelStructureDiagram structure={targetDiagram.structure} />}
      </section>

      <EvaluationSummary evaluation={evaluation} target={targetDiagram?.target} />

      <details className="model-result-metadata">
        <summary>モデル情報をJSONで確認</summary>
        <pre className="codebox">{JSON.stringify(modelInfo, null, 2)}</pre>
      </details>
    </div>,
    host,
  );
}
