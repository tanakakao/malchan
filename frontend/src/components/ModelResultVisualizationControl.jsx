import React, { useEffect, useMemo, useState } from "react";
import { createPortal } from "react-dom";
import { api } from "../api";
import { evaluationMetricNames, metricValue } from "../evaluationMetrics";
import { useWorkbench } from "../context/WorkbenchContext";
import ModelStructureSummaryTable from "./ModelStructureSummaryTable";

function formatMetric(value) {
  if (!Number.isFinite(value)) return "—";
  const magnitude = Math.abs(value);
  if (magnitude >= 1000 || (magnitude > 0 && magnitude < 0.001)) {
    return value.toExponential(4);
  }
  return value
    .toFixed(4)
    .replace(/\.0+$/, "")
    .replace(/(\.\d*?)0+$/, "$1");
}

function summarizeMetric(records, metric) {
  const values = (records || [])
    .map((record) => metricValue(record, metric))
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

function classificationOofMetrics(records) {
  const rows = (records || []).filter(
    (record) => record?.actual !== undefined && record?.predicted !== undefined,
  );
  if (!rows.length) return {};

  const labels = [...new Set(rows.flatMap((record) => [record.actual, record.predicted]))];
  const total = rows.length;
  const accuracy = rows.filter((record) => record.actual === record.predicted).length / total;
  let weightedPrecision = 0;
  let weightedRecall = 0;
  let weightedF1 = 0;

  labels.forEach((label) => {
    const support = rows.filter((record) => record.actual === label).length;
    const truePositive = rows.filter(
      (record) => record.actual === label && record.predicted === label,
    ).length;
    const falsePositive = rows.filter(
      (record) => record.actual !== label && record.predicted === label,
    ).length;
    const falseNegative = rows.filter(
      (record) => record.actual === label && record.predicted !== label,
    ).length;
    const precision = truePositive + falsePositive
      ? truePositive / (truePositive + falsePositive)
      : 0;
    const recall = truePositive + falseNegative
      ? truePositive / (truePositive + falseNegative)
      : 0;
    const f1 = precision + recall ? (2 * precision * recall) / (precision + recall) : 0;
    const weight = support / total;
    weightedPrecision += precision * weight;
    weightedRecall += recall * weight;
    weightedF1 += f1 * weight;
  });

  return {
    accuracy,
    precision: weightedPrecision,
    recall: weightedRecall,
    f1: weightedF1,
  };
}

function regressionOofMetrics(records) {
  const rows = (records || [])
    .map((record) => ({
      actual: Number(record?.actual),
      predicted: Number(record?.predicted),
    }))
    .filter((record) => Number.isFinite(record.actual) && Number.isFinite(record.predicted));
  if (!rows.length) return {};

  const errors = rows.map((record) => record.predicted - record.actual);
  const absoluteErrors = errors.map(Math.abs);
  const squaredErrors = errors.map((value) => value ** 2);
  const actualMean = rows.reduce((sum, record) => sum + record.actual, 0) / rows.length;
  const residualSum = squaredErrors.reduce((sum, value) => sum + value, 0);
  const totalSum = rows.reduce((sum, record) => sum + (record.actual - actualMean) ** 2, 0);
  const percentageErrors = rows
    .filter((record) => record.actual !== 0)
    .map((record) => Math.abs((record.predicted - record.actual) / record.actual));

  return {
    mae: absoluteErrors.reduce((sum, value) => sum + value, 0) / rows.length,
    mse: residualSum / rows.length,
    rmse: Math.sqrt(residualSum / rows.length),
    r2: totalSum > 0 ? 1 - residualSum / totalSum : 0,
    mape: percentageErrors.length
      ? percentageErrors.reduce((sum, value) => sum + value, 0) / percentageErrors.length
      : undefined,
  };
}

function comparisonTargetEvaluation(comparisonResult, task) {
  if (!comparisonResult) return null;
  return {
    task,
    train: comparisonResult.best_cv_scores?.train || [],
    test: comparisonResult.best_cv_scores?.test || [],
    oof: task === "classification"
      ? classificationOofMetrics(comparisonResult.best_cv_predictions?.test)
      : regressionOofMetrics(comparisonResult.best_cv_predictions?.test),
  };
}

function resolvedTargetEvaluation(evaluation, comparison, target, task) {
  const direct = evaluation?.targets?.[target];
  const compared = comparisonTargetEvaluation(comparison?.targets?.[target], task);
  if (!direct) return compared;
  return {
    ...direct,
    oof: direct.oof || compared?.oof || {},
  };
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

function EvaluationSummary({ evaluation, comparison, targets, activeTarget }) {
  const entries = useMemo(
    () => targets
      .map((item) => [
        item.target,
        resolvedTargetEvaluation(evaluation, comparison, item.target, item.task),
      ])
      .filter(([, result]) => result),
    [evaluation, comparison, targets],
  );

  if (!entries.length) {
    return (
      <section className="model-result-evaluation empty">
        <div className="model-result-section-head">
          <div><span>CROSS VALIDATION</span><strong>交差検証による精度評価</strong></div>
          <span className="status-chip">未実施</span>
        </div>
        <p>Model画面で精度検証またはモデル比較を実行すると表示します。</p>
      </section>
    );
  }

  const methodLabel = evaluation
    ? evaluation.method === "loo"
      ? "Leave-One-Out"
      : `${evaluation.n_splits}-fold`
    : "Model comparison CV";

  return (
    <section className="model-result-evaluation bochan-evaluation-summary">
      <div className="model-result-section-head">
        <div><span>CROSS VALIDATION</span><strong>交差検証による精度評価</strong></div>
        <span className="status-chip success">{methodLabel}</span>
      </div>
      <div className="bochan-evaluation-targets">
        {entries.map(([target, result], index) => {
          const metrics = evaluationMetricNames(result);
          return (
            <details
              key={target}
              open={target === activeTarget || (!activeTarget && index === 0)}
            >
              <summary>{target}</summary>
              <div className="table-wrap compact bochan-evaluation-table">
                <table>
                  <thead>
                    <tr>
                      <th>指標</th>
                      <th>Train</th>
                      <th>Validation</th>
                      <th>OOF</th>
                    </tr>
                  </thead>
                  <tbody>
                    {metrics.map((metric) => (
                      <tr key={`${target}-${String(metric).toLowerCase()}`}>
                        <td>{String(metric).toUpperCase()}</td>
                        <td>{formatSummary(summarizeMetric(result.train, metric))}</td>
                        <td>{formatSummary(summarizeMetric(result.test, metric))}</td>
                        <td>{formatMetric(metricValue(result.oof, metric))}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </details>
          );
        })}
      </div>
    </section>
  );
}

function removeHost(current) {
  if (current?.isConnected) current.remove();
  return null;
}

function markExplainPanels(stack) {
  const panels = stack?.querySelectorAll(":scope > .xai-result-panel") || [];
  panels[0]?.classList.add("xai-yy-panel");
  panels[1]?.classList.add("xai-importance-panel");
  panels[2]?.classList.add("xai-relationship-panel");
}

function unmarkExplainPanels(root) {
  root?.querySelectorAll(
    ".xai-yy-panel, .xai-importance-panel, .xai-relationship-panel",
  ).forEach((panel) => {
    panel.classList.remove(
      "xai-yy-panel",
      "xai-importance-panel",
      "xai-relationship-panel",
    );
  });
}

export default function ModelResultVisualizationControl() {
  const { step, modelInfo, comparison } = useWorkbench();
  const [host, setHost] = useState(null);
  const [result, setResult] = useState(null);
  const [liveEvaluation, setLiveEvaluation] = useState(null);
  const [activeTarget, setActiveTarget] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    const isModel = step === "model";
    const isExplain = step === "explain";
    if ((!isModel && !isExplain) || (isModel && !modelInfo?.model_id)) {
      setHost(removeHost);
      return undefined;
    }

    const contentRoot = document.querySelector(".content-inner") || document.body;
    let frameId = null;
    let disposed = false;

    const resolveModelHost = () => {
      const panel = contentRoot.querySelector(".model-registration-panel");
      if (!panel) return null;
      const panelTitle = panel.querySelector(":scope > .panel-title");
      let nextHost = panel.querySelector(":scope > .model-result-visualization-host");
      if (!nextHost) {
        nextHost = document.createElement("div");
        nextHost.className = "model-result-visualization-host";
        nextHost.dataset.location = "registered-model";
        nextHost.setAttribute("aria-label", "登録モデルの構成");
      }
      if (panelTitle) {
        if (panelTitle.nextElementSibling !== nextHost) {
          panelTitle.insertAdjacentElement("afterend", nextHost);
        }
      } else if (panel.firstElementChild !== nextHost) {
        panel.prepend(nextHost);
      }
      return nextHost;
    };

    const resolveExplainHost = () => {
      const stack = contentRoot.querySelector(".xai-results-stack");
      if (!stack) return null;
      markExplainPanels(stack);
      let nextHost = stack.querySelector(":scope > .xai-evaluation-host");
      if (!nextHost) {
        nextHost = document.createElement("div");
        nextHost.className = "xai-evaluation-host";
        nextHost.dataset.location = "explain-evaluation";
        nextHost.setAttribute("aria-label", "交差検証による精度評価");
      }
      const yyPanel = stack.querySelector(":scope > .xai-yy-panel");
      if (yyPanel && yyPanel.nextElementSibling !== nextHost) {
        yyPanel.insertAdjacentElement("afterend", nextHost);
      } else if (!yyPanel && stack.firstElementChild !== nextHost) {
        stack.prepend(nextHost);
      }
      const selectedTarget = contentRoot.querySelector(".xai-overview select")?.value;
      if (selectedTarget) setActiveTarget(selectedTarget);
      return nextHost;
    };

    const resolveHost = () => {
      if (disposed) return;
      const nextHost = isExplain ? resolveExplainHost() : resolveModelHost();
      setHost(nextHost);
    };

    const scheduleResolve = () => {
      if (frameId !== null) window.cancelAnimationFrame(frameId);
      frameId = window.requestAnimationFrame(() => {
        frameId = null;
        resolveHost();
      });
    };

    const handleChange = (event) => {
      if (
        isExplain
        && event.target instanceof HTMLSelectElement
        && event.target.closest(".xai-overview")
      ) {
        setActiveTarget(event.target.value);
      }
      scheduleResolve();
    };

    const observer = new MutationObserver(scheduleResolve);
    observer.observe(contentRoot, { childList: true, subtree: true });
    scheduleResolve();
    contentRoot.addEventListener("click", scheduleResolve);
    contentRoot.addEventListener("change", handleChange);
    return () => {
      disposed = true;
      observer.disconnect();
      contentRoot.removeEventListener("click", scheduleResolve);
      contentRoot.removeEventListener("change", handleChange);
      if (frameId !== null) window.cancelAnimationFrame(frameId);
      unmarkExplainPanels(contentRoot);
      setHost(removeHost);
    };
  }, [step, modelInfo?.model_id]);

  useEffect(() => {
    setLiveEvaluation(null);
    if (!modelInfo?.model_id) {
      setResult(null);
      setError("");
      setLoading(false);
      return undefined;
    }
    let cancelled = false;
    setLoading(true);
    setError("");
    api.modelVisualization(modelInfo.model_id)
      .then((payload) => {
        if (!cancelled) setResult(payload);
      })
      .catch((requestError) => {
        if (!cancelled) setError(requestError.message || String(requestError));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
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

  if (step === "explain") {
    return createPortal(
      <article className="panel xai-result-panel xai-evaluation-panel">
        <div className="xai-result-head">
          <span>02</span>
          <div>
            <strong>MODEL EVALUATION</strong>
            <h3>精度評価の結果</h3>
          </div>
        </div>
        {loading && <p className="empty-state">精度評価を取得しています...</p>}
        {error && <p className="xai-error">{error}</p>}
        {!loading && !error && (
          <EvaluationSummary
            evaluation={evaluation}
            comparison={comparison}
            targets={targets}
            activeTarget={activeTarget}
          />
        )}
      </article>,
      host,
    );
  }

  const createdAt = modelInfo?.created_at
    ? new Date(modelInfo.created_at).toLocaleString("ja-JP")
    : "—";

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
          <div><span>MODEL SUMMARY</span><strong>{targetDiagram?.target || "学習済みモデル"}の構成一覧</strong></div>
          <div className="model-result-model-tags">
            {(targetDiagram?.model_names || []).map((name) => <span key={name}>{name}</span>)}
          </div>
        </div>
        {loading && <p className="settings-note">学習済みモデルの構成を取得しています...</p>}
        {error && <p className="xai-error">{error}</p>}
        {!loading && targetDiagram && (
          <ModelStructureSummaryTable
            structure={targetDiagram.structure}
            featureColumns={modelInfo?.feature_columns || []}
            modelNames={targetDiagram.model_names || []}
          />
        )}
      </section>

      <details className="model-result-metadata">
        <summary>モデル情報をJSONで確認</summary>
        <pre className="codebox">{JSON.stringify(modelInfo, null, 2)}</pre>
      </details>
    </div>,
    host,
  );
}
