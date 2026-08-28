import React, { useMemo, useState } from "react";
import { api } from "../api";
import { Field, SectionHeader } from "../components/Common";
import { useWorkbench } from "../context/WorkbenchContext";
import { useAnalysisColumnSelection } from "../context/useAnalysisColumnSelection";
import { createReportSnapshot } from "../report";
import { downloadInteractiveHtmlReport } from "../report-interactive-export";
import { generateChatGptReportPrompt } from "../report-prompt";
import "../report-page.css";

export default function ReportPage() {
  const {
    reportProblem,
    setReportProblem,
    report,
    setReport,
    fileName,
    rows,
    columns: rawColumns,
    numeric: rawNumeric,
    categorical: rawCategorical,
    targets,
    tasks,
    features,
    stats: rawStats,
    modelInfo,
    comparison,
    diagnostics,
    prediction,
    inverseResult,
    objectives,
    bounds,
    sampler,
    inverseTrials,
    topK,
  } = useWorkbench();
  const { enabledColumns: columns, enabledSet } = useAnalysisColumnSelection(rawColumns, rows);
  const numeric = useMemo(
    () => rawNumeric.filter((column) => enabledSet.has(column)),
    [rawNumeric, enabledSet],
  );
  const categorical = useMemo(
    () => rawCategorical.filter((column) => enabledSet.has(column)),
    [rawCategorical, enabledSet],
  );
  const stats = useMemo(
    () => rawStats.filter((item) => enabledSet.has(item.column)),
    [rawStats, enabledSet],
  );
  const missing = useMemo(
    () => stats.reduce((sum, item) => sum + item.missing, 0),
    [stats],
  );
  const [downloadStatus, setDownloadStatus] = useState("");
  const [downloadError, setDownloadError] = useState("");
  const [preparingReport, setPreparingReport] = useState(false);
  const [promptStatus, setPromptStatus] = useState("");
  const [promptError, setPromptError] = useState("");
  const [preparingPrompt, setPreparingPrompt] = useState(false);

  const snapshot = useMemo(() => createReportSnapshot({
    reportProblem,
    reportText: report,
    fileName,
    rows,
    columns,
    numeric,
    categorical,
    targets,
    tasks,
    features,
    stats,
    missing,
    modelInfo,
    comparison,
    diagnostics,
    prediction,
    inverseResult,
    objectives,
    bounds,
    sampler,
    inverseTrials,
    topK,
  }), [
    reportProblem,
    report,
    fileName,
    rows,
    columns,
    numeric,
    categorical,
    targets,
    tasks,
    features,
    stats,
    missing,
    modelInfo,
    comparison,
    diagnostics,
    prediction,
    inverseResult,
    objectives,
    bounds,
    sampler,
    inverseTrials,
    topK,
  ]);

  const comparisonCount = Object.keys(comparison?.targets || {}).length;
  const inverseCandidateCount = inverseResult?.candidates?.length || 0;
  const reportReady = Boolean(
    rows.length
    || modelInfo
    || comparison
    || inverseResult
    || reportProblem.trim()
    || report.trim(),
  );

  async function downloadReport() {
    if (preparingReport) return;
    setPreparingReport(true);
    setDownloadError("");
    setDownloadStatus(modelInfo?.model_id
      ? "Y–Y、重要度、PD、SHAPを収集しています..."
      : "HTMLレポートを作成しています...");
    try {
      const result = await downloadInteractiveHtmlReport(snapshot, {
        modelId: modelInfo?.model_id,
        targets,
        tasks,
        features,
        rows,
        onProgress: setDownloadStatus,
      });
      if (result.interactiveFigureCount && result.interactiveRuntimeEmbedded) {
        setDownloadStatus(
          `${result.fileName} を作成しました（${result.interactiveFigureCount}図を拡大・編集できます）。`,
        );
      } else if (result.interactiveFigureCount) {
        setDownloadStatus(
          `${result.fileName} を作成しました。図の拡大はできますが、Plotly編集機能を埋め込めませんでした。`,
        );
      } else {
        setDownloadStatus(`${result.fileName} を作成しました。`);
      }
    } catch (error) {
      setDownloadStatus("");
      setDownloadError(error.message || String(error));
    } finally {
      setPreparingReport(false);
    }
  }

  async function generatePrompt() {
    if (preparingPrompt) return;
    setPreparingPrompt(true);
    setPromptError("");
    setPromptStatus(modelInfo?.model_id
      ? "重要度・SHAP・1D PDの傾向を整理しています..."
      : "分析コンテキストを整理しています...");
    try {
      const result = await generateChatGptReportPrompt({
        apiClient: api,
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
        onProgress: setPromptStatus,
      });
      setReport(result.prompt);
      setPromptStatus(result.summarizedFeatureCount
        ? `ChatGPT用プロンプトを作成しました（主要特徴量 ${result.summarizedFeatureCount} 件を要約）。`
        : "ChatGPT用プロンプトを作成しました。XAIが未実施の場合は、利用可能な分析結果のみを含めています。");
    } catch (error) {
      setPromptStatus("");
      setPromptError(error.message || String(error));
    } finally {
      setPreparingPrompt(false);
    }
  }

  const downloadLabel = preparingReport
    ? "レポートを作成中..."
    : "HTMLレポートをダウンロード";
  const promptLabel = preparingPrompt
    ? "分析結果を要約中..."
    : "ChatGPT用プロンプトを生成";

  return (
    <>
      <SectionHeader
        step="8 · REPORT"
        title="分析結果をHTMLレポートにまとめる"
        text="データ概要、モデル比較、Y–Y・重要度・PD・SHAP、逆解析を統合し、図はHTML上で拡大・編集できます。"
        action={(
          <button disabled={!reportReady || preparingReport} onClick={downloadReport}>
            {downloadLabel}
          </button>
        )}
      />

      <article className="panel best-model-panel report-export-panel">
        <div className="panel-title">
          <div>
            <span className="panel-kicker">HTML REPORT</span>
            <h3>共有用の自己完結レポート</h3>
            <p>図をクリックして拡大し、軸レンジ、文字サイズ、表示高さを後から変更できます。</p>
          </div>
          <span className={`status-chip ${reportReady ? "success" : ""}`}>
            {preparingReport ? "Building" : reportReady ? "Ready" : "No result"}
          </span>
        </div>

        <div className="result-metric-grid report-export-metrics">
          <div className="result-metric">
            <span>Data</span>
            <strong>{rows.length ? `${rows.length} rows` : "未読込"}</strong>
          </div>
          <div className="result-metric">
            <span>Model comparison</span>
            <strong>{comparisonCount ? `${comparisonCount} targets` : "未実施"}</strong>
          </div>
          <div className="result-metric">
            <span>Inverse candidates</span>
            <strong>{inverseCandidateCount || "未実施"}</strong>
          </div>
        </div>

        <div className="report-export-grid">
          <div>
            <span>収録</span>
            <strong>データ概要・基本統計</strong>
          </div>
          <div>
            <span>収録</span>
            <strong>モデル設定・比較ランキング</strong>
          </div>
          <div>
            <span>モデル学習後</span>
            <strong>Y–Y・残差／混同行列</strong>
          </div>
          <div>
            <span>XAI計算後</span>
            <strong>重要度・SHAP・PD</strong>
          </div>
          <div>
            <span>HTML上で操作</span>
            <strong>拡大・軸範囲・文字サイズ・PNG保存</strong>
          </div>
          <div>
            <span>条件付き</span>
            <strong>逆解析条件・候補一覧</strong>
          </div>
        </div>

        <p className="settings-note report-export-note">
          印刷用の静止画像に加えて元のPlotly図もHTMLへ埋め込みます。拡大画面ではズーム、パン、軸範囲、文字サイズ、図の高さを変更し、調整後のPNGを保存できます。
        </p>

        <div className="report-download-actions">
          <button disabled={!reportReady || preparingReport} onClick={downloadReport}>
            {downloadLabel}
          </button>
          <span>{fileName || "分析データ未選択"}</span>
        </div>
        {downloadStatus && <p className="report-download-status">{downloadStatus}</p>}
        {downloadError && <p className="report-download-status error">{downloadError}</p>}
      </article>

      <article className="panel">
        <div className="panel-title">
          <div>
            <span className="panel-kicker">LLM REPORT PROMPT</span>
            <h3>分析課題とChatGPT用レポートプロンプト</h3>
            <p>モデル比較、重要度、SHAP、1D PD、逆解析をLLM向けの分析コンテキストへ整理します。</p>
          </div>
          <span className={`status-chip ${report ? "success" : ""}`}>
            {preparingPrompt ? "Building" : report ? "Prompt ready" : "Optional"}
          </span>
        </div>

        <div className="report-prompt-layout">
          <aside className="settings-card report-prompt-settings">
            <Field label="課題">
              <textarea
                rows="10"
                value={reportProblem}
                onChange={(event) => setReportProblem(event.target.value)}
                placeholder="分析の目的、背景、確認したい仮説など"
              />
            </Field>
            <button
              className="full-button"
              disabled={preparingPrompt}
              onClick={generatePrompt}
            >
              {promptLabel}
            </button>
            {promptStatus && <p className="report-download-status">{promptStatus}</p>}
            {promptError && <p className="report-download-status error">{promptError}</p>}
          </aside>
          <div className="report-prompt-output">
            <textarea
              value={report}
              onChange={(event) => setReport(event.target.value)}
              placeholder="ChatGPTへそのまま貼り付けるレポート作成プロンプト"
            />
            <div className="report-prompt-actions">
              <button
                className="secondary"
                disabled={!report}
                onClick={() => navigator.clipboard.writeText(report)}
              >
                コピー
              </button>
            </div>
          </div>
        </div>
      </article>
    </>
  );
}
