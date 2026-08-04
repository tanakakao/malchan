import React, { useMemo, useState } from "react";
import { Field, SectionHeader } from "../components/Common";
import { useWorkbench } from "../context/WorkbenchContext";
import { createReportSnapshot, downloadHtmlReport } from "../report";
import "../report-page.css";

export default function ReportPage() {
  const {
    reportProblem,
    setReportProblem,
    report,
    setReport,
    makeReportPrompt,
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
  } = useWorkbench();
  const [downloadStatus, setDownloadStatus] = useState("");
  const [downloadError, setDownloadError] = useState("");

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

  function downloadReport() {
    setDownloadError("");
    try {
      const name = downloadHtmlReport(snapshot);
      setDownloadStatus(`${name} を作成しました。`);
    } catch (error) {
      setDownloadStatus("");
      setDownloadError(error.message || String(error));
    }
  }

  return (
    <>
      <SectionHeader
        step="8 · REPORT"
        title="分析結果をHTMLレポートにまとめる"
        text="データ概要、モデル比較、精度診断、逆解析、レポート用テキストを1つのHTMLへ統合します。"
        action={(
          <button disabled={!reportReady} onClick={downloadReport}>
            HTMLレポートをダウンロード
          </button>
        )}
      />

      <article className="panel best-model-panel report-export-panel">
        <div className="panel-title">
          <div>
            <span className="panel-kicker">HTML REPORT</span>
            <h3>共有用の自己完結レポート</h3>
            <p>ブラウザで閲覧でき、レポート内のボタンから印刷またはPDF保存もできます。</p>
          </div>
          <span className={`status-chip ${reportReady ? "success" : ""}`}>
            {reportReady ? "Ready" : "No result"}
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
            <span>条件付き</span>
            <strong>実測値・予測値の診断</strong>
          </div>
          <div>
            <span>条件付き</span>
            <strong>逆解析条件・候補一覧</strong>
          </div>
        </div>

        <p className="settings-note report-export-note">
          入力データの全行はHTMLへ埋め込みません。共有しやすいように、統計量と分析結果の要約のみを収録します。
        </p>

        <div className="report-download-actions">
          <button disabled={!reportReady} onClick={downloadReport}>
            HTMLレポートをダウンロード
          </button>
          <span>{fileName || "分析データ未選択"}</span>
        </div>
        {downloadStatus && <p className="report-download-status">{downloadStatus}</p>}
        {downloadError && <p className="report-download-status error">{downloadError}</p>}
      </article>

      <article className="panel">
        <div className="panel-title">
          <div>
            <span className="panel-kicker">REPORT TEXT</span>
            <h3>分析課題とレポート用テキスト</h3>
            <p>作成したテキストはHTMLレポートの最終セクションへそのまま収録されます。</p>
          </div>
          <span className={`status-chip ${report ? "success" : ""}`}>
            {report ? "Text ready" : "Optional"}
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
            <button className="full-button" onClick={makeReportPrompt}>
              プロンプトを作成
            </button>
          </aside>
          <div className="report-prompt-output">
            <textarea
              value={report}
              onChange={(event) => setReport(event.target.value)}
              placeholder="生成AI向けプロンプト、または編集済みの分析所見"
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
