import React from "react";
import { Field, SectionHeader } from "../components/Common";
import { useWorkbench } from "../context/WorkbenchContext";

export default function ReportPage() {
  const {
    reportProblem, setReportProblem, report, setReport, makeReportPrompt,
  } = useWorkbench();
  return (
    <>
      <SectionHeader
        step="7 · REPORT"
        title="分析レポート用プロンプトを作成する"
        text="データ、比較、逆解析結果を生成AI向けに統合します。"
      />
      <article className="panel">
        <div className="report-prompt-layout">
          <aside className="settings-card">
            <Field label="課題">
              <textarea rows="10" value={reportProblem} onChange={(event) => setReportProblem(event.target.value)} />
            </Field>
            <button className="full-button" onClick={makeReportPrompt}>プロンプトを作成</button>
          </aside>
          <div className="report-prompt-output">
            <textarea
              value={report}
              onChange={(event) => setReport(event.target.value)}
              placeholder="生成したプロンプト"
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
