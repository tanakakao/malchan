import React from "react";
import Chart from "../components/SimpleChart";
import { SectionHeader } from "../components/Common";
import { useWorkbench } from "../context/WorkbenchContext";

export default function ExplainPage() {
  const { targets, diagnostics, modelInfo, busy, updateDiagnostics } = useWorkbench();
  const target = targets[0];
  const actual = diagnostics.map((item) => item.actual[target]);
  const predicted = diagnostics.map(
    (item) => item.predicted[target] ?? item.predicted[`pred_${target}`],
  );
  const residual = actual.map((value, index) => Number(value) - Number(predicted[index]));

  return (
    <>
      <SectionHeader
        step="5 · EXPLAIN"
        title="精度とモデル挙動を説明する"
        text="登録モデルによるY-Y図と残差を確認します。"
        action={
          <button disabled={!modelInfo || busy} onClick={updateDiagnostics}>
            診断を更新
          </button>
        }
      />
      {diagnostics.length ? (
        <div className="split">
          <article className="panel">
            <Chart
              data={[
                { type: "scatter", mode: "markers", x: actual, y: predicted, marker: { color: "#6d8cff", size: 8 } },
                { type: "scatter", mode: "lines", x: actual, y: actual, line: { color: "#50d09c", dash: "dash" } },
              ]}
              layout={{ title: "Y-Yプロット", xaxis: { title: "Actual" }, yaxis: { title: "Predicted" } }}
            />
          </article>
          <article className="panel">
            <Chart
              data={[{ type: "scatter", mode: "markers", x: predicted, y: residual, marker: { color: "#f0b85b", size: 8 } }]}
              layout={{ title: "残差プロット", xaxis: { title: "Predicted" }, yaxis: { title: "Actual - Predicted" } }}
            />
          </article>
        </div>
      ) : (
        <article className="panel"><p className="empty-state">診断を更新してください。</p></article>
      )}
      <article className="panel">
        <h3>SHAP・重要度・部分依存</h3>
        <p className="settings-note">
          Python側の可視化機能をHTTP化する次段階の拡張領域です。
        </p>
      </article>
    </>
  );
}
