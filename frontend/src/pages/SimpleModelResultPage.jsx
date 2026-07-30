import React from "react";
import ComparisonTable from "../components/ComparisonTable";
import { SectionHeader } from "../components/Common";
import { useWorkbench } from "../context/WorkbenchContext";

const SIMPLE_MODELS = [
  "線形回帰",
  "ElasticNet",
  "ランダムフォレスト回帰",
  "LightGBM",
];

export default function SimpleModelResultPage() {
  const { comparison, modelInfo, setStep } = useWorkbench();
  const targetResults = comparison?.targets || {};
  const bestModels = Object.entries(targetResults)
    .map(([target, result]) => `${target}: ${result.best_model_name || "—"}`)
    .join(" / ");

  return (
    <>
      <SectionHeader
        step="3 · MODEL"
        title="自動モデル比較の結果"
        text="固定した4候補を同じ5-fold交差検証で比較し、Validation RMSEが最も小さいモデルを有効化します。"
        action={(
          <button
            type="button"
            disabled={!modelInfo}
            onClick={() => setStep("explain")}
          >
            精度と変数影響を確認 →
          </button>
        )}
      />

      <article className="panel simple-model-summary">
        <div className="panel-title">
          <div>
            <span className="panel-kicker">SIMPLE MODE</span>
            <h3>自動選択の条件</h3>
            <p>詳細な前処理・モデル・パラメータ設定は行わず、共通条件で公平に比較します。</p>
          </div>
          <span className={`status-chip ${comparison ? "success" : "warning"}`}>
            {comparison ? "Completed" : "Not run"}
          </span>
        </div>
        <div className="simple-default-grid">
          <span><strong>Task</strong> 単一目的の回帰</span>
          <span><strong>Metric</strong> Validation RMSE</span>
          <span><strong>Validation</strong> 5-fold CV</span>
          <span><strong>Activation</strong> 1位を自動採用</span>
          <span className="simple-model-list"><strong>Models</strong> {SIMPLE_MODELS.join(" / ")}</span>
          <span><strong>Selected</strong> {bestModels || "未実行"}</span>
        </div>
      </article>

      <ComparisonTable comparison={comparison} />

      {!comparison && (
        <div className="button-row simple-result-actions">
          <button type="button" className="secondary" onClick={() => setStep("prepare")}>
            説明変数の選択へ戻る
          </button>
        </div>
      )}
    </>
  );
}
