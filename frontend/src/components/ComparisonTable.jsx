import React from "react";
import { formatNumber } from "../data";

function BestModelEvaluation({ result }) {
  const train = result.best_cv_scores?.train?.[0] || {};
  const validation = result.best_cv_scores?.test?.[0] || {};
  const metrics = [...new Set([...Object.keys(train), ...Object.keys(validation)])];
  const trainPredictions = result.best_cv_predictions?.train || [];
  const validationPredictions = result.best_cv_predictions?.test || [];

  if (!metrics.length && !trainPredictions.length && !validationPredictions.length) return null;

  return (
    <section className="comparison-best-evaluation">
      <div className="comparison-evaluation-head">
        <div>
          <strong>ベストモデル精度評価</strong>
          <span>比較時と同じ交差検証によるTrain／Validationの評価です。</span>
        </div>
        <span className="status-chip success">CV evaluated</span>
      </div>

      {metrics.length > 0 && (
        <div className="table-wrap compact comparison-evaluation-table">
          <table>
            <thead>
              <tr>
                <th>指標</th>
                <th>Train</th>
                <th>Validation</th>
              </tr>
            </thead>
            <tbody>
              {metrics.map((metric) => (
                <tr key={metric}>
                  <td>{metric}</td>
                  <td>{formatNumber(train[metric])}</td>
                  <td>{formatNumber(validation[metric])}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {(trainPredictions.length > 0 || validationPredictions.length > 0) && (
        <div className="comparison-prediction-data">
          <div>
            <span>Train plot data</span>
            <strong>{trainPredictions.length} rows</strong>
          </div>
          <div>
            <span>Validation plot data</span>
            <strong>{validationPredictions.length} rows</strong>
          </div>
          <small>実測値・予測値・行indexを保持し、Y-Y／残差などのプロットに利用できます。</small>
        </div>
      )}
    </section>
  );
}

export default function ComparisonTable({ comparison }) {
  if (!comparison?.targets) return <p className="empty-state">比較結果はまだありません。</p>;
  return (
    <div className="comparison-stack">
      {Object.entries(comparison.targets).map(([target, result]) => {
        const columns = result.ranking?.length ? Object.keys(result.ranking[0]) : [];
        return (
          <article className="comparison-card" key={target}>
            <div className="panel-title">
              <div>
                <span className="panel-kicker">TARGET</span>
                <h4>{target}</h4>
                <p>{result.metric} · {result.higher_is_better ? "大きいほど良い" : "小さいほど良い"}</p>
              </div>
              <span className={`status-chip ${result.best_is_tuned ? "success" : ""}`}>
                {result.best_is_tuned ? "Tuned" : "Compared"}
              </span>
            </div>
            <div className="result-metric-grid">
              <div className="result-metric"><span>Best model</span><strong>{result.best_model_name || "—"}</strong></div>
              <div className="result-metric"><span>Tuned</span><strong>{result.best_is_tuned ? "Yes" : "No"}</strong></div>
              <div className="result-metric"><span>Failures</span><strong>{Object.keys(result.failures || {}).length}</strong></div>
            </div>

            <BestModelEvaluation result={result} />

            <div className="comparison-ranking-head">
              <strong>候補モデルランキング</strong>
              <span>共通の交差検証条件による比較結果</span>
            </div>
            <div className="table-wrap compact">
              <table>
                <thead><tr>{columns.map((column) => <th key={column}>{column}</th>)}</tr></thead>
                <tbody>
                  {(result.ranking || []).map((row, index) => (
                    <tr key={row.model_name || index} className={index === 0 ? "model-rank-best" : ""}>
                      {columns.map((column) => <td key={column}>{formatNumber(row[column])}</td>)}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {result.best_params && (
              <details className="advanced compact-details">
                <summary><strong>採用パラメータ</strong><span>JSON</span></summary>
                <pre className="codebox">{JSON.stringify(result.best_params, null, 2)}</pre>
              </details>
            )}
          </article>
        );
      })}
    </div>
  );
}
