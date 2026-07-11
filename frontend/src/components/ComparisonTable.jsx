import React from "react";
import { formatNumber } from "../data";

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
