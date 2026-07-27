import React from "react";
import DataTable from "../components/DataTable";
import { Field, SectionHeader } from "../components/Common";
import { useWorkbench } from "../context/WorkbenchContext";

export default function OptimizePage() {
  const {
    rows, targets, tasks,
    objectives, setObjectives, numFeatures, bounds, setBounds,
    sampler, setSampler, inverseTrials, setInverseTrials, topK, setTopK,
    inverseResult, runInverseAnalysis, modelInfo, busy,
  } = useWorkbench();

  return (
    <>
      <SectionHeader
        step="7 · OPTIMIZE"
        title="目的条件を満たす入力候補を逆解析する"
        text="予測とは独立したページで、目的変数の条件と説明変数の探索範囲を指定します。"
      />

      <article className="panel">
        <div className="panel-title">
          <div>
            <span className="panel-kicker">INVERSE ANALYSIS</span>
            <h3>目的条件と探索範囲</h3>
            <p>登録済みモデルを使い、指定した目的を満たす入力候補を探索します。</p>
          </div>
        </div>
        {!modelInfo && (
          <p className="settings-note">先にModel画面で逆解析に使用するモデルを学習してください。</p>
        )}

        <div className="objective-grid">
          {targets.map((target) => {
            const objective = objectives[target] || {};
            return (
              <div className="objective-card" key={target}>
                <strong>{target}</strong>
                <select
                  value={objective.mode || "direction"}
                  onChange={(event) =>
                    setObjectives({
                      ...objectives,
                      [target]: {
                        mode: event.target.value,
                        value: event.target.value === "direction" ? "max" : rows[0]?.[target] ?? "",
                      },
                    })
                  }
                >
                  <option value="direction" disabled={tasks[target] === "classification"}>方向</option>
                  <option value="target">目標値 / クラス</option>
                </select>
                {objective.mode === "target" ? (
                  <input
                    value={objective.value ?? ""}
                    onChange={(event) =>
                      setObjectives({
                        ...objectives,
                        [target]: { ...objective, value: event.target.value },
                      })
                    }
                  />
                ) : (
                  <select
                    value={objective.value || "max"}
                    onChange={(event) =>
                      setObjectives({
                        ...objectives,
                        [target]: { ...objective, value: event.target.value },
                      })
                    }
                  >
                    <option value="max">最大化</option>
                    <option value="min">最小化</option>
                  </select>
                )}
              </div>
            );
          })}
        </div>

        <div className="inverse-variable-list">
          {numFeatures.map((column) => (
            <div className="inverse-variable" key={column}>
              <strong>{column}</strong>
              <Field label="min">
                <input
                  type="number"
                  value={bounds[column]?.min ?? 0}
                  onChange={(event) =>
                    setBounds({
                      ...bounds,
                      [column]: { ...bounds[column], min: event.target.value },
                    })
                  }
                />
              </Field>
              <Field label="max">
                <input
                  type="number"
                  value={bounds[column]?.max ?? 1}
                  onChange={(event) =>
                    setBounds({
                      ...bounds,
                      [column]: { ...bounds[column], max: event.target.value },
                    })
                  }
                />
              </Field>
            </div>
          ))}
        </div>

        <div className="form-grid">
          <Field label="Sampler">
            <select value={sampler} onChange={(event) => setSampler(event.target.value)}>
              <option>TPE</option><option>MOTPE</option><option>CmaEs</option>
              <option>GP</option><option>QMS</option><option>NSGAII</option><option>NSGAIII</option>
            </select>
          </Field>
          <Field label="Trials">
            <input
              type="number"
              min="1"
              value={inverseTrials}
              onChange={(event) => setInverseTrials(event.target.value)}
            />
          </Field>
          <Field label="候補数">
            <input
              type="number"
              min="1"
              value={topK}
              onChange={(event) => setTopK(event.target.value)}
            />
          </Field>
        </div>
        <button disabled={!modelInfo || busy} onClick={runInverseAnalysis}>逆解析を実行 →</button>
      </article>

      {inverseResult && (
        <article className="panel">
          <div className="panel-title">
            <div>
              <span className="panel-kicker">CANDIDATES</span>
              <h3>逆解析候補</h3>
            </div>
            <span className="status-chip success">{inverseResult.candidates.length} candidates</span>
          </div>
          <DataTable
            rows={inverseResult.candidates}
            columns={Object.keys(inverseResult.candidates[0] || {})}
          />
        </article>
      )}
    </>
  );
}
