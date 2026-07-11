import React from "react";
import DataTable from "../components/DataTable";
import { Field, SectionHeader } from "../components/Common";
import { formatNumber, uniqueValues } from "../data";
import { useWorkbench } from "../context/WorkbenchContext";

export default function OptimizePage() {
  const {
    rows, categorical, targets, tasks, features,
    predictValues, setPredictValues, prediction, predictOne,
    objectives, setObjectives, numFeatures, bounds, setBounds,
    sampler, setSampler, inverseTrials, setInverseTrials, topK, setTopK,
    inverseResult, runInverseAnalysis, modelInfo, busy,
  } = useWorkbench();

  return (
    <>
      <SectionHeader
        step="6 · OPTIMIZE"
        title="予測と逆解析を実行する"
        text="任意条件予測と目的条件を満たす入力候補探索を行います。"
      />
      <article className="panel">
        <h3>任意条件で予測</h3>
        <div className="form-grid">
          {features.map((column) => (
            <Field key={column} label={column}>
              {categorical.includes(column) ? (
                <select
                  value={predictValues[column] ?? ""}
                  onChange={(event) => setPredictValues({ ...predictValues, [column]: event.target.value })}
                >
                  {uniqueValues(rows, column).map((value) => (
                    <option key={String(value)} value={value}>{String(value)}</option>
                  ))}
                </select>
              ) : (
                <input
                  type="number"
                  step="any"
                  value={predictValues[column] ?? ""}
                  onChange={(event) => setPredictValues({ ...predictValues, [column]: event.target.value })}
                />
              )}
            </Field>
          ))}
        </div>
        <button disabled={!modelInfo || busy} onClick={predictOne}>予測を実行</button>
        {prediction && (
          <div className="prediction-result-grid">
            {Object.entries(prediction).map(([key, value]) => (
              <div className="prediction-chip" key={key}><span>{key}</span><strong>{formatNumber(value)}</strong></div>
            ))}
          </div>
        )}
      </article>

      <article className="panel">
        <h3>逆解析</h3>
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
                      setObjectives({ ...objectives, [target]: { ...objective, value: event.target.value } })
                    }
                  />
                ) : (
                  <select
                    value={objective.value || "max"}
                    onChange={(event) =>
                      setObjectives({ ...objectives, [target]: { ...objective, value: event.target.value } })
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
                    setBounds({ ...bounds, [column]: { ...bounds[column], min: event.target.value } })
                  }
                />
              </Field>
              <Field label="max">
                <input
                  type="number"
                  value={bounds[column]?.max ?? 1}
                  onChange={(event) =>
                    setBounds({ ...bounds, [column]: { ...bounds[column], max: event.target.value } })
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
          <Field label="Trials"><input type="number" min="1" value={inverseTrials} onChange={(event) => setInverseTrials(event.target.value)} /></Field>
          <Field label="候補数"><input type="number" min="1" value={topK} onChange={(event) => setTopK(event.target.value)} /></Field>
        </div>
        <button disabled={!modelInfo || busy} onClick={runInverseAnalysis}>逆解析を実行 →</button>
      </article>

      {inverseResult && (
        <article className="panel">
          <h3>逆解析候補</h3>
          <DataTable
            rows={inverseResult.candidates}
            columns={Object.keys(inverseResult.candidates[0] || {})}
          />
        </article>
      )}
    </>
  );
}
