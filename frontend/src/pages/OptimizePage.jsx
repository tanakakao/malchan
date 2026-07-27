import React, { useEffect, useMemo, useState } from "react";
import {
  setInverseAnalysisPayloadOverride,
} from "../api";
import DataTable from "../components/DataTable";
import { Field, SectionHeader } from "../components/Common";
import { uniqueValues } from "../data";
import { useWorkbench } from "../context/WorkbenchContext";
import "../optimize-variable-settings.css";

const variableSettingsByModel = new Map();

function isIntegerColumn(rows, column) {
  const values = rows.map((row) => row[column]).filter(Number.isFinite);
  return values.length > 0 && values.every(Number.isInteger);
}

function defaultVariableSetting(rows, column, numeric) {
  const firstValue = rows.find((row) => row[column] !== null && row[column] !== undefined)?.[column];
  return {
    fixed: false,
    fixedValue: firstValue ?? "",
    step: numeric && isIntegerColumn(rows, column) ? 1 : "",
  };
}

function objectiveMode(objective, task) {
  if (task === "classification") return "target";
  if (objective?.mode === "target") return "target";
  if (objective?.mode === "min" || objective?.value === "min") return "min";
  return "max";
}

function originalValue(values, selected) {
  return values.find((value) => String(value) === String(selected)) ?? selected;
}

export default function OptimizePage() {
  const {
    rows,
    targets,
    tasks,
    features,
    numFeatures,
    catFeatures,
    objectives,
    setObjectives,
    bounds,
    setBounds,
    sampler,
    setSampler,
    inverseTrials,
    setInverseTrials,
    topK,
    setTopK,
    inverseResult,
    runInverseAnalysis,
    modelInfo,
    busy,
  } = useWorkbench();
  const modelKey = modelInfo?.model_id || "unregistered";
  const featureKey = features.join("\u0000");
  const [variableSettings, setVariableSettings] = useState(() => {
    const saved = variableSettingsByModel.get(modelKey) || {};
    return Object.fromEntries(features.map((column) => [
      column,
      saved[column] || defaultVariableSetting(rows, column, numFeatures.includes(column)),
    ]));
  });
  const [settingsError, setSettingsError] = useState("");

  useEffect(() => {
    const saved = variableSettingsByModel.get(modelKey) || {};
    setVariableSettings(Object.fromEntries(features.map((column) => [
      column,
      saved[column] || defaultVariableSetting(rows, column, numFeatures.includes(column)),
    ])));
  }, [modelKey, featureKey]);

  useEffect(() => {
    variableSettingsByModel.set(modelKey, variableSettings);
  }, [modelKey, variableSettings]);

  useEffect(() => () => {
    setInverseAnalysisPayloadOverride(null);
  }, []);

  const fixedCount = useMemo(
    () => features.filter((column) => variableSettings[column]?.fixed).length,
    [features, variableSettings],
  );
  const searchedCount = features.length - fixedCount;

  function patchVariable(column, patch) {
    setVariableSettings((current) => ({
      ...current,
      [column]: {
        ...(current[column] || defaultVariableSetting(
          rows,
          column,
          numFeatures.includes(column),
        )),
        ...patch,
      },
    }));
  }

  function changeObjective(target, nextMode) {
    const task = tasks[target];
    if (task === "classification" && nextMode !== "target") return;
    const current = objectives[target] || {};
    const values = uniqueValues(rows, target);
    const defaultTarget = task === "classification"
      ? values[0] ?? ""
      : rows.find((row) => Number.isFinite(row[target]))?.[target] ?? 0;
    setObjectives({
      ...objectives,
      [target]: {
        mode: nextMode,
        value: nextMode === "target"
          ? current.mode === "target"
            ? current.value
            : defaultTarget
          : nextMode,
      },
    });
  }

  function changeTargetValue(target, rawValue) {
    const values = uniqueValues(rows, target);
    setObjectives({
      ...objectives,
      [target]: {
        ...(objectives[target] || {}),
        mode: "target",
        value: tasks[target] === "classification"
          ? originalValue(values, rawValue)
          : rawValue,
      },
    });
  }

  function validateSettings() {
    const errors = [];

    targets.forEach((target) => {
      const mode = objectiveMode(objectives[target], tasks[target]);
      if (mode !== "target") return;
      const value = objectives[target]?.value;
      if (tasks[target] === "regression" && !Number.isFinite(Number(value))) {
        errors.push(`${target}: 目標値を数値で入力してください。`);
      }
      if (tasks[target] === "classification" && (value === "" || value === null || value === undefined)) {
        errors.push(`${target}: 目標クラスを選択してください。`);
      }
    });

    features.forEach((column) => {
      const setting = variableSettings[column] || {};
      const numeric = numFeatures.includes(column);
      if (setting.fixed) {
        if (numeric && !Number.isFinite(Number(setting.fixedValue))) {
          errors.push(`${column}: 固定値を数値で入力してください。`);
        }
        if (!numeric && (setting.fixedValue === "" || setting.fixedValue === null || setting.fixedValue === undefined)) {
          errors.push(`${column}: 固定するカテゴリを選択してください。`);
        }
        return;
      }
      if (!numeric) return;

      const lower = Number(bounds[column]?.min);
      const upper = Number(bounds[column]?.max);
      if (!Number.isFinite(lower) || !Number.isFinite(upper) || upper <= lower) {
        errors.push(`${column}: 上限は下限より大きくしてください。`);
      }
      if (setting.step !== "" && setting.step !== null && setting.step !== undefined) {
        const step = Number(setting.step);
        if (!Number.isFinite(step) || step <= 0) {
          errors.push(`${column}: 刻みは0より大きくしてください。`);
        } else if (isIntegerColumn(rows, column) && !Number.isInteger(step)) {
          errors.push(`${column}: 整数変数の刻みは整数にしてください。`);
        }
      }
    });

    if (searchedCount < 1) {
      errors.push("少なくとも1つの説明変数を探索対象にしてください。");
    }
    if (Number(topK) > Number(inverseTrials)) {
      errors.push("候補数はTrials以下にしてください。");
    }
    return errors;
  }

  function inversePayloadOverride() {
    const normalizedObjectives = targets.map((target) => {
      const mode = objectiveMode(objectives[target], tasks[target]);
      if (mode === "target") {
        const value = objectives[target]?.value;
        return {
          target,
          target_value: tasks[target] === "regression" ? Number(value) : value,
        };
      }
      return { target, direction: mode };
    });

    const numericBounds = Object.fromEntries(
      numFeatures
        .filter((column) => !variableSettings[column]?.fixed)
        .map((column) => {
          const range = {
            min: Number(bounds[column]?.min ?? 0),
            max: Number(bounds[column]?.max ?? 1),
            dtype: isIntegerColumn(rows, column) ? "int" : "float",
          };
          const configuredStep = variableSettings[column]?.step;
          if (configuredStep !== "" && configuredStep !== null && configuredStep !== undefined) {
            range.step = Number(configuredStep);
          } else if (range.dtype === "int") {
            range.step = 1;
          }
          return [column, range];
        }),
    );

    const categories = Object.fromEntries(
      catFeatures
        .filter((column) => !variableSettings[column]?.fixed)
        .map((column) => [column, uniqueValues(rows, column)]),
    );

    const fixedValues = Object.fromEntries(
      features
        .filter((column) => variableSettings[column]?.fixed)
        .map((column) => {
          const rawValue = variableSettings[column]?.fixedValue;
          if (numFeatures.includes(column)) return [column, Number(rawValue)];
          return [column, originalValue(uniqueValues(rows, column), rawValue)];
        }),
    );

    return {
      objectives: normalizedObjectives,
      bounds: numericBounds,
      categories,
      fixed_values: fixedValues,
    };
  }

  async function executeInverseAnalysis() {
    const errors = validateSettings();
    if (errors.length) {
      setSettingsError(errors.join("\n"));
      return;
    }
    setSettingsError("");
    setInverseAnalysisPayloadOverride(inversePayloadOverride());
    await runInverseAnalysis();
  }

  return (
    <>
      <SectionHeader
        step="7 · OPTIMIZE"
        title="目的条件を満たす入力候補を逆解析する"
        text="目的変数の条件と、説明変数の探索範囲・刻み・固定値を設定します。"
      />

      {!modelInfo && (
        <article className="panel">
          <p className="settings-note">先にModel画面で逆解析に使用するモデルを学習してください。</p>
        </article>
      )}

      <article className="panel optimize-target-panel">
        <div className="panel-title">
          <div>
            <span className="panel-kicker">1 · OBJECTIVES</span>
            <h3>目的変数の条件</h3>
            <p>最大化・最小化・目標値から選択します。値の入力は目標値を選んだ場合だけ表示します。</p>
          </div>
          <span className="status-chip success">{targets.length} targets</span>
        </div>
        <div className="table-wrap">
          <table className="inverse-objective-table">
            <thead>
              <tr>
                <th>目的変数</th>
                <th>タスク</th>
                <th>条件</th>
                <th>目標値</th>
              </tr>
            </thead>
            <tbody>
              {targets.map((target) => {
                const mode = objectiveMode(objectives[target], tasks[target]);
                const targetValues = uniqueValues(rows, target);
                return (
                  <tr key={target}>
                    <td className="inverse-name-cell"><strong>{target}</strong></td>
                    <td>
                      <span className={`status-chip ${tasks[target] === "classification" ? "categorical-chip" : ""}`}>
                        {tasks[target] === "classification" ? "分類" : "回帰"}
                      </span>
                    </td>
                    <td>
                      <select
                        value={mode}
                        onChange={(event) => changeObjective(target, event.target.value)}
                      >
                        <option value="max" disabled={tasks[target] === "classification"}>最大化</option>
                        <option value="min" disabled={tasks[target] === "classification"}>最小化</option>
                        <option value="target">目標値</option>
                      </select>
                    </td>
                    <td>
                      {mode === "target" ? (
                        tasks[target] === "classification" ? (
                          <select
                            value={String(objectives[target]?.value ?? "")}
                            onChange={(event) => changeTargetValue(target, event.target.value)}
                          >
                            <option value="">選択</option>
                            {targetValues.map((value) => (
                              <option key={String(value)} value={String(value)}>{String(value)}</option>
                            ))}
                          </select>
                        ) : (
                          <input
                            type="number"
                            step="any"
                            value={objectives[target]?.value ?? ""}
                            onChange={(event) => changeTargetValue(target, event.target.value)}
                          />
                        )
                      ) : (
                        <span className="muted-cell">入力不要</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        {targets.some((target) => tasks[target] === "classification") && (
          <p className="settings-note">
            分類では目標クラスを指定するため、「目標値」を使用します。
          </p>
        )}
      </article>

      <article className="panel optimize-variable-panel">
        <div className="panel-title">
          <div>
            <span className="panel-kicker">2 · SEARCH SPACE</span>
            <h3>説明変数の探索範囲</h3>
            <p>bochanの変数設定に合わせ、下限・上限・刻み・固定・固定値を同じ表で設定します。</p>
          </div>
          <div className="optimize-variable-summary">
            <span className="status-chip success">{searchedCount} search</span>
            <span className="status-chip">{fixedCount} fixed</span>
          </div>
        </div>
        <div className="table-wrap optimize-variable-table-wrap">
          <table className="optimize-variable-table">
            <thead>
              <tr>
                <th>変数</th>
                <th>型</th>
                <th>下限</th>
                <th>上限</th>
                <th>刻み</th>
                <th>固定</th>
                <th>固定値</th>
              </tr>
            </thead>
            <tbody>
              {features.map((column) => {
                const numeric = numFeatures.includes(column);
                const setting = variableSettings[column] || defaultVariableSetting(
                  rows,
                  column,
                  numeric,
                );
                const categories = numeric ? [] : uniqueValues(rows, column);
                return (
                  <tr key={column} className={setting.fixed ? "fixed-variable-row" : ""}>
                    <td className="inverse-name-cell"><strong>{column}</strong></td>
                    <td>
                      <span className={`status-chip ${numeric ? "" : "categorical-chip"}`}>
                        {numeric ? "numeric" : "categorical"}
                      </span>
                    </td>
                    <td>
                      {numeric ? (
                        <input
                          type="number"
                          step="any"
                          disabled={setting.fixed}
                          value={bounds[column]?.min ?? ""}
                          onChange={(event) => setBounds({
                            ...bounds,
                            [column]: {
                              ...(bounds[column] || {}),
                              min: event.target.value,
                            },
                          })}
                        />
                      ) : <span className="muted-cell">—</span>}
                    </td>
                    <td>
                      {numeric ? (
                        <input
                          type="number"
                          step="any"
                          disabled={setting.fixed}
                          value={bounds[column]?.max ?? ""}
                          onChange={(event) => setBounds({
                            ...bounds,
                            [column]: {
                              ...(bounds[column] || {}),
                              max: event.target.value,
                            },
                          })}
                        />
                      ) : <span className="muted-cell">—</span>}
                    </td>
                    <td>
                      {numeric ? (
                        <input
                          type="number"
                          min="0"
                          step="any"
                          disabled={setting.fixed}
                          value={setting.step ?? ""}
                          placeholder="任意"
                          onChange={(event) => patchVariable(column, {
                            step: event.target.value,
                          })}
                        />
                      ) : <span className="muted-cell">—</span>}
                    </td>
                    <td className="fixed-checkbox-cell">
                      <input
                        className="table-checkbox"
                        type="checkbox"
                        checked={Boolean(setting.fixed)}
                        onChange={(event) => patchVariable(column, {
                          fixed: event.target.checked,
                          fixedValue: setting.fixedValue,
                        })}
                        aria-label={`${column}を固定`}
                      />
                    </td>
                    <td>
                      {setting.fixed ? (
                        numeric ? (
                          <input
                            type="number"
                            step="any"
                            value={setting.fixedValue ?? ""}
                            onChange={(event) => patchVariable(column, {
                              fixedValue: event.target.value,
                            })}
                          />
                        ) : (
                          <select
                            value={String(setting.fixedValue ?? "")}
                            onChange={(event) => patchVariable(column, {
                              fixedValue: originalValue(categories, event.target.value),
                            })}
                          >
                            <option value="">選択</option>
                            {categories.map((value) => (
                              <option key={String(value)} value={String(value)}>{String(value)}</option>
                            ))}
                          </select>
                        )
                      ) : (
                        <span className="muted-cell">—</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <p className="settings-note">
          固定をONにした変数は探索せず、指定した固定値を`fixed_values`として逆解析へ渡します。
        </p>
      </article>

      <article className="panel optimize-run-panel">
        <div className="panel-title">
          <div>
            <span className="panel-kicker">3 · SEARCH</span>
            <h3>探索設定</h3>
            <p>探索アルゴリズム、試行回数、表示する候補数を設定します。</p>
          </div>
        </div>
        <div className="form-grid optimize-run-grid">
          <Field label="Sampler">
            <select value={sampler} onChange={(event) => setSampler(event.target.value)}>
              <option>TPE</option>
              <option>MOTPE</option>
              <option>CmaEs</option>
              <option>GP</option>
              <option>QMS</option>
              <option>NSGAII</option>
              <option>NSGAIII</option>
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
        {settingsError && (
          <p className="xai-error optimize-settings-error">{settingsError}</p>
        )}
        <button
          disabled={!modelInfo || busy}
          onClick={executeInverseAnalysis}
        >
          逆解析を実行 →
        </button>
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
