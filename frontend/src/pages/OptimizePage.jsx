import React, { useEffect, useMemo, useState } from "react";
import {
  setInverseAnalysisPayloadOverride,
} from "../api";
import DataTable from "../components/DataTable";
import { Field, SectionHeader } from "../components/Common";
import { uniqueValues } from "../data";
import { useWorkbench } from "../context/WorkbenchContext";
import "../optimize-variable-settings.css";

const SINGLE_OBJECTIVE_SAMPLERS = [
  {
    value: "TPE",
    label: "TPE",
    description: "標準的な単目的探索。カテゴリ変数を含む探索にも使いやすい手法です。",
  },
  {
    value: "CmaEs",
    label: "CMA-ES",
    description: "連続変数中心の単目的探索に向く進化戦略です。",
  },
  {
    value: "GP",
    label: "Gaussian Process",
    description: "少ない試行回数で効率的に単目的探索したい場合に適します。",
  },
  {
    value: "QMS",
    label: "QMC",
    description: "低偏差列で探索空間を広く均一にサンプリングします。",
  },
];

const MULTI_OBJECTIVE_SAMPLERS = [
  {
    value: "MOTPE",
    label: "MOTPE",
    description: "TPESamplerによる多目的探索です。比較的少ない試行回数でも利用できます。",
  },
  {
    value: "NSGAII",
    label: "NSGA-II",
    description: "多目的最適化で広く使われる進化計算です。",
  },
  {
    value: "NSGAIII",
    label: "NSGA-III",
    description: "目的数が多い場合の多目的探索に適した進化計算です。",
  },
];

const variableSettingsByModel = new Map();
const constraintSettingsByModel = new Map();
const objectiveSelectionByModel = new Map();

function isIntegerColumn(rows, column) {
  const values = rows.map((row) => row[column]).filter(Number.isFinite);
  return values.length > 0 && values.every(Number.isInteger);
}

function defaultVariableSetting(rows, column, numeric) {
  const firstValue = rows.find(
    (row) => row[column] !== null && row[column] !== undefined,
  )?.[column];
  return {
    fixed: false,
    fixedValue: firstValue ?? "",
    step: numeric && isIntegerColumn(rows, column) ? 1 : "",
  };
}

function normalizeConstraintSetting(saved, numericColumns) {
  return {
    enabled: Boolean(saved?.enabled),
    columns: (saved?.columns || []).filter((column) => numericColumns.includes(column)),
    value: saved?.value ?? "",
  };
}

function normalizeObjectiveSelection(saved, targets) {
  if (!Array.isArray(saved)) return [...targets];
  return targets.filter((target) => saved.includes(target));
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
  const numericFeatureKey = numFeatures.join("\u0000");
  const targetKey = targets.join("\u0000");

  const [selectedTargets, setSelectedTargets] = useState(() => (
    normalizeObjectiveSelection(objectiveSelectionByModel.get(modelKey), targets)
  ));
  const [variableSettings, setVariableSettings] = useState(() => {
    const saved = variableSettingsByModel.get(modelKey) || {};
    return Object.fromEntries(features.map((column) => [
      column,
      saved[column] || defaultVariableSetting(rows, column, numFeatures.includes(column)),
    ]));
  });
  const [sumConstraint, setSumConstraint] = useState(() => normalizeConstraintSetting(
    constraintSettingsByModel.get(modelKey),
    numFeatures,
  ));
  const [settingsError, setSettingsError] = useState("");

  const multiObjective = selectedTargets.length > 1;
  const samplerOptions = multiObjective
    ? MULTI_OBJECTIVE_SAMPLERS
    : SINGLE_OBJECTIVE_SAMPLERS;
  const selectedSampler = samplerOptions.find((option) => option.value === sampler)
    || samplerOptions[0];

  useEffect(() => {
    const saved = objectiveSelectionByModel.get(modelKey);
    setSelectedTargets(normalizeObjectiveSelection(saved, targets));
  }, [modelKey, targetKey]);

  useEffect(() => {
    const saved = variableSettingsByModel.get(modelKey) || {};
    setVariableSettings(Object.fromEntries(features.map((column) => [
      column,
      saved[column] || defaultVariableSetting(rows, column, numFeatures.includes(column)),
    ])));
  }, [modelKey, featureKey, numericFeatureKey]);

  useEffect(() => {
    const saved = constraintSettingsByModel.get(modelKey);
    setSumConstraint(normalizeConstraintSetting(saved, numFeatures));
  }, [modelKey, numericFeatureKey]);

  useEffect(() => {
    objectiveSelectionByModel.set(modelKey, selectedTargets);
  }, [modelKey, selectedTargets]);

  useEffect(() => {
    variableSettingsByModel.set(modelKey, variableSettings);
  }, [modelKey, variableSettings]);

  useEffect(() => {
    constraintSettingsByModel.set(modelKey, sumConstraint);
  }, [modelKey, sumConstraint]);

  useEffect(() => {
    if (!samplerOptions.some((option) => option.value === sampler)) {
      setSampler(samplerOptions[0].value);
    }
  }, [sampler, samplerOptions, setSampler]);

  useEffect(() => () => {
    setInverseAnalysisPayloadOverride(null);
  }, []);

  const fixedCount = useMemo(
    () => features.filter((column) => variableSettings[column]?.fixed).length,
    [features, variableSettings],
  );
  const searchedCount = features.length - fixedCount;
  const constraintRange = useMemo(() => {
    if (!sumConstraint.enabled || !sumConstraint.columns.length) return null;
    let minimum = 0;
    let maximum = 0;
    for (const column of sumConstraint.columns) {
      const setting = variableSettings[column] || {};
      if (setting.fixed) {
        const fixedValue = Number(setting.fixedValue);
        if (!Number.isFinite(fixedValue)) return null;
        minimum += fixedValue;
        maximum += fixedValue;
      } else {
        const lower = Number(bounds[column]?.min);
        const upper = Number(bounds[column]?.max);
        if (!Number.isFinite(lower) || !Number.isFinite(upper)) return null;
        minimum += lower;
        maximum += upper;
      }
    }
    return { minimum, maximum };
  }, [bounds, sumConstraint, variableSettings]);

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

  function toggleTarget(target) {
    setSelectedTargets((current) => (
      current.includes(target)
        ? current.filter((name) => name !== target)
        : targets.filter((name) => name === target || current.includes(name))
    ));
    setSettingsError("");
  }

  function toggleConstraintColumn(column) {
    setSumConstraint((current) => {
      const selected = current.columns.includes(column);
      return {
        ...current,
        columns: selected
          ? current.columns.filter((name) => name !== column)
          : [...current.columns, column],
      };
    });
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

    if (!selectedTargets.length) {
      errors.push("逆解析に使用する目的変数を1つ以上選択してください。");
    }

    selectedTargets.forEach((target) => {
      const mode = objectiveMode(objectives[target], tasks[target]);
      if (mode !== "target") return;
      const value = objectives[target]?.value;
      if (tasks[target] === "regression" && !Number.isFinite(Number(value))) {
        errors.push(`${target}: 目標値を数値で入力してください。`);
      }
      if (
        tasks[target] === "classification"
        && (value === "" || value === null || value === undefined)
      ) {
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
        if (
          !numeric
          && (setting.fixedValue === ""
            || setting.fixedValue === null
            || setting.fixedValue === undefined)
        ) {
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

    if (sumConstraint.enabled) {
      const constraintValue = Number(sumConstraint.value);
      if (!sumConstraint.columns.length) {
        errors.push("合計制約: 対象となる数値説明変数を1つ以上選択してください。");
      }
      if (!Number.isFinite(constraintValue)) {
        errors.push("合計制約: 合計値を数値で入力してください。");
      } else if (constraintRange) {
        const tolerance = 1.0e-9;
        if (
          constraintValue < constraintRange.minimum - tolerance
          || constraintValue > constraintRange.maximum + tolerance
        ) {
          errors.push(
            `合計制約: ${constraintValue}は現在の固定値・探索範囲で達成できません。`
            + ` 達成可能範囲は${constraintRange.minimum}〜${constraintRange.maximum}です。`,
          );
        }
      }
    }

    if (!samplerOptions.some((option) => option.value === sampler)) {
      errors.push(
        `${multiObjective ? "多目的" : "単目的"}探索に対応した探索手法を選択してください。`,
      );
    }
    if (searchedCount < 1) {
      errors.push("少なくとも1つの説明変数を探索対象にしてください。");
    }
    if (Number(topK) > Number(inverseTrials)) {
      errors.push("候補数はTrials以下にしてください。");
    }
    return errors;
  }

  function inversePayloadOverride() {
    const normalizedObjectives = selectedTargets.map((target) => {
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
          if (
            configuredStep !== ""
            && configuredStep !== null
            && configuredStep !== undefined
          ) {
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
      sum_constraint: sumConstraint.enabled
        ? {
            columns: sumConstraint.columns,
            value: Number(sumConstraint.value),
          }
        : null,
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

  const objectiveModeLabel = selectedTargets.length === 0
    ? "未選択"
    : multiObjective
      ? "多目的"
      : "単目的";

  return (
    <>
      <SectionHeader
        step="7 · OPTIMIZE"
        title="目的条件を満たす入力候補を逆解析する"
        text="使用する目的変数、目的条件、説明変数の探索範囲・固定値、合計制約、探索手法を設定します。"
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
            <h3>使用する目的変数と条件</h3>
            <p>逆解析に使う目的変数を選択し、最大化・最小化・目標値を設定します。</p>
          </div>
          <div className="optimize-objective-summary">
            <span className="status-chip success">
              {selectedTargets.length} / {targets.length} use
            </span>
            <button
              type="button"
              className="secondary compact-action"
              onClick={() => setSelectedTargets([...targets])}
            >
              すべて使用
            </button>
            <button
              type="button"
              className="secondary compact-action"
              onClick={() => setSelectedTargets([])}
            >
              選択解除
            </button>
          </div>
        </div>
        <div className="table-wrap">
          <table className="inverse-objective-table selectable-objective-table">
            <thead>
              <tr>
                <th>使用</th>
                <th>目的変数</th>
                <th>タスク</th>
                <th>条件</th>
                <th>目標値</th>
              </tr>
            </thead>
            <tbody>
              {targets.map((target) => {
                const selected = selectedTargets.includes(target);
                const mode = objectiveMode(objectives[target], tasks[target]);
                const targetValues = uniqueValues(rows, target);
                return (
                  <tr
                    key={target}
                    className={selected ? "selected-objective-row" : "excluded-objective-row"}
                  >
                    <td className="objective-checkbox-cell">
                      <input
                        className="table-checkbox"
                        type="checkbox"
                        checked={selected}
                        onChange={() => toggleTarget(target)}
                        aria-label={`${target}を逆解析に使用`}
                      />
                    </td>
                    <td className="inverse-name-cell">
                      <strong>{target}</strong>
                      {!selected && <span className="objective-excluded-label">対象外</span>}
                    </td>
                    <td>
                      <span
                        className={`status-chip ${tasks[target] === "classification" ? "categorical-chip" : ""}`}
                      >
                        {tasks[target] === "classification" ? "分類" : "回帰"}
                      </span>
                    </td>
                    <td>
                      <select
                        value={mode}
                        disabled={!selected}
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
                            disabled={!selected}
                            onChange={(event) => changeTargetValue(target, event.target.value)}
                          >
                            <option value="">選択</option>
                            {targetValues.map((value) => (
                              <option key={String(value)} value={String(value)}>
                                {String(value)}
                              </option>
                            ))}
                          </select>
                        ) : (
                          <input
                            type="number"
                            step="any"
                            disabled={!selected}
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
        <p className="settings-note">
          対象外にした目的変数の設定値は保持されますが、逆解析APIの`objectives`と元関数の`target_cols`には含まれません。
        </p>
        {selectedTargets.some((target) => tasks[target] === "classification") && (
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

      <article className="panel optimize-constraint-panel">
        <div className="panel-title">
          <div>
            <span className="panel-kicker">3 · CONSTRAINT</span>
            <h3>説明変数の合計制約</h3>
            <p>選択した数値説明変数の合計を、指定値と等しくする制約を設定します。</p>
          </div>
          <span className={`status-chip ${sumConstraint.enabled ? "success" : ""}`}>
            {sumConstraint.enabled ? "ON" : "OFF"}
          </span>
        </div>

        <label className={`constraint-toggle-card ${sumConstraint.enabled ? "active" : ""}`}>
          <input
            type="checkbox"
            checked={sumConstraint.enabled}
            onChange={(event) => setSumConstraint((current) => ({
              ...current,
              enabled: event.target.checked,
            }))}
          />
          <span>
            <strong>合計制約を使用する</strong>
            <small>例: 原料1 + 原料2 + 原料3 = 100</small>
          </span>
        </label>

        {sumConstraint.enabled && (
          <div className="constraint-settings-body">
            <div className="constraint-value-row">
              <Field label="合計値">
                <input
                  type="number"
                  step="any"
                  value={sumConstraint.value}
                  onChange={(event) => setSumConstraint((current) => ({
                    ...current,
                    value: event.target.value,
                  }))}
                />
              </Field>
              <div className="constraint-range-summary">
                <span>現在の達成可能範囲</span>
                <strong>
                  {constraintRange
                    ? `${constraintRange.minimum} ～ ${constraintRange.maximum}`
                    : "対象変数を選択してください"}
                </strong>
              </div>
              <div className="constraint-bulk-actions">
                <button
                  type="button"
                  className="secondary"
                  onClick={() => setSumConstraint((current) => ({
                    ...current,
                    columns: [...numFeatures],
                  }))}
                >
                  数値変数を全選択
                </button>
                <button
                  type="button"
                  className="secondary"
                  onClick={() => setSumConstraint((current) => ({
                    ...current,
                    columns: [],
                  }))}
                >
                  選択解除
                </button>
              </div>
            </div>

            <div className="constraint-variable-grid">
              {numFeatures.map((column) => {
                const selected = sumConstraint.columns.includes(column);
                const setting = variableSettings[column] || {};
                const rangeText = setting.fixed
                  ? `固定: ${setting.fixedValue ?? "—"}`
                  : `${bounds[column]?.min ?? "—"} ～ ${bounds[column]?.max ?? "—"}`;
                return (
                  <label
                    key={column}
                    className={`constraint-variable-card ${selected ? "selected" : ""}`}
                  >
                    <input
                      type="checkbox"
                      checked={selected}
                      onChange={() => toggleConstraintColumn(column)}
                    />
                    <span>
                      <strong>{column}</strong>
                      <small>{rangeText}</small>
                    </span>
                  </label>
                );
              })}
            </div>
            <p className="settings-note">
              固定済みの数値変数も合計に含められます。固定値を差し引いた残りを、探索対象変数の範囲内で調整します。
            </p>
          </div>
        )}
      </article>

      <article className="panel optimize-run-panel">
        <div className="panel-title">
          <div>
            <span className="panel-kicker">4 · SEARCH</span>
            <h3>探索設定</h3>
            <p>使用する目的変数の数に対応した探索アルゴリズム、試行回数、候補数を設定します。</p>
          </div>
          <span className="status-chip success">
            {selectedTargets.length === 0
              ? "目的変数未選択"
              : multiObjective
                ? `多目的 (${selectedTargets.length})`
                : "単目的"}
          </span>
        </div>
        <div className="form-grid optimize-run-grid">
          <Field label={`探索手法（${objectiveModeLabel}）`}>
            <select
              value={selectedSampler.value}
              disabled={selectedTargets.length === 0}
              onChange={(event) => setSampler(event.target.value)}
            >
              {samplerOptions.map((option) => (
                <option key={option.value} value={option.value}>{option.label}</option>
              ))}
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
        <div className="sampler-description">
          <strong>{selectedTargets.length ? selectedSampler.label : "目的変数を選択してください"}</strong>
          <span>
            {selectedTargets.length
              ? selectedSampler.description
              : "目的変数の使用チェックに応じて、単目的または多目的の探索手法を表示します。"}
          </span>
        </div>
        {settingsError && (
          <p className="xai-error optimize-settings-error">{settingsError}</p>
        )}
        <button
          disabled={!modelInfo || busy || selectedTargets.length === 0}
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
            <span className="status-chip success">
              {inverseResult.candidates.length} candidates
            </span>
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
