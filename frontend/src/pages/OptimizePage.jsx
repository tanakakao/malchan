import React, { useEffect, useMemo, useState } from "react";
import {
  setInverseAnalysisPayloadOverride,
} from "../api";
import DataTable from "../components/DataTable";
import { Field } from "../components/Common";
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

function OptimizeSectionTitle({ symbol, title, description }) {
  return (
    <div className="optimize-section-title">
      <span className="optimize-section-symbol" aria-hidden="true">{symbol}</span>
      <div>
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

function OptimizeSummary({ symbol, label, value, status = false }) {
  return (
    <div className="optimize-summary-item">
      <span className="optimize-summary-symbol" aria-hidden="true">{symbol}</span>
      <div>
        <span>{label}</span>
        {status ? value : <strong>{value}</strong>}
      </div>
    </div>
  );
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

  const readyForRun = Boolean(modelInfo && selectedTargets.length > 0 && searchedCount > 0);
  const statusLabel = busy ? "実行中" : readyForRun ? "準備完了" : "設定待ち";
  const statusClass = busy ? "running" : readyForRun ? "ready" : "waiting";
  const objectiveModeLabel = selectedTargets.length === 0
    ? "未選択"
    : multiObjective
      ? "多目的"
      : "単目的";

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
    const currentMode = objectiveMode(current, task);
    const defaultTarget = task === "classification"
      ? values[0] ?? ""
      : rows.find((row) => Number.isFinite(row[target]))?.[target] ?? 0;
    setObjectives({
      ...objectives,
      [target]: {
        mode: nextMode,
        direction: nextMode === "target"
          ? current.direction || (currentMode === "min" ? "min" : "max")
          : nextMode,
        value: nextMode === "target"
          ? current.mode === "target"
            ? current.value
            : defaultTarget
          : nextMode,
      },
    });
  }

  function changeObjectiveConstraint(target, nextConstraint) {
    if (tasks[target] === "classification") {
      changeObjective(target, "target");
      return;
    }
    if (nextConstraint === "target") {
      changeObjective(target, "target");
      return;
    }
    const previousDirection = objectives[target]?.direction;
    changeObjective(target, previousDirection === "min" ? "min" : "max");
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
      document.getElementById("optimize-search-settings")?.scrollIntoView({
        behavior: "smooth",
        block: "center",
      });
      return;
    }
    setSettingsError("");
    setInverseAnalysisPayloadOverride(inversePayloadOverride());
    await runInverseAnalysis();
  }

  function showDetailedSettings() {
    document.getElementById("optimize-search-settings")?.scrollIntoView({
      behavior: "smooth",
      block: "start",
    });
  }

  return (
    <div className="optimize-page">
      <header className="optimize-hero">
        <div className="optimize-hero-copy">
          <span className="optimize-hero-kicker">STEP 7 · OPTIMIZE</span>
          <h2>Optimize</h2>
          <p>最適化の目的（目的変数）と探索条件（変数の範囲や制約）を設定します。</p>
        </div>
        <div className="optimize-hero-tools">
          <div className="optimize-summary-strip">
            <OptimizeSummary symbol="◎" label="目的変数" value={`${selectedTargets.length} 件`} />
            <OptimizeSummary symbol="☷" label="探索変数" value={`${searchedCount} 件`} />
            <OptimizeSummary
              symbol="✓"
              label="設定ステータス"
              status
              value={<strong className={`optimize-status-badge ${statusClass}`}>{statusLabel}</strong>}
            />
          </div>
          <div className="optimize-hero-run-actions">
            <button
              type="button"
              className="optimize-primary-run"
              disabled={!modelInfo || busy || selectedTargets.length === 0}
              onClick={executeInverseAnalysis}
            >
              <span aria-hidden="true">▶</span>
              {busy ? "逆解析を実行中" : "逆解析を実行"}
            </button>
            <button
              type="button"
              className="optimize-detail-button"
              onClick={showDetailedSettings}
            >
              詳細設定 <span aria-hidden="true">›</span>
            </button>
          </div>
        </div>
      </header>

      {!modelInfo && (
        <article className="panel optimize-model-warning">
          <span aria-hidden="true">!</span>
          <p>先にModel画面で逆解析に使用するモデルを学習してください。</p>
        </article>
      )}

      <article className="panel optimize-section-card optimize-target-panel">
        <div className="optimize-card-header">
          <OptimizeSectionTitle
            symbol="◎"
            title="目的変数"
            description="最適化の目標となる出力と条件を設定します。"
          />
          <div className="optimize-card-actions">
            <span className="optimize-count-badge">
              {selectedTargets.length} / {targets.length} 使用
            </span>
            <button
              type="button"
              className="optimize-outline-button"
              onClick={() => setSelectedTargets([...targets])}
            >
              すべて使用
            </button>
            <button
              type="button"
              className="optimize-outline-button"
              onClick={() => setSelectedTargets([])}
            >
              選択解除
            </button>
          </div>
        </div>

        <div className="table-wrap optimize-table-wrap objective-table-wrap">
          <table className="inverse-objective-table bochan-objective-table">
            <thead>
              <tr>
                <th>目的変数</th>
                <th>最適化対象</th>
                <th>方向</th>
                <th>制約</th>
                <th>しきい値 / 目標値</th>
                <th>対象クラス</th>
              </tr>
            </thead>
            <tbody>
              {targets.map((target) => {
                const selected = selectedTargets.includes(target);
                const mode = objectiveMode(objectives[target], tasks[target]);
                const targetValues = uniqueValues(rows, target);
                const classification = tasks[target] === "classification";
                const direction = mode === "target"
                  ? objectives[target]?.direction || "max"
                  : mode;
                return (
                  <tr
                    key={target}
                    className={selected ? "selected-objective-row" : "excluded-objective-row"}
                  >
                    <td className="inverse-name-cell optimize-name-cell">
                      <strong>{target}</strong>
                      <span className={`optimize-type-badge ${classification ? "classification" : "regression"}`}>
                        {classification ? "classification" : "regression"}
                      </span>
                    </td>
                    <td className="objective-checkbox-cell">
                      <input
                        className="table-checkbox optimize-checkbox"
                        type="checkbox"
                        checked={selected}
                        onChange={() => toggleTarget(target)}
                        aria-label={`${target}を逆解析に使用`}
                      />
                    </td>
                    <td>
                      {classification ? (
                        <span className="optimize-empty-value">—</span>
                      ) : (
                        <select
                          value={direction}
                          disabled={!selected || mode === "target"}
                          onChange={(event) => changeObjective(target, event.target.value)}
                        >
                          <option value="max">最大化</option>
                          <option value="min">最小化</option>
                        </select>
                      )}
                    </td>
                    <td>
                      <select
                        value={mode === "target" ? "target" : "none"}
                        disabled={!selected || classification}
                        onChange={(event) => changeObjectiveConstraint(target, event.target.value)}
                      >
                        <option value="none">なし</option>
                        <option value="target">目標値</option>
                      </select>
                    </td>
                    <td>
                      {mode === "target" && !classification ? (
                        <input
                          type="number"
                          step="any"
                          disabled={!selected}
                          value={objectives[target]?.value ?? ""}
                          onChange={(event) => changeTargetValue(target, event.target.value)}
                        />
                      ) : (
                        <span className="muted-cell">
                          {classification ? "対象クラスを指定" : "制約なし"}
                        </span>
                      )}
                    </td>
                    <td>
                      {classification ? (
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
                        <span className="optimize-empty-value">—</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <p className="optimize-card-note">
          対象外にした目的変数の設定は保持されますが、逆解析の対象には含まれません。
        </p>
      </article>

      <article className="panel optimize-section-card optimize-variable-panel">
        <div className="optimize-card-header">
          <OptimizeSectionTitle
            symbol="☷"
            title="探索変数（検索空間）"
            description="探索に使用する入力変数の範囲や固定条件を設定します。"
          />
          <div className="optimize-card-actions">
            <span className="optimize-count-badge success">{searchedCount} 探索</span>
            <span className="optimize-count-badge">{fixedCount} 固定</span>
          </div>
        </div>

        <div className="table-wrap optimize-table-wrap optimize-variable-table-wrap">
          <table className="optimize-variable-table bochan-variable-table">
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
                    <td className="inverse-name-cell optimize-name-cell"><strong>{column}</strong></td>
                    <td>
                      <span className={`optimize-type-badge ${numeric ? "numeric" : "categorical"}`}>
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
                      ) : <span className="optimize-empty-value">—</span>}
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
                      ) : <span className="optimize-empty-value">—</span>}
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
                      ) : <span className="optimize-empty-value">—</span>}
                    </td>
                    <td className="fixed-checkbox-cell">
                      <input
                        className="table-checkbox optimize-checkbox"
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
                        <span className="optimize-empty-value">—</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <p className="optimize-card-note">
          固定をONにした変数は探索せず、指定した固定値を逆解析へ渡します。
        </p>
      </article>

      <article className="panel optimize-section-card optimize-constraint-panel">
        <div className="optimize-card-header">
          <OptimizeSectionTitle
            symbol="Σ"
            title="合計制約"
            description="選択した数値説明変数の合計を指定値と等しくします。"
          />
          <span className={`optimize-count-badge ${sumConstraint.enabled ? "success" : ""}`}>
            {sumConstraint.enabled ? "使用中" : "未使用"}
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
                  className="optimize-outline-button"
                  onClick={() => setSumConstraint((current) => ({
                    ...current,
                    columns: [...numFeatures],
                  }))}
                >
                  数値変数を全選択
                </button>
                <button
                  type="button"
                  className="optimize-outline-button"
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
            <p className="optimize-card-note">
              固定済みの数値変数も合計に含められます。固定値を差し引いた残りを探索変数で調整します。
            </p>
          </div>
        )}
      </article>

      <article
        id="optimize-search-settings"
        className="panel optimize-section-card optimize-run-panel"
      >
        <div className="optimize-card-header">
          <OptimizeSectionTitle
            symbol="⚙"
            title="探索設定"
            description="目的数に対応する探索手法、試行回数、候補数を設定します。"
          />
          <span className="optimize-count-badge success">
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
          <span className="sampler-description-symbol" aria-hidden="true">i</span>
          <div>
            <strong>{selectedTargets.length ? selectedSampler.label : "目的変数を選択してください"}</strong>
            <span>
              {selectedTargets.length
                ? selectedSampler.description
                : "目的変数の選択数に応じて、単目的または多目的の探索手法を表示します。"}
            </span>
          </div>
        </div>

        {settingsError && (
          <p className="xai-error optimize-settings-error">{settingsError}</p>
        )}

        <div className="optimize-bottom-action">
          <div>
            <strong>設定内容で逆解析を開始</strong>
            <span>選択した目的条件と探索空間を使用して候補を生成します。</span>
          </div>
          <button
            type="button"
            className="optimize-primary-run"
            disabled={!modelInfo || busy || selectedTargets.length === 0}
            onClick={executeInverseAnalysis}
          >
            <span aria-hidden="true">▶</span>
            {busy ? "実行中" : "逆解析を実行"}
          </button>
        </div>
      </article>

      {inverseResult && (
        <article className="panel optimize-section-card optimize-result-panel">
          <div className="optimize-card-header">
            <OptimizeSectionTitle
              symbol="✓"
              title="逆解析候補"
              description="目的条件に対して評価された候補を表示します。"
            />
            <span className="optimize-count-badge success">
              {inverseResult.candidates.length} candidates
            </span>
          </div>
          <DataTable
            rows={inverseResult.candidates}
            columns={Object.keys(inverseResult.candidates[0] || {})}
          />
        </article>
      )}
    </div>
  );
}
