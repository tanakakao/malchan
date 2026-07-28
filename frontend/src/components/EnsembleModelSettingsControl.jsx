import React, { useEffect, useMemo, useState } from "react";
import { createPortal } from "react-dom";
import { api, setEnsembleTrainingOptions } from "../api";
import { CheckboxList, Field } from "./Common";
import { modelsFor, useWorkbench } from "../context/WorkbenchContext";
import "../model-ensemble.css";

const ENSEMBLE_TYPES = [
  ["アンサンブル", "Voting", "複数モデルの予測を平均・多数決で統合します。"],
  ["スタッキング", "Stacking", "複数モデルの予測を最終モデルで統合します。"],
  ["バギング", "Bagging", "1つのベースモデルを複数の再標本データで学習します。"],
  ["ブースティング", "Boosting", "1つのベースモデルを逐次的に補正しながら学習します。"],
];

const PARAMETER_MODES = [
  ["tuning", "Optunaでチューニング"],
  ["default", "各モデルの既定値"],
  ["manual", "各モデルで設定"],
];

function requiresMultipleModels(ensembleType) {
  return ensembleType === "アンサンブル" || ensembleType === "スタッキング";
}

function usesBaseModelParameters(ensembleType) {
  return ["スタッキング", "バギング", "ブースティング"].includes(ensembleType);
}

function uniqueValues(values) {
  return [...new Set(values.filter(Boolean))];
}

function defaultMembers(task, selectedModel, candidates, multiple) {
  const allowed = modelsFor(task);
  const ordered = uniqueValues([selectedModel, ...(candidates || []), ...allowed])
    .filter((model) => allowed.includes(model));
  return ordered.slice(0, multiple ? 2 : 1);
}

function findSelectionContent() {
  return [...document.querySelectorAll(".model-workflow-content")]
    .find((content) => content.textContent?.includes("パラメータ設定方法")) || null;
}

function schemaKey(task, model) {
  return `${task}\u0001${model}`;
}

function editableParameters(schema) {
  return (schema?.parameters || []).filter((parameter) => parameter.editable);
}

function defaultParameters(schema) {
  return Object.fromEntries(
    editableParameters(schema).map((parameter) => [parameter.name, parameter.default_value]),
  );
}

function valuesEqual(left, right) {
  return JSON.stringify(left) === JSON.stringify(right);
}

function displayValue(value) {
  if (Array.isArray(value)) return value.join(" × ");
  if (value === null) return "None";
  if (value === true) return "有効";
  if (value === false) return "無効";
  return String(value);
}

function normalizeNumericValue(parameter, rawValue) {
  let value = Number(rawValue);
  if (!Number.isFinite(value)) value = Number(parameter.default_value ?? parameter.low ?? 0);
  if (parameter.control === "integer") value = Math.round(value);
  if (parameter.low !== null && parameter.low !== undefined) {
    value = Math.max(Number(parameter.low), value);
  }
  if (parameter.high !== null && parameter.high !== undefined) {
    value = Math.min(Number(parameter.high), value);
  }
  return value;
}

function CompactParameterControl({ parameter, value, onChange }) {
  const label = parameter.label || parameter.name;

  if (parameter.control === "boolean") {
    return (
      <label className="ensemble-parameter-control ensemble-parameter-boolean">
        <span>
          <strong>{label}</strong>
          <small>{parameter.name}</small>
        </span>
        <input
          type="checkbox"
          checked={Boolean(value)}
          onChange={(event) => onChange(event.target.checked)}
        />
      </label>
    );
  }

  if (parameter.control === "categorical") {
    const choices = parameter.choices || [];
    const selectedIndex = Math.max(
      0,
      choices.findIndex((choice) => valuesEqual(choice, value)),
    );
    return (
      <label className="ensemble-parameter-control">
        <span>
          <strong>{label}</strong>
          <small>{parameter.name}</small>
        </span>
        <select
          value={selectedIndex}
          onChange={(event) => onChange(choices[Number(event.target.value)])}
        >
          {choices.map((choice, index) => (
            <option key={`${parameter.name}-${index}`} value={index}>
              {displayValue(choice)}
            </option>
          ))}
        </select>
      </label>
    );
  }

  return (
    <label className="ensemble-parameter-control">
      <span>
        <strong>{label}</strong>
        <small>{parameter.name}</small>
      </span>
      <input
        type="number"
        min={parameter.low ?? undefined}
        max={parameter.high ?? undefined}
        step={parameter.step ?? "any"}
        value={normalizeNumericValue(parameter, value)}
        onChange={(event) => onChange(normalizeNumericValue(parameter, event.target.value))}
      />
    </label>
  );
}

function EnsembleParameterEditor({
  tab,
  schema,
  values,
  loading,
  error,
  onChange,
  onReset,
}) {
  const parameters = editableParameters(schema);

  return (
    <div className="ensemble-parameter-panel" role="tabpanel">
      <div className="ensemble-parameter-panel-head">
        <div>
          <strong>{tab?.model || "モデルを選択してください"}</strong>
          <span>{tab?.role === "final" ? "Stackingの最終モデル" : "アンサンブル構成モデル"}</span>
        </div>
        {parameters.length > 0 && (
          <button type="button" className="secondary compact-button" onClick={onReset}>
            既定値へ戻す
          </button>
        )}
      </div>

      {loading && !schema && <p className="settings-note">パラメータ定義を取得しています...</p>}
      {error && <p className="xai-error ensemble-parameter-error">{error}</p>}
      {!loading && !error && schema && parameters.length === 0 && (
        <p className="settings-note">このモデルにはWeb画面で変更できるパラメータがありません。</p>
      )}
      {parameters.length > 0 && (
        <div className="ensemble-parameter-scroll">
          <div className="ensemble-parameter-grid">
            {parameters.map((parameter) => (
              <CompactParameterControl
                key={parameter.name}
                parameter={parameter}
                value={values?.[parameter.name] ?? parameter.default_value}
                onChange={(nextValue) => onChange(parameter.name, nextValue)}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default function EnsembleModelSettingsControl() {
  const {
    step,
    targets,
    tasks,
    modelNames,
    candidates,
  } = useWorkbench();
  const [host, setHost] = useState(null);
  const [enabled, setEnabled] = useState(false);
  const [ensembleType, setEnsembleType] = useState("アンサンブル");
  const [activeTarget, setActiveTarget] = useState("");
  const [membersByTarget, setMembersByTarget] = useState({});
  const [stackingBaseModel, setStackingBaseModel] = useState("");
  const [parameterMode, setParameterMode] = useState("tuning");
  const [activeParameterTab, setActiveParameterTab] = useState("");
  const [parameterSchemas, setParameterSchemas] = useState({});
  const [memberParamsByTarget, setMemberParamsByTarget] = useState({});
  const [baseParamsByTarget, setBaseParamsByTarget] = useState({});
  const [parameterLoading, setParameterLoading] = useState(false);
  const [parameterError, setParameterError] = useState("");

  const currentTarget = targets.includes(activeTarget) ? activeTarget : targets[0] || "";
  const taskKinds = useMemo(
    () => new Set(targets.map((target) => tasks[target]).filter(Boolean)),
    [targets, tasks],
  );
  const mixedTasks = taskKinds.size > 1;
  const stackingModels = taskKinds.size === 1 && targets.length
    ? modelsFor(tasks[targets[0]])
    : [];

  const parameterEntries = useMemo(() => {
    if (parameterMode !== "manual") return [];
    const entries = [];
    targets.forEach((target) => {
      (membersByTarget[target] || []).forEach((model) => {
        entries.push({
          key: `member:${target}:${model}`,
          target,
          task: tasks[target],
          model,
          role: "member",
        });
      });
      if (ensembleType === "スタッキング" && stackingBaseModel) {
        entries.push({
          key: `final:${target}:${stackingBaseModel}`,
          target,
          task: tasks[target],
          model: stackingBaseModel,
          role: "final",
        });
      }
    });
    return entries;
  }, [parameterMode, targets, tasks, membersByTarget, ensembleType, stackingBaseModel]);

  const parameterTabs = useMemo(() => {
    if (!currentTarget || parameterMode !== "manual") return [];
    const memberTabs = (membersByTarget[currentTarget] || []).map((model) => ({
      key: `member:${currentTarget}:${model}`,
      target: currentTarget,
      task: tasks[currentTarget],
      model,
      role: "member",
      label: requiresMultipleModels(ensembleType) ? model : `ベース: ${model}`,
    }));
    if (ensembleType === "スタッキング" && stackingBaseModel) {
      memberTabs.push({
        key: `final:${currentTarget}:${stackingBaseModel}`,
        target: currentTarget,
        task: tasks[currentTarget],
        model: stackingBaseModel,
        role: "final",
        label: `最終: ${stackingBaseModel}`,
      });
    }
    return memberTabs;
  }, [currentTarget, parameterMode, membersByTarget, tasks, ensembleType, stackingBaseModel]);

  const selectedParameterTab = parameterTabs.find((tab) => tab.key === activeParameterTab)
    || parameterTabs[0]
    || null;
  const selectedSchema = selectedParameterTab
    ? parameterSchemas[schemaKey(selectedParameterTab.task, selectedParameterTab.model)]
    : null;
  const selectedParameterValues = selectedParameterTab?.role === "final"
    ? baseParamsByTarget[selectedParameterTab.target]
    : memberParamsByTarget[selectedParameterTab?.target]?.[selectedParameterTab?.model];

  useEffect(() => {
    setActiveTarget((current) => (targets.includes(current) ? current : targets[0] || ""));
  }, [targets]);

  useEffect(() => {
    if (mixedTasks && ensembleType === "スタッキング") {
      setEnsembleType("アンサンブル");
    }
  }, [mixedTasks, ensembleType]);

  useEffect(() => {
    const multiple = requiresMultipleModels(ensembleType);
    setMembersByTarget((current) => Object.fromEntries(
      targets.map((target) => {
        const allowed = modelsFor(tasks[target]);
        const previous = (current[target] || []).filter((model) => allowed.includes(model));
        const fallback = defaultMembers(
          tasks[target],
          modelNames[target],
          candidates[target],
          multiple,
        );
        let selected = previous.length ? previous : fallback;
        if (multiple && selected.length < 2) {
          selected = uniqueValues([...selected, ...fallback, ...allowed]).slice(0, 2);
        }
        if (!multiple) selected = selected.slice(0, 1);
        return [target, selected];
      }),
    ));
  }, [targets, tasks, modelNames, candidates, ensembleType]);

  useEffect(() => {
    if (!stackingModels.length) {
      setStackingBaseModel("");
      return;
    }
    setStackingBaseModel((current) => (
      stackingModels.includes(current)
        ? current
        : modelNames[targets[0]] || stackingModels[0]
    ));
  }, [stackingModels.join("\u0001"), modelNames, targets]);

  useEffect(() => {
    setActiveParameterTab((current) => (
      parameterTabs.some((tab) => tab.key === current) ? current : parameterTabs[0]?.key || ""
    ));
  }, [parameterTabs]);

  useEffect(() => {
    if (parameterMode !== "manual" || !parameterEntries.length) {
      setParameterLoading(false);
      setParameterError("");
      return undefined;
    }

    const missing = [...new Map(
      parameterEntries
        .filter((entry) => !parameterSchemas[schemaKey(entry.task, entry.model)])
        .map((entry) => [schemaKey(entry.task, entry.model), entry]),
    ).values()];
    if (!missing.length) return undefined;

    let cancelled = false;
    setParameterLoading(true);
    setParameterError("");
    Promise.all(
      missing.map(async (entry) => [
        schemaKey(entry.task, entry.model),
        await api.modelParameters(entry.task, entry.model),
      ]),
    )
      .then((items) => {
        if (!cancelled) {
          setParameterSchemas((current) => ({ ...current, ...Object.fromEntries(items) }));
        }
      })
      .catch((error) => {
        if (!cancelled) setParameterError(error.message || String(error));
      })
      .finally(() => {
        if (!cancelled) setParameterLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [parameterMode, parameterEntries, parameterSchemas]);

  useEffect(() => {
    if (parameterMode !== "manual" || !parameterEntries.length) return;

    setMemberParamsByTarget((current) => {
      let changed = false;
      const next = { ...current };
      parameterEntries.filter((entry) => entry.role === "member").forEach((entry) => {
        const schema = parameterSchemas[schemaKey(entry.task, entry.model)];
        if (!schema) return;
        const targetValues = next[entry.target] || {};
        if (!Object.prototype.hasOwnProperty.call(targetValues, entry.model)) {
          next[entry.target] = {
            ...targetValues,
            [entry.model]: defaultParameters(schema),
          };
          changed = true;
        }
      });
      return changed ? next : current;
    });

    setBaseParamsByTarget((current) => {
      let changed = false;
      const next = { ...current };
      parameterEntries.filter((entry) => entry.role === "final").forEach((entry) => {
        const schema = parameterSchemas[schemaKey(entry.task, entry.model)];
        if (!schema || Object.prototype.hasOwnProperty.call(next, entry.target)) return;
        next[entry.target] = defaultParameters(schema);
        changed = true;
      });
      return changed ? next : current;
    });
  }, [parameterMode, parameterEntries, parameterSchemas]);

  useEffect(() => {
    if (!enabled || step !== "model" || !host) {
      setEnsembleTrainingOptions(null);
      return undefined;
    }
    setEnsembleTrainingOptions({
      ensembleType,
      baseModel: ensembleType === "スタッキング" ? stackingBaseModel : null,
      membersByTarget,
      parameterMode,
      memberParamsByTarget,
      baseParamsByTarget,
    });
    return () => setEnsembleTrainingOptions(null);
  }, [
    enabled,
    step,
    host,
    ensembleType,
    stackingBaseModel,
    membersByTarget,
    parameterMode,
    memberParamsByTarget,
    baseParamsByTarget,
  ]);

  useEffect(() => {
    if (step !== "model") {
      setHost(null);
      return undefined;
    }

    const contentRoot = document.querySelector(".content-inner") || document.body;
    let frameId = null;
    let disposed = false;

    const resolveHost = () => {
      if (disposed) return;
      const content = findSelectionContent();
      if (!content) {
        setHost(null);
        return;
      }
      let nextHost = content.querySelector(":scope > .ensemble-model-settings-host");
      if (!nextHost) {
        nextHost = document.createElement("div");
        nextHost.className = "ensemble-model-settings-host";
        content.prepend(nextHost);
      }
      setHost(nextHost);
    };

    const scheduleResolve = () => {
      if (frameId !== null) window.cancelAnimationFrame(frameId);
      frameId = window.requestAnimationFrame(() => {
        frameId = null;
        resolveHost();
      });
    };

    scheduleResolve();
    contentRoot.addEventListener("click", scheduleResolve);
    return () => {
      disposed = true;
      contentRoot.removeEventListener("click", scheduleResolve);
      if (frameId !== null) window.cancelAnimationFrame(frameId);
      setHost((current) => {
        if (current?.isConnected) current.remove();
        return null;
      });
    };
  }, [step, targets.length]);

  useEffect(() => {
    const panel = host?.closest(".model-selection-panel");
    if (!panel) return undefined;
    panel.classList.toggle("ensemble-active", enabled);
    const summary = panel.querySelector(".model-run-summary small");
    if (summary) {
      const modeLabel = PARAMETER_MODES.find(([value]) => value === parameterMode)?.[1] || parameterMode;
      summary.dataset.ensembleSummary = `${ENSEMBLE_TYPES.find(([value]) => value === ensembleType)?.[1] || ensembleType} · ${modeLabel}`;
    }
    return () => {
      panel.classList.remove("ensemble-active");
      if (summary) delete summary.dataset.ensembleSummary;
    };
  }, [host, enabled, ensembleType, parameterMode]);

  function updateParameter(tab, name, value) {
    if (!tab) return;
    if (tab.role === "final") {
      setBaseParamsByTarget((current) => ({
        ...current,
        [tab.target]: {
          ...(current[tab.target] || {}),
          [name]: value,
        },
      }));
      return;
    }
    setMemberParamsByTarget((current) => ({
      ...current,
      [tab.target]: {
        ...(current[tab.target] || {}),
        [tab.model]: {
          ...(current[tab.target]?.[tab.model] || {}),
          [name]: value,
        },
      },
    }));
  }

  function resetSelectedParameters() {
    if (!selectedParameterTab || !selectedSchema) return;
    const defaults = defaultParameters(selectedSchema);
    if (selectedParameterTab.role === "final") {
      setBaseParamsByTarget((current) => ({
        ...current,
        [selectedParameterTab.target]: defaults,
      }));
      return;
    }
    setMemberParamsByTarget((current) => ({
      ...current,
      [selectedParameterTab.target]: {
        ...(current[selectedParameterTab.target] || {}),
        [selectedParameterTab.model]: defaults,
      },
    }));
  }

  if (!host) return null;

  const multiple = requiresMultipleModels(ensembleType);
  const currentMembers = membersByTarget[currentTarget] || [];
  const currentModels = currentTarget ? modelsFor(tasks[currentTarget]) : [];
  const selectedType = ENSEMBLE_TYPES.find(([value]) => value === ensembleType);

  return createPortal(
    <section className="model-setting-section ensemble-model-settings">
      <div className="model-setting-heading ensemble-setting-heading">
        <div>
          <strong>アンサンブル学習</strong>
          <span>複数モデルの統合、スタッキング、バギング、ブースティングを設定します。</span>
        </div>
        <label className="switch-label">
          <input
            type="checkbox"
            checked={enabled}
            onChange={(event) => setEnabled(event.target.checked)}
          />
          <span />
          使用する
        </label>
      </div>

      {enabled && (
        <div className="ensemble-settings-body">
          <div className="ensemble-top-fields">
            <Field label="アンサンブル方式">
              <select
                value={ensembleType}
                onChange={(event) => setEnsembleType(event.target.value)}
              >
                {ENSEMBLE_TYPES.map(([value, label]) => (
                  <option
                    key={value}
                    value={value}
                    disabled={value === "スタッキング" && mixedTasks}
                  >
                    {label}（{value}）
                  </option>
                ))}
              </select>
            </Field>
            <Field label="パラメータ設定">
              <select value={parameterMode} onChange={(event) => setParameterMode(event.target.value)}>
                {PARAMETER_MODES.map(([value, label]) => (
                  <option key={value} value={value}>{label}</option>
                ))}
              </select>
            </Field>
            {ensembleType === "スタッキング" && (
              <Field label="最終モデル">
                <select
                  value={stackingBaseModel}
                  onChange={(event) => setStackingBaseModel(event.target.value)}
                >
                  {stackingModels.map((model) => <option key={model}>{model}</option>)}
                </select>
              </Field>
            )}
          </div>

          <p className="settings-note ensemble-description">
            {selectedType?.[2]}
            {mixedTasks && " 回帰と分類が混在しているため、共通の最終モデルが必要なStackingは選択できません。"}
          </p>

          {targets.length > 1 && (
            <div className="model-target-tabs ensemble-target-tabs" role="tablist" aria-label="アンサンブル対象の切り替え">
              {targets.map((target) => (
                <button
                  key={target}
                  type="button"
                  role="tab"
                  aria-selected={target === currentTarget}
                  className={target === currentTarget ? "active" : ""}
                  onClick={() => setActiveTarget(target)}
                >
                  <strong>{target}</strong>
                  <span>
                    {tasks[target] === "classification" ? "分類" : "回帰"}
                    {` · ${membersByTarget[target]?.length || 0} models`}
                  </span>
                </button>
              ))}
            </div>
          )}

          {currentTarget && (
            <section className="target-model-card ensemble-target-card">
              <div className="target-model-card-head">
                <div>
                  <strong>{currentTarget}</strong>
                  <span>{multiple ? "構成モデルを複数選択" : "ベースモデルを選択"}</span>
                </div>
                <span className={`status-chip ${currentMembers.length >= (multiple ? 2 : 1) ? "success" : "warning"}`}>
                  {currentMembers.length} models
                </span>
              </div>

              {multiple ? (
                <CheckboxList
                  values={currentModels}
                  selected={currentMembers}
                  onChange={(values) => setMembersByTarget((current) => ({
                    ...current,
                    [currentTarget]: values,
                  }))}
                />
              ) : (
                <Field label="ベースモデル">
                  <select
                    value={currentMembers[0] || ""}
                    onChange={(event) => setMembersByTarget((current) => ({
                      ...current,
                      [currentTarget]: [event.target.value],
                    }))}
                  >
                    {currentModels.map((model) => <option key={model}>{model}</option>)}
                  </select>
                </Field>
              )}

              {parameterMode === "manual" && parameterTabs.length > 0 && (
                <div className="ensemble-manual-parameters">
                  <div className="ensemble-parameter-tabs" role="tablist" aria-label="モデル別パラメータ">
                    {parameterTabs.map((tab) => (
                      <button
                        key={tab.key}
                        type="button"
                        role="tab"
                        aria-selected={tab.key === selectedParameterTab?.key}
                        className={tab.key === selectedParameterTab?.key ? "active" : ""}
                        onClick={() => setActiveParameterTab(tab.key)}
                      >
                        {tab.label}
                      </button>
                    ))}
                  </div>
                  <EnsembleParameterEditor
                    tab={selectedParameterTab}
                    schema={selectedSchema}
                    values={selectedParameterValues}
                    loading={parameterLoading}
                    error={parameterError}
                    onChange={(name, value) => updateParameter(selectedParameterTab, name, value)}
                    onReset={resetSelectedParameters}
                  />
                </div>
              )}
            </section>
          )}

          <p className="ensemble-override-note">
            {parameterMode === "manual"
              ? "選択中のモデルタブだけを表示しています。設定値はモデル順に学習APIへ送信します。"
              : "アンサンブル有効中は、下の単体モデル用設定ではなく、この欄のモデル構成とパラメータ設定を学習APIへ送信します。"}
          </p>
        </div>
      )}
    </section>,
    host,
  );
}
