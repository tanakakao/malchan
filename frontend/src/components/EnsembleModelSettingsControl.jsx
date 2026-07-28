import React, { useEffect, useMemo, useState } from "react";
import { createPortal } from "react-dom";
import { setEnsembleTrainingOptions } from "../api";
import { CheckboxList, Field } from "./Common";
import { modelsFor, useWorkbench } from "../context/WorkbenchContext";
import "../model-ensemble.css";

const ENSEMBLE_TYPES = [
  ["アンサンブル", "Voting", "複数モデルの予測を平均・多数決で統合します。"],
  ["スタッキング", "Stacking", "複数モデルの予測を最終モデルで統合します。"],
  ["バギング", "Bagging", "1つのベースモデルを複数の再標本データで学習します。"],
  ["ブースティング", "Boosting", "1つのベースモデルを逐次的に補正しながら学習します。"],
];

function requiresMultipleModels(ensembleType) {
  return ensembleType === "アンサンブル" || ensembleType === "スタッキング";
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
  const [tuning, setTuning] = useState(true);

  const currentTarget = targets.includes(activeTarget) ? activeTarget : targets[0] || "";
  const taskKinds = useMemo(
    () => new Set(targets.map((target) => tasks[target]).filter(Boolean)),
    [targets, tasks],
  );
  const mixedTasks = taskKinds.size > 1;
  const stackingModels = taskKinds.size === 1 && targets.length
    ? modelsFor(tasks[targets[0]])
    : [];

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
    if (!enabled || step !== "model" || !host) {
      setEnsembleTrainingOptions(null);
      return undefined;
    }
    setEnsembleTrainingOptions({
      ensembleType,
      baseModel: ensembleType === "スタッキング" ? stackingBaseModel : null,
      membersByTarget,
      tuning,
    });
    return () => setEnsembleTrainingOptions(null);
  }, [enabled, step, host, ensembleType, stackingBaseModel, membersByTarget, tuning]);

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
      summary.dataset.ensembleSummary = `${ENSEMBLE_TYPES.find(([value]) => value === ensembleType)?.[1] || ensembleType} · ${tuning ? "Optuna tuning" : "Default parameters"}`;
    }
    return () => {
      panel.classList.remove("ensemble-active");
      if (summary) delete summary.dataset.ensembleSummary;
    };
  }, [host, enabled, ensembleType, tuning]);

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
              <select value={tuning ? "tuning" : "default"} onChange={(event) => setTuning(event.target.value === "tuning")}>
                <option value="tuning">Optunaでチューニング</option>
                <option value="default">各モデルの既定値</option>
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
            </section>
          )}

          <p className="ensemble-override-note">
            アンサンブル有効中は、下の単体モデル用設定ではなく、この欄のモデル構成とパラメータ設定を学習APIへ送信します。
          </p>
        </div>
      )}
    </section>,
    host,
  );
}
