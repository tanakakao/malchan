import React, { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { CheckboxList, Field } from "./Common";
import { useWorkbench } from "../context/WorkbenchContext";
import {
  COMPOSITIONAL_METHODS,
  patchMaterialFeatureSettings,
  pruneCompositionalGroups,
  useMaterialFeatureSettings,
} from "../materialFeatures";

function groupLabel(index) {
  return `組成グループ ${index + 1}`;
}

function selectedColumns(groups, excludedIndex) {
  return groups.flatMap((group, index) => (index === excludedIndex ? [] : group));
}

export default function CompositionalSettingsControl() {
  const { step, numFeatures } = useWorkbench();
  const settings = useMaterialFeatureSettings();
  const [host, setHost] = useState(null);
  const available = numFeatures.length >= 2;

  useEffect(() => {
    pruneCompositionalGroups(numFeatures);
  }, [numFeatures.join("\u0001")]);

  useEffect(() => {
    if (step !== "model" || !available) {
      setHost((current) => {
        if (current?.isConnected) current.remove();
        return null;
      });
      return undefined;
    }

    const contentRoot = document.querySelector(".content-inner") || document.body;
    let frameId = null;
    let disposed = false;

    const resolveHost = () => {
      if (disposed) return;
      const stack = document.querySelector(".preprocessing-panel .model-settings-stack");
      if (!stack) {
        setHost(null);
        return;
      }
      let nextHost = stack.querySelector(":scope > .compositional-settings-host");
      if (!nextHost) {
        nextHost = document.createElement("div");
        nextHost.className = "compositional-settings-host";
        stack.prepend(nextHost);
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
  }, [step, available, numFeatures.join("\u0001")]);

  if (!host || !available) return null;

  const groups = settings.compositionalGroups || [];
  const usedColumns = new Set(groups.flat());
  const remainingColumns = numFeatures.filter((column) => !usedColumns.has(column));
  const groupsComplete = groups.every((group) => group.length >= 2);
  const canAddGroup = groupsComplete && remainingColumns.length >= 2;

  function toggleEnabled(enabled) {
    patchMaterialFeatureSettings({
      compositionalEnabled: enabled,
      compositionalGroups: enabled && groups.length === 0 ? [[]] : groups,
    });
  }

  function patchGroup(index, columns) {
    const nextGroups = groups.map((group, groupIndex) => (
      groupIndex === index ? columns : group
    ));
    patchMaterialFeatureSettings({ compositionalGroups: nextGroups });
  }

  function addGroup() {
    if (!canAddGroup) return;
    patchMaterialFeatureSettings({ compositionalGroups: [...groups, []] });
  }

  function removeGroup(index) {
    patchMaterialFeatureSettings({
      compositionalGroups: groups.filter((_, groupIndex) => groupIndex !== index),
    });
  }

  return createPortal(
    <section className="model-setting-section compositional-settings">
      <div className="model-setting-heading material-descriptor-heading">
        <div>
          <strong>組成比データ</strong>
          <span>足して1（または100）になる数値列をlog-ratio座標へ変換します。</span>
        </div>
        <label className="switch-label">
          <input
            type="checkbox"
            checked={settings.compositionalEnabled}
            onChange={(event) => toggleEnabled(event.target.checked)}
          />
          <span />
          使用する
        </label>
      </div>

      {settings.compositionalEnabled && (
        <div className="material-descriptor-panel" role="tabpanel">
          <div className="material-descriptor-panel-head">
            <div>
              <strong>組成グループ</strong>
              <span>同じ合計制約を持つ列を1グループとして2列以上選択します。</span>
            </div>
            <span className={`status-chip ${groupsComplete && groups.length ? "success" : "warning"}`}>
              {groups.length} groups
            </span>
          </div>

          <div className="model-settings-stack">
            {groups.map((group, index) => {
              const disabled = selectedColumns(groups, index);
              return (
                <div className="model-setting-section" key={`composition-group-${index}`}>
                  <div className="model-setting-heading">
                    <div>
                      <strong>{groupLabel(index)}</strong>
                      <span>{group.length} columns selected</span>
                    </div>
                    <button
                      type="button"
                      className="secondary compact-button"
                      onClick={() => removeGroup(index)}
                    >
                      削除
                    </button>
                  </div>
                  <CheckboxList
                    values={numFeatures}
                    selected={group}
                    disabled={disabled}
                    onChange={(columns) => patchGroup(index, columns)}
                  />
                </div>
              );
            })}
          </div>

          <button
            type="button"
            className="secondary compact-button"
            disabled={!canAddGroup}
            onClick={addGroup}
            title={canAddGroup ? "別の合計制約を持つ組成グループを追加" : "現在のグループを2列以上完成させ、未使用列を2列以上残してください"}
          >
            ＋ 組成グループを追加
          </button>

          <div className="model-inline-fields">
            <Field label="変換方法">
              <select
                value={settings.compositionalMethod}
                onChange={(event) => patchMaterialFeatureSettings({
                  compositionalMethod: event.target.value,
                })}
              >
                {COMPOSITIONAL_METHODS.map((method) => (
                  <option key={method} value={method}>{method}</option>
                ))}
              </select>
            </Field>

            <Field label="ゼロ置換値">
              <input
                type="number"
                min="0"
                max="0.999999999"
                step="any"
                value={settings.compositionalZeroReplacement}
                onChange={(event) => patchMaterialFeatureSettings({
                  compositionalZeroReplacement: event.target.value,
                })}
              />
            </Field>
          </div>

          <div className="model-inline-fields">
            <Field label="変換後スケーリング">
              <select
                value={settings.compositionalScaleType}
                onChange={(event) => patchMaterialFeatureSettings({
                  compositionalScaleType: event.target.value,
                })}
              >
                <option value="">使用しない</option>
                <option value="StandardScaler">標準化</option>
                <option value="MinMaxScaler">Min-Max</option>
                <option value="centering">中心化のみ</option>
                <option value="MaxAbsScaler">最大絶対値</option>
              </select>
            </Field>

            {settings.compositionalMethod === "ALR" && (
              <Field label="ALR基準成分">
                <select
                  value={settings.compositionalAlrReference}
                  onChange={(event) => patchMaterialFeatureSettings({
                    compositionalAlrReference: Number(event.target.value),
                  })}
                >
                  <option value={-1}>各グループの最後の成分</option>
                  <option value={0}>各グループの最初の成分</option>
                </select>
              </Field>
            )}
          </div>

          <label className="switch-label setting-inline-switch">
            <input
              type="checkbox"
              checked={settings.compositionalClosure}
              onChange={(event) => patchMaterialFeatureSettings({
                compositionalClosure: event.target.checked,
              })}
            />
            <span />
            合計を1へ正規化（closure）
          </label>

          <p className="settings-note material-descriptor-note">
            通常はILRを推奨します。選択した組成列は通常の数値前処理から自動的に除外され、
            組成変換だけが1回適用されます。0を含む組成は指定値でmultiplicative replacementします。
          </p>
        </div>
      )}
    </section>,
    host,
  );
}
