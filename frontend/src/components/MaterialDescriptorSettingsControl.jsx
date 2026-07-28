import React, { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { CheckboxList, Field } from "./Common";
import { useWorkbench } from "../context/WorkbenchContext";
import {
  MATMINER_DESCRIPTORS,
  MENDELEEV_PROPERTIES,
  SMILES_FINGERPRINTS,
  THREE_DIMENSIONAL_FINGERPRINTS,
  patchMaterialFeatureSettings,
  useMaterialFeatureSettings,
} from "../materialFeatures";
import "../material-features.css";

function SelectedColumns({ columns }) {
  return (
    <div className="material-selected-columns" aria-label="対象列">
      {columns.map((column) => <span key={column}>{column}</span>)}
    </div>
  );
}

function SmilesDescriptorPanel({ columns, settings }) {
  const threeDimensional = settings.fingerprints.filter(
    (name) => THREE_DIMENSIONAL_FINGERPRINTS.includes(name),
  );
  return (
    <div className="material-descriptor-panel" role="tabpanel">
      <div className="material-descriptor-panel-head">
        <div>
          <strong>分子記述子・Fingerprint</strong>
          <span>選択した記述子を各SMILES列へ個別に適用します。</span>
        </div>
        <span className={`status-chip ${settings.fingerprints.length ? "success" : "warning"}`}>
          {settings.fingerprints.length} selected
        </span>
      </div>
      <SelectedColumns columns={columns} />
      <div className="material-descriptor-checklist">
        <CheckboxList
          values={SMILES_FINGERPRINTS}
          selected={settings.fingerprints}
          onChange={(fingerprints) => patchMaterialFeatureSettings({ fingerprints })}
        />
      </div>
      <p className="settings-note material-descriptor-note">
        ECFPは汎用的な既定値です。Autocorr、E3FP、MORSE、RDFは3次元配座生成を伴います。
        {threeDimensional.length ? ` 現在の3D記述子: ${threeDimensional.join(", ")}` : ""}
      </p>
    </div>
  );
}

function CompositionDescriptorPanel({ columns, settings }) {
  const useMendeleev = settings.compMethod === "mendeleev";
  const values = useMendeleev ? MENDELEEV_PROPERTIES : MATMINER_DESCRIPTORS;
  const selected = useMendeleev
    ? settings.mendeleevProperties
    : settings.matminerDescriptors;
  const patchKey = useMendeleev ? "mendeleevProperties" : "matminerDescriptors";

  return (
    <div className="material-descriptor-panel" role="tabpanel">
      <div className="material-descriptor-panel-head">
        <div>
          <strong>組成記述子</strong>
          <span>組成式を元素組成へ変換し、選択した記述子を生成します。</span>
        </div>
        <span className={`status-chip ${selected.length ? "success" : "warning"}`}>
          {selected.length} selected
        </span>
      </div>
      <SelectedColumns columns={columns} />
      <Field label="特徴量生成方式" className="material-method-field">
        <select
          value={settings.compMethod}
          onChange={(event) => patchMaterialFeatureSettings({ compMethod: event.target.value })}
        >
          <option value="matminer">Matminer</option>
          <option value="mendeleev">Mendeleev元素プロパティ</option>
        </select>
      </Field>
      <div className="material-descriptor-checklist">
        <CheckboxList
          values={values}
          selected={selected}
          onChange={(nextValues) => patchMaterialFeatureSettings({ [patchKey]: nextValues })}
        />
      </div>
      <p className="settings-note material-descriptor-note">
        {useMendeleev
          ? "元素プロパティを組成分率で重み付けし、平均・標準偏差・最小・最大・範囲を生成します。"
          : "ElementPropertyとStoichiometryを既定値とし、必要なMatminer記述子だけを追加してください。"}
      </p>
    </div>
  );
}

export default function MaterialDescriptorSettingsControl() {
  const { step, catFeatures } = useWorkbench();
  const settings = useMaterialFeatureSettings();
  const [host, setHost] = useState(null);
  const [activeTab, setActiveTab] = useState("smiles");

  const smilesColumns = catFeatures.filter((column) => settings.kinds[column] === "smiles");
  const compositionColumns = catFeatures.filter(
    (column) => settings.kinds[column] === "composition",
  );
  const hasSmiles = smilesColumns.length > 0;
  const hasComposition = compositionColumns.length > 0;
  const hasMaterialFeatures = hasSmiles || hasComposition;

  useEffect(() => {
    if (activeTab === "smiles" && !hasSmiles && hasComposition) setActiveTab("composition");
    if (activeTab === "composition" && !hasComposition && hasSmiles) setActiveTab("smiles");
  }, [activeTab, hasSmiles, hasComposition]);

  useEffect(() => {
    if (step !== "model" || !hasMaterialFeatures) {
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
      let nextHost = stack.querySelector(":scope > .material-descriptor-settings-host");
      if (!nextHost) {
        nextHost = document.createElement("div");
        nextHost.className = "material-descriptor-settings-host";
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
  }, [step, hasMaterialFeatures, smilesColumns.join("\u0001"), compositionColumns.join("\u0001")]);

  if (!host || !hasMaterialFeatures) return null;

  const selectedTab = hasSmiles && hasComposition
    ? activeTab
    : hasSmiles ? "smiles" : "composition";

  return createPortal(
    <section className="model-setting-section material-descriptor-settings">
      <div className="model-setting-heading material-descriptor-heading">
        <div>
          <strong>材料・分子記述子</strong>
          <span>Prepare画面で指定した組成式・SMILES列を数値特徴量へ変換します。</span>
        </div>
        <span className="status-chip success">
          {smilesColumns.length + compositionColumns.length} columns
        </span>
      </div>

      {hasSmiles && hasComposition && (
        <div className="material-descriptor-tabs" role="tablist" aria-label="記述子種別">
          <button
            type="button"
            role="tab"
            aria-selected={selectedTab === "smiles"}
            className={selectedTab === "smiles" ? "active" : ""}
            onClick={() => setActiveTab("smiles")}
          >
            分子表記 <span>{smilesColumns.length}</span>
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={selectedTab === "composition"}
            className={selectedTab === "composition" ? "active" : ""}
            onClick={() => setActiveTab("composition")}
          >
            組成式 <span>{compositionColumns.length}</span>
          </button>
        </div>
      )}

      {selectedTab === "smiles" && (
        <SmilesDescriptorPanel columns={smilesColumns} settings={settings} />
      )}
      {selectedTab === "composition" && (
        <CompositionDescriptorPanel columns={compositionColumns} settings={settings} />
      )}
    </section>,
    host,
  );
}
