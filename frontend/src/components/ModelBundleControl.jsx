import React, { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useWorkbench } from "../context/WorkbenchContext";
import { downloadModelBundle } from "../modelBundles";

function bestModelNames(comparison) {
  return Object.fromEntries(
    Object.entries(comparison?.targets || {})
      .filter(([, result]) => result?.best_model_name)
      .map(([target, result]) => [target, result.best_model_name]),
  );
}

function resolvedModelInfo(modelInfo, comparison, activateBest) {
  if (!modelInfo || !activateBest) return modelInfo;
  const bestNames = bestModelNames(comparison);
  if (!Object.keys(bestNames).length) return modelInfo;
  const targetNames = {
    ...(modelInfo.model_names_by_target || {}),
    ...Object.fromEntries(
      Object.entries(bestNames).map(([target, name]) => [target, [name]]),
    ),
  };
  const targets = modelInfo.target_cols || Object.keys(targetNames);
  return {
    ...modelInfo,
    model_names_by_target: targetNames,
    model_names: targets.length === 1 && bestNames[targets[0]]
      ? [bestNames[targets[0]]]
      : modelInfo.model_names,
  };
}

function defaultSaveName(modelInfo) {
  const names = Object.values(modelInfo?.model_names_by_target || {})
    .flat()
    .filter(Boolean);
  if (names.length === 1) return names[0];
  return modelInfo?.model_id ? `malchan-model-${modelInfo.model_id}` : "malchan-model";
}

export default function ModelBundleControl() {
  const {
    step,
    modelInfo,
    comparison,
    activateBest,
    setModelNames,
    busy,
    setToast,
    loadModelBundle,
  } = useWorkbench();
  const [host, setHost] = useState(null);
  const [actionHost, setActionHost] = useState(null);
  const [runAction, setRunAction] = useState({ label: "モデル学習を実行 →", disabled: true });
  const [saveName, setSaveName] = useState("");
  const [downloading, setDownloading] = useState(false);
  const fileInputRef = useRef(null);
  const sourceRunButtonRef = useRef(null);
  const displayedModelInfo = useMemo(
    () => resolvedModelInfo(modelInfo, comparison, activateBest),
    [modelInfo, comparison, activateBest],
  );

  useEffect(() => {
    setSaveName(defaultSaveName(displayedModelInfo));
  }, [displayedModelInfo?.model_id, comparison, activateBest]);

  useEffect(() => {
    if (!activateBest) return;
    const names = bestModelNames(comparison);
    if (!Object.keys(names).length) return;
    setModelNames((current) => {
      const changed = Object.entries(names).some(([target, name]) => current[target] !== name);
      return changed ? { ...current, ...names } : current;
    });
  }, [activateBest, comparison, setModelNames]);

  useEffect(() => {
    if (step !== "model") {
      sourceRunButtonRef.current = null;
      setHost((current) => {
        if (current?.isConnected) current.remove();
        return null;
      });
      setActionHost((current) => {
        if (current?.isConnected) current.remove();
        return null;
      });
      return undefined;
    }

    const contentRoot = document.querySelector(".content-inner") || document.body;
    let frameId = null;
    let disposed = false;

    const resolveHosts = () => {
      if (disposed) return;
      const settings = contentRoot.querySelector(".model-settings-columns");
      const sectionHeader = contentRoot.querySelector(".section-header");
      if (!settings || !sectionHeader) {
        setHost(null);
        setActionHost(null);
        return;
      }

      let nextHost = contentRoot.querySelector(":scope > .model-bundle-control-host");
      if (!nextHost) {
        nextHost = document.createElement("div");
        nextHost.className = "model-bundle-control-host";
        nextHost.dataset.location = "model-page";
      }
      if (settings.previousElementSibling !== nextHost) {
        settings.insertAdjacentElement("beforebegin", nextHost);
      }
      setHost(nextHost);

      let nextActionHost = sectionHeader.querySelector(":scope > .model-run-action-host");
      if (!nextActionHost) {
        nextActionHost = document.createElement("div");
        nextActionHost.className = "model-run-action-host";
        sectionHeader.appendChild(nextActionHost);
      }
      setActionHost(nextActionHost);

      const sourceButton = contentRoot.querySelector(".model-run-summary button");
      sourceRunButtonRef.current = sourceButton;
      if (sourceButton) {
        setRunAction({
          label: sourceButton.textContent?.trim() || "学習を実行 →",
          disabled: sourceButton.disabled,
        });
      }

      const evaluationPanel = contentRoot.querySelector(".evaluation-result-panel");
      if (evaluationPanel) evaluationPanel.hidden = true;

      const registeredModel = contentRoot.querySelector(".model-registration-panel pre");
      if (registeredModel && displayedModelInfo) {
        const serialized = JSON.stringify(displayedModelInfo, null, 2);
        if (registeredModel.textContent !== serialized) registeredModel.textContent = serialized;
      }
    };

    const scheduleResolve = () => {
      if (frameId !== null) window.cancelAnimationFrame(frameId);
      frameId = window.requestAnimationFrame(() => {
        frameId = null;
        resolveHosts();
      });
    };

    const observer = new MutationObserver(scheduleResolve);
    observer.observe(contentRoot, {
      subtree: true,
      childList: true,
      characterData: true,
      attributes: true,
      attributeFilter: ["disabled", "class"],
    });
    scheduleResolve();
    return () => {
      disposed = true;
      observer.disconnect();
      if (frameId !== null) window.cancelAnimationFrame(frameId);
      sourceRunButtonRef.current = null;
      setHost((current) => {
        if (current?.isConnected) current.remove();
        return null;
      });
      setActionHost((current) => {
        if (current?.isConnected) current.remove();
        return null;
      });
    };
  }, [step, displayedModelInfo]);

  async function handleDownload() {
    if (!modelInfo?.model_id || downloading) return;
    setDownloading(true);
    try {
      const result = await downloadModelBundle(modelInfo.model_id, saveName);
      const url = URL.createObjectURL(result.blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = result.filename;
      anchor.style.display = "none";
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 0);
      setToast({ text: `${result.filename}をダウンロードしました。`, type: "success" });
    } catch (error) {
      setToast({ text: error.message || String(error), type: "error" });
    } finally {
      setDownloading(false);
    }
  }

  async function handleFile(event) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (file) await loadModelBundle(file);
  }

  return (
    <>
      {actionHost && createPortal(
        <button
          type="button"
          disabled={runAction.disabled}
          onClick={() => sourceRunButtonRef.current?.click()}
        >
          {runAction.label}
        </button>,
        actionHost,
      )}
      {host && createPortal(
        <article className="panel model-bundle-panel">
          <div className="panel-title">
            <div><h3>モデルの保存・読み込み</h3></div>
          </div>

          <div className="model-bundle-actions">
            <label className="model-bundle-save-name">
              保存名
              <input
                type="text"
                value={saveName}
                placeholder="malchan-model"
                onChange={(event) => setSaveName(event.target.value)}
              />
            </label>
            <button
              type="button"
              className="secondary"
              disabled={!modelInfo || Boolean(busy) || downloading || !saveName.trim()}
              onClick={handleDownload}
            >
              {downloading ? "保存中..." : "モデルをダウンロード"}
            </button>
            <button
              type="button"
              disabled={Boolean(busy) || downloading}
              onClick={() => fileInputRef.current?.click()}
            >
              モデルファイルを読み込む
            </button>
            <input
              ref={fileInputRef}
              className="model-bundle-file-input"
              type="file"
              accept=".malchan,application/vnd.malchan.model"
              onChange={handleFile}
            />
          </div>
        </article>,
        host,
      )}
    </>
  );
}
