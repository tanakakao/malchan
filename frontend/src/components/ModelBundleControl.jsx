import React, { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useWorkbench } from "../context/WorkbenchContext";
import { downloadModelBundle, modelBundleFilename } from "../modelBundles";

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
  if (names.length === 1) return modelBundleFilename(names[0]);
  return modelInfo?.model_id
    ? `malchan-model-${modelInfo.model_id}.malchan`
    : "malchan-model.malchan";
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
  } = useWorkbench();
  const [actionHost, setActionHost] = useState(null);
  const [runAction, setRunAction] = useState({
    label: "モデル学習を実行 →",
    disabled: true,
  });
  const [saveName, setSaveName] = useState("");
  const [downloading, setDownloading] = useState(false);
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
      setActionHost((current) => {
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
      const sectionHeader = contentRoot.querySelector(".section-header");
      if (!sectionHeader) {
        setActionHost(null);
        return;
      }

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
        resolveHost();
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
      setActionHost((current) => {
        if (current?.isConnected) current.remove();
        return null;
      });
    };
  }, [step, displayedModelInfo]);

  async function handleDownload() {
    if (!modelInfo?.model_id || downloading) return;
    const normalizedName = modelBundleFilename(saveName, `malchan-model-${modelInfo.model_id}`);
    setSaveName(normalizedName);
    setDownloading(true);
    try {
      const result = await downloadModelBundle(modelInfo.model_id, normalizedName);
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

  if (!actionHost) return null;

  return createPortal(
    <div className="model-header-actions">
      <div className="model-save-control">
        <label>
          保存名
          <input
            type="text"
            value={saveName}
            placeholder="malchan-model.malchan"
            aria-label="モデル保存名"
            onChange={(event) => setSaveName(event.target.value)}
            onBlur={() => setSaveName(modelBundleFilename(saveName))}
          />
        </label>
        <button
          type="button"
          className="secondary"
          disabled={!modelInfo || Boolean(busy) || downloading || !saveName.trim()}
          onClick={handleDownload}
        >
          {downloading ? "モデル保存中" : "モデルを保存"}
        </button>
      </div>
      <button
        type="button"
        disabled={runAction.disabled}
        onClick={() => sourceRunButtonRef.current?.click()}
      >
        {runAction.label}
      </button>
    </div>,
    actionHost,
  );
}
