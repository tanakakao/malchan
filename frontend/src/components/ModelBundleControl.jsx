import React, { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useWorkbench } from "../context/WorkbenchContext";
import { downloadModelBundle } from "../modelBundles";

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
    busy,
    setToast,
    loadModelBundle,
  } = useWorkbench();
  const [host, setHost] = useState(null);
  const [saveName, setSaveName] = useState("");
  const [downloading, setDownloading] = useState(false);
  const fileInputRef = useRef(null);

  useEffect(() => {
    setSaveName(defaultSaveName(modelInfo));
  }, [modelInfo?.model_id]);

  useEffect(() => {
    if (step !== "model") {
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
      const settings = contentRoot.querySelector(".model-settings-columns");
      if (!settings) {
        setHost(null);
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
  }, [step]);

  if (!host) return null;

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

  return createPortal(
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
  );
}
