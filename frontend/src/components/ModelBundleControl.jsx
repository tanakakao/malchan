import React, { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useWorkbench } from "../context/WorkbenchContext";

export default function ModelBundleControl() {
  const {
    step,
    modelInfo,
    busy,
    downloadActiveModel,
    loadModelBundle,
  } = useWorkbench();
  const [host, setHost] = useState(null);
  const fileInputRef = useRef(null);

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
      if (settings.nextElementSibling !== nextHost) {
        settings.insertAdjacentElement("afterend", nextHost);
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

  async function handleFile(event) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (file) await loadModelBundle(file);
  }

  return createPortal(
    <article className="panel model-bundle-panel">
      <div className="panel-title">
        <div>
          <span className="panel-kicker">MODEL FILE</span>
          <h3>モデルの保存・読み込み</h3>
          <p>
            サーバーへ永続保存せず、モデルファイルをPCへダウンロードして管理します。
          </p>
        </div>
        <span className="status-chip">Server storage off</span>
      </div>

      <div className="model-bundle-content">
        <div className="model-bundle-security-note">
          <strong>bochanと同様の信頼済みファイル方式</strong>
          <span>
            モデルファイルはpickle形式です。malchanからダウンロードしたものなど、
            作成元を信頼できるファイルだけを読み込んでください。
          </span>
          <span>
            モデル内に学習データが保持されている場合、予測入力と最適化条件の
            初期値・範囲・カテゴリ候補へ自動的に反映します。
          </span>
          <span>
            FastAPIの停止後は、読み込んだモデルもメモリから消えます。
          </span>
        </div>

        <div className="model-bundle-actions">
          <button
            type="button"
            className="secondary"
            disabled={!modelInfo || Boolean(busy)}
            onClick={downloadActiveModel}
          >
            モデルをダウンロード
          </button>
          <button
            type="button"
            disabled={Boolean(busy)}
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
      </div>

      <p className="model-bundle-config-note">
        署名用の秘密値は不要です。作成時と互換性のあるmalchanおよび依存ライブラリの
        環境で読み込んでください。
      </p>
    </article>,
    host,
  );
}
