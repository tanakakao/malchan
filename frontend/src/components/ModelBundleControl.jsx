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
            サーバーへ永続保存せず、署名付きモデルファイルをPCへダウンロードして管理します。
          </p>
        </div>
        <span className="status-chip">Server storage off</span>
      </div>

      <div className="model-bundle-content">
        <div className="model-bundle-security-note">
          <strong>署名検証あり</strong>
          <span>
            読み込み時はファイル全体のHMAC署名を確認してからモデルをメモリへ復元します。
            FastAPIの停止後は、読み込んだモデルもメモリから消えます。
          </span>
          <span>
            モデル内部に学習データや説明用キャッシュが含まれる場合があります。
            ダウンロードしたファイルは機密データとして管理してください。
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
        利用前にFastAPI環境へ32文字以上の
        <code>MALCHAN_MODEL_BUNDLE_SECRET</code>
        を設定してください。同じ秘密値とPythonメジャー・マイナーバージョンを使用する環境間で読み込めます。
      </p>
    </article>,
    host,
  );
}
