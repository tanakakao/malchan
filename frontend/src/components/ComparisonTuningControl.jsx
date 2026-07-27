import React, { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { setComparisonTuneBestOverride } from "../api";
import { useWorkbench } from "../context/WorkbenchContext";
import "../comparison-tuning.css";

function findComparisonSettingsSection() {
  return [...document.querySelectorAll(".model-execution-settings")]
    .find((section) => section.textContent?.includes("比較後のモデル")) || null;
}

export default function ComparisonTuningControl() {
  const { step, activateBest } = useWorkbench();
  const [enabled, setEnabled] = useState(false);
  const [host, setHost] = useState(null);

  useEffect(() => {
    if (!activateBest) setEnabled(false);
  }, [activateBest]);

  useEffect(() => {
    setComparisonTuneBestOverride(activateBest && enabled);
    return () => setComparisonTuneBestOverride(false);
  }, [activateBest, enabled]);

  useEffect(() => {
    if (step !== "model") {
      setHost(null);
      return undefined;
    }

    const content = document.querySelector(".content-inner") || document.body;
    const resolveHost = () => {
      const section = findComparisonSettingsSection();
      if (!section) {
        setHost(null);
        return;
      }
      let nextHost = section.querySelector(":scope > .comparison-tuning-control-host");
      if (!nextHost) {
        nextHost = document.createElement("div");
        nextHost.className = "comparison-tuning-control-host";
        const heading = section.querySelector(":scope > .model-setting-heading");
        if (heading) heading.insertAdjacentElement("afterend", nextHost);
        else section.appendChild(nextHost);
      }
      setHost(nextHost);
    };

    resolveHost();
    const observer = new MutationObserver(resolveHost);
    observer.observe(content, { childList: true, subtree: true });
    return () => {
      observer.disconnect();
      setHost((current) => {
        if (current?.isConnected) current.remove();
        return null;
      });
    };
  }, [step]);

  if (!host) return null;

  return createPortal(
    <div className="comparison-tuning-control">
      <label
        className={`switch-label comparison-tuning-switch ${activateBest ? "" : "disabled"}`}
        title={activateBest ? "" : "ベストモデルを有効化すると選択できます。"}
      >
        <input
          type="checkbox"
          checked={activateBest && enabled}
          disabled={!activateBest}
          onChange={(event) => setEnabled(event.target.checked)}
        />
        <span />
        ベストモデルをチューニング
      </label>
      <small>
        {activateBest && enabled
          ? "比較1位をOptunaで調整し、再評価してから有効化します。"
          : "必要な場合だけ、比較後の1位モデルをOptunaで調整します。"}
      </small>
    </div>,
    host,
  );
}
