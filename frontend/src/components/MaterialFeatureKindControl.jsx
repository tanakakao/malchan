import React, { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { useWorkbench } from "../context/WorkbenchContext";
import {
  FEATURE_REPRESENTATIONS,
  pruneMaterialFeatureKinds,
  setMaterialFeatureKind,
  useMaterialFeatureSettings,
} from "../materialFeatures";

function featureCard(column) {
  return [...document.querySelectorAll(".feature-variable-choice")].find((card) => (
    card.querySelector(".variable-choice-main span")?.textContent?.trim() === column
  ));
}

function clearMaterialCardClasses(card) {
  card?.classList.remove("selected-composition", "selected-smiles");
}

export default function MaterialFeatureKindControl() {
  const { step, columns, catFeatures } = useWorkbench();
  const settings = useMaterialFeatureSettings();
  const [hosts, setHosts] = useState({});

  useEffect(() => {
    pruneMaterialFeatureKinds(columns);
  }, [columns]);

  useEffect(() => {
    if (step !== "prepare") {
      setHosts((current) => {
        Object.values(current).forEach((host) => {
          clearMaterialCardClasses(host.closest(".feature-variable-choice"));
          if (host.isConnected) host.remove();
        });
        return {};
      });
      return undefined;
    }

    const contentRoot = document.querySelector(".content-inner") || document.body;
    let frameId = null;
    let disposed = false;

    const connect = () => {
      if (disposed) return;
      const activeColumns = new Set(catFeatures);
      setHosts((current) => {
        const next = {};
        Object.entries(current).forEach(([column, host]) => {
          if (activeColumns.has(column) && host.isConnected) {
            next[column] = host;
          } else {
            clearMaterialCardClasses(host.closest(".feature-variable-choice"));
            if (host.isConnected) host.remove();
          }
        });

        catFeatures.forEach((column) => {
          const card = featureCard(column);
          if (!card) return;
          let host = next[column] || card.querySelector(":scope > .material-feature-kind-host");
          if (!host) {
            host = document.createElement("div");
            host.className = "material-feature-kind-host";
            card.appendChild(host);
          }
          clearMaterialCardClasses(card);
          const kind = settings.kinds[column] || "categorical";
          if (kind === "composition") card.classList.add("selected-composition");
          if (kind === "smiles") card.classList.add("selected-smiles");
          next[column] = host;
        });
        return next;
      });
    };

    const scheduleConnect = () => {
      if (frameId !== null) window.cancelAnimationFrame(frameId);
      frameId = window.requestAnimationFrame(() => {
        frameId = null;
        connect();
      });
    };

    scheduleConnect();
    contentRoot.addEventListener("click", scheduleConnect);
    contentRoot.addEventListener("change", scheduleConnect);
    return () => {
      disposed = true;
      contentRoot.removeEventListener("click", scheduleConnect);
      contentRoot.removeEventListener("change", scheduleConnect);
      if (frameId !== null) window.cancelAnimationFrame(frameId);
      setHosts((current) => {
        Object.values(current).forEach((host) => {
          clearMaterialCardClasses(host.closest(".feature-variable-choice"));
          if (host.isConnected) host.remove();
        });
        return {};
      });
    };
  }, [step, catFeatures.join("\u0001"), settings.kinds]);

  if (step !== "prepare") return null;

  return (
    <>
      {Object.entries(hosts).map(([column, host]) => createPortal(
        <label className="material-feature-kind-select">
          <span>入力表記</span>
          <select
            value={settings.kinds[column] || "categorical"}
            onChange={(event) => setMaterialFeatureKind(column, event.target.value)}
          >
            {FEATURE_REPRESENTATIONS.map(([value, label]) => (
              <option key={value} value={value}>{label}</option>
            ))}
          </select>
        </label>,
        host,
        column,
      ))}
    </>
  );
}
