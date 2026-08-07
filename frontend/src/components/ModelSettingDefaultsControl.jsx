import { useLayoutEffect, useRef } from "react";
import { useWorkbench } from "../context/WorkbenchContext";
import CompositionalSettingsControl from "./CompositionalSettingsControl";

function findButtonByText(root, selector, text) {
  return [...root.querySelectorAll(selector)]
    .find((button) => button.textContent?.trim() === text) || null;
}

function setNativeSelectValue(select, value) {
  const setter = Object.getOwnPropertyDescriptor(
    window.HTMLSelectElement.prototype,
    "value",
  )?.set;
  if (setter) setter.call(select, value);
  else select.value = value;
  select.dispatchEvent(new Event("change", { bubbles: true }));
}

export default function ModelSettingDefaultsControl() {
  const { step } = useWorkbench();
  const modelDefaultApplied = useRef(false);
  const ensembleDefaultApplied = useRef(false);

  useLayoutEffect(() => {
    if (step !== "model") {
      modelDefaultApplied.current = false;
      ensembleDefaultApplied.current = false;
      return undefined;
    }

    const contentRoot = document.querySelector(".content-inner") || document.body;
    let frameId = null;
    let disposed = false;

    const applyDefaults = () => {
      if (disposed) return;

      const manualButton = findButtonByText(
        contentRoot,
        ".parameter-mode-switch button",
        "個別設定",
      );
      if (manualButton && !modelDefaultApplied.current) {
        modelDefaultApplied.current = true;
        if (!manualButton.classList.contains("active")) manualButton.click();
      }

      const ensembleSection = contentRoot.querySelector(".ensemble-model-settings");
      const ensembleEnabled = Boolean(
        ensembleSection?.querySelector(
          ".ensemble-setting-heading input[type='checkbox']",
        )?.checked,
      );
      const ensembleParameterSelect = ensembleSection
        ? [...ensembleSection.querySelectorAll("select")]
          .find((select) => select.querySelector("option[value='manual']"))
        : null;

      if (!ensembleEnabled) {
        ensembleDefaultApplied.current = false;
      } else if (ensembleParameterSelect && !ensembleDefaultApplied.current) {
        ensembleDefaultApplied.current = true;
        if (ensembleParameterSelect.value !== "manual") {
          setNativeSelectValue(ensembleParameterSelect, "manual");
        }
      }
    };

    const scheduleApply = () => {
      if (frameId !== null) window.cancelAnimationFrame(frameId);
      frameId = window.requestAnimationFrame(() => {
        frameId = null;
        applyDefaults();
      });
    };

    applyDefaults();
    contentRoot.addEventListener("click", scheduleApply);
    contentRoot.addEventListener("change", scheduleApply);
    return () => {
      disposed = true;
      contentRoot.removeEventListener("click", scheduleApply);
      contentRoot.removeEventListener("change", scheduleApply);
      if (frameId !== null) window.cancelAnimationFrame(frameId);
    };
  }, [step]);

  return <CompositionalSettingsControl />;
}