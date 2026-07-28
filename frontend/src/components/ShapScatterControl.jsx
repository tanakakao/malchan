import React, { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { api } from "../api";
import PlotlyFigure from "./PlotlyFigure";
import { useWorkbench } from "../context/WorkbenchContext";
import "../shap-scatter-control.css";

function findPartialDependencePanel() {
  return [...document.querySelectorAll(".xai-result-panel")]
    .find((panel) => panel.textContent?.includes("PARTIAL DEPENDENCE")) || null;
}

function findTargetSelect() {
  return document.querySelector(".xai-overview .xai-controls select");
}

function ScatterFigure({ state }) {
  if (state.loading) {
    return <p className="empty-state">show_shap_scatterで散布図を生成しています...</p>;
  }
  if (state.error) return <p className="xai-error">{state.error}</p>;
  if (!state.response?.figure?.data?.length) {
    return <p className="empty-state">XAI計算後にSHAP散布図を表示します。</p>;
  }
  return (
    <PlotlyFigure
      figure={state.response.figure}
      className="plotly-figure shap-scatter-figure"
    />
  );
}

export default function ShapScatterControl() {
  const { step, modelInfo, features } = useWorkbench();
  const targetRef = useRef("");
  const panelRef = useRef(null);
  const [buttonHost, setButtonHost] = useState(null);
  const [contentHost, setContentHost] = useState(null);
  const [active, setActive] = useState(false);
  const [target, setTarget] = useState("");
  const [feature, setFeature] = useState(features[0] || "");
  const [interactiveColumn, setInteractiveColumn] = useState("");
  const [output, setOutput] = useState("");
  const [state, setState] = useState({ response: null, loading: false, error: "" });

  useEffect(() => {
    if (!features.length) {
      setFeature("");
      setInteractiveColumn("");
      return;
    }
    if (!features.includes(feature)) setFeature(features[0]);
    if (interactiveColumn && !features.includes(interactiveColumn)) {
      setInteractiveColumn("");
    }
  }, [features, feature, interactiveColumn]);

  useEffect(() => {
    setOutput("");
    setState({ response: null, loading: false, error: "" });
  }, [modelInfo?.model_id, target]);

  useEffect(() => {
    if (step !== "explain") {
      setButtonHost(null);
      setContentHost(null);
      setActive(false);
      return undefined;
    }

    const content = document.querySelector(".content-inner") || document.body;
    let panel = null;
    let targetSelect = null;
    let nativeButtons = [];

    const syncTarget = () => {
      const nextTarget = targetSelect?.value || "";
      if (targetRef.current === nextTarget) return;
      targetRef.current = nextTarget;
      setTarget(nextTarget);
      setOutput("");
    };

    const deactivateScatter = () => setActive(false);

    const connect = () => {
      const nextPanel = findPartialDependencePanel();
      if (nextPanel !== panel) {
        panel?.classList.remove("shap-scatter-controlled");
        nativeButtons.forEach((button) => button.removeEventListener("click", deactivateScatter));
        nativeButtons = [];
        panel = nextPanel;
        panelRef.current = panel;

        if (panel) {
          const heading = panel.querySelector(".xai-result-head h3");
          if (heading) heading.textContent = "PD 1D / 2D / SHAP散布図";

          const modeToggle = panel.querySelector(".pd-mode-toggle");
          let nextButtonHost = modeToggle?.querySelector(":scope > .shap-scatter-button-host");
          if (modeToggle && !nextButtonHost) {
            nextButtonHost = document.createElement("span");
            nextButtonHost.className = "shap-scatter-button-host";
            modeToggle.append(nextButtonHost);
          }
          setButtonHost(nextButtonHost || null);

          const resultCard = panel.querySelector(".pd-result-card");
          let nextContentHost = panel.querySelector(":scope > .shap-scatter-content-host");
          if (!nextContentHost) {
            nextContentHost = document.createElement("div");
            nextContentHost.className = "shap-scatter-content-host";
            if (resultCard) resultCard.insertAdjacentElement("beforebegin", nextContentHost);
            else panel.append(nextContentHost);
          }
          setContentHost(nextContentHost);

          nativeButtons = [...(modeToggle?.querySelectorAll(":scope > button") || [])];
          nativeButtons.forEach((button) => button.addEventListener("click", deactivateScatter));
        } else {
          setButtonHost(null);
          setContentHost(null);
        }
      } else if (panel) {
        const heading = panel.querySelector(".xai-result-head h3");
        if (heading && heading.textContent !== "PD 1D / 2D / SHAP散布図") {
          heading.textContent = "PD 1D / 2D / SHAP散布図";
        }
      }

      const nextSelect = findTargetSelect();
      if (nextSelect !== targetSelect) {
        targetSelect?.removeEventListener("change", syncTarget);
        targetSelect = nextSelect;
        targetSelect?.addEventListener("change", syncTarget);
      }
      syncTarget();
    };

    connect();
    const observer = new MutationObserver(connect);
    observer.observe(content, { childList: true, subtree: true });

    return () => {
      observer.disconnect();
      targetSelect?.removeEventListener("change", syncTarget);
      nativeButtons.forEach((button) => button.removeEventListener("click", deactivateScatter));
      if (panel) {
        panel.classList.remove("shap-scatter-controlled");
        const heading = panel.querySelector(".xai-result-head h3");
        if (heading) heading.textContent = "PD 1D / 2D";
        panel.querySelector(":scope > .shap-scatter-content-host")?.remove();
        panel.querySelector(".shap-scatter-button-host")?.remove();
      }
      panelRef.current = null;
      targetRef.current = "";
      setButtonHost(null);
      setContentHost(null);
      setActive(false);
    };
  }, [step]);

  useEffect(() => {
    const panel = panelRef.current;
    if (!panel) return;
    panel.classList.toggle("shap-scatter-controlled", active);
  }, [active, buttonHost, contentHost]);

  useEffect(() => {
    let current = true;
    if (
      !active
      || step !== "explain"
      || !modelInfo?.model_id
      || !target
      || !feature
    ) {
      setState({ response: null, loading: false, error: "" });
      return () => {
        current = false;
      };
    }

    setState((value) => ({ ...value, loading: true, error: "" }));
    api.visualizationShapScatter(modelInfo.model_id, target, {
      feature,
      interactive_col: interactiveColumn,
      output,
    })
      .then((response) => {
        if (!current) return;
        setState({ response, loading: false, error: "" });
        if (!output && response.metadata?.selected_output) {
          setOutput(response.metadata.selected_output);
        }
      })
      .catch((error) => {
        if (!current) return;
        setState({
          response: null,
          loading: false,
          error: error.message || String(error),
        });
      });

    return () => {
      current = false;
    };
  }, [active, step, modelInfo?.model_id, target, feature, interactiveColumn, output]);

  const outputs = state.response?.metadata?.outputs || [];

  return (
    <>
      {buttonHost && createPortal(
        <button
          type="button"
          className={`shap-scatter-mode-button ${active ? "active" : ""}`}
          onClick={() => setActive(true)}
        >
          SHAP散布図
        </button>,
        buttonHost,
      )}
      {contentHost && active && createPortal(
        <section className="shap-scatter-control">
          <div className="shap-scatter-toolbar">
            <label>
              SHAP特徴量
              <select value={feature} onChange={(event) => setFeature(event.target.value)}>
                {features.map((column) => <option key={column}>{column}</option>)}
              </select>
            </label>
            <label>
              interactive col
              <select
                value={interactiveColumn}
                onChange={(event) => setInteractiveColumn(event.target.value)}
              >
                <option value="">なし</option>
                {features.map((column) => <option key={column}>{column}</option>)}
              </select>
            </label>
            {outputs.length > 1 && (
              <label>
                SHAP出力
                <select value={output} onChange={(event) => setOutput(event.target.value)}>
                  {outputs.map((name) => <option key={name}>{name}</option>)}
                </select>
              </label>
            )}
          </div>
          <p className="shap-scatter-note">
            横軸に特徴量、縦軸にSHAP値を表示します。interactive colを指定すると、その値で点を色分けします。
          </p>
          <div className="xai-card shap-scatter-result-card">
            <ScatterFigure state={state} />
          </div>
        </section>,
        contentHost,
      )}
    </>
  );
}
