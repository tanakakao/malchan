import React, { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { api } from "../api";
import PlotlyFigure from "./PlotlyFigure";
import { useWorkbench } from "../context/WorkbenchContext";
import "../yy-diagnostics.css";

function findYyPanel() {
  return [...document.querySelectorAll(".xai-result-panel")]
    .find((panel) => panel.textContent?.includes("Y-Y PLOT")) || null;
}

function findTargetSelect() {
  return document.querySelector(".xai-overview .xai-controls select");
}

function FigureContent({ state }) {
  if (state.loading) {
    return <p className="empty-state">yy_plot_mlで診断図を生成しています...</p>;
  }
  if (state.error) return <p className="xai-error">{state.error}</p>;
  if (!state.response?.figure) {
    return <p className="empty-state">登録モデルの診断図を表示できません。</p>;
  }
  return <PlotlyFigure figure={state.response.figure} className="plotly-figure yy-diagnostic-figure" />;
}

export default function YyDiagnosticsControl() {
  const { step, modelInfo } = useWorkbench();
  const targetRef = useRef("");
  const [host, setHost] = useState(null);
  const [target, setTarget] = useState("");
  const [plotType, setPlotType] = useState("prediction");
  const [source, setSource] = useState("fit");
  const [split, setSplit] = useState("test");
  const [state, setState] = useState({ response: null, loading: false, error: "" });

  function resetDiagnosticSelection() {
    setSource("fit");
    setPlotType("prediction");
    setSplit("test");
  }

  useEffect(() => {
    resetDiagnosticSelection();
  }, [modelInfo?.model_id]);

  useEffect(() => {
    if (step !== "explain") {
      setHost(null);
      return undefined;
    }

    const content = document.querySelector(".content-inner") || document.body;
    let panel = null;
    let targetSelect = null;

    const syncTarget = () => {
      const nextTarget = targetSelect?.value || "";
      if (targetRef.current === nextTarget) return;
      targetRef.current = nextTarget;
      resetDiagnosticSelection();
      setTarget(nextTarget);
    };

    const connect = () => {
      const nextPanel = findYyPanel();
      if (nextPanel !== panel) {
        panel?.classList.remove("yy-diagnostic-controlled");
        panel = nextPanel;
        if (panel) {
          panel.classList.add("yy-diagnostic-controlled");
          let nextHost = panel.querySelector(":scope > .yy-diagnostic-control-host");
          if (!nextHost) {
            nextHost = document.createElement("div");
            nextHost.className = "yy-diagnostic-control-host";
            const heading = panel.querySelector(":scope > .xai-result-head");
            if (heading) heading.insertAdjacentElement("afterend", nextHost);
            else panel.prepend(nextHost);
          }
          setHost(nextHost);
        } else {
          setHost(null);
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
      panel?.classList.remove("yy-diagnostic-controlled");
      const currentHost = panel?.querySelector(":scope > .yy-diagnostic-control-host");
      currentHost?.remove();
      targetRef.current = "";
      setHost(null);
    };
  }, [step]);

  useEffect(() => {
    let active = true;
    if (!modelInfo?.model_id || !target || step !== "explain") {
      setState({ response: null, loading: false, error: "" });
      return () => {
        active = false;
      };
    }

    setState((current) => ({ ...current, loading: true, error: "" }));
    api.visualizationYy(modelInfo.model_id, target, {
      cv: source === "cv",
      residual: plotType === "residual",
      split,
    })
      .then((response) => {
        if (!active) return;
        const metadata = response.metadata || {};
        if (!metadata.cv_available && source === "cv") {
          setSource("fit");
          return;
        }
        if (metadata.task === "classification" && plotType === "residual") {
          setPlotType("prediction");
          return;
        }
        setState({ response, loading: false, error: "" });
      })
      .catch((error) => {
        if (active) {
          setState({
            response: null,
            loading: false,
            error: error.message || String(error),
          });
        }
      });

    return () => {
      active = false;
    };
  }, [modelInfo?.model_id, target, source, plotType, split, step]);

  if (!host) return null;

  const metadata = state.response?.metadata || {};
  const task = metadata.task || "";
  const classification = task === "classification";
  const cvAvailable = Boolean(metadata.cv_available);
  const cvSplits = metadata.cv_splits || [];

  return createPortal(
    <section className="yy-diagnostic-control">
      <div className="yy-diagnostic-toolbar">
        <label className="compact-select">
          表示
          <select
            value={plotType}
            onChange={(event) => setPlotType(event.target.value)}
          >
            <option value="prediction">
              {classification ? "混同行列" : "実測値 vs 予測値"}
            </option>
            <option value="residual" disabled={classification}>差分プロット</option>
          </select>
        </label>

        <label className="compact-select">
          予測データ
          <select value={source} onChange={(event) => setSource(event.target.value)}>
            <option value="fit">学習済みモデル</option>
            <option value="cv" disabled={!cvAvailable}>
              {classification ? "交差検証予測" : "CV Train / Validation"}
            </option>
          </select>
        </label>

        {classification && source === "cv" && (
          <label className="compact-select">
            CV区分
            <select value={split} onChange={(event) => setSplit(event.target.value)}>
              <option value="train" disabled={!cvSplits.includes("train")}>Train</option>
              <option value="test" disabled={!cvSplits.includes("test")}>Validation</option>
            </select>
          </label>
        )}
      </div>

      <p className="yy-diagnostic-note">
        {cvAvailable
          ? source === "cv"
            ? classification
              ? `yy_plot_ml(cv=True, train_test="${split}") を表示しています。`
              : "yy_plot_ml(cv=True) によりTrainとValidationを同じ図へ表示しています。"
            : "全データで学習した登録モデルの予測を表示しています。"
          : "CV予測はありません。Model画面で精度評価またはモデル比較を実行すると選択できます。"}
      </p>

      <FigureContent state={state} />
    </section>,
    host,
  );
}
