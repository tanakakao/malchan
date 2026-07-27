import React, { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import { SectionHeader } from "../components/Common";
import PlotlyFigure from "../components/PlotlyFigure";
import { useWorkbench } from "../context/WorkbenchContext";

const METHOD_LABELS = {
  shap: "SHAP重要度",
  pfi: "Permutation Importance",
  model: "モデル固有重要度",
};

const emptyFigureState = () => ({ response: null, loading: false, error: "" });

function FigurePanel({ state, emptyText }) {
  if (state.loading) return <p className="empty-state">visualizationでPlotly図を生成しています...</p>;
  if (state.error) return <p className="xai-error">{state.error}</p>;
  if (!state.response?.figure) return <p className="empty-state">{emptyText}</p>;
  return <PlotlyFigure figure={state.response.figure} />;
}

export default function ExplainPage() {
  const { targets, numeric, modelInfo, busy } = useWorkbench();
  const [summary, setSummary] = useState(null);
  const [xaiTarget, setXaiTarget] = useState("");
  const [method, setMethod] = useState("shap");
  const [pdMode, setPdMode] = useState("1d");
  const [feature, setFeature] = useState("");
  const [secondFeature, setSecondFeature] = useState("");
  const [includeIce, setIncludeIce] = useState(false);
  const [beeswarmOutput, setBeeswarmOutput] = useState("");
  const [pdOutput, setPdOutput] = useState("");
  const [pd2dOutput, setPd2dOutput] = useState("");
  const [refreshKey, setRefreshKey] = useState(0);
  const [summaryBusy, setSummaryBusy] = useState(false);
  const [summaryError, setSummaryError] = useState("");
  const [yyFigure, setYyFigure] = useState(emptyFigureState);
  const [importanceFigure, setImportanceFigure] = useState(emptyFigureState);
  const [beeswarmFigure, setBeeswarmFigure] = useState(emptyFigureState);
  const [pdpFigure, setPdpFigure] = useState(emptyFigureState);
  const [pdp2dFigure, setPdp2dFigure] = useState(emptyFigureState);

  function resetOutputs() {
    setBeeswarmOutput("");
    setPdOutput("");
    setPd2dOutput("");
  }

  function changeTarget(nextTarget) {
    resetOutputs();
    setXaiTarget(nextTarget);
  }

  useEffect(() => {
    let active = true;
    if (!modelInfo?.model_id) {
      setSummary(null);
      setXaiTarget("");
      resetOutputs();
      return () => { active = false; };
    }
    setSummaryBusy(true);
    setSummaryError("");
    api.xaiSummary(modelInfo.model_id)
      .then((response) => {
        if (!active) return;
        setSummary(response);
        const nextTarget = response.targets?.[xaiTarget]
          ? xaiTarget
          : Object.keys(response.targets || {})[0] || targets[0] || "";
        if (nextTarget !== xaiTarget) resetOutputs();
        setXaiTarget(nextTarget);
      })
      .catch((error) => active && setSummaryError(error.message || String(error)))
      .finally(() => active && setSummaryBusy(false));
    return () => { active = false; };
  }, [modelInfo?.model_id, modelInfo?.xai_status, refreshKey]);

  const targetSummary = summary?.targets?.[xaiTarget] || null;
  const pdpFeatures = useMemo(() => {
    const values = [
      ...(targetSummary?.pdp_features || []),
      ...(targetSummary?.features || []),
    ];
    return [...new Set(values)];
  }, [targetSummary]);
  const numericPdpFeatures = useMemo(
    () => pdpFeatures.filter((column) => numeric.includes(column)),
    [pdpFeatures, numeric],
  );
  const xaiReady = targetSummary?.status === "ready";

  useEffect(() => {
    if (!pdpFeatures.length) {
      setFeature("");
      return;
    }
    if (!pdpFeatures.includes(feature)) setFeature(pdpFeatures[0]);
  }, [pdpFeatures, feature]);

  useEffect(() => {
    if (pdMode !== "2d") return;
    const first = numericPdpFeatures.includes(feature) ? feature : numericPdpFeatures[0] || "";
    const second = numericPdpFeatures.find((column) => column !== first) || "";
    if (feature !== first) setFeature(first);
    if (!numericPdpFeatures.includes(secondFeature) || secondFeature === first) {
      setSecondFeature(second);
    }
  }, [pdMode, numericPdpFeatures, feature, secondFeature]);

  useEffect(() => {
    const methods = targetSummary?.importance_methods || [];
    if (methods.length && !methods.includes(method)) setMethod(methods[0]);
  }, [targetSummary, method]);

  useEffect(() => {
    let active = true;
    if (!modelInfo?.model_id || !xaiTarget) {
      setYyFigure(emptyFigureState());
      return () => { active = false; };
    }
    setYyFigure((state) => ({ ...state, loading: true, error: "" }));
    api.visualizationYy(modelInfo.model_id, xaiTarget)
      .then((response) => active && setYyFigure({ response, loading: false, error: "" }))
      .catch((error) => active && setYyFigure({ response: null, loading: false, error: error.message || String(error) }));
    return () => { active = false; };
  }, [modelInfo?.model_id, xaiTarget, refreshKey]);

  useEffect(() => {
    let active = true;
    if (!modelInfo?.model_id || !xaiTarget || !xaiReady) {
      setImportanceFigure(emptyFigureState());
      return () => { active = false; };
    }
    setImportanceFigure((state) => ({ ...state, loading: true, error: "" }));
    api.visualizationImportance(modelInfo.model_id, xaiTarget, {
      method,
      combined: true,
      top_n: 20,
    })
      .then((response) => active && setImportanceFigure({ response, loading: false, error: "" }))
      .catch((error) => active && setImportanceFigure({ response: null, loading: false, error: error.message || String(error) }));
    return () => { active = false; };
  }, [modelInfo?.model_id, xaiTarget, xaiReady, method, refreshKey]);

  useEffect(() => {
    let active = true;
    if (!modelInfo?.model_id || !xaiTarget || !xaiReady) {
      setBeeswarmFigure(emptyFigureState());
      return () => { active = false; };
    }
    setBeeswarmFigure((state) => ({ ...state, loading: true, error: "" }));
    api.visualizationBeeswarm(modelInfo.model_id, xaiTarget, {
      output: beeswarmOutput,
      top_n: 15,
    })
      .then((response) => {
        if (!active) return;
        setBeeswarmFigure({ response, loading: false, error: "" });
        if (!beeswarmOutput && response.metadata?.selected_output) {
          setBeeswarmOutput(response.metadata.selected_output);
        }
      })
      .catch((error) => active && setBeeswarmFigure({ response: null, loading: false, error: error.message || String(error) }));
    return () => { active = false; };
  }, [modelInfo?.model_id, xaiTarget, xaiReady, beeswarmOutput, refreshKey]);

  useEffect(() => {
    let active = true;
    if (pdMode !== "1d" || !modelInfo?.model_id || !xaiTarget || !feature || !xaiReady) {
      setPdpFigure(emptyFigureState());
      return () => { active = false; };
    }
    setPdpFigure((state) => ({ ...state, loading: true, error: "" }));
    api.visualizationPdp(modelInfo.model_id, xaiTarget, {
      feature,
      output: pdOutput,
      include_ice: includeIce,
      max_ice: 30,
    })
      .then((response) => {
        if (!active) return;
        setPdpFigure({ response, loading: false, error: "" });
        if (!pdOutput && response.metadata?.selected_output) {
          setPdOutput(response.metadata.selected_output);
        }
      })
      .catch((error) => active && setPdpFigure({ response: null, loading: false, error: error.message || String(error) }));
    return () => { active = false; };
  }, [pdMode, modelInfo?.model_id, xaiTarget, feature, xaiReady, pdOutput, includeIce, refreshKey]);

  useEffect(() => {
    let active = true;
    if (
      pdMode !== "2d"
      || !modelInfo?.model_id
      || !xaiTarget
      || !feature
      || !secondFeature
      || feature === secondFeature
    ) {
      setPdp2dFigure(emptyFigureState());
      return () => { active = false; };
    }
    setPdp2dFigure((state) => ({ ...state, loading: true, error: "" }));
    api.visualizationPdp2d(modelInfo.model_id, xaiTarget, {
      feature_x: feature,
      feature_y: secondFeature,
      output: pd2dOutput,
    })
      .then((response) => {
        if (!active) return;
        setPdp2dFigure({ response, loading: false, error: "" });
        if (!pd2dOutput && response.metadata?.selected_output) {
          setPd2dOutput(response.metadata.selected_output);
        }
      })
      .catch((error) => active && setPdp2dFigure({ response: null, loading: false, error: error.message || String(error) }));
    return () => { active = false; };
  }, [pdMode, modelInfo?.model_id, xaiTarget, feature, secondFeature, pd2dOutput, refreshKey]);

  async function recompute() {
    if (!modelInfo?.model_id) return;
    setSummaryBusy(true);
    setSummaryError("");
    try {
      const response = await api.recomputeXai(modelInfo.model_id, {
        targets: xaiTarget ? [xaiTarget] : [],
      });
      setSummary(response);
      setRefreshKey((value) => value + 1);
    } catch (error) {
      setSummaryError(error.message || String(error));
    } finally {
      setSummaryBusy(false);
    }
  }

  const beeswarmOutputs = beeswarmFigure.response?.metadata?.outputs || [];
  const pdOutputs = pdpFigure.response?.metadata?.outputs || [];
  const pd2dOutputs = pdp2dFigure.response?.metadata?.outputs || [];

  return (
    <>
      <SectionHeader
        step="5 · EXPLAIN"
        title="精度とモデル挙動を説明する"
        text="すべての結果図をmalchan.visualizationで生成し、Plotly図として縦並びで表示します。"
        action={
          <div className="inline">
            <button className="secondary" disabled={!modelInfo || busy} onClick={() => setRefreshKey((value) => value + 1)}>
              図を更新
            </button>
            <button disabled={!modelInfo || summaryBusy} onClick={recompute}>
              XAIを再計算
            </button>
          </div>
        }
      />

      <article className="panel xai-panel xai-overview">
        <div className="panel-title">
          <div>
            <span className="panel-kicker">VISUALIZATION SETTINGS</span>
            <h3>表示対象</h3>
            <p>FastAPIはmalchan.visualizationのFigureをPlotly JSONとして返します。</p>
          </div>
          <span className={`status-chip ${summary?.status === "ready" ? "success" : ""}`}>
            {summaryBusy ? "読込中" : summary?.status || modelInfo?.xai_status || "未計算"}
          </span>
        </div>
        {!modelInfo && <p className="settings-note">先にModel画面で学習またはモデル比較を実行してください。</p>}
        {modelInfo && targetSummary && !xaiReady && (
          <p className="settings-note">
            Y-Yと2D PDは表示できます。重要度・Beeswarm・1D PDは「XAIを再計算」後に表示します。
          </p>
        )}
        <div className="form-grid xai-controls">
          <label>目的変数
            <select value={xaiTarget} onChange={(event) => changeTarget(event.target.value)}>
              {Object.keys(summary?.targets || {}).map((target) => <option key={target}>{target}</option>)}
            </select>
          </label>
          <label>重要度
            <select value={method} onChange={(event) => setMethod(event.target.value)}>
              {(targetSummary?.importance_methods || ["shap", "pfi", "model"]).map((value) => (
                <option key={value} value={value}>{METHOD_LABELS[value] || value}</option>
              ))}
            </select>
          </label>
        </div>
        {summaryError && <p className="xai-error">{summaryError}</p>}
      </article>

      <div className="xai-results-stack">
        <article className="panel xai-result-panel">
          <div className="xai-result-head"><span>01</span><div><strong>Y-Y PLOT</strong><h3>実測値と予測値</h3></div></div>
          <FigurePanel state={yyFigure} emptyText="登録モデルの診断図を表示できません。" />
        </article>

        <article className="panel xai-result-panel">
          <div className="xai-result-head"><span>02</span><div><strong>IMPORTANCE</strong><h3>特徴量重要度・SHAP Beeswarm</h3></div></div>
          <section className="xai-card">
            <div className="xai-card-head"><span>FEATURE IMPORTANCE</span><strong>{METHOD_LABELS[method] || method}</strong></div>
            <FigurePanel state={importanceFigure} emptyText="XAI計算後に重要度を表示します。" />
          </section>
          <section className="xai-card beeswarm-card">
            <div className="xai-card-head">
              <span>SHAP BEESWARM</span>
              {beeswarmOutputs.length > 1 && (
                <label className="compact-select">出力
                  <select value={beeswarmOutput} onChange={(event) => setBeeswarmOutput(event.target.value)}>
                    {beeswarmOutputs.map((output) => <option key={output}>{output}</option>)}
                  </select>
                </label>
              )}
            </div>
            <FigurePanel state={beeswarmFigure} emptyText="XAI計算後にBeeswarmを表示します。" />
          </section>
        </article>

        <article className="panel xai-result-panel">
          <div className="xai-result-head"><span>03</span><div><strong>PARTIAL DEPENDENCE</strong><h3>PD 1D / 2D</h3></div></div>
          <div className="pd-toolbar">
            <div className="pd-mode-toggle">
              <button className={pdMode === "1d" ? "active" : ""} onClick={() => setPdMode("1d")}>1D</button>
              <button className={pdMode === "2d" ? "active" : ""} onClick={() => setPdMode("2d")}>2D</button>
            </div>
            <label>特徴量 X
              <select value={feature} onChange={(event) => setFeature(event.target.value)}>
                {(pdMode === "2d" ? numericPdpFeatures : pdpFeatures).map((column) => <option key={column}>{column}</option>)}
              </select>
            </label>
            {pdMode === "2d" && (
              <label>特徴量 Y
                <select value={secondFeature} onChange={(event) => setSecondFeature(event.target.value)}>
                  {numericPdpFeatures.filter((column) => column !== feature).map((column) => <option key={column}>{column}</option>)}
                </select>
              </label>
            )}
            {pdMode === "1d" && pdOutputs.length > 1 && (
              <label>予測出力
                <select value={pdOutput} onChange={(event) => setPdOutput(event.target.value)}>
                  {pdOutputs.map((output) => <option key={output}>{output}</option>)}
                </select>
              </label>
            )}
            {pdMode === "2d" && pd2dOutputs.length > 1 && (
              <label>予測出力
                <select value={pd2dOutput} onChange={(event) => setPd2dOutput(event.target.value)}>
                  {pd2dOutputs.map((output) => <option key={output}>{output}</option>)}
                </select>
              </label>
            )}
            {pdMode === "1d" && (
              <label className="switch-label"><input type="checkbox" checked={includeIce} onChange={(event) => setIncludeIce(event.target.checked)} /><span />ICEを表示</label>
            )}
          </div>
          <section className="xai-card pd-result-card">
            {pdMode === "1d" ? (
              <FigurePanel state={pdpFigure} emptyText="XAI計算後に1D PDを表示します。" />
            ) : (
              <FigurePanel state={pdp2dFigure} emptyText="2D PDには異なる2つの数値特徴量が必要です。" />
            )}
          </section>
        </article>
      </div>
    </>
  );
}
