import React, { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import Chart from "../components/SimpleChart";
import DataTable from "../components/DataTable";
import { SectionHeader } from "../components/Common";
import { formatNumber } from "../data";
import { useWorkbench } from "../context/WorkbenchContext";

const METHOD_LABELS = {
  shap: "SHAP重要度",
  pfi: "Permutation Importance",
  model: "モデル固有重要度",
};

function isNumericSeries(values) {
  return values.length > 0 && values.every((value) => Number.isFinite(Number(value)));
}

function ImportanceBars({ response }) {
  const items = response?.items || [];
  const maxValue = Math.max(1e-12, ...items.map((item) => Math.abs(item.value)));
  if (!items.length) return <p className="empty-state">利用可能な重要度がありません。</p>;
  return (
    <div className="xai-importance-list">
      {items.map((item) => (
        <div className="xai-importance-row" key={item.feature}>
          <span title={item.feature}>{item.feature}</span>
          <div><i style={{ width: `${(Math.abs(item.value) / maxValue) * 100}%` }} /></div>
          <strong>{formatNumber(item.value)}</strong>
        </div>
      ))}
    </div>
  );
}

export default function ExplainPage() {
  const { targets, diagnostics, modelInfo, busy, updateDiagnostics } = useWorkbench();
  const diagnosticTarget = targets[0];
  const actual = diagnostics.map((item) => item.actual[diagnosticTarget]);
  const predicted = diagnostics.map(
    (item) => item.predicted[diagnosticTarget] ?? item.predicted[`pred_${diagnosticTarget}`],
  );
  const residual = actual.map((value, index) => Number(value) - Number(predicted[index]));

  const [summary, setSummary] = useState(null);
  const [xaiTarget, setXaiTarget] = useState("");
  const [feature, setFeature] = useState("");
  const [method, setMethod] = useState("shap");
  const [importance, setImportance] = useState(null);
  const [shapData, setShapData] = useState(null);
  const [pdpData, setPdpData] = useState(null);
  const [xaiBusy, setXaiBusy] = useState(false);
  const [xaiError, setXaiError] = useState("");

  useEffect(() => {
    let active = true;
    if (!modelInfo?.model_id) {
      setSummary(null);
      setImportance(null);
      setShapData(null);
      setPdpData(null);
      return () => { active = false; };
    }
    setXaiBusy(true);
    api.xaiSummary(modelInfo.model_id)
      .then((response) => {
        if (!active) return;
        setSummary(response);
        const firstTarget = response.targets?.[xaiTarget]
          ? xaiTarget
          : Object.keys(response.targets || {})[0] || "";
        setXaiTarget(firstTarget);
        setXaiError("");
      })
      .catch((error) => active && setXaiError(error.message))
      .finally(() => active && setXaiBusy(false));
    return () => { active = false; };
  }, [modelInfo?.model_id, modelInfo?.xai_status]);

  const targetSummary = summary?.targets?.[xaiTarget] || null;
  const availableFeatures = useMemo(() => {
    const values = [
      ...(targetSummary?.shap_features || []),
      ...(targetSummary?.pdp_features || []),
      ...(targetSummary?.features || []),
    ];
    return [...new Set(values)];
  }, [targetSummary]);

  useEffect(() => {
    if (!availableFeatures.length) {
      setFeature("");
      return;
    }
    if (!availableFeatures.includes(feature)) setFeature(availableFeatures[0]);
  }, [availableFeatures, feature]);

  useEffect(() => {
    const methods = targetSummary?.importance_methods || [];
    if (methods.length && !methods.includes(method)) setMethod(methods[0]);
  }, [targetSummary, method]);

  useEffect(() => {
    let active = true;
    if (!modelInfo?.model_id || !xaiTarget || !targetSummary) return undefined;
    setXaiBusy(true);
    setXaiError("");
    const importanceRequest = targetSummary.importance_methods.includes(method)
      ? api.xaiImportance(modelInfo.model_id, xaiTarget, {
          method,
          combined: true,
          top_n: 20,
        })
      : Promise.resolve(null);
    const shapRequest = feature && targetSummary.shap_features.includes(feature)
      ? api.xaiShap(modelInfo.model_id, xaiTarget, feature)
      : Promise.resolve(null);
    const pdpRequest = feature && targetSummary.pdp_features.includes(feature)
      ? api.xaiPdp(modelInfo.model_id, xaiTarget, feature, { include_ice: false })
      : Promise.resolve(null);

    Promise.allSettled([importanceRequest, shapRequest, pdpRequest])
      .then(([importanceResult, shapResult, pdpResult]) => {
        if (!active) return;
        setImportance(importanceResult.status === "fulfilled" ? importanceResult.value : null);
        setShapData(shapResult.status === "fulfilled" ? shapResult.value : null);
        setPdpData(pdpResult.status === "fulfilled" ? pdpResult.value : null);
        const failed = [importanceResult, shapResult, pdpResult]
          .find((result) => result.status === "rejected");
        if (failed) setXaiError(failed.reason?.message || String(failed.reason));
      })
      .finally(() => active && setXaiBusy(false));
    return () => { active = false; };
  }, [modelInfo?.model_id, xaiTarget, feature, method, targetSummary]);

  async function recompute() {
    if (!modelInfo?.model_id) return;
    setXaiBusy(true);
    setXaiError("");
    try {
      const response = await api.recomputeXai(modelInfo.model_id, {
        targets: xaiTarget ? [xaiTarget] : [],
      });
      setSummary(response);
    } catch (error) {
      setXaiError(error.message || String(error));
    } finally {
      setXaiBusy(false);
    }
  }

  const shapColumn = shapData?.value_columns?.[0];
  const shapX = shapData?.records?.map((record) => record[feature]) || [];
  const shapY = shapData?.records?.map((record) => record[shapColumn]) || [];
  const shapIsNumeric = isNumericSeries(shapX);
  const pdpIsNumeric = isNumericSeries(pdpData?.x_values || []);
  const pdpRows = (pdpData?.x_values || []).map((xValue, index) => ({
    [feature]: xValue,
    ...Object.fromEntries(
      (pdpData?.series || []).map((series) => [series.name, series.pd_values[index]]),
    ),
  }));

  return (
    <>
      <SectionHeader
        step="5 · EXPLAIN"
        title="精度とモデル挙動を説明する"
        text="予測診断と、学習時に事前計算したSHAP・重要度・PDPを確認します。"
        action={
          <div className="inline">
            <button className="secondary" disabled={!modelInfo || busy} onClick={updateDiagnostics}>
              予測診断を更新
            </button>
            <button disabled={!modelInfo || xaiBusy} onClick={recompute}>
              XAIを再計算
            </button>
          </div>
        }
      />

      {diagnostics.length ? (
        <div className="split">
          <article className="panel">
            <Chart
              data={[
                { type: "scatter", mode: "markers", x: actual, y: predicted, marker: { color: "#6d8cff", size: 8 } },
                { type: "scatter", mode: "lines", x: actual, y: actual, line: { color: "#50d09c", dash: "dash" } },
              ]}
              layout={{ title: "Y-Yプロット", xaxis: { title: "Actual" }, yaxis: { title: "Predicted" } }}
            />
          </article>
          <article className="panel">
            <Chart
              data={[{ type: "scatter", mode: "markers", x: predicted, y: residual, marker: { color: "#f0b85b", size: 8 } }]}
              layout={{ title: "残差プロット", xaxis: { title: "Predicted" }, yaxis: { title: "Actual - Predicted" } }}
            />
          </article>
        </div>
      ) : (
        <article className="panel"><p className="empty-state">予測診断を更新するとY-Y図と残差を表示します。</p></article>
      )}

      <article className="panel xai-panel">
        <div className="panel-title">
          <div>
            <span className="panel-kicker">CACHED XAI</span>
            <h3>SHAP・重要度・部分依存</h3>
            <p>通常の表示操作では再計算せず、モデル作成時に保存した結果を取得します。</p>
          </div>
          <span className={`status-chip ${summary?.status === "ready" ? "success" : ""}`}>
            {xaiBusy ? "読込中" : summary?.status || modelInfo?.xai_status || "未計算"}
          </span>
        </div>

        {!modelInfo && <p className="settings-note">先にModel画面でモデルを作成してください。</p>}
        {modelInfo && summary && summary.status !== "ready" && summary.status !== "partial" && (
          <p className="settings-note">
            XAI状態は「{summary.status}」です。詳細を確認し、必要な場合だけ「XAIを再計算」を実行してください。
          </p>
        )}

        <div className="form-grid xai-controls">
          <label>目的変数
            <select value={xaiTarget} onChange={(event) => setXaiTarget(event.target.value)}>
              {Object.keys(summary?.targets || {}).map((target) => <option key={target}>{target}</option>)}
            </select>
          </label>
          <label>特徴量
            <select value={feature} onChange={(event) => setFeature(event.target.value)}>
              {availableFeatures.map((column) => <option key={column}>{column}</option>)}
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

        {targetSummary?.error && <p className="xai-error">{targetSummary.error}</p>}
        {xaiError && <p className="xai-error">{xaiError}</p>}

        <div className="xai-grid">
          <section className="xai-card">
            <div className="xai-card-head"><span>IMPORTANCE</span><strong>{METHOD_LABELS[method]}</strong></div>
            <ImportanceBars response={importance} />
          </section>

          <section className="xai-card">
            <div className="xai-card-head"><span>SHAP</span><strong>{feature || "特徴量を選択"}</strong></div>
            {shapData && shapColumn ? (
              shapIsNumeric ? (
                <Chart
                  data={[{ type: "scatter", mode: "markers", x: shapX, y: shapY, marker: { color: "#6d8cff", size: 7 } }]}
                  layout={{ title: `SHAP dependence · ${feature}`, xaxis: { title: feature }, yaxis: { title: shapColumn } }}
                />
              ) : (
                <DataTable rows={shapData.records} columns={[feature, ...shapData.value_columns]} pageSize={12} />
              )
            ) : <p className="empty-state">この特徴量のSHAPデータは利用できません。</p>}
          </section>

          <section className="xai-card xai-pdp-card">
            <div className="xai-card-head"><span>PARTIAL DEPENDENCE</span><strong>{feature || "特徴量を選択"}</strong></div>
            {pdpData?.series?.length ? (
              pdpIsNumeric ? (
                <Chart
                  data={pdpData.series.map((series, index) => ({
                    type: "scatter",
                    mode: "lines",
                    name: series.name,
                    x: pdpData.x_values,
                    y: series.pd_values,
                    line: { color: index ? "#50d09c" : "#6d8cff" },
                  }))}
                  layout={{ title: `PDP · ${feature}`, xaxis: { title: feature }, yaxis: { title: "Prediction" } }}
                />
              ) : (
                <DataTable rows={pdpRows} columns={[feature, ...pdpData.series.map((series) => series.name)]} pageSize={20} />
              )
            ) : <p className="empty-state">この特徴量のPDPは利用できません。</p>}
          </section>
        </div>
      </article>
    </>
  );
}
