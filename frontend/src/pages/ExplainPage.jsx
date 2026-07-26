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

function mean(values) {
  const finiteValues = values.map(Number).filter(Number.isFinite);
  if (!finiteValues.length) return null;
  return finiteValues.reduce((sum, value) => sum + value, 0) / finiteValues.length;
}

function numericGrid(rows, column, count = 12) {
  const values = rows.map((row) => Number(row[column])).filter(Number.isFinite);
  if (!values.length) return [];
  const min = Math.min(...values);
  const max = Math.max(...values);
  if (min === max) return [min];
  return Array.from({ length: count }, (_, index) => min + ((max - min) * index) / (count - 1));
}

function backgroundRows(rows, maxRows = 20) {
  if (rows.length <= maxRows) return rows;
  const step = Math.max(1, Math.floor(rows.length / maxRows));
  return rows.filter((_, index) => index % step === 0).slice(0, maxRows);
}

function numericPredictionKeys(predictions) {
  const keys = [];
  predictions.slice(0, 20).forEach((record) => {
    Object.entries(record || {}).forEach(([key, value]) => {
      if (Number.isFinite(Number(value)) && !keys.includes(key)) keys.push(key);
    });
  });
  return keys;
}

async function computePdp2d({ modelId, rows, features, xFeature, yFeature, task }) {
  const xValues = numericGrid(rows, xFeature);
  const yValues = numericGrid(rows, yFeature);
  const background = backgroundRows(rows);
  if (xValues.length < 2 || yValues.length < 2 || !background.length) {
    throw new Error("2D PDには値が異なる2つの数値特徴量が必要です。");
  }

  const data = [];
  yValues.forEach((yValue) => {
    xValues.forEach((xValue) => {
      background.forEach((source) => {
        data.push(Object.fromEntries(
          features.map((column) => [
            column,
            column === xFeature ? xValue : column === yFeature ? yValue : source[column],
          ]),
        ));
      });
    });
  });

  const response = await api.predict(modelId, {
    data,
    proba: task === "classification",
    decode_labels: false,
  });
  const predictions = response.predictions || [];
  const outputKeys = numericPredictionKeys(predictions);
  if (!outputKeys.length) throw new Error("2D PDに利用できる数値予測がありません。");

  const cellSize = background.length;
  const series = outputKeys.map((outputName) => {
    let offset = 0;
    const z = yValues.map(() => xValues.map(() => {
      const cell = predictions.slice(offset, offset + cellSize);
      offset += cellSize;
      return mean(cell.map((record) => record?.[outputName]));
    }));
    return { name: outputName, z };
  });

  return {
    x_values: xValues,
    y_values: yValues,
    series,
    background_size: background.length,
  };
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

function BeeswarmPlot({ response, outputName }) {
  const features = response?.features || [];
  const records = response?.records || [];
  const matrix = response?.shap_values?.[outputName] || [];
  if (!features.length || !matrix.length) {
    return <p className="empty-state">Beeswarm用のSHAP値がありません。</p>;
  }

  const ranked = features
    .map((feature, featureIndex) => {
      const values = matrix
        .map((row) => Number(row?.[featureIndex]))
        .filter(Number.isFinite);
      return {
        feature,
        featureIndex,
        score: values.length
          ? values.reduce((sum, value) => sum + Math.abs(value), 0) / values.length
          : 0,
      };
    })
    .sort((a, b) => b.score - a.score)
    .slice(0, 15);
  const maxAbs = Math.max(
    1e-12,
    ...ranked.flatMap(({ featureIndex }) =>
      matrix.map((row) => Math.abs(Number(row?.[featureIndex]))).filter(Number.isFinite)),
  );
  const width = 920;
  const left = 190;
  const right = 34;
  const top = 28;
  const rowHeight = 31;
  const bottom = 48;
  const height = top + ranked.length * rowHeight + bottom;
  const plotWidth = width - left - right;
  const xPosition = (value) => left + ((Number(value) + maxAbs) / (2 * maxAbs)) * plotWidth;
  const categorical = new Set(response.cat_cols || []);

  function pointColor(feature, featureIndex, rowIndex) {
    if (categorical.has(feature)) return "hsl(265 55% 62%)";
    const rawValues = records.map((record) => Number(record?.[feature])).filter(Number.isFinite);
    const value = Number(records[rowIndex]?.[feature]);
    if (!rawValues.length || !Number.isFinite(value)) return "hsl(218 12% 58%)";
    const min = Math.min(...rawValues);
    const max = Math.max(...rawValues);
    const ratio = max === min ? 0.5 : (value - min) / (max - min);
    const hue = 220 - 210 * Math.max(0, Math.min(1, ratio));
    return `hsl(${hue} 78% 56%)`;
  }

  return (
    <div className="beeswarm-chart">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`SHAP beeswarm ${outputName}`}>
        <line x1={xPosition(0)} x2={xPosition(0)} y1={top - 10} y2={height - bottom + 8} className="beeswarm-zero" />
        {ranked.map(({ feature, featureIndex }, rankIndex) => {
          const yCenter = top + rankIndex * rowHeight + rowHeight / 2;
          return (
            <g key={feature}>
              <text x={left - 12} y={yCenter + 4} textAnchor="end" className="beeswarm-label">{feature}</text>
              <line x1={left} x2={width - right} y1={yCenter} y2={yCenter} className="beeswarm-row-line" />
              {matrix.map((row, rowIndex) => {
                const value = Number(row?.[featureIndex]);
                if (!Number.isFinite(value)) return null;
                const jitter = (((rowIndex * 17 + featureIndex * 7) % 13) - 6) * 0.72;
                return (
                  <circle
                    key={`${feature}-${rowIndex}`}
                    cx={xPosition(value)}
                    cy={yCenter + jitter}
                    r="3.1"
                    fill={pointColor(feature, featureIndex, rowIndex)}
                    opacity="0.72"
                  >
                    <title>{`${feature}: SHAP=${formatNumber(value)}, value=${records[rowIndex]?.[feature] ?? "—"}`}</title>
                  </circle>
                );
              })}
            </g>
          );
        })}
        <text x={left} y={height - 20} textAnchor="middle" className="beeswarm-tick">{formatNumber(-maxAbs)}</text>
        <text x={xPosition(0)} y={height - 20} textAnchor="middle" className="beeswarm-tick">0</text>
        <text x={width - right} y={height - 20} textAnchor="middle" className="beeswarm-tick">{formatNumber(maxAbs)}</text>
        <text x={(left + width - right) / 2} y={height - 3} textAnchor="middle" className="beeswarm-axis-title">SHAP value</text>
      </svg>
      <div className="beeswarm-legend"><span>低い特徴量値</span><i /><span>高い特徴量値</span></div>
    </div>
  );
}

export default function ExplainPage() {
  const {
    targets, tasks, rows, features, numeric, diagnostics,
    modelInfo, busy, updateDiagnostics,
  } = useWorkbench();
  const [summary, setSummary] = useState(null);
  const [xaiTarget, setXaiTarget] = useState("");
  const [feature, setFeature] = useState("");
  const [secondFeature, setSecondFeature] = useState("");
  const [method, setMethod] = useState("shap");
  const [pdMode, setPdMode] = useState("1d");
  const [importance, setImportance] = useState(null);
  const [shapValues, setShapValues] = useState(null);
  const [shapOutput, setShapOutput] = useState("");
  const [pdpData, setPdpData] = useState(null);
  const [pdp2dData, setPdp2dData] = useState(null);
  const [pdOutput, setPdOutput] = useState("");
  const [xaiBusy, setXaiBusy] = useState(false);
  const [pd2dBusy, setPd2dBusy] = useState(false);
  const [xaiError, setXaiError] = useState("");

  const diagnosticTarget = xaiTarget || targets[0];
  const actual = diagnostics.map((item) => item.actual[diagnosticTarget]);
  const predicted = diagnostics.map(
    (item) => item.predicted[diagnosticTarget] ?? item.predicted[`pred_${diagnosticTarget}`],
  );

  useEffect(() => {
    let active = true;
    if (!modelInfo?.model_id) {
      setSummary(null);
      setImportance(null);
      setShapValues(null);
      setPdpData(null);
      setPdp2dData(null);
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
      ...(targetSummary?.pdp_features || []),
      ...(targetSummary?.features || []),
    ];
    return [...new Set(values)];
  }, [targetSummary]);
  const pdpFeatures = targetSummary?.pdp_features?.length
    ? targetSummary.pdp_features
    : availableFeatures;
  const numericPdpFeatures = pdpFeatures.filter((column) => numeric.includes(column));

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
    if (!numericPdpFeatures.includes(secondFeature) || secondFeature === first) setSecondFeature(second);
  }, [pdMode, numericPdpFeatures, feature, secondFeature]);

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
    const shapValuesRequest = targetSummary.shap_features.length
      ? api.xaiShapValues(modelInfo.model_id, xaiTarget)
      : Promise.resolve(null);
    const pdpRequest = pdMode === "1d" && feature && targetSummary.pdp_features.includes(feature)
      ? api.xaiPdp(modelInfo.model_id, xaiTarget, feature, { include_ice: false })
      : Promise.resolve(null);

    Promise.allSettled([importanceRequest, shapValuesRequest, pdpRequest])
      .then(([importanceResult, shapResult, pdpResult]) => {
        if (!active) return;
        setImportance(importanceResult.status === "fulfilled" ? importanceResult.value : null);
        setShapValues(shapResult.status === "fulfilled" ? shapResult.value : null);
        setPdpData(pdpResult.status === "fulfilled" ? pdpResult.value : null);
        const failed = [importanceResult, shapResult, pdpResult]
          .find((result) => result.status === "rejected");
        if (failed) setXaiError(failed.reason?.message || String(failed.reason));
      })
      .finally(() => active && setXaiBusy(false));
    return () => { active = false; };
  }, [modelInfo?.model_id, xaiTarget, feature, method, pdMode, targetSummary]);

  useEffect(() => {
    const outputs = shapValues?.output_names || [];
    if (outputs.length && !outputs.includes(shapOutput)) setShapOutput(outputs[0]);
  }, [shapValues, shapOutput]);

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
      setPdp2dData(null);
      return () => { active = false; };
    }
    setPd2dBusy(true);
    setXaiError("");
    computePdp2d({
      modelId: modelInfo.model_id,
      rows,
      features,
      xFeature: feature,
      yFeature: secondFeature,
      task: tasks[xaiTarget],
    })
      .then((response) => active && setPdp2dData(response))
      .catch((error) => active && setXaiError(error.message || String(error)))
      .finally(() => active && setPd2dBusy(false));
    return () => { active = false; };
  }, [pdMode, modelInfo?.model_id, xaiTarget, feature, secondFeature, rows, features, tasks]);

  useEffect(() => {
    const outputs = pdp2dData?.series?.map((series) => series.name) || [];
    if (outputs.length && !outputs.includes(pdOutput)) setPdOutput(outputs[0]);
  }, [pdp2dData, pdOutput]);

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

  const pdpIsNumeric = isNumericSeries(pdpData?.x_values || []);
  const pdpRows = (pdpData?.x_values || []).map((xValue, index) => ({
    [feature]: xValue,
    ...Object.fromEntries(
      (pdpData?.series || []).map((series) => [series.name, series.pd_values[index]]),
    ),
  }));
  const selected2dSeries = pdp2dData?.series?.find((series) => series.name === pdOutput)
    || pdp2dData?.series?.[0];

  return (
    <>
      <SectionHeader
        step="5 · EXPLAIN"
        title="精度とモデル挙動を説明する"
        text="結果図をY-Yプロット、重要度、部分依存の順に縦並びで確認します。"
        action={
          <div className="inline">
            <button className="secondary" disabled={!modelInfo || busy} onClick={updateDiagnostics}>
              Y-Yを更新
            </button>
            <button disabled={!modelInfo || xaiBusy} onClick={recompute}>
              XAIを再計算
            </button>
          </div>
        }
      />

      <article className="panel xai-panel xai-overview">
        <div className="panel-title">
          <div>
            <span className="panel-kicker">RESULT SETTINGS</span>
            <h3>表示対象</h3>
            <p>FastAPIの予測・キャッシュ済みXAIを利用し、表示切替では再学習しません。</p>
          </div>
          <span className={`status-chip ${summary?.status === "ready" ? "success" : ""}`}>
            {xaiBusy ? "読込中" : summary?.status || modelInfo?.xai_status || "未計算"}
          </span>
        </div>
        {!modelInfo && <p className="settings-note">先にModel画面で学習またはモデル比較を実行してください。</p>}
        {modelInfo && summary && summary.status !== "ready" && summary.status !== "partial" && (
          <p className="settings-note">
            XAI状態は「{summary.status}」です。「XAIを再計算」で重要度・Beeswarm・1D PDを作成できます。
          </p>
        )}
        <div className="form-grid xai-controls">
          <label>目的変数
            <select value={xaiTarget} onChange={(event) => setXaiTarget(event.target.value)}>
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
        {targetSummary?.error && <p className="xai-error">{targetSummary.error}</p>}
        {xaiError && <p className="xai-error">{xaiError}</p>}
      </article>

      <div className="xai-results-stack">
        <article className="panel xai-result-panel">
          <div className="xai-result-head"><span>01</span><div><strong>Y-Y PLOT</strong><h3>実測値と予測値</h3></div></div>
          {diagnostics.length ? (
            <Chart
              data={[
                { type: "scatter", mode: "markers", x: actual, y: predicted, marker: { color: "#6d8cff", size: 8 } },
                { type: "scatter", mode: "lines", x: actual, y: actual, line: { color: "#50d09c", dash: "dash" } },
              ]}
              layout={{ title: `Y-Yプロット · ${diagnosticTarget}`, xaxis: { title: "Actual" }, yaxis: { title: "Predicted" } }}
            />
          ) : (
            <p className="empty-state">「Y-Yを更新」で現在の登録モデルによる予測診断を表示します。</p>
          )}
        </article>

        <article className="panel xai-result-panel">
          <div className="xai-result-head"><span>02</span><div><strong>IMPORTANCE</strong><h3>特徴量重要度・SHAP Beeswarm</h3></div></div>
          <section className="xai-card">
            <div className="xai-card-head"><span>FEATURE IMPORTANCE</span><strong>{METHOD_LABELS[method] || method}</strong></div>
            <ImportanceBars response={importance} />
          </section>
          <section className="xai-card beeswarm-card">
            <div className="xai-card-head">
              <span>SHAP BEESWARM</span>
              <label className="compact-select">出力
                <select value={shapOutput} onChange={(event) => setShapOutput(event.target.value)}>
                  {(shapValues?.output_names || []).map((output) => <option key={output}>{output}</option>)}
                </select>
              </label>
            </div>
            <BeeswarmPlot response={shapValues} outputName={shapOutput} />
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
            {pdMode === "2d" && pdp2dData?.series?.length > 1 && (
              <label>予測出力
                <select value={pdOutput} onChange={(event) => setPdOutput(event.target.value)}>
                  {pdp2dData.series.map((series) => <option key={series.name}>{series.name}</option>)}
                </select>
              </label>
            )}
          </div>

          <section className="xai-card pd-result-card">
            {pdMode === "1d" && (pdpData?.series?.length ? (
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
                  layout={{ title: `1D PD · ${feature}`, xaxis: { title: feature }, yaxis: { title: "Prediction" } }}
                />
              ) : (
                <DataTable rows={pdpRows} columns={[feature, ...pdpData.series.map((series) => series.name)]} pageSize={20} />
              )
            ) : <p className="empty-state">この特徴量の1D PDは利用できません。</p>)}

            {pdMode === "2d" && (pd2dBusy ? (
              <p className="empty-state">現在のFastAPI予測APIで2D PDを計算しています...</p>
            ) : selected2dSeries ? (
              <>
                <Chart
                  data={[{
                    type: "heatmap",
                    x: pdp2dData.x_values,
                    y: pdp2dData.y_values,
                    z: selected2dSeries.z,
                  }]}
                  layout={{
                    title: `2D PD · ${feature} × ${secondFeature} · ${selected2dSeries.name}`,
                    xaxis: { title: feature },
                    yaxis: { title: secondFeature },
                  }}
                />
                <p className="pd-footnote">背景サンプル {pdp2dData.background_size}件の予測平均。表示時のみFastAPIの予測APIを呼び出します。</p>
              </>
            ) : (
              <p className="empty-state">2D PDには2つ以上の数値特徴量が必要です。</p>
            ))}
          </section>
        </article>
      </div>
    </>
  );
}
