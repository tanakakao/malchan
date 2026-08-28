import React, { useMemo } from "react";
import Chart from "../components/SimpleChart";
import { Field, SectionHeader } from "../components/Common";
import { correlationMatrix } from "../data";
import { useWorkbench } from "../context/WorkbenchContext";
import { useAnalysisColumnSelection } from "../context/useAnalysisColumnSelection";

export default function ExplorePage() {
  const {
    rows,
    columns,
    numeric: rawNumeric,
    chartType,
    setChartType,
    chartX,
    setChartX,
    chartY,
    setChartY,
  } = useWorkbench();
  const { enabledSet } = useAnalysisColumnSelection(columns, rows);
  const numeric = useMemo(
    () => rawNumeric.filter((column) => enabledSet.has(column)),
    [rawNumeric, enabledSet],
  );

  const resolvedChartX = numeric.includes(chartX) ? chartX : numeric[0] || "";
  const resolvedChartY = numeric.includes(chartY)
    ? chartY
    : numeric.find((column) => column !== resolvedChartX) || resolvedChartX;

  let data = [];
  let layout = { title: "データを選択してください", xaxis: {}, yaxis: {} };
  if (chartType === "histogram" && resolvedChartX) {
    data = [{
      type: "histogram",
      x: rows.map((row) => row[resolvedChartX]).filter(Number.isFinite),
      marker: { color: "#6d8cff" },
    }];
    layout = {
      title: `${resolvedChartX} の分布`,
      xaxis: { title: resolvedChartX },
      yaxis: { title: "Count" },
    };
  } else if (chartType === "scatter" && resolvedChartX && resolvedChartY) {
    data = [{
      type: "scatter",
      mode: "markers",
      x: rows.map((row) => row[resolvedChartX]),
      y: rows.map((row) => row[resolvedChartY]),
      marker: { color: "#6d8cff", size: 8 },
    }];
    layout = {
      title: `${resolvedChartX} × ${resolvedChartY}`,
      xaxis: { title: resolvedChartX },
      yaxis: { title: resolvedChartY },
    };
  } else if (chartType === "correlation" && numeric.length) {
    data = [{ type: "heatmap", z: correlationMatrix(rows, numeric), x: numeric, y: numeric }];
    layout = { title: "相関ヒートマップ" };
  }

  return (
    <>
      <SectionHeader
        step="2 · EXPLORE"
        title="データを視覚的に探索する"
        text="分布、関係性、相関構造を確認します。DataでOFFにした列は候補から除外されます。"
      />
      <div className="workspace-two">
        <aside className="settings-card">
          <div className="settings-title"><span>GRAPH SETTINGS</span><h3>描画条件</h3></div>
          <div className="settings-stack">
            <Field label="グラフ">
              <select value={chartType} onChange={(event) => setChartType(event.target.value)}>
                <option value="scatter">散布図</option>
                <option value="histogram">ヒストグラム</option>
                <option value="correlation">相関ヒートマップ</option>
              </select>
            </Field>
            <Field label="X / 対象列">
              <select value={resolvedChartX} onChange={(event) => setChartX(event.target.value)}>
                {numeric.map((column) => <option key={column}>{column}</option>)}
              </select>
            </Field>
            {chartType === "scatter" && (
              <Field label="Y">
                <select value={resolvedChartY} onChange={(event) => setChartY(event.target.value)}>
                  {numeric.map((column) => <option key={column}>{column}</option>)}
                </select>
              </Field>
            )}
          </div>
        </aside>
        <article className="panel canvas-panel">
          {numeric.length
            ? <Chart data={data} layout={layout} />
            : <p className="empty-state">Data画面で数値列を1列以上ONにしてください。</p>}
        </article>
      </div>
    </>
  );
}
