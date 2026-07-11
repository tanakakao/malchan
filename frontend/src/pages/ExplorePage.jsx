import React from "react";
import Chart from "../components/SimpleChart";
import { Field, SectionHeader } from "../components/Common";
import { correlationMatrix } from "../data";
import { useWorkbench } from "../context/WorkbenchContext";

export default function ExplorePage() {
  const {
    rows, numeric, chartType, setChartType, chartX, setChartX, chartY, setChartY,
  } = useWorkbench();

  let data = [];
  let layout = { title: "データを選択してください", xaxis: {}, yaxis: {} };
  if (chartType === "histogram" && chartX) {
    data = [{
      type: "histogram",
      x: rows.map((row) => row[chartX]).filter(Number.isFinite),
      marker: { color: "#6d8cff" },
    }];
    layout = { title: `${chartX} の分布`, xaxis: { title: chartX }, yaxis: { title: "Count" } };
  } else if (chartType === "scatter" && chartX && chartY) {
    data = [{
      type: "scatter",
      mode: "markers",
      x: rows.map((row) => row[chartX]),
      y: rows.map((row) => row[chartY]),
      marker: { color: "#6d8cff", size: 8 },
    }];
    layout = { title: `${chartX} × ${chartY}`, xaxis: { title: chartX }, yaxis: { title: chartY } };
  } else if (chartType === "correlation" && numeric.length) {
    data = [{ type: "heatmap", z: correlationMatrix(rows, numeric), x: numeric, y: numeric }];
    layout = { title: "相関ヒートマップ" };
  }

  return (
    <>
      <SectionHeader
        step="2 · EXPLORE"
        title="データを視覚的に探索する"
        text="分布、関係性、相関構造を確認します。"
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
              <select value={chartX} onChange={(event) => setChartX(event.target.value)}>
                {numeric.map((column) => <option key={column}>{column}</option>)}
              </select>
            </Field>
            {chartType === "scatter" && (
              <Field label="Y">
                <select value={chartY} onChange={(event) => setChartY(event.target.value)}>
                  {numeric.map((column) => <option key={column}>{column}</option>)}
                </select>
              </Field>
            )}
          </div>
        </aside>
        <article className="panel canvas-panel"><Chart data={data} layout={layout} /></article>
      </div>
    </>
  );
}
