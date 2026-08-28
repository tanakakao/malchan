import React, { useMemo, useState } from "react";
import DataTable from "../components/DataTable";
import { SectionHeader } from "../components/Common";
import { useWorkbench } from "../context/WorkbenchContext";
import { useAnalysisColumnSelection } from "../context/useAnalysisColumnSelection";
import "../data-column-selection.css";

export default function DataPage() {
  const {
    rows,
    columns,
    numeric,
    categorical,
    fileName,
    stats,
    targets,
    numFeatures,
    setNumFeatures,
    catFeatures,
    setCatFeatures,
    chartX,
    setChartX,
    chartY,
    setChartY,
    modelInfo,
    busy,
    loadFile,
    loadModelBundle,
    changeTargets,
  } = useWorkbench();
  const [dataDragging, setDataDragging] = useState(false);
  const [modelDragging, setModelDragging] = useState(false);
  const dataLoaded = rows.length > 0;
  const modelLoaded = Boolean(modelInfo?.model_id) && !dataLoaded;
  const columnSelectionLocked = Boolean(modelInfo?.model_id) && dataLoaded;
  const {
    enabledColumns,
    enabledSet,
    setEnabledColumns,
  } = useAnalysisColumnSelection(columns, rows);

  const visibleStats = useMemo(
    () => stats.filter((item) => enabledSet.has(item.column)),
    [stats, enabledSet],
  );

  function loadDataFile(file) {
    if (file) loadFile(file);
  }

  async function loadSavedModel(file, input) {
    try {
      if (!file) return;
      const trusted = window.confirm(
        "保存モデルの読込にはpickle形式のデータを使用します。\n\n"
        + "malchanから自分で保存したものなど、作成元を信頼できるファイルだけを選択してください。\n\n"
        + "このモデルファイルを読み込みますか？",
      );
      if (trusted) await loadModelBundle(file);
    } finally {
      input.value = "";
    }
  }

  function applyEnabledColumns(nextColumns) {
    if (columnSelectionLocked) return;

    const nextEnabled = columns.filter((column) => nextColumns.includes(column));
    const nextSet = new Set(nextEnabled);
    const nextNumeric = numeric.filter((column) => nextSet.has(column));
    const nextChartX = nextSet.has(chartX) ? chartX : nextNumeric[0] || "";
    const nextChartY = nextSet.has(chartY)
      ? chartY
      : nextNumeric.find((column) => column !== nextChartX) || nextChartX;

    setEnabledColumns(nextEnabled);
    changeTargets(targets.filter((target) => nextSet.has(target)));
    setNumFeatures(numFeatures.filter((column) => nextSet.has(column)));
    setCatFeatures(catFeatures.filter((column) => nextSet.has(column)));
    setChartX(nextChartX);
    setChartY(nextChartY);
  }

  function toggleColumn(column) {
    const next = enabledSet.has(column)
      ? enabledColumns.filter((name) => name !== column)
      : columns.filter((name) => name === column || enabledSet.has(name));
    applyEnabledColumns(next);
  }

  function columnRole(column) {
    if (targets.includes(column)) return "目的変数";
    if (numFeatures.includes(column) || catFeatures.includes(column)) return "説明変数";
    return "未選択";
  }

  function columnKind(column) {
    if (numeric.includes(column)) return "numeric";
    if (categorical.includes(column)) return "categorical";
    return "other";
  }

  return (
    <>
      <SectionHeader
        step="1 · DATA"
        title="データまたは保存モデルを読み込む"
        text="CSV / XLSXから新しく学習するか、保存したmalchanモデルから作業を再開します。"
      />

      <div className="data-source-grid">
        <article className="panel data-file-panel">
          <div className="panel-title">
            <div>
              <span className="panel-kicker">DATA SOURCE</span>
              <h3>{dataLoaded ? "データを入れ替える" : "データファイル"}</h3>
              <p>対応形式: CSV / XLSX</p>
            </div>
            {dataLoaded && <span className="status-chip success">Loaded</span>}
          </div>
          <label
            className={`dropzone ${dataDragging ? "dragging" : ""}`}
            aria-label="CSVまたはXLSXファイルをドロップまたは選択"
            onDragEnter={(event) => {
              event.preventDefault();
              setDataDragging(true);
            }}
            onDragOver={(event) => {
              event.preventDefault();
              event.dataTransfer.dropEffect = "copy";
              setDataDragging(true);
            }}
            onDragLeave={() => setDataDragging(false)}
            onDrop={(event) => {
              event.preventDefault();
              setDataDragging(false);
              loadDataFile(event.dataTransfer.files[0]);
            }}
          >
            <input
              type="file"
              accept=".csv,.xlsx"
              disabled={Boolean(busy)}
              onChange={(event) => loadDataFile(event.target.files?.[0])}
            />
            <span className="upload-symbol">⇧</span>
            <strong>
              {dataDragging
                ? "ここにドロップして読み込む"
                : dataLoaded
                  ? "別のファイルをドロップまたは選択"
                  : "CSVまたはExcelをドロップまたは選択"}
            </strong>
            <span>
              {dataLoaded && fileName
                ? `読込中のファイル: ${fileName}`
                : "ファイルはブラウザで解析され、現在のワークスペース内に保持されます。"}
            </span>
          </label>
        </article>

        <article className="panel model-artifact-panel">
          <div className="panel-title">
            <div>
              <span className="panel-kicker">SAVED MODEL</span>
              <h3>保存モデルを読み込む</h3>
              <p>対応形式: .malchan</p>
            </div>
            {modelLoaded && <span className="status-chip success">Restored</span>}
          </div>
          <label
            className={`dropzone model-dropzone ${modelDragging ? "dragging" : ""}`}
            aria-label="malchanモデルファイルをドロップまたは選択"
            onDragEnter={(event) => {
              event.preventDefault();
              setModelDragging(true);
            }}
            onDragOver={(event) => {
              event.preventDefault();
              event.dataTransfer.dropEffect = "copy";
              setModelDragging(true);
            }}
            onDragLeave={() => setModelDragging(false)}
            onDrop={(event) => {
              event.preventDefault();
              setModelDragging(false);
              const file = event.dataTransfer.files[0];
              if (!file) return;
              const proxyInput = { value: "" };
              void loadSavedModel(file, proxyInput);
            }}
          >
            <input
              type="file"
              accept=".malchan,application/vnd.malchan.model"
              disabled={Boolean(busy)}
              onChange={(event) => void loadSavedModel(
                event.target.files?.[0],
                event.currentTarget,
              )}
            />
            <span className="upload-symbol">↺</span>
            <strong>
              {modelDragging
                ? "ここにドロップして読み込む"
                : modelLoaded
                  ? "別の保存モデルをドロップまたは選択"
                  : "malchan保存モデルをドロップまたは選択"}
            </strong>
            <span>
              {modelLoaded && fileName
                ? `読込中のモデル: ${fileName}`
                : "malchanから保存した信頼できるモデルファイルを読み込みます。"}
            </span>
          </label>
        </article>
      </div>

      {dataLoaded && (
        <article className="panel data-column-selection-panel">
          <div className="panel-title">
            <div>
              <span className="panel-kicker">ANALYSIS COLUMNS</span>
              <h3>解析に使う列を選択する</h3>
              <p>OFFにした列は元データには残りますが、以降のExplore / Prepare / Modelでは解析対象から外れます。</p>
            </div>
            <span className={`status-chip ${enabledColumns.length ? "success" : "warning"}`}>
              {enabledColumns.length} / {columns.length} ON
            </span>
          </div>

          <div className="data-column-selection-actions">
            <button
              type="button"
              className="secondary"
              disabled={columnSelectionLocked || enabledColumns.length === columns.length}
              onClick={() => applyEnabledColumns(columns)}
            >
              すべてON
            </button>
            <button
              type="button"
              className="secondary"
              disabled={columnSelectionLocked || enabledColumns.length === 0}
              onClick={() => applyEnabledColumns([])}
            >
              すべてOFF
            </button>
          </div>

          {columnSelectionLocked && (
            <p className="data-column-selection-lock-note">
              学習済みモデルとの入力列不整合を防ぐため、モデル登録後は変更できません。列構成を変える場合はデータを読み直して再学習してください。
            </p>
          )}

          <div className="data-column-toggle-grid" role="group" aria-label="解析対象列">
            {columns.map((column) => {
              const enabled = enabledSet.has(column);
              return (
                <label
                  key={column}
                  className={`data-column-toggle ${enabled ? "enabled" : "disabled"}`}
                >
                  <input
                    type="checkbox"
                    checked={enabled}
                    disabled={columnSelectionLocked}
                    onChange={() => toggleColumn(column)}
                  />
                  <span className="data-column-toggle-copy">
                    <strong>{column}</strong>
                    <small>{columnKind(column)} · {columnRole(column)}</small>
                  </span>
                  <em>{enabled ? "ON" : "OFF"}</em>
                </label>
              );
            })}
          </div>
          <p className="data-column-selection-help">
            一度OFFにした列をONへ戻すとPrepareの候補には復帰します。説明変数・目的変数への再選択はPrepareで行ってください。
          </p>
        </article>
      )}

      <article className="panel">
        {rows.length
          ? enabledColumns.length
            ? <DataTable rows={rows} columns={enabledColumns} />
            : <p className="empty-state">解析対象列がありません。上の列スイッチで1列以上をONにしてください。</p>
          : modelLoaded
            ? <p className="empty-state">保存モデルを読み込みました。目的変数・説明変数・モデル設定を復元しています。</p>
            : <p className="empty-state">データまたは保存モデルを読み込むと内容が表示されます。</p>}
      </article>
      {visibleStats.length > 0 && (
        <article className="panel">
          <h3>列統計</h3>
          <DataTable
            rows={visibleStats}
            columns={["column", "count", "missing", "unique", "min", "max", "mean"]}
            pageSize={50}
          />
        </article>
      )}
    </>
  );
}
