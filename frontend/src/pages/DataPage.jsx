import React, { useState } from "react";
import DataTable from "../components/DataTable";
import { SectionHeader } from "../components/Common";
import { useWorkbench } from "../context/WorkbenchContext";

export default function DataPage() {
  const {
    rows,
    columns,
    fileName,
    stats,
    modelInfo,
    busy,
    loadFile,
    loadModelBundle,
  } = useWorkbench();
  const [dataDragging, setDataDragging] = useState(false);
  const [modelDragging, setModelDragging] = useState(false);
  const dataLoaded = rows.length > 0;
  const modelLoaded = Boolean(modelInfo?.model_id) && !dataLoaded;

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

      <article className="panel">
        {rows.length
          ? <DataTable rows={rows} columns={columns} />
          : modelLoaded
            ? <p className="empty-state">保存モデルを読み込みました。目的変数・説明変数・モデル設定を復元しています。</p>
            : <p className="empty-state">データまたは保存モデルを読み込むと内容が表示されます。</p>}
      </article>
      {stats.length > 0 && (
        <article className="panel">
          <h3>列統計</h3>
          <DataTable
            rows={stats}
            columns={["column", "count", "missing", "unique", "min", "max", "mean"]}
            pageSize={50}
          />
        </article>
      )}
    </>
  );
}
