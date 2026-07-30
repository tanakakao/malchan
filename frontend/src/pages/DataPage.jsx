import React, { useState } from "react";
import DataTable from "../components/DataTable";
import { SectionHeader } from "../components/Common";
import { useWorkbench } from "../context/WorkbenchContext";

export default function DataPage() {
  const { rows, columns, fileName, stats, loadFile } = useWorkbench();
  const [dragging, setDragging] = useState(false);
  const dataLoaded = rows.length > 0;

  function loadDataFile(file) {
    if (file) loadFile(file);
  }

  return (
    <>
      <SectionHeader
        step="1 · DATA"
        title="データを読み込む"
        text="CSV / XLSXを読み込み、データの内容を確認します。"
      />
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
          className={`dropzone ${dragging ? "dragging" : ""}`}
          aria-label="CSVまたはXLSXファイルをドロップまたは選択"
          onDragEnter={(event) => {
            event.preventDefault();
            setDragging(true);
          }}
          onDragOver={(event) => {
            event.preventDefault();
            event.dataTransfer.dropEffect = "copy";
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(event) => {
            event.preventDefault();
            setDragging(false);
            loadDataFile(event.dataTransfer.files[0]);
          }}
        >
          <input
            type="file"
            accept=".csv,.xlsx"
            onChange={(event) => loadDataFile(event.target.files?.[0])}
          />
          <span className="upload-symbol">⇧</span>
          <strong>
            {dragging
              ? "ここにドロップして読み込む"
              : dataLoaded
                ? "別のファイルをドロップまたは選択"
                : "CSVまたはExcelをドロップまたは選択"}
          </strong>
          <span>
            {fileName
              ? `読込中のファイル: ${fileName}`
              : "ファイルはブラウザで解析され、現在のワークスペース内に保持されます。"}
          </span>
        </label>
      </article>
      <article className="panel">
        {rows.length
          ? <DataTable rows={rows} columns={columns} />
          : <p className="empty-state">ファイルを読み込むとプレビューが表示されます。</p>}
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
