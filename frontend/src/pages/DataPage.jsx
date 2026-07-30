import React from "react";
import DataTable from "../components/DataTable";
import { SectionHeader } from "../components/Common";
import { useWorkbench } from "../context/WorkbenchContext";

export default function DataPage() {
  const { rows, columns, fileName, stats, loadFile } = useWorkbench();
  return (
    <>
      <SectionHeader
        step="1 · DATA"
        title="データを読み込む"
        text="CSV / XLSXを読み込み、データの内容を確認します。"
      />
      <article className="panel">
        <div className="panel-title">
          <div><span className="panel-kicker">DATA SOURCE</span><h3>ファイルを読み込む</h3></div>
        </div>
        <label className="dropzone">
          <input type="file" accept=".csv,.xlsx" onChange={(event) => loadFile(event.target.files?.[0])} />
          <span className="upload-symbol">⇧</span>
          <strong>CSV / XLSXを選択</strong>
          <span>{fileName || "ブラウザ内で読み込み"}</span>
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
