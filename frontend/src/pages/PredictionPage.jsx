import React, { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import DataTable from "../components/DataTable";
import { Field, SectionHeader } from "../components/Common";
import {
  coerceRows,
  formatNumber,
  parseTabularFile,
  uniqueValues,
} from "../data";
import { useWorkbench } from "../context/WorkbenchContext";

const FILE_PAGE_SIZE = 30;
const SHAP_FEATURE_LIMIT = 10;

function requiredRecords(rows, features) {
  return rows.map((row) => Object.fromEntries(
    features.map((feature) => [feature, row[feature]]),
  ));
}

function PredictionSummary({ prediction }) {
  if (!prediction) return null;
  return (
    <div className="prediction-result-grid">
      {Object.entries(prediction).map(([key, value]) => (
        <div className="prediction-chip" key={key}>
          <span>{key}</span>
          <strong>{formatNumber(value)}</strong>
        </div>
      ))}
    </div>
  );
}

function ShapTargetCard({ result, rowLabels }) {
  const [output, setOutput] = useState(result.output_names?.[0] || "");

  useEffect(() => {
    setOutput(result.output_names?.[0] || "");
  }, [result]);

  const matrices = result.shap_values?.[output] || [];
  const baseValues = result.base_values?.[output] || [];

  return (
    <section className="local-shap-target">
      <div className="local-shap-target-head">
        <div>
          <strong>{result.target}</strong>
          <span>{result.features.length} transformed features</span>
        </div>
        {result.output_names.length > 1 && (
          <label className="compact-select">
            出力
            <select value={output} onChange={(event) => setOutput(event.target.value)}>
              {result.output_names.map((name) => <option key={name}>{name}</option>)}
            </select>
          </label>
        )}
      </div>

      <div className="local-shap-row-list">
        {matrices.map((values, rowIndex) => {
          const contributions = result.features
            .map((feature, featureIndex) => ({
              feature,
              value: result.records?.[rowIndex]?.[feature],
              shap: values?.[featureIndex],
            }))
            .filter((item) => typeof item.shap === "number" && Number.isFinite(item.shap))
            .sort((left, right) => Math.abs(right.shap) - Math.abs(left.shap))
            .slice(0, SHAP_FEATURE_LIMIT);
          return (
            <article className="local-shap-row" key={`${result.target}-${rowIndex}`}>
              <div className="local-shap-row-head">
                <strong>行 {rowLabels?.[rowIndex] ?? rowIndex + 1}</strong>
                <span>base value: {formatNumber(baseValues[rowIndex])}</span>
              </div>
              <div className="table-wrap compact">
                <table>
                  <thead>
                    <tr><th>特徴量</th><th>入力値</th><th>SHAP</th></tr>
                  </thead>
                  <tbody>
                    {contributions.map((item) => (
                      <tr key={item.feature}>
                        <td>{item.feature}</td>
                        <td>{formatNumber(item.value)}</td>
                        <td className={item.shap >= 0 ? "shap-positive" : "shap-negative"}>
                          {formatNumber(item.shap)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </article>
          );
        })}
      </div>
    </section>
  );
}

function LocalShapResults({ response, rowLabels }) {
  if (!response) return null;
  return (
    <article className="panel local-shap-panel">
      <div className="panel-title">
        <div>
          <span className="panel-kicker">LOCAL EXPLANATION</span>
          <h3>選択行のSHAP</h3>
          <p>入力された行だけで再計算したローカルSHAPです。Explain画面のキャッシュは変更しません。</p>
        </div>
        <span className="status-chip success">{response.row_count} rows</span>
      </div>
      <div className="local-shap-target-list">
        {Object.values(response.targets || {}).map((result) => (
          <ShapTargetCard key={result.target} result={result} rowLabels={rowLabels} />
        ))}
      </div>
    </article>
  );
}

function SelectablePredictionRows({
  rows,
  features,
  selected,
  setSelected,
  page,
  setPage,
}) {
  const pageCount = Math.max(1, Math.ceil(rows.length / FILE_PAGE_SIZE));
  const normalizedPage = Math.min(page, pageCount - 1);
  const start = normalizedPage * FILE_PAGE_SIZE;
  const visibleRows = rows.slice(start, start + FILE_PAGE_SIZE);
  const visibleIndexes = visibleRows.map((_, offset) => start + offset);
  const allVisibleSelected = visibleIndexes.length > 0
    && visibleIndexes.every((index) => selected.has(index));

  function toggle(index) {
    setSelected((current) => {
      const next = new Set(current);
      if (next.has(index)) next.delete(index);
      else next.add(index);
      return next;
    });
  }

  function toggleVisible() {
    setSelected((current) => {
      const next = new Set(current);
      visibleIndexes.forEach((index) => {
        if (allVisibleSelected) next.delete(index);
        else next.add(index);
      });
      return next;
    });
  }

  return (
    <div className="prediction-file-table">
      <div className="prediction-file-table-head">
        <span>{rows.length}行を読み込み済み</span>
        <strong>SHAP対象: {selected.size}行</strong>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>
                <input
                  type="checkbox"
                  checked={allVisibleSelected}
                  onChange={toggleVisible}
                  aria-label="表示中の行をすべて選択"
                />
              </th>
              <th>#</th>
              {features.map((feature) => <th key={feature}>{feature}</th>)}
            </tr>
          </thead>
          <tbody>
            {visibleRows.map((row, offset) => {
              const index = start + offset;
              return (
                <tr key={index} className={selected.has(index) ? "selected-row" : ""}>
                  <td>
                    <input
                      type="checkbox"
                      checked={selected.has(index)}
                      onChange={() => toggle(index)}
                      aria-label={`行${index + 1}をSHAP対象にする`}
                    />
                  </td>
                  <td>{index + 1}</td>
                  {features.map((feature) => (
                    <td key={feature}>{formatNumber(row[feature])}</td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="pagination">
        <button
          className="secondary icon-button"
          onClick={() => setPage(Math.max(0, normalizedPage - 1))}
          disabled={normalizedPage === 0}
        >
          ‹
        </button>
        <span>{normalizedPage + 1} / {pageCount}</span>
        <button
          className="secondary icon-button"
          onClick={() => setPage(Math.min(pageCount - 1, normalizedPage + 1))}
          disabled={normalizedPage >= pageCount - 1}
        >
          ›
        </button>
      </div>
    </div>
  );
}

export default function PredictionPage() {
  const {
    rows,
    numeric,
    categorical,
    features,
    predictValues,
    setPredictValues,
    modelInfo,
    busy,
  } = useWorkbench();
  const [mode, setMode] = useState("custom");
  const [customPrediction, setCustomPrediction] = useState(null);
  const [customShap, setCustomShap] = useState(null);
  const [fileName, setFileName] = useState("");
  const [fileRows, setFileRows] = useState([]);
  const [selectedRows, setSelectedRows] = useState(new Set());
  const [filePage, setFilePage] = useState(0);
  const [filePredictions, setFilePredictions] = useState([]);
  const [fileShap, setFileShap] = useState(null);
  const [fileShapLabels, setFileShapLabels] = useState([]);
  const [running, setRunning] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    setCustomPrediction(null);
    setCustomShap(null);
    setFilePredictions([]);
    setFileShap(null);
  }, [modelInfo?.model_id]);

  const predictionRows = useMemo(
    () => filePredictions.map((prediction, index) => ({
      row: index + 1,
      ...prediction,
    })),
    [filePredictions],
  );
  const predictionColumns = useMemo(() => {
    const columns = new Set(["row"]);
    filePredictions.forEach((prediction) => {
      Object.keys(prediction || {}).forEach((column) => columns.add(column));
    });
    return [...columns];
  }, [filePredictions]);

  function customInput() {
    return Object.fromEntries(features.map((feature) => [
      feature,
      numeric.includes(feature)
        ? Number(predictValues[feature])
        : predictValues[feature],
    ]));
  }

  async function runCustomPrediction() {
    if (!modelInfo?.model_id) return;
    setRunning("custom");
    setError("");
    setCustomPrediction(null);
    setCustomShap(null);
    try {
      const data = [customInput()];
      const [predictionResponse, shapResponse] = await Promise.all([
        api.predict(modelInfo.model_id, { data }),
        api.localShap(modelInfo.model_id, { data }),
      ]);
      setCustomPrediction(predictionResponse.predictions?.[0] || null);
      setCustomShap(shapResponse);
    } catch (reason) {
      setError(reason.message || String(reason));
    } finally {
      setRunning("");
    }
  }

  async function loadPredictionFile(file) {
    if (!file) return;
    setRunning("load");
    setError("");
    try {
      const parsed = await parseTabularFile(file);
      const data = coerceRows(parsed.rows);
      const missing = features.filter((feature) => !data.columns.includes(feature));
      if (missing.length) {
        throw new Error(`予測ファイルに必要な列がありません: ${missing.join(", ")}`);
      }
      setFileName(file.name);
      setFileRows(data.rows);
      setSelectedRows(new Set(data.rows.length ? [0] : []));
      setFilePage(0);
      setFilePredictions([]);
      setFileShap(null);
      setFileShapLabels([]);
    } catch (reason) {
      setError(reason.message || String(reason));
    } finally {
      setRunning("");
    }
  }

  async function runFilePrediction() {
    if (!modelInfo?.model_id || !fileRows.length) return;
    setRunning("file-predict");
    setError("");
    try {
      const response = await api.predict(modelInfo.model_id, {
        data: requiredRecords(fileRows, features),
      });
      setFilePredictions(response.predictions || []);
    } catch (reason) {
      setError(reason.message || String(reason));
    } finally {
      setRunning("");
    }
  }

  async function runSelectedShap() {
    if (!modelInfo?.model_id || !selectedRows.size) return;
    const indexes = [...selectedRows].sort((left, right) => left - right);
    const selected = indexes.map((index) => fileRows[index]);
    setRunning("file-shap");
    setError("");
    setFileShap(null);
    try {
      const response = await api.localShap(modelInfo.model_id, {
        data: requiredRecords(selected, features),
      });
      setFileShap(response);
      setFileShapLabels(indexes.map((index) => index + 1));
    } catch (reason) {
      setError(reason.message || String(reason));
    } finally {
      setRunning("");
    }
  }

  const disabled = !modelInfo || Boolean(busy) || Boolean(running);

  return (
    <>
      <SectionHeader
        step="6 · PREDICT"
        title="モデルで予測し、入力ごとのSHAPを確認する"
        text="カスタム入力は実行ごとにSHAPを再計算し、ファイル入力は選択した行だけSHAPを計算します。"
      />

      <article className="panel prediction-mode-panel">
        <div className="prediction-mode-switch" role="tablist" aria-label="予測入力方式">
          <button
            type="button"
            className={mode === "custom" ? "active" : ""}
            onClick={() => setMode("custom")}
          >
            <strong>カスタム入力</strong>
            <span>1条件を入力して予測＋SHAP</span>
          </button>
          <button
            type="button"
            className={mode === "file" ? "active" : ""}
            onClick={() => setMode("file")}
          >
            <strong>ファイル入力</strong>
            <span>CSV / Excelを一括予測</span>
          </button>
        </div>
        {!modelInfo && (
          <p className="settings-note">先にModel画面で予測に使用するモデルを学習してください。</p>
        )}
      </article>

      {mode === "custom" && (
        <article className="panel custom-prediction-panel">
          <div className="panel-title">
            <div>
              <span className="panel-kicker">CUSTOM PREDICTION</span>
              <h3>任意条件で予測</h3>
              <p>予測ボタンを押すたびに、この入力1行についてSHAPも新しく計算します。</p>
            </div>
          </div>
          <div className="form-grid">
            {features.map((column) => (
              <Field key={column} label={column}>
                {categorical.includes(column) ? (
                  <select
                    value={predictValues[column] ?? ""}
                    onChange={(event) => setPredictValues({
                      ...predictValues,
                      [column]: event.target.value,
                    })}
                  >
                    {uniqueValues(rows, column).map((value) => (
                      <option key={String(value)} value={value}>{String(value)}</option>
                    ))}
                  </select>
                ) : (
                  <input
                    type="number"
                    step="any"
                    value={predictValues[column] ?? ""}
                    onChange={(event) => setPredictValues({
                      ...predictValues,
                      [column]: event.target.value,
                    })}
                  />
                )}
              </Field>
            ))}
          </div>
          <button disabled={disabled} onClick={runCustomPrediction}>
            {running === "custom" ? "予測・SHAP計算中..." : "予測とSHAPを計算 →"}
          </button>
          <PredictionSummary prediction={customPrediction} />
        </article>
      )}

      {mode === "file" && (
        <>
          <article className="panel file-prediction-panel">
            <div className="panel-title">
              <div>
                <span className="panel-kicker">BATCH PREDICTION</span>
                <h3>CSV / Excelから予測</h3>
                <p>全行予測とSHAP計算を分離し、SHAPはチェックした行だけ実行します。</p>
              </div>
              {fileName && <span className="status-chip">{fileName}</span>}
            </div>
            <label className="prediction-file-drop">
              <input
                type="file"
                accept=".csv,.xlsx,.xls,text/csv,application/vnd.ms-excel,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                onChange={(event) => loadPredictionFile(event.target.files?.[0])}
              />
              <strong>{running === "load" ? "読み込み中..." : "予測ファイルを選択"}</strong>
              <span>必要列: {features.join(", ")}</span>
            </label>

            {fileRows.length > 0 && (
              <>
                <SelectablePredictionRows
                  rows={fileRows}
                  features={features}
                  selected={selectedRows}
                  setSelected={setSelectedRows}
                  page={filePage}
                  setPage={setFilePage}
                />
                <div className="prediction-file-actions">
                  <button disabled={disabled} onClick={runFilePrediction}>
                    {running === "file-predict" ? "全行を予測中..." : `全${fileRows.length}行を予測`}
                  </button>
                  <button
                    className="secondary"
                    disabled={disabled || !selectedRows.size}
                    onClick={runSelectedShap}
                  >
                    {running === "file-shap"
                      ? "選択行のSHAP計算中..."
                      : `選択${selectedRows.size}行のSHAPを計算`}
                  </button>
                </div>
              </>
            )}
          </article>

          {filePredictions.length > 0 && (
            <article className="panel">
              <div className="panel-title">
                <div>
                  <span className="panel-kicker">PREDICTION RESULT</span>
                  <h3>ファイル予測結果</h3>
                </div>
                <span className="status-chip success">{filePredictions.length} rows</span>
              </div>
              <DataTable rows={predictionRows} columns={predictionColumns} pageSize={30} />
            </article>
          )}
        </>
      )}

      {error && <p className="xai-error prediction-error">{error}</p>}
      {mode === "custom" && <LocalShapResults response={customShap} rowLabels={[1]} />}
      {mode === "file" && <LocalShapResults response={fileShap} rowLabels={fileShapLabels} />}
    </>
  );
}
