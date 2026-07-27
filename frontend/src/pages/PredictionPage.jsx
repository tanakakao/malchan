import React, { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import { Field, SectionHeader } from "../components/Common";
import {
  coerceRows,
  formatNumber,
  parseTabularFile,
  uniqueValues,
} from "../data";
import { useWorkbench } from "../context/WorkbenchContext";
import "../prediction-auto.css";

const FILE_PAGE_SIZE = 30;
const SHAP_FEATURE_LIMIT = 10;

function requiredRecords(rows, features) {
  return rows.map((row) => Object.fromEntries(
    features.map((feature) => [feature, row[feature]]),
  ));
}

function PredictionSummary({ prediction, alwaysVisible = false }) {
  if (!prediction) {
    return alwaysVisible
      ? <p className="empty-state prediction-placeholder">予測実行後、ここに予測値を表示します。</p>
      : null;
  }
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

function LocalShapResults({
  response,
  rowLabels,
  alwaysVisible = false,
  title = "選択行のSHAP",
  emptyText = "SHAP計算後、ここに結果を表示します。",
}) {
  if (!response && !alwaysVisible) return null;
  return (
    <article className="panel local-shap-panel">
      <div className="panel-title">
        <div>
          <span className="panel-kicker">LOCAL EXPLANATION</span>
          <h3>{title}</h3>
          <p>入力された行だけで再計算したローカルSHAPです。Explain画面のキャッシュは変更しません。</p>
        </div>
        {response && <span className="status-chip success">{response.row_count} rows</span>}
      </div>
      {response ? (
        <div className="local-shap-target-list">
          {Object.values(response.targets || {}).map((result) => (
            <ShapTargetCard key={result.target} result={result} rowLabels={rowLabels} />
          ))}
        </div>
      ) : (
        <p className="empty-state prediction-placeholder">{emptyText}</p>
      )}
    </article>
  );
}

function SelectablePredictionRows({
  rows,
  columns,
  predictionColumns,
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
  const predictionColumnSet = new Set(predictionColumns);

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
        <span>{rows.length}行を予測済み</span>
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
              {columns.map((column) => (
                <th
                  key={column}
                  className={predictionColumnSet.has(column) ? "prediction-column" : ""}
                >
                  {column}
                </th>
              ))}
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
                  {columns.map((column) => (
                    <td
                      key={column}
                      className={predictionColumnSet.has(column) ? "prediction-value-cell" : ""}
                    >
                      {formatNumber(row[column])}
                    </td>
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

function predictionColumnEntries(predictions, inputColumns) {
  const predictionKeys = [...new Set(
    predictions.flatMap((prediction) => Object.keys(prediction || {})),
  )];
  const used = new Set(inputColumns);
  return predictionKeys.map((source) => {
    const base = `予測_${source}`;
    let label = base;
    let suffix = 2;
    while (used.has(label)) {
      label = `${base}_${suffix}`;
      suffix += 1;
    }
    used.add(label);
    return { source, label };
  });
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
  const [fileColumns, setFileColumns] = useState([]);
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
    setFileRows([]);
    setFileColumns([]);
    setFilePredictions([]);
    setFileShap(null);
    setFileName("");
    setSelectedRows(new Set());
  }, [modelInfo?.model_id]);

  const filePredictionEntries = useMemo(
    () => predictionColumnEntries(filePredictions, fileColumns),
    [filePredictions, fileColumns],
  );
  const filePredictionColumns = useMemo(
    () => filePredictionEntries.map((entry) => entry.label),
    [filePredictionEntries],
  );
  const fileDisplayColumns = useMemo(
    () => [...fileColumns, ...filePredictionColumns],
    [fileColumns, filePredictionColumns],
  );
  const fileDisplayRows = useMemo(
    () => fileRows.map((row, index) => {
      const prediction = filePredictions[index] || {};
      const predicted = Object.fromEntries(
        filePredictionEntries.map(({ source, label }) => [label, prediction[source] ?? null]),
      );
      return { ...row, ...predicted };
    }),
    [fileRows, filePredictions, filePredictionEntries],
  );

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
    if (!modelInfo?.model_id) {
      setError("先にModel画面で予測に使用するモデルを学習してください。");
      return;
    }
    setRunning("load");
    setError("");
    setFileRows([]);
    setFileColumns([]);
    setFilePredictions([]);
    setFileShap(null);
    try {
      const parsed = await parseTabularFile(file);
      const data = coerceRows(parsed.rows);
      const missing = features.filter((feature) => !data.columns.includes(feature));
      if (missing.length) {
        throw new Error(`予測ファイルに必要な列がありません: ${missing.join(", ")}`);
      }
      const predictionResponse = await api.predict(modelInfo.model_id, {
        data: requiredRecords(data.rows, features),
      });
      setFileName(file.name);
      setFileRows(data.rows);
      setFileColumns(data.columns);
      setFilePredictions(predictionResponse.predictions || []);
      setSelectedRows(new Set());
      setFilePage(0);
      setFileShap(null);
      setFileShapLabels([]);
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
        text="カスタム入力では予測値とSHAPを同時表示し、ファイル入力では読込直後に全行を予測します。"
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
            <span>CSV / Excelを自動予測</span>
          </button>
        </div>
        {!modelInfo && (
          <p className="settings-note">先にModel画面で予測に使用するモデルを学習してください。</p>
        )}
      </article>

      {mode === "custom" && (
        <>
          <article className="panel custom-prediction-panel">
            <div className="panel-title">
              <div>
                <span className="panel-kicker">CUSTOM PREDICTION</span>
                <h3>任意条件で予測</h3>
                <p>実行するたびに現在の入力1行について、予測とSHAPを更新します。</p>
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
              {running === "custom" ? "予測・SHAP計算中..." : "予測とSHAPを更新 →"}
            </button>
          </article>

          <article className="panel custom-prediction-result-panel">
            <div className="panel-title">
              <div>
                <span className="panel-kicker">PREDICTION RESULT</span>
                <h3>予測値</h3>
                <p>最後に実行したカスタム入力の予測結果を常に表示します。</p>
              </div>
            </div>
            <PredictionSummary prediction={customPrediction} alwaysVisible />
          </article>

          <LocalShapResults
            response={customShap}
            rowLabels={[1]}
            alwaysVisible
            title="カスタム入力のSHAP"
            emptyText="予測を実行すると、この入力1行のSHAPを表示します。"
          />
        </>
      )}

      {mode === "file" && (
        <>
          <article className="panel file-prediction-panel">
            <div className="panel-title">
              <div>
                <span className="panel-kicker">BATCH PREDICTION</span>
                <h3>CSV / Excelから予測</h3>
                <p>読込直後に全行を予測し、予測列を入力データフレームの右端へ追加します。</p>
              </div>
              {fileName && <span className="status-chip success">{fileName}</span>}
            </div>
            <label className={`prediction-file-drop ${!modelInfo ? "disabled" : ""}`}>
              <input
                type="file"
                accept=".csv,.xlsx,text/csv,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                disabled={!modelInfo || Boolean(running)}
                onChange={(event) => loadPredictionFile(event.target.files?.[0])}
              />
              <strong>{running === "load" ? "読み込み・予測中..." : "予測ファイルを選択"}</strong>
              <span>必要列: {features.join(", ")}</span>
            </label>

            {fileDisplayRows.length > 0 && (
              <>
                <SelectablePredictionRows
                  rows={fileDisplayRows}
                  columns={fileDisplayColumns}
                  predictionColumns={filePredictionColumns}
                  selected={selectedRows}
                  setSelected={setSelectedRows}
                  page={filePage}
                  setPage={setFilePage}
                />
                <div className="prediction-file-actions">
                  <span className="settings-note">
                    SHAPはチェックした行だけ計算します。予測値はすでに右端へ追加されています。
                  </span>
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

          <LocalShapResults
            response={fileShap}
            rowLabels={fileShapLabels}
            title="選択行のSHAP"
          />
        </>
      )}

      {error && <p className="xai-error prediction-error">{error}</p>}
    </>
  );
}
