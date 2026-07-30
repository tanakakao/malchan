import React, { useEffect, useMemo, useState } from "react";
import { SectionHeader } from "../components/Common";
import { useWorkbench } from "../context/WorkbenchContext";
import { useWorkbenchMode } from "../workbenchMode";

const COLUMN_KIND_LABELS = {
  numeric: "numeric",
  categorical: "categorical",
};

const SIMPLE_REGRESSION_MODELS = [
  "線形回帰",
  "ElasticNet",
  "ランダムフォレスト回帰",
  "LightGBM",
];

const SIMPLE_CLASSIFICATION_MODELS = [
  "ロジスティック回帰",
  "ランダムフォレスト",
  "LightGBM",
];

const SIMPLE_PREPROCESSING = {
  impute: true,
  numImputeType: "mean",
  numScaleType: "StandardScaler",
  catImpute: true,
  poly: false,
  polyDegree: 2,
  polyInteractionOnly: true,
  decomposition: false,
  decompositionMethod: "PCA",
  decNComponents: 2,
  samplingMethod: "",
};

function simpleModelsFor(task) {
  return task === "classification"
    ? SIMPLE_CLASSIFICATION_MODELS
    : SIMPLE_REGRESSION_MODELS;
}

export default function PreparePage() {
  const mode = useWorkbenchMode();
  const {
    columns,
    numeric,
    categorical,
    targets,
    tasks,
    numFeatures,
    setNumFeatures,
    catFeatures,
    setCatFeatures,
    ready,
    busy,
    changeTargets,
    changeTask,
    compareModels,
    setStep,
  } = useWorkbench();
  const [featureTypeOverrides, setFeatureTypeOverrides] = useState({});
  const [simpleRunPending, setSimpleRunPending] = useState(false);

  const targetSet = useMemo(() => new Set(targets), [targets]);
  const numericSet = useMemo(() => new Set(numeric), [numeric]);
  const categoricalSet = useMemo(() => new Set(categorical), [categorical]);
  const numFeatureSet = useMemo(() => new Set(numFeatures), [numFeatures]);
  const catFeatureSet = useMemo(() => new Set(catFeatures), [catFeatures]);
  const featureCandidates = useMemo(
    () => columns.filter((column) => !targetSet.has(column)),
    [columns, targetSet],
  );
  const simpleTargetsSupported = targets.length > 0 && targets.every(
    (target) => tasks[target] === "regression" || tasks[target] === "classification",
  );

  useEffect(() => {
    setFeatureTypeOverrides((current) => Object.fromEntries(
      Object.entries(current).filter(([column]) => columns.includes(column)),
    ));
  }, [columns]);

  function orderedColumns(values) {
    const selected = new Set(values);
    return columns.filter((column) => selected.has(column) && !targetSet.has(column));
  }

  function columnKind(column) {
    return categoricalSet.has(column) ? "categorical" : "numeric";
  }

  function selectedFeatureType(column) {
    if (catFeatureSet.has(column)) return "categorical";
    if (numFeatureSet.has(column)) return "numeric";
    if (featureTypeOverrides[column]) return featureTypeOverrides[column];
    return columnKind(column);
  }

  function toggleTarget(column) {
    const nextTargets = targetSet.has(column)
      ? targets.filter((target) => target !== column)
      : [...targets, column];
    changeTargets(nextTargets);
  }

  function clearTargets() {
    changeTargets([]);
  }

  function toggleFeature(column) {
    if (numFeatureSet.has(column) || catFeatureSet.has(column)) {
      setNumFeatures(numFeatures.filter((feature) => feature !== column));
      setCatFeatures(catFeatures.filter((feature) => feature !== column));
      return;
    }

    if (selectedFeatureType(column) === "categorical") {
      setCatFeatures(orderedColumns([...catFeatures, column]));
    } else {
      setNumFeatures(orderedColumns([...numFeatures, column]));
    }
  }

  function setFeatureCategorical(column, useCategorical) {
    const nextType = useCategorical ? "categorical" : "numeric";
    setFeatureTypeOverrides((current) => ({ ...current, [column]: nextType }));

    if (useCategorical) {
      setNumFeatures(numFeatures.filter((feature) => feature !== column));
      setCatFeatures(orderedColumns([...catFeatures, column]));
    } else {
      setCatFeatures(catFeatures.filter((feature) => feature !== column));
      setNumFeatures(orderedColumns([...numFeatures, column]));
    }
  }

  function replaceFeatureSelection(modeName) {
    if (modeName === "clear") {
      setNumFeatures([]);
      setCatFeatures([]);
      return;
    }

    if (modeName === "numeric") {
      setNumFeatures(featureCandidates.filter((column) => numericSet.has(column)));
      setCatFeatures([]);
      return;
    }

    const nextNumeric = [];
    const nextCategorical = [];
    featureCandidates.forEach((column) => {
      if (selectedFeatureType(column) === "categorical") {
        nextCategorical.push(column);
      } else {
        nextNumeric.push(column);
      }
    });
    setNumFeatures(nextNumeric);
    setCatFeatures(nextCategorical);
  }

  async function executeSimpleMode() {
    if (!ready || !simpleTargetsSupported || busy || simpleRunPending) return;
    setSimpleRunPending(true);
    try {
      await compareModels({
        preprocessing: SIMPLE_PREPROCESSING,
        tuning: false,
        cvMethod: "kfold",
        cvSplits: 5,
        activateBest: true,
        candidatesByTarget: Object.fromEntries(
          targets.map((target) => [target, simpleModelsFor(tasks[target])]),
        ),
      });
      setStep("model");
    } finally {
      setSimpleRunPending(false);
    }
  }

  const selectedFeatureCount = numFeatures.length + catFeatures.length;
  const simpleMode = mode === "simple";
  const selectedTaskSummary = targets.length
    ? targets
        .map((target) => `${target}: ${tasks[target] === "classification" ? "分類" : "回帰"}`)
        .join(" / ")
    : "未選択";

  return (
    <>
      <SectionHeader
        step={simpleMode ? "2 · PREPARE" : "3 · PREPARE"}
        title={simpleMode ? "目的変数と説明変数を選択してモデルを自動決定する" : "目的変数と説明変数を選択する"}
        text={simpleMode
          ? "変数選択は詳細モードと同じです。回帰・分類のタスクに応じた候補モデルを比較し、最良モデルを目的変数ごとに自動採用します。"
          : "bochanと同じカード式の操作で列を選択し、目的変数のタスクと説明変数の型を同じ画面で設定します。"}
        action={simpleMode ? (
          <button
            type="button"
            disabled={!ready || !simpleTargetsSupported || Boolean(busy) || simpleRunPending}
            onClick={executeSimpleMode}
          >
            {simpleRunPending ? "モデルを比較中..." : "モデルを自動選択 →"}
          </button>
        ) : null}
      />

      {simpleMode && (
        <article className="panel compact-panel simple-mode-summary">
          <div className="panel-title">
            <div>
              <span className="panel-kicker">SIMPLE MODE</span>
              <h3>変数選択だけでモデルを自動比較</h3>
              <p>目的変数と説明変数の選択以外は、固定した既定値で自動実行します。</p>
            </div>
            <span className={`status-chip ${simpleTargetsSupported ? "success" : "warning"}`}>
              {simpleTargetsSupported ? "Ready" : "Target required"}
            </span>
          </div>
          <div className="simple-default-grid">
            <span><strong>Target / Task</strong> {selectedTaskSummary}</span>
            <span><strong>Regression</strong> 線形回帰 / ElasticNet / Random Forest / LightGBM</span>
            <span><strong>Classification</strong> ロジスティック回帰 / Random Forest / LightGBM</span>
            <span><strong>Validation</strong> 5-fold CV</span>
            <span><strong>Metric</strong> 回帰=RMSE / 分類=F1</span>
            <span><strong>Activation</strong> 各目的変数の1位を自動採用</span>
          </div>
        </article>
      )}

      <div className="prepare-selection-grid">
        <article className="panel selection-panel">
          <div className="panel-title">
            <div>
              <span className="panel-kicker">TARGET COLUMNS</span>
              <h3>目的変数 Y</h3>
              <p>予測したい列を選択します。選択後、カード内で回帰／分類を設定できます。</p>
            </div>
            <span className={`status-chip ${targets.length ? "success" : "warning"}`}>
              {targets.length ? `${targets.length} selected` : "Required"}
            </span>
          </div>

          <div className="button-row selection-actions">
            <button
              type="button"
              className="secondary"
              disabled={!targets.length}
              onClick={clearTargets}
            >
              解除
            </button>
          </div>

          <div className="variable-selection-list" role="group" aria-label="目的変数">
            {columns.map((column) => {
              const selected = targetSet.has(column);
              const kind = columnKind(column);
              return (
                <div
                  key={column}
                  className={`variable-choice target-variable-choice ${selected ? "selected" : ""}`}
                >
                  <button
                    type="button"
                    className="variable-choice-main"
                    aria-pressed={selected}
                    onClick={() => toggleTarget(column)}
                  >
                    <span>{column}</span>
                    <small>{COLUMN_KIND_LABELS[kind]}</small>
                  </button>
                  {selected && (
                    <label className="target-task-select">
                      <span>タスク</span>
                      <select
                        value={tasks[column] || (kind === "numeric" ? "regression" : "classification")}
                        onChange={(event) => changeTask(column, event.target.value)}
                      >
                        <option value="regression">回帰</option>
                        <option value="classification">分類</option>
                      </select>
                    </label>
                  )}
                </div>
              );
            })}
          </div>
        </article>

        <article className="panel selection-panel">
          <div className="panel-title">
            <div>
              <span className="panel-kicker">FEATURE COLUMNS</span>
              <h3>説明変数 X</h3>
              <p>青は数値、紫はカテゴリ扱いです。型を変更すると、その列も自動的に選択されます。</p>
            </div>
            <span className={`status-chip ${selectedFeatureCount ? "success" : "warning"}`}>
              {selectedFeatureCount ? `${selectedFeatureCount} selected` : "Required"}
            </span>
          </div>

          <div className="button-row selection-actions">
            <button type="button" className="secondary" onClick={() => replaceFeatureSelection("all")}>
              全選択
            </button>
            <button type="button" className="secondary" onClick={() => replaceFeatureSelection("numeric")}>
              数値列のみ
            </button>
            <button type="button" className="secondary" onClick={() => replaceFeatureSelection("clear")}>
              解除
            </button>
          </div>

          <div className="variable-selection-list" role="group" aria-label="説明変数">
            {featureCandidates.map((column) => {
              const selected = numFeatureSet.has(column) || catFeatureSet.has(column);
              const type = selectedFeatureType(column);
              const fixedCategorical = categoricalSet.has(column);
              return (
                <div
                  key={column}
                  className={`variable-choice feature-variable-choice ${selected ? "selected" : ""} ${selected && type === "categorical" ? "selected-categorical" : ""}`}
                >
                  <button
                    type="button"
                    className="variable-choice-main"
                    aria-pressed={selected}
                    onClick={() => toggleFeature(column)}
                  >
                    <span>{column}</span>
                    <small>{type}</small>
                  </button>
                  <label
                    className="feature-type-toggle"
                    title={fixedCategorical ? "入力データ上カテゴリ列のため固定です。" : "カテゴリ変数として扱う"}
                  >
                    <input
                      type="checkbox"
                      checked={type === "categorical"}
                      disabled={fixedCategorical}
                      onChange={(event) => setFeatureCategorical(column, event.target.checked)}
                    />
                    <span>カテゴリ</span>
                  </label>
                </div>
              );
            })}
          </div>
        </article>
      </div>

      {!columns.length && (
        <p className="empty-state prepare-empty-state">
          Data画面でCSVまたはXLSXを読み込むと、変数選択カードが表示されます。
        </p>
      )}
    </>
  );
}
