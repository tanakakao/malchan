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
  const simpleTarget = columns.at(-1) || "";
  const simpleTargetSupported = Boolean(simpleTarget && numericSet.has(simpleTarget));

  useEffect(() => {
    setFeatureTypeOverrides((current) => Object.fromEntries(
      Object.entries(current).filter(([column]) => columns.includes(column)),
    ));
  }, [columns]);

  useEffect(() => {
    if (mode !== "simple" || !simpleTargetSupported) return;
    if (
      targets.length === 1
      && targets[0] === simpleTarget
      && tasks[simpleTarget] === "regression"
    ) {
      return;
    }
    changeTargets([simpleTarget]);
    changeTask(simpleTarget, "regression");
  }, [mode, simpleTarget, simpleTargetSupported, targets, tasks]);

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
    if (!ready || !simpleTargetSupported || busy || simpleRunPending) return;
    setSimpleRunPending(true);
    try {
      await compareModels({
        preprocessing: SIMPLE_PREPROCESSING,
        tuning: false,
        cvMethod: "kfold",
        cvSplits: 5,
        activateBest: true,
        candidatesByTarget: {
          [simpleTarget]: SIMPLE_REGRESSION_MODELS,
        },
      });
      setStep("model");
    } finally {
      setSimpleRunPending(false);
    }
  }

  const selectedFeatureCount = numFeatures.length + catFeatures.length;
  const simpleMode = mode === "simple";

  return (
    <>
      <SectionHeader
        step={simpleMode ? "2 · PREPARE" : "3 · PREPARE"}
        title={simpleMode ? "説明変数を選択してモデルを自動決定する" : "目的変数と説明変数を選択する"}
        text={simpleMode
          ? "最終列を回帰の目的変数として使用します。説明変数を選択すると、4つの候補モデルを同じ条件で比較して最良モデルを自動採用します。"
          : "bochanと同じカード式の操作で列を選択し、目的変数のタスクと説明変数の型を同じ画面で設定します。"}
        action={simpleMode ? (
          <button
            type="button"
            disabled={!ready || !simpleTargetSupported || Boolean(busy) || simpleRunPending}
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
              <h3>説明変数の選択だけで学習</h3>
              <p>前処理、モデル候補、交差検証条件、最良モデルの有効化は固定値で自動実行します。</p>
            </div>
            <span className={`status-chip ${simpleTargetSupported ? "success" : "warning"}`}>
              {simpleTargetSupported ? "Regression" : "Unsupported"}
            </span>
          </div>
          <div className="simple-default-grid">
            <span><strong>Target</strong> {simpleTarget || "未読込"}</span>
            <span><strong>Task</strong> 回帰</span>
            <span><strong>Models</strong> 線形回帰 / ElasticNet / Random Forest / LightGBM</span>
            <span><strong>Validation</strong> 5-fold CV</span>
            <span><strong>Metric</strong> Validation RMSE</span>
            <span><strong>Activation</strong> 1位を自動採用</span>
          </div>
          {!simpleTargetSupported && columns.length > 0 && (
            <p className="simple-mode-warning">
              簡易モードは、最終列が数値の単一目的回帰に対応しています。分類または目的変数を変更する場合は詳細モードを使用してください。
            </p>
          )}
        </article>
      )}

      <div className={simpleMode ? "prepare-selection-grid simple-feature-selection-grid" : "prepare-selection-grid"}>
        {!simpleMode && (
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
        )}

        <article className="panel selection-panel">
          <div className="panel-title">
            <div>
              <span className="panel-kicker">FEATURE COLUMNS</span>
              <h3>説明変数 X</h3>
              <p>{simpleMode
                ? "モデルに使用する列を選択します。数値／カテゴリの型は読み込んだデータから自動判定します。"
                : "青は数値、紫はカテゴリ扱いです。型を変更すると、その列も自動的に選択されます。"}</p>
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
                  {!simpleMode && (
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
                  )}
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
