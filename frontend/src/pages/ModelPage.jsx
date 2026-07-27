import React, { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import ComparisonTable from "../components/ComparisonTable";
import { CheckboxList, Field, SectionHeader } from "../components/Common";
import { modelsFor, useWorkbench } from "../context/WorkbenchContext";

const DEFAULT_PREPROCESSING = {
  impute: false,
  numImputeType: "mean",
  numScaleType: "",
  catImpute: false,
  poly: false,
  polyDegree: 2,
  polyInteractionOnly: true,
  decomposition: false,
  decompositionMethod: "PCA",
  decNComponents: 2,
  samplingMethod: "",
};

const PARAMETER_LABELS = {
  alpha: "正則化強度 α",
  l1_ratio: "L1比率",
  n_components: "成分数",
  epsilon: "ε",
  power: "分布の指数",
  link: "リンク関数",
  max_depth: "最大深さ",
  min_samples_split: "分割に必要な最小サンプル数",
  min_samples_leaf: "葉に必要な最小サンプル数",
  ccp_alpha: "剪定強度",
  n_estimators: "推定器数",
  max_features: "使用特徴量の割合",
  learning_rate: "学習率",
  gamma: "Gamma",
  reg_alpha: "L1正則化",
  reg_lambda: "L2正則化",
  min_child_weight: "子ノード最小重み",
  colsample_bytree: "木ごとの特徴量割合",
  colsample_bylevel: "レベルごとの特徴量割合",
  colsample_bynode: "ノードごとの特徴量割合",
  num_leaves: "葉数",
  min_child_samples: "子ノード最小サンプル数",
  iterations: "反復回数",
  depth: "深さ",
  random_strength: "ランダム強度",
  l2_leaf_reg: "葉のL2正則化",
  kernel: "カーネル",
  C: "誤分類ペナルティ C",
  weights: "近傍の重み",
  n_neighbors: "近傍数",
  penalty: "正則化方式",
  solver: "ソルバー",
  probability: "確率出力",
  priors: "事前確率",
  hidden_layer_sizes: "隠れ層構成",
  activation: "活性化関数",
  learning_rate_init: "初期学習率",
  fit_intercept: "切片を使用",
  alpha_1: "α1",
  alpha_2: "α2",
  lambda_1: "λ1",
  lambda_2: "λ2",
};

function valuesEqual(left, right) {
  return JSON.stringify(left) === JSON.stringify(right);
}

function displayValue(value) {
  if (Array.isArray(value)) return value.join(" × ");
  if (value === null) return "None";
  if (value === true) return "有効";
  if (value === false) return "無効";
  return String(value);
}

function formatMetric(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) return String(value ?? "-");
  if (Math.abs(value) >= 1000 || (Math.abs(value) > 0 && Math.abs(value) < 0.001)) {
    return value.toExponential(4);
  }
  return value.toFixed(4);
}

function normalizeNumericValue(parameter, rawValue) {
  let value = Number(rawValue);
  if (!Number.isFinite(value)) value = Number(parameter.default_value ?? parameter.low ?? 0);
  if (parameter.control === "integer") value = Math.round(value);
  if (parameter.low !== null && parameter.low !== undefined) {
    value = Math.max(Number(parameter.low), value);
  }
  if (parameter.high !== null && parameter.high !== undefined) {
    value = Math.min(Number(parameter.high), value);
  }
  return value;
}

function ParameterControl({ parameter, value, onChange }) {
  const label = PARAMETER_LABELS[parameter.name] || parameter.label || parameter.name;

  if (!parameter.editable || parameter.control === "readonly") {
    return (
      <div className="parameter-control readonly-parameter">
        <div>
          <strong>{label}</strong>
          <span>{parameter.note || "既定値を使用します。"}</span>
        </div>
        <code>{displayValue(parameter.default_value)}</code>
      </div>
    );
  }

  if (parameter.control === "boolean") {
    return (
      <div className="parameter-control boolean-parameter">
        <div>
          <strong>{label}</strong>
          <span>{parameter.name}</span>
        </div>
        <label className="switch-label">
          <input
            type="checkbox"
            checked={Boolean(value)}
            onChange={(event) => onChange(event.target.checked)}
          />
          <span />
          {Boolean(value) ? "有効" : "無効"}
        </label>
      </div>
    );
  }

  if (parameter.control === "categorical") {
    const choices = parameter.choices || [];
    const selectedIndex = Math.max(
      0,
      choices.findIndex((choice) => valuesEqual(choice, value)),
    );
    return (
      <label className="parameter-control categorical-parameter">
        <span className="parameter-label">
          <strong>{label}</strong>
          <small>{parameter.name}</small>
        </span>
        <select
          value={selectedIndex}
          onChange={(event) => onChange(choices[Number(event.target.value)])}
        >
          {choices.map((choice, index) => (
            <option key={`${parameter.name}-${index}`} value={index}>
              {displayValue(choice)}
            </option>
          ))}
        </select>
      </label>
    );
  }

  const low = Number(parameter.low);
  const high = Number(parameter.high);
  const numericValue = normalizeNumericValue(parameter, value);
  const logScale = Boolean(parameter.log && low > 0 && high > low);
  const rangeValue = logScale ? Math.log10(Math.max(numericValue, low)) : numericValue;
  const rangeMin = logScale ? Math.log10(low) : low;
  const rangeMax = logScale ? Math.log10(high) : high;
  const rangeStep = logScale
    ? 0.01
    : Number(parameter.step || Math.max((high - low) / 100, 0.0001));

  function updateFromRange(rawValue) {
    const transformed = logScale ? 10 ** Number(rawValue) : Number(rawValue);
    onChange(normalizeNumericValue(parameter, transformed));
  }

  return (
    <div className="parameter-control numeric-parameter">
      <div className="parameter-control-head">
        <span className="parameter-label">
          <strong>{label}</strong>
          <small>
            {parameter.name}
            {parameter.log ? " · log scale" : ""}
          </small>
        </span>
        <input
          className="parameter-number-input"
          type="number"
          min={parameter.low ?? undefined}
          max={parameter.high ?? undefined}
          step={parameter.step ?? "any"}
          value={numericValue}
          onChange={(event) => onChange(normalizeNumericValue(parameter, event.target.value))}
        />
      </div>
      <input
        className="parameter-slider"
        type="range"
        min={rangeMin}
        max={rangeMax}
        step={rangeStep}
        value={rangeValue}
        onChange={(event) => updateFromRange(event.target.value)}
      />
      <div className="parameter-range-labels">
        <span>{displayValue(parameter.low)}</span>
        <strong>{formatMetric(numericValue)}</strong>
        <span>{displayValue(parameter.high)}</span>
      </div>
    </div>
  );
}

function ParameterEditor({ target, schema, values, loading, onChange, onReset }) {
  if (loading && !schema) {
    return <p className="settings-note">モデルの調整可能パラメータを取得しています...</p>;
  }
  if (!schema?.parameters?.length) {
    return (
      <p className="settings-note">
        このモデルにはWeb画面で調整する探索パラメータがありません。malchanの既定値を使用します。
      </p>
    );
  }

  return (
    <div className="parameter-editor">
      <div className="parameter-editor-head">
        <div>
          <strong>個別パラメータ</strong>
          <span>Optunaの探索範囲を基準に値を指定します。</span>
        </div>
        <button type="button" className="secondary compact-button" onClick={onReset}>
          既定値へ戻す
        </button>
      </div>
      <div className="parameter-control-grid">
        {schema.parameters.map((parameter) => (
          <ParameterControl
            key={`${target}-${parameter.name}`}
            parameter={parameter}
            value={values?.[parameter.name] ?? parameter.default_value}
            onChange={(nextValue) => onChange(parameter.name, nextValue)}
          />
        ))}
      </div>
    </div>
  );
}

function TargetTabs({ targets, tasks, activeTarget, onChange, getDetail }) {
  if (targets.length <= 1) return null;

  return (
    <div className="model-target-tabs" role="tablist" aria-label="目的変数の切り替え">
      {targets.map((target) => (
        <button
          key={target}
          type="button"
          role="tab"
          aria-selected={target === activeTarget}
          className={target === activeTarget ? "active" : ""}
          onClick={() => onChange(target)}
        >
          <strong>{target}</strong>
          <span>
            {tasks[target] === "classification" ? "分類" : "回帰"}
            {getDetail ? ` · ${getDetail(target)}` : ""}
          </span>
        </button>
      ))}
    </div>
  );
}

function EvaluationPanel({ evaluation }) {
  if (!evaluation) return null;

  return (
    <article className="panel evaluation-result-panel">
      <div className="panel-title">
        <div>
          <span className="panel-kicker">ACCURACY VALIDATION</span>
          <h3>交差検証による精度評価</h3>
          <p>
            モデル選択やベストモデルの切り替えは行わず、登録済みモデルの設定をそのまま評価しています。
          </p>
        </div>
        <span className="status-chip success">
          {evaluation.method === "loo" ? "LOO" : `${evaluation.n_splits}-fold`}
        </span>
      </div>

      <div className="evaluation-target-grid">
        {Object.entries(evaluation.targets || {}).map(([target, result]) => {
          const train = result.train?.[0] || {};
          const test = result.test?.[0] || {};
          const metrics = Array.from(new Set([...Object.keys(train), ...Object.keys(test)]));
          return (
            <section className="evaluation-target-card" key={target}>
              <div className="target-model-card-head">
                <div>
                  <strong>{target}</strong>
                  <span>{result.task === "classification" ? "分類" : "回帰"}</span>
                </div>
              </div>
              <div className="table-wrap compact">
                <table>
                  <thead>
                    <tr>
                      <th>指標</th>
                      <th>Train</th>
                      <th>Validation</th>
                    </tr>
                  </thead>
                  <tbody>
                    {metrics.map((metric) => (
                      <tr key={`${target}-${metric}`}>
                        <td>{metric}</td>
                        <td>{formatMetric(train[metric])}</td>
                        <td>{formatMetric(test[metric])}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          );
        })}
      </div>
    </article>
  );
}

function PreprocessingPanel({ features, preprocessing, patchPreprocessing }) {
  return (
    <article className="panel model-settings-panel preprocessing-panel">
      <div className="panel-title">
        <div>
          <span className="panel-kicker">PREPROCESSING</span>
          <h3>前処理の設定</h3>
          <p>モデル選択とモデル比較の両方に共通して適用します。</p>
        </div>
        <span className="status-chip">{features.length} features</span>
      </div>

      <div className="model-settings-stack">
        <section className="model-setting-section">
          <div className="model-setting-heading">
            <div><strong>欠損値処理</strong><span>数値列とカテゴリ列の補完を設定します。</span></div>
            <label className="switch-label">
              <input
                type="checkbox"
                checked={preprocessing.impute}
                onChange={(event) => patchPreprocessing({
                  impute: event.target.checked,
                  numImputeType: event.target.checked
                    ? preprocessing.numImputeType || "mean"
                    : preprocessing.numImputeType,
                })}
              />
              <span />
              欠損を補完
            </label>
          </div>
          {preprocessing.impute && (
            <Field label="数値補完方式">
              <select
                value={preprocessing.numImputeType}
                onChange={(event) => patchPreprocessing({ numImputeType: event.target.value })}
              >
                <option value="mean">平均値</option>
                <option value="median">中央値</option>
                <option value="most_frequent">最頻値</option>
                <option value="knn">KNN補完</option>
                <option value="Multiple">多重代入</option>
              </select>
            </Field>
          )}
          <label className="switch-label setting-inline-switch">
            <input
              type="checkbox"
              checked={preprocessing.catImpute}
              onChange={(event) => patchPreprocessing({ catImpute: event.target.checked })}
            />
            <span />
            カテゴリ列を補完
          </label>
        </section>

        <section className="model-setting-section">
          <div className="model-setting-heading">
            <div><strong>数値スケーリング</strong><span>数値列に適用する変換を指定します。</span></div>
          </div>
          <Field label="スケーラー">
            <select
              value={preprocessing.numScaleType}
              onChange={(event) => patchPreprocessing({ numScaleType: event.target.value })}
            >
              <option value="">使用しない</option>
              <option value="StandardScaler">標準化</option>
              <option value="MinMaxScaler">Min-Max</option>
              <option value="centering">中心化のみ</option>
              <option value="MaxAbsScaler">最大絶対値</option>
            </select>
          </Field>
        </section>

        <section className="model-setting-section">
          <div className="model-setting-heading">
            <div><strong>多項式特徴量</strong><span>非線形項や交互作用項を追加します。</span></div>
            <label className="switch-label">
              <input
                type="checkbox"
                checked={preprocessing.poly}
                onChange={(event) => patchPreprocessing({ poly: event.target.checked })}
              />
              <span />
              使用する
            </label>
          </div>
          {preprocessing.poly && (
            <div className="model-inline-fields">
              <Field label="次数">
                <input
                  type="number"
                  min="1"
                  value={preprocessing.polyDegree}
                  onChange={(event) => patchPreprocessing({ polyDegree: event.target.value })}
                />
              </Field>
              <label className="switch-label setting-inline-switch">
                <input
                  type="checkbox"
                  checked={preprocessing.polyInteractionOnly}
                  onChange={(event) => patchPreprocessing({ polyInteractionOnly: event.target.checked })}
                />
                <span />
                交互作用項のみ
              </label>
            </div>
          )}
        </section>

        <section className="model-setting-section">
          <div className="model-setting-heading">
            <div><strong>次元削減</strong><span>学習前に特徴量を低次元へ圧縮します。</span></div>
            <label className="switch-label">
              <input
                type="checkbox"
                checked={preprocessing.decomposition}
                onChange={(event) => patchPreprocessing({ decomposition: event.target.checked })}
              />
              <span />
              使用する
            </label>
          </div>
          {preprocessing.decomposition && (
            <div className="model-inline-fields">
              <Field label="手法">
                <select
                  value={preprocessing.decompositionMethod}
                  onChange={(event) => patchPreprocessing({ decompositionMethod: event.target.value })}
                >
                  <option value="PCA">PCA</option>
                  <option value="KernelPCA">KernelPCA</option>
                  <option value="NMF">NMF</option>
                  <option value="ICA">ICA</option>
                </select>
              </Field>
              <Field label="成分数">
                <input
                  type="number"
                  min="1"
                  value={preprocessing.decNComponents}
                  onChange={(event) => patchPreprocessing({ decNComponents: event.target.value })}
                />
              </Field>
            </div>
          )}
        </section>

        <section className="model-setting-section">
          <div className="model-setting-heading">
            <div><strong>サンプリング</strong><span>不均衡分類で使用する方式を指定します。</span></div>
          </div>
          <Field label="方式">
            <select
              value={preprocessing.samplingMethod}
              onChange={(event) => patchPreprocessing({ samplingMethod: event.target.value })}
            >
              <option value="">使用しない</option>
              <option value="sample_weight">クラス重み</option>
              <option value="ros">Random Over Sampling</option>
              <option value="rus">Random Under Sampling</option>
              <option value="smote">SMOTE / SMOTENC</option>
            </select>
          </Field>
        </section>
      </div>
    </article>
  );
}

export default function ModelPage() {
  const {
    targets,
    tasks,
    features,
    modelNames,
    setModelNames,
    candidates,
    setCandidates,
    modelInfo,
    comparison,
    cvSplits,
    setCvSplits,
    activateBest,
    setActivateBest,
    ready,
    busy,
    trainModel,
    compareModels,
  } = useWorkbench();

  const [mode, setMode] = useState("selection");
  const [activeTarget, setActiveTarget] = useState("");
  const [preprocessing, setPreprocessing] = useState(DEFAULT_PREPROCESSING);
  const [parameterMode, setParameterMode] = useState("tuning");
  const [parameterSchemas, setParameterSchemas] = useState({});
  const [modelParamsByTarget, setModelParamsByTarget] = useState({});
  const [parameterLoading, setParameterLoading] = useState(false);
  const [accuracyValidation, setAccuracyValidation] = useState(true);
  const [cvMethod, setCvMethod] = useState("kfold");
  const [evaluation, setEvaluation] = useState(null);
  const [evaluating, setEvaluating] = useState(false);
  const [settingsError, setSettingsError] = useState("");

  const selectedCandidateCount = useMemo(
    () => targets.reduce((sum, target) => sum + (candidates[target]?.length || 0), 0),
    [targets, candidates],
  );

  const currentTarget = targets.includes(activeTarget) ? activeTarget : targets[0] || "";

  useEffect(() => {
    setActiveTarget((current) => (targets.includes(current) ? current : targets[0] || ""));
  }, [targets]);

  useEffect(() => {
    let cancelled = false;
    if (!targets.length) {
      setParameterSchemas({});
      setModelParamsByTarget({});
      return undefined;
    }

    setParameterLoading(true);
    Promise.all(
      targets.map(async (target) => {
        const schema = await api.modelParameters(tasks[target], modelNames[target]);
        return [target, schema];
      }),
    )
      .then((entries) => {
        if (cancelled) return;
        setParameterSchemas(Object.fromEntries(entries));
        setModelParamsByTarget((current) => {
          const next = {};
          entries.forEach(([target, schema]) => {
            const editable = (schema.parameters || []).filter((parameter) => parameter.editable);
            const defaults = Object.fromEntries(
              editable.map((parameter) => [parameter.name, parameter.default_value]),
            );
            const previous = current[target] || {};
            editable.forEach((parameter) => {
              if (Object.prototype.hasOwnProperty.call(previous, parameter.name)) {
                defaults[parameter.name] = previous[parameter.name];
              }
            });
            next[target] = defaults;
          });
          return next;
        });
      })
      .catch((error) => {
        if (!cancelled) setSettingsError(error.message || String(error));
      })
      .finally(() => {
        if (!cancelled) setParameterLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [targets, tasks, modelNames]);

  function patchPreprocessing(patch) {
    setPreprocessing((current) => ({ ...current, ...patch }));
  }

  function changeSelectedModel(target, modelName) {
    setModelNames({ ...modelNames, [target]: modelName });
  }

  function changeParameter(target, name, value) {
    setModelParamsByTarget((current) => ({
      ...current,
      [target]: {
        ...(current[target] || {}),
        [name]: value,
      },
    }));
  }

  function resetParameters(target) {
    const schema = parameterSchemas[target];
    setModelParamsByTarget((current) => ({
      ...current,
      [target]: Object.fromEntries(
        (schema?.parameters || [])
          .filter((parameter) => parameter.editable)
          .map((parameter) => [parameter.name, parameter.default_value]),
      ),
    }));
  }

  function resolvedManualParameters() {
    return Object.fromEntries(
      targets.map((target) => {
        const editableNames = new Set(
          (parameterSchemas[target]?.parameters || [])
            .filter((parameter) => parameter.editable)
            .map((parameter) => parameter.name),
        );
        const values = Object.fromEntries(
          Object.entries(modelParamsByTarget[target] || {})
            .filter(([name]) => editableNames.has(name)),
        );
        return [target, Object.keys(values).length ? values : null];
      }),
    );
  }

  async function executeModelSelection() {
    setSettingsError("");
    setEvaluation(null);

    const before = await api.listModels().catch(() => ({ models: [] }));
    await trainModel({
      preprocessing,
      tuning: parameterMode === "tuning",
      modelParamsByTarget: parameterMode === "manual" ? resolvedManualParameters() : {},
    });

    if (!accuracyValidation) return;

    try {
      setEvaluating(true);
      const after = await api.listModels();
      const previousIds = new Set((before.models || []).map((model) => model.model_id));
      const createdModel = [...(after.models || [])]
        .reverse()
        .find((model) => !previousIds.has(model.model_id));
      if (!createdModel) {
        throw new Error("新しく登録されたモデルを確認できなかったため、精度評価を実行できませんでした。");
      }
      const result = await api.evaluate(createdModel.model_id, {
        method: cvMethod,
        n_splits: cvMethod === "loo" ? 2 : Number(cvSplits),
      });
      setEvaluation(result);
    } catch (error) {
      setSettingsError(error.message || String(error));
    } finally {
      setEvaluating(false);
    }
  }

  async function executeModelComparison() {
    setSettingsError("");
    setEvaluation(null);
    await compareModels({
      preprocessing,
      tuning: false,
      modelParamsByTarget: {},
      cvMethod: "kfold",
      cvSplits: 5,
      activateBest,
      candidatesByTarget: candidates,
    });
  }

  const running = Boolean(busy) || evaluating;

  return (
    <>
      <SectionHeader
        step="4 · MODEL"
        title="前処理とモデルを設定する"
        text="左で共通の前処理を設定し、右でモデル選択またはモデル比較のワークフローを選びます。"
      />

      <div className="model-settings-columns">
        <PreprocessingPanel
          features={features}
          preprocessing={preprocessing}
          patchPreprocessing={patchPreprocessing}
        />

        <article className="panel model-settings-panel model-selection-panel">
          <div className="panel-title">
            <div>
              <span className="panel-kicker">MODELING WORKFLOW</span>
              <h3>モデルの設定</h3>
              <p>使用するモデルを決める操作と、複数モデルを比較する操作を分離しています。</p>
            </div>
            <span className={`status-chip ${ready ? "success" : "warning"}`}>
              {ready ? "Ready" : "Required"}
            </span>
          </div>

          <div className="model-mode-switch" role="tablist" aria-label="モデル設定モード">
            <button
              type="button"
              className={mode === "selection" ? "active" : ""}
              aria-selected={mode === "selection"}
              onClick={() => setMode("selection")}
            >
              <strong>モデル選択</strong>
              <span>使用モデルを学習し、必要に応じてCVで精度評価</span>
            </button>
            <button
              type="button"
              className={mode === "comparison" ? "active" : ""}
              aria-selected={mode === "comparison"}
              onClick={() => setMode("comparison")}
            >
              <strong>モデル比較</strong>
              <span>複数候補を比較し、ベストモデルの有効化を選択</span>
            </button>
          </div>

          {mode === "selection" && (
            <div className="model-workflow-content">
              <section className="model-setting-section">
                <div className="model-setting-heading">
                  <div>
                    <strong>パラメータ設定方法</strong>
                    <span>Optunaによる自動調整か、探索範囲を参考にした個別設定を選択します。</span>
                  </div>
                </div>
                <div className="parameter-mode-switch">
                  <button
                    type="button"
                    className={parameterMode === "tuning" ? "active" : ""}
                    onClick={() => setParameterMode("tuning")}
                  >
                    チューニング
                  </button>
                  <button
                    type="button"
                    className={parameterMode === "manual" ? "active" : ""}
                    onClick={() => setParameterMode("manual")}
                  >
                    個別設定
                  </button>
                </div>
                {parameterMode === "tuning" && (
                  <p className="settings-note">
                    選択したモデルのOptuna探索範囲を使用してパラメータを自動調整します。
                  </p>
                )}
              </section>

              <TargetTabs
                targets={targets}
                tasks={tasks}
                activeTarget={currentTarget}
                onChange={setActiveTarget}
                getDetail={(target) => modelNames[target] || "未選択"}
              />

              {currentTarget ? (
                <section
                  className="target-model-card model-target-tab-panel"
                  role="tabpanel"
                  aria-label={`${currentTarget}のモデル設定`}
                >
                  <div className="target-model-card-head">
                    <div>
                      <strong>{currentTarget}</strong>
                      <span>{tasks[currentTarget] === "classification" ? "分類" : "回帰"}</span>
                    </div>
                  </div>
                  <Field label="使用モデル">
                    <select
                      value={modelNames[currentTarget] || ""}
                      onChange={(event) => changeSelectedModel(currentTarget, event.target.value)}
                    >
                      {modelsFor(tasks[currentTarget]).map((model) => (
                        <option key={model}>{model}</option>
                      ))}
                    </select>
                  </Field>

                  {parameterMode === "manual" && (
                    <ParameterEditor
                      target={currentTarget}
                      schema={parameterSchemas[currentTarget]}
                      values={modelParamsByTarget[currentTarget]}
                      loading={parameterLoading}
                      onChange={(name, value) => changeParameter(currentTarget, name, value)}
                      onReset={() => resetParameters(currentTarget)}
                    />
                  )}
                </section>
              ) : (
                <p className="empty-state">PREPARE画面で目的変数を選択してください。</p>
              )}

              <section className="model-setting-section model-execution-settings">
                <div className="model-setting-heading">
                  <div>
                    <strong>精度検証</strong>
                    <span>学習したモデルをcv_scoreで評価するだけで、モデル選択や設定変更は行いません。</span>
                  </div>
                  <label className="switch-label">
                    <input
                      type="checkbox"
                      checked={accuracyValidation}
                      onChange={(event) => setAccuracyValidation(event.target.checked)}
                    />
                    <span />
                    交差検証を実施
                  </label>
                </div>
                {accuracyValidation && (
                  <div className="model-inline-fields cv-fields">
                    <Field label="検証方式">
                      <select value={cvMethod} onChange={(event) => setCvMethod(event.target.value)}>
                        <option value="kfold">K-fold</option>
                        <option value="loo">Leave-One-Out</option>
                      </select>
                    </Field>
                    {cvMethod === "kfold" && (
                      <Field label="分割数">
                        <input
                          type="number"
                          min="2"
                          value={cvSplits}
                          onChange={(event) => setCvSplits(event.target.value)}
                        />
                      </Field>
                    )}
                  </div>
                )}
              </section>

              <div className="model-run-summary">
                <div>
                  <span>MODEL SELECTION</span>
                  <strong>{features.length} features / {targets.length} targets</strong>
                  <small>
                    {parameterMode === "tuning" ? "Optuna tuning" : "Manual parameters"}
                    {" · "}
                    {accuracyValidation
                      ? cvMethod === "loo" ? "LOO evaluation" : `${cvSplits}-fold evaluation`
                      : "No accuracy validation"}
                  </small>
                </div>
                <button disabled={!ready || running} onClick={executeModelSelection}>
                  モデル学習を実行 →
                </button>
              </div>
            </div>
          )}

          {mode === "comparison" && (
            <div className="model-workflow-content">
              <p className="settings-note comparison-note">
                比較候補は使用モデルとは独立しています。候補は共通条件の5-fold評価でランキングします。
              </p>

              <TargetTabs
                targets={targets}
                tasks={tasks}
                activeTarget={currentTarget}
                onChange={setActiveTarget}
                getDetail={(target) => `${candidates[target]?.length || 0} models`}
              />

              {currentTarget ? (
                <section
                  className="target-model-card model-target-tab-panel"
                  role="tabpanel"
                  aria-label={`${currentTarget}の比較モデル設定`}
                >
                  <div className="target-model-card-head">
                    <div>
                      <strong>{currentTarget}</strong>
                      <span>{tasks[currentTarget] === "classification" ? "分類" : "回帰"}</span>
                    </div>
                    <span className="status-chip">
                      {candidates[currentTarget]?.length || 0} models
                    </span>
                  </div>
                  <CheckboxList
                    values={modelsFor(tasks[currentTarget])}
                    selected={candidates[currentTarget] || []}
                    onChange={(values) => setCandidates({
                      ...candidates,
                      [currentTarget]: values,
                    })}
                  />
                </section>
              ) : (
                <p className="empty-state">PREPARE画面で目的変数を選択してください。</p>
              )}

              <section className="model-setting-section model-execution-settings">
                <div className="model-setting-heading">
                  <div>
                    <strong>比較後のモデル</strong>
                    <span>ランキング1位のモデルを後続の予測・可視化で使用するか指定します。</span>
                  </div>
                  <label className="switch-label">
                    <input
                      type="checkbox"
                      checked={activateBest}
                      onChange={(event) => setActivateBest(event.target.checked)}
                    />
                    <span />
                    ベストモデルを有効化
                  </label>
                </div>
              </section>

              <div className="model-run-summary comparison-run-summary">
                <div>
                  <span>MODEL COMPARISON</span>
                  <strong>{selectedCandidateCount} candidate selections</strong>
                  <small>{activateBest ? "Best model will be activated" : "Keep comparison result only"}</small>
                </div>
                <button disabled={!ready || running} onClick={executeModelComparison}>
                  モデル比較を実行 →
                </button>
              </div>
            </div>
          )}

          {settingsError && <p className="xai-error model-settings-error">{settingsError}</p>}
        </article>
      </div>

      {modelInfo && (
        <article className="panel model-registration-panel">
          <div className="panel-title">
            <div>
              <span className="panel-kicker">REGISTERED MODEL</span>
              <h3>登録モデル</h3>
            </div>
            <span className="status-chip success">Active</span>
          </div>
          <pre className="codebox">{JSON.stringify(modelInfo, null, 2)}</pre>
        </article>
      )}

      {mode === "selection" && <EvaluationPanel evaluation={evaluation} />}
      {mode === "comparison" && <ComparisonTable comparison={comparison} />}
    </>
  );
}
