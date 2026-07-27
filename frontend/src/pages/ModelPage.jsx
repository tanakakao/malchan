import React, { useEffect, useMemo, useState } from "react";
import ComparisonTable from "../components/ComparisonTable";
import { CheckboxList, Field, SectionHeader } from "../components/Common";
import { modelsFor, useWorkbench } from "../context/WorkbenchContext";

const MODEL_PARAM_EXAMPLES = {
  Ridge: { alpha: 1.0 },
  Lasso: { alpha: 0.01, max_iter: 2000 },
  ElasticNet: { alpha: 0.01, l1_ratio: 0.5, max_iter: 2000 },
  "ランダムフォレスト回帰": { n_estimators: 300, max_depth: null, random_state: 42 },
  ランダムフォレスト: { n_estimators: 300, max_depth: null, random_state: 42 },
  "Extra-Trees": { n_estimators: 300, max_depth: null, random_state: 42 },
  "Gradient Boosting": { n_estimators: 200, learning_rate: 0.05, max_depth: 3 },
  XGBoost: { n_estimators: 300, learning_rate: 0.05, max_depth: 6 },
  LightGBM: { n_estimators: 300, learning_rate: 0.05, num_leaves: 31 },
  CatBoost: { iterations: 300, learning_rate: 0.05, depth: 6, verbose: false },
  サポートベクター回帰: { C: 1.0, kernel: "rbf", gamma: "scale" },
  サポートベクターマシン: { C: 1.0, kernel: "rbf", gamma: "scale" },
  K近傍法: { n_neighbors: 5, weights: "uniform" },
};

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

function parseParams(text, target) {
  const trimmed = text.trim();
  if (!trimmed || trimmed === "{}") return null;
  let parsed;
  try {
    parsed = JSON.parse(trimmed);
  } catch (error) {
    throw new Error(`${target}のモデルパラメータがJSONとして不正です: ${error.message}`);
  }
  if (!parsed || Array.isArray(parsed) || typeof parsed !== "object") {
    throw new Error(`${target}のモデルパラメータはJSONオブジェクトで指定してください。`);
  }
  return parsed;
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
    tuneBest,
    setTuneBest,
    activateBest,
    setActivateBest,
    trials,
    setTrials,
    ready,
    busy,
    trainModel,
    compareModels,
    tuneBestLater,
  } = useWorkbench();
  const [preprocessing, setPreprocessing] = useState(DEFAULT_PREPROCESSING);
  const [crossValidation, setCrossValidation] = useState(true);
  const [cvMethod, setCvMethod] = useState("kfold");
  const [modelParamsText, setModelParamsText] = useState({});
  const [settingsError, setSettingsError] = useState("");

  useEffect(() => {
    setModelParamsText((current) => Object.fromEntries(
      targets.map((target) => [target, current[target] ?? "{}"]),
    ));
  }, [targets]);

  const selectedCandidateCount = useMemo(
    () => targets.reduce((sum, target) => sum + (candidates[target]?.length || 0), 0),
    [targets, candidates],
  );

  function patchPreprocessing(patch) {
    setPreprocessing((current) => ({ ...current, ...patch }));
  }

  function changeModel(target, modelName) {
    setModelNames({ ...modelNames, [target]: modelName });
    setCandidates({
      ...candidates,
      [target]: candidates[target]?.includes(modelName)
        ? candidates[target]
        : [modelName, ...(candidates[target] || [])],
    });
  }

  function resetParams(target) {
    const example = MODEL_PARAM_EXAMPLES[modelNames[target]] || {};
    setModelParamsText({
      ...modelParamsText,
      [target]: JSON.stringify(example, null, 2),
    });
  }

  function buildTrainingOptions() {
    const modelParamsByTarget = tuneBest
      ? {}
      : Object.fromEntries(
          targets.map((target) => [target, parseParams(modelParamsText[target] || "{}", target)]),
        );
    return {
      preprocessing,
      tuning: tuneBest,
      modelParamsByTarget,
      cvMethod,
      cvSplits: Number(cvSplits),
      trials: Number(trials),
      activateBest,
      candidatesByTarget: candidates,
    };
  }

  async function executeModeling() {
    setSettingsError("");
    let options;
    try {
      options = buildTrainingOptions();
    } catch (error) {
      setSettingsError(error.message || String(error));
      return;
    }
    if (crossValidation) {
      await compareModels(options);
    } else {
      await trainModel(options);
    }
  }

  return (
    <>
      <SectionHeader
        step="4 · MODEL"
        title="前処理とモデルを設定する"
        text="左で学習前のデータ処理、右でモデル・チューニング・手動パラメータ・交差検証を設定します。"
      />

      <div className="model-settings-columns">
        <article className="panel model-settings-panel preprocessing-panel">
          <div className="panel-title">
            <div>
              <span className="panel-kicker">PREPROCESSING</span>
              <h3>前処理の設定</h3>
              <p>選択した説明変数へ共通して適用する処理です。</p>
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
                <div><strong>数値スケーリング</strong><span>未指定の場合はモデルPipelineの既定動作を使用します。</span></div>
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
                <div><strong>サンプリング</strong><span>不均衡分類などで使用するサンプリング方式です。</span></div>
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

        <article className="panel model-settings-panel model-selection-panel">
          <div className="panel-title">
            <div>
              <span className="panel-kicker">MODEL & VALIDATION</span>
              <h3>モデルと検証の設定</h3>
              <p>目的変数ごとにモデルを選択し、学習方法を指定します。</p>
            </div>
            <span className={`status-chip ${ready ? "success" : "warning"}`}>
              {ready ? "Ready" : "Required"}
            </span>
          </div>

          <div className="target-model-cards">
            {targets.map((target) => (
              <section className="target-model-card" key={target}>
                <div className="target-model-card-head">
                  <div><strong>{target}</strong><span>{tasks[target] === "classification" ? "分類" : "回帰"}</span></div>
                </div>
                <Field label="使用モデル">
                  <select
                    value={modelNames[target] || ""}
                    onChange={(event) => changeModel(target, event.target.value)}
                  >
                    {modelsFor(tasks[target]).map((model) => <option key={model}>{model}</option>)}
                  </select>
                </Field>

                {!tuneBest && (
                  <div className="model-parameter-editor">
                    <div className="model-parameter-head">
                      <label htmlFor={`params-${target}`}>モデルパラメータ（JSON）</label>
                      <button type="button" className="secondary compact-button" onClick={() => resetParams(target)}>
                        例を入力
                      </button>
                    </div>
                    <textarea
                      id={`params-${target}`}
                      rows="7"
                      spellCheck="false"
                      value={modelParamsText[target] || "{}"}
                      onChange={(event) => setModelParamsText({
                        ...modelParamsText,
                        [target]: event.target.value,
                      })}
                    />
                  </div>
                )}
                {!tuneBest && crossValidation && (
                  <p className="settings-note">このJSONは選択中の使用モデルへ適用し、その他のCV候補は既定パラメータで評価します。</p>
                )}

                {crossValidation && (
                  <details className="advanced model-candidate-details">
                    <summary>
                      <strong>交差検証の候補モデル</strong>
                      <span>{candidates[target]?.length || 0} models</span>
                    </summary>
                    <CheckboxList
                      values={modelsFor(tasks[target])}
                      selected={candidates[target] || []}
                      onChange={(values) => setCandidates({ ...candidates, [target]: values })}
                    />
                  </details>
                )}
              </section>
            ))}
            {!targets.length && <p className="empty-state">PREPARE画面で目的変数を選択してください。</p>}
          </div>

          <section className="model-setting-section model-execution-settings">
            <div className="model-setting-heading">
              <div><strong>ハイパーパラメータチューニング</strong><span>OFFの場合は上のJSON設定をそのまま使用します。</span></div>
              <label className="switch-label">
                <input type="checkbox" checked={tuneBest} onChange={(event) => setTuneBest(event.target.checked)} />
                <span />
                チューニングする
              </label>
            </div>
            {tuneBest && crossValidation && (
              <Field label="Optuna試行数">
                <input type="number" min="1" value={trials} onChange={(event) => setTrials(event.target.value)} />
              </Field>
            )}
            {tuneBest && !crossValidation && (
              <p className="settings-note">直接学習時のチューニングはPipeline側の既定探索条件を使用します。</p>
            )}
          </section>

          <section className="model-setting-section model-execution-settings">
            <div className="model-setting-heading">
              <div><strong>交差検証</strong><span>OFFは全データで直接学習、ONは候補モデルをCV評価します。</span></div>
              <label className="switch-label">
                <input
                  type="checkbox"
                  checked={crossValidation}
                  onChange={(event) => setCrossValidation(event.target.checked)}
                />
                <span />
                実施する
              </label>
            </div>
            {crossValidation && (
              <div className="model-inline-fields cv-fields">
                <Field label="検証方式">
                  <select value={cvMethod} onChange={(event) => setCvMethod(event.target.value)}>
                    <option value="kfold">K-fold</option>
                    <option value="loo">Leave-One-Out</option>
                  </select>
                </Field>
                {cvMethod === "kfold" && (
                  <Field label="分割数">
                    <input type="number" min="2" value={cvSplits} onChange={(event) => setCvSplits(event.target.value)} />
                  </Field>
                )}
              </div>
            )}
            {crossValidation && (
              <label className="switch-label setting-inline-switch">
                <input type="checkbox" checked={activateBest} onChange={(event) => setActivateBest(event.target.checked)} />
                <span />
                評価後にベストモデルを有効化
              </label>
            )}
          </section>

          {settingsError && <p className="xai-error model-settings-error">{settingsError}</p>}

          <div className="model-run-summary">
            <div>
              <span>{crossValidation ? "CROSS VALIDATION" : "DIRECT TRAINING"}</span>
              <strong>{features.length} features / {targets.length} targets</strong>
              <small>
                {crossValidation
                  ? `${selectedCandidateCount} candidate selections / ${cvMethod === "loo" ? "LOO" : `${cvSplits}-fold`}`
                  : tuneBest ? "Pipeline tuning enabled" : "Manual parameters"}
              </small>
            </div>
            <button disabled={!ready || busy} onClick={executeModeling}>
              {crossValidation ? "交差検証を実行 →" : "モデル学習を実行 →"}
            </button>
          </div>
        </article>
      </div>

      {modelInfo && (
        <article className="panel model-registration-panel">
          <div className="panel-title">
            <div><span className="panel-kicker">REGISTERED MODEL</span><h3>登録モデル</h3></div>
            <span className="status-chip success">Active</span>
          </div>
          <pre className="codebox">{JSON.stringify(modelInfo, null, 2)}</pre>
        </article>
      )}

      {comparison && (
        <article className="panel comparison-followup-panel">
          <div className="panel-title">
            <div>
              <span className="panel-kicker">FOLLOW-UP</span>
              <h3>交差検証後の操作</h3>
              <p>現在の比較結果に対して、後からベストモデルだけを再チューニングできます。</p>
            </div>
            <button className="secondary" disabled={busy} onClick={tuneBestLater}>ベストを再チューニング</button>
          </div>
        </article>
      )}

      <ComparisonTable comparison={comparison} />
    </>
  );
}
