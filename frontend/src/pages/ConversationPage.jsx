import React, { useEffect, useMemo, useRef, useState } from "react";
import ConversationIcon from "../components/ConversationIcon";
import { useWorkbench } from "../context/WorkbenchContext";
import "../conversation-mode.css";

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

const STAGE_ORDER = ["data", "target", "task", "features", "confirm", "result"];
let messageSequence = 0;

function nextMessage(role, text) {
  messageSequence += 1;
  return { id: messageSequence, role, text };
}

function modelsFor(task) {
  return task === "classification"
    ? SIMPLE_CLASSIFICATION_MODELS
    : SIMPLE_REGRESSION_MODELS;
}

function taskLabel(task) {
  return task === "classification" ? "分類" : "回帰";
}

function includesColumn(text, column) {
  return text.toLocaleLowerCase().includes(column.toLocaleLowerCase());
}

export default function ConversationPage() {
  const {
    rows,
    columns,
    numeric,
    categorical,
    targets,
    tasks,
    numFeatures,
    setNumFeatures,
    catFeatures,
    setCatFeatures,
    comparison,
    modelInfo,
    busy,
    toast,
    loadFile,
    changeTargets,
    changeTask,
    compareModels,
    setStep,
  } = useWorkbench();

  const initialTarget = targets[0] || columns.at(-1) || "";
  const [stage, setStage] = useState(rows.length ? "target" : "data");
  const [messages, setMessages] = useState([
    nextMessage(
      "assistant",
      rows.length
        ? "データを確認しました。まず、予測したい目的変数を選んでください。"
        : "機械学習モデルを一緒に作ります。まず、CSVまたはExcelのデータを読み込んでください。",
    ),
  ]);
  const [draftTarget, setDraftTarget] = useState(initialTarget);
  const [draftTask, setDraftTask] = useState(
    initialTarget && categorical.includes(initialTarget) ? "classification" : "regression",
  );
  const [draftFeatures, setDraftFeatures] = useState(
    [...numFeatures, ...catFeatures].filter((column) => column !== initialTarget),
  );
  const [inputText, setInputText] = useState("");
  const [awaitingResult, setAwaitingResult] = useState(false);
  const previousComparison = useRef(comparison);
  const previousRowCount = useRef(rows.length);

  const featureCandidates = useMemo(
    () => columns.filter((column) => column !== draftTarget),
    [columns, draftTarget],
  );
  const selectedFeatureSet = useMemo(() => new Set(draftFeatures), [draftFeatures]);
  const targetResult = draftTarget ? comparison?.targets?.[draftTarget] : null;
  const bestModelName = targetResult?.best_model_name || "—";
  const selectedModels = modelsFor(draftTask);

  function append(role, text) {
    setMessages((current) => [...current, nextMessage(role, text)]);
  }

  function applyFeatures(nextFeatures) {
    const clean = featureCandidates.filter((column) => nextFeatures.includes(column));
    setDraftFeatures(clean);
    setNumFeatures(clean.filter((column) => numeric.includes(column)));
    setCatFeatures(clean.filter((column) => categorical.includes(column)));
  }

  function resetConversation(datasetLoaded = rows.length > 0) {
    const target = targets[0] || columns.at(-1) || "";
    const task = target && categorical.includes(target) ? "classification" : "regression";
    const features = columns.filter((column) => column !== target);
    setDraftTarget(target);
    setDraftTask(task);
    setDraftFeatures(features);
    setStage(datasetLoaded ? "target" : "data");
    setMessages([
      nextMessage(
        "assistant",
        datasetLoaded
          ? "データを確認しました。予測したい目的変数を選んでください。"
          : "機械学習モデルを一緒に作ります。まず、CSVまたはExcelのデータを読み込んでください。",
      ),
    ]);
    setInputText("");
    setAwaitingResult(false);
  }

  useEffect(() => {
    if (rows.length === previousRowCount.current) return;
    previousRowCount.current = rows.length;
    if (rows.length) resetConversation(true);
  }, [rows.length]);

  useEffect(() => {
    if (!awaitingResult || !comparison || comparison === previousComparison.current) return;
    setAwaitingResult(false);
    setStage("result");
    append(
      "assistant",
      `${draftTarget}について候補モデルの比較が完了しました。最良モデルは「${comparison.targets?.[draftTarget]?.best_model_name || "選定済みモデル"}」です。`,
    );
  }, [awaitingResult, comparison, draftTarget]);

  useEffect(() => {
    if (!awaitingResult || busy || toast?.type !== "error") return;
    setAwaitingResult(false);
    setStage("confirm");
    append("assistant", "モデル比較を完了できませんでした。エラー内容を確認して、変数選択を修正してください。");
  }, [awaitingResult, busy, toast]);

  function selectTarget(column) {
    const inferredTask = categorical.includes(column) ? "classification" : "regression";
    const nextFeatures = columns.filter((name) => name !== column);
    changeTargets([column]);
    changeTask(column, inferredTask);
    setDraftTarget(column);
    setDraftTask(inferredTask);
    setDraftFeatures(nextFeatures);
    setNumFeatures(nextFeatures.filter((name) => numeric.includes(name)));
    setCatFeatures(nextFeatures.filter((name) => categorical.includes(name)));
    append("user", `${column}を予測したいです。`);
    append("assistant", `${column}は初期判定では${taskLabel(inferredTask)}です。タスクを確認してください。`);
    setStage("task");
  }

  function selectTask(task) {
    if (!draftTarget) return;
    changeTask(draftTarget, task);
    setDraftTask(task);
    append("user", `${taskLabel(task)}として扱います。`);
    append("assistant", "次に、予測へ使用する説明変数を選んでください。初期状態では目的変数以外をすべて選択しています。");
    setStage("features");
  }

  function toggleFeature(column) {
    const next = selectedFeatureSet.has(column)
      ? draftFeatures.filter((name) => name !== column)
      : [...draftFeatures, column];
    applyFeatures(next);
  }

  function confirmFeatures() {
    if (!draftFeatures.length) return;
    append("user", `${draftFeatures.join("、")}を説明変数として使います。`);
    append("assistant", "設定内容をまとめました。4種類の候補モデルを5-fold交差検証で比較します。確認後に実行してください。");
    setStage("confirm");
  }

  async function executeComparison() {
    if (!draftTarget || !draftFeatures.length || busy || awaitingResult) return;
    previousComparison.current = comparison;
    setAwaitingResult(true);
    append("user", "この内容でモデルを比較してください。 ");
    append("assistant", "候補モデルを同じ条件で交差検証し、最良モデルを自動で有効化します。");
    await compareModels({
      preprocessing: SIMPLE_PREPROCESSING,
      tuning: false,
      cvMethod: "kfold",
      cvSplits: 5,
      activateBest: true,
      candidatesByTarget: { [draftTarget]: selectedModels },
    });
  }

  function openExistingScreen(step) {
    window.history.replaceState(null, "", `${window.location.pathname}${window.location.search}`);
    window.dispatchEvent(new HashChangeEvent("hashchange"));
    setStep(step);
  }

  function handleTextSubmit(event) {
    event.preventDefault();
    const text = inputText.trim();
    if (!text) return;
    setInputText("");

    if (/やり直|最初|リセット/.test(text)) {
      append("user", text);
      resetConversation(rows.length > 0);
      return;
    }

    if (stage === "target") {
      const matched = columns.find((column) => includesColumn(text, column));
      if (matched) return selectTarget(matched);
    }

    if (stage === "task") {
      if (/分類|クラス|判定/.test(text)) return selectTask("classification");
      if (/回帰|数値|予測/.test(text)) return selectTask("regression");
    }

    if (stage === "features") {
      if (/すべて|全部/.test(text)) {
        applyFeatures(featureCandidates);
        append("user", text);
        append("assistant", "目的変数以外をすべて説明変数として選択しました。「この変数で進む」を押してください。");
        return;
      }
      const matched = featureCandidates.filter((column) => includesColumn(text, column));
      if (matched.length) {
        applyFeatures(matched);
        append("user", text);
        append("assistant", `${matched.join("、")}を説明変数として選択しました。`);
        return;
      }
    }

    if (stage === "confirm" && /実行|比較|学習|進め/.test(text)) {
      append("user", text);
      void executeComparison();
      return;
    }

    append("user", text);
    append("assistant", "現在の選択肢から回答するか、列名・「回帰」「分類」「すべて」「比較を実行」などを入力してください。");
  }

  const stageIndex = STAGE_ORDER.indexOf(stage);

  return (
    <div className="conversation-page">
      <div className="conversation-heading">
        <div>
          <span className="conversation-kicker">GUIDED MODELING</span>
          <h2>対話しながらモデルを自動選択</h2>
          <p>目的変数と説明変数を順番に確認し、線形モデル・ElasticNet・ランダムフォレスト・LightGBMを同じ条件で比較します。</p>
        </div>
        <div className="conversation-heading-actions">
          <button type="button" className="secondary" onClick={() => resetConversation(rows.length > 0)}>
            最初からやり直す
          </button>
          {rows.length > 0 && (
            <button type="button" className="secondary" onClick={() => openExistingScreen("prepare")}>
              画面で設定する
            </button>
          )}
        </div>
      </div>

      <div className="conversation-layout">
        <section className="conversation-thread" aria-label="malchanとの対話">
          <div className="conversation-messages" aria-live="polite">
            {messages.map((message) => (
              <div key={message.id} className={`conversation-message ${message.role}`}>
                <ConversationIcon
                  fallback={message.role === "assistant" ? "m" : "自"}
                  className="conversation-avatar"
                />
                <div className="conversation-bubble">{message.text}</div>
              </div>
            ))}

            {stage === "data" && (
              <div className="conversation-action-card">
                <strong>分析データを読み込む</strong>
                <p>CSV、XLSX、XLSに対応しています。最終列を初期の目的変数として推定しますが、次の画面で変更できます。</p>
                <label className="conversation-file-button">
                  <input
                    type="file"
                    accept=".csv,.xlsx,.xls"
                    onChange={(event) => void loadFile(event.target.files?.[0] || null)}
                  />
                  ファイルを選択
                </label>
              </div>
            )}

            {stage === "target" && rows.length > 0 && (
              <div className="conversation-action-card">
                <strong>予測したい目的変数を選ぶ</strong>
                <p>対話モードでは、まず1つの目的変数を対象にモデルを比較します。複数目的は既存のPrepare画面で設定できます。</p>
                <div className="conversation-choice-grid">
                  {columns.map((column) => (
                    <button key={column} type="button" className="secondary" onClick={() => selectTarget(column)}>
                      <strong>{column}</strong>
                      <small>{categorical.includes(column) ? "カテゴリ列 · 分類候補" : "数値列 · 回帰候補"}</small>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {stage === "task" && (
              <div className="conversation-action-card">
                <strong>{draftTarget}をどのタスクとして扱いますか？</strong>
                <div className="conversation-choice-grid two-columns">
                  <button type="button" onClick={() => selectTask("regression")}>
                    <strong>回帰</strong>
                    <small>連続値や数値を予測する</small>
                  </button>
                  <button type="button" className="secondary" onClick={() => selectTask("classification")}>
                    <strong>分類</strong>
                    <small>クラスや良否を判定する</small>
                  </button>
                </div>
              </div>
            )}

            {stage === "features" && (
              <div className="conversation-action-card">
                <strong>予測へ使用する説明変数</strong>
                <p>選択した列をモデル入力として使用します。青いカードが選択中です。</p>
                <div className="conversation-feature-list">
                  {featureCandidates.map((column) => {
                    const selected = selectedFeatureSet.has(column);
                    return (
                      <button
                        key={column}
                        type="button"
                        className={`conversation-feature ${selected ? "selected" : ""}`}
                        aria-pressed={selected}
                        onClick={() => toggleFeature(column)}
                      >
                        <span className="conversation-check">{selected ? "✓" : ""}</span>
                        <span>
                          <strong>{column}</strong>
                          <small>{categorical.includes(column) ? "カテゴリ変数" : "数値変数"}</small>
                        </span>
                      </button>
                    );
                  })}
                </div>
                <div className="conversation-card-actions">
                  <button type="button" className="secondary" onClick={() => applyFeatures(featureCandidates)}>全選択</button>
                  <button type="button" className="secondary" onClick={() => applyFeatures([])}>解除</button>
                  <button type="button" disabled={!draftFeatures.length} onClick={confirmFeatures}>この変数で進む</button>
                </div>
              </div>
            )}

            {stage === "confirm" && (
              <div className="conversation-action-card conversation-confirm-card">
                <strong>この内容でモデルを比較します</strong>
                <dl>
                  <div><dt>目的変数</dt><dd>{draftTarget}</dd></div>
                  <div><dt>タスク</dt><dd>{taskLabel(draftTask)}</dd></div>
                  <div><dt>説明変数</dt><dd>{draftFeatures.join("、")}</dd></div>
                  <div><dt>候補モデル</dt><dd>{selectedModels.join(" / ")}</dd></div>
                  <div><dt>評価</dt><dd>5-fold CV · {draftTask === "classification" ? "F1" : "RMSE"}</dd></div>
                  <div><dt>採用方法</dt><dd>1位のモデルを自動で有効化</dd></div>
                </dl>
                <button type="button" disabled={Boolean(busy) || awaitingResult} onClick={() => void executeComparison()}>
                  {awaitingResult ? "モデルを比較中..." : "モデルを比較して自動選択"}
                </button>
              </div>
            )}

            {stage === "result" && targetResult && (
              <div className="conversation-action-card conversation-result-card">
                <span className="conversation-result-label">BEST MODEL</span>
                <h3>{bestModelName}</h3>
                <div className="conversation-result-values">
                  <div><span>目的変数</span><strong>{draftTarget}</strong></div>
                  <div><span>タスク</span><strong>{taskLabel(draftTask)}</strong></div>
                  <div><span>選定指標</span><strong>{targetResult.metric || (draftTask === "classification" ? "F1" : "RMSE")}</strong></div>
                  <div><span>モデル状態</span><strong>{modelInfo ? "有効化済み" : "比較完了"}</strong></div>
                </div>
                <p>候補モデルを同じ前処理と交差検証条件で比較し、最も評価の高いモデルを採用しました。</p>
                <div className="conversation-result-actions">
                  <button type="button" onClick={() => openExistingScreen("model")}>比較結果を詳しく見る</button>
                  <button type="button" className="secondary" onClick={() => openExistingScreen("explain")}>精度と変数影響を見る</button>
                  <button type="button" className="secondary" onClick={() => setStage("features")}>変数を変えて再比較</button>
                </div>
              </div>
            )}
          </div>

          <form className="conversation-composer" onSubmit={handleTextSubmit}>
            <input
              type="text"
              value={inputText}
              onChange={(event) => setInputText(event.target.value)}
              placeholder={rows.length ? "例：強度を予測、回帰、温度と時間を使う、比較を実行" : "データ読込後に自然文でも回答できます"}
              disabled={!rows.length || Boolean(busy)}
              aria-label="対話モードへの入力"
            />
            <button type="submit" disabled={!rows.length || !inputText.trim() || Boolean(busy)}>送信</button>
          </form>
        </section>

        <aside className="conversation-summary" aria-label="現在のモデル設定">
          <div className="conversation-summary-card">
            <span>CURRENT PLAN</span>
            <h3>現在のモデル設定</h3>
            <dl>
              <div><dt>データ</dt><dd>{rows.length ? `${rows.length}行` : "未読込"}</dd></div>
              <div><dt>目的変数</dt><dd>{draftTarget || "未選択"}</dd></div>
              <div><dt>タスク</dt><dd>{draftTarget ? taskLabel(draftTask) : "—"}</dd></div>
              <div><dt>説明変数</dt><dd>{draftFeatures.length ? `${draftFeatures.length}列` : "未選択"}</dd></div>
              <div><dt>比較モデル</dt><dd>{selectedModels.length}種類</dd></div>
              <div><dt>検証</dt><dd>5-fold CV</dd></div>
            </dl>
          </div>

          <div className="conversation-summary-card conversation-progress-card">
            <span>PROGRESS</span>
            <h3>対話の進行</h3>
            <ol>
              {[
                ["data", "データ読込"],
                ["target", "目的変数"],
                ["task", "回帰・分類"],
                ["features", "説明変数"],
                ["confirm", "内容確認"],
                ["result", "モデル選定"]
              ].map(([id, label], index) => (
                <li key={id} className={index < stageIndex ? "complete" : index === stageIndex ? "active" : ""}>
                  <span>{index < stageIndex ? "✓" : index + 1}</span>{label}
                </li>
              ))}
            </ol>
          </div>

          <div className="conversation-summary-card conversation-note-card">
            <span>ABOUT</span>
            <h3>既存画面との関係</h3>
            <p>対話モードは既存のデータ状態とFastAPIをそのまま使用します。高度な前処理、複数目的、個別モデル設定は既存のPrepare・Model画面から継続できます。</p>
          </div>
        </aside>
      </div>
    </div>
  );
}
