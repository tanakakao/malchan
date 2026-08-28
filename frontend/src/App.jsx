import React from "react";
import { WorkbenchProvider, useWorkbench } from "./context/WorkbenchContext";
import ComparisonTuningControl from "./components/ComparisonTuningControl";
import ConversationIcon from "./components/ConversationIcon";
import EnsembleModelSettingsControl from "./components/EnsembleModelSettingsControl";
import ImportedModelDefaultsControl from "./components/ImportedModelDefaultsControl";
import MaterialDescriptorSettingsControl from "./components/MaterialDescriptorSettingsControl";
import MaterialFeatureKindControl from "./components/MaterialFeatureKindControl";
import ModelBundleControl from "./components/ModelBundleControl";
import ModelResultVisualizationControl from "./components/ModelResultVisualizationControl";
import ModelSettingDefaultsControl from "./components/ModelSettingDefaultsControl";
import PersistentWorkflowPage from "./components/PersistentWorkflowPage";
import YyDiagnosticsControl from "./components/YyDiagnosticsControl";
import OptimizeCategoryCandidatesControl from "./components/OptimizeCategoryCandidatesControl";
import ShapScatterControl from "./components/ShapScatterControl";
import ConversationPage from "./pages/ConversationPage";
import DataPage from "./pages/DataPage";
import ExplorePage from "./pages/ExplorePage";
import PreparePage from "./pages/PreparePage";
import ModelPage from "./pages/ModelPage";
import SimpleModelResultPage from "./pages/SimpleModelResultPage";
import ExplainPage from "./pages/ExplainPage";
import PredictionPage from "./pages/PredictionPage";
import OptimizePage from "./pages/OptimizePage";
import ReportPage from "./pages/ReportPage";
import { getWorkflowCompletion, workflowStatusText } from "./workflowCompletion";
import { setWorkbenchMode, useWorkbenchMode } from "./workbenchMode";

const APP_STEPS = [
  ["data", "Data", "読込・確認"],
  ["explore", "Explore", "統計・可視化"],
  ["prepare", "Prepare", "変数・前処理"],
  ["model", "Model", "学習・比較"],
  ["explain", "Explain", "精度・挙動"],
  ["predict", "Predict", "予測・ローカルSHAP"],
  ["optimize", "Optimize", "逆解析"],
  ["report", "Report", "レポート"],
];

const SIMPLE_STEP_IDS = new Set([
  "data",
  "prepare",
  "model",
  "explain",
  "predict",
  "optimize",
  "report",
]);

const PERSISTENT_PAGE_IDS = new Set(["prepare", "model", "optimize"]);

const PAGES = {
  data: DataPage,
  explore: ExplorePage,
  prepare: PreparePage,
  model: ModelPage,
  explain: ExplainPage,
  predict: PredictionPage,
  optimize: OptimizePage,
  report: ReportPage,
};
const ICONS = {
  data: "▦",
  explore: "◫",
  prepare: "◇",
  model: "⌁",
  explain: "◎",
  predict: "◉",
  optimize: "↗",
  report: "▧",
};
const API_STATUS_LABELS = {
  loading: "確認中",
  ready: "利用可能",
  error: "エラー",
};
const PORTAL_URL = import.meta.env.VITE_PORTAL_URL?.trim() || "http://127.0.0.1:5172";

function currentAuxiliaryPage() {
  return window.location.hash === "#conversation" ? "conversation" : null;
}

function clearAuxiliaryHash() {
  window.history.replaceState(null, "", `${window.location.pathname}${window.location.search}`);
  window.dispatchEvent(new HashChangeEvent("hashchange"));
}

function WorkbenchLayout() {
  const mode = useWorkbenchMode();
  const [auxiliaryPage, setAuxiliaryPage] = React.useState(currentAuxiliaryPage);
  const [persistentPageIds, setPersistentPageIds] = React.useState([]);
  const {
    theme, setTheme, step, setStep, health, busy, toast, setToast,
    fileName, rows, columns, features, targets, ready, modelInfo, comparison,
    diagnostics, prediction, inverseResult, report,
  } = useWorkbench();
  const visibleSteps = mode === "simple"
    ? APP_STEPS.filter(([id]) => SIMPLE_STEP_IDS.has(id))
    : APP_STEPS;
  const workflowCompletion = getWorkflowCompletion({
    rows,
    columns,
    ready,
    modelInfo,
    comparison,
    diagnostics,
    prediction,
    inverseResult,
    report,
  });
  const activeAuxiliaryPage = auxiliaryPage === "conversation" ? "conversation" : null;
  const currentPageUsesCache = !activeAuxiliaryPage
    && PERSISTENT_PAGE_IDS.has(step)
    && !(mode === "simple" && step === "model");
  const persistentPagesToRender = currentPageUsesCache && !persistentPageIds.includes(step)
    ? [...persistentPageIds, step]
    : persistentPageIds;
  const Page = activeAuxiliaryPage === "conversation"
    ? ConversationPage
    : mode === "simple" && step === "model"
      ? SimpleModelResultPage
      : PAGES[step] || DataPage;
  const apiStatusLabel = API_STATUS_LABELS[health.status] || "利用不可";

  React.useEffect(() => {
    const handleHashChange = () => setAuxiliaryPage(currentAuxiliaryPage());
    window.addEventListener("hashchange", handleHashChange);
    return () => window.removeEventListener("hashchange", handleHashChange);
  }, []);

  React.useEffect(() => {
    if (mode === "simple" && step === "explore") {
      setStep(rows.length ? "prepare" : "data");
    }
  }, [mode, rows.length, setStep, step]);

  React.useEffect(() => {
    if (!currentPageUsesCache) return;
    setPersistentPageIds((current) => (
      current.includes(step) ? current : [...current, step]
    ));
  }, [currentPageUsesCache, step]);

  function openStep(id) {
    if (auxiliaryPage) clearAuxiliaryHash();
    setStep(id);
  }

  function openConversation() {
    window.location.hash = "conversation";
  }

  return (
    <div className="app-root">
      <header className="app-header">
        <div className="brand-lockup">
          <div className="brand-mark" aria-hidden="true"><span>m</span></div>
          <div className="brand-wordmark">
            <h1>機械学習</h1>
            <p>Materials Analysis Workbench · malchan</p>
          </div>
        </div>
        <div className="workflow-strip" aria-label="ワークフロー">
          {visibleSteps.map(([id, label], stepIndex) => {
            const stepStatus = workflowCompletion[id] || {};
            const active = !activeAuxiliaryPage && id === step;
            const statusText = workflowStatusText(stepStatus);
            return (
              <React.Fragment key={id}>
                <button
                  className={`workflow-step ${active ? "active" : ""} ${stepStatus.complete ? "complete" : ""} ${stepStatus.available ? "available" : ""} ${stepStatus.optional ? "optional" : ""}`}
                  onClick={() => openStep(id)}
                  aria-current={active ? "step" : undefined}
                  aria-label={`${label} · ${statusText}`}
                  title={`${label}: ${statusText}`}
                  data-workflow-status={stepStatus.complete ? "complete" : stepStatus.optional ? "optional" : stepStatus.available ? "available" : "pending"}
                >
                  <span>{stepStatus.complete ? "✓" : stepIndex + 1}</span><strong>{label}</strong>
                </button>
                {stepIndex < visibleSteps.length - 1 && (
                  <i className={stepStatus.complete ? "complete" : ""} aria-hidden="true" />
                )}
              </React.Fragment>
            );
          })}
        </div>
        <div className="header-actions">
          <div className="runtime-pill" title={`API接続: ${health.text}`}>
            <span className={`dot ${health.status}`} />
            <span className="runtime-copy">
              <small>API接続</small>
              <strong>{apiStatusLabel}</strong>
            </span>
          </div>
          <button
            className="portal-button secondary"
            title="ツール一覧へ戻る"
            onClick={() => window.location.assign(PORTAL_URL)}
          >
            ツール一覧
          </button>
          <button
            className="icon-button secondary"
            title={theme === "dark" ? "ライトテーマへ" : "ダークテーマへ"}
            aria-label={theme === "dark" ? "ライトテーマへ切り替える" : "ダークテーマへ切り替える"}
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
          >
            {theme === "dark" ? "☀" : "☾"}
          </button>
        </div>
      </header>

      <main className="app-shell">
        <aside className="left-rail">
          <button
            type="button"
            className={`conversation-launcher ${activeAuxiliaryPage === "conversation" ? "active" : ""}`}
            onClick={openConversation}
            aria-current={activeAuxiliaryPage === "conversation" ? "page" : undefined}
          >
            <ConversationIcon fallback="m" className="conversation-launcher-icon" />
            <span className="conversation-launcher-copy">
              <strong>対話モード</strong>
              <small>変数を選んでモデルを自動比較</small>
            </span>
            <span className="conversation-launcher-arrow" aria-hidden="true">›</span>
          </button>

          <div className="rail-section-label">MODE</div>
          <div className="workbench-mode-switch" role="group" aria-label="実行モード">
            <button
              type="button"
              className={mode === "simple" ? "active" : ""}
              aria-pressed={mode === "simple"}
              onClick={() => setWorkbenchMode("simple")}
            >
              簡易
            </button>
            <button
              type="button"
              className={mode === "advanced" ? "active" : ""}
              aria-pressed={mode === "advanced"}
              onClick={() => setWorkbenchMode("advanced")}
            >
              詳細
            </button>
          </div>

          <div className="rail-section-label">WORKFLOW</div>
          <nav className="tabs" aria-label="ページナビゲーション">
            {visibleSteps.map(([id, label, detail], stepIndex) => {
              const stepStatus = workflowCompletion[id] || {};
              const active = !activeAuxiliaryPage && step === id;
              const statusText = workflowStatusText(stepStatus);
              return (
                <button
                  key={id}
                  className={`tab ${active ? "active" : ""} ${stepStatus.complete ? "complete" : ""} ${stepStatus.available ? "available" : ""} ${stepStatus.optional ? "optional" : ""}`}
                  onClick={() => openStep(id)}
                  aria-current={active ? "page" : undefined}
                  aria-label={`${label} · ${statusText}`}
                  title={`${label}: ${statusText}`}
                  data-workflow-status={stepStatus.complete ? "complete" : stepStatus.optional ? "optional" : stepStatus.available ? "available" : "pending"}
                >
                  <span className="nav-icon">{stepStatus.complete ? "✓" : ICONS[id]}</span>
                  <span className="tab-copy"><strong>{label}</strong><small>{detail}</small></span>
                  <em className={`tab-status ${stepStatus.complete ? "complete" : stepStatus.optional ? "optional" : stepStatus.available ? "available" : "pending"}`}>
                    {stepStatus.complete ? "完了" : stepStatus.optional ? "任意" : stepStatus.available ? "準備" : stepIndex + 1}
                  </em>
                </button>
              );
            })}
          </nav>
          <div className="rail-spacer" />
          <div className="rail-note">
            <div className="shield-icon">✓</div>
            <div><strong>Browser + API</strong><p>ファイル確認はブラウザ、学習と探索はFastAPIで実行します。</p></div>
          </div>
        </aside>

        <section className="content">
          <div className="content-inner">
            {toast?.type === "error" && (
              <div className="inline-alert error" role="alert">
                <div className="inline-alert-icon" aria-hidden="true">!</div>
                <div className="inline-alert-copy">
                  <span className="eyebrow">ERROR</span>
                  <strong>処理を完了できませんでした</strong>
                  <p>{toast.text}</p>
                  <small>入力内容とAPI接続を確認し、もう一度実行してください。</small>
                </div>
                <button
                  className="alert-close icon-button ghost"
                  aria-label="エラー表示を閉じる"
                  title="閉じる"
                  onClick={() => setToast(null)}
                >
                  ×
                </button>
              </div>
            )}
            {!currentPageUsesCache && <Page />}
            {persistentPagesToRender.map((pageId) => {
              const CachedPage = PAGES[pageId];
              const active = !activeAuxiliaryPage
                && pageId === step
                && !(mode === "simple" && pageId === "model");
              const resetKey = pageId === "optimize"
                ? modelInfo?.model_id || "unregistered"
                : rows;
              return (
                <PersistentWorkflowPage
                  key={pageId}
                  active={active}
                  resetKey={resetKey}
                  Page={CachedPage}
                />
              );
            })}
          </div>
        </section>

        <aside className="right-rail">
          <div className={`side-card runtime-card ${health.status}`}>
            <div className="side-card-title"><span>RUNTIME</span><strong>API接続</strong></div>
            <div className="runtime-large">
              <span className={`dot ${health.status}`} />
              <div><strong>FastAPI</strong><small>{apiStatusLabel} · {health.text}</small></div>
            </div>
          </div>
          <div className="side-card">
            <div className="side-card-title"><span>DATA CONTEXT</span><strong>現在のデータ</strong></div>
            <div className="context-list">
              <div><span>Mode</span><strong>{activeAuxiliaryPage ? "対話" : mode === "simple" ? "簡易" : "詳細"}</strong></div>
              <div><span>File</span><strong>{fileName || "—"}</strong></div>
              <div><span>Rows</span><strong>{rows.length || "—"}</strong></div>
              <div><span>Features</span><strong>{features.length || "—"}</strong></div>
              <div><span>Targets</span><strong>{targets.join(", ") || "—"}</strong></div>
            </div>
          </div>
          <div className="side-card">
            <div className="side-card-title"><span>MODEL CONTEXT</span><strong>登録モデル</strong></div>
            <div className="context-list">
              <div><span>Model ID</span><strong>{modelInfo?.model_id || "—"}</strong></div>
              <div><span>Compared</span><strong>{comparison ? "Yes" : "No"}</strong></div>
            </div>
          </div>
        </aside>
      </main>

      <footer className="statusbar">
        <span><span className={`dot ${health.status}`} /> API接続 {apiStatusLabel}</span>
        <span>{activeAuxiliaryPage ? "Conversation mode" : mode === "simple" ? "Simple mode" : "Advanced mode"}</span>
        <span>{rows.length ? `${rows.length} rows` : "No data"}</span>
        <span className="privacy-status">React + FastAPI</span>
      </footer>
      <ComparisonTuningControl />
      <EnsembleModelSettingsControl />
      <ImportedModelDefaultsControl />
      <MaterialFeatureKindControl />
      <MaterialDescriptorSettingsControl />
      <ModelSettingDefaultsControl />
      <ModelBundleControl />
      <ModelResultVisualizationControl />
      <YyDiagnosticsControl />
      <OptimizeCategoryCandidatesControl />
      <ShapScatterControl />
      {toast && toast.type !== "error" && (
        <button className={`message ${toast.type}`} onClick={() => setToast(null)}>{toast.text}</button>
      )}
      {busy && (
        <div className="overlay" role="status" aria-live="polite" aria-busy="true">
          <div className="busy-card">
            <div className="spinner" aria-hidden="true" />
            <span className="eyebrow">PROCESSING</span>
            <h3>{busy}</h3>
            <p>処理中は画面操作を一時停止しています。</p>
          </div>
        </div>
      )}
    </div>
  );
}

export default function App() {
  return <WorkbenchProvider><WorkbenchLayout /></WorkbenchProvider>;
}
