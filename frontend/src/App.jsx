import React from "react";
import { WorkbenchProvider, useWorkbench } from "./context/WorkbenchContext";
import ComparisonTuningControl from "./components/ComparisonTuningControl";
import EnsembleModelSettingsControl from "./components/EnsembleModelSettingsControl";
import MaterialDescriptorSettingsControl from "./components/MaterialDescriptorSettingsControl";
import MaterialFeatureKindControl from "./components/MaterialFeatureKindControl";
import ModelBundleControl from "./components/ModelBundleControl";
import ModelResultVisualizationControl from "./components/ModelResultVisualizationControl";
import ModelSettingDefaultsControl from "./components/ModelSettingDefaultsControl";
import YyDiagnosticsControl from "./components/YyDiagnosticsControl";
import OptimizeCategoryCandidatesControl from "./components/OptimizeCategoryCandidatesControl";
import ShapScatterControl from "./components/ShapScatterControl";
import DataPage from "./pages/DataPage";
import ExplorePage from "./pages/ExplorePage";
import PreparePage from "./pages/PreparePage";
import ModelPage from "./pages/ModelPage";
import ExplainPage from "./pages/ExplainPage";
import PredictionPage from "./pages/PredictionPage";
import OptimizePage from "./pages/OptimizePage";
import ReportPage from "./pages/ReportPage";

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
const ICONS = ["▦", "◫", "◇", "⌁", "◎", "◉", "↗", "▧"];
const API_STATUS_LABELS = {
  loading: "確認中",
  ready: "利用可能",
  error: "エラー",
};
const PORTAL_URL = import.meta.env.VITE_PORTAL_URL?.trim() || "http://127.0.0.1:5172";

function WorkbenchLayout() {
  const {
    theme, setTheme, step, setStep, health, busy, toast, setToast,
    fileName, rows, features, targets, modelInfo, comparison,
  } = useWorkbench();
  const index = APP_STEPS.findIndex(([id]) => id === step);
  const Page = PAGES[step] || DataPage;
  const apiStatusLabel = API_STATUS_LABELS[health.status] || "利用不可";

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
          {APP_STEPS.map(([id, label], stepIndex) => (
            <React.Fragment key={id}>
              <button
                className={`workflow-step ${id === step ? "active" : ""} ${stepIndex < index ? "complete" : ""}`}
                onClick={() => setStep(id)}
                aria-current={id === step ? "step" : undefined}
              >
                <span>{stepIndex + 1}</span><strong>{label}</strong>
              </button>
              {stepIndex < APP_STEPS.length - 1 && <i />}
            </React.Fragment>
          ))}
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
          <div className="rail-section-label">WORKFLOW</div>
          <nav className="tabs" aria-label="ページナビゲーション">
            {APP_STEPS.map(([id, label, detail], stepIndex) => (
              <button
                key={id}
                className={`tab ${step === id ? "active" : ""} ${stepIndex < index ? "complete" : ""}`}
                onClick={() => setStep(id)}
                aria-current={step === id ? "page" : undefined}
              >
                <span className="nav-icon">{ICONS[stepIndex]}</span>
                <span><strong>{label}</strong><small>{detail}</small></span>
                <em>{stepIndex + 1}</em>
              </button>
            ))}
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
            <Page />
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
        <span>{rows.length ? `${rows.length} rows` : "No data"}</span>
        <span className="privacy-status">React + FastAPI</span>
      </footer>
      <ComparisonTuningControl />
      <EnsembleModelSettingsControl />
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
