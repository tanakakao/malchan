import React from "react";
import { STEPS, WorkbenchProvider, useWorkbench } from "./context/WorkbenchContext";
import DataPage from "./pages/DataPage";
import ExplorePage from "./pages/ExplorePage";
import PreparePage from "./pages/PreparePage";
import ModelPage from "./pages/ModelPage";
import ExplainPage from "./pages/ExplainPage";
import OptimizePage from "./pages/OptimizePage";
import ReportPage from "./pages/ReportPage";

const PAGES = {
  data: DataPage,
  explore: ExplorePage,
  prepare: PreparePage,
  model: ModelPage,
  explain: ExplainPage,
  optimize: OptimizePage,
  report: ReportPage,
};
const ICONS = ["▦", "◫", "◇", "⌁", "◎", "↗", "▧"];

function WorkbenchLayout() {
  const {
    theme, setTheme, step, setStep, health, busy, toast, setToast,
    fileName, rows, features, targets, modelInfo, comparison,
  } = useWorkbench();
  const index = STEPS.findIndex(([id]) => id === step);
  const Page = PAGES[step];

  return (
    <div className="app-root">
      <header className="app-header">
        <div className="brand-lockup">
          <div className="brand-mark"><span>m</span></div>
          <div className="brand-wordmark">
            <h1>キカイガクシュウ</h1>
            <p>MALCHAN MACHINE LEARNING WORKBENCH</p>
          </div>
        </div>
        <div className="workflow-strip">
          {STEPS.map(([id, label], stepIndex) => (
            <React.Fragment key={id}>
              <button
                className={`workflow-step ${id === step ? "active" : ""} ${stepIndex < index ? "complete" : ""}`}
                onClick={() => setStep(id)}
              >
                <span>{stepIndex + 1}</span><strong>{label}</strong>
              </button>
              {stepIndex < STEPS.length - 1 && <i />}
            </React.Fragment>
          ))}
        </div>
        <div className="header-actions">
          <div className="runtime-pill"><span className={`dot ${health.status}`} /><span>{health.text}</span></div>
          <button className="icon-button" onClick={() => setTheme(theme === "dark" ? "light" : "dark")}>◐</button>
        </div>
      </header>

      <main className="app-shell">
        <aside className="left-rail">
          <div className="rail-section-label">WORKSPACE</div>
          <nav className="tabs">
            {STEPS.map(([id, label, detail], stepIndex) => (
              <button
                key={id}
                className={`tab ${step === id ? "active" : ""} ${stepIndex < index ? "complete" : ""}`}
                onClick={() => setStep(id)}
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

        <section className="content"><Page /></section>

        <aside className="right-rail">
          <div className={`side-card runtime-card ${health.status}`}>
            <div className="runtime-large"><span className={`dot ${health.status}`} /><div><strong>FastAPI</strong><small>{health.text}</small></div></div>
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
        <span><span className={`dot ${health.status}`} /> API {health.status}</span>
        <span>{rows.length ? `${rows.length} rows` : "No data"}</span>
        <span className="privacy-status">React + FastAPI</span>
      </footer>
      {toast && <button className={`message ${toast.type}`} onClick={() => setToast(null)}>{toast.text}</button>}
      {busy && (
        <div className="overlay">
          <div className="busy-card"><div className="spinner" /><h3>{busy}</h3><div className="busy-progress"><span /></div></div>
        </div>
      )}
    </div>
  );
}

export default function App() {
  return <WorkbenchProvider><WorkbenchLayout /></WorkbenchProvider>;
}
