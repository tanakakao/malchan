import React from "react";
import ComparisonTable from "../components/ComparisonTable";
import { CheckboxList, Field, SectionHeader } from "../components/Common";
import { modelsFor, useWorkbench } from "../context/WorkbenchContext";

export default function ModelPage() {
  const {
    targets, tasks, features, modelNames, setModelNames, candidates, setCandidates,
    modelInfo, comparison, cvSplits, setCvSplits, tuneBest, setTuneBest,
    activateBest, setActivateBest, trials, setTrials, ready, busy,
    trainModel, compareModels, tuneBestLater,
  } = useWorkbench();
  return (
    <>
      <SectionHeader
        step="4 · MODEL"
        title="モデルを学習・検証する"
        text="学習、候補比較、ベストモデルのチューニングと有効化を行います。"
      />
      <article className="panel">
        <div className="target-model-grid">
          {targets.map((target) => (
            <Field key={target} label={target}>
              <select
                value={modelNames[target] || ""}
                onChange={(event) => setModelNames({ ...modelNames, [target]: event.target.value })}
              >
                {modelsFor(tasks[target]).map((model) => <option key={model}>{model}</option>)}
              </select>
            </Field>
          ))}
        </div>
        <div className="train-launcher">
          <div><strong>FastAPIでモデルを学習</strong><span>{features.length} features / {targets.length} targets</span></div>
          <button disabled={!ready || busy} onClick={trainModel}>モデル学習を実行 →</button>
        </div>
        {modelInfo && <pre className="codebox">{JSON.stringify(modelInfo, null, 2)}</pre>}
      </article>
      <article className="panel best-model-panel">
        <h3>最良モデル選定</h3>
        {targets.map((target) => (
          <details className="advanced" key={target}>
            <summary><strong>{target}</strong><span>{candidates[target]?.length || 0} models</span></summary>
            <CheckboxList
              values={modelsFor(tasks[target])}
              selected={candidates[target] || []}
              onChange={(values) => setCandidates({ ...candidates, [target]: values })}
            />
          </details>
        ))}
        <div className="best-model-actions">
          <Field label="CV分割"><input type="number" min="2" value={cvSplits} onChange={(event) => setCvSplits(event.target.value)} /></Field>
          <Field label="Optuna試行数"><input type="number" min="1" value={trials} onChange={(event) => setTrials(event.target.value)} /></Field>
          <label className="switch-label"><input type="checkbox" checked={tuneBest} onChange={(event) => setTuneBest(event.target.checked)} /><span />ベストだけチューニング</label>
          <label className="switch-label"><input type="checkbox" checked={activateBest} onChange={(event) => setActivateBest(event.target.checked)} /><span />ベストを有効化</label>
          <button disabled={!modelInfo || busy} onClick={compareModels}>候補を比較</button>
          <button className="secondary" disabled={!comparison || busy} onClick={tuneBestLater}>後からチューニング</button>
        </div>
      </article>
      <ComparisonTable comparison={comparison} />
    </>
  );
}
