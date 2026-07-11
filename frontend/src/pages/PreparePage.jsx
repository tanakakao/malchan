import React from "react";
import { CheckboxList, SectionHeader } from "../components/Common";
import { useWorkbench } from "../context/WorkbenchContext";

export default function PreparePage() {
  const {
    columns, numeric, categorical, targets, tasks, numFeatures, setNumFeatures,
    catFeatures, setCatFeatures, changeTargets, changeTask,
  } = useWorkbench();
  return (
    <>
      <SectionHeader
        step="3 · PREPARE"
        title="分析用Pipelineを設計する"
        text="目的変数、タスク、説明変数を設定します。"
      />
      <div className="split">
        <article className="panel">
          <h3>目的変数 Y</h3>
          <CheckboxList values={columns} selected={targets} onChange={changeTargets} />
          <div className="target-config-list">
            {targets.map((target) => (
              <div className="target-config" key={target}>
                <strong>{target}</strong>
                <select value={tasks[target]} onChange={(event) => changeTask(target, event.target.value)}>
                  <option value="regression">回帰</option>
                  <option value="classification">分類</option>
                </select>
              </div>
            ))}
          </div>
        </article>
        <article className="panel">
          <h3>説明変数 X</h3>
          <h4>数値列</h4>
          <CheckboxList
            values={numeric}
            selected={numFeatures}
            disabled={targets}
            onChange={setNumFeatures}
          />
          <h4 className="subsection-title">カテゴリ列</h4>
          <CheckboxList
            values={categorical}
            selected={catFeatures}
            disabled={targets}
            onChange={setCatFeatures}
          />
        </article>
      </div>
    </>
  );
}
