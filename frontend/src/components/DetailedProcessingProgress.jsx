import React, { useEffect, useMemo, useRef, useState } from "react";
import { DETAILED_PROGRESS_EVENT } from "../detailed-progress";
import "../detailed-progress.css";

const DIMENSION_ORDER = ["target", "trial", "fold"];
const COMPLETED_HOLD_MS = 1400;

function dimensionTitle(kind, dimension) {
  if (kind === "target") return `目的変数 ${dimension.current} / ${dimension.total}`;
  if (kind === "trial") return `Optuna ${dimension.current} / ${dimension.total} trials`;
  if (kind === "fold") return `CV fold ${dimension.current} / ${dimension.total}`;
  return `${kind} ${dimension.current} / ${dimension.total}`;
}

function progressPercent(dimension) {
  if (!dimension?.total) return 0;
  return Math.max(0, Math.min(100, (Number(dimension.current) / Number(dimension.total)) * 100));
}

export default function DetailedProcessingProgress() {
  const [state, setState] = useState(null);
  const clearTimerRef = useRef(null);

  useEffect(() => {
    function clearTimer() {
      if (!clearTimerRef.current) return;
      window.clearTimeout(clearTimerRef.current);
      clearTimerRef.current = null;
    }

    function handleProgress(event) {
      const detail = event.detail || {};
      const progress = detail.progress;
      if (!progress) return;

      clearTimer();
      setState({ progressId: detail.progressId, progress });
      if (progress.status !== "running") {
        clearTimerRef.current = window.setTimeout(() => {
          setState((current) => (
            current?.progressId === detail.progressId ? null : current
          ));
          clearTimerRef.current = null;
        }, COMPLETED_HOLD_MS);
      }
    }

    window.addEventListener(DETAILED_PROGRESS_EVENT, handleProgress);
    return () => {
      window.removeEventListener(DETAILED_PROGRESS_EVENT, handleProgress);
      clearTimer();
    };
  }, []);

  const dimensions = useMemo(() => {
    const values = state?.progress?.dimensions || {};
    return DIMENSION_ORDER
      .filter((kind) => values[kind]?.total > 0)
      .map((kind) => [kind, values[kind]]);
  }, [state]);

  if (!state || !dimensions.length) return null;

  const status = state.progress.status;
  const completed = status === "success";
  const failed = status === "error";
  const stateClass = failed ? "failed" : completed ? "completed" : "running";
  const heading = failed
    ? "内部処理でエラーが発生しました"
    : completed
      ? "内部処理が完了しました"
      : "内部処理の実進捗";
  const badge = failed ? "!" : completed ? "✓" : "LIVE";
  const badgeClass = failed ? "error" : completed ? "complete" : "active";

  return (
    <aside
      className={`detailed-progress-card ${stateClass}`}
      role="status"
      aria-live="polite"
      aria-label="詳細な処理進捗"
    >
      <div className="detailed-progress-heading">
        <div>
          <span className="eyebrow">LIVE PROGRESS</span>
          <strong>{heading}</strong>
        </div>
        <span className={`detailed-progress-state ${badgeClass}`}>
          {badge}
        </span>
      </div>

      <div className="detailed-progress-dimensions">
        {dimensions.map(([kind, dimension]) => {
          const percent = progressPercent(dimension);
          return (
            <div className="detailed-progress-dimension" key={kind}>
              <div className="detailed-progress-label">
                <strong>{dimensionTitle(kind, dimension)}</strong>
                {dimension.label && dimension.label !== "Optuna" && dimension.label !== "CV fold" && (
                  <span title={dimension.label}>{dimension.label}</span>
                )}
              </div>
              <div
                className="detailed-progress-track"
                role="progressbar"
                aria-valuemin="0"
                aria-valuemax={dimension.total}
                aria-valuenow={dimension.current}
                aria-label={dimensionTitle(kind, dimension)}
              >
                <span style={{ width: `${percent}%` }} />
              </div>
            </div>
          );
        })}
      </div>

      <p>バックエンドで完了したtarget / trial / foldをそのまま表示しています。</p>
    </aside>
  );
}
