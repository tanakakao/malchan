import React, { useEffect, useMemo, useRef, useState } from "react";
import "../processing-progress.css";

const API_PROGRESS_EVENT = "malchan:api-progress";
const CLOSE_DELAY_MS = 700;
const COMPLETION_NOTICE_MS = 5000;

export function formatProcessingDuration(durationMs) {
  const totalSeconds = Math.max(0, Math.floor(Number(durationMs || 0) / 1000));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return minutes > 0
    ? `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`
    : `${seconds}.${Math.floor((Number(durationMs || 0) % 1000) / 100)}秒`;
}

function snapshotOperation(operation) {
  if (!operation) return null;
  return {
    startedAt: operation.startedAt,
    completed: [...operation.completed],
    active: [...operation.active.values()].sort((left, right) => left.startedAt - right.startedAt),
  };
}

function summarizeOperation(operation) {
  const completedAt = Date.now();
  const stages = [...operation.completed];
  const slowest = [...stages].sort((left, right) => right.durationMs - left.durationMs)[0] || null;
  return {
    startedAt: operation.startedAt,
    completedAt,
    durationMs: Math.max(0, completedAt - operation.startedAt),
    stages,
    slowest,
  };
}

function useProcessingMonitor() {
  const operationRef = useRef(null);
  const ignoredRequestsRef = useRef(new Set());
  const closeTimerRef = useRef(null);
  const completionTimerRef = useRef(null);
  const [operation, setOperation] = useState(null);
  const [lastSummary, setLastSummary] = useState(null);

  useEffect(() => {
    function cancelCloseTimer() {
      if (!closeTimerRef.current) return;
      window.clearTimeout(closeTimerRef.current);
      closeTimerRef.current = null;
    }

    function finishOperation() {
      const current = operationRef.current;
      if (!current || current.active.size) return;

      const summary = summarizeOperation(current);
      operationRef.current = null;
      setOperation(null);
      setLastSummary(summary);
      console.info("[malchan processing summary]", {
        duration_ms: summary.durationMs,
        stages: summary.stages.map((stage) => ({
          label: stage.label,
          duration_ms: stage.durationMs,
          status: stage.status,
        })),
      });

      if (completionTimerRef.current) window.clearTimeout(completionTimerRef.current);
      completionTimerRef.current = window.setTimeout(() => {
        setLastSummary(null);
        completionTimerRef.current = null;
      }, COMPLETION_NOTICE_MS);
    }

    function scheduleFinish() {
      cancelCloseTimer();
      closeTimerRef.current = window.setTimeout(() => {
        closeTimerRef.current = null;
        finishOperation();
      }, CLOSE_DELAY_MS);
    }

    function handleProgress(event) {
      const detail = event.detail || {};
      if (!detail.requestId) return;

      if (detail.phase === "start") {
        if (!operationRef.current && !detail.foreground) {
          ignoredRequestsRef.current.add(detail.requestId);
          return;
        }

        cancelCloseTimer();
        const current = operationRef.current || {
          startedAt: detail.startedAt || Date.now(),
          completed: [],
          active: new Map(),
        };
        current.active.set(detail.requestId, detail);
        operationRef.current = current;
        setOperation(snapshotOperation(current));
        return;
      }

      if (detail.phase !== "complete") return;
      if (ignoredRequestsRef.current.has(detail.requestId)) {
        ignoredRequestsRef.current.delete(detail.requestId);
        return;
      }

      const current = operationRef.current;
      if (!current || !current.active.has(detail.requestId)) return;

      const activeStage = current.active.get(detail.requestId);
      current.active.delete(detail.requestId);
      current.completed.push({
        ...activeStage,
        durationMs: detail.durationMs,
        status: detail.status,
        completedAt: detail.completedAt,
      });
      setOperation(snapshotOperation(current));
      if (!current.active.size) scheduleFinish();
    }

    window.addEventListener(API_PROGRESS_EVENT, handleProgress);
    return () => {
      window.removeEventListener(API_PROGRESS_EVENT, handleProgress);
      cancelCloseTimer();
      if (completionTimerRef.current) window.clearTimeout(completionTimerRef.current);
    };
  }, []);

  return { operation, lastSummary };
}

function useNow(active) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    if (!active) return undefined;
    setNow(Date.now());
    const timer = window.setInterval(() => setNow(Date.now()), 250);
    return () => window.clearInterval(timer);
  }, [active]);
  return now;
}

function StageIcon({ status, active }) {
  if (active) return <span className="api-stage-icon active" aria-hidden="true" />;
  return (
    <span className={`api-stage-icon ${status === "error" ? "error" : "complete"}`} aria-hidden="true">
      {status === "error" ? "!" : "✓"}
    </span>
  );
}

export default function ApiProcessingOverlay() {
  const { operation, lastSummary } = useProcessingMonitor();
  const now = useNow(Boolean(operation));
  const currentStage = operation?.active?.at(-1) || null;
  const recentCompleted = useMemo(
    () => operation?.completed?.slice(-5) || [],
    [operation],
  );

  return (
    <>
      {operation && (
        <div className="api-processing-overlay" role="status" aria-live="polite" aria-busy="true">
          <div className="api-processing-card">
            <div className="api-processing-header">
              <div className="api-processing-spinner" aria-hidden="true" />
              <div>
                <span className="eyebrow">PROCESSING</span>
                <h3>{currentStage?.label || "次の処理を準備しています"}</h3>
                {currentStage?.detail && <p>{currentStage.detail}</p>}
              </div>
              <div className="api-processing-elapsed">
                <span>経過時間</span>
                <strong>{formatProcessingDuration(now - operation.startedAt)}</strong>
              </div>
            </div>

            <div className="api-processing-stage-list" aria-label="処理段階">
              {recentCompleted.map((stage) => (
                <div className="api-processing-stage completed" key={`done-${stage.requestId}`}>
                  <StageIcon status={stage.status} />
                  <div>
                    <strong>{stage.label}</strong>
                    {stage.detail && <small>{stage.detail}</small>}
                  </div>
                  <time>{formatProcessingDuration(stage.durationMs)}</time>
                </div>
              ))}
              {currentStage && (
                <div className="api-processing-stage current">
                  <StageIcon active />
                  <div>
                    <strong>{currentStage.label}</strong>
                    {currentStage.detail && <small>{currentStage.detail}</small>}
                  </div>
                  <time>{formatProcessingDuration(now - currentStage.startedAt)}</time>
                </div>
              )}
              {!currentStage && (
                <div className="api-processing-stage current preparing">
                  <StageIcon active />
                  <div><strong>次のAPI処理を準備しています</strong></div>
                </div>
              )}
            </div>

            <p className="api-processing-note">
              完了率を推測せず、実際に開始・完了した処理段階と所要時間を表示しています。
            </p>
          </div>
        </div>
      )}

      {!operation && lastSummary && (
        <div className="api-processing-complete" role="status">
          <span className="api-stage-icon complete" aria-hidden="true">✓</span>
          <div>
            <strong>処理完了 · {formatProcessingDuration(lastSummary.durationMs)}</strong>
            {lastSummary.slowest && (
              <small>
                最長: {lastSummary.slowest.label} {formatProcessingDuration(lastSummary.slowest.durationMs)}
              </small>
            )}
          </div>
        </div>
      )}
    </>
  );
}
