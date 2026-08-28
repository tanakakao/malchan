function hasText(value) {
  return typeof value === "string" && value.trim().length > 0;
}

function status(complete, available, options = {}) {
  return {
    complete: Boolean(complete),
    available: Boolean(available || complete),
    optional: Boolean(options.optional),
    label: options.label || "",
  };
}

function hasPredictionWorkspaceResult(workspace) {
  return Boolean(
    workspace?.customPrediction
    || (Array.isArray(workspace?.filePredictions) && workspace.filePredictions.length > 0),
  );
}

/**
 * Derive workflow completion from actual workspace artifacts rather than page order.
 *
 * A page only receives a completion check when the state it is responsible for
 * actually exists. Optional pages remain optional until they produce an artifact.
 */
export function getWorkflowCompletion({
  rows = [],
  columns = [],
  ready = false,
  modelInfo = null,
  comparison = null,
  diagnostics = [],
  prediction = null,
  predictionWorkspace = null,
  inverseResult = null,
  inverseCurrent,
  report = "",
  reportCurrent,
} = {}) {
  const hasRawData = rows.length > 0;
  const hasImportedModelContext = Boolean(modelInfo?.model_id) && columns.length > 0;
  const hasDataContext = hasRawData || hasImportedModelContext;
  const hasModel = Boolean(modelInfo?.model_id);
  const hasExplainResult = Boolean(
    comparison
    || diagnostics.length > 0
    || modelInfo?.xai_status === "ready",
  );
  const hasPredictionResult = Boolean(
    hasModel
    && (prediction || hasPredictionWorkspaceResult(predictionWorkspace)),
  );
  const inverseMatchesModelId = Boolean(
    inverseResult
    && hasModel
    && (!inverseResult.model_id || inverseResult.model_id === modelInfo.model_id),
  );
  const hasCurrentInverseResult = inverseCurrent === undefined
    ? inverseMatchesModelId
    : Boolean(inverseResult && hasModel && inverseCurrent);
  const hasCurrentReport = Boolean(
    hasText(report)
    && hasModel
    && (reportCurrent === undefined || reportCurrent),
  );

  return {
    data: status(hasDataContext, true, { label: hasRawData ? "読込済み" : "モデル読込済み" }),
    explore: status(false, hasRawData, { optional: true, label: "任意" }),
    prepare: status(Boolean(ready), hasRawData, { label: "設定済み" }),
    model: status(hasModel, Boolean(ready), { label: "学習済み" }),
    explain: status(hasExplainResult, hasModel, { optional: true, label: "解析済み" }),
    predict: status(hasPredictionResult, hasModel, { optional: true, label: "予測済み" }),
    optimize: status(hasCurrentInverseResult, hasModel, { optional: true, label: "探索済み" }),
    report: status(hasCurrentReport, hasModel, { optional: true, label: "作成済み" }),
  };
}

export function workflowStatusText(stepStatus) {
  if (stepStatus?.complete) return stepStatus.label || "完了";
  if (stepStatus?.optional) return "任意";
  if (stepStatus?.available) return "実行可能";
  return "未完了";
}
