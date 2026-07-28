const API_BASE = (import.meta.env.VITE_API_BASE || "/api").replace(/\/$/, "");

let comparisonTuneBestOverride = false;
let inverseAnalysisPayloadOverride = null;
let inverseCategoryCandidatesOverride = null;
let ensembleTrainingOptions = null;

export function setComparisonTuneBestOverride(enabled) {
  comparisonTuneBestOverride = Boolean(enabled);
}

export function setInverseAnalysisPayloadOverride(payload) {
  inverseAnalysisPayloadOverride = payload && typeof payload === "object"
    ? payload
    : null;
}

export function setInverseCategoryCandidatesOverride(categories) {
  inverseCategoryCandidatesOverride = categories && typeof categories === "object"
    ? categories
    : null;
}

export function setEnsembleTrainingOptions(options) {
  ensembleTrainingOptions = options && typeof options === "object"
    ? options
    : null;
}

export class ApiError extends Error {
  constructor(message, status, detail) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });

  if (response.status === 204) {
    return null;
  }

  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json")
    ? await response.json()
    : await response.text();

  if (!response.ok) {
    const detail = payload?.detail ?? payload;
    throw new ApiError(
      typeof detail === "string" ? detail : JSON.stringify(detail),
      response.status,
      detail,
    );
  }
  return payload;
}

function query(params) {
  const values = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") {
      values.set(key, String(value));
    }
  });
  const encoded = values.toString();
  return encoded ? `?${encoded}` : "";
}

function visualizationPath(modelId, target, chart, options = {}) {
  return `/models/${encodeURIComponent(modelId)}/visualizations/${encodeURIComponent(target)}/${chart}${query(options)}`;
}

function comparisonPayload(payload) {
  if (!payload?.activate_best || !comparisonTuneBestOverride) return payload;
  return { ...payload, tune_best: true };
}

function ensembleTrainingPayload(payload) {
  if (!ensembleTrainingOptions) return payload;

  const {
    ensembleType,
    baseModel,
    membersByTarget = {},
    parameterMode: configuredParameterMode,
    tuning: legacyTuning,
    memberParamsByTarget = {},
    baseParamsByTarget = {},
  } = ensembleTrainingOptions;
  const parameterMode = configuredParameterMode
    || (legacyTuning === false ? "default" : "tuning");
  const manualParameters = parameterMode === "manual";
  const targetColumns = payload?.target_cols?.length
    ? payload.target_cols
    : [payload?.target_col].filter(Boolean);
  const requiresMultiple = ensembleType === "アンサンブル" || ensembleType === "スタッキング";
  const usesBaseParameters = ["スタッキング", "バギング", "ブースティング"].includes(ensembleType);

  if (!ensembleType) {
    throw new Error("アンサンブル方式を選択してください。");
  }
  if (ensembleType === "スタッキング") {
    const taskValues = payload?.tasks?.length ? payload.tasks : [payload?.task].filter(Boolean);
    if (new Set(taskValues).size > 1) {
      throw new Error("回帰と分類が混在する多目的モデルではStackingを使用できません。");
    }
    if (!baseModel) {
      throw new Error("Stackingの最終モデルを選択してください。");
    }
  }

  const normalizedMembers = Object.fromEntries(
    targetColumns.map((target) => {
      const selected = [...new Set((membersByTarget[target] || []).filter(Boolean))];
      const members = requiresMultiple ? selected : selected.slice(0, 1);
      if (members.length < (requiresMultiple ? 2 : 1)) {
        throw new Error(
          requiresMultiple
            ? `${target}の構成モデルを2件以上選択してください。`
            : `${target}のベースモデルを選択してください。`,
        );
      }
      return [target, members];
    }),
  );

  const normalizedModelParams = Object.fromEntries(
    targetColumns.map((target) => [
      target,
      normalizedMembers[target].map((model) => ({
        ...(memberParamsByTarget[target]?.[model] || {}),
      })),
    ]),
  );
  const normalizedBaseParams = Object.fromEntries(
    targetColumns.map((target) => {
      if (ensembleType === "スタッキング") {
        return [target, { ...(baseParamsByTarget[target] || {}) }];
      }
      const baseMember = normalizedMembers[target][0];
      return [target, { ...(memberParamsByTarget[target]?.[baseMember] || {}) }];
    }),
  );

  const common = {
    ensemble: true,
    ens_type: ensembleType,
    base_model: ensembleType === "スタッキング" ? baseModel : null,
    tuning: parameterMode === "tuning",
  };

  if (payload?.target_cols?.length) {
    const merged = {
      ...payload,
      ...common,
      model_names_by_target: normalizedMembers,
      model_params_by_target: manualParameters ? normalizedModelParams : {},
      base_model_params_by_target: manualParameters && usesBaseParameters
        ? normalizedBaseParams
        : {},
    };
    delete merged.model_names;
    delete merged.model_params;
    delete merged.base_model_param;
    return merged;
  }

  const target = targetColumns[0];
  const singleBaseModel = ensembleType === "スタッキング"
    ? baseModel
    : usesBaseParameters
      ? normalizedMembers[target]?.[0] || null
      : null;
  const merged = {
    ...payload,
    ...common,
    base_model: singleBaseModel,
    model_names: normalizedMembers[target] || [],
    model_params: manualParameters ? normalizedModelParams[target] : null,
    base_model_param: manualParameters && usesBaseParameters
      ? normalizedBaseParams[target]
      : null,
  };
  delete merged.model_names_by_target;
  delete merged.model_params_by_target;
  delete merged.base_model_params_by_target;
  return merged;
}

function inverseAnalysisPayload(payload) {
  const mergedPayload = inverseAnalysisPayloadOverride
    ? {
        ...payload,
        ...inverseAnalysisPayloadOverride,
      }
    : payload;

  if (!inverseCategoryCandidatesOverride) return mergedPayload;

  const existingCategories = mergedPayload?.categories || {};
  const selectableOverrides = Object.fromEntries(
    Object.entries(inverseCategoryCandidatesOverride)
      .filter(([column]) => Object.prototype.hasOwnProperty.call(existingCategories, column)),
  );

  return {
    ...mergedPayload,
    categories: {
      ...existingCategories,
      ...selectableOverrides,
    },
  };
}

export const api = {
  health: () => request("/health"),
  modelParameters: (task, modelName) =>
    request(`/model-parameters${query({ task, model_name: modelName })}`),
  train: (payload) => request("/models", {
    method: "POST",
    body: JSON.stringify(ensembleTrainingPayload(payload)),
  }),
  listModels: () => request("/models"),
  modelInfo: (modelId) => request(`/models/${encodeURIComponent(modelId)}`),
  predict: (modelId, payload) =>
    request(`/models/${encodeURIComponent(modelId)}/predict`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  evaluate: (modelId, payload) =>
    request(`/models/${encodeURIComponent(modelId)}/evaluate`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  compare: (modelId, payload) =>
    request(`/models/${encodeURIComponent(modelId)}/compare`, {
      method: "POST",
      body: JSON.stringify(comparisonPayload(payload)),
    }),
  comparison: (modelId) =>
    request(`/models/${encodeURIComponent(modelId)}/comparison`),
  tuneBest: (modelId, payload) =>
    request(`/models/${encodeURIComponent(modelId)}/comparison/tune-best`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  inverse: (modelId, payload) =>
    request(`/models/${encodeURIComponent(modelId)}/inverse-analysis`, {
      method: "POST",
      body: JSON.stringify(inverseAnalysisPayload(payload)),
    }),
  xaiSummary: (modelId) =>
    request(`/models/${encodeURIComponent(modelId)}/xai`),
  recomputeXai: (modelId, payload = {}) =>
    request(`/models/${encodeURIComponent(modelId)}/xai/recompute`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  localShap: (modelId, payload) =>
    request(`/models/${encodeURIComponent(modelId)}/xai/local`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  xaiImportance: (modelId, target, options = {}) =>
    request(
      `/models/${encodeURIComponent(modelId)}/xai/${encodeURIComponent(target)}/importance${query(options)}`,
    ),
  xaiShapValues: (modelId, target) =>
    request(
      `/models/${encodeURIComponent(modelId)}/xai/${encodeURIComponent(target)}/shap-values`,
    ),
  xaiShap: (modelId, target, feature) =>
    request(
      `/models/${encodeURIComponent(modelId)}/xai/${encodeURIComponent(target)}/shap${query({ feature })}`,
    ),
  xaiPdp: (modelId, target, feature, options = {}) =>
    request(
      `/models/${encodeURIComponent(modelId)}/xai/${encodeURIComponent(target)}/pdp${query({ feature, ...options })}`,
    ),
  visualizationYy: (modelId, target, options = {}) =>
    request(visualizationPath(modelId, target, "yy", options)),
  visualizationImportance: (modelId, target, options = {}) =>
    request(visualizationPath(modelId, target, "importance", options)),
  visualizationBeeswarm: (modelId, target, options = {}) =>
    request(visualizationPath(modelId, target, "shap-beeswarm", options)),
  visualizationShapScatter: (modelId, target, options = {}) =>
    request(visualizationPath(modelId, target, "shap-scatter", options)),
  visualizationPdp: (modelId, target, options = {}) =>
    request(visualizationPath(modelId, target, "pdp", options)),
  visualizationPdp2d: (modelId, target, options = {}) =>
    request(visualizationPath(modelId, target, "pdp-2d", options)),
  deleteModel: (modelId) =>
    request(`/models/${encodeURIComponent(modelId)}`, { method: "DELETE" }),
};
