import { applyMaterialFeatureTrainingPayload } from "./materialFeatures";

const API_BASE = (import.meta.env.VITE_API_BASE || "/api").replace(/\/$/, "");
const API_PROGRESS_EVENT = "malchan:api-progress";

let comparisonTuneBestOverride = false;
let inverseAnalysisPayloadOverride = null;
let inverseCategoryCandidatesOverride = null;
let ensembleTrainingOptions = null;
let apiRequestSequence = 0;

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

function requestBody(options = {}) {
  if (typeof options.body !== "string") return null;
  try {
    return JSON.parse(options.body);
  } catch {
    return null;
  }
}

function candidateCount(modelNames) {
  if (Array.isArray(modelNames)) return modelNames.length;
  if (!modelNames || typeof modelNames !== "object") return 0;
  return Object.values(modelNames).reduce(
    (sum, values) => sum + (Array.isArray(values) ? values.length : 0),
    0,
  );
}

function requestProgressDescriptor(path, options = {}) {
  if (path === "/health") return null;

  const method = String(options.method || "GET").toUpperCase();
  const body = requestBody(options);

  if (path === "/models" && method === "POST") {
    const targetCount = body?.target_cols?.length || (body?.target_col ? 1 : 0);
    const featureCount = (body?.num_cols?.length || 0) + (body?.cat_cols?.length || 0);
    return {
      label: body?.tuning ? "モデル学習・パラメータ探索" : "モデル学習",
      detail: [
        targetCount ? `目的変数 ${targetCount}件` : null,
        featureCount ? `説明変数 ${featureCount}件` : null,
      ].filter(Boolean).join(" · "),
      foreground: true,
    };
  }

  if (/\/compare$/.test(path) && method === "POST") {
    const count = candidateCount(body?.model_names);
    const methodLabel = body?.method === "loo"
      ? "LOO"
      : body?.n_splits ? `${body.n_splits}-fold CV` : "交差検証";
    return {
      label: "候補モデルを交差検証・比較",
      detail: [methodLabel, count ? `候補 ${count}件` : null, body?.tune_best ? "チューニングあり" : null]
        .filter(Boolean)
        .join(" · "),
      foreground: true,
    };
  }

  if (/\/evaluate$/.test(path) && method === "POST") {
    return {
      label: "学習済みモデルを精度検証",
      detail: body?.method === "loo" ? "Leave-One-Out" : `${body?.n_splits || "?"}-fold CV`,
      foreground: true,
    };
  }

  if (/\/comparison\/tune-best$/.test(path) && method === "POST") {
    const trialValues = typeof body?.n_trials === "object"
      ? Object.values(body.n_trials)
      : [body?.n_trials];
    const maxTrials = Math.max(0, ...trialValues.map((value) => Number(value) || 0));
    return {
      label: "ベストモデルをチューニング",
      detail: maxTrials ? `最大 ${maxTrials} trials` : "Optuna探索",
      foreground: true,
    };
  }

  if (/\/inverse-analysis$/.test(path) && method === "POST") {
    return {
      label: "逆解析・候補探索",
      detail: [
        body?.trials ? `${body.trials} trials` : null,
        body?.n_candidates ? `上位 ${body.n_candidates}件` : null,
      ].filter(Boolean).join(" · "),
      foreground: true,
    };
  }

  if (/\/predict$/.test(path) && method === "POST") {
    return {
      label: "予測を計算",
      detail: body?.data?.length ? `${body.data.length} records` : "",
      foreground: true,
    };
  }

  if (/\/xai\/local$/.test(path) && method === "POST") {
    return {
      label: "ローカルSHAPを計算",
      detail: body?.data?.length ? `${body.data.length} records` : "",
      foreground: true,
    };
  }

  if (/\/xai\/recompute$/.test(path) && method === "POST") {
    return { label: "XAIを再計算", detail: "", foreground: true };
  }

  if (path === "/models" && method === "GET") {
    return { label: "登録モデルを確認", detail: "", foreground: false };
  }

  if (/^\/models\/[^/]+$/.test(path) && method === "GET") {
    return { label: "モデル情報を反映", detail: "", foreground: false };
  }

  if (path.startsWith("/model-parameters")) {
    return { label: "モデル設定を確認", detail: "", foreground: false };
  }

  return {
    label: method === "GET" ? "APIデータを取得" : "API処理を実行",
    detail: "",
    foreground: false,
  };
}

function dispatchApiProgress(detail) {
  window.dispatchEvent(new CustomEvent(API_PROGRESS_EVENT, { detail }));
}

async function request(path, options = {}) {
  const descriptor = requestProgressDescriptor(path, options);
  apiRequestSequence += 1;
  const requestId = apiRequestSequence;
  const startedAt = Date.now();

  if (descriptor) {
    dispatchApiProgress({
      phase: "start",
      requestId,
      startedAt,
      method: String(options.method || "GET").toUpperCase(),
      ...descriptor,
    });
  }

  let requestStatus = "success";
  try {
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
  } catch (error) {
    requestStatus = "error";
    throw error;
  } finally {
    if (descriptor) {
      const durationMs = Math.max(0, Date.now() - startedAt);
      const timing = {
        phase: "complete",
        requestId,
        startedAt,
        completedAt: Date.now(),
        durationMs,
        status: requestStatus,
        method: String(options.method || "GET").toUpperCase(),
        ...descriptor,
      };
      dispatchApiProgress(timing);
      console.info("[malchan api timing]", {
        label: descriptor.label,
        duration_ms: durationMs,
        status: requestStatus,
      });
    }
  }
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

async function evaluateAndNotify(modelId, payload) {
  const result = await request(`/models/${encodeURIComponent(modelId)}/evaluate`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
  window.dispatchEvent(new CustomEvent("malchan:model-evaluated", { detail: result }));
  return result;
}

function unavailableShapFeature(error) {
  const detail = typeof error?.detail === "string" ? error.detail : error?.message;
  return /Unknown or unavailable SHAP feature/i.test(String(detail || ""));
}

async function requestOptionalShap(modelId, target, feature) {
  try {
    return await request(
      `/models/${encodeURIComponent(modelId)}/xai/${encodeURIComponent(target)}/shap${query({ feature })}`,
    );
  } catch (error) {
    if (!unavailableShapFeature(error)) throw error;
    return {
      feature,
      records: [],
      value_columns: [],
      unavailable: true,
      unavailable_reason: error.message || String(error),
    };
  }
}

export const api = {
  health: () => request("/health"),
  modelParameters: (task, modelName) =>
    request(`/model-parameters${query({ task, model_name: modelName })}`),
  train: (payload) => request("/models", {
    method: "POST",
    body: JSON.stringify(
      ensembleTrainingPayload(applyMaterialFeatureTrainingPayload(payload)),
    ),
  }),
  listModels: () => request("/models"),
  modelInfo: (modelId) => request(`/models/${encodeURIComponent(modelId)}`),
  modelVisualization: (modelId) =>
    request(`/models/${encodeURIComponent(modelId)}/visualization`),
  predict: (modelId, payload) =>
    request(`/models/${encodeURIComponent(modelId)}/predict`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  evaluate: evaluateAndNotify,
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
  xaiShap: requestOptionalShap,
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
