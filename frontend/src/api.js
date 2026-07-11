const API_BASE = (import.meta.env.VITE_API_BASE || "/api").replace(/\/$/, "");

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

export const api = {
  health: () => request("/health"),
  train: (payload) => request("/models", { method: "POST", body: JSON.stringify(payload) }),
  listModels: () => request("/models"),
  modelInfo: (modelId) => request(`/models/${encodeURIComponent(modelId)}`),
  predict: (modelId, payload) =>
    request(`/models/${encodeURIComponent(modelId)}/predict`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  compare: (modelId, payload) =>
    request(`/models/${encodeURIComponent(modelId)}/compare`, {
      method: "POST",
      body: JSON.stringify(payload),
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
      body: JSON.stringify(payload),
    }),
  xaiSummary: (modelId) =>
    request(`/models/${encodeURIComponent(modelId)}/xai`),
  recomputeXai: (modelId, payload = {}) =>
    request(`/models/${encodeURIComponent(modelId)}/xai/recompute`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  xaiImportance: (modelId, target, options = {}) =>
    request(
      `/models/${encodeURIComponent(modelId)}/xai/${encodeURIComponent(target)}/importance${query(options)}`,
    ),
  xaiShap: (modelId, target, feature) =>
    request(
      `/models/${encodeURIComponent(modelId)}/xai/${encodeURIComponent(target)}/shap${query({ feature })}`,
    ),
  xaiPdp: (modelId, target, feature, options = {}) =>
    request(
      `/models/${encodeURIComponent(modelId)}/xai/${encodeURIComponent(target)}/pdp${query({ feature, ...options })}`,
    ),
  deleteModel: (modelId) =>
    request(`/models/${encodeURIComponent(modelId)}`, { method: "DELETE" }),
};
