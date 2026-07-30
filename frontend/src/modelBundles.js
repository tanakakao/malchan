import { ApiError } from "./api";

const API_BASE = (import.meta.env.VITE_API_BASE || "/api").replace(/\/$/, "");
const MODEL_BUNDLE_MEDIA_TYPE = "application/vnd.malchan.model";
const importedRowsByModel = new Map();
let activeImportedRows = [];

async function responseError(response) {
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json")
    ? await response.json()
    : await response.text();
  const detail = payload?.detail ?? payload;
  return new ApiError(
    typeof detail === "string" ? detail : JSON.stringify(detail),
    response.status,
    detail,
  );
}

function attachmentFilename(header, modelId) {
  if (!header) return `malchan-model-${modelId}.malchan`;
  const utf8 = header.match(/filename\*=UTF-8''([^;]+)/i);
  if (utf8) {
    try {
      return decodeURIComponent(utf8[1]);
    } catch {
      return utf8[1];
    }
  }
  const quoted = header.match(/filename="([^"]+)"/i);
  if (quoted) return quoted[1];
  const plain = header.match(/filename=([^;]+)/i);
  return plain?.[1]?.trim() || `malchan-model-${modelId}.malchan`;
}

export function importedModelRows(modelId) {
  if (!modelId) return [];
  return importedRowsByModel.get(modelId) || [];
}

export function activeImportedModelRows() {
  return activeImportedRows;
}

export async function downloadModelBundle(modelId) {
  const response = await fetch(
    `${API_BASE}/models/${encodeURIComponent(modelId)}/export`,
    {
      method: "GET",
      cache: "no-store",
      headers: { Accept: MODEL_BUNDLE_MEDIA_TYPE },
    },
  );
  if (!response.ok) throw await responseError(response);
  return {
    blob: await response.blob(),
    filename: attachmentFilename(
      response.headers.get("content-disposition"),
      modelId,
    ),
  };
}

export async function importModelBundle(file) {
  if (!(file instanceof Blob)) {
    throw new TypeError("読み込むモデルファイルを選択してください。");
  }
  const response = await fetch(`${API_BASE}/model-bundles/import`, {
    method: "POST",
    cache: "no-store",
    headers: {
      "Content-Type": file.type || MODEL_BUNDLE_MEDIA_TYPE,
      "X-Malchan-Filename": encodeURIComponent(file.name || "model.malchan"),
    },
    body: file,
  });
  if (!response.ok) throw await responseError(response);

  const payload = await response.json();
  const modelId = payload?.model?.model_id;
  const trainingRows = Array.isArray(payload?.training_rows)
    ? payload.training_rows
    : [];
  activeImportedRows = trainingRows;
  if (modelId) importedRowsByModel.set(modelId, trainingRows);
  return payload;
}

export { MODEL_BUNDLE_MEDIA_TYPE };
