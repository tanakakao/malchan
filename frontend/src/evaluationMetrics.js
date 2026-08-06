function normalizedMetricName(value) {
  return String(value ?? "").trim().toLowerCase();
}

function directMetricValue(values, normalizedMetric) {
  const key = Object.keys(values).find(
    (candidate) => normalizedMetricName(candidate) === normalizedMetric,
  );
  if (!key) return undefined;
  const value = Number(values[key]);
  return Number.isFinite(value) ? value : undefined;
}

export function metricValue(values, metric) {
  if (!values || typeof values !== "object") return undefined;
  const normalized = normalizedMetricName(metric);
  if (!normalized) return undefined;

  const directValue = directMetricValue(values, normalized);
  if (directValue !== undefined) return directValue;

  // malchanの回帰CVはRMSEを返すため、同じfoldのMSEはRMSE²として補完できる。
  // APIがMSEを直接返す場合は上のdirectValueを優先する。
  if (normalized === "mse") {
    const rmse = directMetricValue(values, "rmse");
    return rmse === undefined ? undefined : rmse ** 2;
  }
  return undefined;
}

export function evaluationMetricNames(result) {
  const names = [];
  const seen = new Set();
  const candidates = [
    ...(result?.train || []).flatMap((record) => Object.keys(record || {})),
    ...(result?.test || []).flatMap((record) => Object.keys(record || {})),
    ...Object.keys(result?.oof || {}),
  ];

  candidates.forEach((name) => {
    const normalized = normalizedMetricName(name);
    if (!normalized || seen.has(normalized)) return;
    seen.add(normalized);
    names.push(name);
  });
  return names;
}
