function normalizedMetricName(value) {
  return String(value ?? "").trim().toLowerCase();
}

export function metricValue(values, metric) {
  if (!values || typeof values !== "object") return undefined;
  const normalized = normalizedMetricName(metric);
  if (!normalized) return undefined;
  const key = Object.keys(values).find(
    (candidate) => normalizedMetricName(candidate) === normalized,
  );
  if (!key) return undefined;
  const value = Number(values[key]);
  return Number.isFinite(value) ? value : undefined;
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
