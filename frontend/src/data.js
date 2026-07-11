import Papa from "papaparse";
import readXlsxFile from "read-excel-file";

export const REGRESSION_MODELS = [
  "線形回帰",
  "Ridge",
  "Lasso",
  "ElasticNet",
  "PLS回帰",
  "ランダムフォレスト回帰",
  "Extra-Trees",
  "Gradient Boosting",
  "HistGradientBoosting",
  "XGBoost",
  "LightGBM",
  "CatBoost",
  "サポートベクター回帰",
  "K近傍法",
  "ガウス過程回帰",
  "ベイズ線形回帰",
];

export const CLASSIFICATION_MODELS = [
  "ロジスティック回帰",
  "決定木",
  "ランダムフォレスト",
  "Extra-Trees",
  "Gradient Boosting",
  "HistGradientBoosting",
  "XGBoost",
  "LightGBM",
  "CatBoost",
  "サポートベクターマシン",
  "K近傍法",
  "ガウス過程分類",
  "ナイーブベイズ",
];

const emptyLike = (value) => value === null || value === undefined || value === "";

function normalizeCell(value) {
  if (value instanceof Date) return value.toISOString();
  if (typeof value === "number" && Number.isNaN(value)) return null;
  return value;
}

function matrixToRecords(matrix) {
  const nonEmptyRows = matrix.filter((row) => row.some((value) => !emptyLike(value)));
  if (nonEmptyRows.length < 2) throw new Error("ヘッダーとデータ行を確認してください。");
  const headers = nonEmptyRows[0].map((value, index) => String(value || `column_${index + 1}`).trim());
  return nonEmptyRows.slice(1).map((row) => Object.fromEntries(headers.map((header, index) => [header, normalizeCell(row[index] ?? null)])));
}

export async function parseTabularFile(file) {
  const extension = file.name.split(".").pop()?.toLowerCase();
  let records;
  let sheetName = "CSV";
  if (extension === "csv") {
    const content = await file.text();
    const parsed = Papa.parse(content.replace(/^\uFEFF/, ""), {
      skipEmptyLines: true,
    });
    if (parsed.errors.length) {
      throw new Error(parsed.errors[0].message);
    }
    records = matrixToRecords(parsed.data);
  } else if (extension === "xlsx") {
    const matrix = await readXlsxFile(file);
    records = matrixToRecords(matrix);
    sheetName = "Sheet 1";
  } else {
    throw new Error("CSVまたはXLSXファイルを選択してください。");
  }
  const rows = records.filter((row) => Object.values(row).some((value) => !emptyLike(value)));
  if (!rows.length) throw new Error("データ行が見つかりませんでした。");
  return { rows, sheetName };
}

export function coerceRows(rows) {
  const columns = Array.from(new Set(rows.flatMap((row) => Object.keys(row))));
  const numericColumns = [];
  const categoricalColumns = [];

  for (const column of columns) {
    const values = rows.map((row) => row[column]).filter((value) => !emptyLike(value));
    const numeric = values.length > 0 && values.every((value) => {
      if (typeof value === "number") return Number.isFinite(value);
      if (typeof value !== "string") return false;
      return value.trim() !== "" && Number.isFinite(Number(value));
    });
    (numeric ? numericColumns : categoricalColumns).push(column);
  }

  const normalizedRows = rows.map((row) => {
    const normalized = {};
    for (const column of columns) {
      const value = row[column];
      if (emptyLike(value)) normalized[column] = null;
      else if (numericColumns.includes(column)) normalized[column] = Number(value);
      else normalized[column] = String(value);
    }
    return normalized;
  });

  return { rows: normalizedRows, columns, numericColumns, categoricalColumns };
}

export function columnStats(rows, columns) {
  return columns.map((column) => {
    const values = rows.map((row) => row[column]);
    const present = values.filter((value) => !emptyLike(value));
    const numeric = present.filter((value) => typeof value === "number" && Number.isFinite(value));
    return {
      column,
      count: present.length,
      missing: values.length - present.length,
      unique: new Set(present.map((value) => String(value))).size,
      min: numeric.length ? Math.min(...numeric) : null,
      max: numeric.length ? Math.max(...numeric) : null,
      mean: numeric.length ? numeric.reduce((sum, value) => sum + value, 0) / numeric.length : null,
    };
  });
}

export function uniqueValues(rows, column, limit = 50) {
  return Array.from(new Set(rows.map((row) => row[column]).filter((value) => !emptyLike(value)))).slice(0, limit);
}

export function pearson(xValues, yValues) {
  const pairs = xValues
    .map((x, index) => [x, yValues[index]])
    .filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y));
  if (pairs.length < 2) return 0;
  const xMean = pairs.reduce((sum, [x]) => sum + x, 0) / pairs.length;
  const yMean = pairs.reduce((sum, [, y]) => sum + y, 0) / pairs.length;
  let numerator = 0;
  let xDenominator = 0;
  let yDenominator = 0;
  for (const [x, y] of pairs) {
    const dx = x - xMean;
    const dy = y - yMean;
    numerator += dx * dy;
    xDenominator += dx * dx;
    yDenominator += dy * dy;
  }
  const denominator = Math.sqrt(xDenominator * yDenominator);
  return denominator ? numerator / denominator : 0;
}

export function correlationMatrix(rows, columns) {
  return columns.map((left) =>
    columns.map((right) => pearson(rows.map((row) => row[left]), rows.map((row) => row[right]))),
  );
}

export function numericSummary(rows, column) {
  const values = rows.map((row) => row[column]).filter((value) => Number.isFinite(value));
  if (!values.length) return { min: 0, max: 1 };
  const min = Math.min(...values);
  const max = Math.max(...values);
  return { min, max: max > min ? max : min + 1 };
}

export function formatNumber(value, digits = 4) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  if (typeof value !== "number") return String(value);
  return new Intl.NumberFormat("ja-JP", { maximumFractionDigits: digits }).format(value);
}
