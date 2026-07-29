import React, { useMemo } from "react";

const GROUPS = [
  { key: "numeric", label: "連続値", match: (name) => name === "num" },
  { key: "categorical", label: "通常カテゴリ", match: (name) => name === "cat" },
  { key: "composition", label: "組成式", match: (name) => name.startsWith("comp_") },
  { key: "smiles", label: "分子表記（SMILES）", match: (name) => name.startsWith("smiles_") },
];

const LABELS = {
  imputer: "欠損値補完",
  scaler: "スケーリング",
  identity: "変換なし",
  "one-hot": "One-Hotエンコード",
  ordinal: "順序エンコード",
  poly: "多項式特徴量",
  polynomial: "多項式特徴量",
  decomposition: "次元削減",
  sampling: "サンプリング",
  sampler: "サンプリング",
  predictor: "予測モデル",
};

const COMMON_NAMES = new Set([
  "num_cat_common",
  "common_preprocess",
  "common_transform",
  "poly",
  "polynomial",
  "decomposition",
  "sampling",
  "sampler",
]);

function walk(node, result = []) {
  if (!node) return result;
  result.push(node);
  (node.children || []).forEach((child) => walk(child, result));
  return result;
}

function unique(values) {
  return [...new Set(values.filter(Boolean))];
}

function operationLabel(node) {
  if (node.kind === "passthrough") return "そのまま使用";
  if (node.kind === "dropped") return "除外";
  const name = LABELS[node.name] || node.name;
  const className = node.class_name || "";
  if (["pipeline", "branch", "ensemble", "reference"].includes(node.kind)) return null;
  if (!name || name === "model") return className || null;
  return className && className !== name ? `${name}（${className}）` : name;
}

function operationsUnder(node) {
  return unique(walk(node, []).map(operationLabel));
}

function buildRows(structure, featureColumns) {
  const nodes = walk(structure, []);
  const rows = GROUPS.map((group) => {
    const roots = nodes.filter((node) => group.match(node.name || ""));
    return {
      key: group.key,
      label: group.label,
      columns: unique(roots.flatMap((node) => node.columns || [])),
      operations: unique(roots.flatMap(operationsUnder)),
    };
  }).filter((row) => row.columns.length > 0);

  const classified = new Set(rows.flatMap((row) => row.columns));
  const remaining = unique(featureColumns).filter((column) => !classified.has(column));
  if (remaining.length) {
    rows.push({ key: "other", label: "その他", columns: remaining, operations: ["列別処理"] });
  }
  if (!rows.length) {
    rows.push({ key: "other", label: "入力変数", columns: unique(featureColumns), operations: ["変換なし"] });
  }
  rows.forEach((row) => {
    if (!row.operations.length) row.operations = ["変換なし"];
  });
  return rows;
}

function buildCommon(structure) {
  const nodes = walk(structure, []);
  const roots = nodes.filter((node) => COMMON_NAMES.has(node.name));
  const operations = unique(roots.flatMap(operationsUnder));
  return operations.length ? operations : ["共通前処理なし"];
}

function buildModels(structure, modelNames) {
  // APIに保存されたモデル名を表示上の正として使用する。
  // 例: RandomForest と RandomForestRegressor を二重表示しない。
  const configured = unique(modelNames || []);
  if (configured.length) return configured;

  const fitted = unique(
    walk(structure, [])
      .filter((node) => node.kind === "estimator")
      .map((node) => node.class_name || node.name),
  );
  return fitted.length ? fitted : ["モデル情報なし"];
}

function TextList({ items }) {
  return (
    <ul className="model-summary-list">
      {items.map((item) => <li key={item}>{item}</li>)}
    </ul>
  );
}

export default function ModelStructureSummaryTable({ structure, featureColumns = [], modelNames = [] }) {
  const summary = useMemo(() => ({
    rows: buildRows(structure, featureColumns),
    common: buildCommon(structure),
    models: buildModels(structure, modelNames),
  }), [structure, featureColumns.join("|"), modelNames.join("|")]);

  return (
    <div className="model-structure-summary-scroll">
      <table className="model-structure-summary-table">
        <thead>
          <tr><th>変数名</th><th>共通前処理</th><th>種別別の前処理</th><th>モデル</th></tr>
        </thead>
        <tbody>
          {summary.rows.map((row, index) => (
            <tr key={row.key}>
              <td className={`model-summary-variable-cell type-${row.key}`}>
                <strong>{row.label}</strong>
                <div className="model-summary-columns">
                  {row.columns.map((column) => <span key={column}>{column}</span>)}
                </div>
              </td>
              {index === 0 && <td rowSpan={summary.rows.length}><TextList items={summary.common} /></td>}
              <td><TextList items={row.operations} /></td>
              {index === 0 && <td rowSpan={summary.rows.length}><TextList items={summary.models} /></td>}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
