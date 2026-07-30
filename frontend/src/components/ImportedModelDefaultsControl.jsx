import { useEffect } from "react";
import { numericSummary } from "../data";
import { importedModelRows } from "../modelBundles";
import { useWorkbench } from "../context/WorkbenchContext";

function medianValue(rows, column) {
  const values = rows
    .map((row) => row[column])
    .filter(Number.isFinite)
    .sort((left, right) => left - right);
  if (!values.length) return "";
  const middle = Math.floor(values.length / 2);
  return values.length % 2
    ? values[middle]
    : (values[middle - 1] + values[middle]) / 2;
}

function modeValue(rows, column) {
  const values = rows
    .map((row) => row[column])
    .filter((value) => value !== null && value !== undefined && value !== "");
  if (!values.length) return "";

  const counts = new Map();
  let selected = values[0];
  let selectedCount = 0;
  values.forEach((value) => {
    const key = `${typeof value}:${String(value)}`;
    const next = (counts.get(key) || 0) + 1;
    counts.set(key, next);
    if (next > selectedCount) {
      selected = value;
      selectedCount = next;
    }
  });
  return selected;
}

export default function ImportedModelDefaultsControl() {
  const {
    modelInfo,
    features,
    numFeatures,
    targets,
    tasks,
    setPredictValues,
    setBounds,
    setObjectives,
  } = useWorkbench();

  useEffect(() => {
    const rows = importedModelRows(modelInfo?.model_id);
    if (!modelInfo?.model_id || !rows.length) return;

    const numericSet = new Set(numFeatures);
    setPredictValues(Object.fromEntries(
      features.map((column) => [
        column,
        numericSet.has(column)
          ? medianValue(rows, column)
          : modeValue(rows, column),
      ]),
    ));
    setBounds(Object.fromEntries(
      numFeatures.map((column) => [column, numericSummary(rows, column)]),
    ));
    setObjectives(Object.fromEntries(
      targets.map((target) => [
        target,
        tasks[target] === "classification"
          ? { mode: "target", value: modeValue(rows, target) }
          : { mode: "direction", value: "max" },
      ]),
    ));
  }, [modelInfo?.model_id]);

  return null;
}
