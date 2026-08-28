import { useCallback, useMemo, useSyncExternalStore } from "react";

const listeners = new Set();
let store = {
  datasetKey: null,
  columns: [],
  enabledColumns: [],
};

function sameColumns(left, right) {
  return left.length === right.length && left.every((column, index) => column === right[index]);
}

function ensureDataset(datasetKey, columns) {
  const normalizedColumns = [...columns];
  if (store.datasetKey === datasetKey && sameColumns(store.columns, normalizedColumns)) return;

  store = {
    datasetKey,
    columns: normalizedColumns,
    enabledColumns: normalizedColumns,
  };
}

function subscribe(listener) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

function getSnapshot() {
  return store.enabledColumns;
}

function emit() {
  listeners.forEach((listener) => listener());
}

/**
 * Keeps the Data-page analysis-column selection alive across workflow navigation.
 *
 * The raw dataset is never mutated. Disabling a column only removes it from the
 * analysis workspace; loading a different dataset resets all columns to enabled.
 */
export function useAnalysisColumnSelection(columns, datasetKey) {
  ensureDataset(datasetKey, columns);
  const enabledColumns = useSyncExternalStore(subscribe, getSnapshot, getSnapshot);

  const enabledSet = useMemo(() => new Set(enabledColumns), [enabledColumns]);

  const setEnabledColumns = useCallback((nextColumns) => {
    const requested = new Set(nextColumns);
    const normalized = store.columns.filter((column) => requested.has(column));
    if (sameColumns(store.enabledColumns, normalized)) return;

    store = {
      ...store,
      enabledColumns: normalized,
    };
    emit();
  }, []);

  const setColumnEnabled = useCallback((column, enabled) => {
    const current = new Set(store.enabledColumns);
    if (enabled) current.add(column);
    else current.delete(column);
    setEnabledColumns([...current]);
  }, [setEnabledColumns]);

  return {
    enabledColumns,
    enabledSet,
    setEnabledColumns,
    setColumnEnabled,
  };
}
