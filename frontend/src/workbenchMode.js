import { useSyncExternalStore } from "react";

const WORKBENCH_MODE_KEY = "malchan-web-workbench-mode";
const listeners = new Set();

function readWorkbenchMode() {
  if (typeof window === "undefined") return "advanced";
  return window.localStorage.getItem(WORKBENCH_MODE_KEY) === "simple"
    ? "simple"
    : "advanced";
}

function subscribe(listener) {
  listeners.add(listener);
  const handleStorage = (event) => {
    if (event.key === WORKBENCH_MODE_KEY) listener();
  };
  window.addEventListener("storage", handleStorage);
  return () => {
    listeners.delete(listener);
    window.removeEventListener("storage", handleStorage);
  };
}

export function setWorkbenchMode(mode) {
  window.localStorage.setItem(WORKBENCH_MODE_KEY, mode);
  listeners.forEach((listener) => listener());
}

export function useWorkbenchMode() {
  return useSyncExternalStore(subscribe, readWorkbenchMode, () => "advanced");
}
