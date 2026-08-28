import { useCallback, useEffect, useRef, useState, useSyncExternalStore } from "react";

const workspacePageStateStore = new Map();
const workspacePageStateListeners = new Map();

function resolveInitialState(initialState) {
  return typeof initialState === "function" ? initialState() : initialState;
}

function listenersFor(scope) {
  if (!workspacePageStateListeners.has(scope)) {
    workspacePageStateListeners.set(scope, new Set());
  }
  return workspacePageStateListeners.get(scope);
}

function emitWorkspacePageState(scope) {
  listenersFor(scope).forEach((listener) => listener());
}

function subscribeWorkspacePageState(scope, listener) {
  const listeners = listenersFor(scope);
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
    if (!listeners.size) workspacePageStateListeners.delete(scope);
  };
}

function cachedState(scope, resetKey) {
  const cached = workspacePageStateStore.get(scope);
  return cached && Object.is(cached.resetKey, resetKey) ? cached.state : null;
}

/**
 * Keeps page workspace state alive while the application is mounted.
 *
 * The state survives page component unmount/remount caused by workflow navigation,
 * but is reset when resetKey changes. This intentionally does not use localStorage:
 * prediction tables and analysis results can be large and should not be serialized
 * into browser storage merely to survive a page switch.
 */
export function useWorkspacePageState(scope, initialState, resetKey = null) {
  const initialStateRef = useRef(initialState);
  const resetKeyRef = useRef(resetKey);
  const stateRef = useRef();
  const mountedRef = useRef(true);

  const [state, setInternalState] = useState(() => {
    const cached = workspacePageStateStore.get(scope);
    const initial = cached && Object.is(cached.resetKey, resetKey)
      ? cached.state
      : resolveInitialState(initialStateRef.current);

    workspacePageStateStore.set(scope, { resetKey, state: initial });
    stateRef.current = initial;
    return initial;
  });

  useEffect(() => () => {
    mountedRef.current = false;
  }, []);

  useEffect(() => {
    if (Object.is(resetKeyRef.current, resetKey)) return;

    resetKeyRef.current = resetKey;
    const nextState = resolveInitialState(initialStateRef.current);
    workspacePageStateStore.set(scope, { resetKey, state: nextState });
    stateRef.current = nextState;
    setInternalState(nextState);
    emitWorkspacePageState(scope);
  }, [resetKey, scope]);

  const setState = useCallback((nextStateOrUpdater) => {
    const cached = workspacePageStateStore.get(scope);
    const currentState = cached && Object.is(cached.resetKey, resetKeyRef.current)
      ? cached.state
      : stateRef.current;
    const nextState = typeof nextStateOrUpdater === "function"
      ? nextStateOrUpdater(currentState)
      : nextStateOrUpdater;

    workspacePageStateStore.set(scope, {
      resetKey: resetKeyRef.current,
      state: nextState,
    });
    stateRef.current = nextState;
    if (mountedRef.current) setInternalState(nextState);
    emitWorkspacePageState(scope);
  }, [scope]);

  return [state, setState];
}

/**
 * Observe one persisted page workspace without owning or mounting that page.
 *
 * A reset-key mismatch returns null immediately, so stale results from a previous
 * model or dataset cannot be mistaken for current workflow completion.
 */
export function useWorkspacePageStateSnapshot(scope, resetKey = null) {
  const subscribe = useCallback(
    (listener) => subscribeWorkspacePageState(scope, listener),
    [scope],
  );
  const getSnapshot = useCallback(
    () => cachedState(scope, resetKey),
    [scope, resetKey],
  );

  return useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
}

/** Clears one persisted page workspace explicitly. */
export function clearWorkspacePageState(scope) {
  workspacePageStateStore.delete(scope);
  emitWorkspacePageState(scope);
}
