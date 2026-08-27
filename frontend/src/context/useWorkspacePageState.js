import { useCallback, useEffect, useRef, useState } from "react";

const workspacePageStateStore = new Map();

function resolveInitialState(initialState) {
  return typeof initialState === "function" ? initialState() : initialState;
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

  const [state, setInternalState] = useState(() => {
    const cached = workspacePageStateStore.get(scope);
    const initial = cached && Object.is(cached.resetKey, resetKey)
      ? cached.state
      : resolveInitialState(initialStateRef.current);

    workspacePageStateStore.set(scope, { resetKey, state: initial });
    stateRef.current = initial;
    return initial;
  });

  useEffect(() => {
    if (Object.is(resetKeyRef.current, resetKey)) return;

    resetKeyRef.current = resetKey;
    const nextState = resolveInitialState(initialStateRef.current);
    workspacePageStateStore.set(scope, { resetKey, state: nextState });
    stateRef.current = nextState;
    setInternalState(nextState);
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
    setInternalState(nextState);
  }, [scope]);

  return [state, setState];
}

/** Clears one persisted page workspace explicitly. */
export function clearWorkspacePageState(scope) {
  workspacePageStateStore.delete(scope);
}
