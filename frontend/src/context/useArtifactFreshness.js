import { useRef } from "react";

/**
 * Associate an artifact with the model/context object that produced it.
 *
 * Object identity is intentional here: malchan can replace the active predictor
 * while keeping the same registry model_id (for example Tune Best). In that case
 * results from the previous predictor must be treated as stale.
 */
export function useArtifactFreshness(artifact, contextKey) {
  const associationRef = useRef({ artifact: undefined, contextKey: undefined });

  if (!Object.is(associationRef.current.artifact, artifact)) {
    associationRef.current = { artifact, contextKey };
  }

  return Boolean(
    artifact
    && contextKey
    && Object.is(associationRef.current.contextKey, contextKey),
  );
}
