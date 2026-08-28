import React, { useEffect, useMemo, useState } from "react";

const objectIdentityTokens = new WeakMap();
let nextObjectIdentity = 1;

function resetToken(value) {
  if ((typeof value === "object" && value !== null) || typeof value === "function") {
    if (!objectIdentityTokens.has(value)) {
      objectIdentityTokens.set(value, nextObjectIdentity);
      nextObjectIdentity += 1;
    }
    return `object:${objectIdentityTokens.get(value)}`;
  }
  return `primitive:${String(value)}`;
}

/**
 * Keeps an already visited workflow page mounted while it is hidden.
 *
 * When resetKey changes, stale page state is discarded. If the page is currently
 * hidden, remounting is deferred until the user opens the page again so hidden
 * pages do not start data/model-dependent effects unnecessarily.
 */
export default function PersistentWorkflowPage({ active, resetKey, Page }) {
  const token = useMemo(() => resetToken(resetKey), [resetKey]);
  const [mountedToken, setMountedToken] = useState(token);
  const staleWhileHidden = !active && mountedToken !== token;

  useEffect(() => {
    if (active && mountedToken !== token) {
      setMountedToken(token);
    }
  }, [active, mountedToken, token]);

  if (staleWhileHidden) {
    return <div hidden aria-hidden="true" data-workflow-page-cache="stale" />;
  }

  return (
    <div
      hidden={!active}
      aria-hidden={!active}
      data-workflow-page-cache={active ? "active" : "cached"}
    >
      <Page key={token} />
    </div>
  );
}
