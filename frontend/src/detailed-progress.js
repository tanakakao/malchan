const DETAILED_PROGRESS_EVENT = "malchan:detailed-progress";
const PROGRESS_HEADER = "X-Malchan-Progress-ID";
const POLL_INTERVAL_MS = 350;

let installed = false;
let fallbackSequence = 0;

function makeProgressId() {
  if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID();
  fallbackSequence += 1;
  return `malchan-${Date.now()}-${fallbackSequence}`;
}

function requestUrl(input) {
  if (input instanceof Request) return new URL(input.url, window.location.origin);
  return new URL(String(input), window.location.origin);
}

function requestMethod(input, init = {}) {
  return String(init.method || (input instanceof Request ? input.method : "GET")).toUpperCase();
}

function requestCredentials(input, init = {}) {
  return init.credentials || (input instanceof Request ? input.credentials : undefined);
}

function supportsDetailedProgress(url, method) {
  if (method !== "POST") return false;
  const path = url.pathname;
  return (
    /\/models\/?$/.test(path)
    || /\/models\/[^/]+\/evaluate\/?$/.test(path)
    || /\/models\/[^/]+\/compare\/?$/.test(path)
    || /\/models\/[^/]+\/comparison\/tune-best\/?$/.test(path)
    || /\/models\/[^/]+\/inverse-analysis\/?$/.test(path)
  );
}

function progressUrlFor(requestUrlValue, progressId) {
  const markerIndex = requestUrlValue.pathname.indexOf("/models");
  if (markerIndex < 0) return null;
  const apiPrefix = requestUrlValue.pathname.slice(0, markerIndex).replace(/\/$/, "");
  return `${requestUrlValue.origin}${apiPrefix}/progress/${encodeURIComponent(progressId)}`;
}

function emitDetailedProgress(progressId, progress) {
  window.dispatchEvent(new CustomEvent(DETAILED_PROGRESS_EVENT, {
    detail: { progressId, progress },
  }));
}

async function readProgress(originalFetch, url, progressId, credentials) {
  try {
    const response = await originalFetch(url, {
      method: "GET",
      cache: "no-store",
      credentials,
      headers: { Accept: "application/json" },
    });
    if (!response.ok) return null;
    const progress = await response.json();
    emitDetailedProgress(progressId, progress);
    return progress;
  } catch {
    return null;
  }
}

function delay(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

async function pollProgress(originalFetch, url, progressId, state, credentials) {
  while (!state.stopped) {
    const progress = await readProgress(originalFetch, url, progressId, credentials);
    if (progress && progress.status !== "running") return;
    await delay(POLL_INTERVAL_MS);
  }
}

/**
 * Adds a progress id to long foreground API requests and polls the lightweight
 * backend progress endpoint while the original synchronous request is running.
 */
export function installDetailedProgressTransport() {
  if (installed || typeof window === "undefined" || typeof window.fetch !== "function") return;
  installed = true;

  const originalFetch = window.fetch.bind(window);
  window.fetch = function detailedProgressFetch(input, init = {}) {
    const url = requestUrl(input);
    const method = requestMethod(input, init);
    if (!supportsDetailedProgress(url, method)) {
      return originalFetch(input, init);
    }

    const progressId = makeProgressId();
    const progressUrl = progressUrlFor(url, progressId);
    if (!progressUrl) return originalFetch(input, init);

    const headers = new Headers(input instanceof Request ? input.headers : undefined);
    new Headers(init.headers || {}).forEach((value, key) => headers.set(key, value));
    headers.set(PROGRESS_HEADER, progressId);

    const credentials = requestCredentials(input, init);
    const state = { stopped: false };
    const requestPromise = originalFetch(input, { ...init, headers });
    void pollProgress(originalFetch, progressUrl, progressId, state, credentials);

    const finish = () => {
      state.stopped = true;
      void readProgress(originalFetch, progressUrl, progressId, credentials);
    };
    void requestPromise.then(finish, finish);
    return requestPromise;
  };
}

export { DETAILED_PROGRESS_EVENT, POLL_INTERVAL_MS, PROGRESS_HEADER };
