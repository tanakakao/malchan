const BASE64_CHUNK_SIZE = 0x8000;

export function sourceToScriptDataUrl(source) {
  const bytes = new TextEncoder().encode(String(source || ""));
  const chunks = [];
  for (let start = 0; start < bytes.length; start += BASE64_CHUNK_SIZE) {
    chunks.push(String.fromCharCode(...bytes.subarray(start, start + BASE64_CHUNK_SIZE)));
  }
  return `data:text/javascript;charset=utf-8;base64,${btoa(chunks.join(""))}`;
}

export function embeddedScriptTag(source) {
  if (!source) return "";
  return `<script src="${sourceToScriptDataUrl(source)}"></script>`;
}
