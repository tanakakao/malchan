const REPORT_SECTION_PREFIX = '<section class="report-section" id="';

function sectionMarker(id) {
  return `${REPORT_SECTION_PREFIX}${id}">`;
}

export function removeReportSection(html, id) {
  const marker = sectionMarker(id);
  const markerIndex = html.indexOf(marker);
  if (markerIndex < 0) return html;

  const lineStart = html.lastIndexOf("\n", markerIndex);
  const start = lineStart >= 0 ? lineStart + 1 : markerIndex;
  const nextSection = html.indexOf(REPORT_SECTION_PREFIX, markerIndex + marker.length);
  const footer = html.indexOf("<footer", markerIndex + marker.length);
  const end = nextSection >= 0
    ? nextSection
    : footer >= 0
      ? footer
      : html.length;
  return `${html.slice(0, start)}${html.slice(end)}`;
}

export function removeReportNavItem(html, id) {
  const marker = `<a href="#${id}">`;
  const start = html.indexOf(marker);
  if (start < 0) return html;
  const close = html.indexOf("</a>", start + marker.length);
  if (close < 0) return html;
  return `${html.slice(0, start)}${html.slice(close + 4)}`;
}
