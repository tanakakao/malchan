import assert from "node:assert/strict";

import {
  removeReportNavItem,
  removeReportSection,
} from "./report-html-sections.js";

const source = `
<nav><a href="#diagnostics">精度診断</a><a href="#optimization">予測・逆解析</a></nav>
<section class="report-section" id="diagnostics">
  <section class="nested">diagnostic chart</section>
</section>
<section class="report-section" id="optimization">optimization</section>
<footer>footer</footer>`;

const withoutDiagnostics = removeReportSection(source, "diagnostics");
assert.doesNotMatch(withoutDiagnostics, /id="diagnostics"/);
assert.doesNotMatch(withoutDiagnostics, /diagnostic chart/);
assert.match(withoutDiagnostics, /id="optimization"/);
assert.match(withoutDiagnostics, /<footer>footer<\/footer>/);

const withoutNav = removeReportNavItem(withoutDiagnostics, "diagnostics");
assert.doesNotMatch(withoutNav, /href="#diagnostics"/);
assert.match(withoutNav, /href="#optimization"/);

assert.equal(removeReportSection(source, "missing"), source);
assert.equal(removeReportNavItem(source, "missing"), source);

console.log("report HTML section checks passed");
