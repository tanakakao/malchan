import assert from "node:assert/strict";
import fs from "node:fs";
import {
  REPORT_TARGET_TABS_CSS,
  reportTargetTabsRuntimeScript,
} from "./report-target-tabs.js";

const source = fs.readFileSync(new URL("./report-visualizations.js", import.meta.url), "utf8");
const apiSource = fs.readFileSync(new URL("./api.js", import.meta.url), "utf8");
const interactiveExport = fs.readFileSync(new URL("./report-interactive-export.js", import.meta.url), "utf8");
const interactiveFigures = fs.readFileSync(new URL("./report-interactive-figures.js", import.meta.url), "utf8");
const interactiveRuntime = fs.readFileSync(new URL("./report-interactive-runtime.js", import.meta.url), "utf8");
const reportPage = fs.readFileSync(new URL("./pages/ReportPage.jsx", import.meta.url), "utf8");
const targetTabsRuntime = reportTargetTabsRuntimeScript();

assert.match(source, /visualizationYy\(modelId, target, \{ cv: true, residual: false \}\)/);
assert.match(source, /visualizationYy\(modelId, target, \{ cv, residual: true \}\)/);
assert.match(source, /IMPORTANCE_METHODS/);
assert.match(source, /Baseline \+ SHAP/);
assert.match(source, /visualizationBeeswarm/);
assert.match(source, /figureToPng/);
assert.match(source, /plotly\.toImage/);

assert.match(apiSource, /function unavailableShapFeature\(error\)/);
assert.match(apiSource, /Unknown or unavailable SHAP feature/);
assert.match(apiSource, /if \(!unavailableShapFeature\(error\)\) throw error/);
assert.match(apiSource, /records:\s*\[\]/);
assert.match(apiSource, /value_columns:\s*\[\]/);
assert.match(apiSource, /unavailable:\s*true/);
assert.match(apiSource, /xaiShap:\s*requestOptionalShap/);

assert.match(interactiveFigures, /collectInteractiveFigures/);
assert.match(interactiveFigures, /combinedImportanceFigure/);
assert.match(interactiveFigures, /legacyPdFigure/);
assert.doesNotMatch(interactiveFigures, /visualizationPdp2d/);
assert.doesNotMatch(interactiveFigures, /pd2d/);
assert.match(interactiveExport, /downloadInteractiveHtmlReport/);
assert.match(interactiveExport, /attachInteractiveRegistry/);
assert.match(interactiveExport, /removeReportSection\(reportHtml, "diagnostics"\)/);
assert.match(interactiveExport, /removeReportNavItem\(reportHtml, "diagnostics"\)/);
assert.match(interactiveExport, /<span>05<\/span>/);
assert.match(interactiveExport, /data-open-report-figure/);
assert.match(interactiveExport, /embeddedScriptTag\(plotlySource\)/);
assert.doesNotMatch(interactiveExport, /safeInlineScript\(plotlySource\)/);
assert.match(interactiveExport, /REPORT_TARGET_TABS_CSS/);
assert.match(interactiveExport, /reportTargetTabsRuntimeScript\(\)/);
assert.match(interactiveExport, /numericFeatures:\s*\[\]/);
assert.doesNotMatch(interactiveExport, /result\.pd2d/);
assert.doesNotMatch(interactiveExport, /2D Partial Dependence/);
assert.doesNotMatch(reportPage, /2D PD/);

assert.match(targetTabsRuntime, /#comparison/);
assert.match(targetTabsRuntime, /#diagnostics/);
assert.match(targetTabsRuntime, /#model-figures/);
assert.match(targetTabsRuntime, /cards\.length <= 1/);
assert.match(targetTabsRuntime, /activateTarget/);
assert.match(targetTabsRuntime, /ArrowRight/);
assert.match(targetTabsRuntime, /aria-selected/);
assert.match(REPORT_TARGET_TABS_CSS, /@media print/);
assert.match(REPORT_TARGET_TABS_CSS, /report-target-panel\[hidden\]/);

assert.match(interactiveRuntime, /plotly\.min\.js\?raw/);
assert.match(interactiveRuntime, /malchan-figure-modal/);
assert.match(interactiveRuntime, /malchan-x-min/);
assert.match(interactiveRuntime, /malchan-font-size/);
assert.match(interactiveRuntime, /Plotly\.relayout/);
assert.match(interactiveRuntime, /Plotly\.downloadImage/);
assert.match(reportPage, /downloadInteractiveHtmlReport/);
assert.match(reportPage, /軸レンジ、文字サイズ、表示高さ/);

console.log("report visualization source checks passed");
