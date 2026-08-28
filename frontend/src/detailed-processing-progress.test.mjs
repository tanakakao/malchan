import assert from "node:assert/strict";
import fs from "node:fs";

const transport = fs.readFileSync(new URL("./detailed-progress.js", import.meta.url), "utf8");
const component = fs.readFileSync(
  new URL("./components/DetailedProcessingProgress.jsx", import.meta.url),
  "utf8",
);
const css = fs.readFileSync(new URL("./detailed-progress.css", import.meta.url), "utf8");
const main = fs.readFileSync(new URL("./main.jsx", import.meta.url), "utf8");
const backendProgress = fs.readFileSync(
  new URL("../../src/malchan/app/progress.py", import.meta.url),
  "utf8",
);
const backendMain = fs.readFileSync(
  new URL("../../src/malchan/app/api/main.py", import.meta.url),
  "utf8",
);

assert.match(transport, /X-Malchan-Progress-ID/);
assert.match(transport, /POLL_INTERVAL_MS = 350/);
assert.match(transport, /\/progress\/\$\{encodeURIComponent\(progressId\)\}/);
assert.match(transport, /supportsDetailedProgress/);
assert.match(transport, /inverse-analysis/);
assert.match(transport, /malchan:detailed-progress/);

assert.match(component, /目的変数 \$\{dimension\.current\} \/ \$\{dimension\.total\}/);
assert.match(component, /Optuna \$\{dimension\.current\} \/ \$\{dimension\.total\} trials/);
assert.match(component, /CV fold \$\{dimension\.current\} \/ \$\{dimension\.total\}/);
assert.match(component, /role="progressbar"/);
assert.match(component, /aria-valuenow=\{dimension\.current\}/);
assert.match(component, /backendで完了したtarget|バックエンドで完了したtarget/);

assert.match(css, /\.detailed-progress-track/);
assert.match(css, /transition: width 0\.2s ease/);
assert.match(css, /font-variant-numeric: tabular-nums/);
assert.match(css, /@media \(max-width: 620px\)/);

assert.match(main, /installDetailedProgressTransport\(\)/);
assert.match(main, /<DetailedProcessingProgress \/>/);

assert.match(backendProgress, /def progress_scope/);
assert.match(backendProgress, /def report_dimension/);
assert.match(backendProgress, /def set_target_plan/);
assert.match(backendProgress, /def mark_target/);
assert.match(backendProgress, /training\.OptunaSearchCV = progress_optuna_search_cv/);
assert.match(backendProgress, /SingleOutputMLModelPipeline\.cv_score = cv_score/);
assert.match(backendProgress, /training\.cv_fit = cv_fit/);
assert.match(backendProgress, /optuna\.study\.Study\.optimize = optimize/);
assert.match(backendMain, /progress_scope\(progress_id, operation=operation\)/);
assert.match(backendMain, /create_progress_router\(\)/);

console.log("detailed processing progress checks passed");
