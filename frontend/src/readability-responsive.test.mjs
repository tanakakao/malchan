import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const readability = fs.readFileSync(path.join(here, "readability.css"), "utf8");
const main = fs.readFileSync(path.join(here, "main.jsx"), "utf8");
const dataTable = fs.readFileSync(path.join(here, "components", "DataTable.jsx"), "utf8");

assert.match(readability, /--ui-body-font-size:\s*14px/);
assert.match(readability, /--ui-control-font-size:\s*14px/);
assert.match(readability, /--ui-label-font-size:\s*13px/);
assert.match(readability, /--ui-secondary-font-size:\s*12px/);
assert.match(readability, /--ui-caption-font-size:\s*11px/);
assert.doesNotMatch(readability, /--ui-control-font-size:\s*15px/);

assert.match(readability, /\.sampler-description span,[\s\S]*font-size:\s*12px\s*!important/);
assert.match(readability, /\.optimize-type-badge,[\s\S]*font-size:\s*11px\s*!important/);
assert.match(readability, /\.inverse-objective-table th,[\s\S]*font-size:\s*12px\s*!important/);
assert.match(readability, /\.inverse-objective-table td,[\s\S]*font-size:\s*13px\s*!important/);

assert.match(readability, /text-overflow:\s*ellipsis/);
assert.match(readability, /overflow-wrap:\s*anywhere/);
assert.match(readability, /@media \(max-width:\s*1480px\)/);
assert.match(readability, /\.workflow-step strong\s*\{[\s\S]*display:\s*none\s*!important/);
assert.match(readability, /@media \(max-width:\s*720px\)/);
assert.match(readability, /button:not\(\.icon-button\)\s*\{[\s\S]*white-space:\s*normal/);

const readabilityIndex = main.indexOf('import "./readability.css";');
const completionIndex = main.indexOf('import "./workflow-completion.css";');
const conversationFixIndex = main.indexOf('import "./conversation-mode-fixes.css";');
assert.ok(readabilityIndex > completionIndex, "readability overrides must load after workflow completion styles");
assert.ok(readabilityIndex > conversationFixIndex, "readability overrides must be the final UI normalization layer");

assert.match(dataTable, /<th key=\{column\} title=\{column\}>/);
assert.match(dataTable, /title=\{String\(displayValue\)\}/);

console.log("readability and responsive regression checks passed");
