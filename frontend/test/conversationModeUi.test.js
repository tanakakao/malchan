import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const fixes = readFileSync(new URL("../src/conversation-mode-fixes.css", import.meta.url), "utf8");
const iconSource = readFileSync(
  new URL("../src/components/ConversationIcon.jsx", import.meta.url),
  "utf8",
);
const mainSource = readFileSync(new URL("../src/main.jsx", import.meta.url), "utf8");

test("conversation launcher uses the full left-rail width", () => {
  assert.match(fixes, /\.conversation-launcher\s*\{[\s\S]*?width:\s*100%/);
  assert.match(fixes, /\.conversation-launcher\s*\{[\s\S]*?margin:\s*0 0 16px/);
  assert.match(fixes, /\.conversation-launcher\s*\{[\s\S]*?flex:\s*0 0 auto/);
});

test("conversation cards cannot collapse inside the scrollable message column", () => {
  assert.match(
    fixes,
    /\.conversation-messages > \.conversation-message,[\s\S]*?\.conversation-messages > \.conversation-action-card\s*\{[\s\S]*?flex:\s*0 0 auto/,
  );
  assert.match(fixes, /\.conversation-result-card\s*\{[\s\S]*?overflow:\s*visible/);
});

test("all assistant icons share one resolved image request", () => {
  assert.match(iconSource, /let resolvedIconUrl;/);
  assert.match(iconSource, /let iconResolutionPromise;/);
  assert.match(iconSource, /function resolveConversationIcon\(\)/);
  assert.doesNotMatch(iconSource, /useState\(0\)/);
});

test("conversation fixes load after readability overrides", () => {
  const readabilityIndex = mainSource.indexOf('import "./readability.css";');
  const fixesIndex = mainSource.indexOf('import "./conversation-mode-fixes.css";');
  assert.ok(readabilityIndex >= 0);
  assert.ok(fixesIndex > readabilityIndex);
});
