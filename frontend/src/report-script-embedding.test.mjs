import assert from "node:assert/strict";

import {
  embeddedScriptTag,
  sourceToScriptDataUrl,
} from "./report-script-embedding.js";

const dangerousSource = 'window.marker = "</script><article>duplicated report</article>";';
const dataUrl = sourceToScriptDataUrl(dangerousSource);
const tag = embeddedScriptTag(dangerousSource);

assert.match(dataUrl, /^data:text\/javascript;charset=utf-8;base64,/);
assert.doesNotMatch(tag, /window\.marker/);
assert.doesNotMatch(tag, /<article>/);
assert.equal((tag.match(/<script/g) || []).length, 1);
assert.equal((tag.match(/<\/script>/g) || []).length, 1);

const encoded = dataUrl.split(",", 2)[1];
assert.equal(Buffer.from(encoded, "base64").toString("utf8"), dangerousSource);
assert.equal(embeddedScriptTag(""), "");

console.log("report script embedding checks passed");
