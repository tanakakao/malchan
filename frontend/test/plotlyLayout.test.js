import assert from "node:assert/strict";
import test from "node:test";

import { withBottomLegend } from "../src/plotlyLayout.js";

test("places Plotly legends below the chart without mutating source layout", () => {
  const source = {
    margin: { l: 60, r: 40, b: 48 },
    legend: { orientation: "v", x: 1.02 },
    xaxis: { title: "Feature" },
  };

  const result = withBottomLegend(source);

  assert.deepEqual(source, {
    margin: { l: 60, r: 40, b: 48 },
    legend: { orientation: "v", x: 1.02 },
    xaxis: { title: "Feature" },
  });
  assert.equal(result.autosize, true);
  assert.equal(result.width, undefined);
  assert.deepEqual(result.margin, { l: 60, r: 40, b: 130 });
  assert.deepEqual(result.legend, {
    orientation: "h",
    x: 0.5,
    xanchor: "center",
    y: -0.18,
    yanchor: "top",
    traceorder: "normal",
  });
  assert.deepEqual(result.xaxis, { title: "Feature" });
});

test("preserves a larger backend-provided bottom margin", () => {
  const result = withBottomLegend({ margin: { b: 180 } });

  assert.equal(result.margin.b, 180);
});
