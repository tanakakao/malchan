const BOTTOM_LEGEND_MARGIN = 130;

/**
 * Keep Plotly legends below the plotting area so the graph retains its full width.
 *
 * This mirrors bochan's shared Plotly layout contract while preserving each
 * visualization's axes, annotations, colors, and other backend-provided settings.
 */
export function withBottomLegend(layout = {}) {
  const source = layout || {};
  const sourceBottomMargin = typeof source.margin?.b === "number"
    ? source.margin.b
    : 0;

  return {
    ...source,
    autosize: true,
    width: undefined,
    margin: {
      ...(source.margin || {}),
      b: Math.max(sourceBottomMargin, BOTTOM_LEGEND_MARGIN),
    },
    legend: {
      ...(source.legend || {}),
      orientation: "h",
      x: 0.5,
      xanchor: "center",
      y: -0.18,
      yanchor: "top",
      traceorder: "normal",
    },
  };
}
