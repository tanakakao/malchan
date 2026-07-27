import React, { useEffect, useRef, useState } from "react";

export default function PlotlyFigure({ figure, className = "plotly-figure" }) {
  const containerRef = useRef(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    let plotly = null;
    const container = containerRef.current;
    if (!container || !figure) return undefined;

    setError("");
    import("plotly.js-dist-min")
      .then((module) => {
        if (!active || !containerRef.current) return;
        plotly = module.default || module;
        const layout = {
          ...(figure.layout || {}),
          autosize: true,
          width: undefined,
        };
        plotly.react(
          containerRef.current,
          figure.data || [],
          layout,
          {
            responsive: true,
            displaylogo: false,
            scrollZoom: true,
            toImageButtonOptions: {
              format: "png",
              filename: "malchan-visualization",
              scale: 2,
            },
          },
        );
      })
      .catch((reason) => {
        if (active) setError(reason?.message || String(reason));
      });

    return () => {
      active = false;
      if (plotly && container) plotly.purge(container);
    };
  }, [figure]);

  if (!figure) return <p className="empty-state">表示できる図がありません。</p>;
  if (error) return <p className="xai-error">Plotly描画エラー: {error}</p>;
  return <div ref={containerRef} className={className} />;
}
