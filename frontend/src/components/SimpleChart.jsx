import React from "react";

const WIDTH = 900;
const HEIGHT = 480;
const MARGIN = { left: 74, right: 24, top: 48, bottom: 64 };

function finite(values) {
  return values.map(Number).filter(Number.isFinite);
}

function extent(values) {
  const cleaned = finite(values);
  if (!cleaned.length) return [0, 1];
  const min = Math.min(...cleaned);
  const max = Math.max(...cleaned);
  return max > min ? [min, max] : [min - 0.5, max + 0.5];
}

function scale(value, domain, range) {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  return r0 + ((Number(value) - d0) / (d1 - d0 || 1)) * (r1 - r0);
}

function ticks(domain, count = 5) {
  const [min, max] = domain;
  return Array.from({ length: count }, (_, index) => min + ((max - min) * index) / (count - 1));
}

function fmt(value) {
  return Number(value).toLocaleString("ja-JP", { maximumFractionDigits: 3 });
}

function Axes({ xDomain, yDomain, xTitle, yTitle }) {
  const x0 = MARGIN.left;
  const x1 = WIDTH - MARGIN.right;
  const y0 = HEIGHT - MARGIN.bottom;
  const y1 = MARGIN.top;
  return (
    <g className="svg-axes">
      <line x1={x0} y1={y0} x2={x1} y2={y0} />
      <line x1={x0} y1={y0} x2={x0} y2={y1} />
      {ticks(xDomain).map((value) => {
        const x = scale(value, xDomain, [x0, x1]);
        return <g key={`x-${value}`}><line x1={x} y1={y0} x2={x} y2={y0 + 6} /><text x={x} y={y0 + 22} textAnchor="middle">{fmt(value)}</text></g>;
      })}
      {ticks(yDomain).map((value) => {
        const y = scale(value, yDomain, [y0, y1]);
        return <g key={`y-${value}`}><line x1={x0 - 6} y1={y} x2={x0} y2={y} /><text x={x0 - 10} y={y + 4} textAnchor="end">{fmt(value)}</text></g>;
      })}
      <text x={(x0 + x1) / 2} y={HEIGHT - 18} textAnchor="middle" className="axis-title">{xTitle}</text>
      <text x="18" y={(y0 + y1) / 2} textAnchor="middle" className="axis-title" transform={`rotate(-90 18 ${(y0 + y1) / 2})`}>{yTitle}</text>
    </g>
  );
}

function ScatterChart({ traces, layout }) {
  const points = traces.flatMap((trace) => (trace.x || []).map((x, index) => [Number(x), Number(trace.y?.[index])])).filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y));
  const xDomain = extent(points.map(([x]) => x));
  const yDomain = extent(points.map(([, y]) => y));
  const xRange = [MARGIN.left, WIDTH - MARGIN.right];
  const yRange = [HEIGHT - MARGIN.bottom, MARGIN.top];
  return (
    <>
      <Axes xDomain={xDomain} yDomain={yDomain} xTitle={layout?.xaxis?.title || "X"} yTitle={layout?.yaxis?.title || "Y"} />
      {traces.map((trace, traceIndex) => {
        const color = trace.marker?.color || trace.line?.color || (traceIndex ? "#50d09c" : "#6d8cff");
        const coordinates = (trace.x || []).map((x, index) => [Number(x), Number(trace.y?.[index])]).filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y));
        const linePoints = coordinates.map(([x, y]) => `${scale(x, xDomain, xRange)},${scale(y, yDomain, yRange)}`).join(" ");
        return (
          <g key={trace.name || traceIndex}>
            {String(trace.mode || "").includes("lines") && <polyline points={linePoints} fill="none" stroke={color} strokeWidth="2" strokeDasharray={trace.line?.dash === "dash" ? "8 6" : undefined} />}
            {String(trace.mode || "markers").includes("markers") && coordinates.map(([x, y], index) => <circle key={index} cx={scale(x, xDomain, xRange)} cy={scale(y, yDomain, yRange)} r={trace.marker?.size ? Math.max(3, Number(trace.marker.size) / 2) : 4} fill={color} opacity={trace.marker?.opacity || 0.75}><title>{`${fmt(x)}, ${fmt(y)}`}</title></circle>)}
          </g>
        );
      })}
    </>
  );
}

function HistogramChart({ trace, layout }) {
  const values = finite(trace.x || []);
  const domain = extent(values);
  const binCount = Math.max(5, Math.min(24, Math.ceil(Math.sqrt(values.length || 1))));
  const width = (domain[1] - domain[0]) / binCount;
  const bins = Array.from({ length: binCount }, (_, index) => ({ start: domain[0] + index * width, count: 0 }));
  values.forEach((value) => {
    const index = Math.min(binCount - 1, Math.max(0, Math.floor((value - domain[0]) / width)));
    bins[index].count += 1;
  });
  const yDomain = [0, Math.max(1, ...bins.map((bin) => bin.count))];
  const xRange = [MARGIN.left, WIDTH - MARGIN.right];
  const yRange = [HEIGHT - MARGIN.bottom, MARGIN.top];
  return (
    <>
      <Axes xDomain={domain} yDomain={yDomain} xTitle={layout?.xaxis?.title || "Value"} yTitle={layout?.yaxis?.title || "Count"} />
      {bins.map((bin, index) => {
        const x = scale(bin.start, domain, xRange);
        const next = scale(bin.start + width, domain, xRange);
        const y = scale(bin.count, yDomain, yRange);
        return <rect key={index} x={x + 1} y={y} width={Math.max(1, next - x - 2)} height={HEIGHT - MARGIN.bottom - y} fill={trace.marker?.color || "#6d8cff"} opacity="0.82"><title>{`${fmt(bin.start)}–${fmt(bin.start + width)}: ${bin.count}`}</title></rect>;
      })}
    </>
  );
}

function heatColor(value, domain) {
  const numericValue = Number(value);
  if (!Number.isFinite(numericValue)) return "transparent";
  const [min, max] = domain;
  const ratio = max === min ? 0.5 : (numericValue - min) / (max - min);
  const clipped = Math.max(0, Math.min(1, ratio));
  const hue = 220 - clipped * 210;
  return `hsl(${hue} 78% 55%)`;
}

function HeatmapChart({ trace, layout }) {
  const labelsX = trace.x || [];
  const labelsY = trace.y || [];
  const matrix = trace.z || [];
  const values = matrix.flatMap((row) => finite(row || []));
  const colorDomain = values.length
    ? [Math.min(...values), Math.max(...values)]
    : [0, 1];
  const left = 150;
  const top = 56;
  const right = 35;
  const bottom = 126;
  const cellWidth = (WIDTH - left - right) / Math.max(1, labelsX.length);
  const cellHeight = (HEIGHT - top - bottom) / Math.max(1, labelsY.length);
  return (
    <g>
      {labelsX.map((label, index) => <text key={`${label}-${index}`} x={left + index * cellWidth + cellWidth / 2} y={HEIGHT - bottom + 18} textAnchor="end" transform={`rotate(-45 ${left + index * cellWidth + cellWidth / 2} ${HEIGHT - bottom + 18})`} className="heat-label">{Number.isFinite(Number(label)) ? fmt(label) : label}</text>)}
      {labelsY.map((label, index) => <text key={`${label}-${index}`} x={left - 10} y={top + index * cellHeight + cellHeight / 2 + 4} textAnchor="end" className="heat-label">{Number.isFinite(Number(label)) ? fmt(label) : label}</text>)}
      {matrix.flatMap((row, rowIndex) => (row || []).map((value, columnIndex) => (
        <g key={`${rowIndex}-${columnIndex}`}>
          <rect x={left + columnIndex * cellWidth} y={top + rowIndex * cellHeight} width={cellWidth} height={cellHeight} fill={heatColor(value, colorDomain)} stroke="rgba(255,255,255,.24)">
            <title>{`${labelsX[columnIndex]}, ${labelsY[rowIndex]}: ${Number.isFinite(Number(value)) ? fmt(value) : "—"}`}</title>
          </rect>
          {cellWidth > 50 && cellHeight > 25 && Number.isFinite(Number(value)) && <text x={left + columnIndex * cellWidth + cellWidth / 2} y={top + rowIndex * cellHeight + cellHeight / 2 + 4} textAnchor="middle" className="heat-value">{fmt(value)}</text>}
        </g>
      )))}
      <text x={(left + WIDTH - right) / 2} y={HEIGHT - 18} textAnchor="middle" className="axis-title">{layout?.xaxis?.title || "X"}</text>
      <text x="24" y={(top + HEIGHT - bottom) / 2} textAnchor="middle" className="axis-title" transform={`rotate(-90 24 ${(top + HEIGHT - bottom) / 2})`}>{layout?.yaxis?.title || "Y"}</text>
      <text x={WIDTH - right} y={top - 15} textAnchor="end" className="heat-range">{`${fmt(colorDomain[0])} – ${fmt(colorDomain[1])}`}</text>
    </g>
  );
}

export default function SimpleChart({ data = [], layout = {}, className = "chart" }) {
  const trace = data[0];
  return (
    <div className={`simple-chart ${className}`}>
      <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-label={layout.title || "chart"}>
        <text x={WIDTH / 2} y="27" textAnchor="middle" className="chart-title">{layout.title || ""}</text>
        {!trace && <text x={WIDTH / 2} y={HEIGHT / 2} textAnchor="middle" className="empty-chart-label">データを選択してください</text>}
        {trace?.type === "histogram" && <HistogramChart trace={trace} layout={layout} />}
        {trace?.type === "heatmap" && <HeatmapChart trace={trace} layout={layout} />}
        {trace?.type === "scatter" && <ScatterChart traces={data} layout={layout} />}
      </svg>
    </div>
  );
}
