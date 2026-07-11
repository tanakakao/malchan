import React from "react";

export default function MetricCard({ icon, label, value, note, warning = false }) {
  return (
    <div className={`metric ${warning ? "warning" : ""}`}>
      <div className="metric-icon">{icon}</div>
      <div><span>{label}</span><strong>{value}</strong><small>{note}</small></div>
    </div>
  );
}
