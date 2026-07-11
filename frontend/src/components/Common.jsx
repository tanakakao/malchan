import React from "react";

export function SectionHeader({ step, title, text, action }) {
  return (
    <div className="section-header">
      <div>
        <span className="eyebrow">STEP {step}</span>
        <h2>{title}</h2>
        <p>{text}</p>
      </div>
      {action}
    </div>
  );
}

export function CheckboxList({ values, selected, disabled = [], onChange }) {
  return (
    <div className="checklist">
      {values.map((value) => (
        <label key={value}>
          <input
            type="checkbox"
            checked={selected.includes(value)}
            disabled={disabled.includes(value)}
            onChange={(event) =>
              onChange(
                event.target.checked
                  ? [...selected, value]
                  : selected.filter((item) => item !== value),
              )
            }
          />
          {value}
        </label>
      ))}
    </div>
  );
}

export function Field({ label, children, className = "" }) {
  return (
    <label className={className}>
      {label}
      {children}
    </label>
  );
}
