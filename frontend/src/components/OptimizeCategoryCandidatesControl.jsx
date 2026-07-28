import React, { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { setInverseCategoryCandidatesOverride } from "../api";
import { uniqueValues } from "../data";
import { useWorkbench } from "../context/WorkbenchContext";
import "../optimize-category-candidates.css";

const candidateSelectionsByModel = new Map();

function categoryValues(rows, column) {
  return uniqueValues(rows, column, Math.max(rows.length, 1));
}

function originalValues(available, selected) {
  const selectedKeys = new Set(selected.map(String));
  return available.filter((value) => selectedKeys.has(String(value)));
}

function normalizedSelections(rows, columns, saved = {}) {
  return Object.fromEntries(columns.map((column) => {
    const available = categoryValues(rows, column);
    const restored = Array.isArray(saved[column])
      ? originalValues(available, saved[column])
      : [];
    return [column, restored.length ? restored : available];
  }));
}

function sameSet(left, right) {
  if (left.size !== right.size) return false;
  return [...left].every((value) => right.has(value));
}

function sameHosts(left, right) {
  const leftKeys = Object.keys(left);
  const rightKeys = Object.keys(right);
  return leftKeys.length === rightKeys.length
    && leftKeys.every((key) => left[key] === right[key]);
}

function CategoryMultiSelect({ column, available, selected, disabled, onChange }) {
  const rootRef = useRef(null);
  const [open, setOpen] = useState(false);
  const [filterText, setFilterText] = useState("");
  const selectedKeys = useMemo(() => new Set(selected.map(String)), [selected]);
  const filteredValues = useMemo(() => {
    const keyword = filterText.trim().toLocaleLowerCase();
    if (!keyword) return available;
    return available.filter((value) => String(value).toLocaleLowerCase().includes(keyword));
  }, [available, filterText]);

  useEffect(() => {
    if (disabled) setOpen(false);
  }, [disabled]);

  useEffect(() => {
    if (!open) return undefined;

    const closeFromOutside = (event) => {
      if (!rootRef.current?.contains(event.target)) setOpen(false);
    };
    const closeFromKeyboard = (event) => {
      if (event.key === "Escape") setOpen(false);
    };

    document.addEventListener("pointerdown", closeFromOutside);
    document.addEventListener("keydown", closeFromKeyboard);
    return () => {
      document.removeEventListener("pointerdown", closeFromOutside);
      document.removeEventListener("keydown", closeFromKeyboard);
    };
  }, [open]);

  function toggle(value) {
    const key = String(value);
    if (selectedKeys.has(key)) {
      if (selected.length <= 1) return;
      onChange(selected.filter((item) => String(item) !== key));
      return;
    }
    onChange(available.filter(
      (item) => selectedKeys.has(String(item)) || String(item) === key,
    ));
  }

  return (
    <div
      ref={rootRef}
      className={`category-candidate-select ${open ? "open" : ""}`}
    >
      <button
        type="button"
        className="category-candidate-trigger"
        disabled={disabled}
        aria-expanded={open}
        onClick={() => setOpen((value) => !value)}
      >
        <span>{disabled ? "固定値を使用" : `${selected.length} / ${available.length} 候補`}</span>
        <span aria-hidden="true">⌄</span>
      </button>
      {open && !disabled && (
        <div className="category-candidate-menu">
          <div className="category-candidate-menu-head">
            <strong>{column} の探索候補</strong>
            <button type="button" onClick={() => onChange([...available])}>全選択</button>
          </div>
          {available.length > 8 && (
            <input
              className="category-candidate-filter"
              type="search"
              value={filterText}
              placeholder="候補を検索"
              onChange={(event) => setFilterText(event.target.value)}
            />
          )}
          <div className="category-candidate-options">
            {filteredValues.map((value) => {
              const checked = selectedKeys.has(String(value));
              return (
                <label key={String(value)}>
                  <input
                    type="checkbox"
                    checked={checked}
                    disabled={checked && selected.length <= 1}
                    onChange={() => toggle(value)}
                  />
                  <span title={String(value)}>{String(value)}</span>
                </label>
              );
            })}
            {!filteredValues.length && (
              <span className="category-candidate-empty">一致する候補がありません。</span>
            )}
          </div>
          <p>探索候補は1つ以上必要です。</p>
        </div>
      )}
    </div>
  );
}

export default function OptimizeCategoryCandidatesControl() {
  const { step, rows, catFeatures, modelInfo } = useWorkbench();
  const modelKey = modelInfo?.model_id || "unregistered";
  const candidateKey = useMemo(
    () => catFeatures.map((column) => (
      `${column}:${categoryValues(rows, column).map(String).join("\u0001")}`
    )).join("\u0002"),
    [rows, catFeatures],
  );
  const [hosts, setHosts] = useState({});
  const [fixedColumns, setFixedColumns] = useState(new Set());
  const [selections, setSelections] = useState(() => normalizedSelections(
    rows,
    catFeatures,
    candidateSelectionsByModel.get(modelKey),
  ));

  useEffect(() => {
    setSelections(normalizedSelections(
      rows,
      catFeatures,
      candidateSelectionsByModel.get(modelKey),
    ));
  }, [modelKey, candidateKey]);

  useEffect(() => {
    candidateSelectionsByModel.set(modelKey, selections);
  }, [modelKey, selections]);

  useEffect(() => {
    if (step !== "optimize" || !catFeatures.length) {
      setInverseCategoryCandidatesOverride(null);
      return undefined;
    }
    setInverseCategoryCandidatesOverride(selections);
    return () => setInverseCategoryCandidatesOverride(null);
  }, [step, catFeatures, selections]);

  useEffect(() => {
    if (step !== "optimize" || !catFeatures.length) {
      setHosts({});
      return undefined;
    }

    const content = document.querySelector(".content-inner") || document.body;

    const connect = () => {
      const panel = document.querySelector(".optimize-variable-panel");
      const tableRows = panel?.querySelectorAll(".optimize-variable-table tbody > tr") || [];
      const nextHosts = {};
      const nextFixed = new Set();

      tableRows.forEach((row) => {
        const column = row.querySelector("td:first-child strong")?.textContent?.trim();
        if (!column || !catFeatures.includes(column)) return;

        const cells = row.querySelectorAll(":scope > td");
        const lowerLimitCell = cells[2];
        if (!lowerLimitCell) return;

        row.classList.add("categorical-candidate-row");
        lowerLimitCell.classList.add("category-candidate-cell");

        let host = lowerLimitCell.querySelector(":scope > .category-candidate-inline-host");
        if (!host) {
          host = document.createElement("div");
          host.className = "category-candidate-inline-host";
          host.dataset.column = column;
          lowerLimitCell.append(host);
        }
        nextHosts[column] = host;
        if (row.classList.contains("fixed-variable-row")) nextFixed.add(column);
      });

      panel?.querySelectorAll(".category-candidate-inline-host").forEach((host) => {
        const column = host.dataset.column;
        if (column && nextHosts[column] === host) return;
        const cell = host.parentElement;
        const row = host.closest("tr");
        host.remove();
        cell?.classList.remove("category-candidate-cell");
        row?.classList.remove("categorical-candidate-row");
      });

      setHosts((current) => (sameHosts(current, nextHosts) ? current : nextHosts));
      setFixedColumns((current) => (sameSet(current, nextFixed) ? current : nextFixed));
    };

    connect();
    const observer = new MutationObserver(connect);
    observer.observe(content, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ["class"],
    });

    return () => {
      observer.disconnect();
      document.querySelectorAll(".category-candidate-inline-host").forEach((host) => {
        const cell = host.parentElement;
        const row = host.closest("tr");
        host.remove();
        cell?.classList.remove("category-candidate-cell");
        row?.classList.remove("categorical-candidate-row");
      });
      setHosts({});
    };
  }, [step, candidateKey, catFeatures]);

  if (!catFeatures.length) return null;

  return (
    <>
      {catFeatures.map((column) => {
        const host = hosts[column];
        if (!host) return null;
        const available = categoryValues(rows, column);
        const selected = selections[column] || available;
        const fixed = fixedColumns.has(column);
        return createPortal(
          <div className={`category-candidate-inline-control ${fixed ? "fixed" : ""}`}>
            <span className="category-candidate-inline-label">
              {fixed ? "固定値を優先" : "探索候補"}
            </span>
            <CategoryMultiSelect
              column={column}
              available={available}
              selected={selected}
              disabled={fixed || available.length === 0}
              onChange={(next) => setSelections((current) => ({
                ...current,
                [column]: next,
              }))}
            />
          </div>,
          host,
        );
      })}
    </>
  );
}
