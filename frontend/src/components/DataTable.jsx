import React, { useMemo, useState } from "react";
import { formatNumber } from "../data";

export default function DataTable({ rows, columns, pageSize = 20 }) {
  const [page, setPage] = useState(0);
  const pageCount = Math.max(1, Math.ceil(rows.length / pageSize));
  const normalizedPage = Math.min(page, pageCount - 1);
  const visibleRows = useMemo(
    () => rows.slice(normalizedPage * pageSize, (normalizedPage + 1) * pageSize),
    [rows, normalizedPage, pageSize],
  );

  return (
    <div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>#</th>
              {columns.map((column) => <th key={column}>{column}</th>)}
            </tr>
          </thead>
          <tbody>
            {visibleRows.map((row, rowIndex) => (
              <tr key={`${normalizedPage}-${rowIndex}`}>
                <td>{normalizedPage * pageSize + rowIndex + 1}</td>
                {columns.map((column) => (
                  <td key={column} className={row[column] === null ? "missing-cell" : ""}>
                    {row[column] === null ? "missing" : formatNumber(row[column])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="pagination">
        <button className="secondary icon-button" onClick={() => setPage(Math.max(0, normalizedPage - 1))} disabled={normalizedPage === 0}>‹</button>
        <span>{normalizedPage + 1} / {pageCount}</span>
        <button className="secondary icon-button" onClick={() => setPage(Math.min(pageCount - 1, normalizedPage + 1))} disabled={normalizedPage >= pageCount - 1}>›</button>
      </div>
    </div>
  );
}
