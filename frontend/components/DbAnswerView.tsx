"use client";

import { useState } from "react";
import type { DbAnswer } from "@/lib/api";

/** Answer from a connected database: summary, the exact SQL that ran, and the rows. */
export function DbAnswerView({ answer }: { answer: DbAnswer }) {
  const [showSql, setShowSql] = useState(false);
  const rows = answer.rows.slice(0, 50);
  return (
    <div className="space-y-2 text-sm">
      <p className="whitespace-pre-wrap">{answer.answer}</p>
      <div className="flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
        <span>{answer.row_count} row{answer.row_count === 1 ? "" : "s"}{answer.truncated ? " (truncated)" : ""}</span>
        <span>{Math.round(answer.latency_ms)} ms</span>
        <span>from {answer.database}</span>
        <button type="button" className="underline" onClick={() => setShowSql((s) => !s)}>{showSql ? "Hide" : "Show"} SQL</button>
      </div>
      {showSql && <pre className="overflow-x-auto rounded bg-muted p-2 text-xs">{answer.sql}</pre>}
      {rows.length > 0 && (
        <div className="max-h-72 overflow-auto rounded border">
          <table className="w-full text-xs">
            <thead className="sticky top-0 bg-muted">
              <tr>{answer.columns.map((c) => <th key={c} className="px-2 py-1 text-left font-medium">{c}</th>)}</tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={i} className="border-t">
                  {r.map((v, j) => <td key={j} className="px-2 py-1">{v === null ? "∅" : String(v)}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
