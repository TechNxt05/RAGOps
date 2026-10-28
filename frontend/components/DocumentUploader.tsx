"use client";

import { useRef, useState } from "react";
import { toast } from "sonner";
import { CheckCircle2, CircleAlert, Copy, Loader2, UploadCloud, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { errorMessage, uploadDocuments, type UploadResult } from "@/lib/api";

const ACCEPTED = ["pdf", "docx", "txt", "md", "markdown", "csv", "tsv", "json", "jsonl", "xml", "yaml", "yml", "html", "htm", "log", "rst", "sql"];
const MAX_MB = 25;
const BATCH_SIZE = 5; // files per request: keeps each request small and progress meaningful

type Row = { name: string; size: number; state: "waiting" | "uploading" | UploadResult["status"] | "error"; note?: string };

function ext(name: string) {
  return name.includes(".") ? name.split(".").pop()!.toLowerCase() : "";
}

export function DocumentUploader({ projectId, onUploaded }: { projectId: number; onUploaded: () => void }) {
  const input = useRef<HTMLInputElement>(null);
  const [rows, setRows] = useState<Row[]>([]);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState(0);
  const [dragging, setDragging] = useState(false);

  const start = async (fileList: FileList | File[]) => {
    const files = Array.from(fileList);
    if (!files.length || busy) return;
    const initial: Row[] = files.map((f) => {
      if (!ACCEPTED.includes(ext(f.name))) return { name: f.name, size: f.size, state: "rejected", note: `.${ext(f.name) || "?"} is not supported` };
      if (f.size > MAX_MB * 1024 * 1024) return { name: f.name, size: f.size, state: "rejected", note: `larger than ${MAX_MB} MB` };
      if (f.size === 0) return { name: f.name, size: f.size, state: "rejected", note: "empty file" };
      return { name: f.name, size: f.size, state: "waiting" };
    });
    setRows(initial);
    const valid = files.filter((_, i) => initial[i].state === "waiting");
    if (!valid.length) return;

    setBusy(true);
    setProgress(0);
    let done = 0;
    let queued = 0;
    for (let i = 0; i < valid.length; i += BATCH_SIZE) {
      const batch = valid.slice(i, i + BATCH_SIZE);
      const names = new Set(batch.map((f) => f.name));
      setRows((r) => r.map((row) => (names.has(row.name) && row.state === "waiting" ? { ...row, state: "uploading" } : row)));
      try {
        const res = await uploadDocuments(projectId, batch, (frac) =>
          setProgress((done + frac * batch.length) / valid.length)
        );
        queued += res.results.filter((x) => x.status === "queued").length;
        setRows((r) =>
          r.map((row) => {
            const hit = res.results.find((x) => x.filename === row.name);
            return hit && row.state === "uploading" ? { ...row, state: hit.status, note: hit.error || hit.message } : row;
          })
        );
      } catch (e) {
        const msg = errorMessage(e, "Upload failed");
        setRows((r) => r.map((row) => (names.has(row.name) && row.state === "uploading" ? { ...row, state: "error", note: msg } : row)));
      }
      done += batch.length;
      setProgress(done / valid.length);
    }
    setBusy(false);
    if (queued) {
      toast.success(`${queued} file${queued > 1 ? "s" : ""} queued for processing`);
      onUploaded();
    }
    if (input.current) input.current.value = ""; // allow re-selecting the same file
  };

  return (
    <div className="space-y-3">
      <div
        role="button"
        tabIndex={0}
        onClick={() => input.current?.click()}
        onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && input.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => { e.preventDefault(); setDragging(false); void start(e.dataTransfer.files); }}
        className={cn(
          "flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed px-6 py-8 text-center transition",
          dragging ? "border-indigo-500 bg-indigo-50/60 dark:bg-indigo-950/30" : "border-border hover:border-indigo-400",
          busy && "pointer-events-none opacity-70"
        )}
      >
        {busy ? <Loader2 className="mb-2 h-8 w-8 animate-spin text-indigo-500" /> : <UploadCloud className="mb-2 h-8 w-8 text-indigo-500" />}
        <p className="text-sm font-medium">{busy ? `Uploading… ${Math.round(progress * 100)}%` : "Drop files here or click to choose"}</p>
        <p className="mt-1 text-xs text-muted-foreground">PDF, DOCX, TXT, MD, CSV, JSON, HTML… up to {MAX_MB} MB each, many at once</p>
        <input
          ref={input}
          type="file"
          multiple
          className="hidden"
          accept={ACCEPTED.map((e) => `.${e}`).join(",")}
          onChange={(e) => e.target.files && void start(e.target.files)}
        />
      </div>

      {busy && (
        <div className="h-1.5 w-full overflow-hidden rounded bg-muted">
          <div className="h-full bg-indigo-500 transition-all" style={{ width: `${Math.round(progress * 100)}%` }} />
        </div>
      )}

      {rows.length > 0 && (
        <ul className="divide-y rounded-lg border text-sm">
          {rows.map((r, i) => (
            <li key={`${r.name}-${i}`} className="flex items-center gap-2 px-3 py-2">
              {r.state === "queued" && <CheckCircle2 className="h-4 w-4 shrink-0 text-emerald-500" />}
              {r.state === "duplicate" && <Copy className="h-4 w-4 shrink-0 text-sky-500" />}
              {(r.state === "rejected" || r.state === "error") && <CircleAlert className="h-4 w-4 shrink-0 text-red-500" />}
              {(r.state === "waiting" || r.state === "uploading") && <Loader2 className="h-4 w-4 shrink-0 animate-spin text-muted-foreground" />}
              <span className="min-w-0 flex-1 truncate">{r.name}</span>
              <span className="shrink-0 text-xs text-muted-foreground">
                {r.state === "queued" ? "queued: processing in background" : r.note || r.state}
              </span>
            </li>
          ))}
          {!busy && (
            <li className="flex justify-end px-3 py-1.5">
              <Button size="sm" variant="ghost" onClick={() => setRows([])}><X className="mr-1 h-3.5 w-3.5" />Clear</Button>
            </li>
          )}
        </ul>
      )}
    </div>
  );
}
