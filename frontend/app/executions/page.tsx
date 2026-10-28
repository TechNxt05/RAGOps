"use client";

import React, { useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import {
  Activity, AlertTriangle, ArrowLeft, ArrowRight, CheckCircle2, Clock, Cpu, Database, FileSearch,
  Filter, GitCommit, Layers, RefreshCw, ShieldCheck, XCircle, Zap,
} from "lucide-react";
import { errorMessage, getExecutions, getProjects, type Execution, type Project } from "@/lib/api";
import { useAuth } from "@/lib/auth-context";

type Stage = { key: string; status: string; duration_ms?: number; start_time?: number; metadata?: Record<string, unknown>; error_message?: string };

const STAGE_ICONS: Record<string, React.ReactNode> = {
  query_understanding: <FileSearch className="h-4 w-4 text-blue-400" />,
  retrieval: <Database className="h-4 w-4 text-cyan-400" />,
  confidence_gate: <ShieldCheck className="h-4 w-4 text-orange-400" />,
  self_correction_triggered: <RefreshCw className="h-4 w-4 text-purple-400" />,
  absence_prover_retry: <Filter className="h-4 w-4 text-amber-400" />,
  llm_generation: <Zap className="h-4 w-4 text-emerald-400" />,
  semantic_cache: <Layers className="h-4 w-4 text-sky-400" />,
  reflective_generation: <Activity className="h-4 w-4 text-teal-400" />,
};

function StatusBadge({ status }: { status: string }) {
  if (status === "success") return <span className="flex items-center gap-1 rounded-full border border-emerald-500/20 bg-emerald-500/10 px-2 py-0.5 text-xs font-semibold text-emerald-400"><CheckCircle2 className="h-3 w-3" /> PASS</span>;
  if (status === "failed" || status === "refused") return <span className="flex items-center gap-1 rounded-full border border-rose-500/20 bg-rose-500/10 px-2 py-0.5 text-xs font-semibold text-rose-400"><XCircle className="h-3 w-3" /> {status.toUpperCase()}</span>;
  if (status === "skipped") return <span className="flex items-center gap-1 rounded-full border border-slate-500/20 bg-slate-500/10 px-2 py-0.5 text-xs font-semibold text-slate-400"><Clock className="h-3 w-3" /> SKIPPED</span>;
  return <span className="flex items-center gap-1 rounded-full border border-amber-500/20 bg-amber-500/10 px-2 py-0.5 text-xs font-semibold text-amber-400"><AlertTriangle className="h-3 w-3" /> {status.toUpperCase()}</span>;
}

const ms = (v?: number | null) => (v == null ? "—" : `${Math.round(v).toLocaleString()} ms`);

export default function ExecutionsPage() {
  const { user, isLoading } = useAuth();
  const [projects, setProjects] = useState<Project[]>([]);
  const [projectId, setProjectId] = useState<number | null>(null);
  const [rows, setRows] = useState<Execution[]>([]);
  const [selected, setSelected] = useState<Execution | null>(null);
  const [stage, setStage] = useState<Stage | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!user) return;
    getProjects().then((p) => {
      setProjects(p);
      if (p.length) setProjectId(p[0].id);
    }).catch((e) => setError(errorMessage(e, "Could not load projects")));
  }, [user]);

  const load = useCallback(async () => {
    if (projectId == null) return;
    try {
      const data = await getExecutions(projectId);
      setRows(data);
      setSelected((cur) => data.find((d) => d.id === cur?.id) ?? data[0] ?? null);
      setError(null);
    } catch (e) {
      setError(errorMessage(e, "Could not load executions (admin only)"));
    }
  }, [projectId]);

  useEffect(() => {
    if (projectId == null) return;
    let cancelled = false;
    getExecutions(projectId)
      .then((data) => {
        if (cancelled) return;
        setRows(data);
        setSelected(data[0] ?? null);
        setError(null);
      })
      .catch((e) => { if (!cancelled) setError(errorMessage(e, "Could not load executions (admin only)")); });
    return () => { cancelled = true; };
  }, [projectId]);

  const stages: Stage[] = useMemo(() => {
    const raw = (selected?.trace?.stages ?? {}) as Record<string, Omit<Stage, "key">>;
    return Object.entries(raw)
      .map(([key, v]) => ({ key, ...v }))
      .sort((a, b) => (a.start_time ?? Number.MAX_SAFE_INTEGER) - (b.start_time ?? Number.MAX_SAFE_INTEGER));
  }, [selected]);

  const timings = (selected?.trace?.timings_ms ?? null) as Record<string, number> | null;
  const timingTotal = timings ? Object.values(timings).reduce((a, b) => a + b, 0) : 0;
  const p95 = useMemo(() => {
    const lat = rows.map((r) => r.latency_ms ?? 0).filter(Boolean).sort((a, b) => a - b);
    return lat.length ? lat[Math.min(lat.length - 1, Math.floor(lat.length * 0.95))] : null;
  }, [rows]);

  if (isLoading) return null;

  return (
    <div className="min-h-screen space-y-6 bg-slate-950 p-6 text-slate-100">
      <div className="flex flex-col justify-between gap-4 border-b border-slate-800 pb-5 md:flex-row md:items-center">
        <div>
          <Link href="/admin" className="mb-2 inline-flex items-center gap-1 text-xs text-slate-400 hover:text-slate-200"><ArrowLeft className="h-3 w-3" /> Admin</Link>
          <h1 className="flex items-center gap-2 text-2xl font-bold tracking-tight text-white">
            <GitCommit className="h-6 w-6 text-purple-400" /> Pipeline executions
          </h1>
          <p className="mt-1 text-sm text-slate-400">Real stage-by-stage traces of recent questions, recorded as they ran.</p>
        </div>
        <div className="flex items-center gap-3">
          <select
            className="rounded-lg border border-slate-800 bg-slate-900 px-3 py-2 text-sm"
            value={projectId ?? ""}
            onChange={(e) => setProjectId(Number(e.target.value))}
          >
            {projects.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
          </select>
          <button type="button" onClick={() => void load()} className="rounded-lg border border-slate-800 bg-slate-900 p-2 hover:border-slate-600" aria-label="Refresh">
            <RefreshCw className="h-4 w-4" />
          </button>
          <div className="rounded-lg border border-slate-800 bg-slate-900 px-4 py-2 text-sm">
            <span className="text-slate-400">p95 latency (last {rows.length}):</span>{" "}
            <span className="font-mono font-bold text-sky-400">{ms(p95)}</span>
          </div>
        </div>
      </div>

      {error && <p className="rounded-lg border border-rose-900 bg-rose-950/40 p-3 text-sm text-rose-300">{error}</p>}

      <div className="grid gap-6 lg:grid-cols-[340px_1fr]">
        <div className="max-h-[75vh] space-y-2 overflow-y-auto pr-1">
          {rows.length === 0 && !error && <p className="text-sm text-slate-500">No questions asked in this project yet.</p>}
          {rows.map((r) => (
            <button
              key={r.id}
              type="button"
              onClick={() => { setSelected(r); setStage(null); }}
              className={`w-full rounded-lg border p-3 text-left transition ${selected?.id === r.id ? "border-purple-500 bg-purple-950/30" : "border-slate-800 bg-slate-900 hover:border-slate-700"}`}
            >
              <p className="line-clamp-2 text-sm text-slate-100">{r.query}</p>
              <div className="mt-2 flex items-center justify-between text-[11px] text-slate-400">
                <span>{new Date(r.created_at).toLocaleString()}</span>
                <span className="font-mono">{ms(r.latency_ms)}</span>
              </div>
            </button>
          ))}
        </div>

        {selected && (
          <div className="space-y-6">
            <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
              {[
                ["Total latency", ms(selected.latency_ms)],
                ["Time to first token", ms(selected.ttft_ms)],
                ["Model", selected.model],
                ["Backend", selected.backend || "—"],
                ["Chunks used", String(selected.chunks_retrieved)],
              ].map(([k, v]) => (
                <div key={k} className="rounded-lg border border-slate-800 bg-slate-900 p-3">
                  <p className="text-[11px] uppercase tracking-wider text-slate-500">{k}</p>
                  <p className="mt-1 truncate font-mono text-sm text-slate-100" title={v}>{v}</p>
                </div>
              ))}
            </div>

            {timings && timingTotal > 0 && (
              <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
                <h2 className="mb-3 text-xs font-semibold uppercase tracking-wider text-slate-400">Where the time went</h2>
                <div className="flex h-6 w-full overflow-hidden rounded">
                  {Object.entries(timings).map(([k, v], i) => (
                    <div key={k} title={`${k}: ${Math.round(v)} ms`} style={{ width: `${(v / timingTotal) * 100}%` }}
                      className={["bg-sky-500", "bg-emerald-500", "bg-purple-500", "bg-amber-500", "bg-rose-500", "bg-teal-500"][i % 6]} />
                  ))}
                </div>
                <div className="mt-2 flex flex-wrap gap-3 text-[11px] text-slate-400">
                  {Object.entries(timings).map(([k, v]) => <span key={k}>{k.replace(/_/g, " ")}: <span className="font-mono text-slate-200">{Math.round(v)} ms</span></span>)}
                </div>
              </div>
            )}

            <div className="overflow-x-auto rounded-xl border border-slate-800 bg-slate-900/60 p-4">
              <h2 className="mb-4 text-xs font-semibold uppercase tracking-wider text-slate-400">Stages</h2>
              <div className="flex min-w-max items-center gap-3 pb-2">
                {stages.length === 0 && <p className="text-sm text-slate-500">No stage data recorded for this execution.</p>}
                {stages.map((s, i) => (
                  <React.Fragment key={s.key}>
                    <button
                      type="button"
                      onClick={() => setStage(s)}
                      className={`flex w-48 flex-col gap-2 rounded-xl border p-3 text-left ${stage?.key === s.key ? "border-purple-500 bg-purple-950/40" : "border-slate-800 bg-slate-900 hover:border-slate-700"}`}
                    >
                      <div className="flex items-center justify-between">
                        <div className="rounded-lg bg-slate-800/80 p-2">{STAGE_ICONS[s.key] ?? <Cpu className="h-4 w-4 text-slate-400" />}</div>
                        <StatusBadge status={s.status} />
                      </div>
                      <p className="text-sm font-semibold capitalize">{s.key.replace(/_/g, " ")}</p>
                      <p className="font-mono text-xs text-slate-400">{ms(s.duration_ms)}</p>
                    </button>
                    {i < stages.length - 1 && <ArrowRight className="h-5 w-5 shrink-0 text-slate-600" />}
                  </React.Fragment>
                ))}
              </div>
            </div>

            <div className="grid gap-6 md:grid-cols-2">
              <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
                <h2 className="mb-2 text-xs font-semibold uppercase tracking-wider text-slate-400">{stage ? `Stage: ${stage.key}` : "Routing decisions"}</h2>
                <pre className="max-h-96 overflow-auto whitespace-pre-wrap text-xs text-slate-300">
                  {JSON.stringify(stage ? { status: stage.status, duration_ms: stage.duration_ms, error: stage.error_message, ...stage.metadata } : {
                    turn_type: selected.trace?.turn_type,
                    retrieval_source: selected.trace?.retrieval_source,
                    rewrite: selected.trace?.rewrite_info,
                    adaptive_router: selected.trace?.adaptive_router_decision,
                    semantic_routing: selected.trace?.semantic_routing,
                    model_chain: selected.trace?.model_chain,
                  }, null, 2)}
                </pre>
              </div>
              <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
                <h2 className="mb-2 text-xs font-semibold uppercase tracking-wider text-slate-400">Answer</h2>
                <p className="max-h-96 overflow-auto whitespace-pre-wrap text-sm text-slate-200">{selected.answer}</p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
