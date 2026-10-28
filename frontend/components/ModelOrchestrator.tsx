"use client";

import { useCallback, useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { toast } from "sonner";
import {
  errorMessage,
  getModelCatalog,
  getRAGConfig,
  patchProjectRagConfig,
  type ModelSpec,
  type RAGConfig,
} from "@/lib/api";
import { ArrowDown, ArrowUp, Plus, X } from "lucide-react";

const EMBEDDINGS = [
  { value: "all-MiniLM-L6-v2", label: "MiniLM-L6 (local · fast · 384d)" },
  { value: "BAAI/bge-small-en-v1.5", label: "BGE-small v1.5 (local · better recall · 384d)" },
  { value: "BAAI/bge-base-en-v1.5", label: "BGE-base v1.5 (local · best local quality · 768d)" },
  { value: "gemini-embedding-001", label: "Gemini embedding-001 (API · free tier)" },
];

const STYLES = ["Concise", "Detailed", "Bullet points", "Technical", "Friendly"];

type ToggleKey =
  | "use_hybrid_search"
  | "enable_query_rewrite"
  | "use_multi_query"
  | "enable_hyde"
  | "enable_adaptive_router"
  | "enable_reranker"
  | "enable_cross_reference"
  | "enable_conflict_detection"
  | "enable_compression"
  | "enable_self_correction"
  | "enable_claim_verification"
  | "enable_reflective_generation"
  | "enable_semantic_cache"
  | "answer_only_from_docs"
  | "enable_cost_routing";

const TECHNIQUES: { key: ToggleKey; title: string; detail: string; defaultOn: boolean }[] = [
  { key: "use_hybrid_search", title: "Hybrid search", detail: "Vector search + BM25 keyword search fused with RRF.", defaultOn: true },
  { key: "enable_query_rewrite", title: "Query rewriting", detail: "Turns follow-ups (\"what about them?\") into standalone questions.", defaultOn: true },
  { key: "use_multi_query", title: "Multi-query (RAG-Fusion)", detail: "Searches several paraphrases of the question and fuses the results.", defaultOn: true },
  { key: "enable_hyde", title: "HyDE", detail: "Searches with a hypothetical answer instead of the question. Extra LLM call.", defaultOn: false },
  { key: "enable_adaptive_router", title: "Adaptive router", detail: "Sizes retrieval per question; routes counts/aggregates to SQL.", defaultOn: true },
  { key: "enable_reranker", title: "Cross-encoder reranker", detail: "BGE reranker re-scores candidates for precision.", defaultOn: true },
  { key: "enable_cross_reference", title: "Cross-reference resolution", detail: "Follows \"see Section 4.2\"-style references to pull the target text.", defaultOn: true },
  { key: "enable_conflict_detection", title: "Conflict detection", detail: "Flags sources that disagree and tells the model to surface it.", defaultOn: true },
  { key: "enable_compression", title: "Contextual compression", detail: "When context exceeds the budget, keeps the most relevant sentences.", defaultOn: true },
  { key: "enable_self_correction", title: "Self-correcting retrieval", detail: "If evidence is weak, rewrites the query and retries once.", defaultOn: true },
  { key: "enable_claim_verification", title: "Claim verification", detail: "Checks each sentence of the answer against the sources.", defaultOn: true },
  { key: "enable_reflective_generation", title: "Reflective generation", detail: "Rewrites answers whose claims aren't supported. Extra LLM call.", defaultOn: false },
  { key: "enable_semantic_cache", title: "Semantic answer cache", detail: "Instant answers for repeated questions; resets when documents change.", defaultOn: true },
  { key: "answer_only_from_docs", title: "Strict grounding", detail: "Never use general knowledge, even when labelled.", defaultOn: false },
  { key: "enable_cost_routing", title: "Cost routing", detail: "Simple questions go to the first fallback model first.", defaultOn: false },
];

function SliderRow({
  label, value, min, max, step, format, onChange, hint,
}: {
  label: string; value: number; min: number; max: number; step: number;
  format?: (v: number) => string; onChange: (v: number) => void; hint?: string;
}) {
  return (
    <div className="space-y-2">
      <div className="flex justify-between">
        <Label>{label}</Label>
        <span className="text-xs tabular-nums text-muted-foreground">{format ? format(value) : value}</span>
      </div>
      <Slider value={[value]} min={min} max={max} step={step} onValueChange={(v) => onChange(v[0])} />
      {hint && <p className="text-[11px] text-muted-foreground">{hint}</p>}
    </div>
  );
}

function ModelSelect({
  value, catalog, onChange,
}: {
  value: ModelSpec; catalog: Record<string, string[]>; onChange: (m: ModelSpec) => void;
}) {
  const options = Object.entries(catalog).flatMap(([provider, names]) => names.map((name) => ({ provider, name })));
  const known = options.some((o) => o.provider === value.provider && o.name === value.name);
  return (
    <select
      className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm"
      value={`${value.provider}|${value.name}`}
      onChange={(e) => {
        const [provider, ...rest] = e.target.value.split("|");
        onChange({ provider, name: rest.join("|") });
      }}
    >
      {!known && <option value={`${value.provider}|${value.name}`}>{value.name} ({value.provider}, not currently listed)</option>}
      {Object.entries(catalog).map(([provider, names]) => (
        <optgroup key={provider} label={provider}>
          {names.map((name) => (
            <option key={`${provider}|${name}`} value={`${provider}|${name}`}>{name}</option>
          ))}
        </optgroup>
      ))}
    </select>
  );
}

export function ModelOrchestrator({ projectId }: { projectId: number }) {
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [cfg, setCfg] = useState<Partial<RAGConfig>>({});
  const [catalog, setCatalog] = useState<Record<string, string[]>>({});
  const [originalEmbedding, setOriginalEmbedding] = useState<string | undefined>();

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const [data, models] = await Promise.all([getRAGConfig(projectId), getModelCatalog().catch(() => ({}))]);
      setCfg(data);
      setCatalog(models);
      setOriginalEmbedding(data.embedding_model);
    } catch (e) {
      toast.error(errorMessage(e, "Failed to load configuration"));
    } finally {
      setLoading(false);
    }
  }, [projectId]);

  useEffect(() => {
    void load();
  }, [load]);

  const set = <K extends keyof RAGConfig>(key: K, value: RAGConfig[K]) => setCfg((c) => ({ ...c, [key]: value }));
  const primary: ModelSpec = cfg.primary_models?.[0] ?? { provider: "groq", name: "openai/gpt-oss-20b" };
  const fallbacks: ModelSpec[] = cfg.fallback_models ?? [];

  const setFallbacks = (list: ModelSpec[]) => set("fallback_models", list);
  const moveFallback = (i: number, dir: -1 | 1) => {
    const list = [...fallbacks];
    const j = i + dir;
    if (j < 0 || j >= list.length) return;
    [list[i], list[j]] = [list[j], list[i]];
    setFallbacks(list);
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      const { id: _id, project_id: _pid, created_at: _c, is_active: _a, ...patch } = cfg as RAGConfig;
      void _id; void _pid; void _c; void _a;
      await patchProjectRagConfig(projectId, patch);
      if (cfg.embedding_model !== originalEmbedding) {
        toast.success("Saved. Re-indexing documents with the new embedding model in the background.");
        setOriginalEmbedding(cfg.embedding_model);
      } else {
        toast.success("Pipeline configuration saved");
      }
    } catch (e) {
      toast.error(errorMessage(e, "Save failed"));
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className="space-y-3">
        <div className="h-36 animate-pulse rounded-xl bg-muted" />
        <div className="h-24 animate-pulse rounded-xl bg-muted" />
      </div>
    );
  }

  const noProviders = Object.keys(catalog).length === 0;

  return (
    <Card className="border-indigo-200/60 dark:border-indigo-900/40">
      <CardHeader>
        <CardTitle>Pipeline studio</CardTitle>
        <CardDescription>Every model and RAG technique for this project. Changes apply to the next question.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-8">
        <section className="space-y-3">
          <Label className="text-base font-semibold">Models</Label>
          {noProviders && (
            <p className="text-xs text-amber-600">No provider API keys detected on the server. Set GROQ_API_KEY or GEMINI_API_KEY.</p>
          )}
          <div className="space-y-1.5">
            <Label className="text-xs text-muted-foreground">Primary</Label>
            <ModelSelect value={primary} catalog={catalog} onChange={(m) => set("primary_models", [m])} />
          </div>
          <div className="space-y-1.5">
            <Label className="text-xs text-muted-foreground">Fallback chain (tried in order on errors or rate limits)</Label>
            {fallbacks.map((m, i) => (
              <div key={i} className="flex items-center gap-2">
                <span className="w-5 text-xs text-muted-foreground">{i + 1}.</span>
                <ModelSelect value={m} catalog={catalog} onChange={(nm) => setFallbacks(fallbacks.map((x, k) => (k === i ? nm : x)))} />
                <Button type="button" size="icon" variant="ghost" aria-label="Move up" onClick={() => moveFallback(i, -1)}><ArrowUp className="h-4 w-4" /></Button>
                <Button type="button" size="icon" variant="ghost" aria-label="Move down" onClick={() => moveFallback(i, 1)}><ArrowDown className="h-4 w-4" /></Button>
                <Button type="button" size="icon" variant="ghost" aria-label="Remove" onClick={() => setFallbacks(fallbacks.filter((_, k) => k !== i))}><X className="h-4 w-4" /></Button>
              </div>
            ))}
            {fallbacks.length < 3 && (
              <Button type="button" variant="outline" size="sm" onClick={() => setFallbacks([...fallbacks, primary])}>
                <Plus className="mr-1 h-3.5 w-3.5" /> Add fallback
              </Button>
            )}
          </div>
        </section>

        <section className="grid gap-6 md:grid-cols-2">
          <div className="space-y-2">
            <Label>Embedding model</Label>
            <select
              className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm"
              value={cfg.embedding_model || "all-MiniLM-L6-v2"}
              onChange={(e) => set("embedding_model", e.target.value)}
            >
              {EMBEDDINGS.map((e) => <option key={e.value} value={e.value}>{e.label}</option>)}
              {!EMBEDDINGS.some((e) => e.value === cfg.embedding_model) && cfg.embedding_model && (
                <option value={cfg.embedding_model}>{cfg.embedding_model}</option>
              )}
            </select>
            <p className="text-[11px] text-muted-foreground">Changing it re-embeds this project&apos;s documents automatically.</p>
          </div>
          <div className="space-y-2">
            <Label>Response style</Label>
            <select
              className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm"
              value={cfg.response_style || "Concise"}
              onChange={(e) => set("response_style", e.target.value)}
            >
              {STYLES.map((s) => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>
        </section>

        <section className="grid gap-8 md:grid-cols-2">
          <SliderRow label="Temperature" value={cfg.temperature ?? 0.7} min={0} max={1.5} step={0.05} format={(v) => v.toFixed(2)} onChange={(v) => set("temperature", v)} />
          <SliderRow label="Top-K chunks" value={cfg.top_k ?? 4} min={1} max={20} step={1} onChange={(v) => set("top_k", v)}
            hint="Used when the adaptive router is off; the router sizes retrieval per question otherwise." />
          <SliderRow label="Chunk size (chars)" value={cfg.chunk_size ?? 1000} min={256} max={4096} step={64} onChange={(v) => set("chunk_size", v)}
            hint="Applies to newly uploaded or re-chunked documents." />
          <SliderRow label="Chunk overlap" value={cfg.chunk_overlap ?? 200} min={0} max={Math.max(0, (cfg.chunk_size ?? 1000) - 64)} step={16} onChange={(v) => set("chunk_overlap", v)} />
          <SliderRow label="Context budget (tokens)" value={cfg.max_context_tokens ?? 2048} min={512} max={16000} step={256} onChange={(v) => set("max_context_tokens", v)}
            hint="Compression only kicks in above this." />
          <SliderRow label="Max answer tokens" value={cfg.max_output_tokens ?? 1024} min={256} max={8192} step={128} onChange={(v) => set("max_output_tokens", v)} />
          <SliderRow label="Confidence gate" value={cfg.confidence_threshold ?? 0.4} min={0} max={0.9} step={0.05} format={(v) => v.toFixed(2)} onChange={(v) => set("confidence_threshold", v)}
            hint="Refuse when evidence is below this. 0.40 is calibrated; raise for stricter answers." />
          <SliderRow label="Minimum chunk relevance" value={cfg.similarity_threshold ?? 0} min={0} max={0.9} step={0.05} format={(v) => v.toFixed(2)} onChange={(v) => set("similarity_threshold", v)}
            hint="Drop retrieved chunks below this relevance (0 = keep all)." />
          {(cfg.use_hybrid_search ?? true) && (
            <SliderRow label="Hybrid weight" value={cfg.semantic_weight ?? 0.6} min={0.1} max={0.9} step={0.05}
              format={(v) => `${Math.round(v * 100)}% semantic / ${Math.round((1 - v) * 100)}% keyword`} onChange={(v) => set("semantic_weight", v)} />
          )}
          {(cfg.enable_semantic_cache ?? true) && (
            <SliderRow label="Cache match strictness" value={cfg.semantic_cache_threshold ?? 0.92} min={0.85} max={0.99} step={0.01} format={(v) => v.toFixed(2)} onChange={(v) => set("semantic_cache_threshold", v)} />
          )}
        </section>

        <section className="space-y-3">
          <Label className="text-base font-semibold">Techniques</Label>
          <div className="grid gap-3 md:grid-cols-2">
            {TECHNIQUES.map((t) => (
              <div key={t.key} className="flex items-start justify-between gap-3 rounded-lg border p-3">
                <div>
                  <p className="text-sm font-medium">{t.title}</p>
                  <p className="text-xs text-muted-foreground">{t.detail}</p>
                </div>
                <Switch
                  checked={(cfg[t.key] as boolean | undefined) ?? t.defaultOn}
                  onCheckedChange={(checked) => set(t.key, checked as never)}
                />
              </div>
            ))}
          </div>
        </section>

        <Button type="button" className="w-full sm:w-auto" disabled={saving} onClick={() => void handleSave()}>
          {saving ? "Saving…" : "Save configuration"}
        </Button>
      </CardContent>
    </Card>
  );
}
