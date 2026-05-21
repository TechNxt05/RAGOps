"use client";

import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { toast } from "sonner";
import { getRAGConfig, patchProjectRagConfig, type RAGConfig } from "@/lib/api";
import { cn } from "@/lib/utils";
import { Switch } from "@/components/ui/switch";

type Provider = "groq" | "google";

interface ModelCard {
  id: string;
  provider: Provider;
  model: string;
  title: string;
  blurb: string;
  costHint: string;
}

const PRIMARY_MODELS: ModelCard[] = [
  {
    id: "gemini-flash",
    provider: "google",
    model: "gemini-1.5-flash",
    title: "Gemini 1.5 Flash",
    blurb: "Fast · cheap · capable",
    costHint: "~$0.0001 / 1K tokens",
  },
  {
    id: "groq-llama",
    provider: "groq",
    model: "llama-3.3-70b-versatile",
    title: "Groq LLaMA 3.3",
    blurb: "Fastest · low cost · 70B",
    costHint: "~$0.0006 / 1K tokens",
  },
  {
    id: "gemini-pro",
    provider: "google",
    model: "gemini-1.5-pro",
    title: "Gemini 1.5 Pro",
    blurb: "Smartest · complex reasoning",
    costHint: "~$0.0013 / 1K tokens",
  },
];

const FALLBACK_OPTIONS: { provider: Provider; model: string; label: string }[] = [
  { provider: "groq", model: "llama-3.3-70b-versatile", label: "Groq LLaMA 3.3 70B" },
  { provider: "groq", model: "llama-3.1-8b-instant", label: "Groq LLaMA 3.1 8B" },
  { provider: "google", model: "gemini-1.5-flash", label: "Gemini 1.5 Flash" },
  { provider: "google", model: "gemini-1.5-pro", label: "Gemini 1.5 Pro" },
];

const EMBEDDINGS = [
  { value: "google-embedding-001", label: "Google embedding-001 (cloud)" },
  { value: "huggingface-minilm", label: "HuggingFace MiniLM (local, free)" },
];

export function ModelOrchestrator({ projectId }: { projectId: number }) {
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [cfg, setCfg] = useState<Partial<RAGConfig>>({});

  const load = async () => {
    setLoading(true);
    try {
      const data = await getRAGConfig(projectId);
      setCfg({
        ...data,
        primary_llm_provider: data.primary_llm_provider || "groq",
        primary_llm_name: data.primary_llm_name || "llama-3.3-70b-versatile",
        fallback_llm_provider: data.fallback_llm_provider || "google",
        fallback_llm_name: data.fallback_llm_name || "gemini-1.5-flash",
        embedding_model: data.embedding_model || "google-embedding-001",
        use_hybrid_search: data.use_hybrid_search !== undefined ? data.use_hybrid_search : true,
        semantic_weight: data.semantic_weight !== undefined ? data.semantic_weight : 0.6,
      });
    } catch {
      toast.error("Failed to load model configuration");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void load();
  }, [projectId]);

  const selectedPrimary = PRIMARY_MODELS.find(
    (m) => m.provider === cfg.primary_llm_provider && m.model === cfg.primary_llm_name
  );

  const handleSave = async () => {
    setSaving(true);
    try {
      await patchProjectRagConfig(projectId, {
        primary_llm_provider: cfg.primary_llm_provider,
        primary_llm_name: cfg.primary_llm_name,
        fallback_llm_provider: cfg.fallback_llm_provider,
        fallback_llm_name: cfg.fallback_llm_name,
        embedding_model: cfg.embedding_model,
        temperature: cfg.temperature,
        chunk_size: cfg.chunk_size,
        top_k: cfg.top_k,
        similarity_threshold: cfg.similarity_threshold,
        use_hybrid_search: cfg.use_hybrid_search,
        semantic_weight: cfg.semantic_weight,
      });
      toast.success("Model configuration saved");
    } catch {
      toast.error("Save failed");
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

  return (
    <Card className="border-indigo-200/60 dark:border-indigo-900/40">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">Model orchestration</CardTitle>
        <CardDescription>Primary, fallback, embeddings, and core RAG parameters for this project.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-8">
        <div>
          <Label className="text-base font-semibold">Primary LLM</Label>
          <div className="mt-3 grid gap-3 sm:grid-cols-3">
            {PRIMARY_MODELS.map((m) => {
              const active =
                cfg.primary_llm_provider === m.provider && cfg.primary_llm_name === m.model;
              return (
                <button
                  key={m.id}
                  type="button"
                  onClick={() =>
                    setCfg((c) => ({
                      ...c,
                      primary_llm_provider: m.provider,
                      primary_llm_name: m.model,
                    }))
                  }
                  className={cn(
                    "rounded-xl border p-4 text-left transition hover:border-indigo-400",
                    active ? "border-indigo-600 bg-indigo-50/80 dark:bg-indigo-950/40" : "border-border bg-card"
                  )}
                >
                  <div className="mb-2 flex items-center gap-2">
                    <span
                      className={cn(
                        "h-8 w-8 rounded-md",
                        m.provider === "google" ? "bg-blue-500/20" : "bg-violet-500/20"
                      )}
                    />
                    <span className="text-sm font-semibold">{m.title}</span>
                  </div>
                  <p className="text-xs text-muted-foreground">{m.blurb}</p>
                  <p className="mt-2 text-[10px] text-muted-foreground">{m.costHint}</p>
                </button>
              );
            })}
          </div>
          {selectedPrimary && (
            <p className="mt-2 text-xs text-muted-foreground">
              Selected: {selectedPrimary.title} ({selectedPrimary.provider})
            </p>
          )}
        </div>

        <div className="grid gap-6 md:grid-cols-2">
          <div className="space-y-2">
            <Label>Fallback LLM</Label>
            <select
              className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm"
              value={`${cfg.fallback_llm_provider}:${cfg.fallback_llm_name}`}
              onChange={(e) => {
                const [provider, model] = e.target.value.split(":") as [Provider, string];
                setCfg((c) => ({ ...c, fallback_llm_provider: provider, fallback_llm_name: model }));
              }}
            >
              {FALLBACK_OPTIONS.map((o) => (
                <option key={`${o.provider}:${o.model}`} value={`${o.provider}:${o.model}`}>
                  {o.label}
                </option>
              ))}
            </select>
          </div>
          <div className="space-y-2">
            <Label>Embedding model</Label>
            <select
              className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm"
              value={cfg.embedding_model || "google-embedding-001"}
              onChange={(e) => setCfg((c) => ({ ...c, embedding_model: e.target.value }))}
            >
              {EMBEDDINGS.map((e) => (
                <option key={e.value} value={e.value}>
                  {e.label}
                </option>
              ))}
            </select>
            <p className="text-xs text-muted-foreground">Changing embeddings requires a full re-index.</p>
          </div>
        </div>

        <div className="grid gap-8 md:grid-cols-2">
          <div className="space-y-2">
            <div className="flex justify-between">
              <Label>Temperature</Label>
              <span className="text-xs text-muted-foreground">{cfg.temperature?.toFixed(2) ?? "0.10"}</span>
            </div>
            <Slider
              value={[cfg.temperature ?? 0.1]}
              min={0}
              max={1}
              step={0.05}
              onValueChange={(v) => setCfg((c) => ({ ...c, temperature: v[0] }))}
            />
          </div>
          <div className="space-y-2">
            <div className="flex justify-between">
              <Label>Chunk size</Label>
              <span className="text-xs text-muted-foreground">{cfg.chunk_size ?? 1000}</span>
            </div>
            <Slider
              value={[cfg.chunk_size ?? 1000]}
              min={256}
              max={2048}
              step={64}
              onValueChange={(v) => setCfg((c) => ({ ...c, chunk_size: v[0] }))}
            />
          </div>
          <div className="space-y-2">
            <div className="flex justify-between">
              <Label>Top-K chunks</Label>
              <span className="text-xs text-muted-foreground">{cfg.top_k ?? 4}</span>
            </div>
            <Slider
              value={[cfg.top_k ?? 4]}
              min={1}
              max={20}
              step={1}
              onValueChange={(v) => setCfg((c) => ({ ...c, top_k: v[0] }))}
            />
          </div>
          <div className="space-y-2">
            <div className="flex justify-between">
              <Label>Similarity threshold</Label>
              <span className="text-xs text-muted-foreground">{cfg.similarity_threshold?.toFixed(2) ?? "0"}</span>
            </div>
            <Slider
              value={[cfg.similarity_threshold ?? 0]}
              min={0}
              max={1}
              step={0.05}
              onValueChange={(v) => setCfg((c) => ({ ...c, similarity_threshold: v[0] }))}
            />
          </div>

          <div className="rounded-xl border border-indigo-100 dark:border-indigo-950/80 p-4 space-y-4 md:col-span-2">
            <div className="flex items-center justify-between">
              <div className="space-y-0.5 pr-4">
                <Label className="text-sm font-semibold">Hybrid Search (Lexical + Semantic)</Label>
                <p className="text-xs text-muted-foreground">
                  Combines FAISS semantic search meaning-matching and BM25 local lexical keyword-matching using Reciprocal Rank Fusion (RRF).
                </p>
              </div>
              <Switch
                checked={cfg.use_hybrid_search ?? true}
                onCheckedChange={(checked) => setCfg((c) => ({ ...c, use_hybrid_search: checked }))}
              />
            </div>

            {(cfg.use_hybrid_search ?? true) && (
              <div className="space-y-2 pt-2 border-t border-indigo-50 dark:border-indigo-900/40">
                <div className="flex justify-between text-xs">
                  <span className="font-medium">Fusion Weight Balance</span>
                  <span className="text-indigo-600 dark:text-indigo-400 font-semibold">
                    {Math.round((cfg.semantic_weight ?? 0.6) * 100)}% Semantic / {Math.round((1 - (cfg.semantic_weight ?? 0.6)) * 100)}% Lexical
                  </span>
                </div>
                <Slider
                  value={[cfg.semantic_weight ?? 0.6]}
                  min={0.1}
                  max={0.9}
                  step={0.05}
                  onValueChange={(v) => setCfg((c) => ({ ...c, semantic_weight: v[0] }))}
                />
                <p className="text-[10px] text-muted-foreground pt-1">
                  Adjust the slider to prioritize conceptual matching (Semantic) or exact keyword matching (Lexical).
                </p>
              </div>
            )}
          </div>
        </div>

        <Button type="button" className="w-full sm:w-auto" disabled={saving} onClick={() => void handleSave()}>
          {saving ? "Saving…" : "Save configuration"}
        </Button>
      </CardContent>
    </Card>
  );
}
