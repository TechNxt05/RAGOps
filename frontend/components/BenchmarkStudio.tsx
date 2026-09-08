"use client";

import { useState, useEffect, useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import {
  Play,
  Square,
  CheckCircle2,
  Clock,
  DollarSign,
  Zap,
  Layers,
  Sparkles,
  Download,
  Copy,
  AlertCircle,
  BarChart2,
  RefreshCw,
  Award,
  Cpu,
  HelpCircle,
  ChevronRight,
  TrendingUp
} from "lucide-react";
import { toast } from "sonner";
import { motion, AnimatePresence } from "framer-motion";
import {
  getBenchmarkModels,
  getBenchmarkVariants,
  getBenchmarkDatasets,
  startBenchmark,
  getBenchmarkStatus,
  getBenchmarkResults,
  cancelBenchmark,
  ModelProfile,
  PipelineVariantInfo,
  BenchmarkDatasetInfo,
  BenchmarkVariantSummary,
  BenchmarkStatusResponse
} from "@/lib/api";

interface BenchmarkStudioProps {
  projectId: number;
}

export function BenchmarkStudio({ projectId }: BenchmarkStudioProps) {
  // Config state
  const [models, setModels] = useState<ModelProfile[]>([]);
  const [variants, setVariants] = useState<PipelineVariantInfo[]>([]);
  const [datasets, setDatasets] = useState<BenchmarkDatasetInfo[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string>("openrouter_gpt4o_mini");
  const [selectedVariants, setSelectedVariants] = useState<string[]>([
    "advanced",
    "hybrid",
    "full_ragops",
    "adaptive",
    "policy_driven"
  ]);
  const [selectedDataset, setSelectedDataset] = useState<string>("smoke_5");
  const [delayMs, setDelayMs] = useState<number>(0);

  // Execution state
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<BenchmarkStatusResponse | null>(null);
  const [results, setResults] = useState<BenchmarkVariantSummary[] | null>(null);
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [sortBy, setSortBy] = useState<string>("avg_overall_score");
  const [sortAsc, setSortAsc] = useState<boolean>(false);

  const pollTimerRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    loadMetadata();
    return () => {
      if (pollTimerRef.current) clearInterval(pollTimerRef.current);
    };
  }, []);

  const loadMetadata = async () => {
    try {
      const [m, v, d] = await Promise.all([
        getBenchmarkModels(),
        getBenchmarkVariants(),
        getBenchmarkDatasets()
      ]);
      setModels(m);
      setVariants(v);
      setDatasets(d);
    } catch (e: any) {
      toast.error("Failed to load benchmark configurations: " + e.message);
    }
  };

  const toggleVariant = (vName: string) => {
    if (selectedVariants.includes(vName)) {
      if (selectedVariants.length === 1) {
        toast.warning("At least one variant must be selected.");
        return;
      }
      setSelectedVariants(selectedVariants.filter((v) => v !== vName));
    } else {
      setSelectedVariants([...selectedVariants, vName]);
    }
  };

  const selectCore4 = () => {
    setSelectedVariants(["advanced", "hybrid", "full_ragops", "adaptive"]);
  };

  const selectAllVariants = () => {
    setSelectedVariants(variants.map((v) => v.name));
  };

  const handleStartBenchmark = async () => {
    if (selectedVariants.length === 0) {
      toast.error("Please select at least one pipeline variant.");
      return;
    }

    setIsRunning(true);
    setResults(null);
    setJobStatus(null);

    try {
      const res = await startBenchmark({
        project_id: projectId,
        model_profile_id: selectedModelId,
        variants: selectedVariants,
        dataset_type: selectedDataset,
        delay_ms: delayMs,
        resume: false
      });

      setActiveJobId(res.job_id);
      toast.success("AutoRAGBench started! Monitoring progress...");

      // Start polling
      if (pollTimerRef.current) clearInterval(pollTimerRef.current);
      pollTimerRef.current = setInterval(() => pollJobStatus(res.job_id), 1500);
    } catch (e: any) {
      setIsRunning(false);
      toast.error("Failed to launch benchmark: " + (e.response?.data?.detail || e.message));
    }
  };

  const pollJobStatus = async (jobId: string) => {
    try {
      const statusData = await getBenchmarkStatus(jobId);
      setJobStatus(statusData);

      if (statusData.status === "completed") {
        if (pollTimerRef.current) clearInterval(pollTimerRef.current);
        setIsRunning(false);
        toast.success("Benchmark run completed successfully!");

        // Fetch final full results
        const resData = await getBenchmarkResults(jobId);
        setResults(resData.summaries);
      } else if (statusData.status === "failed" || statusData.status === "cancelled") {
        if (pollTimerRef.current) clearInterval(pollTimerRef.current);
        setIsRunning(false);
        if (statusData.status === "failed") {
          toast.error("Benchmark run failed: " + (statusData.error || "Unknown error"));
        } else {
          toast.info("Benchmark run was cancelled.");
        }
      }
    } catch (e: any) {
      console.error("Poll status error:", e);
    }
  };

  const handleCancel = async () => {
    if (!activeJobId) return;
    try {
      await cancelBenchmark(activeJobId);
      setIsRunning(false);
      toast.info("Cancelling benchmark run...");
    } catch (e: any) {
      toast.error("Failed to cancel: " + e.message);
    }
  };

  const handleCopyMarkdown = () => {
    if (!results || results.length === 0) return;
    let md = "| Variant | Overall (95% CI) | Context Rel | Faithfulness | Ans Rel | Groundedness | Latency (ms) | Cost ($) | Cache Hit % |\n";
    md += "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n";
    results.forEach((s) => {
      const ci = s.ci_lower_95 !== undefined ? `[${s.ci_lower_95.toFixed(2)}-${s.ci_upper_95?.toFixed(2)}]` : "";
      md += `| **${s.label}** | ${s.avg_overall_score.toFixed(3)} ${ci} | ${s.avg_context_relevance.toFixed(3)} | ${s.avg_faithfulness.toFixed(3)} | ${s.avg_answer_relevance.toFixed(3)} | ${s.avg_groundedness.toFixed(3)} | ${Math.round(s.avg_latency_ms)}ms | $${s.total_token_cost.toFixed(6)} | ${(s.cache_hit_rate * 100).toFixed(1)}% |\n`;
    });
    navigator.clipboard.writeText(md);
    toast.success("Leaderboard Markdown copied to clipboard!");
  };

  const handleDownloadJSON = () => {
    if (!results) return;
    const blob = new Blob([JSON.stringify({ model_id: selectedModelId, results }, null, 2)], {
      type: "application/json"
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `autoragbench_${selectedModelId}_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
    toast.success("Report downloaded as JSON.");
  };

  // Sort summaries
  const sortedSummaries = results
    ? [...results].sort((a: any, b: any) => {
        const valA = a[sortBy] ?? 0;
        const valB = b[sortBy] ?? 0;
        return sortAsc ? valA - valB : valB - valA;
      })
    : [];

  const bestScore = results ? Math.max(...results.map((r) => r.avg_overall_score)) : 0;
  const lowestCost = results ? Math.min(...results.map((r) => r.total_token_cost)) : 0;
  const lowestLatency = results ? Math.min(...results.map((r) => r.avg_latency_ms)) : 0;

  return (
    <div className="space-y-8 max-w-7xl mx-auto pb-12">
      {/* Header Banner */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 p-6 rounded-2xl bg-gradient-to-r from-indigo-950 via-slate-900 to-indigo-900 border border-indigo-500/20 shadow-xl">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <Sparkles className="w-6 h-6 text-indigo-400" />
            <h2 className="text-2xl font-bold text-white tracking-tight">AutoRAGBench Studio</h2>
            <Badge variant="outline" className="text-indigo-300 border-indigo-500/40 bg-indigo-950/60">
              1-Click Multi-Model
            </Badge>
          </div>
          <p className="text-sm text-slate-300">
            Systematically benchmark and compare RAG architectures across LLM models with RAGAS Triad, Groundedness, Token Cost, and Paired Statistical Testing.
          </p>
        </div>

        <div className="flex items-center gap-3">
          {isRunning ? (
            <Button
              variant="destructive"
              onClick={handleCancel}
              className="gap-2 shadow-lg hover:shadow-red-500/20"
            >
              <Square className="w-4 h-4" /> Cancel Benchmark
            </Button>
          ) : (
            <Button
              onClick={handleStartBenchmark}
              className="gap-2 bg-gradient-to-r from-indigo-600 to-violet-600 hover:from-indigo-500 hover:to-violet-500 text-white shadow-lg shadow-indigo-600/30 px-6 font-semibold"
            >
              <Play className="w-4 h-4 fill-white" /> Launch AutoRAGBench
            </Button>
          )}
        </div>
      </div>

      {/* Grid: 1. Model Selection + 2. Dataset Scope */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Step 1: Model Preset Cards (2 cols) */}
        <Card className="lg:col-span-2 border-slate-800 bg-slate-900/60 backdrop-blur-sm shadow-md">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Cpu className="w-5 h-5 text-indigo-400" />
                <CardTitle className="text-base font-semibold">1. Choose Target Model Profile</CardTitle>
              </div>
              <span className="text-xs text-slate-400">Pre-configured 1-click presets</span>
            </div>
            <CardDescription className="text-xs">
              Select the LLM generation engine to evaluate across the RAG pipelines.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {models.map((m) => {
                const isSelected = selectedModelId === m.id;
                return (
                  <div
                    key={m.id}
                    onClick={() => m.is_available && setSelectedModelId(m.id)}
                    className={`relative p-4 rounded-xl border transition-all cursor-pointer flex flex-col justify-between ${
                      !m.is_available
                        ? "opacity-50 border-slate-800 bg-slate-950/40 cursor-not-allowed"
                        : isSelected
                        ? "border-indigo-500 bg-indigo-950/40 shadow-md shadow-indigo-500/10 ring-1 ring-indigo-500"
                        : "border-slate-800 bg-slate-950/60 hover:border-slate-700 hover:bg-slate-900"
                    }`}
                  >
                    <div>
                      <div className="flex items-center justify-between gap-2 mb-1.5">
                        <span className="font-semibold text-sm text-slate-100">{m.name}</span>
                        <Badge
                          variant="secondary"
                          className={`text-[10px] px-2 py-0.5 font-medium ${
                            m.badge === "Recommended"
                              ? "bg-emerald-950 text-emerald-300 border border-emerald-500/30"
                              : "bg-slate-800 text-slate-300"
                          }`}
                        >
                          {m.badge}
                        </Badge>
                      </div>
                      <p className="text-xs text-slate-400 leading-relaxed mb-3">{m.description}</p>
                    </div>

                    <div className="flex items-center justify-between text-[11px] text-slate-400 border-t border-slate-800/80 pt-2 mt-auto">
                      <span className="capitalize font-mono text-indigo-300">{m.provider}</span>
                      <span>
                        ${m.cost_per_1m_input}/M in &bull; ${m.cost_per_1m_output}/M out
                      </span>
                    </div>

                    {isSelected && (
                      <div className="absolute top-2 right-2 w-2 h-2 rounded-full bg-indigo-400 ring-4 ring-indigo-500/20" />
                    )}
                  </div>
                );
              })}
            </div>
          </CardContent>
        </Card>

        {/* Step 2: Evaluation Scope / Dataset */}
        <Card className="border-slate-800 bg-slate-900/60 backdrop-blur-sm shadow-md flex flex-col justify-between">
          <div>
            <CardHeader className="pb-3">
              <div className="flex items-center gap-2">
                <BarChart2 className="w-5 h-5 text-indigo-400" />
                <CardTitle className="text-base font-semibold">2. Benchmark Scope</CardTitle>
              </div>
              <CardDescription className="text-xs">Choose query suite depth</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              {datasets.map((d) => {
                const isSelected = selectedDataset === d.id;
                return (
                  <div
                    key={d.id}
                    onClick={() => setSelectedDataset(d.id)}
                    className={`p-3.5 rounded-xl border transition-all cursor-pointer ${
                      isSelected
                        ? "border-indigo-500 bg-indigo-950/40 shadow-sm ring-1 ring-indigo-500"
                        : "border-slate-800 bg-slate-950/60 hover:border-slate-700"
                    }`}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-semibold text-xs text-slate-200">{d.name}</span>
                      <Badge variant="outline" className="text-[10px] text-indigo-300 border-indigo-500/30">
                        {d.query_count} Queries
                      </Badge>
                    </div>
                    <p className="text-[11px] text-slate-400 leading-relaxed">{d.description}</p>
                  </div>
                );
              })}

              <div className="pt-2">
                <Label className="text-xs text-slate-300">Inter-query Rate Delay (ms)</Label>
                <Input
                  type="number"
                  min="0"
                  step="250"
                  value={delayMs}
                  onChange={(e) => setDelayMs(parseInt(e.target.value) || 0)}
                  className="mt-1 bg-slate-950 border-slate-800 text-xs h-8"
                  placeholder="0 (no delay)"
                />
              </div>
            </CardContent>
          </div>

          <div className="p-6 pt-0">
            <div className="bg-slate-950/80 p-3 rounded-lg border border-slate-800/80 text-[11px] text-slate-400 flex items-start gap-2">
              <HelpCircle className="w-4 h-4 text-indigo-400 flex-shrink-0 mt-0.5" />
              <span>
                Deterministic greedy sampling (temp=0.0) is enforced to ensure reproducible evaluation across runs.
              </span>
            </div>
          </div>
        </Card>
      </div>

      {/* Step 3: RAG Pipeline Variant Matrix */}
      <Card className="border-slate-800 bg-slate-900/60 backdrop-blur-sm shadow-md">
        <CardHeader className="pb-3">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <Layers className="w-5 h-5 text-indigo-400" />
              <CardTitle className="text-base font-semibold">3. Select RAG Pipeline Variants ({selectedVariants.length} selected)</CardTitle>
            </div>
            <div className="flex items-center gap-2">
              <Button variant="ghost" size="sm" onClick={selectCore4} className="text-xs h-7 text-indigo-300 hover:text-white">
                Select Core 4
              </Button>
              <Button variant="ghost" size="sm" onClick={selectAllVariants} className="text-xs h-7 text-indigo-300 hover:text-white">
                Select All 10
              </Button>
            </div>
          </div>
          <CardDescription className="text-xs">
            Toggle the exact retrieval architectures, guardrails, and agentic workflows to evaluate in parallel.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {variants.map((v) => {
              const isChecked = selectedVariants.includes(v.name);
              return (
                <div
                  key={v.name}
                  onClick={() => toggleVariant(v.name)}
                  className={`p-3 rounded-xl border transition-all cursor-pointer flex items-start gap-3 ${
                    isChecked
                      ? "border-indigo-500/70 bg-indigo-950/30 ring-1 ring-indigo-500/30"
                      : "border-slate-800/80 bg-slate-950/40 opacity-70 hover:opacity-100 hover:border-slate-700"
                  }`}
                >
                  <div
                    className={`mt-0.5 w-4 h-4 rounded flex items-center justify-center border transition-colors ${
                      isChecked ? "bg-indigo-600 border-indigo-500 text-white" : "border-slate-700 bg-slate-900"
                    }`}
                  >
                    {isChecked && <CheckCircle2 className="w-3.5 h-3.5" />}
                  </div>
                  <div>
                    <span className="font-semibold text-xs text-slate-200 block">{v.label}</span>
                    <p className="text-[11px] text-slate-400 leading-snug mt-0.5">{v.description}</p>
                  </div>
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>

      {/* Real-Time Execution Progress Banner */}
      <AnimatePresence>
        {(isRunning || (jobStatus && jobStatus.status === "running")) && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
          >
            <Card className="border-indigo-500/30 bg-slate-900/90 shadow-2xl ring-1 ring-indigo-500/20">
              <CardContent className="p-6 space-y-4">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                  <div className="flex items-center gap-3">
                    <RefreshCw className="w-5 h-5 text-indigo-400 animate-spin" />
                    <div>
                      <h4 className="font-semibold text-sm text-white">
                        Benchmarking in Progress: {jobStatus?.model_name || "Active Model"}
                      </h4>
                      <p className="text-xs text-slate-400">
                        Running {jobStatus?.current_variant || "pipeline"} &bull; {jobStatus?.completed_queries || 0} / {jobStatus?.total_queries || 0} queries completed
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-4 text-xs font-mono text-slate-300">
                    <span className="flex items-center gap-1">
                      <Clock className="w-3.5 h-3.5 text-indigo-400" />
                      {jobStatus?.elapsed_seconds ? `${jobStatus.elapsed_seconds}s` : "0s"}
                    </span>
                    <span className="font-bold text-indigo-400">{jobStatus?.progress_pct || 0}%</span>
                  </div>
                </div>

                {/* Progress Bar */}
                <div className="w-full bg-slate-800 rounded-full h-2.5 overflow-hidden">
                  <div
                    className="bg-gradient-to-r from-indigo-500 to-violet-500 h-2.5 rounded-full transition-all duration-300"
                    style={{ width: `${jobStatus?.progress_pct || 0}%` }}
                  />
                </div>

                {/* Live Query Ticker */}
                {jobStatus?.current_query && (
                  <div className="text-xs text-slate-400 flex items-center gap-2 bg-slate-950 p-2.5 rounded-lg border border-slate-800 truncate">
                    <span className="text-indigo-400 font-semibold font-mono">CURRENT QUERY:</span>
                    <span className="truncate text-slate-200">"{jobStatus.current_query}"</span>
                  </div>
                )}
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Leaderboard & Results View */}
      {results && results.length > 0 && (
        <Card className="border-slate-800 bg-slate-900/70 backdrop-blur-md shadow-xl">
          <CardHeader>
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
              <div>
                <div className="flex items-center gap-2">
                  <Award className="w-5 h-5 text-amber-400" />
                  <CardTitle className="text-lg font-bold">AutoRAGBench Leaderboard</CardTitle>
                </div>
                <CardDescription className="text-xs mt-1">
                  Evaluated with {jobStatus?.model_name || "Selected Model"} across {results.length} pipeline architectures.
                </CardDescription>
              </div>

              <div className="flex items-center gap-2">
                <Button variant="outline" size="sm" onClick={handleCopyMarkdown} className="gap-1.5 text-xs border-slate-700 bg-slate-950">
                  <Copy className="w-3.5 h-3.5" /> Copy Markdown
                </Button>
                <Button variant="outline" size="sm" onClick={handleDownloadJSON} className="gap-1.5 text-xs border-slate-700 bg-slate-950">
                  <Download className="w-3.5 h-3.5" /> Export JSON
                </Button>
              </div>
            </div>

            {/* Quick Badges Summary */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-3">
              <div className="p-3 rounded-lg bg-emerald-950/30 border border-emerald-500/20 flex items-center gap-3">
                <Award className="w-5 h-5 text-emerald-400 flex-shrink-0" />
                <div>
                  <span className="text-[11px] text-emerald-300 font-medium">Highest Quality</span>
                  <p className="text-xs font-bold text-white">
                    {results.find((r) => r.avg_overall_score === bestScore)?.label || "N/A"} ({bestScore.toFixed(3)})
                  </p>
                </div>
              </div>

              <div className="p-3 rounded-lg bg-cyan-950/30 border border-cyan-500/20 flex items-center gap-3">
                <Zap className="w-5 h-5 text-cyan-400 flex-shrink-0" />
                <div>
                  <span className="text-[11px] text-cyan-300 font-medium">Fastest Latency</span>
                  <p className="text-xs font-bold text-white">
                    {results.find((r) => r.avg_latency_ms === lowestLatency)?.label || "N/A"} ({Math.round(lowestLatency)}ms)
                  </p>
                </div>
              </div>

              <div className="p-3 rounded-lg bg-indigo-950/30 border border-indigo-500/20 flex items-center gap-3">
                <DollarSign className="w-5 h-5 text-indigo-400 flex-shrink-0" />
                <div>
                  <span className="text-[11px] text-indigo-300 font-medium">Lowest Token Cost</span>
                  <p className="text-xs font-bold text-white">
                    {results.find((r) => r.total_token_cost === lowestCost)?.label || "N/A"} (${lowestCost.toFixed(5)})
                  </p>
                </div>
              </div>
            </div>
          </CardHeader>

          <CardContent>
            <div className="overflow-x-auto rounded-xl border border-slate-800">
              <table className="w-full text-left text-xs text-slate-300">
                <thead className="bg-slate-950 text-slate-400 uppercase text-[10px] tracking-wider border-b border-slate-800">
                  <tr>
                    <th className="px-4 py-3 font-semibold">Variant</th>
                    <th
                      className="px-3 py-3 font-semibold cursor-pointer hover:text-white"
                      onClick={() => {
                        setSortBy("avg_overall_score");
                        setSortAsc(!sortAsc);
                      }}
                    >
                      Overall (95% CI) {sortBy === "avg_overall_score" && (sortAsc ? "▲" : "▼")}
                    </th>
                    <th className="px-3 py-3 font-semibold">Context Rel</th>
                    <th className="px-3 py-3 font-semibold">Faithful</th>
                    <th className="px-3 py-3 font-semibold">Ans Rel</th>
                    <th className="px-3 py-3 font-semibold">Grounded</th>
                    <th
                      className="px-3 py-3 font-semibold cursor-pointer hover:text-white"
                      onClick={() => {
                        setSortBy("avg_latency_ms");
                        setSortAsc(!sortAsc);
                      }}
                    >
                      Latency {sortBy === "avg_latency_ms" && (sortAsc ? "▲" : "▼")}
                    </th>
                    <th
                      className="px-3 py-3 font-semibold cursor-pointer hover:text-white"
                      onClick={() => {
                        setSortBy("total_token_cost");
                        setSortAsc(!sortAsc);
                      }}
                    >
                      Cost ($) {sortBy === "total_token_cost" && (sortAsc ? "▲" : "▼")}
                    </th>
                    <th className="px-3 py-3 font-semibold">Cache Hit%</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 bg-slate-900/40">
                  {sortedSummaries.map((s, idx) => {
                    const isBest = s.avg_overall_score === bestScore;
                    return (
                      <tr
                        key={s.variant}
                        className={`hover:bg-slate-800/40 transition-colors ${
                          isBest ? "bg-indigo-950/20" : ""
                        }`}
                      >
                        <td className="px-4 py-3 font-medium text-slate-100 flex items-center gap-2">
                          {isBest && <Award className="w-4 h-4 text-amber-400" />}
                          <span>{s.label}</span>
                        </td>
                        <td className="px-3 py-3 font-mono font-bold text-indigo-300">
                          {s.avg_overall_score.toFixed(3)}
                          {s.ci_lower_95 !== undefined && (
                            <span className="text-[10px] text-slate-500 font-normal ml-1">
                              [{s.ci_lower_95.toFixed(2)}-{s.ci_upper_95?.toFixed(2)}]
                            </span>
                          )}
                        </td>
                        <td className="px-3 py-3 font-mono">{s.avg_context_relevance.toFixed(3)}</td>
                        <td className="px-3 py-3 font-mono">{s.avg_faithfulness.toFixed(3)}</td>
                        <td className="px-3 py-3 font-mono">{s.avg_answer_relevance.toFixed(3)}</td>
                        <td className="px-3 py-3 font-mono">{s.avg_groundedness.toFixed(3)}</td>
                        <td className="px-3 py-3 font-mono">{Math.round(s.avg_latency_ms)}ms</td>
                        <td className="px-3 py-3 font-mono text-emerald-400">${s.total_token_cost.toFixed(5)}</td>
                        <td className="px-3 py-3 font-mono text-slate-400">{(s.cache_hit_rate * 100).toFixed(1)}%</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {/* Paired Significance & Interpretation Note */}
            <div className="mt-4 p-4 rounded-xl bg-slate-950/70 border border-slate-800 text-xs text-slate-400 space-y-2">
              <div className="flex items-center gap-2 text-indigo-300 font-semibold">
                <TrendingUp className="w-4 h-4" /> Statistical Interpretation
              </div>
              <p>
                Metrics are aggregated with paired bootstrap confidence intervals (1,000 resamples at 95% CI). Small differences between variants with overlapping confidence intervals reflect sampling variance on small query samples rather than intrinsic pipeline degradation.
              </p>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
