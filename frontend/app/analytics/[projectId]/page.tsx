"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useAuth } from "@/lib/auth-context";
import { getProjectAnalytics, getProjects, type ProjectAnalytics, type Project } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import { ArrowLeft, BarChart3 } from "lucide-react";


const COLORS = ["#6366f1", "#22c55e", "#f97316", "#ec4899", "#14b8a6"];

export default function ProjectAnalyticsPage() {
  const params = useParams();
  const router = useRouter();
  const projectId = Number(params.projectId);
  const { user, isLoading } = useAuth();
  const [days, setDays] = useState<7 | 30 | 90>(30);
  const [data, setData] = useState<ProjectAnalytics | null>(null);
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!isLoading && (!user || user.role !== "admin")) {
      if (user && user.role !== "admin") router.replace("/chat");
      else router.replace("/login");
    }
  }, [user, isLoading, router]);

  useEffect(() => {
    if (user?.role === "admin") {
      getProjects().then(setProjects).catch(() => {});
    }
  }, [user]);

  useEffect(() => {
    if (!user || user.role !== "admin" || !Number.isFinite(projectId)) return;
    setLoading(true);
    getProjectAnalytics(projectId, days)
      .then(setData)
      .catch(() => setData(null))
      .finally(() => setLoading(false));
  }, [user, projectId, days]);

  if (isLoading || !user || user.role !== "admin") {
    return (
      <div className="flex min-h-screen items-center justify-center bg-slate-50 dark:bg-slate-950">
        <div className="h-40 w-full max-w-4xl animate-pulse rounded-xl bg-muted" />
      </div>
    );
  }

  const projectName = projects.find((p) => p.id === projectId)?.name ?? `Project ${projectId}`;

  const pieData =
    data?.model_breakdown.map((m) => ({
      name: m.model,
      value: m.count,
    })) ?? [];

  const qualityLine =
    data?.quality_daily.map((d) => ({
      date: d.date,
      hallucination: d.avg_hallucination ?? 0,
      faithfulness: d.avg_faithfulness ?? 0,
      context_relevance: d.avg_context_relevance ?? 0,
      ragas_faithfulness: d.avg_ragas_faithfulness ?? 0,
      answer_relevance: d.avg_answer_relevance ?? 0,
      groundedness: d.avg_groundedness ?? 0,
      overall_ragas: d.avg_overall_ragas ?? 0,
      compression_ratio: d.avg_compression_ratio ?? 0,
    })) ?? [];

  return (
    <div className="min-h-screen bg-slate-50 px-4 py-8 dark:bg-slate-950 md:px-10">
      <div className="mx-auto max-w-6xl space-y-6">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <Link
              href="/admin"
              className="mb-2 inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground"
            >
              <ArrowLeft className="h-4 w-4" />
              Back to admin
            </Link>
            <h1 className="flex items-center gap-2 text-2xl font-bold md:text-3xl">
              <BarChart3 className="h-8 w-8 text-indigo-600" />
              Analytics — {projectName}
            </h1>
          </div>
          <div className="flex gap-2">
            {([7, 30, 90] as const).map((d) => (
              <Button key={d} variant={days === d ? "default" : "outline"} size="sm" onClick={() => setDays(d)}>
                {d}d
              </Button>
            ))}
          </div>
        </div>

        <ErrorBoundary section="Analytics KPIs">
          {loading ? (
            <div className="grid gap-4 md:grid-cols-4">
              {[1, 2, 3, 4, 5, 6, 7, 8].map((i) => (
                <div key={i} className="h-28 animate-pulse rounded-xl bg-muted" />
              ))}
            </div>
          ) : (
            <div className="grid gap-4 md:grid-cols-4">
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium text-muted-foreground">Total queries</CardTitle>
                </CardHeader>
                <CardContent className="text-2xl font-bold">{data?.total_queries ?? 0}</CardContent>
              </Card>
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium text-muted-foreground">Avg latency</CardTitle>
                </CardHeader>
                <CardContent className="text-2xl font-bold">{Math.round(data?.avg_latency_ms ?? 0)} ms</CardContent>
              </Card>
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium text-muted-foreground">Hallucination risk</CardTitle>
                </CardHeader>
                <CardContent className="text-2xl font-bold">
                  {(data?.avg_hallucination_score ?? 0).toFixed(2)}
                  <span className="ml-2 text-xs font-normal text-muted-foreground">lower is better</span>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium text-muted-foreground">Citation engagement</CardTitle>
                </CardHeader>
                <CardContent className="text-2xl font-bold">
                  {((data?.citation_engagement_rate ?? 0) * 100).toFixed(0)}%
                </CardContent>
              </Card>

              {/* Advanced Enterprise Retrieval Layer KPIs */}
              <Card className="border-indigo-100 bg-indigo-50/20 dark:border-indigo-950 dark:bg-indigo-950/10">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium text-indigo-700 dark:text-indigo-300">Hybrid Search usage</CardTitle>
                </CardHeader>
                <CardContent className="text-2xl font-bold text-indigo-600 dark:text-indigo-400">
                  {Math.round(data?.hybrid_search_usage_pct ?? 0)}%
                  <span className="ml-2 text-[10px] font-normal text-muted-foreground">FAISS + BM25</span>
                </CardContent>
              </Card>
              <Card className="border-indigo-100 bg-indigo-50/20 dark:border-indigo-950 dark:bg-indigo-950/10">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium text-indigo-700 dark:text-indigo-300">Context pruning saving</CardTitle>
                </CardHeader>
                <CardContent className="text-2xl font-bold text-indigo-600 dark:text-indigo-400">
                  {Math.round(data?.avg_pruning_reduction_pct ?? 0)}%
                  <span className="ml-2 text-[10px] font-normal text-muted-foreground">less LLM tokens</span>
                </CardContent>
              </Card>
              <Card className="border-indigo-100 bg-indigo-50/20 dark:border-indigo-950 dark:bg-indigo-950/10">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium text-indigo-700 dark:text-indigo-300">Avg chunks retrieved</CardTitle>
                </CardHeader>
                <CardContent className="text-2xl font-bold text-indigo-600 dark:text-indigo-400">
                  {(data?.avg_chunks_before_pruning ?? 0).toFixed(1)}
                  <span className="ml-2 text-[10px] font-normal text-muted-foreground">before prune</span>
                </CardContent>
              </Card>
              <Card className="border-indigo-100 bg-indigo-50/20 dark:border-indigo-950 dark:bg-indigo-950/10">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium text-indigo-700 dark:text-indigo-300">Avg chunks sent</CardTitle>
                </CardHeader>
                <CardContent className="text-2xl font-bold text-indigo-600 dark:text-indigo-400">
                  {(data?.avg_chunks_after_pruning ?? 0).toFixed(1)}
                  <span className="ml-2 text-[10px] font-normal text-muted-foreground">to LLM context</span>
                </CardContent>
              </Card>
              <Card className="border-emerald-100 bg-emerald-50/20 dark:border-emerald-950 dark:bg-emerald-950/10">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium text-emerald-705 dark:text-emerald-300">Prompt cache monthly savings</CardTitle>
                </CardHeader>
                <CardContent className="text-2xl font-bold text-emerald-600 dark:text-emerald-450">
                  ${(data?.avg_cache_savings_usd ?? 0.0).toFixed(2)}
                  <span className="ml-2 text-[10px] font-normal text-muted-foreground">USD / Month</span>
                </CardContent>
              </Card>

              {/* Agentic RAG Performance Card */}
              {data?.agentic_metrics && data.agentic_metrics.total_agentic_queries > 0 && (
                <Card className="border-indigo-200 bg-indigo-50/30 dark:border-indigo-900 dark:bg-indigo-950/20 col-span-full md:col-span-4 grid gap-4 grid-cols-1 md:grid-cols-3 p-4">
                  <div className="flex flex-col justify-center">
                    <h3 className="text-xs font-semibold text-indigo-700 dark:text-indigo-400 uppercase tracking-wider">Agentic Query Success Rate</h3>
                    <p className="text-3xl font-bold text-indigo-600 dark:text-indigo-300 mt-1">{data.agentic_metrics.agentic_success_rate}%</p>
                    <p className="text-xs text-muted-foreground mt-1">Queries answered successfully vs cannot_answer ({data.agentic_metrics.total_agentic_queries} total)</p>
                  </div>
                  <div className="flex flex-col justify-center border-l dark:border-indigo-900 pl-4">
                    <h3 className="text-xs font-semibold text-indigo-700 dark:text-indigo-400 uppercase tracking-wider">Average Attempts / Query</h3>
                    <p className="text-3xl font-bold text-indigo-600 dark:text-indigo-300 mt-1">{data.agentic_metrics.avg_agentic_attempts}</p>
                    <p className="text-xs text-muted-foreground mt-1">Goal is 1.0 (lower is better, max 3.0)</p>
                  </div>
                  <div className="flex flex-col justify-center border-l dark:border-indigo-900 pl-4">
                    <h3 className="text-xs font-semibold text-indigo-700 dark:text-indigo-400 uppercase tracking-wider">Common Fallbacks Tried</h3>
                    <div className="flex flex-wrap gap-1.5 mt-2">
                      {data.agentic_metrics.most_common_fallbacks && data.agentic_metrics.most_common_fallbacks.length > 0 ? (
                        data.agentic_metrics.most_common_fallbacks.map((item: any, idx: number) => (
                          <Badge key={idx} variant="outline" className="text-[10px] uppercase border-indigo-200 text-indigo-700 dark:text-indigo-355 dark:border-indigo-800">
                            {item.strategy} ({item.count})
                          </Badge>
                        ))
                      ) : (
                        <span className="text-xs text-muted-foreground">No fallback attempts logged yet</span>
                      )}
                    </div>
                    <p className="text-xs text-muted-foreground mt-2">Strategies triggered during replanning steps</p>
                  </div>
                </Card>
              )}
            </div>
          )}
        </ErrorBoundary>

        <ErrorBoundary section="Charts">
          {loading ? (
            <div className="grid gap-4 md:grid-cols-2">
              <div className="h-72 animate-pulse rounded-xl bg-muted" />
              <div className="h-72 animate-pulse rounded-xl bg-muted" />
            </div>
          ) : (
            <div className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle>Query volume</CardTitle>
                </CardHeader>
                <CardContent className="h-72 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={data?.daily_volume ?? []}>
                      <defs>
                        <linearGradient id="vol" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#6366f1" stopOpacity={0.35} />
                          <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                      <XAxis dataKey="date" tick={{ fontSize: 11 }} />
                      <YAxis tick={{ fontSize: 11 }} />
                      <Tooltip />
                      <Area type="monotone" dataKey="count" stroke="#6366f1" fill="url(#vol)" name="Queries" />
                    </AreaChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              <div className="grid gap-6 md:grid-cols-2">
                <Card>
                  <CardHeader>
                    <CardTitle>Latency trend</CardTitle>
                  </CardHeader>
                  <CardContent className="h-64 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={data?.daily_volume ?? []}>
                        <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                        <XAxis dataKey="date" tick={{ fontSize: 11 }} />
                        <YAxis tick={{ fontSize: 11 }} />
                        <Tooltip />
                        <Line type="monotone" dataKey="avg_latency" stroke="#22c55e" dot={false} name="Avg ms" />
                      </LineChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader>
                    <CardTitle>Model usage</CardTitle>
                  </CardHeader>
                  <CardContent className="h-64 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie data={pieData} dataKey="value" nameKey="name" outerRadius={90} label>
                          {pieData.map((_, i) => (
                            <Cell key={i} fill={COLORS[i % COLORS.length]} />
                          ))}
                        </Pie>
                        <Tooltip />
                        <Legend />
                      </PieChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              </div>

              <Card>
                <CardHeader>
                  <CardTitle>Response quality trend</CardTitle>
                </CardHeader>
                <CardContent className="h-72 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={qualityLine}>
                      <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                      <XAxis dataKey="date" tick={{ fontSize: 11 }} />
                      <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} />
                      <Tooltip />
                      <Line
                        type="monotone"
                        dataKey="hallucination"
                        stroke="#ef4444"
                        dot={false}
                        name="Hallucination risk"
                      />
                      <Line
                        type="monotone"
                        dataKey="faithfulness"
                        stroke="#3b82f6"
                        dot={false}
                        name="Faithfulness"
                      />
                      <ReferenceLine
                        y={0.15}
                        stroke="#94a3b8"
                        strokeDasharray="4 4"
                        label={{ value: "Hallucination target", position: "insideTopRight" }}
                      />
                      <ReferenceLine
                        y={0.8}
                        stroke="#22c55e"
                        strokeDasharray="4 4"
                        label={{ value: "Faithfulness target", position: "insideBottomRight" }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>RAGAS daily metrics trend</CardTitle>
                </CardHeader>
                <CardContent className="h-72 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={qualityLine}>
                      <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                      <XAxis dataKey="date" tick={{ fontSize: 11 }} />
                      <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} />
                      <Tooltip />
                      <Legend />
                      <Line
                        type="monotone"
                        dataKey="context_relevance"
                        stroke="#6366f1"
                        dot={false}
                        name="Context Relevance"
                      />
                      <Line
                        type="monotone"
                        dataKey="ragas_faithfulness"
                        stroke="#22c55e"
                        dot={false}
                        name="Faithfulness"
                      />
                      <Line
                        type="monotone"
                        dataKey="answer_relevance"
                        stroke="#f59e0b"
                        dot={false}
                        name="Answer Relevance"
                      />
                      <Line
                        type="monotone"
                        dataKey="groundedness"
                        stroke="#ec4899"
                        dot={false}
                        name="Groundedness"
                      />
                      <Line
                        type="monotone"
                        dataKey="overall_ragas"
                        stroke="#14b8a6"
                        strokeWidth={2}
                        dot={false}
                        name="Overall RAGAS Score"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Context compression ratio trend</CardTitle>
                </CardHeader>
                <CardContent className="h-72 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={qualityLine}>
                      <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                      <XAxis dataKey="date" tick={{ fontSize: 11 }} />
                      <YAxis domain={[0, 1]} tickFormatter={(val) => `${(val * 100).toFixed(0)}%`} tick={{ fontSize: 11 }} />
                      <Tooltip formatter={(value: any) => `${(Number(value) * 100).toFixed(1)}%`} />
                      <Legend />
                      <Line
                        type="monotone"
                        dataKey="compression_ratio"
                        stroke="#8b5cf6"
                        strokeWidth={2}
                        dot={false}
                        name="Context Compression Ratio"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </div>
          )}
        </ErrorBoundary>
      </div>
    </div>
  );
}
