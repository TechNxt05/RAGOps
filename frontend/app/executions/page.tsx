"use client";

import React, { useState } from "react";
import {
  GitCommit,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  Clock,
  ArrowRight,
  Cpu,
  Layers,
  Database,
  Filter,
  ShieldCheck,
  Zap,
  Activity,
  FileSearch,
} from "lucide-react";

interface PipelineNode {
  id: string;
  name: string;
  stage: string;
  status: "success" | "warning" | "failed" | "pending";
  latency_ms: number;
  description: string;
  details: Record<string, any>;
}

export default function ExecutionsPage() {
  const [selectedNode, setSelectedNode] = useState<PipelineNode | null>(null);

  // Sample execution flow matching real pipeline traces
  const pipelineNodes: PipelineNode[] = [
    {
      id: "node-1",
      name: "Query Input",
      stage: "query_input",
      status: "success",
      latency_ms: 1.2,
      description: "Incoming user query received and sanitized.",
      details: {
        raw_query: "Calculate total Q3 marketing budget across all approved invoices in INV-2024",
        query_length: 74,
        timestamp: "2026-07-28 03:15:00 UTC",
      },
    },
    {
      id: "node-2",
      name: "TurnTypeRouter",
      stage: "turn_type_router",
      status: "success",
      latency_ms: 3.4,
      description: "Classifies turn intent (NEW_QUERY, FOLLOW_UP, CHIT_CHAT).",
      details: {
        classified_turn: "NEW_QUERY",
        confidence: 0.98,
        session_has_context: false,
      },
    },
    {
      id: "node-3",
      name: "AdaptiveRetrievalRouter",
      stage: "adaptive_router",
      status: "success",
      latency_ms: 4.8,
      description: "Feature-score query classifier dispatching optimal backend (BM25, Computation, Graph, Hybrid).",
      details: {
        chosen_backend: "computation",
        confidence: 0.92,
        feature_scores: {
          has_id_pattern: 0.8,
          has_agg_keywords: 0.92,
          is_multihop_question: 0.2,
          keyword_density: 0.75,
        },
        fallback_triggered: false,
        reasoning: "Structured aggregation keywords detected (total, sum, invoice numbers). Dispatched to ComputationRouter.",
      },
    },
    {
      id: "node-4",
      name: "Retriever",
      stage: "retrieval",
      status: "success",
      latency_ms: 145.6,
      description: "Database computation engine & hybrid vector/keyword search.",
      details: {
        retrieval_source: "computation_engine",
        chunks_retrieved: 4,
        top_k: 4,
        similarity_threshold: 0.0,
      },
    },
    {
      id: "node-5",
      name: "BGE Reranker",
      stage: "reranker",
      status: "success",
      latency_ms: 68.2,
      description: "Cross-encoder relevance re-scoring.",
      details: {
        top_scores: [0.96, 0.91, 0.87, 0.82],
        rerank_model: "bge-reranker-large",
      },
    },
    {
      id: "node-6",
      name: "Contextual Compressor",
      stage: "compressor",
      status: "success",
      latency_ms: 22.1,
      description: "Token-budget pruning and prompt compression.",
      details: {
        chunks_before: 8,
        chunks_after: 4,
        reduction_pct: "50.0%",
        estimated_token_savings: 840,
      },
    },
    {
      id: "node-7",
      name: "Confidence Gate",
      stage: "confidence_gate",
      status: "warning",
      latency_ms: 8.5,
      description: "Pre-generation hallucination & threshold safety gate.",
      details: {
        gate_passed: true,
        max_confidence: 0.74,
        threshold: 0.65,
        borderline_warning: "Borderline confidence (0.74 vs 0.65 threshold). Proceeding with self-correction log.",
      },
    },
    {
      id: "node-8",
      name: "LLM Generation",
      stage: "llm_generation",
      status: "success",
      latency_ms: 840.0,
      description: "Language model response generation with TTFT / TPOT tracking.",
      details: {
        model: "groq/llama-3.3-70b-versatile",
        ttft_ms: 235.2,
        tpot_ms: 12.4,
        tokens_generated: 48,
        reasoning_effort: "high",
      },
    },
    {
      id: "node-9",
      name: "Claim Verifier",
      stage: "claim_verification",
      status: "success",
      latency_ms: 45.0,
      description: "NLI-based sentence claim citation verification.",
      details: {
        verification_score: 1.0,
        claims_checked: 3,
        verified_claims: 3,
        unverified_claims: [],
      },
    },
    {
      id: "node-10",
      name: "Output Contract",
      stage: "output",
      status: "success",
      latency_ms: 2.1,
      description: "Final output contract verification & schema formatting.",
      details: {
        contract_compliant: true,
        citations_attached: 4,
        format: "concise",
      },
    },
  ];

  const totalLatency = pipelineNodes.reduce((acc, n) => acc + n.latency_ms, 0);

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "success":
        return (
          <span className="flex items-center gap-1 text-xs font-semibold px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
            <CheckCircle2 className="w-3 h-3" /> PASS
          </span>
        );
      case "warning":
        return (
          <span className="flex items-center gap-1 text-xs font-semibold px-2 py-0.5 rounded-full bg-amber-500/10 text-amber-400 border border-amber-500/20">
            <AlertTriangle className="w-3 h-3" /> BORDERLINE
          </span>
        );
      case "failed":
        return (
          <span className="flex items-center gap-1 text-xs font-semibold px-2 py-0.5 rounded-full bg-rose-500/10 text-rose-400 border border-rose-500/20">
            <XCircle className="w-3 h-3" /> FAIL
          </span>
        );
      default:
        return (
          <span className="flex items-center gap-1 text-xs font-semibold px-2 py-0.5 rounded-full bg-slate-500/10 text-slate-400 border border-slate-500/20">
            <Clock className="w-3 h-3" /> PENDING
          </span>
        );
    }
  };

  const getNodeIcon = (stage: string) => {
    switch (stage) {
      case "query_input":
        return <FileSearch className="w-4 h-4 text-blue-400" />;
      case "turn_type_router":
        return <GitCommit className="w-4 h-4 text-indigo-400" />;
      case "adaptive_router":
        return <Cpu className="w-4 h-4 text-purple-400" />;
      case "retrieval":
        return <Database className="w-4 h-4 text-cyan-400" />;
      case "reranker":
        return <Layers className="w-4 h-4 text-sky-400" />;
      case "compressor":
        return <Filter className="w-4 h-4 text-amber-400" />;
      case "confidence_gate":
        return <ShieldCheck className="w-4 h-4 text-orange-400" />;
      case "llm_generation":
        return <Zap className="w-4 h-4 text-emerald-400" />;
      case "claim_verification":
        return <Activity className="w-4 h-4 text-teal-400" />;
      default:
        return <CheckCircle2 className="w-4 h-4 text-slate-400" />;
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-6 space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-5">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
            <GitCommit className="w-6 h-6 text-purple-400" /> Pipeline Execution Graph & Telemetry
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Real-time stage-by-stage execution trace, latency breakdown, and Adaptive Router decision inspector.
          </p>
        </div>

        <div className="flex items-center gap-4 text-sm bg-slate-900 border border-slate-800 rounded-lg px-4 py-2">
          <div>
            <span className="text-slate-400">Total Latency:</span>{" "}
            <span className="font-mono font-bold text-emerald-400">{totalLatency.toFixed(1)} ms</span>
          </div>
          <div className="h-4 w-px bg-slate-800" />
          <div>
            <span className="text-slate-400">24h SLA p95:</span>{" "}
            <span className="font-mono font-bold text-sky-400">1,140 ms</span>
          </div>
        </div>
      </div>

      {/* Execution Flow Diagram (Nodes + Connectors) */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-6 overflow-x-auto">
        <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-400 mb-6 flex items-center gap-2">
          <Activity className="w-4 h-4 text-purple-400" /> Interactive Execution Pipeline Graph
        </h2>

        <div className="flex items-center gap-3 min-w-max pb-4">
          {pipelineNodes.map((node, index) => {
            const isSelected = selectedNode?.id === node.id;
            return (
              <React.Fragment key={node.id}>
                <button
                  onClick={() => setSelectedNode(node)}
                  className={`flex flex-col gap-2 p-4 rounded-xl border transition-all text-left w-52 hover:scale-[1.02] cursor-pointer ${
                    isSelected
                      ? "bg-purple-950/40 border-purple-500 shadow-lg shadow-purple-500/10"
                      : "bg-slate-900 border-slate-800 hover:border-slate-700"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div className="p-2 rounded-lg bg-slate-800/80">{getNodeIcon(node.stage)}</div>
                    {getStatusBadge(node.status)}
                  </div>

                  <div>
                    <h3 className="font-semibold text-sm text-slate-100">{node.name}</h3>
                    <p className="text-xs font-mono text-slate-400 mt-1">{node.latency_ms} ms</p>
                  </div>
                </button>

                {index < pipelineNodes.length - 1 && (
                  <ArrowRight className="w-5 h-5 text-slate-600 shrink-0" />
                )}
              </React.Fragment>
            );
          })}
        </div>
      </div>

      {/* Stage Details Inspector */}
      {selectedNode ? (
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-4">
            <div className="flex items-center gap-3">
              <div className="p-2.5 rounded-lg bg-purple-500/10 border border-purple-500/20">
                {getNodeIcon(selectedNode.stage)}
              </div>
              <div>
                <h2 className="text-lg font-bold text-white">{selectedNode.name} Inspector</h2>
                <p className="text-xs text-slate-400">{selectedNode.description}</p>
              </div>
            </div>
            {getStatusBadge(selectedNode.status)}
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-slate-950 p-4 rounded-lg border border-slate-800">
              <h4 className="text-xs font-semibold text-slate-400 uppercase mb-2">Stage Telemetry</h4>
              <div className="space-y-1 text-sm font-mono">
                <div className="flex justify-between py-1 border-b border-slate-900">
                  <span className="text-slate-400">Stage Identifier:</span>
                  <span className="text-purple-300">{selectedNode.stage}</span>
                </div>
                <div className="flex justify-between py-1 border-b border-slate-900">
                  <span className="text-slate-400">Execution Latency:</span>
                  <span className="text-emerald-400">{selectedNode.latency_ms} ms</span>
                </div>
                <div className="flex justify-between py-1">
                  <span className="text-slate-400">Execution Status:</span>
                  <span className="text-slate-200 capitalize">{selectedNode.status}</span>
                </div>
              </div>
            </div>

            <div className="bg-slate-950 p-4 rounded-lg border border-slate-800">
              <h4 className="text-xs font-semibold text-slate-400 uppercase mb-2">Stage Metadata & Decision Rationale</h4>
              <pre className="text-xs font-mono text-slate-300 overflow-x-auto p-2 bg-slate-900/80 rounded border border-slate-800/80">
                {JSON.stringify(selectedNode.details, null, 2)}
              </pre>
            </div>
          </div>
        </div>
      ) : (
        <div className="bg-slate-900/40 border border-slate-800/80 rounded-xl p-8 text-center text-slate-400">
          <p className="text-sm">Click any pipeline node in the execution graph above to inspect detailed feature scores, stage latencies, and decision rationales.</p>
        </div>
      )}
    </div>
  );
}
