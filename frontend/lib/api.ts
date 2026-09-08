import axios from 'axios';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface Project {
  id: number;
  name: string;
  description: string;
  created_at: string;
}

export interface RAGConfig {
  id?: number;
  project_id: number;
  chunk_size: number;
  chunk_overlap: number;
  max_tokens: number;

  primary_llm_provider?: string;
  primary_llm_name?: string;
  fallback_llm_provider?: string;
  fallback_llm_name?: string;
  embedding_model?: string;

  temperature?: number;
  top_p?: number;
  max_output_tokens?: number;
  response_style?: string;
  top_k?: number;
  similarity_threshold?: number;
  max_context_tokens?: number;
  answer_only_from_docs?: boolean;
  hallucination_guard?: boolean;

  is_active: boolean;
  created_at?: string;

  // Hybrid Search Config
  use_hybrid_search?: boolean;
  semantic_weight?: number;
}

export interface Document {
  id: number;
  filename: string;
  project_id: number;
  processed: boolean;
  uploaded_at: string;
  content?: string | null;
  file_size_bytes?: number | null;
  page_count?: number | null;
  chunk_count?: number | null;
  chunk_size_used?: number | null;
  embedding_model_used?: string | null;
  version?: number;
  is_active?: boolean;
  processing_status?: string;
  processing_error?: string | null;
  uploaded_by?: number | null;
  redaction_log?: Record<string, any> | null;
  chunking_strategy?: string | null;
  chunking_metrics?: {
    composite_score: number;
    size_compliance: number;
    intrachunk_cohesion: number;
    contextual_coherence: number;
    block_integrity: number;
    reference_completeness: number;
    all_strategy_scores: Record<string, {
      composite: number;
      size_compliance: number;
      intrachunk_cohesion: number;
      contextual_coherence: number;
      block_integrity: number;
      reference_completeness: number;
      chunk_count: number;
    }>;
  } | null;
}

export interface Chunk {
  id: number;
  content: string;
  token_count: number;
}

export interface DebugSearchResult {
  chunk_id: number;
  content: string;
  score: number;
  document_name: string;
}

export interface DebugSearchRequest {
  project_id: number;
  query: string;
  top_k: number;
  similarity_threshold: number;
}

export interface QualityScores {
  hallucination_score: number;
  faithfulness_score: number;
  overall_quality_score: number;
  quality_label: string;
}

export interface AgenticStep {
  step: string;
  strategy: string;
  confidence: number;
  action_taken: string;
  latency_ms: number;
}

export interface ChatMessageResponse {
  session_id: number;
  role: string;
  content: string;
  sources: { source: string; doc_id: number }[];
  usage_metadata: Record<string, unknown> | null;
  query_log_id: number | null;
  quality: QualityScores | null;
  agent_trace?: AgenticStep[];
  attempts?: number;
  strategies_tried?: string[];
  answered?: boolean;
  conflict_detection?: Record<string, any> | null;
  not_found_proof?: string | null;
  ragas_metrics?: {
    context_relevance: { score: number; label: string; description: string };
    faithfulness: { score: number; label: string; description: string };
    answer_relevance: { score: number; label: string; description: string };
    groundedness: { score: number; label: string; description: string };
    overall_score: number;
    hallucination_risk: string;
    evaluation_method?: string;
  } | null;
  output_contract?: {
    format: string;
    max_tokens: number;
    contract_compliant: boolean;
    checks: {
      length_compliant: boolean;
      has_citations: boolean;
      sentence_count_compliant: boolean;
      has_bullet_structure: boolean;
    };
    approximate_tokens: number;
  } | null;
  pipeline_trace?: Record<string, any> | null;
}


export interface ProjectAnalytics {
  total_queries: number;
  avg_latency_ms: number;
  avg_hallucination_score: number;
  avg_faithfulness_score: number;
  citation_engagement_rate: number;
  avg_chunks_before_pruning: number;
  avg_chunks_after_pruning: number;
  avg_pruning_reduction_pct: number;
  hybrid_search_usage_pct: number;
  daily_volume: { date: string; count: number; avg_latency: number }[];
  model_breakdown: { model: string; count: number; avg_latency: number }[];
  quality_daily: {
    date: string;
    avg_hallucination: number | null;
    avg_faithfulness: number | null;
    avg_context_relevance: number | null;
    avg_ragas_faithfulness: number | null;
    avg_answer_relevance: number | null;
    avg_groundedness: number | null;
    avg_overall_ragas: number | null;
    avg_compression_ratio: number | null;
  }[];
  agentic_metrics?: {
    total_agentic_queries: number;
    agentic_success_rate: number;
    avg_agentic_attempts: number;
    most_common_fallbacks: { strategy: string; count: number }[];
  };
  avg_context_relevance: number;
  avg_ragas_faithfulness: number;
  avg_answer_relevance: number;
  avg_groundedness: number;
  avg_compression_ratio: number;
  avg_cache_savings_usd: number;
  sla_metrics_24h?: {
    p95_latency_ms: number;
    sla_warning_threshold_ms: number;
    sla_warning_triggered: boolean;
  };
  latency_breakdown?: {
    avg_ttft_ms: number;
    avg_tpot_ms: number;
  };
  adaptive_router_breakdown?: {
    backend: string;
    count: number;
    pct: number;
  }[];
}

export interface CompareSide {
  provider: string;
  model: string;
  content: string;
  latency_ms: number;
  hallucination_score: number;
  faithfulness_score: number;
  overall_quality_score: number;
  quality_label: string;
  citations_count: number;
}

export interface CompareModelsResponse {
  winner: string;
  left: CompareSide;
  right: CompareSide;
}

export const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

api.interceptors.request.use((config) => {
  if (typeof window !== 'undefined') {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
  }
  return config;
});

api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response && error.response.status === 401) {
      if (typeof window !== 'undefined') {
        localStorage.removeItem('token');
        window.location.href = '/login';
      }
    }
    return Promise.reject(error);
  }
);

export const getProjects = async () => {
  const response = await api.get<Project[]>('/rag/projects/');
  return response.data;
};

export const createProject = async (name: string, description: string) => {
  const response = await api.post<Project>('/rag/projects/', { name, description });
  return response.data;
};

export const deleteProject = async (projectId: number) => {
  const response = await api.delete(`/rag/projects/${projectId}`);
  return response.data;
};

export const getRAGConfig = async (projectId: number) => {
  const response = await api.get<RAGConfig>(`/rag/config/?project_id=${projectId}`);
  return response.data;
};

export const updateRAGConfig = async (config: Partial<RAGConfig> & { project_id: number }) => {
  const response = await api.post<RAGConfig>('/rag/config/', config);
  return response.data;
};

export const patchProjectRagConfig = async (
  projectId: number,
  patch: Partial<RAGConfig>
): Promise<RAGConfig> => {
  const response = await api.patch<RAGConfig>(`/rag/projects/${projectId}/config`, patch);
  return response.data;
};

export const uploadDocument = async (projectId: number, file: File) => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('project_id', projectId.toString());

  const response = await api.post(`/rag/ingest/upload`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data as { message: string; doc_id: number; status: string };
};

export const getDocuments = async (projectId: number, includeInactive = false) => {
  const response = await api.get<Document[]>(`/rag/ingest/`, {
    params: { project_id: projectId, include_inactive: includeInactive },
  });
  return response.data;
};

export const getDocumentStatus = async (docId: number) => {
  const response = await api.get<{ status: string; error: string | null; processed: boolean }>(
    `/rag/ingest/documents/${docId}/status`
  );
  return response.data;
};

export const deleteDocument = async (docId: number) => {
  const response = await api.delete(`/rag/ingest/documents/${docId}`);
  return response.data as { deleted: boolean; doc_id: number };
};

export const rechunkDocument = async (docId: number, chunkSize: number, chunkOverlap: number) => {
  const response = await api.post(`/rag/ingest/documents/${docId}/rechunk`, {
    chunk_size: chunkSize,
    chunk_overlap: chunkOverlap,
  });
  return response.data as { status: string; doc_id: number };
};

export const getDocumentChunksPaged = async (docId: number, page = 1, limit = 20) => {
  const response = await api.get<{
    chunks: Chunk[];
    total: number;
    page: number;
    pages: number;
  }>(`/rag/ingest/documents/${docId}/chunks`, { params: { page, limit } });
  return response.data;
};

export const getDocumentChunks = async (documentId: number) => {
  const response = await api.get<Chunk[]>(`/rag/inspector/documents/${documentId}/chunks`);
  return response.data;
};

export interface DebugSearchResponse {
  results: DebugSearchResult[];
  query_analysis?: Record<string, any> | null;
  pipeline_trace?: Record<string, any> | null;
}

export const debugSearch = async (data: DebugSearchRequest): Promise<DebugSearchResponse> => {
  const response = await api.post<DebugSearchResponse>('/rag/inspector/search', data);
  return response.data;
};

export const sendMessage = async (
  content: string,
  projectId?: number,
  sessionId?: number,
  temp?: number,
  modelProvider: string = 'groq',
  modelName: string = 'llama-3.3-70b-versatile',
  historyLimit: number = 5,
  projectContextLimit: number = 2,
  contextSessionIds: number[] = [],
  title?: string
): Promise<ChatMessageResponse> => {
  const response = await api.post<ChatMessageResponse>('/chat/message', {
    content,
    project_id: projectId ?? null,
    session_id: sessionId ?? null,
    temperature: temp,
    model_provider: modelProvider,
    model_name: modelName,
    history_limit: historyLimit,
    project_context_limit: projectContextLimit,
    context_session_ids: contextSessionIds,
    title: title ?? null,
  });
  return response.data;
};

export const sendAgenticMessage = async (
  content: string,
  projectId?: number,
  sessionId?: number,
  temp?: number,
  modelProvider: string = 'groq',
  modelName: string = 'llama-3.3-70b-versatile',
  historyLimit: number = 5,
  projectContextLimit: number = 2,
  contextSessionIds: number[] = [],
  title?: string
): Promise<ChatMessageResponse> => {
  const response = await api.post<ChatMessageResponse>('/api/query/agentic', {
    content,
    project_id: projectId ?? null,
    session_id: sessionId ?? null,
    temperature: temp,
    model_provider: modelProvider,
    model_name: modelName,
    history_limit: historyLimit,
    project_context_limit: projectContextLimit,
    context_session_ids: contextSessionIds,
    title: title ?? null,
  });
  return response.data;
};


export const compareModels = async (projectId: number, query: string) => {
  const response = await api.post<CompareModelsResponse>('/chat/compare-models', {
    project_id: projectId,
    query,
  });
  return response.data;
};

export const getProjectAnalytics = async (projectId: number, days: 7 | 30 | 90 = 30) => {
  const response = await api.get<ProjectAnalytics>(`/api/analytics/${projectId}`, {
    params: { days },
  });
  return response.data;
};

export const trackCitationClick = async (queryLogId: number, citationIndex = 0) => {
  await api.post('/api/analytics/citation-click', {
    query_log_id: queryLogId,
    citation_index: citationIndex,
  });
};

export const getSessions = async (projectId?: number) => {
  const response = await api.get('/chat/sessions', { params: { project_id: projectId } });
  return response.data;
};

export const deleteSession = async (sessionId: number) => {
  const response = await api.delete(`/chat/sessions/${sessionId}`);
  return response.data;
};

export const getHistory = async (sessionId: number) => {
  const response = await api.get(`/chat/history/${sessionId}`);
  return response.data;
};

// ==========================================
// AutoRAGBench Evaluation Studio Types & API
// ==========================================

export interface ModelProfile {
  id: string;
  name: string;
  provider: string;
  model_name: string;
  description: string;
  context_window: number;
  cost_per_1m_input: number;
  cost_per_1m_output: number;
  is_available: boolean;
  badge: string;
  recommended_for_benchmark: boolean;
}

export interface PipelineVariantInfo {
  name: string;
  label: string;
  description: string;
}

export interface BenchmarkDatasetInfo {
  id: string;
  name: string;
  description: string;
  query_count: number;
  recommended: boolean;
}

export interface BenchmarkStartRequest {
  project_id: number;
  model_profile_id: string;
  variants: string[];
  dataset_type: string;
  custom_queries?: Array<{ id?: string; query: string; category?: string }>;
  delay_ms?: number;
  resume?: boolean;
}

export interface BenchmarkStatusResponse {
  job_id: string;
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';
  model_name: string;
  model_provider: string;
  progress_pct: number;
  completed_queries: number;
  total_queries: number;
  current_variant: string;
  current_query: string;
  elapsed_seconds: number;
  live_events: Array<{
    event: string;
    variant: string;
    variant_label: string;
    query_id: string;
    query_text: string;
    category: string;
    ragas_overall: number;
    latency_ms: number;
    token_cost: number;
    error?: string | null;
  }>;
  error?: string | null;
}

export interface BenchmarkVariantSummary {
  variant: string;
  label: string;
  description: string;
  total_queries: number;
  avg_latency_ms: number;
  avg_context_relevance: number;
  avg_faithfulness: number;
  avg_answer_relevance: number;
  avg_groundedness: number;
  avg_overall_score: number;
  avg_recall_at_5: number;
  avg_precision_at_5: number;
  avg_mrr: number;
  avg_ndcg_at_5: number;
  total_token_cost: number;
  cache_hit_rate: number;
  error_rate: number;
  ci_lower_95?: number;
  ci_upper_95?: number;
}

export interface BenchmarkResultsResponse {
  job_id: string;
  status: string;
  model_name: string;
  model_provider: string;
  elapsed_seconds: number;
  summaries: BenchmarkVariantSummary[];
  output_file?: string;
}

export const getBenchmarkModels = async (): Promise<ModelProfile[]> => {
  const res = await api.get<{ models: ModelProfile[] }>('/api/benchmark/models');
  return res.data.models;
};

export const getBenchmarkVariants = async (): Promise<PipelineVariantInfo[]> => {
  const res = await api.get<{ variants: PipelineVariantInfo[] }>('/api/benchmark/variants');
  return res.data.variants;
};

export const getBenchmarkDatasets = async (): Promise<BenchmarkDatasetInfo[]> => {
  const res = await api.get<{ datasets: BenchmarkDatasetInfo[] }>('/api/benchmark/datasets');
  return res.data.datasets;
};

export const startBenchmark = async (req: BenchmarkStartRequest): Promise<{ job_id: string; status: string; message: string }> => {
  const res = await api.post('/api/benchmark/start', req);
  return res.data;
};

export const getBenchmarkStatus = async (jobId: string): Promise<BenchmarkStatusResponse> => {
  const res = await api.get<BenchmarkStatusResponse>(`/api/benchmark/status/${jobId}`);
  return res.data;
};

export const getBenchmarkResults = async (jobId: string): Promise<BenchmarkResultsResponse> => {
  const res = await api.get<BenchmarkResultsResponse>(`/api/benchmark/results/${jobId}`);
  return res.data;
};

export const cancelBenchmark = async (jobId: string) => {
  const res = await api.post(`/api/benchmark/cancel/${jobId}`);
  return res.data;
};

export default api;
