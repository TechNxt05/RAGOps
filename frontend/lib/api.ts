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

export interface ChatMessageResponse {
  session_id: number;
  role: string;
  content: string;
  sources: { source: string; doc_id: number }[];
  usage_metadata: Record<string, unknown> | null;
  query_log_id: number | null;
  quality: QualityScores | null;
}

export interface ProjectAnalytics {
  total_queries: number;
  avg_latency_ms: number;
  avg_hallucination_score: number;
  avg_faithfulness_score: number;
  citation_engagement_rate: number;
  daily_volume: { date: string; count: number; avg_latency: number }[];
  model_breakdown: { model: string; count: number; avg_latency: number }[];
  quality_daily: {
    date: string;
    avg_hallucination: number | null;
    avg_faithfulness: number | null;
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

export const debugSearch = async (data: DebugSearchRequest) => {
  const response = await api.post<DebugSearchResult[]>('/rag/inspector/search', data);
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

export default api;
