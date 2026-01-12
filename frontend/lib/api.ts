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

  // Advanced Settings
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
}

// Inspector Interfaces
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


const api = axios.create({
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

export const uploadDocument = async (projectId: number, file: File) => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('project_id', projectId.toString());

  const response = await api.post(`/rag/ingest/upload`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};

export const getDocuments = async (projectId: number) => {
  const response = await api.get<Document[]>(`/rag/ingest/?project_id=${projectId}`);
  return response.data;
};

// Inspector API
export const getDocumentChunks = async (documentId: number) => {
  const response = await api.get<Chunk[]>(`/rag/inspector/documents/${documentId}/chunks`);
  return response.data;
};

export const debugSearch = async (data: DebugSearchRequest) => {
  const response = await api.post<DebugSearchResult[]>('/rag/inspector/search', data);
  return response.data;
};


export const sendMessage = async (content: string, projectId?: number, sessionId?: number, temp?: number, modelProvider: string = "groq", modelName: string = "llama-3.3-70b-versatile", historyLimit: number = 5, projectContextLimit: number = 2, title?: string) => {
  const response = await api.post('/chat/message', null, {
    params: {
      content,
      project_id: projectId,
      session_id: sessionId,
      temperature: temp,
      model_provider: modelProvider,
      model_name: modelName,
      history_limit: historyLimit,
      project_context_limit: projectContextLimit,
      title: title
    }
  });
  return response.data;
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
