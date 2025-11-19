import apiClient from './client';

export interface QueryRequest {
  query: string;
  options?: Record<string, any>;
}

export interface PipelineRunResponse {
  run_id: string;
  status: string;
  message: string;
  created_at: string;
}

export interface StageFlags {
  ingestion: boolean;
  validation: boolean;
  transformation: boolean;
  completed: boolean;
}

export interface PipelineStatusResponse {
  run_id: string;
  query: string;
  status: string;
  progress: number;
  current_stage: string;
  message: string;
  started_at: string;
  completed_at?: string;
  error?: string;
  stage_flags: StageFlags;
}

export interface DataFrameInfo {
  index: number;
  shape: [number, number];
  columns: string[];
  preview: Record<string, any>[];
}

export interface PipelineResultsResponse {
  run_id: string;
  status: string;
  dataframes: DataFrameInfo[];
  enrichment_features: string[];
  validation_report?: any;
  transformation_report?: any;
}

export interface HistoryItem {
  run_id: string;
  query: string;
  status: string;
  created_at: string;
  completed_at?: string;
  duration?: number;
}

export const pipelineApi = {
  runPipeline: async (data: QueryRequest): Promise<PipelineRunResponse> => {
    const response = await apiClient.post('/api/v1/pipeline/run', data);
    return response.data;
  },

  getStatus: async (runId: string): Promise<PipelineStatusResponse> => {
    const response = await apiClient.get(`/api/v1/pipeline/status/${runId}`);
    return response.data;
  },

  getResults: async (runId: string): Promise<PipelineResultsResponse> => {
    const response = await apiClient.get(`/api/v1/pipeline/results/${runId}`);
    return response.data;
  },

  getHistory: async (): Promise<HistoryItem[]> => {
    const response = await apiClient.get('/api/v1/pipeline/history');
    return response.data;
  },

  downloadFile: (runId: string, filename: string): string => {
    const baseUrl = apiClient.defaults.baseURL || 'http://localhost:8000';
    return `${baseUrl}/api/v1/pipeline/download/${runId}/${filename}`;
  },
};

export default pipelineApi;
