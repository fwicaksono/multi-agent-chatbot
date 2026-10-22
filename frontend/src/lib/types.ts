// Types mirroring the backend API schemas (backend/app/schemas/api.py).

export type DocumentStatus = "processing" | "ready" | "failed";

export interface DocumentCounts {
  total: number;
  ready: number;
  processing: number;
  failed: number;
}

export interface Agent {
  id: string;
  name: string;
  description: string;
  instructions: string;
  provider: string;
  model: string;
  fallback_provider: string | null;
  fallback_model: string | null;
  temperature: number;
  created_at: string;
  updated_at: string;
  document_counts: DocumentCounts;
}

export type AgentInput = Pick<
  Agent,
  "name" | "description" | "instructions" | "provider" | "model" | "temperature"
> & { fallback_provider: string | null; fallback_model: string | null };

export interface KnowledgeDocument {
  id: string;
  agent_id: string;
  filename: string;
  file_type: "pdf" | "docx";
  size_bytes: number;
  status: DocumentStatus;
  stage: string;
  error_code: string | null;
  error_message: string | null;
  page_count: number | null;
  ocr_page_count: number;
  chunk_count: number;
  char_count: number;
  embedding_model: string | null;
  attempts: number;
  created_at: string;
  updated_at: string;
  processed_at: string | null;
}

export interface UploadResult {
  accepted: KnowledgeDocument[];
  rejected: { filename: string; code: string; message: string }[];
}

export interface Source {
  index: number;
  chunk_id: string;
  document_id: string;
  document_name: string;
  file_type: string;
  page_number: number | null;
  section: string | null;
  location: string;
  snippet: string;
  score: number;
  vector_similarity: number | null;
  matched_by: string[];
}

export interface FallbackEvent {
  failed_provider: string;
  failed_model: string;
  next_provider: string;
  next_model: string;
  error_code: string;
  error_message: string;
}

export interface ApiErrorBody {
  code: string;
  message: string;
  details?: Record<string, unknown>;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  status: string;
  citations: Source[];
  sources: Source[];
  provider: string | null;
  model: string | null;
  rewritten_query: string | null;
  fallback_events: FallbackEvent[];
  error: ApiErrorBody | null;
  prompt_tokens: number;
  completion_tokens: number;
  cost_usd: number | null;
  latency_ms: number | null;
  created_at: string;
}

export interface Conversation {
  id: string;
  agent_id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
}

export interface ConversationDetail extends Conversation {
  messages: Message[];
  summary: string | null; // the assistant's condensed memory of older messages
}

export interface ProviderInfo {
  name: string;
  display_name: string;
  configured: boolean;
  required_env: string[];
  suggested_models: string[];
}

export interface ProvidersResponse {
  providers: ProviderInfo[];
  default_provider: string;
  default_model: string;
  embedding: { provider: string; model: string; dimensions: number };
  fallback_chain: string[];
}

export interface ModelInfo {
  id: string;
  name?: string | null;
  context_length?: number | null;
  input_price_per_mtok?: number | null;
  output_price_per_mtok?: number | null;
}

export interface ModelsResponse {
  provider: string;
  source: "live" | "suggested";
  models: ModelInfo[];
  warning: string | null;
}

export interface UsageTotals {
  requests: number;
  prompt_tokens: number;
  completion_tokens: number;
  cost_usd: number;
}

export interface AgentUsage {
  agent_id: string;
  totals: UsageTotals;
  by_model: (UsageTotals & { provider: string; model: string; kind: string })[];
  daily: (UsageTotals & { day: string })[];
  unpriced_requests: number;
}

export interface AgentUsageSummary extends UsageTotals {
  agent_id: string;
  agent_name: string;
}

// Server-Sent Events emitted by POST /api/conversations/{id}/messages
export type ChatStreamEvent =
  | { event: "meta"; data: { conversation_id: string; user_message_id: string; provider: string; model: string } }
  | { event: "rewrite"; data: { query: string } }
  | { event: "sources"; data: { sources: Source[] } }
  | { event: "fallback"; data: FallbackEvent }
  | { event: "token"; data: { text: string } }
  | {
      event: "done";
      data: {
        message_id: string;
        content: string;
        citations: Source[];
        provider: string | null;
        model: string | null;
        rewritten_query: string | null;
        fallback_events: FallbackEvent[];
        usage: { prompt_tokens: number; completion_tokens: number; cost_usd: number };
        latency_ms: number;
      };
    }
  | { event: "error"; data: ApiErrorBody & { message_id?: string } };
