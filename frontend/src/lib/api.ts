import type {
  Agent,
  AgentInput,
  AgentUsage,
  AgentUsageSummary,
  ApiErrorBody,
  ChatStreamEvent,
  Conversation,
  ConversationDetail,
  KnowledgeDocument,
  ModelsResponse,
  ProvidersResponse,
  UploadResult,
} from "./types";

const BASE = "/api";

export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly details?: Record<string, unknown>;

  constructor(status: number, body: ApiErrorBody) {
    super(body.message);
    this.status = status;
    this.code = body.code;
    this.details = body.details;
  }
}

async function parseError(res: Response): Promise<ApiError> {
  try {
    const body = await res.json();
    if (body?.error?.message) return new ApiError(res.status, body.error);
  } catch {
    /* non-JSON error body */
  }
  if (res.status === 413)
    return new ApiError(413, {
      code: "payload_too_large",
      message: "The upload is too large for the server. Upload fewer or smaller files at a time.",
    });
  const fallback =
    res.status >= 500
      ? "The server is unavailable. Check that the API container is running."
      : `Request failed (HTTP ${res.status}).`;
  return new ApiError(res.status, { code: `http_${res.status}`, message: fallback });
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, {
      ...init,
      headers: init?.body instanceof FormData ? init.headers : { "Content-Type": "application/json", ...init?.headers },
    });
  } catch {
    throw new ApiError(0, {
      code: "network_error",
      message: "Cannot reach the server. Check your connection and that the backend is running.",
    });
  }
  if (!res.ok) throw await parseError(res);
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

const json = (body: unknown) => JSON.stringify(body);

export const api = {
  providers: () => request<ProvidersResponse>("/providers"),
  models: (provider: string) => request<ModelsResponse>(`/providers/${encodeURIComponent(provider)}/models`),

  listAgents: () => request<Agent[]>("/agents"),
  getAgent: (id: string) => request<Agent>(`/agents/${id}`),
  createAgent: (data: AgentInput) => request<Agent>("/agents", { method: "POST", body: json(data) }),
  updateAgent: (id: string, data: Partial<AgentInput>) =>
    request<Agent>(`/agents/${id}`, { method: "PATCH", body: json(data) }),
  deleteAgent: (id: string) => request<void>(`/agents/${id}`, { method: "DELETE" }),

  listDocuments: (agentId: string) => request<KnowledgeDocument[]>(`/agents/${agentId}/documents`),
  uploadDocuments: (agentId: string, files: File[]) => {
    const form = new FormData();
    files.forEach((f) => form.append("files", f));
    return request<UploadResult>(`/agents/${agentId}/documents`, { method: "POST", body: form });
  },
  deleteDocument: (agentId: string, docId: string) =>
    request<void>(`/agents/${agentId}/documents/${docId}`, { method: "DELETE" }),
  reprocessDocument: (agentId: string, docId: string) =>
    request<KnowledgeDocument>(`/agents/${agentId}/documents/${docId}/reprocess`, { method: "POST" }),
  documentUrl: (agentId: string, docId: string, page?: number | null) =>
    `${BASE}/agents/${agentId}/documents/${docId}/file${page ? `#page=${page}` : ""}`,

  listConversations: (agentId: string) => request<Conversation[]>(`/agents/${agentId}/conversations`),
  createConversation: (agentId: string) =>
    request<Conversation>(`/agents/${agentId}/conversations`, { method: "POST", body: json({}) }),
  getConversation: (id: string) => request<ConversationDetail>(`/conversations/${id}`),
  renameConversation: (id: string, title: string) =>
    request<Conversation>(`/conversations/${id}`, { method: "PATCH", body: json({ title }) }),
  deleteConversation: (id: string) => request<void>(`/conversations/${id}`, { method: "DELETE" }),

  agentUsage: (agentId: string) => request<AgentUsage>(`/agents/${agentId}/usage`),
  usageSummary: () => request<AgentUsageSummary[]>("/usage"),
};

/**
 * Send a chat message and consume the Server-Sent-Events response.
 * EventSource only supports GET, so the stream is read with fetch().
 */
export async function streamChat(
  conversationId: string,
  content: string,
  onEvent: (event: ChatStreamEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  let res: Response;
  try {
    res = await fetch(`${BASE}/conversations/${conversationId}/messages`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: json({ content }),
      signal,
    });
  } catch (err) {
    if ((err as Error).name === "AbortError") throw err;
    throw new ApiError(0, { code: "network_error", message: "Cannot reach the server." });
  }
  if (!res.ok || !res.body) throw await parseError(res);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let boundary: number;
    while ((boundary = buffer.indexOf("\n\n")) !== -1) {
      const raw = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      let event = "message";
      const data: string[] = [];
      for (const line of raw.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
      }
      if (!data.length) continue;
      try {
        onEvent({ event, data: JSON.parse(data.join("\n")) } as ChatStreamEvent);
      } catch {
        /* ignore malformed event */
      }
    }
  }
}
