import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "./api";
import type { AgentInput } from "./types";

export const keys = {
  providers: ["providers"] as const,
  models: (p: string) => ["models", p] as const,
  agents: ["agents"] as const,
  agent: (id: string) => ["agents", id] as const,
  documents: (id: string) => ["agents", id, "documents"] as const,
  conversations: (id: string) => ["agents", id, "conversations"] as const,
  conversation: (id: string) => ["conversations", id] as const,
  agentUsage: (id: string) => ["agents", id, "usage"] as const,
  usage: ["usage"] as const,
};

export const useProviders = () =>
  useQuery({ queryKey: keys.providers, queryFn: api.providers, staleTime: 60_000 });

export const useModels = (provider: string | undefined) =>
  useQuery({
    queryKey: keys.models(provider ?? ""),
    queryFn: () => api.models(provider!),
    enabled: !!provider,
    staleTime: 5 * 60_000,
  });

// Agents poll while any of their documents is processing, so document counts
// (and the chat's "no ready documents" hint) update when ingestion finishes.
export const useAgents = () =>
  useQuery({
    queryKey: keys.agents,
    queryFn: api.listAgents,
    refetchInterval: (q) => (q.state.data?.some((a) => a.document_counts.processing > 0) ? 3000 : false),
  });

export const useAgent = (id: string | undefined) =>
  useQuery({
    queryKey: keys.agent(id ?? ""),
    queryFn: () => api.getAgent(id!),
    enabled: !!id,
    refetchInterval: (q) => (q.state.data?.document_counts.processing ? 3000 : false),
  });

export function useCreateAgent() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: AgentInput) => api.createAgent(data),
    onSuccess: () => qc.invalidateQueries({ queryKey: keys.agents }),
  });
}

export function useUpdateAgent(id: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: Partial<AgentInput>) => api.updateAgent(id, data),
    onSuccess: (agent) => {
      qc.setQueryData(keys.agent(id), agent);
      qc.invalidateQueries({ queryKey: keys.agents, exact: true });
    },
  });
}

export function useDeleteAgent() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api.deleteAgent(id),
    onSuccess: (_d, id) => {
      qc.removeQueries({ queryKey: keys.agent(id) });
      qc.invalidateQueries({ queryKey: keys.agents });
      qc.invalidateQueries({ queryKey: keys.usage });
    },
  });
}

/** Documents list; polls every 2s while any document is still processing. */
export const useDocuments = (agentId: string) =>
  useQuery({
    queryKey: keys.documents(agentId),
    queryFn: () => api.listDocuments(agentId),
    refetchInterval: (q) => (q.state.data?.some((d) => d.status === "processing") ? 2000 : false),
  });

export function useDocumentMutations(agentId: string) {
  const qc = useQueryClient();
  const refresh = () => {
    qc.invalidateQueries({ queryKey: keys.documents(agentId) });
    qc.invalidateQueries({ queryKey: keys.agent(agentId) });
    qc.invalidateQueries({ queryKey: keys.agents, exact: true });
  };
  return {
    upload: useMutation({ mutationFn: (files: File[]) => api.uploadDocuments(agentId, files), onSuccess: refresh }),
    remove: useMutation({ mutationFn: (docId: string) => api.deleteDocument(agentId, docId), onSuccess: refresh }),
    reprocess: useMutation({
      mutationFn: (docId: string) => api.reprocessDocument(agentId, docId),
      onSuccess: refresh,
    }),
  };
}

export const useConversations = (agentId: string) =>
  useQuery({ queryKey: keys.conversations(agentId), queryFn: () => api.listConversations(agentId) });

export const useConversation = (id: string | undefined) =>
  useQuery({
    queryKey: keys.conversation(id ?? ""),
    queryFn: () => api.getConversation(id!),
    enabled: !!id,
  });

export const useAgentUsage = (agentId: string) =>
  useQuery({ queryKey: keys.agentUsage(agentId), queryFn: () => api.agentUsage(agentId) });

export const useUsageSummary = () => useQuery({ queryKey: keys.usage, queryFn: api.usageSummary });
