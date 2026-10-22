import { useCallback, useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from "react";
import { Link, useOutletContext, useSearchParams } from "react-router-dom";
import { useQueryClient } from "@tanstack/react-query";
import { ArrowUp, FileText, MessageSquarePlus, PanelLeft, Repeat, Square, Trash2 } from "lucide-react";
import { api, ApiError, streamChat } from "@/lib/api";
import { keys, useConversation, useConversations, useUpdateAgent } from "@/lib/queries";
import type { Message, Source } from "@/lib/types";
import { AssistantMessage, UserMessage, type AssistantView } from "@/components/chat/MessageItem";
import { SourcePanel } from "@/components/chat/SourcePanel";
import { ProviderModelPicker } from "@/components/agents/ProviderModelPicker";
import { Alert, Button, Skeleton, Textarea } from "@/components/ui/primitives";
import { ConfirmDialog, Dialog } from "@/components/ui/Dialog";
import { useToast } from "@/components/ui/Toast";
import { cn, formatRelative, providerLabel } from "@/lib/format";
import type { WorkspaceContext } from "./AgentWorkspace";

function toView(m: Message): AssistantView {
  return {
    content: m.content,
    status: m.status === "error" ? "error" : "complete",
    stopped: m.status === "cancelled",
    citations: m.citations,
    sources: m.sources,
    provider: m.provider,
    model: m.model,
    rewrittenQuery: m.rewritten_query,
    fallbacks: m.fallback_events,
    error: m.error,
    tokens: m.prompt_tokens + m.completion_tokens,
    cost: m.cost_usd,
    latencyMs: m.latency_ms,
  };
}

const emptyView = (): AssistantView => ({
  content: "",
  status: "streaming",
  phase: "thinking",
  citations: [],
  sources: [],
  provider: null,
  model: null,
  rewrittenQuery: null,
  fallbacks: [],
  error: null,
});

function ModelSwitchDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { agent } = useOutletContext<WorkspaceContext>();
  const update = useUpdateAgent(agent.id);
  const toast = useToast();
  const [provider, setProvider] = useState(agent.provider);
  const [model, setModel] = useState(agent.model);

  useEffect(() => {
    if (open) {
      setProvider(agent.provider);
      setModel(agent.model);
    }
  }, [open, agent.provider, agent.model]);

  const save = () =>
    update.mutate(
      { provider, model: model.trim() },
      {
        onSuccess: (a) => {
          toast.success(`Now using ${providerLabel(a.provider)} · ${a.model}`, "Conversation history is preserved.");
          onClose();
        },
        onError: (e) => toast.error("Couldn't switch model", (e as Error).message),
      },
    );

  return (
    <Dialog
      open={open}
      onClose={onClose}
      title="Switch LLM provider or model"
      description="Takes effect on the next message. History and knowledge base are unchanged."
      footer={
        <>
          <Button variant="secondary" onClick={onClose}>
            Cancel
          </Button>
          <Button onClick={save} loading={update.isPending} disabled={!provider || !model.trim()}>
            Switch
          </Button>
        </>
      }
    >
      <ProviderModelPicker
        idPrefix="switch"
        compact
        provider={provider}
        model={model}
        onChange={(p, m) => {
          setProvider(p);
          setModel(m);
        }}
      />
    </Dialog>
  );
}

function ConversationList({
  agentId,
  activeId,
  onSelect,
  onNew,
}: {
  agentId: string;
  activeId: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
}) {
  const { data, isLoading } = useConversations(agentId);
  const qc = useQueryClient();
  const toast = useToast();
  const [toDelete, setToDelete] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);

  const remove = async () => {
    if (!toDelete) return;
    setDeleting(true);
    try {
      await api.deleteConversation(toDelete);
      await qc.invalidateQueries({ queryKey: keys.conversations(agentId) });
      if (toDelete === activeId) onNew();
      setToDelete(null);
    } catch (e) {
      toast.error("Couldn't delete conversation", (e as Error).message);
    } finally {
      setDeleting(false);
    }
  };

  return (
    <div className="flex h-full flex-col">
      <div className="p-3">
        <Button variant="secondary" className="w-full justify-center" onClick={onNew} icon={<MessageSquarePlus className="h-4 w-4" />}>
          New conversation
        </Button>
      </div>
      <ul className="min-h-0 flex-1 space-y-0.5 overflow-y-auto px-2 pb-3" aria-label="Conversations">
        {isLoading && [0, 1, 2].map((i) => <Skeleton key={i} className="mx-2 my-2 h-8" />)}
        {data?.length === 0 && <li className="px-3 py-2 text-xs text-slate-400">No conversations yet</li>}
        {data?.map((c) => (
          <li key={c.id} className="group relative">
            <button
              onClick={() => onSelect(c.id)}
              aria-current={c.id === activeId ? "true" : undefined}
              className={cn(
                "w-full rounded-lg px-3 py-2 pr-8 text-left text-sm transition-colors",
                c.id === activeId ? "bg-white font-medium text-slate-900 shadow-card ring-1 ring-slate-200" : "text-slate-600 hover:bg-white/70",
              )}
            >
              <span className="block truncate">{c.title}</span>
              <span className="block text-[11px] text-slate-400">
                {formatRelative(c.updated_at)} · {c.message_count} message{c.message_count === 1 ? "" : "s"}
              </span>
            </button>
            <button
              onClick={() => setToDelete(c.id)}
              className="absolute right-1.5 top-2 rounded p-1 text-slate-300 opacity-0 hover:bg-rose-50 hover:text-rose-600 focus:opacity-100 group-hover:opacity-100"
              aria-label={`Delete conversation ${c.title}`}
            >
              <Trash2 className="h-3.5 w-3.5" />
            </button>
          </li>
        ))}
      </ul>
      <ConfirmDialog
        open={!!toDelete}
        onClose={() => setToDelete(null)}
        onConfirm={remove}
        loading={deleting}
        title="Delete conversation?"
        description="Its messages will be permanently deleted."
      />
    </div>
  );
}

const SUGGESTIONS = [
  "What topics do your documents cover?",
  "Summarise the key rules I should know.",
  "What deadlines or notice periods are mentioned?",
];

export function ChatTab() {
  const { agent } = useOutletContext<WorkspaceContext>();
  const [params, setParams] = useSearchParams();
  const conversationId = params.get("c");
  const { data: conversation, isLoading: loadingConv, error: convError } = useConversation(conversationId ?? undefined);
  const qc = useQueryClient();

  const [input, setInput] = useState("");
  const [pending, setPending] = useState<{ question: string; view: AssistantView } | null>(null);
  const [activeSource, setActiveSource] = useState<Source | null>(null);
  const [switchOpen, setSwitchOpen] = useState(false);
  const [listOpen, setListOpen] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const messages = conversation?.messages ?? [];
  const streaming = pending?.view.status === "streaming";
  const noDocs = agent.document_counts.ready === 0;

  // Keep the newest content in view while streaming.
  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTo({ top: el.scrollHeight, behavior: streaming ? "auto" : "smooth" });
  }, [messages.length, pending?.view.content, pending?.view.status, streaming]);

  useEffect(() => () => abortRef.current?.abort(), []);

  const selectConversation = (id: string | null) => {
    abortRef.current?.abort();
    setPending(null);
    setListOpen(false);
    setParams(id ? { c: id } : {}, { replace: false });
    setTimeout(() => inputRef.current?.focus(), 0);
  };

  const send = useCallback(
    async (question: string) => {
      const text = question.trim();
      if (!text || streaming) return;
      setInput("");
      if (inputRef.current) inputRef.current.style.height = "auto";
      setPending({ question: text, view: emptyView() });

      let convId = conversationId;
      try {
        if (!convId) {
          convId = (await api.createConversation(agent.id)).id;
          setParams({ c: convId }, { replace: true });
        }
      } catch (e) {
        setPending({ question: text, view: { ...emptyView(), status: "error", error: { code: "network_error", message: (e as Error).message } } });
        return;
      }

      const controller = new AbortController();
      abortRef.current = controller;
      const patch = (fn: (v: AssistantView) => AssistantView) =>
        setPending((p) => (p ? { ...p, view: fn(p.view) } : p));

      try {
        await streamChat(
          convId,
          text,
          (ev) => {
            switch (ev.event) {
              case "meta":
                patch((v) => ({ ...v, provider: ev.data.provider, model: ev.data.model, phase: "searching" }));
                break;
              case "rewrite":
                patch((v) => ({ ...v, rewrittenQuery: ev.data.query }));
                break;
              case "sources":
                patch((v) => ({ ...v, sources: ev.data.sources, phase: "writing" }));
                break;
              case "fallback":
                patch((v) => ({ ...v, fallbacks: [...v.fallbacks, ev.data] }));
                break;
              case "token":
                patch((v) => ({ ...v, content: v.content + ev.data.text }));
                break;
              case "done":
                patch((v) => ({
                  ...v,
                  status: "complete",
                  content: ev.data.content,
                  citations: ev.data.citations,
                  provider: ev.data.provider,
                  model: ev.data.model,
                  tokens: ev.data.usage.prompt_tokens + ev.data.usage.completion_tokens,
                  cost: ev.data.usage.cost_usd,
                  latencyMs: ev.data.latency_ms,
                }));
                break;
              case "error":
                patch((v) => ({ ...v, status: "error", error: { code: ev.data.code, message: ev.data.message } }));
                break;
            }
          },
          controller.signal,
        );
      } catch (e) {
        if ((e as Error).name === "AbortError") {
          patch((v) => ({ ...v, status: "error", error: { code: "cancelled", message: "You stopped this response." } }));
        } else {
          const err = e as ApiError;
          patch((v) => ({ ...v, status: "error", error: { code: err.code ?? "error", message: err.message } }));
        }
      } finally {
        abortRef.current = null;
        // Server state is the source of truth once the turn is persisted. Setting
        // the fresh data and clearing `pending` in the same tick avoids a flash
        // of duplicated messages (React batches both updates).
        try {
          qc.setQueryData(keys.conversation(convId), await api.getConversation(convId));
        } catch {
          /* keep the optimistic view */
        }
        qc.invalidateQueries({ queryKey: keys.conversations(agent.id) });
        qc.invalidateQueries({ queryKey: keys.agentUsage(agent.id) });
        setPending((p) => (p?.view.error?.code === "cancelled" ? p : null));
        if (controller.signal.aborted) {
          // The server saves the partial answer right after the disconnect;
          // swap the local "stopped" view for the saved message shortly after.
          setTimeout(async () => {
            try {
              qc.setQueryData(keys.conversation(convId!), await api.getConversation(convId!));
              qc.invalidateQueries({ queryKey: keys.conversations(agent.id) });
            } catch {
              /* keep the local view */
            }
            setPending((p) => (p?.view.error?.code === "cancelled" ? null : p));
          }, 1200);
        }
      }
    },
    [agent.id, conversationId, qc, setParams, streaming],
  );

  const onSubmit = (e: FormEvent) => {
    e.preventDefault();
    send(input);
  };
  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      send(input);
    }
  };

  const lastUserQuestion = [...messages].reverse().find((m) => m.role === "user")?.content;

  return (
    <div className="flex h-full">
      {/* Conversation list */}
      <aside className="hidden w-64 shrink-0 border-r border-slate-200 bg-slate-50/60 md:block">
        <ConversationList agentId={agent.id} activeId={conversationId} onSelect={selectConversation} onNew={() => selectConversation(null)} />
      </aside>
      {listOpen && (
        <div className="fixed inset-0 z-40 md:hidden">
          <div className="absolute inset-0 bg-slate-900/30" onClick={() => setListOpen(false)} aria-hidden />
          <aside className="absolute inset-y-0 left-0 w-72 bg-slate-50 shadow-pop">
            <ConversationList agentId={agent.id} activeId={conversationId} onSelect={selectConversation} onNew={() => selectConversation(null)} />
          </aside>
        </div>
      )}

      <section className="flex min-w-0 flex-1 flex-col" aria-label="Chat">
        <div className="flex items-center justify-between gap-2 border-b border-slate-200 bg-white/80 px-4 py-2 backdrop-blur">
          <div className="flex min-w-0 items-center gap-2">
            <Button variant="ghost" size="icon" className="md:hidden" onClick={() => setListOpen(true)} aria-label="Show conversations">
              <PanelLeft className="h-4 w-4" />
            </Button>
            <h2 className="truncate text-sm font-medium text-slate-700">{conversation?.title ?? "New conversation"}</h2>
          </div>
          <button
            onClick={() => setSwitchOpen(true)}
            className="inline-flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-2.5 py-1.5 text-xs font-medium text-slate-700 shadow-sm hover:border-brand-300 hover:text-brand-700"
            title="Switch provider or model"
          >
            <Repeat className="h-3.5 w-3.5" aria-hidden />
            <span className="hidden sm:inline">{providerLabel(agent.provider)} ·</span>
            <span className="max-w-[10rem] truncate font-mono">{agent.model}</span>
          </button>
        </div>

        <div ref={scrollRef} className="min-h-0 flex-1 overflow-y-auto">
          <div className="mx-auto max-w-3xl space-y-6 px-4 py-6">
            {noDocs && (
              <Alert tone="warning" title="This agent has no ready documents">
                It will answer “I don’t know” until its knowledge base has content.{" "}
                <Link to="../knowledge" relative="path" className="font-medium underline">
                  Upload documents
                </Link>
              </Alert>
            )}
            {convError && <Alert title="Couldn't load conversation">{(convError as Error).message}</Alert>}
            {loadingConv && conversationId && (
              <div className="space-y-4">
                <Skeleton className="ml-auto h-10 w-1/2" />
                <Skeleton className="h-24 w-3/4" />
              </div>
            )}

            {!conversationId && !pending && (
              <div className="flex flex-col items-center py-10 text-center">
                <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-2xl bg-brand-50 text-brand-600 ring-1 ring-brand-100">
                  <FileText className="h-6 w-6" aria-hidden />
                </div>
                <h3 className="text-base font-semibold text-slate-900">Ask {agent.name}</h3>
                <p className="mt-1 max-w-md text-sm text-slate-500">
                  Answers come only from this agent’s {agent.document_counts.ready} ready document
                  {agent.document_counts.ready === 1 ? "" : "s"}, with clickable citations. Follow-up questions keep the context.
                </p>
                <div className="mt-6 flex flex-wrap justify-center gap-2">
                  {SUGGESTIONS.map((s) => (
                    <button
                      key={s}
                      onClick={() => send(s)}
                      className="rounded-full border border-slate-200 bg-white px-3.5 py-1.5 text-sm text-slate-600 shadow-sm hover:border-brand-300 hover:text-brand-700"
                    >
                      {s}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {messages.map((m, i) =>
              m.role === "user" ? (
                <UserMessage key={m.id} content={m.content} />
              ) : (
                <AssistantMessage
                  key={m.id}
                  agentName={agent.name}
                  view={toView(m)}
                  onCite={setActiveSource}
                  onRetry={i === messages.length - 1 && m.status === "error" && lastUserQuestion ? () => send(lastUserQuestion) : undefined}
                />
              ),
            )}

            {pending && (
              <>
                {!(messages.at(-1)?.role === "user" && messages.at(-1)?.content === pending.question) && (
                  <UserMessage content={pending.question} />
                )}
                <AssistantMessage
                  agentName={agent.name}
                  view={pending.view}
                  onCite={setActiveSource}
                  onRetry={pending.view.status === "error" ? () => send(pending.question) : undefined}
                />
              </>
            )}
          </div>
        </div>

        <form onSubmit={onSubmit} className="border-t border-slate-200 bg-white px-4 py-3">
          <div className="mx-auto max-w-3xl">
            <div className="flex items-end gap-2 rounded-2xl border border-slate-200 bg-white p-2 shadow-card focus-within:border-brand-300 focus-within:ring-2 focus-within:ring-brand-100">
              <label htmlFor="chat-input" className="sr-only">
                Message {agent.name}
              </label>
              <Textarea
                id="chat-input"
                ref={inputRef}
                rows={1}
                value={input}
                onChange={(e) => {
                  setInput(e.target.value);
                  e.target.style.height = "auto";
                  e.target.style.height = `${Math.min(e.target.scrollHeight, 200)}px`;
                }}
                onKeyDown={onKeyDown}
                placeholder={`Ask ${agent.name} a question…`}
                maxLength={8000}
                className="max-h-[200px] resize-none border-0 shadow-none ring-0 focus:ring-0 focus-visible:outline-none"
              />
              {streaming ? (
                <Button type="button" variant="secondary" size="icon" onClick={() => abortRef.current?.abort()} aria-label="Stop generating">
                  <Square className="h-3.5 w-3.5 fill-current" />
                </Button>
              ) : (
                <Button type="submit" size="icon" disabled={!input.trim()} aria-label="Send message">
                  <ArrowUp className="h-4 w-4" />
                </Button>
              )}
            </div>
            <p className="mt-1.5 text-center text-[11px] text-slate-400">
              Enter to send · Shift+Enter for a new line · Answers are grounded in this agent’s documents only
            </p>
          </div>
        </form>
      </section>

      <SourcePanel agentId={agent.id} source={activeSource} onClose={() => setActiveSource(null)} />
      <ModelSwitchDialog open={switchOpen} onClose={() => setSwitchOpen(false)} />
    </div>
  );
}
