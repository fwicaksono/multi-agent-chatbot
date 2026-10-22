import { useState } from "react";
import { AlertTriangle, ChevronDown, Clock, Coins, FileText, RefreshCw, Search, Shuffle } from "lucide-react";
import type { FallbackEvent, Source } from "@/lib/types";
import { Markdown } from "./Markdown";
import { Badge, Button } from "@/components/ui/primitives";
import { AgentAvatar } from "@/components/layout/AppShell";
import { cn, formatCost, formatNumber, providerLabel } from "@/lib/format";

export interface AssistantView {
  content: string;
  status: "streaming" | "complete" | "error";
  stopped?: boolean; // saved after the user stopped it (or closed the tab) mid-answer
  phase?: "thinking" | "searching" | "writing";
  citations: Source[];
  sources: Source[];
  provider: string | null;
  model: string | null;
  rewrittenQuery: string | null;
  fallbacks: FallbackEvent[];
  error: { code: string; message: string } | null;
  tokens?: number;
  cost?: number | null;
  latencyMs?: number | null;
}

export function UserMessage({ content }: { content: string }) {
  return (
    <div className="flex justify-end">
      <div className="max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-br-md bg-brand-600 px-4 py-2.5 text-sm text-white shadow-sm">
        {content}
      </div>
    </div>
  );
}

function SourceList({ view, onCite }: { view: AssistantView; onCite: (s: Source) => void }) {
  const [showAll, setShowAll] = useState(false);
  const cited = new Set(view.citations.map((c) => c.index));
  const list = showAll ? view.sources : view.citations;
  if (!view.sources.length) return null;

  // Group citations by document so users see which documents were used.
  return (
    <div className="mt-3 space-y-2">
      <div className="flex flex-wrap items-center gap-2 text-xs text-slate-500">
        <FileText className="h-3.5 w-3.5" aria-hidden />
        {view.citations.length > 0 ? (
          <span>
            Cited {view.citations.length} of {view.sources.length} consulted passages from{" "}
            {new Set(view.citations.map((c) => c.document_name)).size} document(s)
          </span>
        ) : (
          <span>Consulted {view.sources.length} passages; none were cited</span>
        )}
        <button
          type="button"
          onClick={() => setShowAll((s) => !s)}
          className="inline-flex items-center gap-0.5 font-medium text-brand-700 hover:underline"
          aria-expanded={showAll}
        >
          {showAll ? "Show cited only" : "Show all consulted"}
          <ChevronDown className={cn("h-3 w-3 transition-transform", showAll && "rotate-180")} aria-hidden />
        </button>
      </div>
      {list.length > 0 && (
        <ul className="grid gap-1.5 sm:grid-cols-2">
          {list.map((s) => (
            <li key={s.chunk_id}>
              <button
                type="button"
                onClick={() => onCite(s)}
                className={cn(
                  "flex w-full items-start gap-2 rounded-lg border px-2.5 py-2 text-left text-xs transition-colors hover:border-brand-300 hover:bg-brand-50/50",
                  cited.has(s.index) ? "border-slate-200 bg-white" : "border-dashed border-slate-200 bg-slate-50/50",
                )}
              >
                <span
                  className={cn(
                    "flex h-5 min-w-[20px] items-center justify-center rounded-md px-1 text-[10.5px] font-semibold",
                    cited.has(s.index) ? "bg-brand-100 text-brand-700" : "bg-slate-200 text-slate-600",
                  )}
                >
                  {s.index}
                </span>
                <span className="min-w-0">
                  <span className="block truncate font-medium text-slate-800">{s.document_name}</span>
                  <span className="block truncate text-slate-500">{s.location}</span>
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export function AssistantMessage({
  agentName,
  view,
  onCite,
  onRetry,
}: {
  agentName: string;
  view: AssistantView;
  onCite: (s: Source) => void;
  onRetry?: () => void;
}) {
  const streaming = view.status === "streaming";
  return (
    <div className="flex gap-3">
      <AgentAvatar name={agentName} size="md" />
      <div className="min-w-0 flex-1">
        {view.rewrittenQuery && (
          <p className="mb-1.5 flex items-center gap-1.5 text-xs text-slate-500">
            <Search className="h-3 w-3" aria-hidden />
            Searched the knowledge base for: <span className="italic">“{view.rewrittenQuery}”</span>
          </p>
        )}
        {view.fallbacks.map((f, i) => (
          <p
            key={i}
            className="mb-2 flex items-start gap-1.5 rounded-lg bg-amber-50 px-3 py-2 text-xs text-amber-800 ring-1 ring-amber-200"
          >
            <Shuffle className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden />
            <span>
              <strong>
                {providerLabel(f.failed_provider)} ({f.failed_model}) failed
              </strong>{" "}
              — switched to {providerLabel(f.next_provider)} ({f.next_model}). {f.error_message}
            </span>
          </p>
        ))}

        <div className="rounded-2xl rounded-tl-md border border-slate-200 bg-white px-4 py-3 shadow-card">
          {streaming && !view.content ? (
            <p className="flex items-center gap-2 text-sm text-slate-500" role="status">
              <span className="flex gap-1" aria-hidden>
                <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-slate-400 [animation-delay:-0.2s]" />
                <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-slate-400 [animation-delay:-0.1s]" />
                <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-slate-400" />
              </span>
              {view.phase === "searching"
                ? "Searching the knowledge base…"
                : view.phase === "writing"
                  ? "Writing answer…"
                  : "Understanding your question…"}
            </p>
          ) : view.content ? (
            <div aria-live={streaming ? "polite" : undefined}>
              <Markdown content={view.content} sources={view.sources} onCite={onCite} />
              {streaming && <span className="ml-0.5 inline-block h-4 w-1.5 animate-blink bg-brand-500 align-text-bottom" aria-hidden />}
            </div>
          ) : null}

          {view.stopped && (
            <p className={cn("text-xs italic text-slate-500", view.content && "mt-2")}>
              Stopped before the answer was complete.
            </p>
          )}

          {view.status === "error" && view.error && (
            <div
              role="alert"
              className={cn(
                "flex items-start gap-2 rounded-lg bg-rose-50 px-3 py-2.5 text-sm text-rose-800",
                view.content && "mt-3",
              )}
            >
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
              <div className="min-w-0 flex-1">
                <p className="font-medium">The answer couldn’t be generated</p>
                <p className="mt-0.5 text-rose-700">{view.error.message}</p>
                <p className="mt-1 font-mono text-[11px] text-rose-500">{view.error.code}</p>
              </div>
              {onRetry && (
                <Button size="sm" variant="secondary" onClick={onRetry} icon={<RefreshCw className="h-3.5 w-3.5" />}>
                  Retry
                </Button>
              )}
            </div>
          )}

          {!streaming && <SourceList view={view} onCite={onCite} />}
        </div>

        {view.status === "complete" && (
          <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 px-1 text-[11px] text-slate-400">
            {view.provider && (
              <Badge className="px-1.5 py-0 text-[10.5px]">
                {providerLabel(view.provider)} · {view.model}
              </Badge>
            )}
            {view.tokens != null && view.tokens > 0 && (
              <span className="inline-flex items-center gap-1" title="Tokens used for this answer (incl. query rewrite)">
                <Coins className="h-3 w-3" aria-hidden /> {formatNumber(view.tokens)} tokens · {formatCost(view.cost)}
              </span>
            )}
            {view.latencyMs != null && (
              <span className="inline-flex items-center gap-1">
                <Clock className="h-3 w-3" aria-hidden /> {(view.latencyMs / 1000).toFixed(1)}s
              </span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
