import { useEffect, useRef } from "react";
import { createPortal } from "react-dom";
import { ExternalLink, FileText, FileType2, X } from "lucide-react";
import { api } from "@/lib/api";
import type { Source } from "@/lib/types";
import { Badge, buttonClass } from "@/components/ui/primitives";

/** Slide-over showing the exact passage behind a citation, with a deep link to the file. */
export function SourcePanel({ agentId, source, onClose }: { agentId: string; source: Source | null; onClose: () => void }) {
  const closeRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!source) return;
    closeRef.current?.focus();
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [source, onClose]);

  if (!source) return null;
  const Icon = source.file_type === "pdf" ? FileText : FileType2;
  const href = api.documentUrl(agentId, source.document_id, source.file_type === "pdf" ? source.page_number : null);

  return createPortal(
    <div className="fixed inset-0 z-50 flex justify-end">
      <div className="absolute inset-0 bg-slate-900/20" onClick={onClose} aria-hidden />
      <aside
        role="dialog"
        aria-modal="true"
        aria-label={`Source ${source.index}`}
        className="relative flex h-full w-full max-w-md animate-fade-in flex-col bg-white shadow-pop"
      >
        <div className="flex items-start justify-between gap-3 border-b border-slate-100 px-5 py-4">
          <div className="flex min-w-0 items-start gap-3">
            <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-brand-50 text-brand-600">
              <Icon className="h-4 w-4" aria-hidden />
            </span>
            <div className="min-w-0">
              <p className="text-xs font-medium uppercase tracking-wide text-slate-400">Source {source.index}</p>
              <h2 className="truncate text-sm font-semibold text-slate-900" title={source.document_name}>
                {source.document_name}
              </h2>
              <p className="mt-0.5 text-xs text-slate-500">{source.location}</p>
            </div>
          </div>
          <button
            ref={closeRef}
            onClick={onClose}
            className="rounded-md p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-700"
            aria-label="Close source"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        <div className="flex-1 space-y-4 overflow-y-auto px-5 py-4">
          <div className="flex flex-wrap gap-1.5">
            {source.page_number && <Badge>Page {source.page_number}</Badge>}
            {source.section && <Badge className="max-w-full truncate">§ {source.section}</Badge>}
            {source.matched_by.map((m) => (
              <Badge key={m} tone="info">
                {m === "vector" ? "Semantic match" : "Keyword match"}
              </Badge>
            ))}
            {source.vector_similarity != null && (
              <Badge tone="neutral">similarity {source.vector_similarity.toFixed(2)}</Badge>
            )}
          </div>
          <figure>
            <figcaption className="mb-2 text-xs font-medium text-slate-500">Passage used by the agent</figcaption>
            <blockquote className="whitespace-pre-wrap rounded-lg border-l-4 border-brand-300 bg-slate-50 px-4 py-3 text-sm leading-relaxed text-slate-700">
              {source.snippet}
              {source.snippet.length >= 600 && "…"}
            </blockquote>
          </figure>
        </div>

        <div className="border-t border-slate-100 px-5 py-3">
          <a href={href} target="_blank" rel="noreferrer" className={buttonClass("primary", "md", "w-full justify-center")}>
            <ExternalLink className="h-4 w-4" aria-hidden />
            Open document{source.file_type === "pdf" && source.page_number ? ` at page ${source.page_number}` : ""}
          </a>
        </div>
      </aside>
    </div>,
    document.body,
  );
}
