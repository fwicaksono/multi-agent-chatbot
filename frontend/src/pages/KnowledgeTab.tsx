import { useState } from "react";
import { useOutletContext } from "react-router-dom";
import { AlertCircle, CheckCircle2, ExternalLink, FileText, FileType2, Loader2, RotateCw, ScanText, Trash2 } from "lucide-react";
import { api } from "@/lib/api";
import { useDocumentMutations, useDocuments, useProviders } from "@/lib/queries";
import type { KnowledgeDocument } from "@/lib/types";
import { Alert, Badge, Button, Card, EmptyState, Skeleton } from "@/components/ui/primitives";
import { ConfirmDialog } from "@/components/ui/Dialog";
import { useToast } from "@/components/ui/Toast";
import { UploadDropzone } from "@/components/knowledge/UploadDropzone";
import { STAGE_LABELS, formatBytes, formatRelative } from "@/lib/format";
import type { WorkspaceContext } from "./AgentWorkspace";

const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? "" : "s"}`;

const STAGE_ORDER = ["queued", "extracting", "chunking", "embedding", "indexing"];

function StatusBadge({ doc }: { doc: KnowledgeDocument }) {
  if (doc.status === "ready")
    return (
      <Badge tone="success">
        <CheckCircle2 className="h-3 w-3" aria-hidden /> Ready
      </Badge>
    );
  if (doc.status === "failed")
    return (
      <Badge tone="danger">
        <AlertCircle className="h-3 w-3" aria-hidden /> Failed
      </Badge>
    );
  const step = Math.max(STAGE_ORDER.indexOf(doc.stage), 0) + 1;
  return (
    <span className="inline-flex flex-col gap-1">
      <Badge tone="info">
        <Loader2 className="h-3 w-3 animate-spin" aria-hidden /> Processing
      </Badge>
      <span className="text-[11px] text-slate-500">
        {STAGE_LABELS[doc.stage] ?? doc.stage} · step {step}/{STAGE_ORDER.length}
      </span>
    </span>
  );
}

function DocumentRow({
  doc,
  agentId,
  onDelete,
  onRetry,
  retrying,
}: {
  doc: KnowledgeDocument;
  agentId: string;
  onDelete: () => void;
  onRetry: () => void;
  retrying: boolean;
}) {
  const Icon = doc.file_type === "pdf" ? FileText : FileType2;
  return (
    <li className="px-4 py-4 sm:px-5">
      <div className="flex items-start gap-3">
        <span
          className={
            doc.file_type === "pdf"
              ? "flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-rose-50 text-rose-600"
              : "flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-sky-50 text-sky-600"
          }
        >
          <Icon className="h-4 w-4" aria-hidden />
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
            <span className="truncate text-sm font-medium text-slate-900">{doc.filename}</span>
            <StatusBadge doc={doc} />
          </div>
          <p className="mt-1 flex flex-wrap gap-x-3 text-xs text-slate-500">
            <span>{doc.file_type.toUpperCase()}</span>
            <span>{formatBytes(doc.size_bytes)}</span>
            {doc.page_count != null && <span>{plural(doc.page_count, "page")}</span>}
            {doc.status === "ready" && <span>{plural(doc.chunk_count, "chunk")} indexed</span>}
            {doc.ocr_page_count > 0 && (
              <span className="inline-flex items-center gap-1 text-violet-700">
                <ScanText className="h-3 w-3" aria-hidden /> OCR on {doc.ocr_page_count} page{doc.ocr_page_count > 1 ? "s" : ""}
              </span>
            )}
            <span>Added {formatRelative(doc.created_at)}</span>
          </p>
          {doc.status === "ready" && doc.error_message && (
            <p className="mt-2 rounded-md bg-amber-50 px-3 py-2 text-xs text-amber-800" role="note">
              {doc.error_message}
            </p>
          )}
          {doc.status === "failed" && doc.error_message && (
            <p className="mt-2 rounded-md bg-rose-50 px-3 py-2 text-xs text-rose-700" role="note">
              {doc.error_message}
            </p>
          )}
        </div>
        <div className="flex shrink-0 items-center gap-1">
          {doc.status === "failed" && (
            <Button variant="ghost" size="sm" onClick={onRetry} loading={retrying} icon={<RotateCw className="h-3.5 w-3.5" />}>
              Retry
            </Button>
          )}
          <a
            href={api.documentUrl(agentId, doc.id)}
            target="_blank"
            rel="noreferrer"
            className="rounded-md p-2 text-slate-400 hover:bg-slate-100 hover:text-slate-700"
            aria-label={`Open ${doc.filename}`}
            title="Open original"
          >
            <ExternalLink className="h-4 w-4" />
          </a>
          <button
            onClick={onDelete}
            className="rounded-md p-2 text-slate-400 hover:bg-rose-50 hover:text-rose-600"
            aria-label={`Delete ${doc.filename}`}
            title="Delete"
          >
            <Trash2 className="h-4 w-4" />
          </button>
        </div>
      </div>
    </li>
  );
}

export function KnowledgeTab() {
  const { agent } = useOutletContext<WorkspaceContext>();
  const { data: docs, isLoading, error } = useDocuments(agent.id);
  const { data: providers } = useProviders();
  const { upload, remove, reprocess } = useDocumentMutations(agent.id);
  const toast = useToast();
  const [toDelete, setToDelete] = useState<KnowledgeDocument | null>(null);
  const [rejected, setRejected] = useState<{ filename: string; message: string }[]>([]);

  const onFiles = (files: File[]) => {
    setRejected([]);
    upload.mutate(files, {
      onSuccess: (res) => {
        if (res.accepted.length)
          toast.success(
            `${res.accepted.length} document${res.accepted.length > 1 ? "s" : ""} queued`,
            "Processing runs in the background; status updates live below.",
          );
        setRejected(res.rejected);
      },
      onError: (e) => toast.error("Upload failed", (e as Error).message),
    });
  };

  const counts = {
    ready: docs?.filter((d) => d.status === "ready").length ?? 0,
    processing: docs?.filter((d) => d.status === "processing").length ?? 0,
    failed: docs?.filter((d) => d.status === "failed").length ?? 0,
  };

  return (
    <div className="mx-auto max-w-4xl space-y-6 px-4 py-6 sm:px-6">
      <UploadDropzone onFiles={onFiles} disabled={upload.isPending} />
      {upload.isPending && (
        <p className="flex items-center gap-2 text-sm text-slate-500" role="status">
          <Loader2 className="h-4 w-4 animate-spin" aria-hidden /> Uploading…
        </p>
      )}

      {rejected.length > 0 && (
        <Alert tone="warning" title={`${rejected.length} file${rejected.length > 1 ? "s were" : " was"} not added`}>
          <ul className="list-disc space-y-0.5 pl-4">
            {rejected.map((r, i) => (
              <li key={`${i}-${r.filename}`}>{r.message}</li>
            ))}
          </ul>
        </Alert>
      )}

      <Card>
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-100 px-5 py-3.5">
          <div>
            <h2 className="text-sm font-semibold text-slate-900">Documents</h2>
            <p className="text-xs text-slate-500">
              Only this agent can retrieve these documents.
              {providers && ` Embeddings: ${providers.embedding.provider} / ${providers.embedding.model}.`}
            </p>
          </div>
          <div className="flex gap-2 text-xs" aria-live="polite">
            <Badge tone="success">{counts.ready} ready</Badge>
            {counts.processing > 0 && <Badge tone="info">{counts.processing} processing</Badge>}
            {counts.failed > 0 && <Badge tone="danger">{counts.failed} failed</Badge>}
          </div>
        </div>

        {error && (
          <div className="p-5">
            <Alert title="Couldn't load documents">{(error as Error).message}</Alert>
          </div>
        )}
        {isLoading && (
          <div className="space-y-3 p-5">
            <Skeleton className="h-10 w-full" />
            <Skeleton className="h-10 w-full" />
          </div>
        )}
        {docs && docs.length === 0 && (
          <EmptyState
            icon={<FileText className="h-6 w-6" />}
            title="No documents yet"
            description="Upload PDF or Word files. The agent answers only from what's in here."
          />
        )}
        {docs && docs.length > 0 && (
          <ul className="divide-y divide-slate-100" aria-label="Documents">
            {docs.map((d) => (
              <DocumentRow
                key={d.id}
                doc={d}
                agentId={agent.id}
                onDelete={() => setToDelete(d)}
                retrying={reprocess.isPending && reprocess.variables === d.id}
                onRetry={() =>
                  reprocess.mutate(d.id, { onError: (e) => toast.error("Retry failed", (e as Error).message) })
                }
              />
            ))}
          </ul>
        )}
      </Card>

      <ConfirmDialog
        open={!!toDelete}
        onClose={() => setToDelete(null)}
        title="Delete document?"
        description={
          <>
            <strong>{toDelete?.filename}</strong> and all of its indexed chunks and embeddings will be permanently removed.
            The agent will no longer use it in answers.
          </>
        }
        loading={remove.isPending}
        onConfirm={() =>
          toDelete &&
          remove.mutate(toDelete.id, {
            onSuccess: () => {
              toast.success("Document deleted");
              setToDelete(null);
            },
            onError: (e) => toast.error("Couldn't delete document", (e as Error).message),
          })
        }
      />
    </div>
  );
}
