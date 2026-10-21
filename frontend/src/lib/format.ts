import clsx, { type ClassValue } from "clsx";

export const cn = (...inputs: ClassValue[]) => clsx(inputs);

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

export function formatCost(usd: number | null | undefined): string {
  if (usd == null) return "—";
  if (usd === 0) return "$0.00";
  if (usd < 0.01) return `$${usd.toFixed(5)}`;
  return `$${usd.toFixed(2)}`;
}

export function formatNumber(n: number): string {
  return new Intl.NumberFormat("en-US").format(n);
}

export function formatRelative(iso: string): string {
  const diff = (Date.now() - new Date(iso).getTime()) / 1000;
  if (diff < 45) return "just now";
  if (diff < 3600) return `${Math.round(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.round(diff / 3600)}h ago`;
  if (diff < 86400 * 7) return `${Math.round(diff / 86400)}d ago`;
  return new Date(iso).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}

const PROVIDER_LABELS: Record<string, string> = {
  openai: "OpenAI",
  litellm: "LiteLLM",
  openrouter: "OpenRouter",
};

export const providerLabel = (name: string | null | undefined) =>
  name ? (PROVIDER_LABELS[name] ?? name) : "—";

export const STAGE_LABELS: Record<string, string> = {
  queued: "Queued",
  extracting: "Extracting text",
  ocr: "Running OCR",
  chunking: "Chunking",
  embedding: "Generating embeddings",
  indexing: "Indexing",
  done: "Ready",
  failed: "Failed",
};
