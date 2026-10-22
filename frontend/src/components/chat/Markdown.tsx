import { memo } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Source } from "@/lib/types";
import { cn } from "@/lib/format";

/**
 * Turn "[1]" / "[1, 3]" / "[2-3]" citation markers into links "#cite-n" that
 * are rendered as clickable chips. Fenced/inline code is left untouched.
 */
function linkCitations(text: string, maxIndex: number): string {
  return text
    .split(/(```[\s\S]*?```|`[^`]*`)/g)
    .map((part, i) =>
      i % 2 === 1
        ? part
        : part.replace(/\[(\d+(?:\s*[,\-–]\s*\d+)*)\](?!\()/g, (_m: string, group: string) => {
            const nums: number[] = [];
            for (const piece of group.split(/\s*,\s*/)) {
              const range = piece.split(/\s*[-–]\s*/).map(Number);
              if (range.length === 2 && range[1] - range[0] < 20)
                for (let n = range[0]; n <= range[1]; n++) nums.push(n);
              else nums.push(Number(piece));
            }
            // Leave "[2023]" and other non-source numbers as plain text.
            if (!nums.length || nums.some((n) => !Number.isInteger(n) || n < 1 || n > maxIndex)) return _m;
            return nums.map((n) => `[${n}](#cite-${n})`).join("");
          }),
    )
    .join("");
}

export function CitationChip({
  n,
  source,
  onClick,
}: {
  n: number;
  source?: Source;
  onClick?: (s: Source) => void;
}) {
  const label = source ? `Source ${n}: ${source.document_name}, ${source.location}` : `Source ${n}`;
  return (
    <button
      type="button"
      onClick={() => source && onClick?.(source)}
      title={label}
      aria-label={label}
      disabled={!source}
      className={cn(
        "mx-0.5 inline-flex h-[18px] min-w-[18px] -translate-y-px items-center justify-center rounded-md px-1 align-middle text-[10.5px] font-semibold leading-none transition-colors",
        source
          ? "bg-brand-100 text-brand-700 hover:bg-brand-600 hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-500"
          : "bg-slate-100 text-slate-500",
      )}
    >
      {n}
    </button>
  );
}

export const Markdown = memo(function Markdown({
  content,
  sources,
  onCite,
}: {
  content: string;
  sources: Source[];
  onCite: (s: Source) => void;
}) {
  const byIndex = new Map(sources.map((s) => [s.index, s]));
  return (
    <div className="prose prose-sm prose-slate max-w-none prose-p:my-2 prose-ul:my-2 prose-ol:my-2 prose-li:my-0.5 prose-pre:bg-slate-900 prose-table:my-2 prose-th:px-2 prose-td:px-2">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ href, children, ...props }) => {
            const m = href?.match(/^#cite-(\d+)$/);
            if (m) {
              const n = Number(m[1]);
              return <CitationChip n={n} source={byIndex.get(n)} onClick={onCite} />;
            }
            return (
              <a href={href} target="_blank" rel="noreferrer" {...props}>
                {children}
              </a>
            );
          },
        }}
      >
        {linkCitations(content, sources.length)}
      </ReactMarkdown>
    </div>
  );
});
