import { useRef, useState, type DragEvent } from "react";
import { UploadCloud } from "lucide-react";
import { cn } from "@/lib/format";

const ACCEPT = ".pdf,.docx,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document";

export function UploadDropzone({ onFiles, disabled }: { onFiles: (files: File[]) => void; disabled?: boolean }) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const handle = (list: FileList | null) => {
    const files = Array.from(list ?? []);
    if (files.length) onFiles(files);
  };

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragging(false);
    if (!disabled) handle(e.dataTransfer.files);
  };

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
      className={cn(
        "relative flex flex-col items-center justify-center rounded-xl border-2 border-dashed px-6 py-9 text-center transition-colors",
        dragging ? "border-brand-400 bg-brand-50/60" : "border-slate-200 bg-white hover:border-slate-300",
        disabled && "opacity-60",
      )}
    >
      <div className="mb-3 flex h-11 w-11 items-center justify-center rounded-full bg-brand-50 text-brand-600 ring-1 ring-brand-100">
        <UploadCloud className="h-5 w-5" aria-hidden />
      </div>
      <p className="text-sm text-slate-700">
        <button
          type="button"
          disabled={disabled}
          onClick={() => inputRef.current?.click()}
          className="font-semibold text-brand-700 underline-offset-2 hover:underline focus:outline-none focus-visible:ring-2 focus-visible:ring-brand-500"
        >
          Choose files
        </button>{" "}
        or drag and drop them here
      </p>
      <p className="mt-1 text-xs text-slate-500">PDF or DOCX · multiple files · up to 25 MB each · scanned PDFs are OCR’d</p>
      <input
        ref={inputRef}
        type="file"
        multiple
        accept={ACCEPT}
        className="sr-only"
        aria-label="Upload documents"
        onChange={(e) => {
          handle(e.target.files);
          e.target.value = "";
        }}
      />
    </div>
  );
}
