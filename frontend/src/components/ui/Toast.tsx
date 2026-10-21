import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from "react";
import { AlertCircle, CheckCircle2, Info, X } from "lucide-react";
import { cn } from "@/lib/format";

type ToastTone = "success" | "error" | "info";
interface Toast {
  id: number;
  tone: ToastTone;
  title: string;
  description?: string;
}

interface ToastApi {
  success: (title: string, description?: string) => void;
  error: (title: string, description?: string) => void;
  info: (title: string, description?: string) => void;
}

const ToastContext = createContext<ToastApi | null>(null);

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const dismiss = useCallback((id: number) => setToasts((t) => t.filter((x) => x.id !== id)), []);
  const push = useCallback(
    (tone: ToastTone, title: string, description?: string) => {
      const id = Date.now() + Math.random();
      setToasts((t) => [...t.slice(-3), { id, tone, title, description }]);
      setTimeout(() => dismiss(id), tone === "error" ? 8000 : 4000);
    },
    [dismiss],
  );
  const api = useMemo<ToastApi>(
    () => ({
      success: (t, d) => push("success", t, d),
      error: (t, d) => push("error", t, d),
      info: (t, d) => push("info", t, d),
    }),
    [push],
  );

  const icons = {
    success: <CheckCircle2 className="h-5 w-5 text-emerald-500" aria-hidden />,
    error: <AlertCircle className="h-5 w-5 text-rose-500" aria-hidden />,
    info: <Info className="h-5 w-5 text-brand-500" aria-hidden />,
  };

  return (
    <ToastContext.Provider value={api}>
      {children}
      <div
        aria-live="polite"
        className="pointer-events-none fixed bottom-4 right-4 z-[60] flex w-full max-w-sm flex-col gap-2"
      >
        {toasts.map((t) => (
          <div
            key={t.id}
            role={t.tone === "error" ? "alert" : "status"}
            className={cn(
              "pointer-events-auto flex animate-fade-in items-start gap-3 rounded-xl border bg-white p-4 shadow-pop",
              t.tone === "error" ? "border-rose-200" : "border-slate-200",
            )}
          >
            {icons[t.tone]}
            <div className="min-w-0 flex-1">
              <p className="text-sm font-medium text-slate-900">{t.title}</p>
              {t.description && <p className="mt-0.5 text-sm text-slate-600">{t.description}</p>}
            </div>
            <button
              onClick={() => dismiss(t.id)}
              className="rounded p-0.5 text-slate-400 hover:text-slate-600"
              aria-label="Dismiss notification"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}

export function useToast(): ToastApi {
  const ctx = useContext(ToastContext);
  if (!ctx) throw new Error("useToast must be used inside <ToastProvider>");
  return ctx;
}
