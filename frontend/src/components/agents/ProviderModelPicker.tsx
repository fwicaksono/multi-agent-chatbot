import { useId } from "react";
import { AlertTriangle, CheckCircle2 } from "lucide-react";
import { useModels, useProviders } from "@/lib/queries";
import { Field, Input, Select } from "@/components/ui/primitives";
import { cn } from "@/lib/format";

interface Props {
  provider: string;
  model: string;
  onChange: (provider: string, model: string) => void;
  idPrefix: string;
  label?: string;
  allowNone?: boolean;
  compact?: boolean;
}

/**
 * Provider dropdown + model input with a live, searchable model list
 * (fetched from the provider when possible, curated suggestions otherwise).
 */
export function ProviderModelPicker({ provider, model, onChange, idPrefix, label = "LLM", allowNone, compact }: Props) {
  const { data: providers } = useProviders();
  const { data: models, isFetching } = useModels(provider || undefined);
  const listId = useId();
  const info = providers?.providers.find((p) => p.name === provider);

  const handleProvider = (next: string) => {
    if (!next) return onChange("", "");
    const nextInfo = providers?.providers.find((p) => p.name === next);
    // Keep the model when switching providers only if the new one suggests it.
    const keep = nextInfo?.suggested_models.includes(model);
    onChange(next, keep ? model : (nextInfo?.suggested_models[0] ?? ""));
  };

  return (
    <div className={cn("grid gap-4", compact ? "grid-cols-1" : "sm:grid-cols-2")}>
      <Field
        label={`${label} provider`}
        htmlFor={`${idPrefix}-provider`}
        required={!allowNone}
        hint={
          info ? (
            info.configured ? (
              <span className="inline-flex items-center gap-1 text-emerald-700">
                <CheckCircle2 className="h-3.5 w-3.5" aria-hidden /> Configured
              </span>
            ) : (
              <span className="inline-flex items-center gap-1 text-amber-700">
                <AlertTriangle className="h-3.5 w-3.5" aria-hidden /> Not configured: set {info.required_env.join(" / ")}
              </span>
            )
          ) : allowNone ? (
            "Optional: used if the primary provider fails"
          ) : undefined
        }
      >
        <Select id={`${idPrefix}-provider`} value={provider} onChange={(e) => handleProvider(e.target.value)}>
          {allowNone && <option value="">None</option>}
          {!allowNone && !provider && <option value="">Select a provider…</option>}
          {providers?.providers.map((p) => (
            <option key={p.name} value={p.name}>
              {p.display_name}
              {p.configured ? "" : " (not configured)"}
            </option>
          ))}
        </Select>
      </Field>

      <Field
        label={`${label} model`}
        htmlFor={`${idPrefix}-model`}
        required={!allowNone}
        hint={
          provider
            ? isFetching
              ? "Loading models…"
              : models?.source === "live"
                ? `${models.models.length} models available — type to search`
                : "Suggested models — any model ID the provider accepts works"
            : undefined
        }
      >
        <Input
          id={`${idPrefix}-model`}
          list={listId}
          value={model}
          disabled={!provider}
          placeholder={provider ? "e.g. gpt-4.1-mini" : "Choose a provider first"}
          onChange={(e) => onChange(provider, e.target.value)}
          autoComplete="off"
          spellCheck={false}
        />
        <datalist id={listId}>
          {(models?.models ?? []).slice(0, 500).map((m) => (
            <option key={m.id} value={m.id}>
              {m.name && m.name !== m.id ? m.name : ""}
              {m.input_price_per_mtok != null ? ` · $${m.input_price_per_mtok}/$${m.output_price_per_mtok} per 1M` : ""}
            </option>
          ))}
        </datalist>
      </Field>
    </div>
  );
}
