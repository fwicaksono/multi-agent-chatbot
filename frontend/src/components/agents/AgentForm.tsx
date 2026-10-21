import { useEffect, useState, type FormEvent } from "react";
import { ChevronDown, Sparkles } from "lucide-react";
import { useProviders } from "@/lib/queries";
import type { Agent, AgentInput } from "@/lib/types";
import { Button, Field, Input, Textarea } from "@/components/ui/primitives";
import { ProviderModelPicker } from "./ProviderModelPicker";
import { cn } from "@/lib/format";

const TEMPLATES: { label: string; description: string; instructions: string }[] = [
  {
    label: "HR assistant",
    description: "Answers employee questions from the employee handbook.",
    instructions:
      "You are the HR assistant for our employees. Answer questions about leave, benefits, working hours and company policies using only the employee handbook. Be friendly and concise, and mention the exact policy numbers (days, amounts, deadlines) when relevant.",
  },
  {
    label: "Legal assistant",
    description: "Explains contract templates and legal policies.",
    instructions:
      "You are the legal assistant. Explain clauses from our contract templates and legal documents precisely. Quote clause numbers and defined terms, highlight obligations and deadlines, and never give advice beyond what the documents state.",
  },
  {
    label: "Support agent",
    description: "Answers product questions from documentation.",
    instructions:
      "You are a customer support agent. Answer product questions step by step using the documentation. Use numbered steps for procedures.",
  },
];

type Errors = Partial<Record<"name" | "provider" | "model" | "fallback", string>>;

export function AgentForm({
  initial,
  submitLabel,
  onSubmit,
  submitting,
  onCancel,
}: {
  initial?: Agent;
  submitLabel: string;
  onSubmit: (data: AgentInput) => void;
  submitting?: boolean;
  onCancel?: () => void;
}) {
  const { data: providers } = useProviders();
  const [form, setForm] = useState<AgentInput>({
    name: initial?.name ?? "",
    description: initial?.description ?? "",
    instructions: initial?.instructions ?? "",
    provider: initial?.provider ?? "",
    model: initial?.model ?? "",
    temperature: initial?.temperature ?? 0.1,
    fallback_provider: initial?.fallback_provider ?? null,
    fallback_model: initial?.fallback_model ?? null,
  });
  const [errors, setErrors] = useState<Errors>({});
  const [advanced, setAdvanced] = useState(Boolean(initial?.fallback_provider));

  // Pre-fill defaults for new agents once provider info has loaded.
  useEffect(() => {
    if (!initial && providers && !form.provider) {
      setForm((f) => ({ ...f, provider: providers.default_provider, model: providers.default_model }));
    }
  }, [providers, initial, form.provider]);

  const set = <K extends keyof AgentInput>(key: K, value: AgentInput[K]) => setForm((f) => ({ ...f, [key]: value }));

  const validate = (): Errors => {
    const e: Errors = {};
    if (!form.name.trim()) e.name = "Give the agent a name.";
    if (!form.provider) e.provider = "Choose an LLM provider.";
    if (!form.model.trim()) e.model = "Enter a model ID.";
    if (Boolean(form.fallback_provider) !== Boolean(form.fallback_model?.trim()))
      e.fallback = "Set both a fallback provider and model, or leave both empty.";
    return e;
  };

  const submit = (ev: FormEvent) => {
    ev.preventDefault();
    const e = validate();
    setErrors(e);
    if (Object.keys(e).length) return;
    onSubmit({
      ...form,
      name: form.name.trim(),
      model: form.model.trim(),
      fallback_provider: form.fallback_provider || null,
      fallback_model: form.fallback_model?.trim() || null,
    });
  };

  return (
    <form onSubmit={submit} className="space-y-8" noValidate>
      <section className="space-y-5">
        <div>
          <h2 className="text-sm font-semibold text-slate-900">Identity</h2>
          <p className="text-sm text-slate-500">How the agent appears to people choosing who to chat with.</p>
        </div>
        <Field label="Name" htmlFor="agent-name" required error={errors.name}>
          <Input
            id="agent-name"
            value={form.name}
            maxLength={120}
            placeholder="e.g. Agent HR"
            aria-invalid={!!errors.name}
            aria-describedby={errors.name ? "agent-name-error" : undefined}
            onChange={(e) => set("name", e.target.value)}
          />
        </Field>
        <Field label="Description" htmlFor="agent-description" hint="One sentence about what this agent knows.">
          <Input
            id="agent-description"
            value={form.description}
            maxLength={2000}
            placeholder="Answers questions about the employee handbook"
            onChange={(e) => set("description", e.target.value)}
          />
        </Field>
      </section>

      <section className="space-y-5">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <h2 className="text-sm font-semibold text-slate-900">Instructions</h2>
            <p className="text-sm text-slate-500">
              The system prompt. Grounding rules (answer only from documents, cite sources, say “I don’t know”) are
              always added automatically.
            </p>
          </div>
        </div>
        <div className="flex flex-wrap gap-2" aria-label="Instruction templates">
          {TEMPLATES.map((t) => (
            <button
              key={t.label}
              type="button"
              title={t.description}
              onClick={() => {
                set("instructions", t.instructions);
                if (!form.description) set("description", t.description);
              }}
              className="inline-flex items-center gap-1.5 rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-medium text-slate-600 hover:border-brand-300 hover:text-brand-700"
            >
              <Sparkles className="h-3 w-3" aria-hidden /> {t.label}
            </button>
          ))}
        </div>
        <Field label="System prompt" htmlFor="agent-instructions" hint={`${form.instructions.length.toLocaleString()} / 20,000 characters`}>
          <Textarea
            id="agent-instructions"
            rows={7}
            maxLength={20000}
            value={form.instructions}
            placeholder="You are the HR assistant. Answer questions about company policies…"
            onChange={(e) => set("instructions", e.target.value)}
          />
        </Field>
      </section>

      <section className="space-y-5">
        <div>
          <h2 className="text-sm font-semibold text-slate-900">Language model</h2>
          <p className="text-sm text-slate-500">
            Switch provider or model any time — conversation history and the knowledge base are kept.
          </p>
        </div>
        <ProviderModelPicker
          idPrefix="agent"
          provider={form.provider}
          model={form.model}
          onChange={(provider, model) => setForm((f) => ({ ...f, provider, model }))}
        />
        {(errors.provider || errors.model) && (
          <p className="text-xs text-rose-600" role="alert">
            {errors.provider ?? errors.model}
          </p>
        )}

        <div className="rounded-xl border border-slate-200">
          <button
            type="button"
            onClick={() => setAdvanced((a) => !a)}
            aria-expanded={advanced}
            className="flex w-full items-center justify-between px-4 py-3 text-left text-sm font-medium text-slate-700"
          >
            Advanced: temperature & fallback provider
            <ChevronDown className={cn("h-4 w-4 transition-transform", advanced && "rotate-180")} aria-hidden />
          </button>
          {advanced && (
            <div className="space-y-5 border-t border-slate-100 px-4 py-4">
              <Field
                label={`Temperature: ${form.temperature.toFixed(1)}`}
                htmlFor="agent-temperature"
                hint="Lower is more factual and consistent. 0–0.3 recommended for document Q&A."
              >
                <input
                  id="agent-temperature"
                  type="range"
                  min={0}
                  max={1}
                  step={0.1}
                  value={form.temperature}
                  onChange={(e) => set("temperature", Number(e.target.value))}
                  className="w-full accent-brand-600"
                />
              </Field>
              <ProviderModelPicker
                idPrefix="fallback"
                label="Fallback"
                allowNone
                provider={form.fallback_provider ?? ""}
                model={form.fallback_model ?? ""}
                onChange={(provider, model) =>
                  setForm((f) => ({ ...f, fallback_provider: provider || null, fallback_model: model || null }))
                }
              />
              {errors.fallback && (
                <p className="text-xs text-rose-600" role="alert">
                  {errors.fallback}
                </p>
              )}
              {providers?.fallback_chain.length ? (
                <p className="text-xs text-slate-500">
                  Global fallback chain (from <code className="font-mono">LLM_FALLBACK_CHAIN</code>):{" "}
                  {providers.fallback_chain.join(" → ")}
                </p>
              ) : null}
            </div>
          )}
        </div>
      </section>

      <div className="flex justify-end gap-2 border-t border-slate-200 pt-5">
        {onCancel && (
          <Button type="button" variant="secondary" onClick={onCancel}>
            Cancel
          </Button>
        )}
        <Button type="submit" loading={submitting}>
          {submitLabel}
        </Button>
      </div>
    </form>
  );
}
