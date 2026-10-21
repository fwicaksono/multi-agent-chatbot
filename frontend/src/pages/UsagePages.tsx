import { Link, useOutletContext } from "react-router-dom";
import { BarChart3, Info } from "lucide-react";
import { useAgentUsage, useUsageSummary } from "@/lib/queries";
import { PageHeader } from "@/components/layout/AppShell";
import { Alert, Card, EmptyState, Skeleton } from "@/components/ui/primitives";
import { formatCost, formatNumber, providerLabel } from "@/lib/format";
import type { UsageTotals } from "@/lib/types";
import type { WorkspaceContext } from "./AgentWorkspace";

const KIND_LABELS: Record<string, string> = {
  chat: "Chat answers",
  query_rewrite: "Follow-up rewriting",
  embedding_query: "Query embeddings",
  embedding_ingest: "Document embeddings",
};

function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <Card className="p-4">
      <p className="text-xs font-medium text-slate-500">{label}</p>
      <p className="mt-1 text-2xl font-semibold tabular-nums tracking-tight text-slate-900">{value}</p>
      {sub && <p className="mt-0.5 text-xs text-slate-400">{sub}</p>}
    </Card>
  );
}

function Totals({ totals }: { totals: UsageTotals }) {
  return (
    <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
      <Stat label="Estimated cost" value={formatCost(Number(totals.cost_usd))} sub="USD" />
      <Stat label="Requests" value={formatNumber(totals.requests)} />
      <Stat label="Input tokens" value={formatNumber(totals.prompt_tokens)} />
      <Stat label="Output tokens" value={formatNumber(totals.completion_tokens)} />
    </div>
  );
}

function DailyBars({ daily }: { daily: { day: string; cost_usd: number; requests: number }[] }) {
  if (!daily.length) return null;
  const max = Math.max(...daily.map((d) => Number(d.cost_usd)), 0.000001);
  return (
    <Card className="p-5">
      <h3 className="text-sm font-semibold text-slate-900">Daily cost (last 30 days)</h3>
      <div className="mt-4 flex h-32 items-end gap-1.5" role="img" aria-label="Bar chart of daily cost">
        {daily.map((d) => (
          <div key={d.day} className="group flex flex-1 flex-col items-center justify-end" title={`${d.day}: ${formatCost(Number(d.cost_usd))}, ${d.requests} requests`}>
            <div
              className="w-full max-w-[28px] rounded-t bg-brand-500 transition-colors group-hover:bg-brand-600"
              style={{ height: `${Math.max((Number(d.cost_usd) / max) * 100, 3)}%` }}
            />
            <span className="mt-1 text-[10px] text-slate-400">{d.day.slice(5)}</span>
          </div>
        ))}
      </div>
    </Card>
  );
}

export function AgentUsageTab() {
  const { agent } = useOutletContext<WorkspaceContext>();
  const { data, isLoading, error } = useAgentUsage(agent.id);

  return (
    <div className="mx-auto max-w-5xl space-y-6 px-4 py-6 sm:px-6">
      {error && <Alert title="Couldn't load usage">{(error as Error).message}</Alert>}
      {isLoading && <Skeleton className="h-24 w-full" />}
      {data && (
        <>
          <Totals totals={data.totals} />
          <DailyBars daily={data.daily} />
          <Card>
            <div className="border-b border-slate-100 px-5 py-3.5">
              <h3 className="text-sm font-semibold text-slate-900">Breakdown by model and purpose</h3>
            </div>
            {data.by_model.length === 0 ? (
              <EmptyState icon={<BarChart3 className="h-6 w-6" />} title="No usage yet" description="Upload documents or chat to see token usage." />
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-xs uppercase tracking-wide text-slate-500">
                      <th scope="col" className="px-5 py-2.5 font-medium">Purpose</th>
                      <th scope="col" className="px-5 py-2.5 font-medium">Provider · model</th>
                      <th scope="col" className="px-5 py-2.5 text-right font-medium">Requests</th>
                      <th scope="col" className="px-5 py-2.5 text-right font-medium">Input</th>
                      <th scope="col" className="px-5 py-2.5 text-right font-medium">Output</th>
                      <th scope="col" className="px-5 py-2.5 text-right font-medium">Cost</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {data.by_model.map((r) => (
                      <tr key={`${r.provider}-${r.model}-${r.kind}`}>
                        <td className="px-5 py-2.5 text-slate-700">{KIND_LABELS[r.kind] ?? r.kind}</td>
                        <td className="px-5 py-2.5">
                          <span className="text-slate-700">{providerLabel(r.provider)}</span>{" "}
                          <span className="font-mono text-xs text-slate-500">{r.model}</span>
                        </td>
                        <td className="px-5 py-2.5 text-right tabular-nums">{formatNumber(r.requests)}</td>
                        <td className="px-5 py-2.5 text-right tabular-nums">{formatNumber(r.prompt_tokens)}</td>
                        <td className="px-5 py-2.5 text-right tabular-nums">{formatNumber(r.completion_tokens)}</td>
                        <td className="px-5 py-2.5 text-right font-medium tabular-nums">{formatCost(Number(r.cost_usd))}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Card>
          <p className="flex items-start gap-2 text-xs text-slate-500">
            <Info className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden />
            Costs use the provider-reported amount when available (OpenRouter), otherwise an estimate from list prices.
            {data.unpriced_requests > 0 && ` ${data.unpriced_requests} request(s) used models without known pricing and are counted as $0.`}
          </p>
        </>
      )}
    </div>
  );
}

export function UsagePage() {
  const { data, isLoading, error } = useUsageSummary();
  const total = (data ?? []).reduce(
    (acc, r) => ({
      requests: acc.requests + r.requests,
      prompt_tokens: acc.prompt_tokens + r.prompt_tokens,
      completion_tokens: acc.completion_tokens + r.completion_tokens,
      cost_usd: acc.cost_usd + Number(r.cost_usd),
    }),
    { requests: 0, prompt_tokens: 0, completion_tokens: 0, cost_usd: 0 },
  );
  const max = Math.max(...(data ?? []).map((r) => Number(r.cost_usd)), 0.000001);

  return (
    <div className="mx-auto max-w-5xl space-y-6 px-4 py-8 sm:px-8">
      <PageHeader title="Usage & cost" description="Token usage and estimated LLM spend across all agents." />
      {error && <Alert title="Couldn't load usage">{(error as Error).message}</Alert>}
      {isLoading && <Skeleton className="h-24 w-full" />}
      {data && (
        <>
          <Totals totals={total} />
          <Card>
            <div className="border-b border-slate-100 px-5 py-3.5">
              <h2 className="text-sm font-semibold text-slate-900">By agent</h2>
            </div>
            {data.length === 0 ? (
              <EmptyState icon={<BarChart3 className="h-6 w-6" />} title="No agents yet" />
            ) : (
              <ul className="divide-y divide-slate-100">
                {data.map((r) => (
                  <li key={r.agent_id} className="flex items-center gap-4 px-5 py-3">
                    <Link to={`/agents/${r.agent_id}/usage`} className="w-48 shrink-0 truncate text-sm font-medium text-slate-800 hover:text-brand-700">
                      {r.agent_name}
                    </Link>
                    <div className="h-2 flex-1 overflow-hidden rounded-full bg-slate-100" aria-hidden>
                      <div className="h-full rounded-full bg-brand-500" style={{ width: `${(Number(r.cost_usd) / max) * 100}%` }} />
                    </div>
                    <span className="w-24 text-right text-xs tabular-nums text-slate-500">{formatNumber(r.prompt_tokens + r.completion_tokens)} tok</span>
                    <span className="w-20 text-right text-sm font-medium tabular-nums">{formatCost(Number(r.cost_usd))}</span>
                  </li>
                ))}
              </ul>
            )}
          </Card>
        </>
      )}
    </div>
  );
}
