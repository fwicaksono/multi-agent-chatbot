import { Link } from "react-router-dom";
import { AlertTriangle, Bot, FileText, Loader2, MessageSquare, Plus, Settings2 } from "lucide-react";
import { useAgents, useProviders } from "@/lib/queries";
import { AgentAvatar, PageHeader } from "@/components/layout/AppShell";
import { Alert, Badge, Button, Card, EmptyState, LinkButton, Skeleton } from "@/components/ui/primitives";
import { formatRelative, providerLabel } from "@/lib/format";
import type { Agent } from "@/lib/types";

function AgentCard({ agent }: { agent: Agent }) {
  const c = agent.document_counts;
  return (
    <Card className="flex flex-col p-5 transition-shadow hover:shadow-pop">
      <div className="flex items-start gap-3">
        <AgentAvatar name={agent.name} size="lg" />
        <div className="min-w-0 flex-1">
          <Link
            to={`/agents/${agent.id}/chat`}
            className="block truncate font-semibold text-slate-900 hover:text-brand-700"
          >
            {agent.name}
          </Link>
          <p className="mt-0.5 line-clamp-2 min-h-[2.5rem] text-sm text-slate-500">
            {agent.description || "No description"}
          </p>
        </div>
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-2">
        <Badge tone="brand">{providerLabel(agent.provider)}</Badge>
        <Badge className="max-w-[12rem] truncate font-mono">{agent.model}</Badge>
        {agent.fallback_provider && <Badge tone="info">+ fallback</Badge>}
      </div>

      <dl className="mt-4 flex items-center gap-4 text-xs text-slate-500">
        <div className="flex items-center gap-1.5">
          <FileText className="h-3.5 w-3.5" aria-hidden />
          <dt className="sr-only">Documents ready</dt>
          <dd>
            {c.ready} ready{c.total !== c.ready ? ` / ${c.total}` : ""}
          </dd>
        </div>
        {c.processing > 0 && (
          <div className="flex items-center gap-1.5 text-sky-700">
            <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden />
            <dd>{c.processing} processing</dd>
          </div>
        )}
        {c.failed > 0 && (
          <div className="flex items-center gap-1.5 text-rose-600">
            <AlertTriangle className="h-3.5 w-3.5" aria-hidden />
            <dd>{c.failed} failed</dd>
          </div>
        )}
        <span className="ml-auto">Updated {formatRelative(agent.updated_at)}</span>
      </dl>

      <div className="mt-4 flex gap-2 border-t border-slate-100 pt-4">
        <LinkButton to={`/agents/${agent.id}/chat`} size="sm" className="flex-1" icon={<MessageSquare className="h-3.5 w-3.5" />}>
          Chat
        </LinkButton>
        <LinkButton to={`/agents/${agent.id}/knowledge`} size="sm" variant="secondary" icon={<FileText className="h-3.5 w-3.5" />}>
          Knowledge
        </LinkButton>
        <LinkButton
          to={`/agents/${agent.id}/settings`}
          size="icon"
          variant="secondary"
          aria-label={`Settings for ${agent.name}`}
          icon={<Settings2 className="h-3.5 w-3.5" />}
        />
      </div>
    </Card>
  );
}

export function AgentsPage() {
  const { data: agents, isLoading, error, refetch } = useAgents();
  const { data: providers } = useProviders();
  const unconfigured = providers?.providers.filter((p) => !p.configured) ?? [];

  return (
    <div className="mx-auto max-w-6xl space-y-6 px-4 py-8 sm:px-8">
      <PageHeader
        title="Agents"
        description="Each agent has its own instructions, LLM and isolated knowledge base."
        actions={
          <LinkButton to="/agents/new" icon={<Plus className="h-4 w-4" />}>
            New agent
          </LinkButton>
        }
      />

      {unconfigured.length > 0 && providers && (
        <Alert tone="info" title="Some providers are not configured">
          {unconfigured.map((p) => `${p.display_name} (${p.required_env.join(", ")})`).join(" · ")}. Agents using them
          will automatically fall back to the configured fallback chain.
        </Alert>
      )}

      {error && (
        <Alert
          title="Couldn't load agents"
          action={
            <Button size="sm" variant="secondary" onClick={() => refetch()}>
              Retry
            </Button>
          }
        >
          {(error as Error).message}
        </Alert>
      )}

      {isLoading && (
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {Array.from({ length: 3 }).map((_, i) => (
            <Card key={i} className="space-y-4 p-5">
              <div className="flex gap-3">
                <Skeleton className="h-11 w-11" />
                <div className="flex-1 space-y-2">
                  <Skeleton className="h-4 w-1/2" />
                  <Skeleton className="h-3 w-full" />
                </div>
              </div>
              <Skeleton className="h-8 w-full" />
            </Card>
          ))}
        </div>
      )}

      {agents && agents.length === 0 && (
        <Card>
          <EmptyState
            icon={<Bot className="h-6 w-6" />}
            title="Create your first agent"
            description="For example an “Agent HR” that knows the employee handbook, or an “Agent Legal” for contract templates."
            action={
              <LinkButton to="/agents/new" icon={<Plus className="h-4 w-4" />}>
                New agent
              </LinkButton>
            }
          />
        </Card>
      )}

      {agents && agents.length > 0 && (
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {agents.map((a) => (
            <AgentCard key={a.id} agent={a} />
          ))}
        </div>
      )}
    </div>
  );
}
