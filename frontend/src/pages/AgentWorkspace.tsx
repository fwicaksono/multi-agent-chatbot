import { NavLink, Outlet, useParams } from "react-router-dom";
import { BarChart3, FileText, MessageSquare, Settings2 } from "lucide-react";
import { useAgent } from "@/lib/queries";
import { AgentAvatar } from "@/components/layout/AppShell";
import { Alert, Badge, Skeleton } from "@/components/ui/primitives";
import { cn, providerLabel } from "@/lib/format";
import type { Agent } from "@/lib/types";

export type WorkspaceContext = { agent: Agent };

const TABS = [
  { to: "chat", label: "Chat", icon: MessageSquare },
  { to: "knowledge", label: "Knowledge", icon: FileText },
  { to: "settings", label: "Settings", icon: Settings2 },
  { to: "usage", label: "Usage", icon: BarChart3 },
];

export function AgentWorkspace() {
  const { agentId } = useParams();
  const { data: agent, error, isLoading } = useAgent(agentId);

  if (error)
    return (
      <div className="mx-auto max-w-3xl p-8">
        <Alert title="Agent not available">{(error as Error).message}</Alert>
      </div>
    );

  return (
    <div className="flex h-full flex-col">
      <div className="border-b border-slate-200 bg-white px-4 pt-4 sm:px-6">
        <div className="flex items-center gap-3">
          {isLoading || !agent ? (
            <>
              <Skeleton className="h-11 w-11" />
              <Skeleton className="h-5 w-48" />
            </>
          ) : (
            <>
              <AgentAvatar name={agent.name} size="lg" />
              <div className="min-w-0 flex-1">
                <h1 className="truncate text-lg font-semibold tracking-tight">{agent.name}</h1>
                <p className="truncate text-sm text-slate-500">{agent.description || "No description"}</p>
              </div>
              <div className="hidden items-center gap-2 sm:flex">
                <Badge tone="brand">{providerLabel(agent.provider)}</Badge>
                <Badge className="font-mono">{agent.model}</Badge>
              </div>
            </>
          )}
        </div>
        <nav aria-label="Agent sections" className="-mb-px mt-3 flex gap-1 overflow-x-auto">
          {TABS.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                cn(
                  "inline-flex items-center gap-2 border-b-2 px-3 py-2.5 text-sm font-medium transition-colors",
                  isActive
                    ? "border-brand-600 text-brand-700"
                    : "border-transparent text-slate-500 hover:border-slate-300 hover:text-slate-800",
                )
              }
            >
              <Icon className="h-4 w-4" aria-hidden />
              {label}
              {to === "knowledge" && agent && (
                <span className="rounded-full bg-slate-100 px-1.5 text-xs text-slate-600">
                  {agent.document_counts.total}
                </span>
              )}
            </NavLink>
          ))}
        </nav>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto">{agent && <Outlet context={{ agent } satisfies WorkspaceContext} />}</div>
    </div>
  );
}
