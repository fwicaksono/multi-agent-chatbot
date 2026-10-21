import { useState, type ReactNode } from "react";
import { Link, NavLink, Outlet, useParams } from "react-router-dom";
import { BarChart3, Bot, Menu, MessageSquareText, Plus, X } from "lucide-react";
import { useAgents } from "@/lib/queries";
import { cn, providerLabel } from "@/lib/format";
import { Skeleton } from "@/components/ui/primitives";

function NavItem({ to, icon, children, end }: { to: string; icon: ReactNode; children: ReactNode; end?: boolean }) {
  return (
    <NavLink
      to={to}
      end={end}
      className={({ isActive }) =>
        cn(
          "flex items-center gap-2.5 rounded-lg px-2.5 py-2 text-sm font-medium transition-colors",
          isActive ? "bg-white text-slate-900 shadow-card ring-1 ring-slate-200" : "text-slate-600 hover:bg-slate-100 hover:text-slate-900",
        )
      }
    >
      {icon}
      {children}
    </NavLink>
  );
}

function Sidebar({ onNavigate }: { onNavigate?: () => void }) {
  const { data: agents, isLoading } = useAgents();
  const { agentId } = useParams();

  return (
    <nav aria-label="Main" className="flex h-full flex-col gap-6 px-3 py-4" onClick={onNavigate}>
      <Link to="/" className="flex items-center gap-2.5 px-2.5">
        <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-brand-600 text-white shadow-sm">
          <MessageSquareText className="h-4 w-4" aria-hidden />
        </span>
        <span className="text-[15px] font-semibold tracking-tight text-slate-900">Agentdesk</span>
      </Link>

      <div className="space-y-1">
        <NavItem to="/agents" end icon={<Bot className="h-4 w-4" aria-hidden />}>
          All agents
        </NavItem>
        <NavItem to="/usage" icon={<BarChart3 className="h-4 w-4" aria-hidden />}>
          Usage & cost
        </NavItem>
      </div>

      <div className="min-h-0 flex-1">
        <div className="mb-2 flex items-center justify-between px-2.5">
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">Agents</span>
          <Link
            to="/agents/new"
            className="rounded-md p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-700"
            aria-label="Create agent"
            title="Create agent"
          >
            <Plus className="h-4 w-4" />
          </Link>
        </div>
        <ul className="max-h-full space-y-0.5 overflow-y-auto pb-4">
          {isLoading &&
            Array.from({ length: 3 }).map((_, i) => (
              <li key={i} className="px-2.5 py-2">
                <Skeleton className="h-4 w-32" />
              </li>
            ))}
          {agents?.map((a) => (
            <li key={a.id}>
              <Link
                to={`/agents/${a.id}/chat`}
                aria-current={a.id === agentId ? "page" : undefined}
                className={cn(
                  "group flex items-center gap-2.5 rounded-lg px-2.5 py-1.5 text-sm transition-colors",
                  a.id === agentId ? "bg-brand-50 text-brand-800" : "text-slate-600 hover:bg-slate-100 hover:text-slate-900",
                )}
              >
                <AgentAvatar name={a.name} size="sm" />
                <span className="min-w-0 flex-1">
                  <span className="block truncate font-medium">{a.name}</span>
                  <span className="block truncate text-xs text-slate-400">{providerLabel(a.provider)} · {a.model}</span>
                </span>
              </Link>
            </li>
          ))}
          {agents?.length === 0 && <li className="px-2.5 py-2 text-xs text-slate-400">No agents yet</li>}
        </ul>
      </div>
    </nav>
  );
}

const AVATAR_COLORS = [
  "bg-indigo-100 text-indigo-700",
  "bg-emerald-100 text-emerald-700",
  "bg-amber-100 text-amber-800",
  "bg-sky-100 text-sky-700",
  "bg-rose-100 text-rose-700",
  "bg-violet-100 text-violet-700",
  "bg-teal-100 text-teal-700",
];

export function AgentAvatar({ name, size = "md" }: { name: string; size?: "sm" | "md" | "lg" }) {
  const hash = [...name].reduce((h, c) => (h * 31 + c.charCodeAt(0)) >>> 0, 7);
  const initials = name
    .split(/\s+/)
    .filter((w) => /^[\p{L}\p{N}]/u.test(w))
    .slice(0, 2)
    .map((w) => w[0]?.toUpperCase())
    .join("");
  return (
    <span
      aria-hidden
      className={cn(
        "flex shrink-0 items-center justify-center rounded-lg font-semibold",
        AVATAR_COLORS[hash % AVATAR_COLORS.length],
        { sm: "h-6 w-6 text-[10px]", md: "h-9 w-9 text-xs", lg: "h-11 w-11 text-sm" }[size],
      )}
    >
      {initials || "A"}
    </span>
  );
}

export function AppShell() {
  const [mobileOpen, setMobileOpen] = useState(false);
  return (
    <div className="flex h-screen overflow-hidden">
      <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:left-2 focus:top-2 focus:z-50 focus:rounded focus:bg-white focus:px-3 focus:py-2">
        Skip to content
      </a>
      <aside className="hidden w-64 shrink-0 border-r border-slate-200 bg-slate-50/80 lg:block">
        <Sidebar />
      </aside>

      {mobileOpen && (
        <div className="fixed inset-0 z-40 lg:hidden">
          <div className="absolute inset-0 bg-slate-900/30" onClick={() => setMobileOpen(false)} aria-hidden />
          <aside className="absolute inset-y-0 left-0 w-72 bg-slate-50 shadow-pop">
            <button
              className="absolute right-3 top-4 rounded-md p-1 text-slate-500 hover:bg-slate-100"
              onClick={() => setMobileOpen(false)}
              aria-label="Close navigation"
            >
              <X className="h-5 w-5" />
            </button>
            <Sidebar onNavigate={() => setMobileOpen(false)} />
          </aside>
        </div>
      )}

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-14 items-center gap-3 border-b border-slate-200 bg-white px-4 lg:hidden">
          <button
            onClick={() => setMobileOpen(true)}
            className="rounded-md p-1.5 text-slate-600 hover:bg-slate-100"
            aria-label="Open navigation"
          >
            <Menu className="h-5 w-5" />
          </button>
          <span className="font-semibold">Agentdesk</span>
        </header>
        <main id="main" className="min-h-0 flex-1 overflow-y-auto">
          <Outlet />
        </main>
      </div>
    </div>
  );
}

export function PageHeader({
  title,
  description,
  actions,
  leading,
}: {
  title: ReactNode;
  description?: ReactNode;
  actions?: ReactNode;
  leading?: ReactNode;
}) {
  return (
    <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
      <div className="flex min-w-0 items-center gap-3">
        {leading}
        <div className="min-w-0">
          <h1 className="truncate text-xl font-semibold tracking-tight text-slate-900">{title}</h1>
          {description && <p className="mt-0.5 text-sm text-slate-500">{description}</p>}
        </div>
      </div>
      {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
    </div>
  );
}
