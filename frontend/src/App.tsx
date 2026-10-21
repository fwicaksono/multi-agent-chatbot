import { Link, Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "@/components/layout/AppShell";
import { AgentsPage } from "@/pages/AgentsPage";
import { NewAgentPage } from "@/pages/NewAgentPage";
import { AgentWorkspace } from "@/pages/AgentWorkspace";
import { ChatTab } from "@/pages/ChatTab";
import { KnowledgeTab } from "@/pages/KnowledgeTab";
import { SettingsTab } from "@/pages/SettingsTab";
import { AgentUsageTab, UsagePage } from "@/pages/UsagePages";

function NotFound() {
  return (
    <div className="flex h-full flex-col items-center justify-center p-8 text-center">
      <p className="text-sm font-semibold text-brand-600">404</p>
      <h1 className="mt-1 text-xl font-semibold">Page not found</h1>
      <Link to="/agents" className="mt-4 text-sm font-medium text-brand-700 hover:underline">
        Back to agents
      </Link>
    </div>
  );
}

export default function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route index element={<Navigate to="/agents" replace />} />
        <Route path="agents" element={<AgentsPage />} />
        <Route path="agents/new" element={<NewAgentPage />} />
        <Route path="agents/:agentId" element={<AgentWorkspace />}>
          <Route index element={<Navigate to="chat" replace />} />
          <Route path="chat" element={<ChatTab />} />
          <Route path="knowledge" element={<KnowledgeTab />} />
          <Route path="settings" element={<SettingsTab />} />
          <Route path="usage" element={<AgentUsageTab />} />
        </Route>
        <Route path="usage" element={<UsagePage />} />
        <Route path="*" element={<NotFound />} />
      </Route>
    </Routes>
  );
}
