import { useState } from "react";
import { useNavigate, useOutletContext } from "react-router-dom";
import { Trash2 } from "lucide-react";
import { useDeleteAgent, useUpdateAgent } from "@/lib/queries";
import { AgentForm } from "@/components/agents/AgentForm";
import { Button, Card, Field, Input } from "@/components/ui/primitives";
import { ConfirmDialog } from "@/components/ui/Dialog";
import { useToast } from "@/components/ui/Toast";
import type { WorkspaceContext } from "./AgentWorkspace";

export function SettingsTab() {
  const { agent } = useOutletContext<WorkspaceContext>();
  const update = useUpdateAgent(agent.id);
  const remove = useDeleteAgent();
  const toast = useToast();
  const navigate = useNavigate();
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [confirmText, setConfirmText] = useState("");

  return (
    <div className="mx-auto max-w-3xl space-y-6 px-4 py-6 sm:px-6">
      <Card className="p-6">
        <AgentForm
          key={agent.updated_at}
          initial={agent}
          submitLabel="Save changes"
          submitting={update.isPending}
          onSubmit={(data) =>
            update.mutate(data, {
              onSuccess: () => toast.success("Agent updated", "Changes apply to the next message."),
              onError: (e) => toast.error("Couldn't save changes", (e as Error).message),
            })
          }
        />
      </Card>

      <Card className="border-rose-200 p-6">
        <h2 className="text-sm font-semibold text-rose-700">Delete agent</h2>
        <p className="mt-1 text-sm text-slate-600">
          Permanently deletes the agent, its {agent.document_counts.total} document(s), all embeddings and indexed data,
          conversations and usage history. This cannot be undone.
        </p>
        <Button variant="danger" className="mt-4" icon={<Trash2 className="h-4 w-4" />} onClick={() => setConfirmOpen(true)}>
          Delete agent
        </Button>
      </Card>

      <ConfirmDialog
        open={confirmOpen}
        onClose={() => {
          setConfirmOpen(false);
          setConfirmText("");
        }}
        title={`Delete ${agent.name}?`}
        description="The knowledge base, embeddings, conversations and usage records are deleted permanently."
        confirmLabel="Delete permanently"
        loading={remove.isPending}
        confirmDisabled={confirmText !== agent.name}
        onConfirm={() => {
          if (confirmText !== agent.name) return;
          remove.mutate(agent.id, {
            onSuccess: () => {
              toast.success(`${agent.name} deleted`);
              navigate("/agents");
            },
            onError: (e) => toast.error("Couldn't delete agent", (e as Error).message),
          });
        }}
      >
        <Field label={`Type “${agent.name}” to confirm`} htmlFor="confirm-name">
          <Input id="confirm-name" value={confirmText} onChange={(e) => setConfirmText(e.target.value)} autoComplete="off" />
        </Field>
      </ConfirmDialog>
    </div>
  );
}
