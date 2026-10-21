import { useNavigate } from "react-router-dom";
import { useCreateAgent } from "@/lib/queries";
import { AgentForm } from "@/components/agents/AgentForm";
import { PageHeader } from "@/components/layout/AppShell";
import { Card } from "@/components/ui/primitives";
import { useToast } from "@/components/ui/Toast";

export function NewAgentPage() {
  const navigate = useNavigate();
  const create = useCreateAgent();
  const toast = useToast();

  return (
    <div className="mx-auto max-w-3xl space-y-6 px-4 py-8 sm:px-8">
      <PageHeader title="Create agent" description="Next you'll upload documents to its knowledge base." />
      <Card className="p-6">
        <AgentForm
          submitLabel="Create agent"
          submitting={create.isPending}
          onCancel={() => navigate(-1)}
          onSubmit={(data) =>
            create.mutate(data, {
              onSuccess: (agent) => {
                toast.success(`${agent.name} created`, "Upload documents to build its knowledge base.");
                navigate(`/agents/${agent.id}/knowledge`);
              },
              onError: (e) => toast.error("Couldn't create agent", (e as Error).message),
            })
          }
        />
      </Card>
    </div>
  );
}
