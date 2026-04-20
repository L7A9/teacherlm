import type { Metadata } from "next";

import { Workspace } from "@/components/workspace/Workspace";

export const metadata: Metadata = {
  title: "Workspace · TeacherLM",
};

interface PageProps {
  params: { id: string };
}

export default function ConversationPage({ params }: PageProps) {
  return <Workspace conversationId={params.id} />;
}
