"use client";

import { useMemo } from "react";

import { Package, Sparkles } from "lucide-react";

import { ArtifactRenderer } from "@/components/artifacts/ArtifactRenderer";
import { LearnerProgress } from "@/components/progress/LearnerProgress";
import { useMessages } from "@/hooks/useMessages";
import type { Artifact, UUID } from "@/lib/types";
import { cn } from "@/lib/utils";
import { useConversationStore } from "@/stores/conversationStore";

interface Props {
  conversationId: UUID;
  className?: string;
}

interface ArtifactGroup {
  messageId: string;
  createdAt: string;
  outputType: string | null;
  artifacts: Artifact[];
}

export function ProgressPanel({ conversationId, className }: Props) {
  const { data } = useMessages(conversationId);
  const streaming = useConversationStore(
    (s) => s.streamingByConversation[conversationId] ?? null,
  );

  const groups = useMemo<ArtifactGroup[]>(() => {
    const out: ArtifactGroup[] = [];
    for (const m of data?.items ?? []) {
      if (m.role !== "assistant" || m.artifacts.length === 0) continue;
      out.push({
        messageId: m.id,
        createdAt: m.created_at,
        outputType: m.output_type ?? null,
        artifacts: m.artifacts,
      });
    }
    if (streaming && streaming.artifacts.length > 0) {
      out.push({
        messageId: `stream-${conversationId}`,
        createdAt: new Date(streaming.startedAt).toISOString(),
        outputType: streaming.outputType ?? null,
        artifacts: streaming.artifacts,
      });
    }
    return out.reverse();
  }, [conversationId, data, streaming]);

  return (
    <aside
      className={cn(
        "flex h-full min-h-0 flex-col overflow-hidden border-l border-border bg-background",
        className,
      )}
      aria-label="Progress"
    >
      <header className="flex items-center gap-2 border-b border-border px-4 py-3">
        <Sparkles className="h-4 w-4 text-primary" />
        <h2 className="text-sm font-semibold">Progress</h2>
      </header>

      <div className="flex-1 overflow-y-auto">
        <section className="border-b border-border px-4 py-4">
          <LearnerProgress conversationId={conversationId} />
        </section>

        <section className="flex flex-col gap-3 px-4 py-4">
          <div className="flex items-center gap-1.5 text-[11px] uppercase tracking-wide text-muted-foreground">
            <Package className="h-3 w-3" />
            Generated items
          </div>

          {groups.length === 0 ? (
            <p className="text-xs text-muted-foreground">
              Quizzes, flashcards, diagrams, and other generated items will
              appear here once you create them.
            </p>
          ) : (
            <div className="flex flex-col gap-3">
              {groups.map((group) => (
                <div
                  key={group.messageId}
                  className="flex flex-col gap-2 rounded-lg border border-border bg-surface p-2"
                >
                  {group.outputType && (
                    <div className="px-1 text-[11px] uppercase tracking-wide text-muted-foreground">
                      {group.outputType}
                    </div>
                  )}
                  <div className="flex flex-col gap-2">
                    {group.artifacts.map((a, idx) => (
                      <ArtifactRenderer
                        key={`${a.url}-${idx}`}
                        artifact={a}
                        siblings={group.artifacts.filter((_, i) => i !== idx)}
                        conversationId={conversationId}
                      />
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>
      </div>
    </aside>
  );
}
