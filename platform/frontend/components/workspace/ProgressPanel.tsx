"use client";

import { useMemo } from "react";

import { Package, X } from "lucide-react";

import { ArtifactModalButton } from "@/components/artifacts/ArtifactModalButton";
import { ArtifactRenderer } from "@/components/artifacts/ArtifactRenderer";
import { Button } from "@/components/ui/Button";
import { useMessages } from "@/hooks/useMessages";
import type { Artifact, UUID } from "@/lib/types";
import { cn } from "@/lib/utils";
import { useConversationStore } from "@/stores/conversationStore";

// Mirrors MessageBubble: these output types render in the generated-items rail as a
// clickable button that opens a modal preview, and are NOT shown inline in
// chat. Other output types (chart, report, presentation) stay inline here.
const MODAL_OUTPUT_TYPES: ReadonlySet<string> = new Set([
  "quiz",
  "podcast",
  "mindmap",
]);

interface Props {
  conversationId: UUID;
  className?: string;
  onClose?: () => void;
}

interface ArtifactGroup {
  messageId: string;
  createdAt: string;
  outputType: string | null;
  artifacts: Artifact[];
}

export function ProgressPanel({ conversationId, className, onClose }: Props) {
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
        "app-pane flex h-full min-h-0 flex-col overflow-hidden border-l border-border",
        className,
      )}
      aria-label="Generated items"
    >
      <header className="app-chrome flex h-11 items-center justify-between gap-2 border-b border-border px-4">
        <div className="flex min-w-0 items-center gap-2">
          <Package className="h-4 w-4 text-primary" />
          <h2 className="truncate text-sm font-semibold">Generated items</h2>
        </div>
        {onClose && (
          <Button
            type="button"
            variant="ghost"
            size="icon"
            className="h-8 w-8 lg:hidden"
            onClick={onClose}
            aria-label="Close generated items"
            title="Close"
          >
            <X className="h-4 w-4" />
          </Button>
        )}
      </header>

      <div className="flex-1 overflow-y-auto px-4 py-4">
        {groups.length === 0 ? (
          <p className="app-chrome text-xs leading-5 text-muted-foreground">
            Generated items will appear here.
          </p>
        ) : (
          <div className="flex flex-col gap-3">
            {groups.map((group) =>
              MODAL_OUTPUT_TYPES.has(group.outputType ?? "") ? (
                <ArtifactModalButton
                  key={group.messageId}
                  outputType={group.outputType ?? ""}
                  artifacts={group.artifacts}
                  conversationId={conversationId}
                  createdAt={group.createdAt}
                />
              ) : (
                <div
                  key={group.messageId}
                  className="flex flex-col gap-2 rounded-md border border-border bg-surface p-2"
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
              ),
            )}
          </div>
        )}
      </div>
    </aside>
  );
}
