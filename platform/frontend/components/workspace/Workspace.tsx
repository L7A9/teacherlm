"use client";

import { useEffect } from "react";

import Link from "next/link";

import { GraduationCap, PanelLeft, PanelRight, Settings } from "lucide-react";

import { GeneratorDialog } from "@/components/chat/GeneratorDialog";
import { Button } from "@/components/ui/Button";
import { ChatPanel } from "@/components/workspace/ChatPanel";
import { ProgressPanel } from "@/components/workspace/ProgressPanel";
import { SourcesPanel } from "@/components/workspace/SourcesPanel";
import type { UUID } from "@/lib/types";
import { cn } from "@/lib/utils";
import { useConversationStore } from "@/stores/conversationStore";
import { useUiStore } from "@/stores/uiStore";

interface Props {
  conversationId: UUID;
}

export function Workspace({ conversationId }: Props) {
  const setActive = useConversationStore((s) => s.setActive);
  const sourcesCollapsed = useUiStore((s) => s.sourcesCollapsed);
  const progressCollapsed = useUiStore((s) => s.progressCollapsed);
  const toggleSources = useUiStore((s) => s.toggleSources);
  const toggleProgress = useUiStore((s) => s.toggleProgress);

  useEffect(() => {
    setActive(conversationId);
    return () => setActive(null);
  }, [conversationId, setActive]);

  return (
    <div className="flex h-dvh flex-col bg-background text-foreground">
      <TopBar
        sourcesCollapsed={sourcesCollapsed}
        progressCollapsed={progressCollapsed}
        onToggleSources={toggleSources}
        onToggleProgress={toggleProgress}
      />

      <div
        className={cn("grid min-h-0 flex-1 overflow-hidden")}
        style={{
          gridTemplateColumns: gridTemplateColumns(
            sourcesCollapsed,
            progressCollapsed,
          ),
          gridTemplateRows: "minmax(0, 1fr)",
        }}
      >
        {!sourcesCollapsed && (
          <SourcesPanel conversationId={conversationId} />
        )}
        <ChatPanel conversationId={conversationId} />
        {!progressCollapsed && (
          <ProgressPanel conversationId={conversationId} />
        )}
      </div>

      <GeneratorDialog />
    </div>
  );
}

function gridTemplateColumns(
  sourcesCollapsed: boolean,
  progressCollapsed: boolean,
): string {
  const cols: string[] = [];
  if (!sourcesCollapsed) cols.push("minmax(240px, 300px)");
  cols.push("minmax(0, 1fr)");
  if (!progressCollapsed) cols.push("minmax(280px, 360px)");
  return cols.join(" ");
}

interface TopBarProps {
  sourcesCollapsed: boolean;
  progressCollapsed: boolean;
  onToggleSources: () => void;
  onToggleProgress: () => void;
}

function TopBar({
  sourcesCollapsed,
  progressCollapsed,
  onToggleSources,
  onToggleProgress,
}: TopBarProps) {
  return (
    <header className="flex items-center justify-between border-b border-border bg-background px-4 py-2">
      <div className="flex items-center gap-2">
        <Button
          variant="ghost"
          size="icon"
          onClick={onToggleSources}
          aria-label={sourcesCollapsed ? "Show sources" : "Hide sources"}
          title={sourcesCollapsed ? "Show sources" : "Hide sources"}
        >
          <PanelLeft className="h-4 w-4" />
        </Button>
        <div className="flex items-center gap-2">
          <div className="flex h-7 w-7 items-center justify-center rounded-md bg-primary/15 text-primary">
            <GraduationCap className="h-4 w-4" />
          </div>
          <span className="text-sm font-semibold">TeacherLM</span>
        </div>
      </div>

      <div className="flex items-center gap-1">
        <Button variant="ghost" size="icon" asChild title="Settings">
          <Link href="/settings" aria-label="Settings">
            <Settings className="h-4 w-4" />
          </Link>
        </Button>
        <Button
          variant="ghost"
          size="icon"
          onClick={onToggleProgress}
          aria-label={progressCollapsed ? "Show progress" : "Hide progress"}
          title={progressCollapsed ? "Show progress" : "Hide progress"}
        >
          <PanelRight className="h-4 w-4" />
        </Button>
      </div>
    </header>
  );
}
