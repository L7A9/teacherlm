"use client";

import { useRef } from "react";

import CloseRoundedIcon from "@mui/icons-material/CloseRounded";
import { IconButton, Tooltip } from "@mui/material";
import { Sparkles } from "lucide-react";

import type { ChatInputHandle } from "@/components/chat/ChatInput";
import { ChatInput } from "@/components/chat/ChatInput";
import { MessageList } from "@/components/chat/MessageList";
import { OutputTypeButtons } from "@/components/chat/OutputTypeButtons";
import { useLearnerState } from "@/hooks/useConversations";
import { useFiles } from "@/hooks/useFiles";
import { useSourceFileSelection } from "@/hooks/useSourceFileSelection";
import type { UploadedFile, UUID } from "@/lib/types";
import { cn } from "@/lib/utils";
import { useProgressStore } from "@/stores/progressStore";

interface Props {
  conversationId: UUID;
  className?: string;
  onClose?: () => void;
}

const EMPTY_UPLOADED_FILES: UploadedFile[] = [];

export function ChatPanel({ conversationId, className, onClose }: Props) {
  const inputRef = useRef<ChatInputHandle>(null);
  useLearnerState(conversationId);
  const { data: files, isLoading: filesLoading } = useFiles(conversationId);
  const learner = useProgressStore((s) => s.stateByConversation[conversationId]);

  const hint = buildHint(learner);
  const fileItems = files?.items ?? EMPTY_UPLOADED_FILES;
  const { readyFiles, selectedSourceFileIds } = useSourceFileSelection(
    conversationId,
    fileItems,
  );
  const hasCourseFiles = readyFiles.length > 0;
  const hasSelectedSourceFiles = selectedSourceFileIds.length > 0;
  const actionsDisabled = filesLoading || !hasCourseFiles || !hasSelectedSourceFiles;
  const noFilesReason = filesLoading
    ? "Checking uploaded files..."
    : hasCourseFiles
      ? "Select at least one source file."
      : "Wait until at least one course file is ready.";

  return (
    <section
      className={cn(
        "flex h-full min-h-0 min-w-0 flex-col overflow-hidden bg-background",
        className,
      )}
      aria-label="Chat"
    >
      <header className="app-chrome flex h-11 items-center justify-between gap-3 border-b border-border px-4">
        <div className="min-w-0 flex-1">
          {hint && (
            <p className="flex items-center gap-1.5 truncate text-[11px] text-muted-foreground">
              <Sparkles className="h-3 w-3 text-primary" />
              {hint}
            </p>
          )}
        </div>
        {onClose && (
          <Tooltip title="Close chat">
            <IconButton
              aria-label="Close chat"
              onClick={onClose}
              size="small"
              sx={{
                color: "hsl(var(--muted-foreground))",
                "&:hover": {
                  bgcolor: "hsl(var(--muted))",
                  color: "hsl(var(--foreground))",
                },
              }}
            >
              <CloseRoundedIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        )}
      </header>

      <MessageList
        conversationId={conversationId}
        className="flex-1 min-h-0"
      />

      <footer className="app-pane flex flex-col gap-2 border-t border-border px-3 py-3 sm:px-4">
        <OutputTypeButtons
          onSelectChat={() => inputRef.current?.focus()}
          className="app-chrome"
          disabled={actionsDisabled}
          disabledReason={noFilesReason}
        />
        <ChatInput
          ref={inputRef}
          conversationId={conversationId}
          disabled={actionsDisabled}
          disabledReason={noFilesReason}
        />
      </footer>
    </section>
  );
}

function buildHint(
  learner: ReturnType<typeof useProgressStore.getState>["stateByConversation"][string] | undefined,
): string | null {
  if (!learner) return null;
  const strong = learner.understood_concepts[0];
  const weak = learner.struggling_concepts[0];
  if (strong && weak) return `You're strong on ${strong} — let's review ${weak}.`;
  if (strong) return `Great progress on ${strong}.`;
  if (weak) return `Let's revisit ${weak}.`;
  return null;
}
