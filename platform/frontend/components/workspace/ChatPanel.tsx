"use client";

import { useRef } from "react";

import { Sparkles } from "lucide-react";

import type { ChatInputHandle } from "@/components/chat/ChatInput";
import { ChatInput } from "@/components/chat/ChatInput";
import { MessageList } from "@/components/chat/MessageList";
import { OutputTypeButtons } from "@/components/chat/OutputTypeButtons";
import { useConversation } from "@/hooks/useConversations";
import type { UUID } from "@/lib/types";
import { cn } from "@/lib/utils";
import { useProgressStore } from "@/stores/progressStore";

interface Props {
  conversationId: UUID;
  className?: string;
}

export function ChatPanel({ conversationId, className }: Props) {
  const inputRef = useRef<ChatInputHandle>(null);
  const { data: conversation } = useConversation(conversationId);
  const learner = useProgressStore((s) => s.stateByConversation[conversationId]);

  const hint = buildHint(learner);

  return (
    <section
      className={cn("flex h-full min-w-0 flex-col bg-background", className)}
      aria-label="Chat"
    >
      <header className="flex items-center justify-between gap-3 border-b border-border px-4 py-3">
        <div className="min-w-0">
          <h1 className="truncate text-sm font-semibold">
            {conversation?.title ?? "Your teacher"}
          </h1>
          {hint && (
            <p className="mt-0.5 flex items-center gap-1.5 truncate text-[11px] text-muted-foreground">
              <Sparkles className="h-3 w-3 text-primary" />
              {hint}
            </p>
          )}
        </div>
      </header>

      <MessageList
        conversationId={conversationId}
        className="flex-1 min-h-0"
      />

      <footer className="flex flex-col gap-2 border-t border-border bg-background px-4 py-3">
        <OutputTypeButtons onSelectChat={() => inputRef.current?.focus()} />
        <ChatInput ref={inputRef} conversationId={conversationId} />
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
