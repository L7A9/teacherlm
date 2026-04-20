"use client";

import { useEffect, useMemo, useRef } from "react";

import { AlertCircle, Loader2, MessageSquare } from "lucide-react";

import { MessageBubble } from "@/components/chat/MessageBubble";
import { useMessages } from "@/hooks/useMessages";
import type { Message, UUID } from "@/lib/types";
import { cn } from "@/lib/utils";
import { useConversationStore } from "@/stores/conversationStore";

interface Props {
  conversationId: UUID;
  className?: string;
}

export function MessageList({ conversationId, className }: Props) {
  const { data, isLoading, error } = useMessages(conversationId);
  const streaming = useConversationStore(
    (s) => s.streamingByConversation[conversationId] ?? null,
  );
  const streamError = useConversationStore(
    (s) => s.streamErrorByConversation[conversationId] ?? null,
  );
  const asTempMessage = useConversationStore((s) => s.asTempMessage);

  const scrollRef = useRef<HTMLDivElement | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  const messages = useMemo<Message[]>(() => {
    const persisted = data?.items ?? [];
    const temp = streaming ? asTempMessage(conversationId) : null;
    if (!temp) return persisted;
    // Avoid duplicating the assistant turn once the server persists it and the
    // query cache refetches before `endStream` fires.
    const last = persisted.at(-1);
    if (last?.role === "assistant" && streaming && last.content === streaming.text) {
      return persisted;
    }
    return [...persisted, temp];
  }, [asTempMessage, conversationId, data, streaming]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages.length, streaming?.text.length]);

  if (isLoading) {
    return (
      <div className={cn("flex h-full items-center justify-center gap-2 text-sm text-muted-foreground", className)}>
        <Loader2 className="h-4 w-4 animate-spin" />
        Loading messages…
      </div>
    );
  }

  if (error) {
    return (
      <div className={cn("flex h-full flex-col items-center justify-center gap-2 text-sm text-[hsl(var(--danger))]", className)}>
        <AlertCircle className="h-5 w-5" />
        {(error as Error).message}
      </div>
    );
  }

  if (messages.length === 0) {
    return (
      <div className={cn("flex h-full flex-col items-center justify-center gap-3 px-6 text-center", className)}>
        <div className="flex h-12 w-12 items-center justify-center rounded-full bg-primary/15 text-primary">
          <MessageSquare className="h-5 w-5" />
        </div>
        <div className="text-sm font-medium">Start learning</div>
        <p className="max-w-sm text-xs text-muted-foreground">
          Upload a file and ask your teacher a question, or click one of the
          buttons above the input to generate a quiz, report, or more.
        </p>
      </div>
    );
  }

  return (
    <div
      ref={scrollRef}
      className={cn("flex flex-col gap-4 overflow-y-auto px-4 py-5", className)}
    >
      {messages.map((message) => (
        <MessageBubble
          key={message.id}
          message={message}
          conversationId={conversationId}
          streaming={message.id === `stream-${conversationId}`}
        />
      ))}

      {streamError && (
        <div className="flex items-center gap-2 rounded-md border border-[hsl(var(--danger)/0.4)] bg-[hsl(var(--danger)/0.08)] px-3 py-2 text-xs text-[hsl(var(--danger))]">
          <AlertCircle className="h-3.5 w-3.5" />
          {streamError}
        </div>
      )}

      <div ref={bottomRef} aria-hidden />
    </div>
  );
}
