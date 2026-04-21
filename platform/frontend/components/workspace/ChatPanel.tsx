"use client";

import { useEffect, useRef, useState } from "react";

import { Sparkles } from "lucide-react";

import type { ChatInputHandle } from "@/components/chat/ChatInput";
import { ChatInput } from "@/components/chat/ChatInput";
import { MessageList } from "@/components/chat/MessageList";
import { OutputTypeButtons } from "@/components/chat/OutputTypeButtons";
import {
  useConversation,
  useUpdateConversation,
} from "@/hooks/useConversations";
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
      className={cn(
        "flex h-full min-h-0 min-w-0 flex-col overflow-hidden bg-background",
        className,
      )}
      aria-label="Chat"
    >
      <header className="flex items-center justify-between gap-3 border-b border-border px-4 py-3">
        <div className="min-w-0">
          <EditableTitle
            conversationId={conversationId}
            title={conversation?.title ?? ""}
          />
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

interface EditableTitleProps {
  conversationId: UUID;
  title: string;
}

function EditableTitle({ conversationId, title }: EditableTitleProps) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(title);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const { mutate, isPending } = useUpdateConversation(conversationId);

  useEffect(() => {
    if (!editing) setDraft(title);
  }, [title, editing]);

  useEffect(() => {
    if (editing) {
      inputRef.current?.focus();
      inputRef.current?.select();
    }
  }, [editing]);

  const commit = () => {
    const next = draft.trim();
    if (!next || next === title) {
      setDraft(title);
      setEditing(false);
      return;
    }
    mutate(
      { title: next },
      {
        onSuccess: () => setEditing(false),
        onError: () => {
          setDraft(title);
          setEditing(false);
        },
      },
    );
  };

  if (editing) {
    return (
      <input
        ref={inputRef}
        value={draft}
        disabled={isPending}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            e.preventDefault();
            commit();
          } else if (e.key === "Escape") {
            e.preventDefault();
            setDraft(title);
            setEditing(false);
          }
        }}
        className="w-full truncate rounded-sm bg-muted px-1.5 py-0.5 text-sm font-semibold outline-none ring-1 ring-ring"
        aria-label="Conversation title"
      />
    );
  }

  return (
    <button
      type="button"
      onClick={() => setEditing(true)}
      title="Click to rename"
      className="block w-full truncate rounded-sm px-1.5 py-0.5 text-left text-sm font-semibold hover:bg-muted"
    >
      {title || "Your teacher"}
    </button>
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
