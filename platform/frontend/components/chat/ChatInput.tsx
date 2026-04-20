"use client";

import {
  forwardRef,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from "react";

import { Send, Square } from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/Button";
import { Textarea } from "@/components/ui/Input";
import { useChatStream } from "@/hooks/useChatStream";
import type { UUID } from "@/lib/types";
import { cn } from "@/lib/utils";
import { useConversationStore } from "@/stores/conversationStore";

export interface ChatInputHandle {
  focus: () => void;
}

interface Props {
  conversationId: UUID;
  className?: string;
  placeholder?: string;
}

const MAX_HEIGHT_PX = 220;
const MIN_HEIGHT_PX = 44;

export const ChatInput = forwardRef<ChatInputHandle, Props>(function ChatInput(
  { conversationId, className, placeholder = "Ask your teacher anything…" },
  ref,
) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const runChat = useChatStream();

  const isStreaming = useConversationStore((s) =>
    Boolean(s.streamingByConversation[conversationId]),
  );
  const abortStream = useConversationStore((s) => s.abortStream);

  useImperativeHandle(ref, () => ({
    focus: () => textareaRef.current?.focus(),
  }));

  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = `${MIN_HEIGHT_PX}px`;
    const next = Math.min(el.scrollHeight, MAX_HEIGHT_PX);
    el.style.height = `${Math.max(MIN_HEIGHT_PX, next)}px`;
  }, [value]);

  const submit = async () => {
    const trimmed = value.trim();
    if (!trimmed || isStreaming) return;
    setValue("");
    try {
      await runChat(conversationId, { user_message: trimmed, options: {} });
    } catch (err) {
      toast.error(`Chat failed: ${(err as Error).message}`);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      void submit();
    }
  };

  const canSubmit = value.trim().length > 0 && !isStreaming;

  return (
    <div
      className={cn(
        "flex items-end gap-2 rounded-2xl border border-border bg-surface p-2",
        "focus-within:border-primary/60 focus-within:ring-1 focus-within:ring-primary/40",
        className,
      )}
    >
      <Textarea
        ref={textareaRef}
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={placeholder}
        rows={1}
        className={cn(
          "min-h-[44px] resize-none border-0 bg-transparent px-2 py-2",
          "focus-visible:ring-0 focus-visible:ring-offset-0",
        )}
        aria-label="Chat message"
      />

      {isStreaming ? (
        <Button
          type="button"
          variant="secondary"
          size="icon"
          onClick={() => abortStream(conversationId)}
          aria-label="Stop generation"
          title="Stop generation"
        >
          <Square className="h-4 w-4 fill-current" />
        </Button>
      ) : (
        <Button
          type="button"
          variant="primary"
          size="icon"
          onClick={() => void submit()}
          disabled={!canSubmit}
          aria-label="Send message"
          title="Send (Enter)"
        >
          <Send className="h-4 w-4" />
        </Button>
      )}
    </div>
  );
});
