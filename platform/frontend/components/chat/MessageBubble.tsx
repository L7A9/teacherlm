"use client";

import { useState } from "react";

import { BookOpen, ChevronDown, ChevronRight, GraduationCap, User } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { ArtifactRenderer } from "@/components/artifacts/ArtifactRenderer";
import { Badge } from "@/components/ui/Badge";
import type { Message, UUID } from "@/lib/types";
import { cn } from "@/lib/utils";

interface Props {
  message: Message;
  conversationId: UUID;
  streaming?: boolean;
}

export function MessageBubble({ message, conversationId, streaming }: Props) {
  const isUser = message.role === "user";
  const hasArtifacts = message.artifacts.length > 0;
  const hasSources = message.sources.length > 0;

  return (
    <div
      className={cn(
        "flex gap-3",
        isUser ? "flex-row-reverse" : "flex-row",
      )}
    >
      <Avatar role={message.role} />

      <div
        className={cn(
          "flex max-w-[85%] flex-col gap-2",
          isUser ? "items-end" : "items-start",
        )}
      >
        <div
          className={cn(
            "rounded-2xl px-4 py-2.5 text-sm leading-relaxed",
            isUser
              ? "bg-primary text-primary-foreground"
              : "bg-surface text-surface-foreground border border-border",
          )}
        >
          {message.content ? (
            <div className="prose prose-invert max-w-none prose-p:my-1.5 prose-pre:my-2 prose-headings:my-2 prose-ul:my-1.5 prose-ol:my-1.5 prose-sm">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {message.content}
              </ReactMarkdown>
            </div>
          ) : streaming ? (
            <TypingIndicator />
          ) : (
            <span className="text-xs text-muted-foreground italic">
              (no response)
            </span>
          )}
          {streaming && message.content && (
            <span className="ml-0.5 inline-block h-3 w-1 align-baseline bg-current animate-pulse" />
          )}
        </div>

        {hasArtifacts && (
          <div className="flex w-full flex-col gap-3">
            {message.artifacts.map((a, idx) => (
              <ArtifactRenderer
                key={`${a.url}-${idx}`}
                artifact={a}
                siblings={message.artifacts.filter((_, i) => i !== idx)}
                conversationId={conversationId}
              />
            ))}
          </div>
        )}

        {hasSources && <Sources message={message} />}
      </div>
    </div>
  );
}

function Avatar({ role }: { role: Message["role"] }) {
  const isUser = role === "user";
  return (
    <div
      className={cn(
        "flex h-8 w-8 shrink-0 items-center justify-center rounded-full",
        isUser
          ? "bg-muted text-muted-foreground"
          : "bg-primary/15 text-primary",
      )}
      aria-hidden
    >
      {isUser ? (
        <User className="h-4 w-4" />
      ) : (
        <GraduationCap className="h-4 w-4" />
      )}
    </div>
  );
}

function TypingIndicator() {
  return (
    <span className="inline-flex items-center gap-1">
      <Dot delay={0} />
      <Dot delay={150} />
      <Dot delay={300} />
    </span>
  );
}

function Dot({ delay }: { delay: number }) {
  return (
    <span
      className="h-1.5 w-1.5 rounded-full bg-current opacity-70 animate-pulse"
      style={{ animationDelay: `${delay}ms` }}
    />
  );
}

function Sources({ message }: { message: Message }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="w-full">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="inline-flex items-center gap-1.5 text-[11px] text-muted-foreground hover:text-foreground"
      >
        {open ? (
          <ChevronDown className="h-3 w-3" />
        ) : (
          <ChevronRight className="h-3 w-3" />
        )}
        <BookOpen className="h-3 w-3" />
        {message.sources.length} source
        {message.sources.length === 1 ? "" : "s"}
      </button>

      {open && (
        <ol className="mt-1.5 flex flex-col gap-1.5">
          {message.sources.map((s, idx) => (
            <li
              key={`${s.chunk_id ?? idx}-${idx}`}
              className="rounded-md border border-border bg-surface p-2 text-[11px]"
            >
              <div className="mb-1 flex items-center justify-between gap-2">
                <span className="font-medium truncate" title={s.source}>
                  {s.source}
                </span>
                <Badge variant="muted">score {s.score.toFixed(2)}</Badge>
              </div>
              <p className="text-muted-foreground line-clamp-4 whitespace-pre-wrap">
                {s.text}
              </p>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}
