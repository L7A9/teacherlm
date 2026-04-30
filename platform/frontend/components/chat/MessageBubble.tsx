"use client";

import { useState } from "react";

import {
  BookOpen,
  Check,
  ChevronDown,
  ChevronRight,
  Copy,
  GraduationCap,
  User,
} from "lucide-react";
import ReactMarkdown, { type Components } from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/cjs/styles/prism";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";

import "katex/dist/katex.min.css";

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
            "rounded-2xl px-4 py-3 text-sm",
            isUser
              ? "bg-primary text-primary-foreground leading-7"
              : "border border-slate-800 bg-slate-900 text-slate-200",
          )}
        >
          {message.content ? (
            isUser ? (
              <p className="whitespace-pre-wrap leading-7">{message.content}</p>
            ) : (
              <AssistantMarkdown content={message.content} />
            )
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

const markdownComponents: Components = {
  // Headings
  h1: ({ node, ...props }) => (
    <h1 className="mt-6 mb-4 text-2xl font-bold text-slate-100" {...props} />
  ),
  h2: ({ node, ...props }) => (
    <h2
      className="mt-5 mb-3 border-b border-slate-700 pb-1.5 text-xl font-semibold text-slate-100"
      {...props}
    />
  ),
  h3: ({ node, ...props }) => (
    <h3 className="mt-4 mb-2 text-lg font-semibold text-slate-100" {...props} />
  ),
  h4: ({ node, ...props }) => (
    <h4 className="mt-3 mb-1.5 text-base font-semibold text-slate-200" {...props} />
  ),
  h5: ({ node, ...props }) => (
    <h5
      className="mt-3 mb-1 text-sm font-semibold uppercase tracking-wide text-slate-300"
      {...props}
    />
  ),
  h6: ({ node, ...props }) => (
    <h6
      className="mt-3 mb-1 text-xs font-semibold uppercase tracking-wide text-slate-400"
      {...props}
    />
  ),

  // Paragraphs
  p: ({ node, ...props }) => (
    <p className="mb-4 leading-7 text-slate-300 last:mb-0" {...props} />
  ),

  // Lists
  ul: ({ node, ...props }) => (
    <ul
      className="mb-4 ml-6 list-disc space-y-2 text-slate-300 marker:text-slate-500 last:mb-0"
      {...props}
    />
  ),
  ol: ({ node, ...props }) => (
    <ol
      className="mb-4 ml-6 list-decimal space-y-2 text-slate-300 marker:text-slate-500 last:mb-0"
      {...props}
    />
  ),
  li: ({ node, ...props }) => <li className="leading-7" {...props} />,

  // Code (inline + fenced)
  code: ({ node, className, children, ...props }) => {
    const match = /language-(\w+)/.exec(className ?? "");
    const codeText = String(children).replace(/\n$/, "");
    const lang = match?.[1];
    if (lang) {
      return <CodeBlock language={lang} code={codeText} />;
    }
    return (
      <code
        className="rounded bg-slate-800 px-1.5 py-0.5 font-mono text-[0.85em] text-pink-400"
        {...props}
      >
        {children}
      </code>
    );
  },
  // `pre` wraps fenced code blocks; CodeBlock renders its own container,
  // so we make `pre` a passthrough to avoid a double wrapper.
  pre: ({ children }) => <>{children}</>,

  // Blockquotes
  blockquote: ({ node, ...props }) => (
    <blockquote
      className="my-4 border-l-4 border-teal-500 bg-slate-800/50 px-4 py-2 not-italic text-slate-300"
      {...props}
    />
  ),

  // Tables
  table: ({ node, ...props }) => (
    <div className="my-4 overflow-x-auto">
      <table
        className="min-w-full border-collapse border border-slate-700 text-left"
        {...props}
      />
    </div>
  ),
  thead: ({ node, ...props }) => <thead className="bg-slate-800" {...props} />,
  th: ({ node, ...props }) => (
    <th
      className="border border-slate-700 px-4 py-2 text-left font-semibold text-slate-200"
      {...props}
    />
  ),
  td: ({ node, ...props }) => (
    <td
      className="border border-slate-700 px-4 py-2 align-top text-slate-300"
      {...props}
    />
  ),
  tr: ({ node, ...props }) => <tr className="even:bg-slate-800/30" {...props} />,

  // Links
  a: ({ node, ...props }) => (
    <a
      className="text-teal-400 underline-offset-2 hover:text-teal-300 hover:underline"
      target="_blank"
      rel="noreferrer"
      {...props}
    />
  ),

  // Horizontal rule
  hr: ({ node, ...props }) => <hr className="my-6 border-slate-700" {...props} />,

  // Strong / em
  strong: ({ node, ...props }) => (
    <strong className="font-semibold text-slate-100" {...props} />
  ),
  em: ({ node, ...props }) => <em className="italic text-slate-300" {...props} />,
};

function AssistantMarkdown({ content }: { content: string }) {
  return (
    <div className="prose prose-invert prose-slate max-w-none">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
        components={markdownComponents}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}

function CodeBlock({ language, code }: { language: string; code: string }) {
  const [copied, setCopied] = useState(false);

  const onCopy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // clipboard may be unavailable (e.g. insecure context); silently no-op.
    }
  };

  return (
    <div className="group relative my-4 overflow-hidden rounded-lg border border-slate-800 bg-slate-950">
      <div className="flex items-center justify-between border-b border-slate-800 bg-slate-900/80 px-3 py-1.5 text-[11px] text-slate-400">
        <span className="font-mono uppercase tracking-wide">{language}</span>
        <button
          type="button"
          onClick={onCopy}
          className="inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-slate-400 transition-colors hover:bg-slate-800 hover:text-slate-200"
          aria-label={copied ? "Copied" : "Copy code"}
        >
          {copied ? <Check className="h-3 w-3" /> : <Copy className="h-3 w-3" />}
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <SyntaxHighlighter
        language={language}
        style={oneDark}
        PreTag="div"
        customStyle={{
          margin: 0,
          padding: "0.875rem 1rem",
          background: "transparent",
          fontSize: "0.85em",
          lineHeight: 1.6,
        }}
        codeTagProps={{
          style: { fontFamily: "var(--font-mono)" },
        }}
      >
        {code}
      </SyntaxHighlighter>
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
