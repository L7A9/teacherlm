"use client";

import { useEffect, useRef, useState, type CSSProperties } from "react";

import Link from "next/link";

import ChatRoundedIcon from "@mui/icons-material/ChatRounded";
import { Dialog, Fab, Tooltip } from "@mui/material";
import {
  GraduationCap,
  GripVertical,
  PanelLeft,
  PanelRight,
  Settings,
} from "lucide-react";

import { GeneratorDialog } from "@/components/chat/GeneratorDialog";
import { ReviewTestDialog } from "@/components/review/ReviewTestDialog";
import { Button } from "@/components/ui/Button";
import { ChatPanel } from "@/components/workspace/ChatPanel";
import { CoursePanel } from "@/components/workspace/CoursePanel";
import { ProgressPanel } from "@/components/workspace/ProgressPanel";
import { SourcesPanel } from "@/components/workspace/SourcesPanel";
import { useConversation, useUpdateConversation } from "@/hooks/useConversations";
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
  const mainRef = useRef<HTMLDivElement | null>(null);
  const [courseWidth, setCourseWidth] = useState(48);
  const [isNarrow, setIsNarrow] = useState(() =>
    typeof window !== "undefined" ? window.innerWidth < 1024 : false,
  );
  const [mobileSourcesOpen, setMobileSourcesOpen] = useState(false);
  const [mobileGeneratedOpen, setMobileGeneratedOpen] = useState(false);
  const [mobileChatOpen, setMobileChatOpen] = useState(false);
  const sourcesVisible = isNarrow ? mobileSourcesOpen : !sourcesCollapsed;
  const generatedVisible = isNarrow ? mobileGeneratedOpen : !progressCollapsed;

  useEffect(() => {
    setActive(conversationId);
    return () => setActive(null);
  }, [conversationId, setActive]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const update = () => setIsNarrow(window.innerWidth < 1024);
    update();
    window.addEventListener("resize", update);
    return () => window.removeEventListener("resize", update);
  }, []);

  useEffect(() => {
    if (!isNarrow) {
      setMobileSourcesOpen(false);
      setMobileGeneratedOpen(false);
      setMobileChatOpen(false);
      return;
    }

    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      setMobileSourcesOpen(false);
      setMobileGeneratedOpen(false);
      setMobileChatOpen(false);
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [isNarrow]);

  const closeMobileDrawers = () => {
    setMobileSourcesOpen(false);
    setMobileGeneratedOpen(false);
  };

  return (
    <div className="flex h-dvh flex-col bg-background text-foreground">
      <TopBar
        conversationId={conversationId}
        sourcesVisible={sourcesVisible}
        generatedVisible={generatedVisible}
        onToggleSources={() => {
          if (isNarrow) setMobileSourcesOpen((open) => !open);
          else toggleSources();
        }}
        onToggleProgress={() => {
          if (isNarrow) setMobileGeneratedOpen((open) => !open);
          else toggleProgress();
        }}
      />

      <div className="relative grid min-h-0 flex-1 grid-cols-1 overflow-hidden lg:grid-cols-[auto_minmax(0,1fr)_auto]">
        {isNarrow && (mobileSourcesOpen || mobileGeneratedOpen) && (
          <button
            type="button"
            aria-label="Close side panels"
            className="absolute inset-0 z-10 bg-background/60 backdrop-blur-sm lg:hidden"
            onClick={closeMobileDrawers}
          />
        )}
        {sourcesVisible && (
          <SourcesPanel
            conversationId={conversationId}
            onClose={closeMobileDrawers}
            className="absolute inset-y-0 left-0 z-20 w-[min(88vw,320px)] shadow-2xl lg:static lg:z-auto lg:h-full lg:w-[300px] lg:shadow-none"
          />
        )}
        <main
          ref={mainRef}
          className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden md:flex-row"
          aria-label="Learning workspace"
        >
          <div
            className={cn(
              "min-h-0 min-w-0 border-border",
              isNarrow
                ? "flex-1"
                : "flex-shrink-0 basis-[44%] border-b md:basis-[var(--course-pane-width)] md:border-b-0 md:border-r",
            )}
            style={
              {
                "--course-pane-width": `clamp(320px, ${courseWidth}%, calc(100% - 360px))`,
              } as CSSProperties
            }
          >
            <CoursePanel conversationId={conversationId} />
          </div>
          {!isNarrow && (
            <>
              <ResizeHandle
                onDrag={(clientX) => {
                  const rect = mainRef.current?.getBoundingClientRect();
                  if (!rect || rect.width <= 0) return;
                  const next = ((clientX - rect.left) / rect.width) * 100;
                  setCourseWidth(Math.min(68, Math.max(32, next)));
                }}
              />
              <div className="min-h-0 min-w-0 flex-1">
                <ChatPanel conversationId={conversationId} />
              </div>
            </>
          )}
        </main>
        {generatedVisible && (
          <ProgressPanel
            conversationId={conversationId}
            onClose={closeMobileDrawers}
            className="absolute inset-y-0 right-0 z-20 w-[min(88vw,340px)] shadow-2xl lg:static lg:z-auto lg:h-full lg:w-[320px] lg:shadow-none"
          />
        )}

        {isNarrow && (
          <>
            <Tooltip title="Ask your teacher">
              <Fab
                color="primary"
                aria-label="Open chat"
                onClick={() => setMobileChatOpen(true)}
                sx={{
                  position: "absolute",
                  bottom: { xs: 16, sm: 20 },
                  right: { xs: 16, sm: 20 },
                  zIndex: 30,
                  boxShadow: "0 18px 42px hsl(var(--background) / 0.45)",
                }}
              >
                <ChatRoundedIcon />
              </Fab>
            </Tooltip>
            <Dialog
              open={mobileChatOpen}
              onClose={() => setMobileChatOpen(false)}
              aria-label="Chat with your teacher"
              sx={{
                "& .MuiBackdrop-root": {
                  bgcolor: "hsl(var(--background) / 0.62)",
                  backdropFilter: "blur(10px)",
                },
                "& .MuiDialog-container": {
                  alignItems: "flex-end",
                  justifyContent: { xs: "center", sm: "flex-end" },
                },
                "& .MuiDialog-paper": {
                  m: { xs: 1, sm: 2 },
                  width: { xs: "calc(100% - 16px)", sm: 460 },
                  maxWidth: "none",
                  height: { xs: "min(82dvh, 700px)", sm: "min(78dvh, 720px)" },
                  overflow: "hidden",
                  border: "1px solid hsl(var(--border))",
                  bgcolor: "hsl(var(--background))",
                  borderRadius: "8px",
                },
              }}
            >
              <ChatPanel
                conversationId={conversationId}
                onClose={() => setMobileChatOpen(false)}
              />
            </Dialog>
          </>
        )}
      </div>

      <GeneratorDialog />
      <ReviewTestDialog conversationId={conversationId} />
    </div>
  );
}

interface TopBarProps {
  conversationId: UUID;
  sourcesVisible: boolean;
  generatedVisible: boolean;
  onToggleSources: () => void;
  onToggleProgress: () => void;
}

function TopBar({
  conversationId,
  sourcesVisible,
  generatedVisible,
  onToggleSources,
  onToggleProgress,
}: TopBarProps) {
  const { data: conversation } = useConversation(conversationId);

  return (
    <header className="app-chrome app-pane flex h-12 shrink-0 items-center justify-between border-b border-border px-3 sm:px-4">
      <div className="flex min-w-0 items-center gap-2">
        <Button
          variant="ghost"
          size="icon"
          onClick={onToggleSources}
          aria-label={sourcesVisible ? "Hide sources" : "Show sources"}
          title={sourcesVisible ? "Hide sources" : "Show sources"}
        >
          <PanelLeft className="h-4 w-4" />
        </Button>
        <Link
          href="/"
          className="flex items-center gap-2 rounded-md px-1 py-0.5 transition-colors hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          aria-label="Go to TeacherLM home"
        >
          <div className="flex h-7 w-7 items-center justify-center rounded-md bg-primary/15 text-primary">
            <GraduationCap className="h-4 w-4" />
          </div>
          <span className="text-sm font-semibold tracking-tight">TeacherLM</span>
        </Link>
        <div className="hidden h-5 w-px shrink-0 bg-border sm:block" />
        <EditableConversationTitle
          conversationId={conversationId}
          title={conversation?.title ?? ""}
        />
      </div>

      <div className="flex shrink-0 items-center gap-1">
        <Button variant="ghost" size="icon" asChild title="Settings">
          <Link href="/settings" aria-label="Settings">
            <Settings className="h-4 w-4" />
          </Link>
        </Button>
        <Button
          variant="ghost"
          size="icon"
          onClick={onToggleProgress}
          aria-label={generatedVisible ? "Hide generated items" : "Show generated items"}
          title={generatedVisible ? "Hide generated items" : "Show generated items"}
        >
          <PanelRight className="h-4 w-4" />
        </Button>
      </div>
    </header>
  );
}

interface EditableConversationTitleProps {
  conversationId: UUID;
  title: string;
}

function EditableConversationTitle({
  conversationId,
  title,
}: EditableConversationTitleProps) {
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
        onChange={(event) => setDraft(event.target.value)}
        onBlur={commit}
        onKeyDown={(event) => {
          if (event.key === "Enter") {
            event.preventDefault();
            commit();
          } else if (event.key === "Escape") {
            event.preventDefault();
            setDraft(title);
            setEditing(false);
          }
        }}
        className="min-w-0 max-w-[34vw] truncate rounded-sm bg-muted px-1.5 py-0.5 text-sm font-medium outline-none ring-1 ring-ring sm:max-w-[42vw] lg:max-w-[30vw]"
        aria-label="Conversation title"
      />
    );
  }

  return (
    <button
      type="button"
      onClick={() => setEditing(true)}
      title="Click to rename"
      className="min-w-0 max-w-[34vw] truncate rounded-sm px-1.5 py-0.5 text-left text-sm font-medium text-muted-foreground hover:bg-muted hover:text-foreground sm:max-w-[42vw] lg:max-w-[30vw]"
    >
      {title || "Your teacher"}
    </button>
  );
}

function ResizeHandle({ onDrag }: { onDrag: (clientX: number) => void }) {
  return (
    <div
      role="separator"
      aria-label="Resize course and chat"
      aria-orientation="vertical"
      className="hidden w-2 cursor-col-resize items-center justify-center bg-border/40 text-muted-foreground transition-colors hover:bg-primary/25 md:flex"
      onPointerDown={(event) => {
        event.currentTarget.setPointerCapture(event.pointerId);
        onDrag(event.clientX);
      }}
      onPointerMove={(event) => {
        if (event.buttons !== 1) return;
        onDrag(event.clientX);
      }}
    >
      <GripVertical className="h-4 w-4" />
    </div>
  );
}
