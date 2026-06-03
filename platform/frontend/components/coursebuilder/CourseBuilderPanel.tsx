"use client";

import { useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";

import {
  AlertCircle,
  BookOpen,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  ClipboardCheck,
  FileText,
  Lock,
  PlayCircle,
  RefreshCw,
  Sparkles,
} from "lucide-react";
import { toast } from "sonner";

import { AssistantMarkdown } from "@/components/chat/MessageBubble";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import {
  useCourseBuilder,
  useCourseBuilderProgress,
  useGenerateCourseBuilder,
  useRebuildCourseBuilder,
} from "@/hooks/useCourseBuilder";
import type {
  CourseBuilderChapter,
  CourseBuilderLesson,
  CourseBuilderQuiz as CourseBuilderQuizType,
  CourseBuilderStatus,
  UUID,
} from "@/lib/types";
import { cn } from "@/lib/utils";

import { LessonBlockRenderer } from "./LessonBlockRenderer";
import { CourseBuilderQuiz } from "./Quiz";

interface Props {
  conversationId: UUID;
}

const RUNNING = new Set<CourseBuilderStatus>([
  "queued",
  "analyzing",
  "generating_outline",
  "generating_chapters",
  "generating_lessons",
  "generating_quizzes",
  "validating",
]);

export function CourseBuilderPanel({ conversationId }: Props) {
  const { data, isLoading } = useCourseBuilder(conversationId);
  const generate = useGenerateCourseBuilder(conversationId);
  const rebuild = useRebuildCourseBuilder(conversationId);
  const running = data ? Boolean(data.id && RUNNING.has(data.status)) : false;
  useCourseBuilderProgress(conversationId, running);
  const generateCourse = () =>
    generate.mutate(undefined, {
      onError: (err) => toast.error(`Course generation failed: ${err.message}`),
    });
  const rebuildCourse = () =>
    rebuild.mutate(undefined, {
      onError: (err) => toast.error(`Course rebuild failed: ${err.message}`),
    });

  const chapters = useMemo(() => data?.chapters ?? [], [data?.chapters]);
  const [activeChapterId, setActiveChapterId] = useState<UUID | null>(null);
  const [activeContentKey, setActiveContentKey] = useState<string | null>(null);
  const activeChapter = useMemo(
    () =>
      chapters.find((chapter) => chapter.id === activeChapterId && !chapter.is_locked) ??
      chapters.find((chapter) => !chapter.is_locked) ??
      chapters[0] ??
      null,
    [activeChapterId, chapters],
  );

  const selectChapter = (chapterId: UUID) => {
    if (chapterId !== activeChapterId) {
      setActiveContentKey(null);
    }
    setActiveChapterId(chapterId);
  };

  const toggleContent = (contentKey: string) => {
    setActiveContentKey((current) => (current === contentKey ? null : contentKey));
  };

  useEffect(() => {
    if (activeChapter && activeChapter.id !== activeChapterId) {
      setActiveChapterId(activeChapter.id);
    }
  }, [activeChapter, activeChapterId]);

  useEffect(() => {
    if (!activeChapter || !activeContentKey) return;
    const validKeys = new Set(activeChapter.lessons.map((lesson) => lessonContentKey(lesson.id)));
    if (activeChapter.quiz) {
      validKeys.add(quizContentKey(activeChapter.id));
    }
    if (!validKeys.has(activeContentKey)) {
      setActiveContentKey(null);
    }
  }, [activeChapter, activeContentKey]);

  if (isLoading) {
    return (
      <div className="px-4 py-4 text-xs text-muted-foreground">
        Loading the generated course...
      </div>
    );
  }

  if (!data || data.total_file_count === 0) {
    return (
      <div className="px-4 py-4 text-xs leading-5 text-muted-foreground">
        Upload course files first. After processing, TeacherLM will build a
        structured text course here.
      </div>
    );
  }

  if (data.pending_file_count > 0) {
    return (
      <div className="flex flex-col gap-3 px-4 py-4">
        <StateCard
          icon={<BookOpen className="h-4 w-4 text-primary" />}
          title="Course will be generated after processing"
          body={`${data.pending_file_count} of ${data.total_file_count} files are not ready yet.`}
        />
      </div>
    );
  }

  if (running) {
    const latest = data.progress_events.at(-1);
    const percent = latest?.percent ?? 4;
    return (
      <div className="flex flex-col gap-3 px-4 py-4">
        <StateCard
          icon={<Sparkles className="h-4 w-4 text-primary" />}
          title="Building your course"
          body={latest?.message || "Preparing a structured course from your files."}
        />
        <ProgressBar percent={percent} />
        {data.status === "queued" && (
          <Button
            size="sm"
            variant="secondary"
            onClick={generateCourse}
            disabled={generate.isPending}
          >
            <Sparkles className="h-3.5 w-3.5" />
            Start generation
          </Button>
        )}
        <EventList events={data.progress_events.slice(-5)} />
      </div>
    );
  }

  if (data.status === "failed") {
    return (
      <div className="flex flex-col gap-3 px-4 py-4">
        <StateCard
          icon={<AlertCircle className="h-4 w-4 text-danger" />}
          title="Course generation failed"
          body={data.error || "TeacherLM could not build the course."}
        />
        <Button
          size="sm"
          variant="secondary"
          onClick={rebuildCourse}
          disabled={rebuild.isPending}
        >
          <RefreshCw className="h-3.5 w-3.5" />
          Rebuild course
        </Button>
      </div>
    );
  }

  if (chapters.length === 0) {
    return (
      <div className="flex flex-col gap-3 px-4 py-4">
        <StateCard
          icon={<BookOpen className="h-4 w-4 text-primary" />}
          title="Ready to generate"
          body="All files are ready. Generate the structured course when you are ready."
        />
        <Button
          size="sm"
          onClick={generateCourse}
          disabled={generate.isPending}
        >
          <Sparkles className="h-3.5 w-3.5" />
          Generate course
        </Button>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4 px-4 py-4">
      <section className="flex flex-col gap-2">
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0">
            <h3 className="text-sm font-semibold">{data.title || "Generated course"}</h3>
            {data.description && (
              <p className="mt-1 line-clamp-3 text-xs leading-5 text-muted-foreground">
                {data.description}
              </p>
            )}
          </div>
          <Badge variant="primary">{chapters.length} chapters</Badge>
        </div>
        <Button
          size="sm"
          variant="ghost"
          className="justify-start px-0 text-xs"
          onClick={rebuildCourse}
          disabled={rebuild.isPending}
        >
          <RefreshCw className="h-3.5 w-3.5" />
          Rebuild
        </Button>
      </section>

      <CourseBuilderAccordion
        conversationId={conversationId}
        chapters={chapters}
        activeChapterId={activeChapter?.id ?? null}
        activeContentKey={activeContentKey}
        onSelectChapter={selectChapter}
        onToggleContent={toggleContent}
      />
    </div>
  );
}

function CourseBuilderAccordion({
  conversationId,
  chapters,
  activeChapterId,
  activeContentKey,
  onSelectChapter,
  onToggleContent,
}: {
  conversationId: UUID;
  chapters: CourseBuilderChapter[];
  activeChapterId: UUID | null;
  activeContentKey: string | null;
  onSelectChapter: (chapterId: UUID) => void;
  onToggleContent: (contentKey: string) => void;
}) {
  return (
    <ol className="app-chrome flex flex-col gap-2">
      {chapters.map((chapter) => (
        <li key={chapter.id}>
          <ChapterAccordionItem
            conversationId={conversationId}
            chapter={chapter}
            open={chapter.id === activeChapterId && !chapter.is_locked}
            activeContentKey={activeContentKey}
            onSelectChapter={onSelectChapter}
            onToggleContent={onToggleContent}
          />
        </li>
      ))}
    </ol>
  );
}

function ChapterAccordionItem({
  conversationId,
  chapter,
  open,
  activeContentKey,
  onSelectChapter,
  onToggleContent,
}: {
  conversationId: UUID;
  chapter: CourseBuilderChapter;
  open: boolean;
  activeContentKey: string | null;
  onSelectChapter: (chapterId: UUID) => void;
  onToggleContent: (contentKey: string) => void;
}) {
  return (
    <section
      className={cn(
        "rounded-md border border-border bg-surface transition-colors",
        chapter.is_locked ? "opacity-70" : "hover:bg-muted/40",
        open && "border-primary/60 bg-primary/10",
      )}
    >
      <button
        type="button"
        className="flex w-full items-start gap-2 px-3 py-2 text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        onClick={() => onSelectChapter(chapter.id)}
        disabled={chapter.is_locked}
        aria-expanded={open}
      >
        <ChapterStatusIcon chapter={chapter} />
        <div className="min-w-0 flex-1">
          <p className="line-clamp-2 text-xs font-medium">{chapter.title}</p>
          <div className="mt-1 flex flex-wrap items-center gap-1.5">
            {chapter.completed && <Badge variant="success">Completed</Badge>}
            {chapter.is_locked && <Badge variant="muted">Locked</Badge>}
            <Badge variant="muted">{chapter.lessons.length} subchapters</Badge>
            {chapter.quiz && <Badge variant="primary">Quiz</Badge>}
            {chapter.attempts > 0 && (
              <span className="text-[11px] text-muted-foreground">
                best {Math.round(chapter.best_score * 100)}%
              </span>
            )}
          </div>
        </div>
        {open ? (
          <ChevronDown className="mt-0.5 h-3.5 w-3.5 shrink-0 text-muted-foreground" />
        ) : (
          <ChevronRight className="mt-0.5 h-3.5 w-3.5 shrink-0 text-muted-foreground" />
        )}
      </button>

      {open && (
        <div className="flex flex-col gap-3 border-t border-border bg-background/60 px-3 py-3">
          {chapter.summary && (
            <div className="course-markdown text-xs leading-5 text-muted-foreground">
              <AssistantMarkdown content={chapter.summary} />
            </div>
          )}

          <div className="flex flex-col gap-1.5">
            {chapter.lessons.map((lesson, index) => (
              <LessonAccordionItem
                key={lesson.id}
                lesson={lesson}
                index={index}
                open={activeContentKey === lessonContentKey(lesson.id)}
                onToggle={() => onToggleContent(lessonContentKey(lesson.id))}
              />
            ))}
            {chapter.quiz && (
              <QuizAccordionItem
                conversationId={conversationId}
                chapterId={chapter.id}
                quiz={chapter.quiz}
                open={activeContentKey === quizContentKey(chapter.id)}
                onToggle={() => onToggleContent(quizContentKey(chapter.id))}
              />
            )}
            {chapter.lessons.length === 0 && !chapter.quiz && (
              <p className="text-xs leading-5 text-muted-foreground">
                No subchapters or quiz were generated for this chapter.
              </p>
            )}
          </div>
        </div>
      )}
    </section>
  );
}

function LessonAccordionItem({
  lesson,
  index,
  open,
  onToggle,
}: {
  lesson: CourseBuilderLesson;
  index: number;
  open: boolean;
  onToggle: () => void;
}) {
  return (
    <section className="flex flex-col">
      <button
        type="button"
        className={cn(
          "flex w-full items-center gap-2 rounded-md px-2 py-2 text-left text-xs transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
          open ? "bg-primary/10 text-primary" : "text-foreground hover:bg-muted/70",
        )}
        onClick={onToggle}
        aria-expanded={open}
      >
        <span
          className={cn(
            "flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-[11px] font-semibold",
            open ? "bg-primary text-primary-foreground" : "bg-muted text-muted-foreground",
          )}
        >
          {index + 1}
        </span>
        <FileText className={cn("h-3.5 w-3.5 shrink-0", open ? "text-primary" : "text-muted-foreground")} />
        <span className="min-w-0 flex-1 truncate font-medium">{lesson.title}</span>
        <Badge variant={lesson.support_status === "supported" ? "success" : "warning"}>
          {lesson.support_status === "supported" ? "Supported" : "Needs source"}
        </Badge>
        {open ? (
          <ChevronDown className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
        ) : (
          <ChevronRight className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
        )}
      </button>
      {open && <LessonDetail lesson={lesson} />}
    </section>
  );
}

function LessonDetail({ lesson }: { lesson: CourseBuilderLesson }) {
  return (
    <div className="flex flex-col gap-3 border-l border-border py-2 pl-4">
      {lesson.learning_objectives.length > 0 && (
        <ul className="list-disc space-y-1 pl-4 text-[11px] leading-4 text-muted-foreground">
          {lesson.learning_objectives.map((objective) => (
            <li key={objective}>{objective}</li>
          ))}
        </ul>
      )}
      {lesson.blocks.length > 0 ? (
        <div className="flex flex-col gap-2">
          {lesson.blocks.map((block) => (
            <LessonBlockRenderer key={block.id} block={block} />
          ))}
        </div>
      ) : (
        <p className="text-xs leading-5 text-muted-foreground">
          No lesson blocks were generated for this subchapter.
        </p>
      )}
    </div>
  );
}

function QuizAccordionItem({
  conversationId,
  chapterId,
  quiz,
  open,
  onToggle,
}: {
  conversationId: UUID;
  chapterId: UUID;
  quiz: CourseBuilderQuizType;
  open: boolean;
  onToggle: () => void;
}) {
  return (
    <section className="flex flex-col">
      <button
        type="button"
        className={cn(
          "flex w-full items-center gap-2 rounded-md px-2 py-2 text-left text-xs transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
          open ? "bg-primary/10 text-primary" : "text-foreground hover:bg-muted/70",
        )}
        onClick={onToggle}
        aria-expanded={open}
      >
        <span className={cn("flex h-5 w-5 shrink-0 items-center justify-center rounded-full", open ? "bg-primary text-primary-foreground" : "bg-muted text-muted-foreground")}>
          <ClipboardCheck className="h-3.5 w-3.5" />
        </span>
        <span className="min-w-0 flex-1 truncate font-medium">Chapter quiz</span>
        <Badge variant="muted">Pass {Math.round(quiz.pass_score * 100)}%</Badge>
        {open ? (
          <ChevronDown className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
        ) : (
          <ChevronRight className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
        )}
      </button>
      {open && (
        <div className="border-l border-border py-2 pl-4">
          <CourseBuilderQuiz
            conversationId={conversationId}
            chapterId={chapterId}
            quiz={quiz}
          />
        </div>
      )}
    </section>
  );
}

function ChapterStatusIcon({ chapter }: { chapter: CourseBuilderChapter }) {
  if (chapter.completed) return <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-[hsl(var(--success))]" />;
  if (chapter.is_locked) return <Lock className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />;
  return <PlayCircle className="mt-0.5 h-4 w-4 shrink-0 text-primary" />;
}

function lessonContentKey(lessonId: UUID): string {
  return `lesson:${lessonId}`;
}

function quizContentKey(chapterId: UUID): string {
  return `quiz:${chapterId}`;
}

function StateCard({
  icon,
  title,
  body,
}: {
  icon: ReactNode;
  title: string;
  body: string;
}) {
  return (
    <section className="rounded-md border border-border bg-surface p-3">
      <div className="flex items-start gap-2">
        {icon}
        <div className="min-w-0">
          <h3 className="text-sm font-semibold">{title}</h3>
          <p className="mt-1 text-xs leading-5 text-muted-foreground">{body}</p>
        </div>
      </div>
    </section>
  );
}

function ProgressBar({ percent }: { percent: number }) {
  return (
    <div className="h-2 overflow-hidden rounded-full bg-muted">
      <div
        className="h-full bg-primary transition-all"
        style={{ width: `${Math.max(0, Math.min(100, percent))}%` }}
      />
    </div>
  );
}

function EventList({
  events,
}: {
  events: Array<{ id: UUID; stage: string; message: string }>;
}) {
  if (events.length === 0) return null;
  return (
    <ol className="flex flex-col gap-1 text-[11px] text-muted-foreground">
      {events.map((event) => (
        <li key={event.id} className={cn("rounded-md bg-muted px-2 py-1")}>
          <span className="font-medium text-foreground">{event.stage}</span>
          {event.message ? ` - ${event.message}` : ""}
        </li>
      ))}
    </ol>
  );
}
