"use client";

import { useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";

import { AlertCircle, BookOpen, RefreshCw, Sparkles } from "lucide-react";
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
import type { CourseBuilderChapter, CourseBuilderStatus, UUID } from "@/lib/types";
import { cn } from "@/lib/utils";

import { CourseBuilderChapterList } from "./ChapterList";
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
  const activeChapter = useMemo(
    () =>
      chapters.find((chapter) => chapter.id === activeChapterId && !chapter.is_locked) ??
      chapters.find((chapter) => !chapter.is_locked) ??
      chapters[0] ??
      null,
    [activeChapterId, chapters],
  );

  useEffect(() => {
    if (activeChapter && activeChapter.id !== activeChapterId) {
      setActiveChapterId(activeChapter.id);
    }
  }, [activeChapter, activeChapterId]);

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

      <CourseBuilderChapterList
        chapters={chapters}
        activeChapterId={activeChapter?.id ?? null}
        onSelect={setActiveChapterId}
      />

      {activeChapter && (
        <ChapterDetail
          conversationId={conversationId}
          chapter={activeChapter}
        />
      )}
    </div>
  );
}

function ChapterDetail({
  conversationId,
  chapter,
}: {
  conversationId: UUID;
  chapter: CourseBuilderChapter;
}) {
  if (chapter.is_locked) {
    return (
      <StateCard
        icon={<BookOpen className="h-4 w-4 text-muted-foreground" />}
        title="Chapter locked"
        body="Pass the previous chapter quiz to unlock this chapter."
      />
    );
  }
  return (
    <section className="flex flex-col gap-4 border-t border-border pt-4">
      <header>
        <h4 className="text-sm font-semibold">{chapter.title}</h4>
        {chapter.summary && (
          <div className="course-markdown mt-1 text-xs leading-5 text-muted-foreground">
            <AssistantMarkdown content={chapter.summary} />
          </div>
        )}
      </header>

      {chapter.lessons.map((lesson) => (
        <section key={lesson.id} className="flex flex-col gap-2">
          <div>
            <h5 className="text-xs font-semibold">{lesson.title}</h5>
            {lesson.learning_objectives.length > 0 && (
              <ul className="mt-1 list-disc space-y-1 pl-4 text-[11px] leading-4 text-muted-foreground">
                {lesson.learning_objectives.map((objective) => (
                  <li key={objective}>{objective}</li>
                ))}
              </ul>
            )}
          </div>
          <div className="flex flex-col gap-2">
            {lesson.blocks.map((block) => (
              <LessonBlockRenderer key={block.id} block={block} />
            ))}
          </div>
        </section>
      ))}

      {chapter.quiz && (
        <CourseBuilderQuiz
          conversationId={conversationId}
          chapterId={chapter.id}
          quiz={chapter.quiz}
        />
      )}
    </section>
  );
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
