"use client";

import { CheckCircle2, ChevronRight, Lock, PlayCircle } from "lucide-react";

import { Badge } from "@/components/ui/Badge";
import type { CourseBuilderChapter, UUID } from "@/lib/types";
import { cn } from "@/lib/utils";

interface Props {
  chapters: CourseBuilderChapter[];
  activeChapterId: UUID | null;
  onSelect: (chapterId: UUID) => void;
}

export function CourseBuilderChapterList({
  chapters,
  activeChapterId,
  onSelect,
}: Props) {
  return (
    <ol className="flex flex-col gap-2">
      {chapters.map((chapter) => (
        <li
          key={chapter.id}
          className={cn(
            "rounded-md border border-border bg-surface",
            activeChapterId === chapter.id && "border-primary/60",
          )}
        >
          <button
            type="button"
            className="flex w-full items-start gap-2 px-3 py-2 text-left"
            onClick={() => onSelect(chapter.id)}
            disabled={chapter.is_locked}
          >
            <StatusIcon chapter={chapter} />
            <div className="min-w-0 flex-1">
              <p className="line-clamp-2 text-xs font-medium">{chapter.title}</p>
              <div className="mt-1 flex flex-wrap items-center gap-1.5">
                {chapter.completed && <Badge variant="success">Completed</Badge>}
                {chapter.is_locked && <Badge variant="muted">Locked</Badge>}
                {chapter.attempts > 0 && (
                  <span className="text-[11px] text-muted-foreground">
                    best {Math.round(chapter.best_score * 100)}%
                  </span>
                )}
              </div>
            </div>
            <ChevronRight className="mt-0.5 h-3.5 w-3.5 shrink-0 text-muted-foreground" />
          </button>
        </li>
      ))}
    </ol>
  );
}

function StatusIcon({ chapter }: { chapter: CourseBuilderChapter }) {
  if (chapter.completed) return <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-[hsl(var(--success))]" />;
  if (chapter.is_locked) return <Lock className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />;
  return <PlayCircle className="mt-0.5 h-4 w-4 shrink-0 text-primary" />;
}
