"use client";

import { BookOpen } from "lucide-react";

import { CourseBuilderPanel } from "@/components/coursebuilder/CourseBuilderPanel";
import type { UUID } from "@/lib/types";
import { cn } from "@/lib/utils";

interface Props {
  conversationId: UUID;
  className?: string;
}

export function CoursePanel({ conversationId, className }: Props) {
  return (
    <section
      className={cn(
        "flex h-full min-h-0 min-w-0 flex-col overflow-hidden bg-background",
        className,
      )}
      aria-label="Generated course"
    >
      <header className="app-chrome flex h-11 items-center gap-2 border-b border-border px-4">
        <BookOpen className="h-4 w-4 text-primary" />
        <h2 className="truncate text-sm font-semibold">Generated course</h2>
      </header>

      <div className="min-h-0 flex-1 overflow-y-auto">
        <CourseBuilderPanel conversationId={conversationId} />
      </div>
    </section>
  );
}
