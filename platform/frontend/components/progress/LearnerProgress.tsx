"use client";

import { useMemo } from "react";

import { CheckCircle2, Flame, Sparkles } from "lucide-react";

import { Badge } from "@/components/ui/Badge";
import type { LearnerState, UUID } from "@/lib/types";
import { cn } from "@/lib/utils";
import { useProgressStore } from "@/stores/progressStore";

interface Props {
  conversationId: UUID;
  className?: string;
}

const EMPTY: LearnerState = {
  conversation_id: "",
  understood_concepts: [],
  struggling_concepts: [],
  mastery_scores: {},
  session_turns: 0,
  turns_since_progress: 0,
};

export function LearnerProgress({ conversationId, className }: Props) {
  const state =
    useProgressStore((s) => s.stateByConversation[conversationId]) ?? EMPTY;

  const ranked = useMemo(() => {
    return Object.entries(state.mastery_scores)
      .map(([concept, score]) => ({ concept, score }))
      .sort((a, b) => b.score - a.score);
  }, [state.mastery_scores]);

  return (
    <div className={cn("flex flex-col gap-4", className)}>
      <header className="flex items-center justify-between">
        <h2 className="text-sm font-semibold">Your learning progress</h2>
        <Badge variant="muted">
          <Flame className="h-3 w-3" />
          {state.session_turns} turn{state.session_turns === 1 ? "" : "s"}
        </Badge>
      </header>

      {ranked.length === 0 ? (
        <p className="text-xs text-muted-foreground">
          Start chatting or generate a quiz — your concept mastery will appear
          here.
        </p>
      ) : (
        <MasteryBars ranked={ranked} />
      )}

      <ConceptList
        title="Concepts mastered"
        icon={<CheckCircle2 className="h-3.5 w-3.5 text-[hsl(var(--success))]" />}
        tone="success"
        concepts={state.understood_concepts}
        emptyHint="Keep practicing — mastered concepts will show up here."
      />

      <ConceptList
        title="Needs review"
        icon={<Sparkles className="h-3.5 w-3.5 text-[hsl(var(--warning))]" />}
        tone="warning"
        concepts={state.struggling_concepts}
        emptyHint="Nothing to review yet."
      />
    </div>
  );
}

function MasteryBars({
  ranked,
}: {
  ranked: { concept: string; score: number }[];
}) {
  return (
    <ul className="flex flex-col gap-2">
      {ranked.map(({ concept, score }) => (
        <li key={concept} className="flex flex-col gap-1">
          <div className="flex items-center justify-between text-xs">
            <span className="truncate" title={concept}>
              {concept}
            </span>
            <span className="tabular-nums text-muted-foreground">
              {Math.round(score * 100)}%
            </span>
          </div>
          <div className="h-2 overflow-hidden rounded-full bg-muted">
            <div
              className={cn("h-full rounded-full", masteryColor(score))}
              style={{ width: `${Math.max(4, score * 100)}%` }}
            />
          </div>
        </li>
      ))}
    </ul>
  );
}

function ConceptList({
  title,
  icon,
  concepts,
  tone,
  emptyHint,
}: {
  title: string;
  icon: React.ReactNode;
  concepts: string[];
  tone: "success" | "warning";
  emptyHint: string;
}) {
  return (
    <section className="flex flex-col gap-2">
      <h3 className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
        {icon}
        {title}
      </h3>
      {concepts.length === 0 ? (
        <p className="text-[11px] text-muted-foreground">{emptyHint}</p>
      ) : (
        <ul className="flex flex-wrap gap-1.5">
          {concepts.map((c) => (
            <li key={c}>
              <Badge variant={tone}>{c}</Badge>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

function masteryColor(score: number): string {
  if (score >= 0.7) return "bg-[hsl(var(--success))]";
  if (score <= 0.3) return "bg-[hsl(var(--danger))]";
  return "bg-[hsl(var(--warning))]";
}
