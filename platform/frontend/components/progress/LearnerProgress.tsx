"use client";

import { useMemo, useState } from "react";

import {
  CheckCircle2,
  ChevronDown,
  ClipboardCheck,
  Flame,
  Send,
  Sparkles,
} from "lucide-react";

import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { useLearnerState } from "@/hooks/useConversations";
import { useFiles } from "@/hooks/useFiles";
import {
  useStartKnowledgeCheck,
  useSubmitKnowledgeCheck,
} from "@/hooks/useKnowledgeChecks";
import type {
  ConceptProgress,
  KnowledgeCheckQuestion,
  KnownConcept,
  LearnerState,
  ObjectiveProgress,
  PhaseProgress,
  UUID,
} from "@/lib/types";
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
  known_concepts: [],
  concept_progress: [],
  learning_phases: [],
  objective_progress: [],
  phase_progress: [],
};

type ProgressView = "course" | "progressed" | "weakest";

export function LearnerProgress({ conversationId, className }: Props) {
  useFiles(conversationId);
  useLearnerState(conversationId);
  const startCheck = useStartKnowledgeCheck(conversationId);
  const submitCheck = useSubmitKnowledgeCheck(conversationId);
  const [activeCheck, setActiveCheck] = useState<KnowledgeCheckQuestion | null>(null);
  const [answer, setAnswer] = useState<string>("");
  const [view, setView] = useState<ProgressView>("course");
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});

  const state =
    useProgressStore((s) => s.stateByConversation[conversationId]) ?? EMPTY;

  const ranked = useMemo(() => {
    const canonical = state.concept_progress ?? [];
    if (canonical.length > 0) {
      return canonical
        .map((item) => ({
          conceptId: item.concept_id,
          concept: item.name,
          score: item.mastery,
        }))
        .sort((a, b) => b.score - a.score || a.concept.localeCompare(b.concept));
    }
    return Object.entries(state.mastery_scores)
      .map(([concept, score]) => ({ conceptId: undefined, concept, score }))
      .sort((a, b) => b.score - a.score || a.concept.localeCompare(b.concept));
  }, [state.concept_progress, state.mastery_scores]);

  const phaseRows = useMemo(
    () => buildPhaseRows(state, view),
    [state, view],
  );
  const hasLearningMap = phaseRows.length > 0;

  const beginCheck = async (filters: {
    conceptId?: UUID;
    phaseId?: UUID;
    objectiveId?: UUID;
  } = {}) => {
    const response = await startCheck.mutateAsync({
      concept_id: filters.conceptId ?? null,
      phase_id: filters.phaseId ?? null,
      objective_id: filters.objectiveId ?? null,
      count: 1,
    });
    setActiveCheck(response.checks[0] ?? null);
    setAnswer("");
  };

  const submitActiveCheck = async () => {
    if (!activeCheck) return;
    await submitCheck.mutateAsync({
      checkId: activeCheck.id,
      body: { answer },
    });
    setActiveCheck(null);
    setAnswer("");
  };

  return (
    <div className={cn("flex flex-col gap-4", className)}>
      <header className="flex items-center justify-between">
        <h2 className="text-sm font-semibold">Your learning progress</h2>
        <Badge variant="muted">
          <Flame className="h-3 w-3" />
          {state.session_turns} turn{state.session_turns === 1 ? "" : "s"}
        </Badge>
      </header>

      {!hasLearningMap && ranked.length === 0 ? (
        <p className="text-xs text-muted-foreground">
          Upload a document first. After chunking, the course phases will
          appear here.
        </p>
      ) : (
        <>
          {activeCheck && (
            <KnowledgeCheckCard
              check={activeCheck}
              answer={answer}
              pending={submitCheck.isPending}
              onAnswer={setAnswer}
              onSubmit={() => void submitActiveCheck()}
            />
          )}
          {hasLearningMap ? (
            <>
              <ProgressViewToggle value={view} onChange={setView} />
              <PhaseMap
                rows={phaseRows}
                expanded={expanded}
                pending={startCheck.isPending}
                onToggle={(phaseId) =>
                  setExpanded((current) => ({
                    ...current,
                    [phaseId]: !(current[phaseId] ?? true),
                  }))
                }
                onCheckPhase={(phaseId) => void beginCheck({ phaseId })}
                onCheckObjective={(objectiveId) => void beginCheck({ objectiveId })}
                onCheckConcept={(conceptId) => void beginCheck({ conceptId })}
              />
            </>
          ) : (
            <MasteryBars
              ranked={ranked}
              pending={startCheck.isPending}
              onCheck={(conceptId) => void beginCheck({ conceptId })}
            />
          )}
        </>
      )}

      <ConceptList
        title="Concepts mastered"
        icon={<CheckCircle2 className="h-3.5 w-3.5 text-[hsl(var(--success))]" />}
        tone="success"
        concepts={state.understood_concepts}
        emptyHint="Keep practicing. Mastered concepts will show up here."
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

interface PhaseRow {
  phase: PhaseProgress;
  synthetic?: boolean;
  objectives: Array<
    ObjectiveProgress & {
      concepts: ConceptProgress[];
    }
  >;
}

function buildPhaseRows(state: LearnerState, view: ProgressView): PhaseRow[] {
  const phaseProgress = state.phase_progress ?? [];
  const objectiveProgress = state.objective_progress ?? [];
  const conceptProgress = state.concept_progress ?? [];
  if (phaseProgress.length === 0) {
    return buildSyntheticPhaseRows(state, view);
  }

  const conceptById = new Map(conceptProgress.map((item) => [item.concept_id, item]));
  const objectivesByPhase = new Map<string, PhaseRow["objectives"]>();
  for (const objective of objectiveProgress) {
    const concepts = objective.concept_ids
      .map((conceptId) => conceptById.get(conceptId))
      .filter((item): item is ConceptProgress => Boolean(item));
    const next = { ...objective, concepts };
    const list = objectivesByPhase.get(objective.phase_id) ?? [];
    list.push(next);
    objectivesByPhase.set(objective.phase_id, list);
  }

  const rows = phaseProgress.map((phase) => ({
    phase,
    synthetic: false,
    objectives: (objectivesByPhase.get(phase.phase_id) ?? []).sort(
      (a, b) => a.order_index - b.order_index || a.objective_text.localeCompare(b.objective_text),
    ),
  }));

  if (view === "progressed") {
    return rows.sort((a, b) => b.phase.mastery - a.phase.mastery || a.phase.order_index - b.phase.order_index);
  }
  if (view === "weakest") {
    return rows.sort((a, b) => a.phase.mastery - b.phase.mastery || a.phase.order_index - b.phase.order_index);
  }
  return rows.sort((a, b) => a.phase.order_index - b.phase.order_index || a.phase.title.localeCompare(b.phase.title));
}

function buildSyntheticPhaseRows(state: LearnerState, view: ProgressView): PhaseRow[] {
  const concepts = state.concept_progress ?? [];
  if (concepts.length === 0) return [];
  const knownById = new Map((state.known_concepts ?? []).map((item) => [item.id, item]));
  const groups = new Map<string, { title: string; concepts: ConceptProgress[]; order: number }>();
  const hasCourseParts = concepts.some((concept) => {
    const known = knownById.get(concept.concept_id);
    const title = known?.course_parts?.find((part) => part.title?.trim())?.title?.trim();
    return Boolean(title && !isNoisyPhaseTitle(title));
  });

  for (const [index, concept] of concepts.entries()) {
    const known = knownById.get(concept.concept_id);
    const title = hasCourseParts
      ? phaseTitleForKnownConcept(known)
      : genericPhaseTitle(index, concepts.length);
    const key = title.toLocaleLowerCase();
    const existing = groups.get(key) ?? {
      title,
      concepts: [],
      order: groups.size,
    };
    existing.concepts.push(concept);
    groups.set(key, existing);
  }

  let rows = Array.from(groups.values()).map((group) => {
    const mastery =
      group.concepts.reduce((sum, concept) => sum + concept.mastery, 0) /
      Math.max(1, group.concepts.length);
    const phaseId = `synthetic-phase-${group.order}`;
    return {
      phase: {
        phase_id: phaseId,
        title: group.title,
        mastery,
        objectives_total: group.concepts.length,
        objectives_mastered: group.concepts.filter((concept) => concept.mastery >= 0.7).length,
        struggle_evidence: group.concepts.reduce(
          (sum, concept) => sum + concept.struggle_evidence,
          0,
        ),
        order_index: group.order,
      },
      synthetic: true,
      objectives: group.concepts.map((concept, index) => ({
        objective_id: `${phaseId}-objective-${index}`,
        phase_id: phaseId,
        objective_text: `Understand ${concept.name}`,
        bloom_level: knownById.get(concept.concept_id)?.bloom_level ?? "understand",
        mastery: concept.mastery,
        encounters: concept.encounters,
        struggle_evidence: concept.struggle_evidence,
        concept_ids: [concept.concept_id],
        order_index: index,
        concepts: [concept],
      })),
    };
  });

  if (view === "progressed") {
    rows = rows.sort((a, b) => b.phase.mastery - a.phase.mastery || a.phase.order_index - b.phase.order_index);
  } else if (view === "weakest") {
    rows = rows.sort((a, b) => a.phase.mastery - b.phase.mastery || a.phase.order_index - b.phase.order_index);
  } else {
    rows = rows.sort((a, b) => a.phase.order_index - b.phase.order_index);
  }
  return rows;
}

function phaseTitleForKnownConcept(concept: KnownConcept | undefined): string {
  const title = concept?.course_parts?.find((part) => part.title?.trim())?.title?.trim();
  if (title && !isNoisyPhaseTitle(title)) return title;
  return "Course Learning Path";
}

function genericPhaseTitle(index: number, total: number): string {
  const titles = [
    "Course Foundations",
    "Core Concepts",
    "Methods and Processes",
    "Practice and Applications",
    "Evaluation and Review",
  ];
  const phaseCount = Math.min(titles.length, Math.max(1, Math.ceil(total / 12)));
  const bucketSize = Math.ceil(total / phaseCount);
  return titles[Math.min(titles.length - 1, Math.floor(index / bucketSize))] ?? "Course Foundations";
}

function isNoisyPhaseTitle(title: string): boolean {
  const normalized = title.toLocaleLowerCase();
  return (
    normalized.length > 140 ||
    /<\/?\w+|\\begin|\\frac|\$|^\d+(?:\.\d+)*$/.test(normalized) ||
    /^(introduction|conclusion|summary|overview|agenda|plan|course|cours)$/.test(normalized)
  );
}

function ProgressViewToggle({
  value,
  onChange,
}: {
  value: ProgressView;
  onChange: (value: ProgressView) => void;
}) {
  const items: Array<{ value: ProgressView; label: string }> = [
    { value: "course", label: "Course" },
    { value: "progressed", label: "Most" },
    { value: "weakest", label: "Weakest" },
  ];
  return (
    <div className="grid grid-cols-3 overflow-hidden rounded-md border border-border text-xs">
      {items.map((item) => (
        <button
          key={item.value}
          type="button"
          onClick={() => onChange(item.value)}
          className={cn(
            "h-8 border-r border-border px-2 last:border-r-0",
            value === item.value
              ? "bg-primary text-primary-foreground"
              : "bg-surface text-muted-foreground hover:bg-muted hover:text-foreground",
          )}
        >
          {item.label}
        </button>
      ))}
    </div>
  );
}

function PhaseMap({
  rows,
  expanded,
  pending,
  onToggle,
  onCheckPhase,
  onCheckObjective,
  onCheckConcept,
}: {
  rows: PhaseRow[];
  expanded: Record<string, boolean>;
  pending: boolean;
  onToggle: (phaseId: string) => void;
  onCheckPhase: (phaseId: UUID) => void;
  onCheckObjective: (objectiveId: UUID) => void;
  onCheckConcept: (conceptId: UUID) => void;
}) {
  return (
    <ul className="flex flex-col gap-3">
      {rows.map((row) => {
        const isExpanded = expanded[row.phase.phase_id] ?? false;
        return (
          <li key={row.phase.phase_id} className="flex flex-col gap-2 rounded-md border border-border bg-surface p-3">
            <div className="flex items-start justify-between gap-2">
              <button
                type="button"
                className="flex min-w-0 items-center gap-1 text-left"
                onClick={() => onToggle(row.phase.phase_id)}
                aria-label={isExpanded ? "Collapse phase" : "Expand phase"}
              >
                <ChevronDown
                  className={cn(
                    "h-3.5 w-3.5 shrink-0 text-muted-foreground transition-transform",
                    !isExpanded && "-rotate-90",
                  )}
                />
                <span className="line-clamp-2 text-xs font-medium">{row.phase.title}</span>
              </button>
              <div className="flex shrink-0 items-center gap-1.5">
                <span className="tabular-nums text-xs text-muted-foreground">
                  {Math.round(row.phase.mastery * 100)}%
                </span>
                <button
                  type="button"
                  className="rounded p-1 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground disabled:opacity-50"
                  onClick={() => {
                    const firstConcept = row.objectives[0]?.concepts[0]?.concept_id;
                    if (row.synthetic && firstConcept) {
                      onCheckConcept(firstConcept);
                      return;
                    }
                    onCheckPhase(row.phase.phase_id);
                  }}
                  disabled={pending || (row.synthetic && !row.objectives[0]?.concepts[0])}
                  title={`Check ${row.phase.title}`}
                  aria-label={`Check ${row.phase.title}`}
                >
                  <ClipboardCheck className="h-3.5 w-3.5" />
                </button>
              </div>
            </div>
            <ProgressBar score={row.phase.mastery} />
            {isExpanded && (
              <ul className="flex flex-col gap-2 pt-1">
                {row.objectives.map((objective) => (
                  <li key={objective.objective_id} className="flex flex-col gap-1.5 border-l border-border pl-3">
                    <div className="flex items-start justify-between gap-2">
                      <div className="min-w-0">
                        <p className="line-clamp-2 text-xs">{objective.objective_text}</p>
                        <p className="mt-0.5 text-[11px] text-muted-foreground">
                          {Math.round(objective.mastery * 100)}% - {objective.bloom_level}
                        </p>
                      </div>
                      <button
                        type="button"
                        className="rounded p-1 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground disabled:opacity-50"
                        onClick={() => {
                          const firstConcept = objective.concepts[0]?.concept_id;
                          if (row.synthetic && firstConcept) {
                            onCheckConcept(firstConcept);
                            return;
                          }
                          onCheckObjective(objective.objective_id);
                        }}
                        disabled={pending}
                        title="Check objective"
                        aria-label="Check objective"
                      >
                        <ClipboardCheck className="h-3.5 w-3.5" />
                      </button>
                    </div>
                    <ProgressBar score={objective.mastery} compact />
                    {objective.concepts.length > 0 && (
                      <ul className="flex flex-wrap gap-1">
                        {objective.concepts.slice(0, 5).map((concept) => (
                          <li key={concept.concept_id}>
                            <button
                              type="button"
                              className="rounded border border-border px-1.5 py-0.5 text-[10px] text-muted-foreground hover:bg-muted hover:text-foreground disabled:opacity-50"
                              disabled={pending}
                              onClick={() => onCheckConcept(concept.concept_id)}
                            >
                              {concept.name}
                            </button>
                          </li>
                        ))}
                      </ul>
                    )}
                  </li>
                ))}
              </ul>
            )}
          </li>
        );
      })}
    </ul>
  );
}

function MasteryBars({
  ranked,
  pending,
  onCheck,
}: {
  ranked: { conceptId?: string; concept: string; score: number }[];
  pending: boolean;
  onCheck: (conceptId: UUID) => void;
}) {
  return (
    <ul className="flex flex-col gap-2">
      {ranked.map(({ conceptId, concept, score }) => (
        <li key={concept} className="flex flex-col gap-1">
          <div className="flex items-center justify-between gap-2 text-xs">
            <span className="truncate" title={concept}>
              {concept}
            </span>
            <div className="flex shrink-0 items-center gap-1.5">
              <span className="tabular-nums text-muted-foreground">
                {Math.round(score * 100)}%
              </span>
              {conceptId && (
                <button
                  type="button"
                  className="rounded p-1 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground disabled:opacity-50"
                  onClick={() => onCheck(conceptId)}
                  disabled={pending}
                  title={`Check ${concept}`}
                  aria-label={`Check ${concept}`}
                >
                  <ClipboardCheck className="h-3.5 w-3.5" />
                </button>
              )}
            </div>
          </div>
          <ProgressBar score={score} />
        </li>
      ))}
    </ul>
  );
}

function ProgressBar({ score, compact = false }: { score: number; compact?: boolean }) {
  return (
    <div className={cn("overflow-hidden rounded-full bg-muted", compact ? "h-1.5" : "h-2")}>
      <div
        className={cn("h-full rounded-full", masteryColor(score))}
        style={{ width: `${Math.max(4, score * 100)}%` }}
      />
    </div>
  );
}

function KnowledgeCheckCard({
  check,
  answer,
  pending,
  onAnswer,
  onSubmit,
}: {
  check: KnowledgeCheckQuestion;
  answer: string;
  pending: boolean;
  onAnswer: (value: string) => void;
  onSubmit: () => void;
}) {
  const objective = check.question_type === "mcq" || check.question_type === "true_false";
  const options = check.question_type === "true_false" ? ["True", "False"] : check.options;

  return (
    <section className="flex flex-col gap-3 rounded-md border border-border bg-surface p-3">
      <div className="flex items-center justify-between gap-2">
        <Badge variant="primary">{check.concept_name}</Badge>
        <Badge variant="muted">{check.bloom_level}</Badge>
      </div>
      <p className="text-sm leading-6">{check.prompt}</p>
      {objective ? (
        <div className="flex flex-col gap-1.5">
          {options.map((option) => (
            <button
              key={option}
              type="button"
              disabled={pending}
              onClick={() => onAnswer(option)}
              className={cn(
                "rounded-md border px-3 py-2 text-left text-xs transition-colors",
                answer === option
                  ? "border-primary bg-primary/10"
                  : "border-border hover:bg-muted",
              )}
            >
              {option}
            </button>
          ))}
        </div>
      ) : (
        <Input
          disabled={pending}
          value={answer}
          placeholder="Your answer"
          onChange={(event) => onAnswer(event.target.value)}
        />
      )}
      <div className="flex justify-end">
        <Button size="sm" onClick={onSubmit} disabled={!answer.trim() || pending}>
          <Send className="h-3.5 w-3.5" />
          Submit
        </Button>
      </div>
    </section>
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
