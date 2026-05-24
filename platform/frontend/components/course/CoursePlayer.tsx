"use client";

import { useEffect, useMemo, useState } from "react";

import {
  BookOpen,
  CheckCircle2,
  ChevronRight,
  Lock,
  PlayCircle,
  Route,
  Send,
  Unlock,
} from "lucide-react";

import { AssistantMarkdown } from "@/components/chat/MessageBubble";
import { LearnerProgress } from "@/components/progress/LearnerProgress";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import {
  useCoursePlayer,
  useSubmitChapterQuiz,
  useUnlockCourseChapter,
} from "@/hooks/useCoursePlayer";
import type { CourseChapter, KnowledgeCheckQuestion, RemediationPath, UUID } from "@/lib/types";
import { cn } from "@/lib/utils";

interface Props {
  conversationId: UUID;
}

type AnswerMap = Record<string, string>;

export function CoursePlayer({ conversationId }: Props) {
  const { data, isLoading } = useCoursePlayer(conversationId);
  const [activeChapterId, setActiveChapterId] = useState<UUID | null>(null);
  const [answers, setAnswers] = useState<AnswerMap>({});
  const [remediationPaths, setRemediationPaths] = useState<RemediationPath[]>([]);
  const chapters = useMemo(() => data?.chapters ?? [], [data?.chapters]);
  const unlock = useUnlockCourseChapter(conversationId);
  const activeChapter = useMemo(
    () =>
      chapters.find((chapter) => chapter.id === activeChapterId) ??
      chapters.find((chapter) => chapter.state === "available") ??
      chapters[0] ??
      null,
    [activeChapterId, chapters],
  );
  const submitQuiz = useSubmitChapterQuiz(conversationId, activeChapter?.id ?? null);

  useEffect(() => {
    if (!activeChapterId && activeChapter) setActiveChapterId(activeChapter.id);
  }, [activeChapter, activeChapterId]);

  useEffect(() => {
    setAnswers({});
    setRemediationPaths([]);
  }, [activeChapter?.id]);

  if (isLoading) {
    return (
      <div className="px-4 py-4 text-xs text-muted-foreground">
        Building your course player...
      </div>
    );
  }

  if (data?.course_status === "waiting_for_files") {
    const pending = data.pending_file_count ?? 0;
    const total = data.total_file_count ?? 0;
    return (
      <div className="flex flex-col gap-4 px-4 py-4">
        <section className="rounded-md border border-border bg-surface p-3">
          <div className="flex items-start gap-2">
            <BookOpen className="mt-0.5 h-4 w-4 shrink-0 text-primary" />
            <div className="min-w-0">
              <h3 className="text-sm font-semibold">Course path is being prepared</h3>
              <p className="mt-1 text-xs leading-5 text-muted-foreground">
                Chapters will appear after every uploaded file is ready, then the course will be
                structured from fundamentals to the final topics.
              </p>
              {total > 0 && (
                <p className="mt-2 text-[11px] text-muted-foreground">
                  {pending} of {total} files are not ready yet.
                </p>
              )}
            </div>
          </div>
        </section>
        <LearnerProgress conversationId={conversationId} />
      </div>
    );
  }

  if (!data || chapters.length === 0) {
    return (
      <div className="flex flex-col gap-4 px-4 py-4">
        <p className="text-xs text-muted-foreground">
          Upload course files first. After processing, your guided course will
          appear here.
        </p>
        <LearnerProgress conversationId={conversationId} />
      </div>
    );
  }

  const submit = async () => {
    if (!activeChapter?.quiz) return;
    const result = await submitQuiz.mutateAsync({
      answers: activeChapter.quiz.questions.map((question) => ({
        check_id: question.id,
        answer: answers[question.id],
      })),
    });
    setRemediationPaths(
      result.results.flatMap((item) => item.remediation_paths ?? []),
    );
  };

  return (
    <div className="flex flex-col gap-4 px-4 py-4">
      <section className="flex flex-col gap-2">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold">Course path</h3>
          <Badge variant="muted">{chapters.length} chapters</Badge>
        </div>
        <ol className="flex flex-col gap-2">
          {chapters.map((chapter) => (
            <ChapterRow
              key={chapter.id}
              chapter={chapter}
              active={chapter.id === activeChapter?.id}
              pending={unlock.isPending}
              onOpen={() => setActiveChapterId(chapter.id)}
              onUnlock={() => void unlock.mutateAsync(chapter.id)}
            />
          ))}
        </ol>
      </section>

      {activeChapter && (
        <section className="flex flex-col gap-4 border-t border-border pt-4">
          <ChapterDetail
            chapter={activeChapter}
            answers={answers}
            remediationPaths={remediationPaths}
            pending={submitQuiz.isPending}
            onAnswer={(checkId, answer) =>
              setAnswers((current) => ({ ...current, [checkId]: answer }))
            }
            onSubmit={() => void submit()}
          />
        </section>
      )}

      <section className="border-t border-border pt-4">
        <LearnerProgress conversationId={conversationId} />
      </section>
    </div>
  );
}

function ChapterRow({
  chapter,
  active,
  pending,
  onOpen,
  onUnlock,
}: {
  chapter: CourseChapter;
  active: boolean;
  pending: boolean;
  onOpen: () => void;
  onUnlock: () => void;
}) {
  const locked = chapter.state === "locked";
  return (
    <li className={cn("rounded-md border border-border bg-surface", active && "border-primary/60")}>
      <button
        type="button"
        className="flex w-full items-start gap-2 px-3 py-2 text-left"
        onClick={onOpen}
      >
        <StatusIcon state={chapter.state} />
        <div className="min-w-0 flex-1">
          <p className="line-clamp-2 text-xs font-medium">{chapter.title}</p>
          <div className="mt-1 flex items-center gap-2 text-[11px] text-muted-foreground">
            <span>{Math.round(chapter.progress * 100)}%</span>
            {chapter.attempts > 0 && <span>best {Math.round(chapter.best_score * 100)}%</span>}
          </div>
          <ProgressBar score={chapter.progress} />
        </div>
        <ChevronRight className="mt-0.5 h-3.5 w-3.5 shrink-0 text-muted-foreground" />
      </button>
      {locked && (
        <div className="border-t border-border px-3 py-2">
          <Button
            variant="secondary"
            size="sm"
            onClick={onUnlock}
            disabled={pending}
            className="w-full"
          >
            <Unlock className="h-3.5 w-3.5" />
            Continue anyway
          </Button>
        </div>
      )}
    </li>
  );
}

function ChapterDetail({
  chapter,
  answers,
  remediationPaths,
  pending,
  onAnswer,
  onSubmit,
}: {
  chapter: CourseChapter;
  answers: AnswerMap;
  remediationPaths: RemediationPath[];
  pending: boolean;
  onAnswer: (checkId: UUID, answer: string) => void;
  onSubmit: () => void;
}) {
  const questions = chapter.quiz?.questions ?? [];
  const allAnswered =
    questions.length > 0 && questions.every((question) => answers[question.id]?.trim());
  return (
    <>
      <header className="flex flex-col gap-2">
        <div className="flex items-start gap-2">
          <BookOpen className="mt-0.5 h-4 w-4 shrink-0 text-primary" />
          <div className="min-w-0">
            <h3 className="text-sm font-semibold">{chapter.title}</h3>
            <div className="course-markdown mt-1 text-xs leading-5 text-muted-foreground">
              <AssistantMarkdown content={chapter.summary} />
            </div>
          </div>
        </div>
      </header>

      <div className="flex flex-col gap-3">
        {chapter.lessons.map((lesson) => (
          <article key={lesson.id} className="flex flex-col gap-2 rounded-md border border-border bg-surface p-3">
            <h4 className="text-xs font-semibold">{lesson.title}</h4>
            <GraphHints hints={lesson.graph_hints} />
            {lesson.blocks.map((block) => (
              <div key={block.id} className="flex flex-col gap-1">
                <Badge variant="muted" className="w-fit capitalize">
                  {block.block_type}
                </Badge>
                {block.title && <p className="text-xs font-medium">{block.title}</p>}
                <div className="course-markdown text-xs leading-5 text-muted-foreground">
                  <AssistantMarkdown content={block.content} />
                </div>
              </div>
            ))}
          </article>
        ))}
      </div>

      {questions.length > 0 && (
        <section className="flex flex-col gap-3 rounded-md border border-border bg-surface p-3">
          <div className="flex items-center justify-between">
            <h4 className="text-xs font-semibold">Chapter quiz</h4>
            <Badge variant={chapter.best_score >= 0.7 ? "success" : "muted"}>
              pass at {Math.round((chapter.quiz?.pass_score ?? 0.7) * 100)}%
            </Badge>
          </div>
          {questions.map((question, index) => (
            <QuizQuestion
              key={question.id}
              index={index}
              question={question}
              answer={answers[question.id] ?? ""}
              disabled={pending}
              onAnswer={(answer) => onAnswer(question.id, answer)}
            />
          ))}
          <Button onClick={onSubmit} disabled={!allAnswered || pending}>
            <Send className="h-3.5 w-3.5" />
            Submit quiz
          </Button>
          <RemediationHints paths={remediationPaths} />
        </section>
      )}
    </>
  );
}

function GraphHints({ hints }: { hints?: Record<string, unknown> }) {
  const prerequisites = stringList(hints?.prerequisites);
  const next = stringList(hints?.next);
  const examples = stringList(hints?.related_examples);
  if (prerequisites.length === 0 && next.length === 0 && examples.length === 0) return null;
  return (
    <div className="rounded-md border border-border bg-muted/30 p-2 text-[11px] leading-5 text-muted-foreground">
      <div className="mb-1 flex items-center gap-1.5 font-medium text-foreground">
        <Route className="h-3.5 w-3.5 text-primary" />
        Why this next?
      </div>
      {prerequisites.length > 0 && (
        <p>Review first: {prerequisites.slice(0, 3).join(", ")}</p>
      )}
      {next.length > 0 && <p>This helps with: {next.slice(0, 3).join(", ")}</p>}
      {examples.length > 0 && <p>Example/formula: {examples.slice(0, 2).join(" | ")}</p>}
    </div>
  );
}

function RemediationHints({ paths }: { paths: RemediationPath[] }) {
  const steps = paths.flatMap((path) => path.steps ?? []).slice(0, 4);
  if (steps.length === 0) return null;
  return (
    <div className="rounded-md border border-warning/40 bg-warning/10 p-2 text-[11px] leading-5">
      <div className="font-medium text-foreground">Review first</div>
      {steps.map((step) => (
        <p key={`${step.concept_id ?? step.concept_name}-${step.reason}`} className="text-muted-foreground">
          {step.concept_name}: {step.reason}
        </p>
      ))}
    </div>
  );
}

function stringList(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
}

function QuizQuestion({
  index,
  question,
  answer,
  disabled,
  onAnswer,
}: {
  index: number;
  question: KnowledgeCheckQuestion;
  answer: string;
  disabled: boolean;
  onAnswer: (answer: string) => void;
}) {
  const objective = question.question_type === "mcq" || question.question_type === "true_false";
  const options = question.question_type === "true_false" ? ["True", "False"] : question.options;
  return (
    <div className="flex flex-col gap-2 border-t border-border pt-3 first:border-t-0 first:pt-0">
      <p className="text-xs font-medium">Q{index + 1}. {question.prompt}</p>
      {objective ? (
        <div className="flex flex-col gap-1">
          {options.map((option) => (
            <button
              key={option}
              type="button"
              disabled={disabled}
              onClick={() => onAnswer(option)}
              className={cn(
                "rounded-md border px-2 py-1.5 text-left text-xs",
                answer === option ? "border-primary bg-primary/10" : "border-border hover:bg-muted",
              )}
            >
              {option}
            </button>
          ))}
        </div>
      ) : (
        <Input
          disabled={disabled}
          value={answer}
          placeholder="Your answer"
          onChange={(event) => onAnswer(event.target.value)}
        />
      )}
    </div>
  );
}

function StatusIcon({ state }: { state: CourseChapter["state"] }) {
  if (state === "completed") return <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-[hsl(var(--success))]" />;
  if (state === "locked") return <Lock className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />;
  return <PlayCircle className="mt-0.5 h-4 w-4 shrink-0 text-primary" />;
}

function ProgressBar({ score }: { score: number }) {
  return (
    <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-muted">
      <div
        className="h-full rounded-full bg-primary transition-all"
        style={{ width: `${Math.max(4, Math.round(score * 100))}%` }}
      />
    </div>
  );
}
