"use client";

import { useMemo, useState } from "react";

import { CheckCircle2, RotateCcw, XCircle } from "lucide-react";

import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import type { QuizPayload, QuizQuestion } from "@/lib/types";
import { cn } from "@/lib/utils";

interface Props {
  payload: QuizPayload;
}

type AnswerMap = Record<number, string>;

export function QuizRenderer({ payload }: Props) {
  const { title, questions } = payload;
  const [answers, setAnswers] = useState<AnswerMap>({});
  const [submitted, setSubmitted] = useState(false);

  const answeredCount = Object.keys(answers).length;
  const total = questions.length;
  const progress = total === 0 ? 0 : (answeredCount / total) * 100;
  const score = useMemo(
    () =>
      submitted
        ? questions.reduce(
            (sum, q, idx) => sum + (isCorrect(q, answers[idx]) ? 1 : 0),
            0,
          )
        : 0,
    [submitted, questions, answers],
  );

  const reset = () => {
    setAnswers({});
    setSubmitted(false);
  };

  if (total === 0) {
    return (
      <div className="text-xs text-muted-foreground">
        The quiz didn't contain any questions.
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      <header className="flex items-center justify-between gap-2">
        <div>
          <h3 className="text-sm font-semibold">{title ?? "Quiz"}</h3>
          <p className="text-[11px] text-muted-foreground">
            {submitted
              ? `You scored ${score}/${total}`
              : `${answeredCount}/${total} answered`}
          </p>
        </div>
        {submitted && (
          <Button variant="secondary" size="sm" onClick={reset}>
            <RotateCcw className="h-3.5 w-3.5" />
            Try again
          </Button>
        )}
      </header>

      <div className="h-1.5 overflow-hidden rounded-full bg-muted">
        <div
          className="h-full rounded-full bg-primary transition-all"
          style={{ width: `${submitted ? 100 : progress}%` }}
        />
      </div>

      <ol className="flex flex-col gap-4">
        {questions.map((q, idx) => (
          <QuestionCard
            key={idx}
            index={idx}
            question={q}
            answer={answers[idx]}
            submitted={submitted}
            onAnswer={(value) =>
              setAnswers((prev) => ({ ...prev, [idx]: value }))
            }
          />
        ))}
      </ol>

      {!submitted && (
        <div className="flex justify-end">
          <Button
            disabled={answeredCount < total}
            onClick={() => setSubmitted(true)}
          >
            Submit quiz
          </Button>
        </div>
      )}
    </div>
  );
}

function QuestionCard({
  index,
  question,
  answer,
  submitted,
  onAnswer,
}: {
  index: number;
  question: QuizQuestion;
  answer?: string;
  submitted: boolean;
  onAnswer: (value: string) => void;
}) {
  const options = resolveOptions(question);
  const correct = isCorrect(question, answer);
  const correctLabel = correctAnswerLabel(question);

  return (
    <li className="rounded-lg border border-border bg-surface p-4">
      <div className="mb-3 flex items-start gap-2">
        <span className="text-xs font-semibold text-muted-foreground">
          Q{index + 1}
        </span>
        <p className="text-sm font-medium">{question.question}</p>
      </div>

      {options ? (
        <ul className="flex flex-col gap-1.5">
          {options.map((opt) => {
            const selected = answer === opt;
            const isAnswer = correctLabel === opt;
            const state = !submitted
              ? selected
                ? "selected"
                : "idle"
              : isAnswer
                ? "correct"
                : selected
                  ? "wrong"
                  : "idle";
            return (
              <li key={opt}>
                <button
                  type="button"
                  disabled={submitted}
                  onClick={() => onAnswer(opt)}
                  className={cn(
                    "w-full rounded-md border px-3 py-2 text-left text-sm transition-colors",
                    state === "idle" && "border-border hover:bg-muted",
                    state === "selected" && "border-primary bg-primary/10",
                    state === "correct" &&
                      "border-[hsl(var(--success))] bg-[hsl(var(--success)/0.15)]",
                    state === "wrong" &&
                      "border-[hsl(var(--danger))] bg-[hsl(var(--danger)/0.15)]",
                  )}
                >
                  {opt}
                </button>
              </li>
            );
          })}
        </ul>
      ) : (
        <Input
          disabled={submitted}
          placeholder="Type your answer"
          value={answer ?? ""}
          onChange={(e) => onAnswer(e.target.value)}
        />
      )}

      {submitted && (
        <div className="mt-3 flex flex-col gap-1.5 text-xs">
          <div className="flex items-center gap-1.5">
            {correct ? (
              <Badge variant="success">
                <CheckCircle2 className="h-3 w-3" />
                Correct
              </Badge>
            ) : (
              <Badge variant="danger">
                <XCircle className="h-3 w-3" />
                Correct: {correctLabel ?? "—"}
              </Badge>
            )}
          </div>
          {question.explanation && (
            <p className="text-muted-foreground">{question.explanation}</p>
          )}
        </div>
      )}
    </li>
  );
}

function resolveOptions(q: QuizQuestion): string[] | null {
  if (q.type === "mcq") return q.options ?? null;
  if (q.type === "true_false") return ["True", "False"];
  return null;
}

function correctAnswerLabel(q: QuizQuestion): string | null {
  if (q.type === "mcq") {
    const idx = q.correct_index;
    if (idx == null || !q.options) return null;
    return q.options[idx] ?? null;
  }
  if (q.type === "true_false") return q.answer ? "True" : "False";
  if (q.type === "fill_blank") return q.answer ?? null;
  return null;
}

function isCorrect(q: QuizQuestion, given: string | undefined): boolean {
  if (given === undefined) return false;
  const expected = correctAnswerLabel(q);
  if (expected == null) return false;
  const norm = (s: string) => s.trim().toLowerCase();
  if (norm(expected) === norm(given)) return true;
  if (q.type === "fill_blank") {
    const accepted = q.accepted_answers ?? [];
    return accepted.some((a) => norm(a) === norm(given));
  }
  return false;
}
