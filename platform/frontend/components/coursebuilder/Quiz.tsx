"use client";

import { useMemo, useState } from "react";

import { CheckCircle2, Send, XCircle } from "lucide-react";

import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { useSubmitCourseBuilderQuiz } from "@/hooks/useCourseBuilder";
import type {
  CourseBuilderQuiz as CourseBuilderQuizType,
  CourseBuilderQuizResult,
  UUID,
} from "@/lib/types";
import { cn } from "@/lib/utils";

interface Props {
  conversationId: UUID;
  chapterId: UUID;
  quiz: CourseBuilderQuizType;
}

type AnswerMap = Record<string, number>;

export function CourseBuilderQuiz({ conversationId, chapterId, quiz }: Props) {
  const [answers, setAnswers] = useState<AnswerMap>({});
  const [results, setResults] = useState<CourseBuilderQuizResult[]>([]);
  const submit = useSubmitCourseBuilderQuiz(conversationId, chapterId);
  const resultByQuestion = useMemo(
    () => new Map(results.map((result) => [result.question_id, result])),
    [results],
  );
  const allAnswered =
    quiz.questions.length > 0 &&
    quiz.questions.every((question) => answers[question.id] !== undefined);

  const onSubmit = async () => {
    const response = await submit.mutateAsync({
      answers: quiz.questions.map((question) => ({
        question_id: question.id,
        answer: answers[question.id] ?? "",
      })),
    });
    setResults(response.results);
  };

  return (
    <section className="flex flex-col gap-3 rounded-md border border-border bg-background p-3">
      <div className="flex items-center justify-between gap-2">
        <h4 className="text-xs font-semibold">Chapter quiz</h4>
        <Badge variant="muted">Pass {Math.round(quiz.pass_score * 100)}%</Badge>
      </div>
      <div className="flex flex-col gap-3">
        {quiz.questions.map((question, index) => {
          const result = resultByQuestion.get(question.id);
          return (
            <div key={question.id} className="flex flex-col gap-2">
              <p className="text-xs font-medium leading-5">
                {index + 1}. {question.prompt}
              </p>
              <div className="grid gap-1">
                {question.options.map((option, optionIndex) => (
                  <button
                    key={`${question.id}-${optionIndex}`}
                    type="button"
                    className={cn(
                      "rounded-md border border-border px-2 py-1.5 text-left text-xs leading-4 transition-colors",
                      answers[question.id] === optionIndex && "border-primary bg-primary/10",
                      result?.correct_index === optionIndex && "border-[hsl(var(--success))] bg-[hsl(var(--success)/0.12)]",
                      result &&
                        result.selected_index === optionIndex &&
                        !result.is_correct &&
                        "border-danger bg-danger/10",
                    )}
                    disabled={submit.isPending}
                    onClick={() =>
                      setAnswers((current) => ({
                        ...current,
                        [question.id]: optionIndex,
                      }))
                    }
                  >
                    {option}
                  </button>
                ))}
              </div>
              {result && (
                <div
                  className={cn(
                    "flex items-start gap-1.5 rounded-md px-2 py-1.5 text-[11px] leading-4",
                    result.is_correct
                      ? "bg-[hsl(var(--success)/0.12)] text-[hsl(var(--success))]"
                      : "bg-danger/10 text-danger",
                  )}
                >
                  {result.is_correct ? (
                    <CheckCircle2 className="mt-0.5 h-3 w-3 shrink-0" />
                  ) : (
                    <XCircle className="mt-0.5 h-3 w-3 shrink-0" />
                  )}
                  <span>{result.feedback}</span>
                </div>
              )}
            </div>
          );
        })}
      </div>
      <Button
        size="sm"
        onClick={() => void onSubmit()}
        disabled={!allAnswered || submit.isPending}
      >
        <Send className="h-3.5 w-3.5" />
        Submit quiz
      </Button>
      {submit.data && (
        <div className="rounded-md bg-muted px-2 py-1.5 text-xs">
          Score {Math.round(submit.data.score * 100)}%
          {submit.data.passed ? " - chapter passed" : " - review and try again"}
        </div>
      )}
    </section>
  );
}
