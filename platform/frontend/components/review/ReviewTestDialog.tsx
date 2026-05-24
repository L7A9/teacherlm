"use client";

import { useEffect, useMemo, useState } from "react";

import { ClipboardCheck, Send } from "lucide-react";
import { toast } from "sonner";

import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/Dialog";
import { Input } from "@/components/ui/Input";
import {
  useDismissReviewTest,
  useReviewTestStatus,
  useSnoozeReviewTest,
  useStartReviewTest,
  useSubmitReviewTest,
} from "@/hooks/useReviewTests";
import type { KnowledgeCheckQuestion, UUID } from "@/lib/types";
import { cn } from "@/lib/utils";

interface Props {
  conversationId: UUID;
}

type AnswerMap = Record<string, string>;

export function ReviewTestDialog({ conversationId }: Props) {
  const status = useReviewTestStatus(conversationId);
  const startReview = useStartReviewTest(conversationId);
  const [open, setOpen] = useState(false);
  const [checks, setChecks] = useState<KnowledgeCheckQuestion[]>([]);
  const [answers, setAnswers] = useState<AnswerMap>({});
  const activeWindow = startReview.data?.window ?? status.data?.window ?? null;
  const submitReview = useSubmitReviewTest(conversationId, activeWindow?.id ?? null);
  const snoozeReview = useSnoozeReviewTest(conversationId);
  const dismissReview = useDismissReviewTest(conversationId);

  useEffect(() => {
    if (status.data?.due && status.data.window && checks.length === 0) {
      setOpen(true);
    }
  }, [checks.length, status.data?.due, status.data?.window]);

  const answeredCount = useMemo(
    () => checks.filter((check) => answers[check.id]?.trim()).length,
    [answers, checks],
  );
  const allAnswered = checks.length > 0 && answeredCount === checks.length;
  const pending =
    startReview.isPending ||
    submitReview.isPending ||
    snoozeReview.isPending ||
    dismissReview.isPending;

  const start = async () => {
    try {
      const response = await startReview.mutateAsync();
      setChecks(response.checks);
      setAnswers({});
      setOpen(true);
    } catch (err) {
      toast.error(`Review could not start: ${(err as Error).message}`);
    }
  };

  const later = async () => {
    if (!activeWindow) return setOpen(false);
    await snoozeReview.mutateAsync(activeWindow.id);
    setOpen(false);
  };

  const dismiss = async () => {
    if (!activeWindow) return setOpen(false);
    await dismissReview.mutateAsync(activeWindow.id);
    setChecks([]);
    setAnswers({});
    setOpen(false);
  };

  const submit = async () => {
    if (!activeWindow || !allAnswered) return;
    await submitReview.mutateAsync({
      answers: checks.map((check) => ({
        check_id: check.id,
        answer: answers[check.id],
      })),
    });
    setChecks([]);
    setAnswers({});
    setOpen(false);
  };

  if (!status.data?.window && checks.length === 0) return null;

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <ClipboardCheck className="h-4 w-4" />
            Quick review
          </DialogTitle>
          <DialogDescription>
            {checks.length === 0
              ? "A review is ready from your recent answered course questions."
              : `${answeredCount}/${checks.length} answered`}
          </DialogDescription>
        </DialogHeader>

        {checks.length === 0 ? (
          <div className="rounded-md border border-border bg-muted p-3 text-sm">
            This review focuses on the concepts you discussed in the last 10
            answered course questions.
          </div>
        ) : (
          <ol className="max-h-[60vh] space-y-3 overflow-y-auto pr-1">
            {checks.map((check, index) => (
              <ReviewQuestion
                key={check.id}
                index={index}
                check={check}
                answer={answers[check.id] ?? ""}
                disabled={pending}
                onAnswer={(answer) =>
                  setAnswers((current) => ({ ...current, [check.id]: answer }))
                }
              />
            ))}
          </ol>
        )}

        <DialogFooter>
          {checks.length === 0 ? (
            <>
              <Button variant="ghost" onClick={() => void dismiss()} disabled={pending}>
                Dismiss
              </Button>
              <Button variant="secondary" onClick={() => void later()} disabled={pending}>
                Later
              </Button>
              <Button onClick={() => void start()} disabled={pending}>
                Start
              </Button>
            </>
          ) : (
            <>
              <Button variant="secondary" onClick={() => void later()} disabled={pending}>
                Later
              </Button>
              <Button onClick={() => void submit()} disabled={!allAnswered || pending}>
                <Send className="h-3.5 w-3.5" />
                Submit review
              </Button>
            </>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function ReviewQuestion({
  index,
  check,
  answer,
  disabled,
  onAnswer,
}: {
  index: number;
  check: KnowledgeCheckQuestion;
  answer: string;
  disabled: boolean;
  onAnswer: (answer: string) => void;
}) {
  const objective = check.question_type === "mcq" || check.question_type === "true_false";
  const options = check.question_type === "true_false" ? ["True", "False"] : check.options;

  return (
    <li className="rounded-md border border-border bg-surface p-3">
      <div className="mb-2 flex items-center justify-between gap-2">
        <Badge variant="muted">Q{index + 1}</Badge>
        <Badge variant="primary">{check.bloom_level}</Badge>
      </div>
      <p className="mb-3 text-sm font-medium leading-6">{check.prompt}</p>
      {objective ? (
        <div className="flex flex-col gap-1.5">
          {options.map((option) => (
            <button
              key={option}
              type="button"
              disabled={disabled}
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
          disabled={disabled}
          value={answer}
          placeholder="Your answer"
          onChange={(event) => onAnswer(event.target.value)}
        />
      )}
    </li>
  );
}
