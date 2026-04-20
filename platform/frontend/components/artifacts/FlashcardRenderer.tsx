"use client";

import { useEffect, useMemo, useState } from "react";

import { CheckCircle2, ChevronLeft, ChevronRight, Sparkles } from "lucide-react";

import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import type { FlashcardPayload, UUID } from "@/lib/types";
import { cn } from "@/lib/utils";
import { useProgressStore } from "@/stores/progressStore";

interface Props {
  payload: FlashcardPayload;
  conversationId?: UUID;
}

export function FlashcardRenderer({ payload, conversationId }: Props) {
  const cards = payload.cards;
  const total = cards.length;
  const [index, setIndex] = useState(0);
  const [flipped, setFlipped] = useState(false);
  const [mastered, setMastered] = useState<Set<number>>(new Set());

  const applyOptimistic = useProgressStore((s) => s.applyOptimistic);
  const card = cards[index];

  useEffect(() => {
    setFlipped(false);
  }, [index]);

  const next = () => setIndex((i) => Math.min(i + 1, total - 1));
  const prev = () => setIndex((i) => Math.max(i - 1, 0));

  const masteredCount = mastered.size;
  const progress = useMemo(
    () => (total === 0 ? 0 : (masteredCount / total) * 100),
    [masteredCount, total],
  );

  const markMastered = () => {
    setMastered((prev) => {
      const next = new Set(prev);
      next.add(index);
      return next;
    });
    if (conversationId && card?.concept) {
      applyOptimistic(conversationId, {
        concepts_covered: [card.concept],
        concepts_demonstrated: [card.concept],
        concepts_struggled: [],
      });
    }
    if (index < total - 1) {
      // small delay so the user sees the flip state settle before advancing
      setTimeout(() => setIndex((i) => Math.min(i + 1, total - 1)), 180);
    }
  };

  if (total === 0 || !card) {
    return (
      <div className="text-xs text-muted-foreground">
        No flashcards were generated.
      </div>
    );
  }

  const isMastered = mastered.has(index);

  return (
    <div className="flex flex-col gap-3">
      <header className="flex items-center justify-between gap-2">
        <div>
          <h3 className="text-sm font-semibold">{payload.title ?? "Flashcards"}</h3>
          <p className="text-[11px] text-muted-foreground">
            Card {index + 1} of {total} · {masteredCount} mastered
          </p>
        </div>
        <Badge variant="muted">Click card to flip</Badge>
      </header>

      <div className="h-1.5 overflow-hidden rounded-full bg-muted">
        <div
          className="h-full rounded-full bg-[hsl(var(--success))] transition-all"
          style={{ width: `${progress}%` }}
        />
      </div>

      <div
        role="button"
        tabIndex={0}
        aria-label={flipped ? "Back of flashcard" : "Front of flashcard"}
        onClick={() => setFlipped((f) => !f)}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            setFlipped((f) => !f);
          }
        }}
        className="relative aspect-[3/2] w-full cursor-pointer select-none outline-none"
        style={{ perspective: 1200 }}
      >
        <div
          className={cn(
            "absolute inset-0 preserve-3d transition-transform duration-300",
            flipped && "[transform:rotateY(180deg)]",
          )}
        >
          <CardFace side="front">{card.front}</CardFace>
          <CardFace side="back">{card.back}</CardFace>
        </div>
      </div>

      <div className="flex items-center justify-between gap-2">
        <Button
          variant="secondary"
          size="sm"
          onClick={prev}
          disabled={index === 0}
        >
          <ChevronLeft className="h-4 w-4" />
          Previous
        </Button>

        <Button
          variant={isMastered ? "secondary" : "primary"}
          size="sm"
          onClick={markMastered}
          disabled={isMastered}
        >
          {isMastered ? (
            <>
              <CheckCircle2 className="h-4 w-4" />
              Mastered
            </>
          ) : (
            <>
              <Sparkles className="h-4 w-4" />
              Mark as mastered
            </>
          )}
        </Button>

        <Button
          variant="secondary"
          size="sm"
          onClick={next}
          disabled={index === total - 1}
        >
          Next
          <ChevronRight className="h-4 w-4" />
        </Button>
      </div>
    </div>
  );
}

function CardFace({
  side,
  children,
}: {
  side: "front" | "back";
  children: React.ReactNode;
}) {
  return (
    <div
      className={cn(
        "flip-card-face absolute inset-0 flex items-center justify-center rounded-xl border border-border p-6 text-center text-lg font-medium shadow-sm",
        side === "front"
          ? "bg-surface text-surface-foreground"
          : "bg-primary/10 text-foreground [transform:rotateY(180deg)]",
      )}
    >
      <p className="leading-snug">{children}</p>
    </div>
  );
}
