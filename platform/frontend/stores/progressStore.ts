import { create } from "zustand";

import type { LearnerState, LearnerUpdates, UUID } from "@/lib/types";

// Local mirror of LearnerState. Canonical state lives server-side; hooks sync
// this store after chat/generate `done` events and on conversation load.

const DEMONSTRATE_STEP = 0.2;
const STRUGGLE_DECAY = 0.7;
const UNDERSTOOD_THRESHOLD = 0.7;
const STRUGGLING_THRESHOLD = 0.3;

function emptyState(conversationId: UUID): LearnerState {
  return {
    conversation_id: conversationId,
    understood_concepts: [],
    struggling_concepts: [],
    mastery_scores: {},
    session_turns: 0,
    turns_since_progress: 0,
  };
}

interface ProgressState {
  stateByConversation: Record<UUID, LearnerState>;

  setState: (conversationId: UUID, state: LearnerState) => void;
  applyOptimistic: (conversationId: UUID, updates: LearnerUpdates) => void;
  reset: (conversationId: UUID) => void;
  get: (conversationId: UUID) => LearnerState;
}

export const useProgressStore = create<ProgressState>((set, get) => ({
  stateByConversation: {},

  setState: (conversationId, state) =>
    set((prev) => ({
      stateByConversation: {
        ...prev.stateByConversation,
        [conversationId]: state,
      },
    })),

  applyOptimistic: (conversationId, updates) =>
    set((prev) => {
      const current =
        prev.stateByConversation[conversationId] ?? emptyState(conversationId);
      const mastery = { ...current.mastery_scores };

      for (const concept of updates.concepts_covered) {
        mastery[concept] = mastery[concept] ?? 0;
      }
      for (const concept of updates.concepts_demonstrated) {
        const v = mastery[concept] ?? 0;
        mastery[concept] = Math.min(1, v + DEMONSTRATE_STEP * (1 - v));
      }
      for (const concept of updates.concepts_struggled) {
        const v = mastery[concept] ?? 0;
        mastery[concept] = Math.max(0, v * STRUGGLE_DECAY);
      }

      const understood = Object.entries(mastery)
        .filter(([, v]) => v >= UNDERSTOOD_THRESHOLD)
        .map(([k]) => k)
        .sort();
      const struggling = Object.entries(mastery)
        .filter(([, v]) => v <= STRUGGLING_THRESHOLD)
        .map(([k]) => k)
        .sort();

      return {
        stateByConversation: {
          ...prev.stateByConversation,
          [conversationId]: {
            ...current,
            mastery_scores: mastery,
            understood_concepts: understood,
            struggling_concepts: struggling,
            session_turns: current.session_turns + 1,
          },
        },
      };
    }),

  reset: (conversationId) =>
    set((prev) => ({
      stateByConversation: {
        ...prev.stateByConversation,
        [conversationId]: emptyState(conversationId),
      },
    })),

  get: (conversationId) =>
    get().stateByConversation[conversationId] ?? emptyState(conversationId),
}));
