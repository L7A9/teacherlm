import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { reviewTestsApi } from "@/lib/api";
import type { ReviewTestSubmitRequest, UUID } from "@/lib/types";
import { useProgressStore } from "@/stores/progressStore";
import {
  forcedLanguageToOptions,
  useSettingsStore,
} from "@/stores/settingsStore";

export function useReviewTestStatus(conversationId: UUID) {
  return useQuery({
    queryKey: ["review-tests", "status", conversationId],
    queryFn: () => reviewTestsApi.status(conversationId),
    refetchInterval: 15_000,
  });
}

export function useStartReviewTest(conversationId: UUID) {
  const setProgressState = useProgressStore((s) => s.setState);
  const forcedLanguage = useSettingsStore((s) => s.forcedLanguage);
  return useMutation({
    mutationFn: () =>
      reviewTestsApi.start(conversationId, {
        options: forcedLanguageToOptions(forcedLanguage),
      }),
    onSuccess: (data) => {
      setProgressState(conversationId, data.learner_state);
    },
  });
}

export function useSubmitReviewTest(conversationId: UUID, windowId: UUID | null) {
  const qc = useQueryClient();
  const setProgressState = useProgressStore((s) => s.setState);
  const forcedLanguage = useSettingsStore((s) => s.forcedLanguage);
  return useMutation({
    mutationFn: (body: Omit<ReviewTestSubmitRequest, "options">) => {
      if (!windowId) throw new Error("No review window is active.");
      return reviewTestsApi.submit(conversationId, windowId, {
        ...body,
        options: forcedLanguageToOptions(forcedLanguage),
      });
    },
    onSuccess: (data) => {
      setProgressState(conversationId, data.learner_state);
      qc.invalidateQueries({ queryKey: ["review-tests", "status", conversationId] });
      qc.invalidateQueries({
        queryKey: ["conversations", "learner-state", conversationId],
      });
    },
  });
}

export function useSnoozeReviewTest(conversationId: UUID) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (windowId: UUID) => reviewTestsApi.snooze(conversationId, windowId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["review-tests", "status", conversationId] });
    },
  });
}

export function useDismissReviewTest(conversationId: UUID) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (windowId: UUID) => reviewTestsApi.dismiss(conversationId, windowId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["review-tests", "status", conversationId] });
    },
  });
}
