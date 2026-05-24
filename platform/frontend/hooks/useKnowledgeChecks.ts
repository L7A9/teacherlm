import { useMutation, useQueryClient } from "@tanstack/react-query";

import { knowledgeChecksApi } from "@/lib/api";
import type {
  KnowledgeCheckStartRequest,
  KnowledgeCheckSubmitRequest,
  QuizAttemptRequest,
  UUID,
} from "@/lib/types";
import { useProgressStore } from "@/stores/progressStore";
import {
  modelSettingsToOptions,
  useSettingsStore,
} from "@/stores/settingsStore";

export function useStartKnowledgeCheck(conversationId: UUID) {
  const setProgressState = useProgressStore((s) => s.setState);
  const modelSettings = useSettingsStore((s) => s.modelSettings);
  const forcedLanguage = useSettingsStore((s) => s.forcedLanguage);
  return useMutation({
    mutationFn: (body: KnowledgeCheckStartRequest = {}) =>
      knowledgeChecksApi.start(conversationId, {
        ...body,
        options: {
          ...modelSettingsToOptions(modelSettings),
          ...(forcedLanguage ? { language: forcedLanguage } : {}),
        },
      }),
    onSuccess: (data) => {
      setProgressState(conversationId, data.learner_state);
    },
  });
}

export function useSubmitKnowledgeCheck(conversationId: UUID) {
  const qc = useQueryClient();
  const setProgressState = useProgressStore((s) => s.setState);
  const modelSettings = useSettingsStore((s) => s.modelSettings);
  const forcedLanguage = useSettingsStore((s) => s.forcedLanguage);
  return useMutation({
    mutationFn: ({
      checkId,
      body,
    }: {
      checkId: UUID;
      body: KnowledgeCheckSubmitRequest;
    }) =>
      knowledgeChecksApi.submit(conversationId, checkId, {
        ...body,
        options: {
          ...modelSettingsToOptions(modelSettings),
          ...(forcedLanguage ? { language: forcedLanguage } : {}),
        },
      }),
    onSuccess: (data) => {
      setProgressState(conversationId, data.learner_state);
      qc.invalidateQueries({
        queryKey: ["conversations", "learner-state", conversationId],
      });
    },
  });
}

export function useSubmitQuizAttempt(conversationId: UUID | undefined) {
  const qc = useQueryClient();
  const setProgressState = useProgressStore((s) => s.setState);
  const modelSettings = useSettingsStore((s) => s.modelSettings);
  const forcedLanguage = useSettingsStore((s) => s.forcedLanguage);
  return useMutation({
    mutationFn: (body: QuizAttemptRequest) => {
      if (!conversationId) {
        throw new Error("No conversation is selected.");
      }
      return knowledgeChecksApi.submitQuiz(conversationId, {
        ...body,
        options: {
          ...modelSettingsToOptions(modelSettings),
          ...(forcedLanguage ? { language: forcedLanguage } : {}),
        },
      });
    },
    onSuccess: (data) => {
      if (!conversationId) return;
      setProgressState(conversationId, data.learner_state);
      qc.invalidateQueries({
        queryKey: ["conversations", "learner-state", conversationId],
      });
    },
  });
}
