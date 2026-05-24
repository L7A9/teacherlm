import { useEffect } from "react";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { API_BASE_URL, courseBuilderApi } from "@/lib/api";
import type {
  CourseBuilderGenerateRequest,
  CourseBuilderQuizSubmitRequest,
  UUID,
} from "@/lib/types";
import {
  modelSettingsToOptions,
  useSettingsStore,
} from "@/stores/settingsStore";

const RUNNING_STATUSES = new Set([
  "queued",
  "analyzing",
  "generating_outline",
  "generating_chapters",
  "generating_lessons",
  "generating_quizzes",
  "validating",
]);

export function useCourseBuilder(conversationId: UUID) {
  return useQuery({
    queryKey: ["coursebuilder", conversationId],
    queryFn: () => courseBuilderApi.get(conversationId),
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      if (!status) return 3000;
      return RUNNING_STATUSES.has(status) ? 2500 : false;
    },
  });
}

export function useCourseBuilderProgress(conversationId: UUID, enabled: boolean) {
  const qc = useQueryClient();

  useEffect(() => {
    if (!enabled) return;
    const source = new EventSource(
      `${API_BASE_URL}/api/conversations/${conversationId}/coursebuilder/events`,
      { withCredentials: true },
    );
    const refresh = () => {
      void qc.invalidateQueries({ queryKey: ["coursebuilder", conversationId] });
    };
    source.addEventListener("snapshot", refresh);
    source.addEventListener("error", refresh);
    return () => source.close();
  }, [conversationId, enabled, qc]);
}

export function useGenerateCourseBuilder(conversationId: UUID) {
  const qc = useQueryClient();
  const request = useCourseBuilderRequest();
  return useMutation({
    mutationFn: () => courseBuilderApi.generate(conversationId, request),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["coursebuilder", conversationId] });
    },
  });
}

export function useRebuildCourseBuilder(conversationId: UUID) {
  const qc = useQueryClient();
  const request = useCourseBuilderRequest();
  return useMutation({
    mutationFn: () => courseBuilderApi.rebuild(conversationId, request),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["coursebuilder", conversationId] });
    },
  });
}

export function useSubmitCourseBuilderQuiz(
  conversationId: UUID,
  chapterId: UUID | null,
) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: CourseBuilderQuizSubmitRequest) => {
      if (!chapterId) throw new Error("No chapter is selected.");
      return courseBuilderApi.submitQuiz(conversationId, chapterId, body);
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["coursebuilder", conversationId] });
      qc.invalidateQueries({
        queryKey: ["conversations", "learner-state", conversationId],
      });
    },
  });
}

function useCourseBuilderRequest(): CourseBuilderGenerateRequest {
  const modelSettings = useSettingsStore((s) => s.modelSettings);
  const forcedLanguage = useSettingsStore((s) => s.forcedLanguage);
  return {
    options: {
      ...modelSettingsToOptions(modelSettings),
      ...(forcedLanguage ? { language: forcedLanguage } : {}),
    },
  };
}
