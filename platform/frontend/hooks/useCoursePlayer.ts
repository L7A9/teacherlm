import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { coursePlayerApi } from "@/lib/api";
import type { ChapterQuizSubmitRequest, UUID } from "@/lib/types";
import { useProgressStore } from "@/stores/progressStore";
import {
  modelSettingsToOptions,
  useSettingsStore,
} from "@/stores/settingsStore";

export function useCoursePlayer(conversationId: UUID) {
  const setProgressState = useProgressStore((s) => s.setState);
  return useQuery({
    queryKey: ["course-player", conversationId],
    queryFn: async () => {
      const data = await coursePlayerApi.get(conversationId);
      setProgressState(conversationId, data.learner_state);
      return data;
    },
    refetchInterval: (query) =>
      query.state.data?.course_status === "waiting_for_files" ? 3000 : false,
  });
}

export function useUnlockCourseChapter(conversationId: UUID) {
  const qc = useQueryClient();
  const setProgressState = useProgressStore((s) => s.setState);
  return useMutation({
    mutationFn: (chapterId: UUID) => coursePlayerApi.unlock(conversationId, chapterId),
    onSuccess: (data) => {
      setProgressState(conversationId, data.learner_state);
      qc.invalidateQueries({ queryKey: ["course-player", conversationId] });
    },
  });
}

export function useSubmitChapterQuiz(conversationId: UUID, chapterId: UUID | null) {
  const qc = useQueryClient();
  const setProgressState = useProgressStore((s) => s.setState);
  const modelSettings = useSettingsStore((s) => s.modelSettings);
  const forcedLanguage = useSettingsStore((s) => s.forcedLanguage);
  return useMutation({
    mutationFn: (body: Omit<ChapterQuizSubmitRequest, "options">) => {
      if (!chapterId) throw new Error("No chapter is selected.");
      return coursePlayerApi.submitQuiz(conversationId, chapterId, {
        ...body,
        options: {
          ...modelSettingsToOptions(modelSettings),
          ...(forcedLanguage ? { language: forcedLanguage } : {}),
        },
      });
    },
    onSuccess: (data) => {
      setProgressState(conversationId, data.learner_state);
      qc.invalidateQueries({ queryKey: ["course-player", conversationId] });
      qc.invalidateQueries({
        queryKey: ["conversations", "learner-state", conversationId],
      });
    },
  });
}
