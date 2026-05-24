import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { filesApi } from "@/lib/api";
import type { UploadedFile, UploadedFileList, UUID } from "@/lib/types";
import {
  modelSettingsToOptions,
  useSettingsStore,
} from "@/stores/settingsStore";

const ROOT_KEY = ["files"] as const;

// If any file is mid-pipeline (parsing/chunking/embedding), poll every 3s so
// the status badges move from "uploaded" → "ready" without manual refresh.
const PIPELINE_STATUSES = new Set([
  "uploaded",
  "parsing",
  "chunking",
  "extracting_concepts",
  "building_course",
  "embedding",
]);
const POLL_INTERVAL_MS = 3000;

export function useFiles(conversationId: UUID | null | undefined) {
  return useQuery<UploadedFileList>({
    queryKey: [...ROOT_KEY, conversationId ?? null],
    queryFn: () => filesApi.list(conversationId as UUID),
    enabled: Boolean(conversationId),
    refetchInterval: (query) => {
      const data = query.state.data as UploadedFileList | undefined;
      if (!data) return false;
      const anyPending = data.items.some((f) => PIPELINE_STATUSES.has(f.status));
      return anyPending ? POLL_INTERVAL_MS : false;
    },
  });
}

export function useUploadFile(conversationId: UUID) {
  const qc = useQueryClient();
  const modelSettings = useSettingsStore((state) => state.modelSettings);
  const forcedLanguage = useSettingsStore((state) => state.forcedLanguage);
  return useMutation<UploadedFile, Error, File>({
    mutationFn: (file) =>
      filesApi.upload(
        conversationId,
        file,
        {
          ...modelSettingsToOptions(modelSettings),
          ...(forcedLanguage ? { language: forcedLanguage } : {}),
        },
      ),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: [...ROOT_KEY, conversationId] });
    },
  });
}

export function useDeleteFile(conversationId: UUID) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (filePk: UUID) => filesApi.remove(conversationId, filePk),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: [...ROOT_KEY, conversationId] });
    },
  });
}
