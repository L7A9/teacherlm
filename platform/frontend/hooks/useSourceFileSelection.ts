"use client";

import { useEffect, useMemo, useRef } from "react";

import type { UploadedFile, UUID } from "@/lib/types";
import { useUiStore } from "@/stores/uiStore";

const EMPTY_SOURCE_FILE_IDS: string[] = [];

export function useSourceFileSelection(
  conversationId: UUID,
  files: UploadedFile[],
) {
  const readyFiles = useMemo(() => files.filter(isUsableSourceFile), [files]);
  const selectedSourceFileIds = useUiStore(
    (s) => s.sourceFileSelectionByConversation[conversationId] ?? EMPTY_SOURCE_FILE_IDS,
  );
  const setSelected = useUiStore((s) => s.setSourceFileSelection);
  const previousFileIdsRef = useRef<string[]>([]);
  const fileIds = useMemo(() => readyFiles.map((file) => file.file_id), [readyFiles]);

  useEffect(() => {
    const previousFileIds = previousFileIdsRef.current;
    let next: string[] | null = null;

    if (fileIds.length === 0) {
      if (selectedSourceFileIds.length > 0) {
        next = [];
      }
    } else if (fileIds.length === 1) {
      const onlyFileId = fileIds[0];
      if (!onlyFileId) {
        return;
      }
      next = [onlyFileId];
    } else {
      const stillPresent = selectedSourceFileIds.filter((id) => fileIds.includes(id));
      if (selectedSourceFileIds.length === 0 || stillPresent.length === 0) {
        next = fileIds;
      } else {
        const previous = new Set(previousFileIds);
        const newlyAvailable = fileIds.filter((id) => !previous.has(id));
        next = [
          ...stillPresent,
          ...newlyAvailable.filter((id) => !stillPresent.includes(id)),
        ];
      }
    }

    previousFileIdsRef.current = fileIds;
    if (next && !sameStringList(selectedSourceFileIds, next)) {
      setSelected(conversationId, next);
    }
  }, [conversationId, fileIds, selectedSourceFileIds, setSelected]);

  const selectedSet = useMemo(
    () => new Set(selectedSourceFileIds),
    [selectedSourceFileIds],
  );
  const activeCount = fileIds.filter((id) => selectedSet.has(id)).length;

  const toggleFile = (fileId: string) => {
    if (!fileIds.includes(fileId)) {
      return;
    }

    if (selectedSet.has(fileId)) {
      if (activeCount <= 1) {
        return;
      }
      setSelected(
        conversationId,
        selectedSourceFileIds.filter((id) => id !== fileId),
      );
      return;
    }

    setSelected(
      conversationId,
      fileIds.filter((id) => id === fileId || selectedSet.has(id)),
    );
  };

  return {
    activeCount,
    readyFiles,
    readyFileIds: fileIds,
    selectedSet,
    selectedSourceFileIds,
    showFileCheckboxes: readyFiles.length > 1,
    toggleFile,
  };
}

export function isUsableSourceFile(file: UploadedFile) {
  return file.status === "ready" && file.chunk_count > 0;
}

function sameStringList(a: string[], b: string[]) {
  return a.length === b.length && a.every((value, index) => value === b[index]);
}
