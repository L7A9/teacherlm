"use client";

import { useQuery } from "@tanstack/react-query";

import { messagesApi } from "@/lib/api";
import type { UUID } from "@/lib/types";

export function useMessages(conversationId: UUID | null | undefined) {
  return useQuery({
    queryKey: ["messages", conversationId],
    queryFn: () => {
      if (!conversationId) throw new Error("no conversationId");
      return messagesApi.list(conversationId);
    },
    enabled: Boolean(conversationId),
    staleTime: 0,
  });
}
