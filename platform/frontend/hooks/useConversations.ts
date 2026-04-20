import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { conversationsApi } from "@/lib/api";
import type {
  Conversation,
  ConversationCreate,
  ConversationList,
  ConversationUpdate,
  UUID,
} from "@/lib/types";

const ROOT_KEY = ["conversations"] as const;

export function useConversations(params?: { limit?: number; offset?: number }) {
  return useQuery<ConversationList>({
    queryKey: [...ROOT_KEY, "list", params ?? {}],
    queryFn: () => conversationsApi.list(params),
  });
}

export function useConversation(id: UUID | null | undefined) {
  return useQuery<Conversation>({
    queryKey: [...ROOT_KEY, "detail", id],
    queryFn: () => conversationsApi.get(id as UUID),
    enabled: Boolean(id),
  });
}

export function useCreateConversation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: ConversationCreate = {}) => conversationsApi.create(body),
    onSuccess: (created) => {
      qc.invalidateQueries({ queryKey: [...ROOT_KEY, "list"] });
      qc.setQueryData([...ROOT_KEY, "detail", created.id], created);
    },
  });
}

export function useUpdateConversation(id: UUID) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: ConversationUpdate) => conversationsApi.update(id, body),
    onSuccess: (updated) => {
      qc.setQueryData([...ROOT_KEY, "detail", id], updated);
      qc.invalidateQueries({ queryKey: [...ROOT_KEY, "list"] });
    },
  });
}

export function useDeleteConversation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: UUID) => conversationsApi.remove(id),
    onSuccess: (_void, id) => {
      qc.removeQueries({ queryKey: [...ROOT_KEY, "detail", id] });
      qc.invalidateQueries({ queryKey: [...ROOT_KEY, "list"] });
    },
  });
}
