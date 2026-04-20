import { create } from "zustand";

import type { Artifact, Message, SourceRef, UUID } from "@/lib/types";

// Represents an assistant turn that is still being streamed in. It becomes a
// committed `Message` once the backend emits `done` and the query cache refetches.
export interface StreamingMessage {
  conversationId: UUID;
  role: "assistant";
  generatorId?: string;
  outputType?: string;
  text: string;
  sources: SourceRef[];
  artifacts: Artifact[];
  startedAt: number;
}

interface ConversationState {
  activeConversationId: UUID | null;
  streamingByConversation: Record<UUID, StreamingMessage | null>;
  abortByConversation: Record<UUID, AbortController | null>;
  streamErrorByConversation: Record<UUID, string | null>;

  setActive: (id: UUID | null) => void;

  startStream: (conversationId: UUID, controller: AbortController) => void;
  appendChunk: (conversationId: UUID, text: string) => void;
  mergeSources: (conversationId: UUID, sources: SourceRef[]) => void;
  addArtifact: (conversationId: UUID, artifact: Artifact) => void;
  setStreamMeta: (
    conversationId: UUID,
    meta: { generatorId?: string; outputType?: string },
  ) => void;
  setStreamError: (conversationId: UUID, error: string | null) => void;
  endStream: (conversationId: UUID) => void;
  abortStream: (conversationId: UUID) => void;

  getStreaming: (conversationId: UUID) => StreamingMessage | null;
  isStreaming: (conversationId: UUID) => boolean;
  asTempMessage: (conversationId: UUID) => Message | null;
}

export const useConversationStore = create<ConversationState>((set, get) => ({
  activeConversationId: null,
  streamingByConversation: {},
  abortByConversation: {},
  streamErrorByConversation: {},

  setActive: (id) => set({ activeConversationId: id }),

  startStream: (conversationId, controller) =>
    set((state) => ({
      streamingByConversation: {
        ...state.streamingByConversation,
        [conversationId]: {
          conversationId,
          role: "assistant",
          text: "",
          sources: [],
          artifacts: [],
          startedAt: Date.now(),
        },
      },
      abortByConversation: {
        ...state.abortByConversation,
        [conversationId]: controller,
      },
      streamErrorByConversation: {
        ...state.streamErrorByConversation,
        [conversationId]: null,
      },
    })),

  appendChunk: (conversationId, text) =>
    set((state) => {
      const current = state.streamingByConversation[conversationId];
      if (!current) return {};
      return {
        streamingByConversation: {
          ...state.streamingByConversation,
          [conversationId]: { ...current, text: current.text + text },
        },
      };
    }),

  mergeSources: (conversationId, sources) =>
    set((state) => {
      const current = state.streamingByConversation[conversationId];
      if (!current) return {};
      return {
        streamingByConversation: {
          ...state.streamingByConversation,
          [conversationId]: { ...current, sources },
        },
      };
    }),

  addArtifact: (conversationId, artifact) =>
    set((state) => {
      const current = state.streamingByConversation[conversationId];
      if (!current) return {};
      return {
        streamingByConversation: {
          ...state.streamingByConversation,
          [conversationId]: {
            ...current,
            artifacts: [...current.artifacts, artifact],
          },
        },
      };
    }),

  setStreamMeta: (conversationId, meta) =>
    set((state) => {
      const current = state.streamingByConversation[conversationId];
      if (!current) return {};
      return {
        streamingByConversation: {
          ...state.streamingByConversation,
          [conversationId]: {
            ...current,
            generatorId: meta.generatorId ?? current.generatorId,
            outputType: meta.outputType ?? current.outputType,
          },
        },
      };
    }),

  setStreamError: (conversationId, error) =>
    set((state) => ({
      streamErrorByConversation: {
        ...state.streamErrorByConversation,
        [conversationId]: error,
      },
    })),

  endStream: (conversationId) =>
    set((state) => ({
      streamingByConversation: {
        ...state.streamingByConversation,
        [conversationId]: null,
      },
      abortByConversation: {
        ...state.abortByConversation,
        [conversationId]: null,
      },
    })),

  abortStream: (conversationId) => {
    const controller = get().abortByConversation[conversationId];
    controller?.abort();
    set((state) => ({
      abortByConversation: {
        ...state.abortByConversation,
        [conversationId]: null,
      },
      streamingByConversation: {
        ...state.streamingByConversation,
        [conversationId]: null,
      },
    }));
  },

  getStreaming: (conversationId) =>
    get().streamingByConversation[conversationId] ?? null,

  isStreaming: (conversationId) =>
    get().streamingByConversation[conversationId] != null,

  asTempMessage: (conversationId) => {
    const s = get().streamingByConversation[conversationId];
    if (!s) return null;
    return {
      id: `stream-${conversationId}`,
      conversation_id: conversationId,
      role: "assistant",
      content: s.text,
      generator_id: s.generatorId ?? null,
      output_type: (s.outputType as Message["output_type"]) ?? null,
      artifacts: s.artifacts,
      sources: s.sources,
      created_at: new Date(s.startedAt).toISOString(),
    };
  },
}));
