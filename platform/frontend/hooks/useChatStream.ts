"use client";

import { useCallback } from "react";

import { useQueryClient } from "@tanstack/react-query";

import { ssePost } from "@/lib/sse";
import type {
  Artifact,
  ChatRequest,
  DoneEventData,
  ErrorEventData,
  GenerateRequest,
  Message,
  MessageList,
  SourceRef,
  SseEvent,
  UUID,
} from "@/lib/types";
import { useConversationStore } from "@/stores/conversationStore";
import { useProgressStore } from "@/stores/progressStore";

type StreamPath = "chat" | "generate";

interface RunStreamArgs {
  conversationId: UUID;
  path: StreamPath;
  body: ChatRequest | GenerateRequest;
}

function extractText(data: unknown): string {
  if (typeof data === "string") return data;
  if (data && typeof data === "object") {
    const d = data as Record<string, unknown>;
    for (const key of ["text", "delta", "content", "chunk"] as const) {
      const v = d[key];
      if (typeof v === "string") return v;
    }
  }
  return "";
}

function useRunStream() {
  const qc = useQueryClient();
  const {
    startStream,
    appendChunk,
    mergeSources,
    addArtifact,
    setStreamMeta,
    setStreamError,
    endStream,
  } = useConversationStore.getState();
  const applyOptimistic = useProgressStore.getState().applyOptimistic;

  return useCallback(
    async ({ conversationId, path, body }: RunStreamArgs) => {
      const controller = new AbortController();
      startStream(conversationId, controller);
      addOptimisticUserMessage(qc, conversationId, path, body);

      const fullPath = `/api/conversations/${conversationId}/${path}`;

      try {
        for await (const event of ssePost({
          path: fullPath,
          body,
          signal: controller.signal,
        })) {
          handleEvent(conversationId, event, {
            appendChunk,
            mergeSources,
            addArtifact,
            setStreamMeta,
            onDone: (done) => {
              if (done.learner_updates) {
                applyOptimistic(conversationId, done.learner_updates);
              }
            },
            onError: (err) => setStreamError(conversationId, err.message),
          });
        }
      } catch (err) {
        if ((err as Error).name !== "AbortError") {
          setStreamError(conversationId, (err as Error).message);
        }
      } finally {
        endStream(conversationId);
        qc.invalidateQueries({ queryKey: ["messages", conversationId] });
        qc.invalidateQueries({ queryKey: ["conversations", "detail", conversationId] });
      }
    },
    [
      addArtifact,
      appendChunk,
      applyOptimistic,
      endStream,
      mergeSources,
      qc,
      setStreamError,
      setStreamMeta,
      startStream,
    ],
  );
}

function addOptimisticUserMessage(
  qc: ReturnType<typeof useQueryClient>,
  conversationId: UUID,
  path: StreamPath,
  body: ChatRequest | GenerateRequest,
) {
  if (path !== "chat" || !("user_message" in body)) return;

  const content = body.user_message.trim();
  if (!content) return;

  const message: Message = {
    id: `optimistic-user-${conversationId}-${Date.now()}`,
    conversation_id: conversationId,
    role: "user",
    content,
    generator_id: null,
    output_type: null,
    artifacts: [],
    sources: [],
    created_at: new Date().toISOString(),
  };

  qc.setQueryData<MessageList>(["messages", conversationId], (current) => {
    if (!current) return { items: [message], total: 1 };
    const alreadyPresent = current.items.some(
      (item) => item.role === "user" && item.content === content,
    );
    if (alreadyPresent) return current;
    return {
      ...current,
      items: [...current.items, message],
      total: current.total + 1,
    };
  });
}

interface HandleEventCallbacks {
  appendChunk: (id: UUID, text: string) => void;
  mergeSources: (id: UUID, sources: SourceRef[]) => void;
  addArtifact: (id: UUID, artifact: Artifact) => void;
  setStreamMeta: (id: UUID, meta: { generatorId?: string; outputType?: string }) => void;
  onDone: (done: DoneEventData) => void;
  onError: (err: ErrorEventData) => void;
}

function handleEvent(
  conversationId: UUID,
  event: SseEvent,
  cb: HandleEventCallbacks,
) {
  switch (event.event) {
    case "chunk":
    case "token": {
      const text = extractText(event.data);
      if (text) cb.appendChunk(conversationId, text);
      break;
    }
    case "sources": {
      let arr: unknown = event.data;
      if (
        arr &&
        typeof arr === "object" &&
        !Array.isArray(arr) &&
        Array.isArray((arr as Record<string, unknown>).sources)
      ) {
        arr = (arr as Record<string, unknown>).sources;
      }
      if (Array.isArray(arr)) {
        cb.mergeSources(conversationId, arr as SourceRef[]);
      }
      break;
    }
    case "artifact":
      if (event.data && typeof event.data === "object") {
        cb.addArtifact(conversationId, event.data as Artifact);
      }
      break;
    case "done":
      if (event.data && typeof event.data === "object") {
        const done = event.data as DoneEventData;
        if (done.generator_id || done.output_type) {
          cb.setStreamMeta(conversationId, {
            generatorId: done.generator_id,
            outputType: done.output_type,
          });
        }
        cb.onDone(done);
      }
      break;
    case "error":
      if (event.data && typeof event.data === "object") {
        cb.onError(event.data as ErrorEventData);
      }
      break;
    default:
      break;
  }
}

export function useChatStream() {
  const run = useRunStream();
  return useCallback(
    (conversationId: UUID, body: ChatRequest) =>
      run({ conversationId, path: "chat", body }),
    [run],
  );
}

export function useGenerateStream() {
  const run = useRunStream();
  return useCallback(
    (conversationId: UUID, body: GenerateRequest) =>
      run({ conversationId, path: "generate", body }),
    [run],
  );
}
