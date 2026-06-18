import type {
  Artifact,
  Conversation,
  CourseBuilderRead,
  GeneratorManifest,
  LearnerState,
  Message,
  ParserSettings,
  ProviderRead,
  RetrievalSettings,
  SourceFile,
  StreamEvent
} from "./types";

const API_BASE = import.meta.env.VITE_TEACHERLM_API_URL ?? "http://127.0.0.1:8765/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: init?.body instanceof FormData ? undefined : { "Content-Type": "application/json" },
    ...init
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || response.statusText);
  }
  if (response.status === 204) {
    return undefined as T;
  }
  return (await response.json()) as T;
}

export const api = {
  health: () => request<{ ok: boolean; data_dir: string }>("/health"),
  listConversations: () => request<{ conversations: Conversation[] }>("/conversations"),
  getConversation: (conversationId: string) =>
    request<Conversation>(`/conversations/${conversationId}`),
  createConversation: (title: string) =>
    request<Conversation>("/conversations", { method: "POST", body: JSON.stringify({ title }) }),
  updateConversation: (conversationId: string, payload: Partial<Pick<Conversation, "title">>) =>
    request<Conversation>(`/conversations/${conversationId}`, {
      method: "PATCH",
      body: JSON.stringify(payload)
    }),
  deleteConversation: (conversationId: string) =>
    request<void>(`/conversations/${conversationId}`, { method: "DELETE" }),
  listMessages: (conversationId: string) =>
    request<{ messages: Message[] }>(`/conversations/${conversationId}/messages`),
  listFiles: (conversationId: string) =>
    request<{ files: SourceFile[]; items?: SourceFile[]; total?: number }>(`/conversations/${conversationId}/files`),
  getFile: (conversationId: string, fileId: string) =>
    request<SourceFile>(`/conversations/${conversationId}/files/${fileId}`),
  uploadFile: async (conversationId: string, file: File) => {
    const form = new FormData();
    form.append("upload", file, file.name);
    return request<SourceFile>(`/conversations/${conversationId}/files`, {
      method: "POST",
      body: form
    });
  },
  deleteFile: (conversationId: string, fileId: string) =>
    request<void>(`/conversations/${conversationId}/files/${fileId}`, { method: "DELETE" }),
  retryFile: (conversationId: string, fileId: string) =>
    request<SourceFile>(`/conversations/${conversationId}/files/${fileId}/retry`, { method: "POST" }),
  learnerState: (conversationId: string) =>
    request<LearnerState>(`/conversations/${conversationId}/learner-state`),
  artifacts: (conversationId: string) =>
    request<{ artifacts: Artifact[] }>(`/conversations/${conversationId}/artifacts`),
  coursebuilder: (conversationId: string) =>
    request<CourseBuilderRead>(`/conversations/${conversationId}/coursebuilder`),
  rebuildCoursebuilder: (conversationId: string) =>
    request<CourseBuilderRead>(`/conversations/${conversationId}/coursebuilder/rebuild`, { method: "POST" }),
  generators: () => request<{ generators: GeneratorManifest[] }>("/generators"),
  providers: () => request<{ providers: ProviderRead[] }>("/settings/llm-providers"),
  createProvider: (payload: Record<string, unknown>) =>
    request<ProviderRead>("/settings/llm-providers", {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  patchProvider: (providerId: string, payload: Record<string, unknown>) =>
    request<ProviderRead>(`/settings/llm-providers/${providerId}`, {
      method: "PATCH",
      body: JSON.stringify(payload)
    }),
  deleteProvider: (providerId: string) =>
    request<{ ok: boolean }>(`/settings/llm-providers/${providerId}`, { method: "DELETE" }),
  testProvider: (providerId: string) =>
    request<ProviderRead>(`/settings/llm-providers/${providerId}/test`, { method: "POST" }),
  parserSettings: () => request<ParserSettings>("/settings/parse"),
  updateParserSettings: (payload: Record<string, unknown>) =>
    request<ParserSettings>("/settings/parse", {
      method: "PATCH",
      body: JSON.stringify(payload)
    }),
  retrievalSettings: () => request<RetrievalSettings>("/settings/retrieval"),
  updateRetrievalSettings: (payload: Record<string, unknown>) =>
    request<RetrievalSettings>("/settings/retrieval", {
      method: "PATCH",
      body: JSON.stringify(payload)
    }),
  rebuildIndexes: (conversationId: string) =>
    request<{ ok: boolean; index_status: RetrievalSettings["index_status"] }>(`/conversations/${conversationId}/indexes/rebuild`, {
      method: "POST"
    }),
  artifactUrl: (artifactId: string) => `${API_BASE}/artifacts/${artifactId}`
};

export async function streamChat(
  conversationId: string,
  message: string,
  sourceFileIds: string[],
  onEvent: (event: StreamEvent) => void
) {
  await streamPost(`/conversations/${conversationId}/chat`, {
    message,
    source_file_ids: sourceFileIds,
    options: {}
  }, onEvent);
}

export async function streamGenerate(
  conversationId: string,
  outputType: string,
  prompt: string,
  sourceFileIds: string[],
  onEvent: (event: StreamEvent) => void,
  options: Record<string, unknown> = {}
) {
  await streamPost(`/conversations/${conversationId}/generate`, {
    output_type: outputType,
    prompt,
    source_file_ids: sourceFileIds,
    options
  }, onEvent);
}

async function streamPost(path: string, payload: unknown, onEvent: (event: StreamEvent) => void) {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  if (!response.ok || !response.body) {
    throw new Error(await response.text());
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const blocks = buffer.split("\n\n");
    buffer = blocks.pop() ?? "";
    for (const block of blocks) {
      const parsed = parseSseBlock(block);
      if (parsed) onEvent(parsed);
    }
  }
  const parsed = parseSseBlock(buffer);
  if (parsed) onEvent(parsed);
}

function parseSseBlock(block: string): StreamEvent | null {
  const eventLine = block.split("\n").find((line) => line.startsWith("event:"));
  const dataLine = block.split("\n").find((line) => line.startsWith("data:"));
  if (!eventLine || !dataLine) return null;
  const event = eventLine.replace("event:", "").trim();
  const raw = dataLine.replace("data:", "").trim();
  try {
    return { event, data: JSON.parse(raw) } as StreamEvent;
  } catch {
    return { event, data: raw } as StreamEvent;
  }
}
