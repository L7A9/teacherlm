import type {
  Conversation,
  ConversationCreate,
  ConversationList,
  ConversationUpdate,
  GeneratorListResponse,
  GeneratorView,
  LivenessResponse,
  MessageList,
  ReadinessResponse,
  UploadedFile,
  UploadedFileList,
  UUID,
} from "./types";

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/+$/, "") ??
  "http://localhost:8000";

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
    public body?: unknown,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

type FetchOptions = Omit<RequestInit, "body"> & {
  body?: unknown;
  query?: Record<string, string | number | boolean | undefined>;
};

function buildUrl(path: string, query?: FetchOptions["query"]): string {
  const url = new URL(
    path.startsWith("/") ? `${API_BASE_URL}${path}` : `${API_BASE_URL}/${path}`,
  );
  if (query) {
    for (const [key, value] of Object.entries(query)) {
      if (value !== undefined) url.searchParams.set(key, String(value));
    }
  }
  return url.toString();
}

export async function apiFetch<T>(
  path: string,
  { body, headers, query, ...init }: FetchOptions = {},
): Promise<T> {
  const isForm = body instanceof FormData;
  const response = await fetch(buildUrl(path, query), {
    ...init,
    headers: {
      Accept: "application/json",
      ...(isForm || body === undefined
        ? {}
        : { "Content-Type": "application/json" }),
      ...headers,
    },
    body: isForm ? body : body === undefined ? undefined : JSON.stringify(body),
    credentials: "include",
  });

  if (response.status === 204) return undefined as T;

  const contentType = response.headers.get("content-type") ?? "";
  const payload: unknown = contentType.includes("application/json")
    ? await response.json()
    : await response.text();

  if (!response.ok) {
    const detail =
      typeof payload === "object" && payload !== null && "detail" in payload
        ? String((payload as { detail: unknown }).detail)
        : response.statusText;
    throw new ApiError(response.status, detail, payload);
  }

  return payload as T;
}

// ---------- conversations ----------

export const conversationsApi = {
  list: (params?: { limit?: number; offset?: number }) =>
    apiFetch<ConversationList>("/api/conversations", { query: params }),
  get: (id: UUID) => apiFetch<Conversation>(`/api/conversations/${id}`),
  create: (body: ConversationCreate) =>
    apiFetch<Conversation>("/api/conversations", { method: "POST", body }),
  update: (id: UUID, body: ConversationUpdate) =>
    apiFetch<Conversation>(`/api/conversations/${id}`, {
      method: "PATCH",
      body,
    }),
  remove: (id: UUID) =>
    apiFetch<void>(`/api/conversations/${id}`, { method: "DELETE" }),
};

// ---------- messages ----------

export const messagesApi = {
  list: (conversationId: UUID, params?: { limit?: number; offset?: number }) =>
    apiFetch<MessageList>(`/api/conversations/${conversationId}/messages`, {
      query: params,
    }),
};

// ---------- files ----------

export const filesApi = {
  list: (conversationId: UUID) =>
    apiFetch<UploadedFileList>(
      `/api/conversations/${conversationId}/files`,
    ),
  get: (conversationId: UUID, filePk: UUID) =>
    apiFetch<UploadedFile>(
      `/api/conversations/${conversationId}/files/${filePk}`,
    ),
  upload: (conversationId: UUID, file: File) => {
    const form = new FormData();
    form.append("upload", file, file.name);
    return apiFetch<UploadedFile>(
      `/api/conversations/${conversationId}/files`,
      { method: "POST", body: form },
    );
  },
  remove: (conversationId: UUID, filePk: UUID) =>
    apiFetch<void>(
      `/api/conversations/${conversationId}/files/${filePk}`,
      { method: "DELETE" },
    ),
};

// ---------- generators ----------

export const generatorsApi = {
  list: (includeDisabled = false) =>
    apiFetch<GeneratorListResponse>("/api/generators", {
      query: { include_disabled: includeDisabled },
    }),
  get: (id: string) => apiFetch<GeneratorView>(`/api/generators/${id}`),
};

// ---------- health ----------

export const healthApi = {
  liveness: () => apiFetch<LivenessResponse>("/api/health"),
  readiness: () => apiFetch<ReadinessResponse>("/api/health/ready"),
};
