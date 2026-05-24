import type {
  Conversation,
  ConversationCreate,
  ConversationList,
  ConversationUpdate,
  ChapterQuizSubmitRequest,
  ChapterQuizSubmitResponse,
  CourseBuilderGenerateRequest,
  CourseBuilderQuizSubmitRequest,
  CourseBuilderQuizSubmitResponse,
  CourseBuilderResponse,
  CoursePlayerResponse,
  CoursePlayerUnlockResponse,
  GeneratorListResponse,
  GeneratorView,
  KnowledgeGraphResponse,
  KnowledgeCheckStartRequest,
  KnowledgeCheckStartResponse,
  KnowledgeCheckSubmitRequest,
  KnowledgeCheckSubmitResponse,
  LearnerState,
  LivenessResponse,
  MessageList,
  QuizAttemptRequest,
  QuizAttemptResponse,
  ReadinessResponse,
  ReviewTestActionResponse,
  ReviewTestStartRequest,
  ReviewTestStartResponse,
  ReviewTestStatusResponse,
  ReviewTestSubmitRequest,
  ReviewTestSubmitResponse,
  UploadedFile,
  UploadedFileList,
  UUID,
} from "./types";

type UploadOptions = Record<string, unknown>;

const CONFIGURED_API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/+$/, "") ??
  "http://localhost:8000";

export const API_BASE_URL = resolveApiBaseUrl();

function resolveApiBaseUrl(): string {
  if (typeof window === "undefined") return CONFIGURED_API_BASE_URL;

  try {
    const configured = new URL(CONFIGURED_API_BASE_URL);
    const pageHost = window.location.hostname;
    const configuredHost = configured.hostname;
    if (
      configuredHost === "localhost" &&
      pageHost &&
      pageHost !== "localhost" &&
      pageHost !== "127.0.0.1"
    ) {
      configured.hostname = pageHost;
      return configured.toString().replace(/\/+$/, "");
    }
  } catch {
    return CONFIGURED_API_BASE_URL;
  }

  return CONFIGURED_API_BASE_URL;
}

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
  learnerState: (id: UUID) =>
    apiFetch<LearnerState>(`/api/conversations/${id}/learner-state`),
  remove: (id: UUID) =>
    apiFetch<void>(`/api/conversations/${id}`, { method: "DELETE" }),
};

// ---------- knowledge checks ----------

export const knowledgeChecksApi = {
  start: (conversationId: UUID, body: KnowledgeCheckStartRequest = {}) =>
    apiFetch<KnowledgeCheckStartResponse>(
      `/api/conversations/${conversationId}/knowledge-checks/start`,
      { method: "POST", body },
    ),
  submit: (
    conversationId: UUID,
    checkId: UUID,
    body: KnowledgeCheckSubmitRequest,
  ) =>
    apiFetch<KnowledgeCheckSubmitResponse>(
      `/api/conversations/${conversationId}/knowledge-checks/${checkId}/submit`,
      { method: "POST", body },
    ),
  submitQuiz: (conversationId: UUID, body: QuizAttemptRequest) =>
    apiFetch<QuizAttemptResponse>(
      `/api/conversations/${conversationId}/quiz-attempts`,
      { method: "POST", body },
    ),
};

export const reviewTestsApi = {
  status: (conversationId: UUID) =>
    apiFetch<ReviewTestStatusResponse>(
      `/api/conversations/${conversationId}/review-tests/status`,
    ),
  start: (conversationId: UUID, body: ReviewTestStartRequest = {}) =>
    apiFetch<ReviewTestStartResponse>(
      `/api/conversations/${conversationId}/review-tests/start`,
      { method: "POST", body },
    ),
  submit: (
    conversationId: UUID,
    windowId: UUID,
    body: ReviewTestSubmitRequest,
  ) =>
    apiFetch<ReviewTestSubmitResponse>(
      `/api/conversations/${conversationId}/review-tests/${windowId}/submit`,
      { method: "POST", body },
    ),
  snooze: (conversationId: UUID, windowId: UUID) =>
    apiFetch<ReviewTestActionResponse>(
      `/api/conversations/${conversationId}/review-tests/${windowId}/snooze`,
      { method: "POST" },
    ),
  dismiss: (conversationId: UUID, windowId: UUID) =>
    apiFetch<ReviewTestActionResponse>(
      `/api/conversations/${conversationId}/review-tests/${windowId}/dismiss`,
      { method: "POST" },
    ),
};

export const coursePlayerApi = {
  get: (conversationId: UUID) =>
    apiFetch<CoursePlayerResponse>(
      `/api/conversations/${conversationId}/course-player`,
    ),
  rebuild: (conversationId: UUID) =>
    apiFetch<CoursePlayerResponse>(
      `/api/conversations/${conversationId}/course-player/rebuild`,
      { method: "POST" },
    ),
  unlock: (conversationId: UUID, chapterId: UUID) =>
    apiFetch<CoursePlayerUnlockResponse>(
      `/api/conversations/${conversationId}/course-player/chapters/${chapterId}/unlock`,
      { method: "POST" },
    ),
  submitQuiz: (
    conversationId: UUID,
    chapterId: UUID,
    body: ChapterQuizSubmitRequest,
  ) =>
    apiFetch<ChapterQuizSubmitResponse>(
      `/api/conversations/${conversationId}/course-player/chapters/${chapterId}/quiz/submit`,
      { method: "POST", body },
    ),
};

export const courseBuilderApi = {
  get: (conversationId: UUID) =>
    apiFetch<CourseBuilderResponse>(
      `/api/conversations/${conversationId}/coursebuilder`,
    ),
  generate: (conversationId: UUID, body: CourseBuilderGenerateRequest = {}) =>
    apiFetch<CourseBuilderResponse>(
      `/api/conversations/${conversationId}/coursebuilder/generate`,
      { method: "POST", body },
    ),
  rebuild: (conversationId: UUID, body: CourseBuilderGenerateRequest = {}) =>
    apiFetch<CourseBuilderResponse>(
      `/api/conversations/${conversationId}/coursebuilder/rebuild`,
      { method: "POST", body },
    ),
  submitQuiz: (
    conversationId: UUID,
    chapterId: UUID,
    body: CourseBuilderQuizSubmitRequest,
  ) =>
    apiFetch<CourseBuilderQuizSubmitResponse>(
      `/api/conversations/${conversationId}/coursebuilder/chapters/${chapterId}/quiz/submit`,
      { method: "POST", body },
    ),
};

export const knowledgeGraphApi = {
  get: (conversationId: UUID) =>
    apiFetch<KnowledgeGraphResponse>(
      `/api/conversations/${conversationId}/knowledge-graph`,
    ),
  rebuild: (conversationId: UUID, options?: Record<string, unknown>) =>
    apiFetch<KnowledgeGraphResponse>(
      `/api/conversations/${conversationId}/knowledge-graph/rebuild`,
      { method: "POST", body: { options: options ?? {} } },
    ),
  remediation: (conversationId: UUID, conceptId: UUID) =>
    apiFetch<{ target_concept_id: UUID; target_concept_name: string; steps: unknown[]; source: string }>(
      `/api/conversations/${conversationId}/knowledge-graph/remediation`,
      { query: { concept_id: conceptId } },
    ),
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
  upload: (conversationId: UUID, file: File, options?: UploadOptions) => {
    const form = new FormData();
    form.append("upload", file, file.name);
    if (options && Object.keys(options).length > 0) {
      form.append("llm_options_json", JSON.stringify(options));
    }
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
