// Mirror of backend Pydantic schemas. Keep field names in sync with
// platform/backend/schemas/*.py and the teacherlm_core schemas.

// ---------- shared primitives ----------

export type UUID = string;
export type ISODateTime = string;

export type Role = "user" | "assistant" | "system";

export type OutputType =
  | "text"
  | "quiz"
  | "report"
  | "presentation"
  | "flashcards"
  | "chart"
  | "podcast";

export type FileStatus =
  | "uploaded"
  | "parsing"
  | "chunking"
  | "embedding"
  | "ready"
  | "failed";

// ---------- conversations ----------

export interface Conversation {
  id: UUID;
  title: string;
  created_at: ISODateTime;
  updated_at: ISODateTime;
}

export interface ConversationList {
  items: Conversation[];
  total: number;
}

export interface ConversationCreate {
  title?: string;
}

export interface ConversationUpdate {
  title: string;
}

// ---------- messages ----------

export interface Artifact {
  type: string;
  url: string;
  filename?: string | null;
}

export interface SourceRef {
  text: string;
  source: string;
  score: number;
  chunk_id?: string | null;
}

export interface Message {
  id: UUID;
  conversation_id: UUID;
  role: Role;
  content: string;
  generator_id?: string | null;
  output_type?: OutputType | null;
  artifacts: Artifact[];
  sources: SourceRef[];
  created_at: ISODateTime;
}

export interface MessageList {
  items: Message[];
  total: number;
}

export interface ChatRequest {
  user_message: string;
  options?: Record<string, unknown>;
}

export interface GenerateRequest {
  output_type: OutputType;
  options?: Record<string, unknown>;
  topic?: string | null;
}

// ---------- files ----------

export interface UploadedFile {
  id: UUID;
  conversation_id: UUID;
  filename: string;
  file_id: string;
  status: FileStatus;
  chunk_count: number;
  parsed_markdown_path?: string | null;
  error?: string | null;
  created_at: ISODateTime;
}

export interface UploadedFileList {
  items: UploadedFile[];
  total: number;
}

// ---------- generators ----------

export interface GeneratorView {
  id: string;
  name?: string | null;
  output_type: string;
  icon?: string | null;
  description?: string | null;
  is_chat_default: boolean;
  enabled: boolean;
}

export interface GeneratorListResponse {
  items: GeneratorView[];
}

// ---------- learner state ----------

export interface LearnerState {
  conversation_id: string;
  understood_concepts: string[];
  struggling_concepts: string[];
  mastery_scores: Record<string, number>;
  session_turns: number;
  turns_since_progress: number;
}

export interface LearnerUpdates {
  concepts_covered: string[];
  concepts_demonstrated: string[];
  concepts_struggled: string[];
}

// ---------- SSE events from chat/generate ----------

export type SseEventName =
  | "chunk"
  | "sources"
  | "artifact"
  | "progress"
  | "done"
  | "error"
  | "message";

export interface SseEvent<T = unknown> {
  event: SseEventName | string;
  data: T;
}

export interface ChunkEvent {
  text?: string;
  delta?: string;
  content?: string;
  chunk?: string;
}

export interface DoneEventData {
  response?: string;
  generator_id?: string;
  output_type?: OutputType;
  artifacts?: Artifact[];
  sources?: SourceRef[];
  learner_updates?: LearnerUpdates;
  metadata?: Record<string, unknown>;
}

export interface ErrorEventData {
  message: string;
}

// ---------- artifact-specific payload shapes (rendered client-side) ----------

// Mirrors generators/quiz_gen/schemas.py — MCQ | TrueFalse | FillBlank.
export interface QuizQuestionMCQ {
  type: "mcq";
  bloom_level?: string;
  question: string;
  options: string[];
  correct_index: number;
  explanation?: string;
  concept?: string;
  source_chunk_id?: string;
}

export interface QuizQuestionTrueFalse {
  type: "true_false";
  bloom_level?: string;
  question: string;
  answer: boolean;
  explanation?: string;
  concept?: string;
  source_chunk_id?: string;
}

export interface QuizQuestionFillBlank {
  type: "fill_blank";
  bloom_level?: string;
  question: string;
  answer: string;
  accepted_answers?: string[];
  explanation?: string;
  concept?: string;
  source_chunk_id?: string;
}

export type QuizQuestion =
  | QuizQuestionMCQ
  | QuizQuestionTrueFalse
  | QuizQuestionFillBlank;

export interface QuizPayload {
  title?: string;
  intro_message?: string;
  questions: QuizQuestion[];
  bloom_distribution?: Record<string, number>;
}

export interface BasicFlashcard {
  type?: "basic";
  id?: string;
  front: string;
  back: string;
  concept?: string;
}

export interface ClozeFlashcard {
  type: "cloze";
  id?: string;
  // Anki-style cloze, e.g. "{{c1::photosynthesis}} is the process..."
  text: string;
  answer: string;
  concept?: string;
}

export type FlashcardItem = BasicFlashcard | ClozeFlashcard;

export interface FlashcardPayload {
  title?: string;
  cards: FlashcardItem[];
}

export interface ChartArtifactMetadata {
  mermaid_code?: string;
  diagram_type?: string;
}

export interface PodcastArtifactMetadata {
  transcript?: string;
  duration_seconds?: number;
}

// ---------- health ----------

export interface LivenessResponse {
  status: string;
  app: string;
  environment: string;
}

export interface ReadinessCheck {
  ok: boolean;
  error?: string | null;
}

export interface ReadinessResponse {
  ready: boolean;
  checks: Record<string, ReadinessCheck>;
}
