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
  | "chart"
  | "podcast"
  | "mindmap";

export type FileStatus =
  | "uploaded"
  | "parsing"
  | "chunking"
  | "extracting_concepts"
  | "building_course"
  | "embedding"
  | "ready"
  | "failed";

export type LlmProvider =
  | "ollama"
  | "openai"
  | "anthropic"
  | "openai_compatible";

export interface LlmRuntimeSettings {
  enabled: boolean;
  provider: LlmProvider;
  model: string;
  api_link: string;
  api_key_set: boolean;
}

export interface ParserRuntimeSettings {
  api_key_set: boolean;
}

export interface RuntimeSettingsResponse {
  llm: LlmRuntimeSettings;
  parser: ParserRuntimeSettings;
}

export interface RuntimeSettingsUpdate {
  llm?: {
    enabled?: boolean;
    provider?: LlmProvider;
    model?: string;
    api_link?: string;
    api_key?: string | null;
  };
  parser?: {
    api_key?: string | null;
  };
}

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
  key?: string | null;
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
  source_file_ids?: string[] | null;
}

export interface GenerateRequest {
  output_type: OutputType;
  options?: Record<string, unknown>;
  topic?: string | null;
  source_file_ids?: string[] | null;
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
  known_concepts?: KnownConcept[];
  concept_progress?: ConceptProgress[];
  learning_phases?: LearningPhase[];
  objective_progress?: ObjectiveProgress[];
  phase_progress?: PhaseProgress[];
  remediation_paths?: RemediationPath[];
}

export interface LearnerUpdates {
  concepts_covered: string[];
  concepts_demonstrated: string[];
  concepts_struggled: string[];
}

export interface KnownConcept {
  id: string;
  name: string;
  aliases: string[];
  description: string;
  bloom_level: string;
  importance: number;
  course_parts?: Array<{ section_id?: string; title?: string }>;
}

export interface ConceptProgress {
  concept_id: string;
  name: string;
  mastery: number;
  encounters: number;
  struggle_evidence: number;
}

export interface LearningPhase {
  id: UUID;
  title: string;
  summary: string;
  order_index: number;
  objective_ids: UUID[];
}

export interface ObjectiveProgress {
  objective_id: UUID;
  phase_id: UUID;
  objective_text: string;
  bloom_level: string;
  mastery: number;
  encounters: number;
  struggle_evidence: number;
  concept_ids: UUID[];
  order_index: number;
}

export interface PhaseProgress {
  phase_id: UUID;
  title: string;
  mastery: number;
  objectives_total: number;
  objectives_mastered: number;
  struggle_evidence: number;
  order_index: number;
}

// ---------- knowledge checks ----------

export type KnowledgeQuestionType =
  | "mcq"
  | "true_false"
  | "fill_blank"
  | "short_answer";

export interface KnowledgeCheckQuestion {
  id: UUID;
  concept_id: UUID;
  concept_name: string;
  phase_id?: UUID | null;
  objective_id?: UUID | null;
  question_type: KnowledgeQuestionType;
  bloom_level: string;
  prompt: string;
  options: string[];
  source_chunk_ids: string[];
}

export interface KnowledgeCheckStartRequest {
  concept_id?: UUID | null;
  phase_id?: UUID | null;
  objective_id?: UUID | null;
  count?: number;
  question_types?: KnowledgeQuestionType[] | null;
  options?: Record<string, unknown>;
}

export interface KnowledgeCheckStartResponse {
  checks: KnowledgeCheckQuestion[];
  learner_state: LearnerState;
}

export interface KnowledgeCheckSubmitRequest {
  answer: unknown;
  options?: Record<string, unknown>;
}

export interface KnowledgeCheckResult {
  check_id: UUID;
  concept_id: UUID;
  concept_name: string;
  question_index?: number | null;
  score: number;
  is_correct: boolean;
  feedback: string;
  evidence_strength: string;
  mastery_delta: number;
  remediation_paths?: RemediationPath[];
}

export interface RemediationStep {
  concept_id?: UUID | null;
  concept_name: string;
  mastery: number;
  reason: string;
  source_chunk_ids: string[];
}

export interface RemediationPath {
  target_concept_id: UUID;
  target_concept_name: string;
  steps: RemediationStep[];
  source: string;
}

export interface KnowledgeCheckSubmitResponse {
  result: KnowledgeCheckResult;
  learner_state: LearnerState;
}

export interface QuizAttemptAnswer {
  question_index: number;
  answer: unknown;
}

export interface QuizAttemptRequest {
  questions: QuizQuestion[];
  answers: QuizAttemptAnswer[];
  options?: Record<string, unknown>;
}

export interface QuizAttemptResponse {
  results: KnowledgeCheckResult[];
  learner_state: LearnerState;
  score: number;
  total: number;
}

// ---------- review tests ----------

export interface ReviewWindowSummary {
  id: UUID;
  status: "pending" | "started" | "completed" | "snoozed" | "dismissed" | string;
  answered_count: number;
  due_count: number;
  snooze_until_count?: number | null;
  concept_ids: UUID[];
  objective_ids: UUID[];
  phase_ids: UUID[];
  source_chunk_ids: string[];
  generated_check_ids: UUID[];
}

export interface ReviewTestStatusResponse {
  answered_count: number;
  pending_count: number;
  due: boolean;
  window?: ReviewWindowSummary | null;
  learner_state?: LearnerState | null;
}

export interface ReviewTestStartRequest {
  options?: Record<string, unknown>;
}

export interface ReviewTestStartResponse {
  window: ReviewWindowSummary;
  checks: KnowledgeCheckQuestion[];
  learner_state: LearnerState;
}

export interface ReviewTestAnswer {
  check_id: UUID;
  answer: unknown;
}

export interface ReviewTestSubmitRequest {
  answers: ReviewTestAnswer[];
  options?: Record<string, unknown>;
}

export interface ReviewTestSubmitResponse {
  window: ReviewWindowSummary;
  results: KnowledgeCheckResult[];
  learner_state: LearnerState;
}

export interface ReviewTestActionResponse {
  window: ReviewWindowSummary;
  answered_count: number;
  due: boolean;
}

// ---------- course player ----------

export interface CourseLessonBlock {
  id: UUID;
  lesson_id: UUID;
  block_type: "definition" | "explanation" | "example" | "procedure" | "formula" | "summary" | "quiz" | string;
  title: string;
  content: string;
  order_index: number;
  source_chunk_ids: string[];
  metadata: Record<string, unknown>;
}

export interface CourseLesson {
  id: UUID;
  chapter_id: UUID;
  objective_id?: UUID | null;
  title: string;
  summary: string;
  order_index: number;
  concept_ids: UUID[];
  source_chunk_ids: string[];
  prerequisite_concept_ids: UUID[];
  next_concept_ids: UUID[];
  related_example_ids: UUID[];
  remediation_objective_ids: UUID[];
  graph_hints: Record<string, unknown>;
  blocks: CourseLessonBlock[];
}

export interface KnowledgeGraphNode {
  id: UUID;
  node_type: string;
  label: string;
  description: string;
  ref_id?: string | null;
  source_chunk_ids: string[];
  metadata: Record<string, unknown>;
}

export interface KnowledgeGraphEdge {
  id: UUID;
  source_node_id: UUID;
  target_node_id: UUID;
  relation_type: string;
  confidence: number;
  source_chunk_ids: string[];
  metadata: Record<string, unknown>;
}

export interface KnowledgeGraphResponse {
  conversation_id: UUID;
  nodes: KnowledgeGraphNode[];
  edges: KnowledgeGraphEdge[];
  node_count: number;
  edge_count: number;
}

export interface ChapterQuiz {
  id: UUID;
  chapter_id: UUID;
  pass_score: number;
  question_ids: UUID[];
  questions: KnowledgeCheckQuestion[];
}

export interface CourseChapter {
  id: UUID;
  phase_id?: UUID | null;
  title: string;
  summary: string;
  order_index: number;
  objective_ids: UUID[];
  concept_ids: UUID[];
  source_chunk_ids: string[];
  state: "locked" | "available" | "completed";
  best_score: number;
  attempts: number;
  soft_lock_overridden: boolean;
  progress: number;
  lessons: CourseLesson[];
  quiz?: ChapterQuiz | null;
}

export interface CoursePlayerResponse {
  conversation_id: UUID;
  chapters: CourseChapter[];
  learner_state: LearnerState;
  course_status?: "waiting_for_files" | "ready";
  pending_file_count?: number;
  total_file_count?: number;
}

export interface CoursePlayerUnlockResponse {
  chapter: CourseChapter;
  learner_state: LearnerState;
}

export interface ChapterQuizSubmitRequest {
  answers: ReviewTestAnswer[];
  options?: Record<string, unknown>;
}

export interface ChapterQuizSubmitResponse {
  chapter: CourseChapter;
  results: KnowledgeCheckResult[];
  score: number;
  passed: boolean;
  learner_state: LearnerState;
}

// ---------- course builder ----------

export type CourseBuilderStatus =
  | "queued"
  | "analyzing"
  | "generating_outline"
  | "generating_chapters"
  | "generating_lessons"
  | "generating_quizzes"
  | "validating"
  | "ready"
  | "failed"
  | string;

export interface CourseBuilderCitation {
  chunk_id: string;
  source: string;
  page_start?: number | null;
  page_end?: number | null;
  section?: string;
  snippet?: string;
}

export interface CourseBuilderLessonBlock {
  id: UUID;
  lesson_id: UUID;
  block_type:
    | "explanation"
    | "definition"
    | "example"
    | "table"
    | "equation"
    | "chart"
    | "diagram"
    | "procedure"
    | "warning"
    | "summary"
    | string;
  title: string;
  content: string;
  order_index: number;
  data_json: Record<string, unknown>;
  source_citations: CourseBuilderCitation[];
  validation_status: string;
}

export interface CourseBuilderLesson {
  id: UUID;
  chapter_id: UUID;
  title: string;
  order_index: number;
  learning_objectives: string[];
  source_chunk_ids: string[];
  support_status: string;
  blocks: CourseBuilderLessonBlock[];
}

export interface CourseBuilderQuizQuestion {
  id: UUID;
  quiz_id: UUID;
  chapter_id: UUID;
  question_type: "mcq" | string;
  prompt: string;
  options: string[];
  explanation: string;
  order_index: number;
  source_citations: CourseBuilderCitation[];
}

export interface CourseBuilderQuiz {
  id: UUID;
  chapter_id: UUID;
  pass_score: number;
  question_count: number;
  source_chunk_ids: string[];
  questions: CourseBuilderQuizQuestion[];
}

export interface CourseBuilderChapter {
  id: UUID;
  course_id: UUID;
  title: string;
  description: string;
  order_index: number;
  summary: string;
  source_chunk_ids: string[];
  is_locked: boolean;
  unlock_rule: Record<string, unknown>;
  best_score: number;
  attempts: number;
  completed: boolean;
  lessons: CourseBuilderLesson[];
  quiz?: CourseBuilderQuiz | null;
}

export interface CourseBuilderProgressEvent {
  id: UUID;
  conversation_id: UUID;
  course_id?: UUID | null;
  stage: CourseBuilderStatus;
  message: string;
  percent: number;
  metadata: Record<string, unknown>;
  created_at: ISODateTime;
}

export interface CourseBuilderResponse {
  id?: UUID | null;
  conversation_id: UUID;
  title: string;
  description: string;
  learning_objectives: string[];
  prerequisites: string[];
  status: CourseBuilderStatus;
  language?: string | null;
  error?: string | null;
  generation_metadata: Record<string, unknown>;
  chapters: CourseBuilderChapter[];
  progress_events: CourseBuilderProgressEvent[];
  pending_file_count: number;
  total_file_count: number;
}

export interface CourseBuilderGenerateRequest {
  options?: Record<string, unknown>;
}

export interface CourseBuilderQuizAnswer {
  question_id: UUID;
  answer: string | number;
}

export interface CourseBuilderQuizSubmitRequest {
  answers: CourseBuilderQuizAnswer[];
}

export interface CourseBuilderQuizResult {
  question_id: UUID;
  is_correct: boolean;
  correct_index: number;
  selected_index?: number | null;
  feedback: string;
}

export interface CourseBuilderQuizSubmitResponse {
  chapter: CourseBuilderChapter;
  score: number;
  passed: boolean;
  results: CourseBuilderQuizResult[];
  course: CourseBuilderResponse;
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
  learner_state?: LearnerState;
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
  concept_id?: string;
  concept?: string;
  phase_id?: string;
  objective_id?: string;
  source_chunk_id?: string;
}

export interface QuizQuestionTrueFalse {
  type: "true_false";
  bloom_level?: string;
  question: string;
  answer: boolean;
  explanation?: string;
  concept_id?: string;
  concept?: string;
  phase_id?: string;
  objective_id?: string;
  source_chunk_id?: string;
}

export interface QuizQuestionFillBlank {
  type: "fill_blank";
  bloom_level?: string;
  question: string;
  answer: string;
  accepted_answers?: string[];
  explanation?: string;
  concept_id?: string;
  concept?: string;
  phase_id?: string;
  objective_id?: string;
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

export interface ChartArtifactMetadata {
  mermaid_code?: string;
  diagram_type?: string;
}

export interface MindmapPayload {
  markdown: string;
  central_topic?: string;
  main_branches?: string[];
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
