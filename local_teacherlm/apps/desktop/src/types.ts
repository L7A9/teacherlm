export type Chunk = {
  text: string;
  source: string;
  score: number;
  chunk_id: string;
  metadata: Record<string, unknown>;
};

export type Artifact = {
  type: string;
  url: string;
  filename: string;
  key?: string | null;
  mime_type?: string | null;
  created_at?: string | null;
};

export type MindmapNode = {
  text: string;
  children?: MindmapNode[];
};

export type MindmapPayload = {
  markdown?: string;
  central_topic?: string;
  main_branches?: string[];
  branches?: MindmapNode[];
};

export type Message = {
  id: string;
  conversation_id: string;
  role: "user" | "assistant" | string;
  content: string;
  output_type: string;
  artifacts: Artifact[];
  sources: Chunk[];
  metadata: Record<string, unknown>;
  created_at: string;
};

export type Conversation = {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
};

export type SourceFile = {
  id: string;
  conversation_id: string;
  filename: string;
  status:
    | "uploaded"
    | "parsing"
    | "chunking"
    | "extracting_concepts"
    | "planning_course"
    | "building_course"
    | "embedding"
    | "ready"
    | "failed"
    | string;
  chunk_count?: number;
  parser_used?: string | null;
  error?: string | null;
  created_at: string;
  updated_at?: string;
};

export type LearnerState = {
  conversation_id: string;
  understood_concepts: string[];
  struggling_concepts: string[];
  mastery_scores: Record<string, number>;
  session_turns: number;
};

export type GeneratorManifest = {
  generator_id: string;
  display_name: string;
  output_type: string;
  enabled: boolean;
  transport: string;
  retrieval_mode: string;
  artifact_types: string[];
  is_chat_default?: boolean;
};

export type ProviderRead = {
  id: string;
  display_name: string;
  provider_type: string;
  base_url: string;
  model_name: string;
  api_key_set: boolean;
  is_default_chat: boolean;
  is_default_embedding: boolean;
  status: string;
};

export type ParserSettings = {
  llama_cloud_api_key_set: boolean;
  use_local_parsers_only: boolean;
  status: string;
};

export type RetrievalSettings = {
  embedding_model: string;
  embedding_dim: number;
  embedding_batch_size: number;
  embedding_model_candidates: string[];
  retrieval_top_k: number;
  retrieval_dense_candidate_k: number;
  retrieval_sparse_candidate_k: number;
  retrieval_hyde_enabled: boolean;
  retrieval_rerank_enabled: boolean;
  retrieval_reranker_model: string;
  retrieval_graph_enabled: boolean;
  index_status: {
    embedding_model?: string;
    embedding_dim?: number;
    embedding_batch_size?: number;
    chunk_count?: number;
    embedded_chunk_count?: number;
    stale_chunk_count?: number;
    graph_node_count?: number;
    graph_edge_count?: number;
    ready?: boolean;
    [key: string]: unknown;
  };
};

export type CourseCitation = {
  chunk_id: string;
  source: string;
  section?: string;
  snippet?: string;
};

export type CourseLessonBlock = {
  id: string;
  block_type: string;
  title: string;
  content: string;
  data_json?: {
    expression?: string;
    headers?: string[];
    rows?: string[][];
    events?: Array<{ date: string; description: string }>;
    [key: string]: unknown;
  };
  source_chunk_ids: string[];
  citations: CourseCitation[];
};

export type CourseQuizOption = {
  id: string;
  text: string;
};

export type CourseQuizQuestion = {
  id: string;
  type: "mcq" | string;
  prompt: string;
  options: CourseQuizOption[];
  source_chunk_ids: string[];
  citations: CourseCitation[];
};

export type CourseQuiz = {
  id: string;
  title: string;
  scope: "chapter" | "course" | string;
  questions: CourseQuizQuestion[];
  pass_score: number;
  is_locked?: boolean;
  is_passed?: boolean;
  attempt_count?: number;
};

export type CourseLesson = {
  id: string;
  title: string;
  order_index?: number;
  summary: string;
  learning_objectives: string[];
  prerequisite_lesson_ids?: string[];
  source_chunk_ids: string[];
  citations: CourseCitation[];
  blocks: CourseLessonBlock[];
  content_fingerprint?: string;
  generation_status?: "pending" | "building" | "ready" | string;
  lesson_stage?: "introduction" | "content" | "conclusion" | string;
  is_locked?: boolean;
  is_completed?: boolean;
};

export type CourseChapter = {
  id: string;
  title: string;
  order_index?: number;
  description: string;
  summary: string;
  learning_objectives?: string[];
  prerequisite_chapter_ids?: string[];
  source_chunk_ids: string[];
  citations: CourseCitation[];
  lessons: CourseLesson[];
  quiz?: CourseQuiz;
  generation_status?: "pending" | "building" | "ready" | string;
  is_locked?: boolean;
  is_complete?: boolean;
};

export type CourseProgress = {
  completed_lesson_ids: string[];
  passed_quiz_ids: string[];
  quiz_scores: Record<string, number>;
  quiz_attempt_counts: Record<string, number>;
  completed_lesson_count: number;
  passed_chapter_count: number;
  course_completed: boolean;
};

export type CourseBuilderRead = {
  id?: string;
  status: "empty" | "waiting_for_files" | "building" | "ready" | "failed" | string;
  title?: string;
  description?: string;
  learning_objectives?: string[];
  chapters: CourseChapter[];
  final_quiz?: CourseQuiz | null;
  progress?: CourseProgress;
  metadata?: Record<string, unknown>;
  files_total?: number;
  files_pending?: number;
  files_failed?: number;
  course_plan?: Record<string, unknown>;
};

export type CourseQuizSubmissionResult = {
  score: number;
  passed: boolean;
  pass_score: number;
  attempt_count: number;
  results: Array<{
    question_id: string;
    selected_option_id: string;
    correct_option_id: string;
    correct: boolean;
    explanation: string;
  }>;
  review_lesson_ids: string[];
  course: CourseBuilderRead;
};

export type StreamEvent =
  | { event: "analysis"; data: unknown }
  | { event: "progress"; data: unknown }
  | { event: "sources"; data: Chunk[] }
  | { event: "token"; data: string }
  | { event: "artifact"; data: Artifact }
  | { event: "done"; data: { response: string; artifacts: Artifact[]; sources: Chunk[]; output_type: string } }
  | { event: "error"; data: { message: string } }
  | { event: string; data: unknown };
