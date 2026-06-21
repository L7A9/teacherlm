import React, {
  ChangeEvent,
  DragEvent,
  FormEvent,
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
} from "react";
import { createPortal } from "react-dom";
import {
  AlertCircle,
  ArrowLeft,
  BarChart3,
  BookOpen,
  Bot,
  Check,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Copy,
  Download,
  Eye,
  FileText,
  FolderOpen,
  GraduationCap,
  GripVertical,
  KeyRound,
  Loader2,
  MessageCircle,
  MessageSquare,
  Mic2,
  Moon,
  Network,
  Package,
  Palette,
  PanelLeft,
  PanelRight,
  Play,
  Plus,
  Presentation,
  RotateCcw,
  Save,
  ScrollText,
  Send,
  Server,
  Settings,
  Sparkles,
  Square,
  Sun,
  Trash2,
  UploadCloud,
  User,
  X,
} from "lucide-react";
import { api, streamChat, streamGenerate } from "./api";
import { AssistantMarkdown } from "./components/AssistantMarkdown";
import {
  cleanCourseCitationSnippet,
  dedupeCourseBlocks,
  shouldShowLessonSummary,
} from "./components/courseContent";
import { MindmapRenderer } from "./components/MindmapRenderer";
import type {
  Artifact,
  Conversation,
  CourseBuilderRead,
  CourseBuilderSettings,
  CourseChapter,
  CourseLesson,
  CourseLessonBlock,
  CourseQuiz,
  CourseQuizSubmissionResult,
  GeneratorManifest,
  GeneratorSettings,
  LearnerState,
  Message,
  MindmapPayload,
  ParserSettings,
  ProviderRead,
  RetrievalSettings,
  SourceFile,
  StreamEvent,
} from "./types";

type Theme = "dark" | "light";
type AppRoute =
  | { kind: "home" }
  | { kind: "settings" }
  | { kind: "conversation"; conversationId: string };

type ProviderForm = {
  display_name: string;
  provider_type: string;
  base_url: string;
  model_name: string;
  api_key: string;
  is_default_chat: boolean;
};

type QuizQuestionType = "mcq" | "true_false";

type QuizForm = {
  question_count: number;
  question_type: QuizQuestionType;
};

type PodcastForm = {
  topic: string;
  duration_minutes: number;
};

type ButtonVariant = "primary" | "secondary" | "ghost" | "danger" | "link";
type ButtonSize = "sm" | "md" | "lg" | "icon";

const DEFAULT_PROVIDER_FORM: ProviderForm = {
  display_name: "Ollama",
  provider_type: "ollama",
  base_url: "http://localhost:11434",
  model_name: "llama3.2",
  api_key: "",
  is_default_chat: true,
};

const DEFAULT_QUIZ_FORM: QuizForm = {
  question_count: 8,
  question_type: "mcq",
};

const DEFAULT_PODCAST_FORM: PodcastForm = {
  topic: "",
  duration_minutes: 6,
};

const OUTPUT_BUTTONS = [
  { output_type: "text", display_name: "Chat", hint: "Talk to your teacher", Icon: GraduationCap },
  { output_type: "quiz", display_name: "Quiz", hint: "Test yourself", Icon: FileText },
  { output_type: "report", display_name: "Report", hint: "Study report", Icon: ScrollText },
  { output_type: "chart", display_name: "Diagram", hint: "Concept diagram", Icon: BarChart3 },
  { output_type: "mindmap", display_name: "Mind map", hint: "Bird's-eye view of your materials", Icon: Network },
  { output_type: "podcast", display_name: "Podcast", hint: "Listen-along audio", Icon: Mic2 },
  { output_type: "presentation", display_name: "Presentation", hint: "Slide deck", Icon: Presentation },
];

const PIPELINE_STATUSES = new Set([
  "uploaded",
  "parsing",
  "chunking",
  "extracting_concepts",
  "planning_course",
  "building_course",
  "embedding",
]);

export default function App() {
  const [route, setRoute] = useState<AppRoute>(() => routeFromLocation());
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [conversation, setConversation] = useState<Conversation | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [files, setFiles] = useState<SourceFile[]>([]);
  const [selectedFiles, setSelectedFiles] = useState<string[]>([]);
  const [learner, setLearner] = useState<LearnerState | null>(null);
  const [generators, setGenerators] = useState<GeneratorManifest[]>([]);
  const [providers, setProviders] = useState<ProviderRead[]>([]);
  const [parser, setParser] = useState<ParserSettings | null>(null);
  const [courseBuilderSettings, setCourseBuilderSettings] = useState<CourseBuilderSettings | null>(null);
  const [generatorSettings, setGeneratorSettings] = useState<GeneratorSettings | null>(null);
  const [retrieval, setRetrieval] = useState<RetrievalSettings | null>(null);
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [course, setCourse] = useState<CourseBuilderRead | null>(null);
  const [input, setInput] = useState("");
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [chatStreaming, setChatStreaming] = useState(false);
  const [status, setStatus] = useState("Starting");
  const [bootstrapped, setBootstrapped] = useState(false);
  const [editingTitle, setEditingTitle] = useState(false);
  const [titleDraft, setTitleDraft] = useState("");
  const [conversationToDelete, setConversationToDelete] = useState<Conversation | null>(null);
  const [homeBusyId, setHomeBusyId] = useState<string | null>(null);
  const [editingProviderId, setEditingProviderId] = useState<string | null>(null);
  const [theme, setTheme] = useState<Theme>(() => readSavedTheme());
  const [parserKey, setParserKey] = useState("");
  const [retrievalBusy, setRetrievalBusy] = useState(false);
  const [courseBuilderSettingsBusy, setCourseBuilderSettingsBusy] = useState(false);
  const [courseBuilderSettingsError, setCourseBuilderSettingsError] = useState("");
  const [generatorSettingsBusy, setGeneratorSettingsBusy] = useState(false);
  const [generatorSettingsError, setGeneratorSettingsError] = useState("");
  const [settingsConversationId, setSettingsConversationId] = useState<string | null>(null);
  const [sourcesCollapsed, setSourcesCollapsed] = useState(false);
  const [progressCollapsed, setProgressCollapsed] = useState(false);
  const [providerForm, setProviderForm] = useState<ProviderForm>(DEFAULT_PROVIDER_FORM);
  const [quizDialogOpen, setQuizDialogOpen] = useState(false);
  const [quizForm, setQuizForm] = useState<QuizForm>(DEFAULT_QUIZ_FORM);
  const [podcastDialogOpen, setPodcastDialogOpen] = useState(false);
  const [podcastForm, setPodcastForm] = useState<PodcastForm>(DEFAULT_PODCAST_FORM);
  const activeChatController = useRef<AbortController | null>(null);

  const routeConversationId = route.kind === "conversation" ? route.conversationId : null;

  useEffect(() => {
    void bootstrap();
  }, []);

  useEffect(() => {
    const handlePopState = () => setRoute(routeFromLocation());
    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, []);

  useEffect(() => {
    if (!bootstrapped) return;
    if (route.kind === "conversation") {
      void syncRouteConversation(route.conversationId);
      return;
    }
    clearConversationData();
    setStatus("Ready");
    if (route.kind === "home") {
      void refreshConversations(null);
    }
  }, [bootstrapped, route.kind, routeConversationId]);

  useEffect(() => {
    if (!conversation?.id) return;
    if (!files.some((file) => PIPELINE_STATUSES.has(file.status)) && course?.status !== "building") return;
    const timer = window.setInterval(() => {
      void loadConversation(conversation.id);
    }, 3000);
    return () => window.clearInterval(timer);
  }, [conversation?.id, files, course?.status]);

  useEffect(() => {
    applyTheme(theme);
  }, [theme]);

  async function bootstrap() {
    setStatus("Connecting");
    await api.health();
    const [conversationResponse, generatorResponse, providerResponse, parserResponse, courseBuilderResponse, generatorSettingsResponse, retrievalResponse] = await Promise.all([
      api.listConversations(),
      api.generators(),
      api.providers(),
      api.parserSettings(),
      api.coursebuilderSettings(),
      api.generatorSettings(),
      api.retrievalSettings(),
    ]);
    setConversations(conversationResponse.conversations);
    clearConversationData();
    setGenerators(generatorResponse.generators);
    setProviders(providerResponse.providers);
    setParser(parserResponse);
    setCourseBuilderSettings(courseBuilderResponse);
    setGeneratorSettings(generatorSettingsResponse);
    setRetrieval(retrievalResponse);
    setStatus("Ready");
    setBootstrapped(true);
  }

  async function syncRouteConversation(conversationId: string) {
    setStatus("Loading");
    let active: Conversation | null = null;
    try {
      const response = await api.listConversations();
      setConversations(response.conversations);
      active = response.conversations.find((item) => item.id === conversationId) ?? null;
      active = active ?? (await api.getConversation(conversationId));
    } catch {
      active = null;
    }

    if (!active) {
      clearConversationData();
      navigateHome(true);
      setStatus("Ready");
      return;
    }

    setConversation(active);
    setTitleDraft(active.title);
    setEditingTitle(false);
    setInput("");
    setDraft("");
    await loadConversation(active.id, active);
    setStatus("Ready");
  }

  function clearConversationData() {
    activeChatController.current?.abort();
    activeChatController.current = null;
    setChatStreaming(false);
    setConversation(null);
    setTitleDraft("");
    setEditingTitle(false);
    setMessages([]);
    setFiles([]);
    setSelectedFiles([]);
    setLearner(null);
    setArtifacts([]);
    setCourse(null);
    setInput("");
    setDraft("");
  }

  function setAppRoute(next: AppRoute, replace = false) {
    const nextPath = pathForRoute(next);
    if (window.location.pathname !== nextPath) {
      if (replace) {
        window.history.replaceState({}, "", nextPath);
      } else {
        window.history.pushState({}, "", nextPath);
      }
    }
    setRoute(next);
  }

  function navigateToConversation(conversationId: string, replace = false) {
    setAppRoute({ kind: "conversation", conversationId }, replace);
  }

  function navigateHome(replace = false) {
    setAppRoute({ kind: "home" }, replace);
  }

  function navigateSettings(replace = false) {
    setSettingsConversationId(conversation?.id ?? routeConversationId ?? null);
    setAppRoute({ kind: "settings" }, replace);
  }

  async function loadConversation(conversationId = conversation?.id, activeOverride?: Conversation) {
    if (!conversationId) return;
    const active = activeOverride ?? conversations.find((item) => item.id === conversationId) ?? conversation;
    const [messageResponse, fileResponse, learnerResponse, artifactResponse, courseResponse] = await Promise.all([
      api.listMessages(conversationId),
      api.listFiles(conversationId),
      api.learnerState(conversationId),
      api.artifacts(conversationId),
      api.coursebuilder(conversationId),
    ]);
    if (active?.id === conversationId) {
      setConversation(active);
      setTitleDraft(active.title);
    }
    setMessages(messageResponse.messages);
    setFiles(fileResponse.files);
    setLearner(learnerResponse);
    setArtifacts(artifactResponse.artifacts);
    setCourse(courseResponse);
    setSelectedFiles((current) => {
      const readyIds = new Set(fileResponse.files.filter((file) => file.status === "ready").map((file) => file.id));
      const kept = current.filter((id) => readyIds.has(id));
      return kept.length ? kept : [...readyIds];
    });
  }

  async function refreshConversations(activeId?: string | null) {
    const response = await api.listConversations();
    setConversations(response.conversations);
    if (activeId) {
      const nextActive = response.conversations.find((item) => item.id === activeId);
      if (nextActive) {
        setConversation(nextActive);
        setTitleDraft(nextActive.title);
      } else if (conversation?.id === activeId) {
        clearConversationData();
      }
    }
  }

  async function createConversation() {
    setStatus("Creating");
    const created = await api.createConversation("New conversation");
    setConversations((current) => [created, ...current.filter((item) => item.id !== created.id)]);
    navigateToConversation(created.id);
  }

  async function openConversation(target: Conversation | string) {
    const next = typeof target === "string" ? conversations.find((item) => item.id === target) : target;
    if (!next) return;
    navigateToConversation(next.id);
  }

  async function deleteConversation(conversationId: string) {
    setHomeBusyId(conversationId);
    setStatus("Deleting");
    try {
      await api.deleteConversation(conversationId);
      setConversationToDelete(null);
      const isActive = conversation?.id === conversationId || routeConversationId === conversationId;
      setConversations((current) => current.filter((item) => item.id !== conversationId));
      if (isActive) {
        navigateHome(true);
      } else {
        await refreshConversations(routeConversationId);
      }
      setStatus("Ready");
    } finally {
      setHomeBusyId(null);
    }
  }

  async function saveConversationTitle() {
    if (!conversation) return;
    const title = titleDraft.trim();
    if (!title) {
      setTitleDraft(conversation.title);
      setEditingTitle(false);
      return;
    }
    if (title === conversation.title) {
      setEditingTitle(false);
      return;
    }
    const updated = await api.updateConversation(conversation.id, { title });
    setConversation(updated);
    setTitleDraft(updated.title);
    setEditingTitle(false);
    await refreshConversations(updated.id);
  }

  async function handleUpload(event: ChangeEvent<HTMLInputElement>) {
    if (!event.target.files?.length) return;
    await uploadFiles(event.target.files);
    event.target.value = "";
  }

  async function uploadFiles(fileList: FileList | File[]) {
    if (!conversation || fileList.length === 0) return;
    setBusy(true);
    setStatus("Indexing");
    try {
      const response = await api.uploadFiles(conversation.id, Array.from(fileList));
      const uploadedIds = new Set(response.files.map((file) => file.id));
      setFiles((current) => [...response.files, ...current.filter((item) => !uploadedIds.has(item.id))]);
      await loadConversation(conversation.id);
      setStatus("Ready");
    } finally {
      setBusy(false);
    }
  }

  async function deleteFile(fileId: string) {
    if (!conversation) return;
    setBusy(true);
    try {
      await api.deleteFile(conversation.id, fileId);
      await loadConversation(conversation.id);
    } finally {
      setBusy(false);
    }
  }

  async function retryFile(fileId: string) {
    if (!conversation) return;
    setBusy(true);
    try {
      const retried = await api.retryFile(conversation.id, fileId);
      setFiles((current) => current.map((item) => (item.id === retried.id ? retried : item)));
      await loadConversation(conversation.id);
    } finally {
      setBusy(false);
    }
  }

  async function submitChat(event?: FormEvent) {
    event?.preventDefault();
    if (!conversation || !input.trim() || busy) return;
    const message = input.trim();
    setInput("");
    setBusy(true);
    setChatStreaming(true);
    setDraft("");
    setMessages((current) => [
      ...current,
      optimisticMessage(conversation.id, "user", message, "text"),
    ]);
    const controller = new AbortController();
    activeChatController.current = controller;
    try {
      await streamChat(conversation.id, message, selectedFiles, handleStreamEvent, controller.signal);
      await loadConversation(conversation.id);
      setDraft("");
    } catch (error) {
      if (isAbortError(error)) {
        setDraft("");
        await loadConversation(conversation.id);
      } else {
        setDraft(error instanceof Error ? error.message : "Chat failed");
      }
    } finally {
      if (activeChatController.current === controller) {
        activeChatController.current = null;
      }
      setChatStreaming(false);
      setBusy(false);
    }
  }

  function stopChat() {
    activeChatController.current?.abort();
  }

  function handleGeneratorSelect(outputType: string) {
    if (outputType === "quiz") {
      setQuizDialogOpen(true);
      return;
    }
    if (outputType === "podcast") {
      setPodcastDialogOpen(true);
      return;
    }
    void runGenerator(outputType);
  }

  async function runGenerator(
    outputType: string,
    options: Record<string, unknown> = {},
    promptOverride?: string,
    preserveInput = false,
  ) {
    if (!conversation || busy) return;
    const sourceFileIds = [...selectedFiles];
    if (outputType === "mindmap" && sourceFileIds.length === 0) return;
    setBusy(true);
    setDraft(
      outputType === "quiz"
        ? "Generating a fresh quiz from all selected chunks and their knowledge graph…"
        : "",
    );
    const prompt = promptOverride?.trim() || input.trim() || `Generate ${outputType}`;
    if (!preserveInput) setInput("");
    setMessages((current) => [
      ...current,
      optimisticMessage(conversation.id, "user", prompt, outputType),
    ]);
    try {
      await streamGenerate(conversation.id, outputType, prompt, sourceFileIds, handleStreamEvent, options);
      await loadConversation(conversation.id);
      setDraft("");
    } catch (error) {
      setDraft(error instanceof Error ? error.message : `${outputType} generation failed`);
    } finally {
      setBusy(false);
    }
  }

  function startQuizFromDialog() {
    const questionCount = Math.max(1, Math.min(20, Math.trunc(Number(quizForm.question_count) || DEFAULT_QUIZ_FORM.question_count)));
    setQuizForm((current) => ({ ...current, question_count: questionCount }));
    setQuizDialogOpen(false);
    void runGenerator("quiz", {
      question_count: questionCount,
      question_type: quizForm.question_type,
      question_types: [quizForm.question_type],
    });
  }

  function startPodcastFromDialog() {
    const durationMinutes = Math.max(
      3,
      Math.min(15, Math.trunc(Number(podcastForm.duration_minutes) || DEFAULT_PODCAST_FORM.duration_minutes)),
    );
    const topic = podcastForm.topic.trim().slice(0, 200);
    setPodcastForm({ topic, duration_minutes: durationMinutes });
    setPodcastDialogOpen(false);
    const prompt = topic || `Generate a ${durationMinutes}-minute podcast from the selected course material`;
    void runGenerator("podcast", { topic, duration_minutes: durationMinutes }, prompt, true);
  }

  function handleStreamEvent(event: StreamEvent) {
    if (event.event === "progress" && typeof event.data === "object" && event.data && "stage" in event.data) {
      const stage = String(event.data.stage);
      if (stage === "sending_full_context_to_llm") {
        setDraft("The model is generating your quiz from the selected files. This can take about a minute…");
      } else if (stage === "llm_quiz_validated") {
        setDraft("Quiz generated—validating answers and sources…");
      } else if (stage === "podcast_extracting_arc") {
        setDraft("Planning the podcast from your selected sources…");
      } else if (stage === "podcast_writing_dialogue") {
        setDraft("Alex and Sam are writing their grounded dialogue…");
      } else if (stage === "podcast_transcript_ready") {
        setDraft("Transcript ready—preparing the local voices…");
      } else if (stage === "podcast_preparing_voices") {
        setDraft("Preparing the local voice model. The first run may download about 23 MB…");
      } else if (stage === "podcast_assembling_audio") {
        setDraft("Assembling and encoding the podcast audio…");
      }
    }
    if (event.event === "token" && typeof event.data === "string") {
      setDraft(event.data);
    }
    if (event.event === "artifact") {
      setArtifacts((current) => [event.data as Artifact, ...current]);
    }
    if (event.event === "error") {
      const message =
        typeof event.data === "object" && event.data && "message" in event.data
          ? String(event.data.message)
          : "Stream failed";
      setDraft(message);
    }
  }

  async function saveProvider(event: FormEvent) {
    event.preventDefault();
    const apiKey = providerForm.api_key.trim();
    if (editingProviderId) {
      const payload: Record<string, unknown> = {
        display_name: providerForm.display_name.trim(),
        provider_type: providerForm.provider_type,
        base_url: providerForm.base_url.trim(),
        model_name: providerForm.model_name.trim(),
        is_default_chat: providerForm.is_default_chat,
      };
      if (apiKey) {
        payload.api_key = apiKey;
      }
      await api.patchProvider(editingProviderId, payload);
    } else {
      await api.createProvider({
        display_name: providerForm.display_name.trim(),
        provider_type: providerForm.provider_type,
        base_url: providerForm.base_url.trim(),
        model_name: providerForm.model_name.trim(),
        api_key: apiKey || null,
        is_default_chat: providerForm.is_default_chat,
      });
    }
    await refreshProviders();
    resetProviderForm();
  }

  async function refreshProviders() {
    const response = await api.providers();
    setProviders(response.providers);
  }

  function resetProviderForm() {
    setEditingProviderId(null);
    setProviderForm(DEFAULT_PROVIDER_FORM);
  }

  function editProvider(provider: ProviderRead) {
    setEditingProviderId(provider.id);
    setProviderForm({
      display_name: provider.display_name,
      provider_type: provider.provider_type,
      base_url: provider.base_url,
      model_name: provider.model_name,
      api_key: "",
      is_default_chat: provider.is_default_chat,
    });
  }

  async function setDefaultProvider(providerId: string) {
    await api.patchProvider(providerId, { is_default_chat: true });
    await refreshProviders();
    if (editingProviderId === providerId) {
      setProviderForm((current) => ({ ...current, is_default_chat: true }));
    }
  }

  async function deleteProvider(providerId: string) {
    await api.deleteProvider(providerId);
    if (editingProviderId === providerId) {
      resetProviderForm();
    }
    await refreshProviders();
  }

  async function testProvider(providerId: string) {
    const updated = await api.testProvider(providerId);
    setProviders((current) => current.map((provider) => (provider.id === providerId ? updated : provider)));
  }

  async function setParserMode(useLocalParsersOnly: boolean) {
    const updated = await api.updateParserSettings({
      use_local_parsers_only: useLocalParsersOnly,
    });
    setParser(updated);
  }

  async function saveParserKey(event: FormEvent) {
    event.preventDefault();
    if (!parserKey.trim()) return;
    const updated = await api.updateParserSettings({
      llama_cloud_api_key: parserKey.trim(),
      use_local_parsers_only: false,
    });
    setParser(updated);
    setParserKey("");
  }

  async function clearParserKey() {
    const updated = await api.updateParserSettings({
      clear_llama_cloud_api_key: true,
    });
    setParser(updated);
    setParserKey("");
  }

  async function setSequentialCourseUnlocking(enabled: boolean) {
    setCourseBuilderSettingsBusy(true);
    setCourseBuilderSettingsError("");
    try {
      const updated = await api.updateCoursebuilderSettings({
        sequential_unlocking_enabled: enabled,
      });
      setCourseBuilderSettings(updated);
    } catch (error) {
      setCourseBuilderSettingsError(error instanceof Error ? error.message : "Could not save course progression.");
    } finally {
      setCourseBuilderSettingsBusy(false);
    }
  }

  async function setPodcastAudioEnabled(enabled: boolean) {
    setGeneratorSettingsBusy(true);
    setGeneratorSettingsError("");
    try {
      const updated = await api.updateGeneratorSettings({
        podcast_audio_enabled: enabled,
      });
      setGeneratorSettings(updated);
    } catch (error) {
      setGeneratorSettingsError(error instanceof Error ? error.message : "Could not save podcast audio preference.");
    } finally {
      setGeneratorSettingsBusy(false);
    }
  }

  async function rebuildIndexes() {
    if (!settingsConversationId) return;
    setRetrievalBusy(true);
    try {
      const rebuilt = await api.rebuildIndexes(settingsConversationId);
      const refreshed = await api.retrievalSettings();
      setRetrieval({
        ...refreshed,
        index_status: { ...refreshed.index_status, ...rebuilt.index_status },
      });
    } finally {
      setRetrievalBusy(false);
    }
  }

  const enabledOutputs = useMemo(
    () => generators.filter((generator) => generator.enabled && generator.output_type !== "text"),
    [generators],
  );
  const readyFiles = files.filter((file) => file.status === "ready");
  const actionsDisabled = readyFiles.length === 0 || selectedFiles.length === 0;
  const disabledReason =
    readyFiles.length === 0 ? "Wait until at least one course file is ready." : "Select at least one source file.";
  const hint = buildHint(learner);

  if (route.kind === "settings") {
    return (
      <SettingsPage
        providers={providers}
        parser={parser}
        courseBuilderSettings={courseBuilderSettings}
        generatorSettings={generatorSettings}
        retrieval={retrieval}
        providerForm={providerForm}
        editingProviderId={editingProviderId}
        parserKey={parserKey}
        retrievalBusy={retrievalBusy}
        courseBuilderSettingsBusy={courseBuilderSettingsBusy}
        courseBuilderSettingsError={courseBuilderSettingsError}
        generatorSettingsBusy={generatorSettingsBusy}
        generatorSettingsError={generatorSettingsError}
        canRebuildIndexes={Boolean(settingsConversationId)}
        theme={theme}
        onBack={() => navigateHome()}
        onProviderFormChange={setProviderForm}
        onEditProvider={editProvider}
        onCancelProviderEdit={resetProviderForm}
        onSetDefaultProvider={(providerId) => void setDefaultProvider(providerId)}
        onDeleteProvider={(providerId) => void deleteProvider(providerId)}
        onParserKeyChange={setParserKey}
        onThemeChange={setTheme}
        onSaveProvider={saveProvider}
        onSaveParserKey={saveParserKey}
        onClearParserKey={() => void clearParserKey()}
        onTestProvider={(providerId) => void testProvider(providerId)}
        onSetParserMode={(useLocalParsersOnly) => void setParserMode(useLocalParsersOnly)}
        onSetSequentialCourseUnlocking={(enabled) => void setSequentialCourseUnlocking(enabled)}
        onSetPodcastAudioEnabled={(enabled) => void setPodcastAudioEnabled(enabled)}
        onRebuildIndexes={() => void rebuildIndexes()}
      />
    );
  }

  if (!conversation) {
    return (
      <>
        <HomePage
          conversations={conversations}
          loading={status === "Connecting" || (route.kind === "conversation" && status === "Loading")}
          creating={status === "Creating"}
          deletingId={homeBusyId}
          onCreate={() => void createConversation()}
          onOpen={(target) => void openConversation(target)}
          onRequestDelete={setConversationToDelete}
          onOpenSettings={() => navigateSettings()}
        />
        {conversationToDelete && (
          <DeleteConversationDialog
            conversation={conversationToDelete}
            pending={homeBusyId === conversationToDelete.id}
            onCancel={() => setConversationToDelete(null)}
            onConfirm={() => void deleteConversation(conversationToDelete.id)}
          />
        )}
      </>
    );
  }

  return (
    <>
      <WorkspacePage
        conversation={conversation}
        titleDraft={titleDraft}
        editingTitle={editingTitle}
        files={files}
        selectedFiles={selectedFiles}
        messages={messages}
        draft={draft}
        input={input}
        busy={busy}
        chatStreaming={chatStreaming}
        artifacts={artifacts}
        course={course}
        onCourseChange={setCourse}
        outputs={enabledOutputs}
        hint={hint}
        sourcesCollapsed={sourcesCollapsed}
        progressCollapsed={progressCollapsed}
        actionsDisabled={actionsDisabled}
        disabledReason={disabledReason}
        onNavigateHome={() => navigateHome()}
        onNavigateSettings={() => navigateSettings()}
        onToggleSources={() => setSourcesCollapsed((collapsed) => !collapsed)}
        onToggleProgress={() => setProgressCollapsed((collapsed) => !collapsed)}
        onTitleDraftChange={setTitleDraft}
        onEditingTitleChange={setEditingTitle}
        onSaveTitle={() => void saveConversationTitle()}
        onUploadFiles={(uploads) => void uploadFiles(uploads)}
        onDeleteFile={(fileId) => void deleteFile(fileId)}
        onRetryFile={(fileId) => void retryFile(fileId)}
        onToggleFile={(fileId, checked) => {
          setSelectedFiles((current) => (checked ? [...current, fileId] : current.filter((id) => id !== fileId)));
        }}
        onSubmitChat={(event) => void submitChat(event)}
        onStopChat={stopChat}
        onInputChange={setInput}
        onRunGenerator={handleGeneratorSelect}
      />
      {quizDialogOpen && (
        <QuizSetupDialog
          value={quizForm}
          busy={busy}
          onChange={setQuizForm}
          onCancel={() => setQuizDialogOpen(false)}
          onConfirm={startQuizFromDialog}
        />
      )}
      {podcastDialogOpen && (
        <PodcastSetupDialog
          value={podcastForm}
          busy={busy}
          onChange={setPodcastForm}
          onCancel={() => setPodcastDialogOpen(false)}
          onConfirm={startPodcastFromDialog}
        />
      )}
      {conversationToDelete && (
        <DeleteConversationDialog
          conversation={conversationToDelete}
          pending={homeBusyId === conversationToDelete.id}
          onCancel={() => setConversationToDelete(null)}
          onConfirm={() => void deleteConversation(conversationToDelete.id)}
        />
      )}
    </>
  );
}

function HomePage({
  conversations,
  loading,
  creating,
  deletingId,
  onCreate,
  onOpen,
  onRequestDelete,
  onOpenSettings,
}: {
  conversations: Conversation[];
  loading: boolean;
  creating: boolean;
  deletingId: string | null;
  onCreate: () => void;
  onOpen: (conversation: Conversation) => void;
  onRequestDelete: (conversation: Conversation) => void;
  onOpenSettings: () => void;
}) {
  return (
    <main className="min-h-dvh bg-background text-foreground">
      <header className="app-chrome app-pane sticky top-0 z-10 border-b border-border">
        <div className="mx-auto flex max-w-5xl items-center justify-between px-4 py-3 sm:px-6">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-md bg-primary/15 text-primary">
              <GraduationCap className="h-5 w-5" />
            </div>
            <div>
              <h1 className="text-lg font-semibold">TeacherLM</h1>
              <p className="hidden text-xs text-muted-foreground sm:block">
                Your AI teacher, grounded in the files you upload.
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <Button variant="ghost" size="icon" title="Settings" onClick={onOpenSettings} aria-label="Settings">
              <Settings className="h-4 w-4" />
            </Button>
            <Button variant="primary" onClick={onCreate} disabled={creating} className="px-3 sm:px-4">
              {creating ? <Loader2 className="h-4 w-4 animate-spin" /> : <Plus className="h-4 w-4" />}
              <span className="hidden sm:inline">New conversation</span>
              <span className="sm:hidden">New</span>
            </Button>
          </div>
        </div>
      </header>

      <div className="mx-auto max-w-5xl px-4 py-6 sm:px-6">
        {loading ? (
          <LoadingState />
        ) : conversations.length === 0 ? (
          <EmptyState onCreate={onCreate} pending={creating} />
        ) : (
          <ConversationList
            items={conversations}
            deletingId={deletingId}
            onOpen={onOpen}
            onRequestDelete={onRequestDelete}
          />
        )}
      </div>
    </main>
  );
}

function LoadingState() {
  return (
    <div className="app-chrome flex items-center justify-center gap-2 py-16 text-sm text-muted-foreground">
      <Loader2 className="h-4 w-4 animate-spin" />
      Loading conversations...
    </div>
  );
}

function EmptyState({
  onCreate,
  pending,
}: {
  onCreate: () => void;
  pending: boolean;
}) {
  return (
    <div className="app-chrome flex flex-col items-center gap-4 rounded-md border border-dashed border-border bg-surface px-6 py-16 text-center">
      <div className="flex h-10 w-10 items-center justify-center rounded-md bg-primary/15 text-primary">
        <Sparkles className="h-5 w-5" />
      </div>
      <div className="flex flex-col gap-1">
        <h2 className="text-base font-semibold">Start your first conversation</h2>
        <p className="max-w-md text-sm text-muted-foreground">
          Create a conversation, upload your course files, and chat with a teacher that stays grounded in what you gave it.
        </p>
      </div>
      <Button variant="primary" onClick={onCreate} disabled={pending}>
        {pending ? <Loader2 className="h-4 w-4 animate-spin" /> : <Plus className="h-4 w-4" />}
        New conversation
      </Button>
    </div>
  );
}

function ConversationList({
  items,
  deletingId,
  onOpen,
  onRequestDelete,
}: {
  items: Conversation[];
  deletingId: string | null;
  onOpen: (conversation: Conversation) => void;
  onRequestDelete: (conversation: Conversation) => void;
}) {
  return (
    <ul className="overflow-hidden rounded-md border border-border bg-surface">
      {items.map((conversation) => (
        <li
          key={conversation.id}
          className="group relative border-b border-border transition-colors last:border-b-0 hover:bg-muted/60"
        >
          <a
            href={conversationPath(conversation.id)}
            className="flex flex-col gap-2 px-4 py-3"
            onClick={(event) => {
              event.preventDefault();
              onOpen(conversation);
            }}
          >
            <div className="flex items-start gap-2">
              <MessageSquare className="mt-0.5 h-4 w-4 shrink-0 text-primary" />
              <div className="min-w-0 flex-1">
                <div className="truncate text-sm font-medium">{conversation.title}</div>
                <div className="text-[11px] text-muted-foreground">
                  Updated {formatRelativeTime(conversation.updated_at)}
                </div>
              </div>
            </div>
          </a>

          <button
            type="button"
            aria-label="Delete conversation"
            onClick={(event) => {
              event.preventDefault();
              event.stopPropagation();
              onRequestDelete(conversation);
            }}
            disabled={deletingId === conversation.id}
            className={cn(
              "absolute right-2 top-2 rounded-md p-1.5 text-muted-foreground opacity-0 transition",
              "group-hover:opacity-100 hover:bg-muted hover:text-[hsl(var(--danger))]",
              "focus-visible:opacity-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
              deletingId === conversation.id && "opacity-100",
            )}
          >
            {deletingId === conversation.id ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <Trash2 className="h-3.5 w-3.5" />
            )}
          </button>
        </li>
      ))}
    </ul>
  );
}

function WorkspacePage({
  conversation,
  titleDraft,
  editingTitle,
  files,
  selectedFiles,
  messages,
  draft,
  input,
  busy,
  chatStreaming,
  artifacts,
  course,
  onCourseChange,
  outputs,
  hint,
  sourcesCollapsed,
  progressCollapsed,
  actionsDisabled,
  disabledReason,
  onNavigateHome,
  onNavigateSettings,
  onToggleSources,
  onToggleProgress,
  onTitleDraftChange,
  onEditingTitleChange,
  onSaveTitle,
  onUploadFiles,
  onDeleteFile,
  onRetryFile,
  onToggleFile,
  onSubmitChat,
  onStopChat,
  onInputChange,
  onRunGenerator,
}: {
  conversation: Conversation;
  titleDraft: string;
  editingTitle: boolean;
  files: SourceFile[];
  selectedFiles: string[];
  messages: Message[];
  draft: string;
  input: string;
  busy: boolean;
  chatStreaming: boolean;
  artifacts: Artifact[];
  course: CourseBuilderRead | null;
  onCourseChange: (course: CourseBuilderRead) => void;
  outputs: GeneratorManifest[];
  hint: string | null;
  sourcesCollapsed: boolean;
  progressCollapsed: boolean;
  actionsDisabled: boolean;
  disabledReason: string;
  onNavigateHome: () => void;
  onNavigateSettings: () => void;
  onToggleSources: () => void;
  onToggleProgress: () => void;
  onTitleDraftChange: (value: string) => void;
  onEditingTitleChange: (value: boolean) => void;
  onSaveTitle: () => void;
  onUploadFiles: (files: FileList | File[]) => void;
  onDeleteFile: (fileId: string) => void;
  onRetryFile: (fileId: string) => void;
  onToggleFile: (fileId: string, checked: boolean) => void;
  onSubmitChat: (event?: FormEvent) => void;
  onStopChat: () => void;
  onInputChange: (value: string) => void;
  onRunGenerator: (outputType: string) => void;
}) {
  const mainRef = useRef<HTMLDivElement | null>(null);
  const [courseWidth, setCourseWidth] = useState(48);
  const [isNarrow, setIsNarrow] = useState(() => window.innerWidth < 1024);
  const [mobileSourcesOpen, setMobileSourcesOpen] = useState(false);
  const [mobileGeneratedOpen, setMobileGeneratedOpen] = useState(false);
  const [mobileMainView, setMobileMainView] = useState<"course" | "chat">("course");
  const inputRef = useRef<HTMLTextAreaElement | null>(null);
  const sourcesVisible = isNarrow ? mobileSourcesOpen : !sourcesCollapsed;
  const generatedVisible = isNarrow ? mobileGeneratedOpen : !progressCollapsed;

  useEffect(() => {
    const update = () => setIsNarrow(window.innerWidth < 1024);
    update();
    window.addEventListener("resize", update);
    return () => window.removeEventListener("resize", update);
  }, []);

  useEffect(() => {
    if (!isNarrow) {
      setMobileSourcesOpen(false);
      setMobileGeneratedOpen(false);
      setMobileMainView("course");
      return;
    }

    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      setMobileSourcesOpen(false);
      setMobileGeneratedOpen(false);
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [isNarrow]);

  const closeMobileDrawers = () => {
    setMobileSourcesOpen(false);
    setMobileGeneratedOpen(false);
  };

  return (
    <div className="flex h-dvh flex-col bg-background text-foreground">
      <WorkspaceTopBar
        conversation={conversation}
        titleDraft={titleDraft}
        editingTitle={editingTitle}
        sourcesVisible={sourcesVisible}
        generatedVisible={generatedVisible}
        mobileMainView={mobileMainView}
        showMobileChatToggle={isNarrow}
        onNavigateHome={onNavigateHome}
        onNavigateSettings={onNavigateSettings}
        onTitleDraftChange={onTitleDraftChange}
        onEditingTitleChange={onEditingTitleChange}
        onSaveTitle={onSaveTitle}
        onToggleSources={() => {
          if (isNarrow) setMobileSourcesOpen((open) => !open);
          else onToggleSources();
        }}
        onToggleProgress={() => {
          if (isNarrow) setMobileGeneratedOpen((open) => !open);
          else onToggleProgress();
        }}
        onToggleMobileMainView={() => {
          setMobileSourcesOpen(false);
          setMobileGeneratedOpen(false);
          setMobileMainView((view) => (view === "chat" ? "course" : "chat"));
        }}
      />

      <div className="relative grid min-h-0 flex-1 grid-cols-1 overflow-hidden lg:grid-cols-[auto_minmax(0,1fr)_auto]">
        {isNarrow && (mobileSourcesOpen || mobileGeneratedOpen) && (
          <button
            type="button"
            aria-label="Close side panels"
            className="absolute inset-0 z-10 bg-background/60 backdrop-blur-sm lg:hidden"
            onClick={closeMobileDrawers}
          />
        )}
        {sourcesVisible && (
          <SourcesPanel
            files={files}
            selectedFiles={selectedFiles}
            busy={busy}
            onUploadFiles={onUploadFiles}
            onDeleteFile={onDeleteFile}
            onRetryFile={onRetryFile}
            onToggleFile={onToggleFile}
            onClose={closeMobileDrawers}
            className="absolute inset-y-0 left-0 z-20 w-[min(88vw,320px)] shadow-2xl lg:static lg:z-auto lg:h-full lg:w-[300px] lg:shadow-none"
          />
        )}
        <main
          ref={mainRef}
          className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden md:flex-row"
          aria-label="Learning workspace"
        >
          {isNarrow ? (
            <div className="min-h-0 min-w-0 flex-1">
              {mobileMainView === "chat" ? (
                <ChatPanel
                  conversationId={conversation.id}
                  messages={messages}
                  draft={draft}
                  input={input}
                  busy={busy}
                  chatStreaming={chatStreaming}
                  hint={hint}
                  outputs={outputs}
                  disabled={actionsDisabled}
                  disabledReason={disabledReason}
                  inputRef={inputRef}
                  onInputChange={onInputChange}
                  onSubmitChat={onSubmitChat}
                  onStopChat={onStopChat}
                  onRunGenerator={onRunGenerator}
                />
              ) : (
                <CoursePanel conversationId={conversation.id} files={files} course={course} onCourseChange={onCourseChange} />
              )}
            </div>
          ) : (
            <>
              <div
                className="min-h-0 min-w-0 flex-shrink-0 basis-[44%] border-border border-b md:basis-[var(--course-pane-width)] md:border-b-0 md:border-r"
                style={
                  {
                    "--course-pane-width": `clamp(320px, ${courseWidth}%, calc(100% - 360px))`,
                  } as CSSProperties
                }
              >
                <CoursePanel conversationId={conversation.id} files={files} course={course} onCourseChange={onCourseChange} />
              </div>
              <ResizeHandle
                onDrag={(clientX) => {
                  const rect = mainRef.current?.getBoundingClientRect();
                  if (!rect || rect.width <= 0) return;
                  const next = ((clientX - rect.left) / rect.width) * 100;
                  setCourseWidth(Math.min(68, Math.max(32, next)));
                }}
              />
              <div className="min-h-0 min-w-0 flex-1">
                <ChatPanel
                  conversationId={conversation.id}
                  messages={messages}
                  draft={draft}
                  input={input}
                  busy={busy}
                  chatStreaming={chatStreaming}
                  hint={hint}
                  outputs={outputs}
                  disabled={actionsDisabled}
                  disabledReason={disabledReason}
                  inputRef={inputRef}
                  onInputChange={onInputChange}
                  onSubmitChat={onSubmitChat}
                  onStopChat={onStopChat}
                  onRunGenerator={onRunGenerator}
                />
              </div>
            </>
          )}
        </main>
        {generatedVisible && (
          <GeneratedItemsPanel
            conversationId={conversation.id}
            messages={messages}
            artifacts={artifacts}
            onClose={closeMobileDrawers}
            className="absolute inset-y-0 right-0 z-20 w-[min(88vw,340px)] shadow-2xl lg:static lg:z-auto lg:h-full lg:w-[320px] lg:shadow-none"
          />
        )}
      </div>
    </div>
  );
}

function WorkspaceTopBar({
  conversation,
  titleDraft,
  editingTitle,
  sourcesVisible,
  generatedVisible,
  mobileMainView,
  showMobileChatToggle,
  onNavigateHome,
  onNavigateSettings,
  onTitleDraftChange,
  onEditingTitleChange,
  onSaveTitle,
  onToggleSources,
  onToggleProgress,
  onToggleMobileMainView,
}: {
  conversation: Conversation;
  titleDraft: string;
  editingTitle: boolean;
  sourcesVisible: boolean;
  generatedVisible: boolean;
  mobileMainView: "course" | "chat";
  showMobileChatToggle: boolean;
  onNavigateHome: () => void;
  onNavigateSettings: () => void;
  onTitleDraftChange: (value: string) => void;
  onEditingTitleChange: (value: boolean) => void;
  onSaveTitle: () => void;
  onToggleSources: () => void;
  onToggleProgress: () => void;
  onToggleMobileMainView: () => void;
}) {
  const mobileChatActive = mobileMainView === "chat";
  return (
    <header className="app-chrome app-pane flex h-12 shrink-0 items-center justify-between border-b border-border px-3 sm:px-4">
      <div className="flex min-w-0 items-center gap-2">
        <Button
          variant="ghost"
          size="icon"
          onClick={onToggleSources}
          aria-label={sourcesVisible ? "Hide sources" : "Show sources"}
          title={sourcesVisible ? "Hide sources" : "Show sources"}
        >
          <PanelLeft className="h-4 w-4" />
        </Button>
        <a
          href="/"
          className="flex items-center gap-2 rounded-md px-1 py-0.5 transition-colors hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          aria-label="Go to TeacherLM home"
          onClick={(event) => {
            event.preventDefault();
            onNavigateHome();
          }}
        >
          <div className="flex h-7 w-7 items-center justify-center rounded-md bg-primary/15 text-primary">
            <GraduationCap className="h-4 w-4" />
          </div>
          <span className="text-sm font-semibold tracking-tight">TeacherLM</span>
        </a>
        <div className="hidden h-5 w-px shrink-0 bg-border sm:block" />
        <EditableConversationTitle
          title={conversation.title}
          draft={titleDraft}
          editing={editingTitle}
          onDraftChange={onTitleDraftChange}
          onEditingChange={onEditingTitleChange}
          onSave={onSaveTitle}
        />
      </div>

      <div className="flex shrink-0 items-center gap-1">
        {showMobileChatToggle && (
          <Button
            variant="ghost"
            size="icon"
            onClick={onToggleMobileMainView}
            aria-label={mobileChatActive ? "Show course" : "Show chat"}
            title={mobileChatActive ? "Show course" : "Show chat"}
            className="lg:hidden"
          >
            {mobileChatActive ? <BookOpen className="h-4 w-4" /> : <MessageCircle className="h-4 w-4" />}
          </Button>
        )}
        <Button variant="ghost" size="icon" title="Settings" aria-label="Settings" onClick={onNavigateSettings}>
          <Settings className="h-4 w-4" />
        </Button>
        <Button
          variant="ghost"
          size="icon"
          onClick={onToggleProgress}
          aria-label={generatedVisible ? "Hide generated items" : "Show generated items"}
          title={generatedVisible ? "Hide generated items" : "Show generated items"}
        >
          <PanelRight className="h-4 w-4" />
        </Button>
      </div>
    </header>
  );
}

function EditableConversationTitle({
  title,
  draft,
  editing,
  onDraftChange,
  onEditingChange,
  onSave,
}: {
  title: string;
  draft: string;
  editing: boolean;
  onDraftChange: (value: string) => void;
  onEditingChange: (value: boolean) => void;
  onSave: () => void;
}) {
  const inputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    if (editing) {
      inputRef.current?.focus();
      inputRef.current?.select();
    }
  }, [editing]);

  if (editing) {
    return (
      <input
        ref={inputRef}
        value={draft}
        onChange={(event) => onDraftChange(event.target.value)}
        onBlur={onSave}
        onKeyDown={(event) => {
          if (event.key === "Enter") {
            event.preventDefault();
            onSave();
          } else if (event.key === "Escape") {
            event.preventDefault();
            onDraftChange(title);
            onEditingChange(false);
          }
        }}
        className="min-w-0 max-w-[34vw] truncate rounded-sm bg-muted px-1.5 py-0.5 text-sm font-medium outline-none ring-1 ring-ring sm:max-w-[42vw] lg:max-w-[30vw]"
        aria-label="Conversation title"
      />
    );
  }

  return (
    <button
      type="button"
      onClick={() => onEditingChange(true)}
      title="Click to rename"
      className="min-w-0 max-w-[34vw] truncate rounded-sm px-1.5 py-0.5 text-left text-sm font-medium text-muted-foreground hover:bg-muted hover:text-foreground sm:max-w-[42vw] lg:max-w-[30vw]"
    >
      {title || "Your teacher"}
    </button>
  );
}

function ResizeHandle({ onDrag }: { onDrag: (clientX: number) => void }) {
  return (
    <div
      role="separator"
      aria-label="Resize course and chat"
      aria-orientation="vertical"
      className="hidden w-2 cursor-col-resize items-center justify-center bg-border/40 text-muted-foreground transition-colors hover:bg-primary/25 md:flex"
      onPointerDown={(event) => {
        event.currentTarget.setPointerCapture(event.pointerId);
        onDrag(event.clientX);
      }}
      onPointerMove={(event) => {
        if (event.buttons !== 1) return;
        onDrag(event.clientX);
      }}
    >
      <GripVertical className="h-4 w-4" />
    </div>
  );
}

function SourcesPanel({
  files,
  selectedFiles,
  busy,
  className,
  onUploadFiles,
  onDeleteFile,
  onRetryFile,
  onToggleFile,
  onClose,
}: {
  files: SourceFile[];
  selectedFiles: string[];
  busy: boolean;
  className?: string;
  onUploadFiles: (files: FileList | File[]) => void;
  onDeleteFile: (fileId: string) => void;
  onRetryFile: (fileId: string) => void;
  onToggleFile: (fileId: string, checked: boolean) => void;
  onClose?: () => void;
}) {
  return (
    <aside
      className={cn("app-pane flex h-full min-h-0 flex-col overflow-hidden border-r border-border", className)}
      aria-label="Sources"
    >
      <header className="app-chrome flex h-11 items-center justify-between gap-2 border-b border-border px-4">
        <div className="flex min-w-0 items-center gap-2">
          <FolderOpen className="h-4 w-4 text-primary" />
          <h2 className="truncate text-sm font-semibold">Sources</h2>
        </div>
        {onClose && (
          <Button type="button" variant="ghost" size="icon" className="h-8 w-8 lg:hidden" onClick={onClose} aria-label="Close sources" title="Close">
            <X className="h-4 w-4" />
          </Button>
        )}
      </header>

      <div className="app-chrome px-4 py-3">
        <FileUploader busy={busy} onUploadFiles={onUploadFiles} />
      </div>

      <div className="flex-1 overflow-y-auto px-2 pb-4">
        <FileList
          files={files}
          selectedFiles={selectedFiles}
          onDeleteFile={onDeleteFile}
          onRetryFile={onRetryFile}
          onToggleFile={onToggleFile}
        />
      </div>
    </aside>
  );
}

function FileUploader({
  busy,
  onUploadFiles,
}: {
  busy: boolean;
  onUploadFiles: (files: FileList | File[]) => void;
}) {
  const [dragActive, setDragActive] = useState(false);

  function handleDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    setDragActive(false);
    if (!busy && event.dataTransfer.files.length > 0) {
      onUploadFiles(event.dataTransfer.files);
    }
  }

  return (
    <label
      className={cn(
        "flex cursor-pointer flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed px-4 py-6 text-center transition-colors",
        "border-border bg-surface hover:border-primary/60 hover:bg-primary/5",
        dragActive && "border-primary bg-primary/10",
        busy && "cursor-not-allowed opacity-60",
      )}
      onDragOver={(event) => {
        event.preventDefault();
        setDragActive(true);
      }}
      onDragLeave={() => setDragActive(false)}
      onDrop={handleDrop}
    >
      <input type="file" multiple hidden disabled={busy} onChange={(event) => event.target.files && onUploadFiles(event.target.files)} />
      <UploadCloud className="h-6 w-6 text-muted-foreground" />
      <div className="text-sm font-medium">{dragActive ? "Drop to upload" : "Drag files here or click to browse"}</div>
      <div className="text-[11px] text-muted-foreground">PDF, DOCX, PPTX, TXT, MD, HTML - up to 50 MB</div>
    </label>
  );
}

function FileList({
  files,
  selectedFiles,
  onDeleteFile,
  onRetryFile,
  onToggleFile,
}: {
  files: SourceFile[];
  selectedFiles: string[];
  onDeleteFile: (fileId: string) => void;
  onRetryFile: (fileId: string) => void;
  onToggleFile: (fileId: string, checked: boolean) => void;
}) {
  const [fileToDelete, setFileToDelete] = useState<SourceFile | null>(null);
  const readyFiles = files.filter((file) => file.status === "ready");
  const selectedSet = new Set(selectedFiles);
  const activeCount = readyFiles.filter((file) => selectedSet.has(file.id)).length;
  const showFileCheckboxes = readyFiles.length > 0;

  if (files.length === 0) {
    return <div className="text-xs text-muted-foreground">No files yet. Upload a document to start teaching.</div>;
  }

  return (
    <div>
      {showFileCheckboxes && (
        <div className="mb-2 px-1 text-[11px] text-muted-foreground">
          Using {activeCount}/{readyFiles.length} ready files
        </div>
      )}
      <ul className="flex flex-col gap-1.5">
        {files.map((file) => {
          const selectable = file.status === "ready";
          const checked = selectedSet.has(file.id);
          const locked = selectable && checked && activeCount <= 1;
          return (
            <li
              key={file.id}
              className={cn(
                "group flex items-center gap-2 rounded-md border border-border bg-surface px-2.5 py-2",
                selectable && checked && "border-primary/35 bg-primary/5",
              )}
            >
              {selectable ? (
                <input
                  type="checkbox"
                  checked={checked}
                  disabled={locked}
                  onChange={(event) => onToggleFile(file.id, event.target.checked)}
                  className="h-4 w-4 shrink-0 rounded border-border accent-primary"
                  aria-label={`Use ${file.filename}`}
                  title={locked ? "At least one source file must stay active" : `Use ${file.filename}`}
                />
              ) : (
                <FileText className="h-4 w-4 shrink-0 text-muted-foreground" />
              )}
              <div className="min-w-0 flex-1">
                <div className="truncate text-sm font-medium" title={file.filename}>
                  {file.filename}
                </div>
                <div className="flex items-center gap-2 text-[11px] text-muted-foreground">
                  <span>{formatRelativeTime(file.created_at)}</span>
                  {file.status === "ready" && Number(file.chunk_count ?? 0) > 0 && (
                    <span>- {file.chunk_count} chunks</span>
                  )}
                  {file.status !== "ready" && file.parser_used && <span>- {file.parser_used}</span>}
                </div>
              </div>
              <FileStatusBadge status={file.status} error={file.error} />
              {file.status === "failed" && (
                <Button
                  variant="ghost"
                  size="icon"
                  className="transition-opacity"
                  aria-label={`Retry ${file.filename}`}
                  title="Retry from the beginning"
                  onClick={() => onRetryFile(file.id)}
                >
                  <RotateCcw className="h-4 w-4" />
                </Button>
              )}
              <Button
                variant="ghost"
                size="icon"
                className="opacity-0 transition-opacity group-hover:opacity-100 focus-visible:opacity-100"
                aria-label={`Delete ${file.filename}`}
                title="Delete file"
                onClick={() => setFileToDelete(file)}
              >
                <Trash2 className="h-4 w-4" />
              </Button>
            </li>
          );
        })}
      </ul>
      {fileToDelete && (
        <DeleteFileDialog
          file={fileToDelete}
          onCancel={() => setFileToDelete(null)}
          onConfirm={() => {
            onDeleteFile(fileToDelete.id);
            setFileToDelete(null);
          }}
        />
      )}
    </div>
  );
}

function DeleteFileDialog({
  file,
  onCancel,
  onConfirm,
}: {
  file: SourceFile;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-background/60 px-4 backdrop-blur-sm" role="presentation" onClick={onCancel}>
      <section
        className="w-full max-w-lg overflow-hidden rounded-lg border border-border bg-surface shadow-2xl"
        role="dialog"
        aria-modal="true"
        aria-label="Delete file"
        onClick={(event) => event.stopPropagation()}
      >
        <header className="flex items-start justify-between gap-4 border-b border-border px-5 py-4">
          <div>
            <h2 className="text-base font-semibold">Delete file?</h2>
            <p className="mt-1 text-sm leading-5 text-muted-foreground">
              "{file.filename}" will be removed from this conversation. This can't be undone.
            </p>
          </div>
          <Button variant="ghost" size="icon" title="Close" aria-label="Close delete dialog" onClick={onCancel}>
            <X className="h-4 w-4" />
          </Button>
        </header>
        <div className="flex justify-end gap-2 px-5 py-4">
          <Button variant="secondary" onClick={onCancel}>
            Cancel
          </Button>
          <Button variant="danger" onClick={onConfirm}>
            <Trash2 className="h-4 w-4" />
            Delete
          </Button>
        </div>
      </section>
    </div>
  );
}

function QuizSetupDialog({
  value,
  busy,
  onChange,
  onCancel,
  onConfirm,
}: {
  value: QuizForm;
  busy: boolean;
  onChange: (value: QuizForm) => void;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  const count = Math.max(1, Math.min(20, Math.trunc(Number(value.question_count) || DEFAULT_QUIZ_FORM.question_count)));

  return createPortal(
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-background/60 px-4 backdrop-blur-sm" role="presentation" onClick={busy ? undefined : onCancel}>
      <section
        className="w-full max-w-md overflow-hidden rounded-lg border border-border bg-surface shadow-2xl"
        role="dialog"
        aria-modal="true"
        aria-label="Quiz setup"
        onClick={(event) => event.stopPropagation()}
      >
        <header className="flex items-start justify-between gap-4 border-b border-border px-5 py-4">
          <div className="flex min-w-0 items-center gap-3">
            <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-md bg-primary/15 text-primary">
              <FileText className="h-4 w-4" />
            </div>
            <div className="min-w-0">
              <h2 className="truncate text-base font-semibold">Quiz</h2>
              <p className="mt-1 text-sm leading-5 text-muted-foreground">Choose the question format.</p>
            </div>
          </div>
          <Button variant="ghost" size="icon" title="Close" aria-label="Close quiz setup" disabled={busy} onClick={onCancel}>
            <X className="h-4 w-4" />
          </Button>
        </header>

        <div className="flex flex-col gap-4 px-5 py-4">
          <Field label="Number of questions">
            <Input
              type="number"
              min={1}
              max={20}
              value={value.question_count}
              onChange={(event) =>
                onChange({
                  ...value,
                  question_count: Math.max(1, Math.min(20, Math.trunc(Number(event.target.value) || 1))),
                })
              }
              disabled={busy}
            />
          </Field>

          <Field label="Question type">
            <div className="grid grid-cols-2 gap-1 rounded-md border border-border bg-background p-1" role="radiogroup" aria-label="Question type">
              <button
                type="button"
                role="radio"
                aria-checked={value.question_type === "mcq"}
                disabled={busy}
                onClick={() => onChange({ ...value, question_type: "mcq" })}
                className={cn(
                  "flex h-10 items-center justify-center rounded-sm px-3 text-sm font-medium transition-colors",
                  value.question_type === "mcq" ? "bg-primary text-primary-foreground shadow-sm" : "text-muted-foreground hover:bg-muted hover:text-foreground",
                )}
              >
                Multiple choice
              </button>
              <button
                type="button"
                role="radio"
                aria-checked={value.question_type === "true_false"}
                disabled={busy}
                onClick={() => onChange({ ...value, question_type: "true_false" })}
                className={cn(
                  "flex h-10 items-center justify-center rounded-sm px-3 text-sm font-medium transition-colors",
                  value.question_type === "true_false" ? "bg-primary text-primary-foreground shadow-sm" : "text-muted-foreground hover:bg-muted hover:text-foreground",
                )}
              >
                True / false
              </button>
            </div>
          </Field>
        </div>

        <footer className="flex justify-end gap-2 border-t border-border px-5 py-4">
          <Button variant="secondary" onClick={onCancel} disabled={busy}>
            Cancel
          </Button>
          <Button variant="primary" onClick={onConfirm} disabled={busy || count < 1}>
            {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Check className="h-4 w-4" />}
            Generate
          </Button>
        </footer>
      </section>
    </div>,
    document.body,
  );
}

function PodcastSetupDialog({
  value,
  busy,
  onChange,
  onCancel,
  onConfirm,
}: {
  value: PodcastForm;
  busy: boolean;
  onChange: (value: PodcastForm) => void;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  const minutes = Math.max(3, Math.min(15, Math.trunc(Number(value.duration_minutes) || 6)));

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !busy) onCancel();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [busy, onCancel]);

  return createPortal(
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-background/60 px-4 backdrop-blur-sm"
      role="presentation"
      onClick={busy ? undefined : onCancel}
    >
      <section
        className="w-full max-w-md overflow-hidden rounded-lg border border-border bg-surface shadow-2xl"
        role="dialog"
        aria-modal="true"
        aria-label="Podcast setup"
        onClick={(event) => event.stopPropagation()}
      >
        <header className="flex items-start justify-between gap-4 border-b border-border px-5 py-4">
          <div className="flex min-w-0 items-center gap-3">
            <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-md bg-primary/15 text-primary">
              <Mic2 className="h-4 w-4" />
            </div>
            <div className="min-w-0">
              <h2 className="truncate text-base font-semibold">Podcast</h2>
              <p className="mt-1 text-sm leading-5 text-muted-foreground">Create an English or French conversation between two hosts.</p>
            </div>
          </div>
          <Button variant="ghost" size="icon" title="Close" aria-label="Close podcast setup" disabled={busy} onClick={onCancel}>
            <X className="h-4 w-4" />
          </Button>
        </header>

        <div className="flex flex-col gap-4 px-5 py-4">
          <Field label="Topic (optional)">
            <Input
              value={value.topic}
              maxLength={200}
              placeholder="Leave blank to cover the selected material"
              disabled={busy}
              onChange={(event) => onChange({ ...value, topic: event.target.value.slice(0, 200) })}
            />
          </Field>
          <p className="-mt-2 text-xs leading-5 text-muted-foreground">
            {value.topic.length}/200 characters. The podcast stays grounded in the checked course files.
          </p>

          <Field label="Length in minutes">
            <Input
              type="number"
              min={3}
              max={15}
              value={value.duration_minutes}
              disabled={busy}
              onChange={(event) =>
                onChange({
                  ...value,
                  duration_minutes: Math.max(3, Math.min(15, Math.trunc(Number(event.target.value) || 3))),
                })
              }
            />
          </Field>
          <p className="-mt-2 text-xs leading-5 text-muted-foreground">Choose between 3 and 15 minutes. The default is 6.</p>
        </div>

        <footer className="flex justify-end gap-2 border-t border-border px-5 py-4">
          <Button variant="secondary" onClick={onCancel} disabled={busy}>
            Cancel
          </Button>
          <Button variant="primary" onClick={onConfirm} disabled={busy || minutes < 3 || minutes > 15}>
            {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Mic2 className="h-4 w-4" />}
            Generate podcast
          </Button>
        </footer>
      </section>
    </div>,
    document.body,
  );
}

function CoursePanel({
  conversationId,
  files,
  course,
  onCourseChange,
}: {
  conversationId: string;
  files: SourceFile[];
  course: CourseBuilderRead | null;
  onCourseChange: (course: CourseBuilderRead) => void;
}) {
  const [focusMode, setFocusMode] = useState(false);
  return (
    <section
      className={cn(
        "flex h-full min-h-0 min-w-0 flex-col overflow-hidden bg-background",
        focusMode && "fixed inset-0 z-50 h-dvh w-screen",
      )}
      aria-label="Generated course"
    >
      <header className="app-chrome flex h-11 items-center gap-2 border-b border-border px-4">
        <BookOpen className="h-4 w-4 text-primary" />
        <h2 className="min-w-0 flex-1 truncate text-sm font-semibold">Generated course</h2>
        <Button
          variant="ghost"
          size="sm"
          className="h-8 px-2 text-xs"
          onClick={() => setFocusMode((value) => !value)}
          title={focusMode ? "Return to workspace" : "Open course focus mode"}
        >
          {focusMode ? <X className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />}
          {focusMode ? "Exit focus" : "Focus"}
        </Button>
      </header>
      <CourseBuilderLikePanel
        conversationId={conversationId}
        files={files}
        course={course}
        onCourseChange={onCourseChange}
        focusMode={focusMode}
      />
    </section>
  );
}

function CourseBuilderLikePanel({
  conversationId,
  files,
  course,
  onCourseChange,
  focusMode,
}: {
  conversationId: string;
  files: SourceFile[];
  course: CourseBuilderRead | null;
  onCourseChange: (course: CourseBuilderRead) => void;
  focusMode: boolean;
}) {
  const readyFiles = files.filter((file) => file.status === "ready");
  const pending = files.filter((file) => file.status !== "ready" && file.status !== "error" && file.status !== "failed").length;
  const [selectedItemId, setSelectedItemId] = useState<string | null>(null);
  const [courseAction, setCourseAction] = useState<"build" | "stop" | null>(null);
  const [rebuildError, setRebuildError] = useState("");

  async function rebuildWithImprovedQuality() {
    setCourseAction("build");
    setRebuildError("");
    try {
      const improvedQuality = course?.status === "ready";
      const updated = await api.rebuildCoursebuilder(conversationId, improvedQuality);
      setSelectedItemId(null);
      onCourseChange(updated);
    } catch (error) {
      setRebuildError(error instanceof Error ? error.message : "Could not rebuild the course.");
    } finally {
      setCourseAction(null);
    }
  }

  async function stopCourseBuild() {
    setCourseAction("stop");
    setRebuildError("");
    try {
      const updated = await api.stopCoursebuilder(conversationId);
      onCourseChange(updated);
    } catch (error) {
      setRebuildError(error instanceof Error ? error.message : "Could not stop course building.");
    } finally {
      setCourseAction(null);
    }
  }

  if (files.length === 0) {
    return (
      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4 text-xs leading-5 text-muted-foreground">
        Upload course files first. After processing, TeacherLM will build a structured text course here.
      </div>
    );
  }

  if (pending > 0) {
    return (
      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4">
        <div className="flex flex-col gap-3">
        <StateCard
          icon={<BookOpen className="h-4 w-4 text-primary" />}
          title="Course will be generated after processing"
          body={`${pending} of ${files.length} files are not ready yet.`}
        />
        <ProgressBar percent={Math.round((readyFiles.length / files.length) * 100)} />
        </div>
      </div>
    );
  }

  if (readyFiles.length === 0) {
    return (
      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4">
        <div className="flex flex-col gap-3">
        <StateCard
          icon={<AlertCircle className="h-4 w-4 text-danger" />}
          title="Course files need attention"
          body="No ready source files are available for this conversation yet."
        />
        </div>
      </div>
    );
  }

  if (!course || (!course.chapters.length && !["ready", "stopped"].includes(course.status))) {
    const buildStage = String(course?.metadata?.stage || "planning").replace(/_/g, " ");
    return (
      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4">
        <div className="flex flex-col gap-3">
        <StateCard
          icon={<Loader2 className="h-4 w-4 animate-spin text-primary" />}
          title="Building course"
          body={`TeacherLM is ${buildStage}: organizing the ready sources into a cumulative course.`}
        />
        {course?.status === "building" && (
          <Button variant="secondary" onClick={stopCourseBuild} disabled={courseAction !== null}>
            {courseAction === "stop" ? <Loader2 className="h-4 w-4 animate-spin" /> : <Square className="h-3.5 w-3.5 fill-current" />}
            Stop building
          </Button>
        )}
        {rebuildError && <p role="alert" className="text-xs text-[hsl(var(--danger))]">{rebuildError}</p>}
        </div>
      </div>
    );
  }

  if (course.status === "stopped" && !course.chapters.length) {
    return (
      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4">
        <div className="flex flex-col gap-3">
          <StateCard
            icon={<Square className="h-4 w-4 text-[hsl(var(--warning))]" />}
            title="Course building stopped"
            body="No chapters were completed yet. Build again whenever you are ready."
          />
          <Button onClick={rebuildWithImprovedQuality} disabled={courseAction !== null}>
            {courseAction === "build" ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
            Build course
          </Button>
          {rebuildError && <p role="alert" className="text-xs text-[hsl(var(--danger))]">{rebuildError}</p>}
        </div>
      </div>
    );
  }

  const allItems = course.chapters.flatMap((chapter) => [
    ...chapter.lessons.map((lesson) => lesson.id),
    ...(chapter.quiz ? [chapter.quiz.id] : []),
  ]).concat(course.final_quiz ? [course.final_quiz.id] : []);
  const defaultItem = course.chapters
    .flatMap((chapter) => chapter.lessons)
    .find((lesson) => !lesson.is_locked && !lesson.is_completed)?.id
    ?? course.chapters.flatMap((chapter) => chapter.lessons).find((lesson) => !lesson.is_locked)?.id
    ?? course.chapters[0]?.lessons[0]?.id
    ?? null;
  const activeId = selectedItemId && allItems.includes(selectedItemId) ? selectedItemId : defaultItem;
  const selectedLesson = course.chapters
    .flatMap((chapter) => chapter.lessons.map((lesson) => ({ chapter, lesson })))
    .find((item) => item.lesson.id === activeId);
  const selectedQuiz = [
    ...course.chapters.flatMap((chapter) => chapter.quiz ? [{ chapter, quiz: chapter.quiz }] : []),
    ...(course.final_quiz ? [{ chapter: null, quiz: course.final_quiz }] : []),
  ].find((item) => item.quiz.id === activeId);

  const lessonTotal = course.chapters.reduce((total, chapter) => total + chapter.lessons.length, 0);
  const completedLessons = course.progress?.completed_lesson_count ?? 0;
  const passedChapters = course.progress?.passed_chapter_count ?? 0;
  const progressUnits = Math.max(1, lessonTotal + course.chapters.length + 1);
  const completedUnits = completedLessons + passedChapters + (course.progress?.course_completed ? 1 : 0);
  const buildReady = Number(course.metadata?.ready_chapter_count ?? course.chapters.length);
  const buildTotal = Number(course.metadata?.total_chapter_count ?? course.chapters.length);
  const qualityMode = String(course.metadata?.quality_mode || "fallback");
  const buildProfile = String(course.metadata?.build_profile || "fast");
  const qualityUpgradeAvailable = !course.metadata?.quality_pipeline_version;
  const isBuilding = course.status === "building";
  const isStopped = course.status === "stopped";

  return (
    <div className={cn(
      "grid min-h-0 flex-1 grid-cols-1 overflow-hidden",
      focusMode && "sm:grid-cols-[minmax(230px,25%)_minmax(0,1fr)]",
    )}>
      <aside className={cn(
        "app-chrome min-h-0 overflow-y-auto border-b border-border bg-surface/60 p-3",
        focusMode ? "sm:border-b-0 sm:border-r" : "max-h-[42vh]",
      )}>
        <div className="mb-3">
          <div className="flex items-start justify-between gap-2">
            <h3 className="text-sm font-semibold leading-5">{course.title || "Generated course"}</h3>
            <Badge variant={course.status === "ready" ? "success" : isStopped ? "muted" : "primary"}>
              {course.status === "ready" ? "Ready" : isStopped ? "Stopped" : `${buildReady}/${buildTotal}`}
            </Badge>
          </div>
          <Button
            variant="secondary"
            size="sm"
            className={cn("mt-2 w-full justify-center", isBuilding && "text-[hsl(var(--danger))]")}
            onClick={isBuilding ? stopCourseBuild : rebuildWithImprovedQuality}
            disabled={courseAction !== null}
            title={
              isBuilding
                ? "Stop course building"
                : isStopped
                  ? "Resume building from completed compatible chapters"
                  : "Replan from the uploaded source structure and regenerate each teaching block with validation"
            }
          >
            {courseAction ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : isBuilding ? (
              <Square className="h-3.5 w-3.5 fill-current" />
            ) : isStopped ? (
              <Play className="h-3.5 w-3.5" />
            ) : (
              <RotateCcw className="h-3.5 w-3.5" />
            )}
            {isBuilding ? "Stop building" : isStopped ? "Build course" : "Rebuild with improved quality"}
          </Button>
          {rebuildError && <p role="alert" className="mt-1 text-[10px] text-[hsl(var(--danger))]">{rebuildError}</p>}
          {qualityUpgradeAvailable && course.status === "ready" && (
            <p className="mt-1 text-[10px] leading-4 text-muted-foreground">
              This existing course can be manually upgraded without deleting matching progress.
            </p>
          )}
          <p className="mt-1 line-clamp-3 text-[11px] leading-4 text-muted-foreground">{course.description}</p>
          <div className="mt-3">
            <div className="mb-1 flex justify-between text-[10px] text-muted-foreground">
              <span>Learning progress</span>
              <span>{Math.round((completedUnits / progressUnits) * 100)}%</span>
            </div>
            <ProgressBar percent={(completedUnits / progressUnits) * 100} />
          </div>
          {course.status === "building" && (
            <div className="mt-2 rounded-md border border-primary/20 bg-primary/10 p-2 text-[10px] leading-4 text-primary">
              <Loader2 className="mr-1 inline h-3 w-3 animate-spin" />
              Publishing chapter {Math.min(buildReady + 1, buildTotal)} of {buildTotal}
            </div>
          )}
          {isStopped && (
            <div className="mt-2 rounded-md border border-[hsl(var(--warning)/0.35)] bg-[hsl(var(--warning)/0.1)] p-2 text-[10px] leading-4 text-foreground">
              Building is stopped. Completed chapters are kept and will be reused when you continue.
            </div>
          )}
          {qualityMode !== "llm" && (
            <div className="mt-2 rounded-md border border-[hsl(var(--warning)/0.35)] bg-[hsl(var(--warning)/0.1)] p-2 text-[10px] leading-4 text-foreground">
              {buildProfile === "fast"
                ? "Fast grounded build is ready. Use Rebuild with improved quality only when you want deeper synthesis."
                : qualityMode === "mixed"
                  ? "Some sections use source-extracted fallback content."
                  : "Source-extracted course; configure a chat model for deeper synthesis."}
            </div>
          )}
        </div>
        <ol className="flex flex-col gap-2">
        {course.chapters.map((chapter, index) => (
          <li key={chapter.id}>
            <div className={cn("rounded-md border bg-background", chapter.is_locked ? "border-border/60 opacity-65" : "border-border")}>
              <div className="flex items-start gap-2 px-2.5 py-2">
                {chapter.is_complete ? <CheckCircle2 className="mt-0.5 h-3.5 w-3.5 text-[hsl(var(--success))]" /> : chapter.is_locked ? <KeyRound className="mt-0.5 h-3.5 w-3.5 text-muted-foreground" /> : <BookOpen className="mt-0.5 h-3.5 w-3.5 text-primary" />}
                <p className="min-w-0 flex-1 text-[11px] font-semibold leading-4">{index + 1}. {chapter.title}</p>
              </div>
              <div className="border-t border-border/70 p-1">
                {chapter.lessons.map((lesson, lessonIndex) => (
                  <button
                    key={lesson.id}
                    type="button"
                    disabled={lesson.is_locked}
                    onClick={() => setSelectedItemId(lesson.id)}
                    className={cn(
                      "flex w-full items-center gap-2 rounded px-2 py-1.5 text-left text-[11px] transition-colors",
                      activeId === lesson.id ? "bg-primary/15 text-primary" : "text-muted-foreground hover:bg-muted hover:text-foreground",
                      lesson.is_locked && "cursor-not-allowed opacity-60",
                    )}
                  >
                    {lesson.is_completed ? <Check className="h-3 w-3 text-[hsl(var(--success))]" /> : lesson.is_locked ? <KeyRound className="h-3 w-3" /> : <span className="w-3 text-center text-[9px]">{lessonIndex + 1}</span>}
                    <span className="min-w-0 flex-1 truncate">{lesson.title}</span>
                  </button>
                ))}
                {chapter.quiz && (
                  <button
                    type="button"
                    disabled={chapter.quiz.is_locked}
                    onClick={() => setSelectedItemId(chapter.quiz!.id)}
                    className={cn(
                      "flex w-full items-center gap-2 rounded px-2 py-1.5 text-left text-[11px] transition-colors",
                      activeId === chapter.quiz.id ? "bg-primary/15 text-primary" : "text-muted-foreground hover:bg-muted hover:text-foreground",
                      chapter.quiz.is_locked && "cursor-not-allowed opacity-60",
                    )}
                  >
                    {chapter.quiz.is_passed ? <CheckCircle2 className="h-3 w-3 text-[hsl(var(--success))]" /> : chapter.quiz.is_locked ? <KeyRound className="h-3 w-3" /> : <FileText className="h-3 w-3" />}
                    <span>Chapter quiz</span>
                  </button>
                )}
              </div>
            </div>
          </li>
        ))}
        {course.final_quiz && (
          <li>
            <button
              type="button"
              disabled={course.final_quiz.is_locked}
              onClick={() => setSelectedItemId(course.final_quiz!.id)}
              className={cn(
                "flex w-full items-center gap-2 rounded-md border border-border bg-background px-2.5 py-2 text-left text-[11px] font-semibold",
                activeId === course.final_quiz.id && "border-primary/40 bg-primary/10 text-primary",
                course.final_quiz.is_locked && "cursor-not-allowed opacity-60",
              )}
            >
              {course.final_quiz.is_passed ? <CheckCircle2 className="h-3.5 w-3.5 text-[hsl(var(--success))]" /> : course.final_quiz.is_locked ? <KeyRound className="h-3.5 w-3.5" /> : <GraduationCap className="h-3.5 w-3.5" />}
              Final course quiz
            </button>
          </li>
        )}
        </ol>
      </aside>

      <main className="content-selectable min-h-0 overflow-y-auto bg-background">
        {selectedLesson ? (
          <CourseLessonView
            conversationId={conversationId}
            chapter={selectedLesson.chapter}
            lesson={selectedLesson.lesson}
            course={course}
            onCourseChange={onCourseChange}
            onNavigate={setSelectedItemId}
          />
        ) : selectedQuiz ? (
          <CourseQuizView
            conversationId={conversationId}
            chapter={selectedQuiz.chapter}
            quiz={selectedQuiz.quiz}
            onCourseChange={onCourseChange}
            onNavigateLesson={setSelectedItemId}
          />
        ) : (
          <div className="flex min-h-full items-center justify-center p-8 text-center text-sm text-muted-foreground">
            {isBuilding
              ? "The first chapter is being prepared."
              : isStopped
                ? "Building is stopped. Select Build course when you want to continue."
                : "Select an unlocked subchapter to begin."}
          </div>
        )}
      </main>
    </div>
  );
}

function CourseLessonView({
  conversationId,
  chapter,
  lesson,
  course,
  onCourseChange,
  onNavigate,
}: {
  conversationId: string;
  chapter: CourseChapter;
  lesson: CourseLesson;
  course: CourseBuilderRead;
  onCourseChange: (course: CourseBuilderRead) => void;
  onNavigate: (id: string) => void;
}) {
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const lessonIndex = chapter.lessons.findIndex((item) => item.id === lesson.id);
  const visibleBlocks = useMemo(() => dedupeCourseBlocks(lesson.blocks), [lesson.blocks]);
  const showSummary = shouldShowLessonSummary(lesson.summary, visibleBlocks);
  const remainingLessonCitations = useMemo(() => {
    const visibleCitationIds = new Set(
      visibleBlocks.flatMap((block) => block.citations.map((citation) => citation.chunk_id)),
    );
    return lesson.citations.filter((citation) => !visibleCitationIds.has(citation.chunk_id));
  }, [lesson.citations, visibleBlocks]);

  async function completeLesson() {
    setSaving(true);
    setError(null);
    try {
      const updated = await api.completeCourseLesson(conversationId, lesson.id);
      onCourseChange(updated);
      const updatedChapter = updated.chapters.find((item) => item.id === chapter.id);
      const nextLesson = updatedChapter?.lessons[lessonIndex + 1];
      if (nextLesson && !nextLesson.is_locked) onNavigate(nextLesson.id);
      else if (updatedChapter?.quiz && !updatedChapter.quiz.is_locked) onNavigate(updatedChapter.quiz.id);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not complete this subchapter.");
    } finally {
      setSaving(false);
    }
  }

  return (
    <article className="mx-auto flex w-full max-w-4xl flex-col gap-5 px-4 py-5 sm:px-7 sm:py-7">
      <header className="border-b border-border pb-4">
        <div className="flex flex-wrap items-center gap-2 text-[11px] text-muted-foreground">
          <Badge variant="primary">Chapter {chapter.order_index !== undefined ? chapter.order_index + 1 : course.chapters.indexOf(chapter) + 1}</Badge>
          <span>Subchapter {lessonIndex + 1} of {chapter.lessons.length}</span>
          {lesson.lesson_stage && (
            <Badge variant="muted">
              {lesson.lesson_stage === "introduction" ? "Introduction" : lesson.lesson_stage === "conclusion" ? "Conclusion" : "Core content"}
            </Badge>
          )}
          {lesson.is_completed && <Badge variant="success"><Check className="h-3 w-3" /> Completed</Badge>}
        </div>
        <h1 className="mt-3 text-xl font-semibold tracking-tight sm:text-2xl">{lesson.title}</h1>
        {showSummary && (
          <div className="course-markdown mt-2 text-sm text-muted-foreground">
            <AssistantMarkdown content={lesson.summary} />
          </div>
        )}
        {lesson.learning_objectives.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-1.5">
            {lesson.learning_objectives.map((objective) => <Badge key={objective} variant="muted">{objective}</Badge>)}
          </div>
        )}
      </header>

      {visibleBlocks.map((block) => <CourseBlockRenderer key={block.id} block={block} />)}

      <CourseCitations citations={remainingLessonCitations} />

      <footer className="flex flex-col items-start gap-2 border-t border-border pt-4 sm:flex-row sm:items-center sm:justify-between">
        <p className="text-xs leading-5 text-muted-foreground">
          Complete this subchapter to unlock the next foundation.
        </p>
        <Button onClick={completeLesson} disabled={saving || lesson.is_completed}>
          {saving ? <Loader2 className="h-4 w-4 animate-spin" /> : <CheckCircle2 className="h-4 w-4" />}
          {lesson.is_completed ? "Completed" : "Mark complete"}
        </Button>
      </footer>
      {error && <p role="alert" className="text-xs text-[hsl(var(--danger))]">{error}</p>}
    </article>
  );
}

function CourseBlockRenderer({ block }: { block: CourseLessonBlock }) {
  const special = ["equation", "matrix", "chemical_equation", "table", "timeline"].includes(block.block_type);
  const insufficient = block.validation_status === "insufficient_source_material";
  if (block.block_type === "table" && block.data_json?.headers && block.data_json?.rows) {
    return (
      <section className="rounded-lg border border-border bg-surface p-4">
        <h2 className="mb-3 text-sm font-semibold">{block.title}</h2>
        <div className="overflow-x-auto rounded-md border border-border">
          <table className="min-w-full border-collapse text-left text-xs">
            <caption className="sr-only">{block.title}</caption>
            <thead className="bg-muted">
              <tr>{block.data_json.headers.map((header, index) => <th key={`${header}-${index}`} scope="col" className="border-b border-r border-border px-3 py-2 font-semibold last:border-r-0">{header}</th>)}</tr>
            </thead>
            <tbody>{block.data_json.rows.map((row, rowIndex) => <tr key={rowIndex} className="even:bg-muted/40">{row.map((cell, cellIndex) => <td key={cellIndex} className="border-b border-r border-border px-3 py-2 align-top last:border-r-0">{cell}</td>)}</tr>)}</tbody>
          </table>
        </div>
        <CourseCitations citations={block.citations} compact />
      </section>
    );
  }
  if (block.block_type === "timeline" && block.data_json?.events) {
    return (
      <section className="rounded-lg border border-border bg-surface p-4">
        <h2 className="mb-4 text-sm font-semibold">{block.title}</h2>
        <ol className="relative ml-2 border-l border-primary/35 pl-5">
          {block.data_json.events.map((event, index) => (
            <li key={`${event.date}-${index}`} className="relative pb-4 last:pb-0">
              <span className="absolute -left-[1.52rem] top-1 h-2.5 w-2.5 rounded-full border-2 border-primary bg-background" />
              <time className="text-xs font-semibold text-primary">{event.date}</time>
              <p className="mt-1 text-sm leading-6 text-surface-foreground">{event.description}</p>
            </li>
          ))}
        </ol>
        <CourseCitations citations={block.citations} compact />
      </section>
    );
  }
  return (
    <section className={cn(
      "course-markdown",
      special && "rounded-lg border border-primary/20 bg-primary/5 p-4",
      insufficient && "rounded-lg border border-[hsl(var(--warning)/0.35)] bg-[hsl(var(--warning)/0.08)] p-4",
    )}>
      {block.title && block.block_type !== "markdown" && <h2 className="mb-2 text-sm font-semibold">{block.title}</h2>}
      {insufficient && <Badge variant="muted">Insufficient source support</Badge>}
      <AssistantMarkdown content={block.content} />
      <CourseCitations citations={block.citations} compact />
    </section>
  );
}

function CourseCitations({ citations, compact = false }: { citations: CourseLessonBlock["citations"]; compact?: boolean }) {
  if (!citations.length) return null;
  return (
    <details className={cn("rounded-md border border-border bg-muted/35", compact ? "mt-3" : "p-1")}>
      <summary className="cursor-pointer select-none px-2 py-1.5 text-[11px] font-medium text-muted-foreground">
        {citations.length} grounded source{citations.length === 1 ? "" : "s"}
      </summary>
      <ul className="space-y-2 border-t border-border p-2">
        {citations.map((citation) => (
          <li key={citation.chunk_id} className="text-[11px] leading-4 text-muted-foreground">
            <strong className="text-foreground">{citation.source}</strong>{citation.section ? ` · ${citation.section}` : ""}
            {citation.snippet && <p className="mt-0.5 line-clamp-3">{cleanCourseCitationSnippet(citation.snippet)}</p>}
          </li>
        ))}
      </ul>
    </details>
  );
}

function CourseQuizView({
  conversationId,
  chapter,
  quiz,
  onCourseChange,
  onNavigateLesson,
}: {
  conversationId: string;
  chapter: CourseChapter | null;
  quiz: CourseQuiz;
  onCourseChange: (course: CourseBuilderRead) => void;
  onNavigateLesson: (id: string) => void;
}) {
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [result, setResult] = useState<CourseQuizSubmissionResult | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setAnswers({});
    setResult(null);
    setError(null);
  }, [quiz.id]);

  async function submit() {
    setSubmitting(true);
    setError(null);
    try {
      const response = await api.submitCourseQuiz(
        conversationId,
        quiz.id,
        Object.entries(answers).map(([question_id, option_id]) => ({ question_id, option_id })),
      );
      setResult(response);
      onCourseChange(response.course);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not submit this quiz.");
    } finally {
      setSubmitting(false);
    }
  }

  const resultByQuestion = new Map(result?.results.map((item) => [item.question_id, item]) ?? []);
  return (
    <article className="mx-auto flex w-full max-w-4xl flex-col gap-5 px-4 py-5 sm:px-7 sm:py-7">
      <header className="border-b border-border pb-4">
        <div className="flex flex-wrap items-center gap-2">
          <Badge variant={quiz.scope === "course" ? "primary" : "muted"}>{quiz.scope === "course" ? "Final assessment" : "Chapter quiz"}</Badge>
          <Badge variant="muted">{quiz.questions.length} questions</Badge>
          <Badge variant="muted">Pass at {Math.round(quiz.pass_score * 100)}%</Badge>
          {quiz.is_passed && <Badge variant="success"><CheckCircle2 className="h-3 w-3" /> Mastered</Badge>}
        </div>
        <h1 className="mt-3 text-xl font-semibold sm:text-2xl">{quiz.title}</h1>
        <p className="mt-2 text-sm leading-6 text-muted-foreground">
          {chapter ? `Show that you are ready to build on ${chapter.title}.` : "Connect ideas across the entire course."}
        </p>
      </header>

      {quiz.questions.map((question, questionIndex) => {
        const questionResult = resultByQuestion.get(question.id);
        return (
          <fieldset key={question.id} className="rounded-lg border border-border bg-surface p-4">
            <legend className="px-1 text-sm font-semibold leading-6">{questionIndex + 1}. {question.prompt}</legend>
            <div className="mt-3 grid gap-2">
              {question.options.map((option) => {
                const selected = answers[question.id] === option.id;
                const correctAfterSubmit = questionResult?.correct_option_id === option.id;
                const wrongSelection = Boolean(questionResult && selected && !questionResult.correct);
                return (
                  <label key={option.id} className={cn(
                    "flex cursor-pointer items-start gap-3 rounded-md border px-3 py-2.5 text-sm leading-5 transition-colors",
                    selected ? "border-primary/50 bg-primary/10" : "border-border hover:bg-muted/60",
                    correctAfterSubmit && "border-[hsl(var(--success)/0.6)] bg-[hsl(var(--success)/0.1)]",
                    wrongSelection && "border-[hsl(var(--danger)/0.6)] bg-[hsl(var(--danger)/0.08)]",
                  )}>
                    <input
                      type="radio"
                      name={`course-${quiz.id}-${question.id}`}
                      value={option.id}
                      checked={selected}
                      disabled={Boolean(result)}
                      onChange={() => setAnswers((current) => ({ ...current, [question.id]: option.id }))}
                      className="mt-1 accent-[hsl(var(--primary))]"
                    />
                    <span>{option.text}</span>
                  </label>
                );
              })}
            </div>
            {questionResult && (
              <div className={cn("mt-3 rounded-md p-3 text-xs leading-5", questionResult.correct ? "bg-[hsl(var(--success)/0.1)]" : "bg-[hsl(var(--warning)/0.12)]")}>
                <strong>{questionResult.correct ? "Correct." : "Review this foundation."}</strong> {questionResult.explanation}
              </div>
            )}
          </fieldset>
        );
      })}

      {result ? (
        <section className={cn("rounded-lg border p-4", result.passed ? "border-[hsl(var(--success)/0.5)] bg-[hsl(var(--success)/0.1)]" : "border-[hsl(var(--warning)/0.5)] bg-[hsl(var(--warning)/0.1)]")}>
          <h2 className="text-base font-semibold">{result.passed ? "Foundation mastered" : "A little review will make this stick"}</h2>
          <p className="mt-1 text-sm">Score: {Math.round(result.score * 100)}% · Attempt {result.attempt_count}</p>
          {!result.passed && result.review_lesson_ids.length > 0 && (
            <div className="mt-3 flex flex-wrap gap-2">
              {result.review_lesson_ids.map((lessonId, index) => (
                <Button key={lessonId} variant="secondary" size="sm" onClick={() => onNavigateLesson(lessonId)}>
                  Review section {index + 1}
                </Button>
              ))}
            </div>
          )}
          {!result.passed && (
            <Button
              className="mt-3"
              size="sm"
              onClick={() => {
                setAnswers({});
                setResult(null);
              }}
            >
              <RotateCcw className="h-3.5 w-3.5" /> Retry quiz
            </Button>
          )}
        </section>
      ) : (
        <div className="flex justify-end">
          <Button onClick={submit} disabled={submitting || Object.keys(answers).length !== quiz.questions.length}>
            {submitting ? <Loader2 className="h-4 w-4 animate-spin" /> : <GraduationCap className="h-4 w-4" />}
            Submit quiz
          </Button>
        </div>
      )}
      {error && <p role="alert" className="text-xs text-[hsl(var(--danger))]">{error}</p>}
    </article>
  );
}

function ChatPanel({
  conversationId,
  messages,
  draft,
  input,
  busy,
  chatStreaming,
  hint,
  outputs,
  disabled,
  disabledReason,
  inputRef,
  onInputChange,
  onSubmitChat,
  onStopChat,
  onRunGenerator,
}: {
  conversationId: string;
  messages: Message[];
  draft: string;
  input: string;
  busy: boolean;
  chatStreaming: boolean;
  hint: string | null;
  outputs: GeneratorManifest[];
  disabled: boolean;
  disabledReason: string;
  inputRef: React.RefObject<HTMLTextAreaElement>;
  onInputChange: (value: string) => void;
  onSubmitChat: (event?: FormEvent) => void;
  onStopChat: () => void;
  onRunGenerator: (outputType: string) => void;
}) {
  return (
    <section className="flex h-full min-h-0 min-w-0 flex-col overflow-hidden bg-background" aria-label="Chat">
      <header className="app-chrome flex h-11 items-center justify-between gap-3 border-b border-border px-4">
        <div className="min-w-0 flex-1">
          {hint && (
            <p className="flex items-center gap-1.5 truncate text-[11px] text-muted-foreground">
              <Sparkles className="h-3 w-3 text-primary" />
              {hint}
            </p>
          )}
        </div>
      </header>

      <MessageList messages={messages} draft={draft} conversationId={conversationId} className="min-h-0 flex-1" />

      <footer className="app-pane flex flex-col gap-2 border-t border-border px-3 py-3 sm:px-4">
        <OutputTypeButtons
          outputs={outputs}
          disabled={disabled || busy}
          disabledReason={disabledReason}
          onSelectChat={() => inputRef.current?.focus()}
          onGenerate={onRunGenerator}
          className="app-chrome"
        />
        <ChatInput
          ref={inputRef}
          value={input}
          busy={busy}
          streaming={chatStreaming}
          disabled={disabled}
          disabledReason={disabledReason}
          onValueChange={onInputChange}
          onSubmit={onSubmitChat}
          onStop={onStopChat}
        />
      </footer>
    </section>
  );
}

function MessageList({
  messages,
  draft,
  conversationId,
  className,
}: {
  messages: Message[];
  draft: string;
  conversationId: string;
  className?: string;
}) {
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const items = useMemo(() => {
    if (!draft) return messages;
    return [...messages, optimisticMessage(conversationId, "assistant", draft, "text")];
  }, [conversationId, draft, messages]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "auto", block: "end" });
  }, [items.length, draft.length]);

  if (items.length === 0) {
    return (
      <div className={cn("app-chrome flex h-full flex-col items-center justify-center gap-3 px-6 text-center", className)}>
        <div className="flex h-10 w-10 items-center justify-center rounded-md bg-primary/15 text-primary">
          <MessageSquare className="h-5 w-5" />
        </div>
        <div className="text-sm font-medium">Start learning</div>
        <p className="max-w-sm text-xs text-muted-foreground">Upload course files, then ask a question.</p>
      </div>
    );
  }

  return (
    <div className={cn("content-selectable flex flex-col gap-4 overflow-y-auto px-3 py-5 sm:px-4", className)}>
      {items.map((message) => (
        <MessageBubble
          key={message.id}
          message={message}
          conversationId={conversationId}
          streaming={message.id.startsWith("optimistic-assistant-")}
        />
      ))}
      <div ref={bottomRef} aria-hidden />
    </div>
  );
}

function MessageBubble({
  message,
  conversationId,
  streaming,
}: {
  message: Message;
  conversationId: string;
  streaming?: boolean;
}) {
  const isUser = message.role === "user";
  const messageBodyRef = useRef<HTMLDivElement | null>(null);
  const hasArtifacts = message.artifacts.length > 0;
  const hasSources = message.sources.length > 0;
  const canCopy = !isUser && message.content.trim().length > 0;
  const showInlineArtifacts = hasArtifacts && !["quiz", "podcast", "mindmap"].includes(message.output_type);

  return (
    <div className={cn("flex gap-3", isUser ? "flex-row-reverse" : "flex-row")}>
      <Avatar role={message.role} />

      <div className={cn("flex min-w-0 max-w-[min(88%,760px)] flex-col gap-2", isUser ? "items-end" : "items-start")}>
        <div
          ref={messageBodyRef}
          className={cn(
            "content-selectable rounded-lg px-4 py-3 text-sm",
            isUser ? "bg-primary text-primary-foreground leading-7" : "border border-border bg-surface text-surface-foreground",
          )}
        >
          {message.content ? (
            <AssistantMarkdown content={message.content} variant={isUser ? "user" : "assistant"} />
          ) : streaming ? (
            <TypingIndicator />
          ) : (
            <span className="text-xs italic text-muted-foreground">(no response)</span>
          )}
          {streaming && message.content && <span className="ml-0.5 inline-block h-3 w-1 animate-pulse align-baseline bg-current" />}
        </div>

        {canCopy && <CopyMessageButton content={message.content} contentRef={messageBodyRef} disabled={Boolean(streaming)} />}

        {showInlineArtifacts && (
          <div className="flex w-full flex-col gap-3">
            {message.artifacts.map((artifact) => (
              <ArtifactLink key={artifact.key ?? artifact.url} artifact={artifact} />
            ))}
          </div>
        )}

        {hasSources && <SourcesDisclosure message={message} />}
      </div>
    </div>
  );
}

function Avatar({ role }: { role: Message["role"] }) {
  const isUser = role === "user";
  return (
    <div
      className={cn(
        "app-chrome flex h-8 w-8 shrink-0 items-center justify-center rounded-md",
        isUser ? "bg-muted text-muted-foreground" : "bg-primary/15 text-primary",
      )}
      aria-hidden
    >
      {isUser ? <User className="h-4 w-4" /> : <GraduationCap className="h-4 w-4" />}
    </div>
  );
}

function CopyMessageButton({
  content,
  contentRef,
  disabled,
}: {
  content: string;
  contentRef: React.RefObject<HTMLElement>;
  disabled?: boolean;
}) {
  const [copied, setCopied] = useState(false);

  const onCopy = async () => {
    if (disabled) return;
    const renderedText = normalizeCopiedText(contentRef.current?.innerText ?? "");
    const text = renderedText || content;
    if (!text.trim()) return;
    const ok = await copyToClipboard(text);
    if (!ok) return;
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1500);
  };

  return (
    <button
      type="button"
      onClick={() => void onCopy()}
      disabled={disabled}
      className={cn(
        "app-chrome inline-flex h-7 items-center gap-1.5 rounded-md border border-border px-2 text-[11px]",
        "text-muted-foreground transition-colors hover:bg-muted hover:text-foreground",
        "disabled:pointer-events-none disabled:opacity-50",
      )}
      aria-label={copied ? "Copied message" : "Copy message"}
      title={disabled ? "Copy is available when generation finishes" : "Copy message"}
    >
      {copied ? <Check className="h-3 w-3" /> : <Copy className="h-3 w-3" />}
      <span>{copied ? "Copied" : "Copy message"}</span>
    </button>
  );
}

function normalizeCopiedText(value: string): string {
  return value
    .replace(/\n{3,}/g, "\n\n")
    .replace(/[ \t]+\n/g, "\n")
    .trim();
}

function TypingIndicator() {
  return (
    <span className="inline-flex items-center gap-1">
      <Dot delay={0} />
      <Dot delay={150} />
      <Dot delay={300} />
    </span>
  );
}

function Dot({ delay }: { delay: number }) {
  return <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-current opacity-70" style={{ animationDelay: `${delay}ms` }} />;
}

function SourcesDisclosure({ message }: { message: Message }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="w-full">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        className="app-chrome inline-flex items-center gap-1.5 text-[11px] text-muted-foreground hover:text-foreground"
      >
        {open ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
        <BookOpen className="h-3 w-3" />
        {message.sources.length} source{message.sources.length === 1 ? "" : "s"}
      </button>

      {open && (
        <ol className="mt-1.5 flex flex-col gap-1.5">
          {message.sources.map((source, index) => (
            <li key={`${source.chunk_id ?? index}-${index}`} className="rounded-md border border-border bg-surface p-2 text-[11px]">
              <div className="mb-1 flex items-center justify-between gap-2">
                <span className="truncate font-medium" title={source.source}>
                  {source.source}
                </span>
                <Badge variant="muted">score {source.score.toFixed(2)}</Badge>
              </div>
              <p className="line-clamp-4 whitespace-pre-wrap text-muted-foreground">{source.text}</p>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}

const ChatInput = React.forwardRef<
  HTMLTextAreaElement,
  {
    value: string;
    busy: boolean;
    streaming: boolean;
    disabled: boolean;
    disabledReason: string;
    onValueChange: (value: string) => void;
    onSubmit: (event?: FormEvent) => void;
    onStop: () => void;
  }
>(function ChatInput({ value, busy, streaming, disabled, disabledReason, onValueChange, onSubmit, onStop }, ref) {
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    const element = textareaRef.current;
    if (!element) return;
    element.style.height = "44px";
    const next = Math.min(element.scrollHeight, 220);
    element.style.height = `${Math.max(44, next)}px`;
  }, [value]);

  return (
    <form
      className={cn(
        "flex items-end gap-2 rounded-lg border border-border bg-surface p-2",
        "focus-within:border-primary/60 focus-within:ring-1 focus-within:ring-primary/40",
        disabled && "opacity-75",
      )}
      onSubmit={(event) => onSubmit(event)}
    >
      <textarea
        ref={(node) => {
          textareaRef.current = node;
          if (typeof ref === "function") ref(node);
          else if (ref) ref.current = node;
        }}
        value={value}
        onChange={(event) => onValueChange(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
            event.preventDefault();
            onSubmit();
          }
        }}
        placeholder={disabled ? disabledReason : "Ask your teacher anything..."}
        rows={1}
        disabled={busy}
        className={cn(
          "flex min-h-[44px] w-full resize-none rounded-md border-0 bg-transparent px-2 py-2 text-sm",
          "placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-0 focus-visible:ring-offset-0",
          "disabled:cursor-not-allowed disabled:opacity-50",
        )}
        aria-label="Chat message"
      />
      <Button
        type={streaming ? "button" : "submit"}
        variant={streaming ? "secondary" : "primary"}
        size="icon"
        onClick={streaming ? onStop : undefined}
        disabled={streaming ? false : busy || disabled || !value.trim()}
        aria-label={streaming ? "Stop response" : "Send message"}
        title={streaming ? "Stop response" : disabled ? disabledReason : "Send (Enter)"}
        className={streaming ? "text-[hsl(var(--danger))]" : undefined}
      >
        {streaming ? (
          <Square className="h-3.5 w-3.5 fill-current" />
        ) : busy ? (
          <Loader2 className="h-4 w-4 animate-spin" />
        ) : (
          <Send className="h-4 w-4" />
        )}
      </Button>
    </form>
  );
});

function OutputTypeButtons({
  outputs,
  disabled,
  disabledReason,
  onSelectChat,
  onGenerate,
  className,
}: {
  outputs: GeneratorManifest[];
  disabled: boolean;
  disabledReason: string;
  onSelectChat: () => void;
  onGenerate: (outputType: string) => void;
  className?: string;
}) {
  const enabled = new Set(["text", ...outputs.map((output) => output.output_type)]);
  const metaByType = new Map(outputs.map((output) => [output.output_type, output]));
  const visibleButtons = OUTPUT_BUTTONS.filter((button) => enabled.has(button.output_type));

  return (
    <div className={cn("flex items-center gap-1.5 overflow-x-auto pb-0.5", className)}>
      {visibleButtons.map(({ output_type, display_name, hint, Icon }) => {
        const manifest = metaByType.get(output_type);
        const handleClick = () => {
          if (disabled) return;
          if (output_type === "text") {
            onSelectChat();
            return;
          }
          onGenerate(output_type);
        };

        return (
          <Button
            key={output_type}
            variant="secondary"
            size="sm"
            onClick={handleClick}
            disabled={disabled}
            aria-label={display_name}
            title={disabled ? disabledReason : hint}
            className={cn("h-8 shrink-0 gap-1.5 px-2.5", output_type === "text" && "border-primary/40 bg-primary/10")}
          >
            {manifest?.transport === "emoji" ? <span className="text-base leading-none">{manifest.transport}</span> : <Icon className="h-4 w-4" />}
            <span className="hidden md:inline">{manifest?.display_name ?? display_name}</span>
          </Button>
        );
      })}
    </div>
  );
}

function GeneratedItemsPanel({
  conversationId,
  messages,
  artifacts,
  className,
  onClose,
}: {
  conversationId: string;
  messages: Message[];
  artifacts: Artifact[];
  className?: string;
  onClose?: () => void;
}) {
  const groups = useMemo(() => buildArtifactGroups(messages, artifacts), [artifacts, messages]);

  return (
    <aside
      className={cn("app-pane flex h-full min-h-0 flex-col overflow-hidden border-l border-border", className)}
      aria-label="Generated items"
    >
      <header className="app-chrome flex h-11 items-center justify-between gap-2 border-b border-border px-4">
        <div className="flex min-w-0 items-center gap-2">
          <Package className="h-4 w-4 text-primary" />
          <h2 className="truncate text-sm font-semibold">Generated items</h2>
        </div>
        {onClose && (
          <Button type="button" variant="ghost" size="icon" className="h-8 w-8 lg:hidden" onClick={onClose} aria-label="Close generated items" title="Close">
            <X className="h-4 w-4" />
          </Button>
        )}
      </header>

      <div className="flex-1 overflow-y-auto px-4 py-4">
        {groups.length === 0 ? (
          <p className="app-chrome text-xs leading-5 text-muted-foreground">Generated items will appear here.</p>
        ) : (
          <div className="flex flex-col gap-3">
            {groups.map((group) => (
              <ArtifactModalButton key={group.id} group={group} />
            ))}
          </div>
        )}
      </div>
    </aside>
  );
}

type ArtifactGroup = {
  id: string;
  outputType: string | null;
  createdAt?: string;
  artifacts: Artifact[];
};

function buildArtifactGroups(messages: Message[], artifacts: Artifact[]): ArtifactGroup[] {
  const groups: ArtifactGroup[] = [];
  const seen = new Set<string>();

  for (const message of messages) {
    if (message.role !== "assistant" || message.artifacts.length === 0) continue;
    groups.push({
      id: message.id,
      outputType: message.output_type || normalizeOutputType(message.artifacts),
      createdAt: message.created_at,
      artifacts: message.artifacts,
    });
    for (const artifact of message.artifacts) {
      seen.add(artifactId(artifact));
    }
  }

  const orphans = artifacts.filter((artifact) => !seen.has(artifactId(artifact)));
  const orphanMindmapArtifacts = orphans.filter((artifact) => ["mindmap", "html"].includes(normalizeArtifactKind(artifact)));
  if (orphanMindmapArtifacts.some((artifact) => normalizeArtifactKind(artifact) === "mindmap")) {
    groups.push({
      id: `orphan-mindmap-${orphanMindmapArtifacts.map(artifactId).join("-")}`,
      outputType: "mindmap",
      artifacts: orphanMindmapArtifacts,
    });
  }

  for (const artifact of orphans) {
    if (orphanMindmapArtifacts.includes(artifact)) continue;
    groups.push({
      id: artifactId(artifact),
      outputType: normalizeArtifactKind(artifact),
      artifacts: [artifact],
    });
  }

  return groups.reverse();
}

function normalizeOutputType(artifacts: Artifact[]): string | null {
  if (artifacts.some((artifact) => normalizeArtifactKind(artifact) === "mindmap")) return "mindmap";
  return artifacts[0]?.type ?? null;
}

function ArtifactModalButton({
  group,
}: {
  group: ArtifactGroup;
}) {
  const [open, setOpen] = useState(false);
  const { Icon, label } = artifactGroupMeta(group.outputType);
  const createdAt = group.createdAt ?? group.artifacts.find((artifact) => artifact.created_at)?.created_at ?? undefined;
  const date = createdAt
    ? new Date(createdAt).toLocaleString(undefined, {
        month: "short",
        day: "numeric",
        hour: "numeric",
        minute: "2-digit",
      })
    : null;

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        className={cn(
          "flex w-full items-center gap-3 rounded-lg border border-border bg-surface p-3 text-left",
          "transition-colors hover:border-primary/40 hover:bg-primary/5",
          "focus:outline-none focus-visible:ring-2 focus-visible:ring-ring",
        )}
      >
        <span className="inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-md bg-primary/15 text-primary">
          <Icon className="h-4 w-4" />
        </span>
        <span className="flex min-w-0 flex-1 flex-col gap-0.5">
          <span className="text-sm font-medium">{label}</span>
          <span className="truncate text-[11px] text-muted-foreground">
            {date ? `Generated ${date} - click to open` : "Click to open"}
          </span>
        </span>
      </button>

      {open && <ArtifactModal group={group} title={label} Icon={Icon} onClose={() => setOpen(false)} />}
    </>
  );
}

function ArtifactModal({
  group,
  title,
  Icon,
  onClose,
}: {
  group: ArtifactGroup;
  title: string;
  Icon: React.ComponentType<{ className?: string }>;
  onClose: () => void;
}) {
  const isMindmapGroup = group.outputType === "mindmap" || group.artifacts.some((artifact) => normalizeArtifactKind(artifact) === "mindmap");
  const downloadArtifact = isMindmapGroup ? null : findDownloadArtifact(group.artifacts);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  useEffect(() => {
    const originalOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = originalOverflow;
    };
  }, []);

  const modal = (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-background/85 p-4 backdrop-blur-sm sm:p-6" role="presentation" onClick={onClose}>
      <section
        className="flex h-[min(90vh,54rem)] w-[min(92vw,78rem)] flex-col overflow-hidden rounded-lg border border-border bg-surface shadow-2xl"
        role="dialog"
        aria-modal="true"
        aria-label={title}
        onClick={(event) => event.stopPropagation()}
      >
        <header className="app-chrome flex items-center justify-between gap-3 border-b border-border px-4 py-3">
          <div className="flex min-w-0 items-center gap-2">
            <Icon className="h-4 w-4 shrink-0 text-primary" />
            <h2 className="truncate text-sm font-semibold">{title}</h2>
          </div>
          <div className="flex shrink-0 items-center gap-2">
            {downloadArtifact && (
              <a
                href={artifactHref(downloadArtifact)}
                download={downloadArtifact.filename || undefined}
                className={cn(
                  "inline-flex h-9 items-center justify-center gap-2 rounded-md border border-border bg-surface px-3 text-sm font-medium",
                  "text-surface-foreground shadow-sm shadow-black/5 transition-colors hover:bg-muted active:bg-muted/80 dark:shadow-none",
                )}
              >
                <Download className="h-4 w-4" />
                Download
              </a>
            )}
            <Button type="button" variant="ghost" size="icon" onClick={onClose} aria-label="Close" title="Close">
              <X className="h-4 w-4" />
            </Button>
          </div>
        </header>

        <div className="min-h-0 flex-1 overflow-y-auto p-4">
          <div className="flex flex-col gap-3">
            {previewArtifactsForGroup(group.artifacts).map((artifact) => (
              <ArtifactRenderer key={artifactId(artifact)} artifact={artifact} siblings={group.artifacts} />
            ))}
          </div>
        </div>
      </section>
    </div>
  );

  return createPortal(modal, document.body);
}

function ArtifactRenderer({
  artifact,
  siblings,
}: {
  artifact: Artifact;
  siblings?: Artifact[];
}) {
  const kind = normalizeArtifactKind(artifact);

  if (kind === "mindmap") {
    return (
      <JsonArtifactBoundary<MindmapPayload> artifact={artifact}>
        {(payload) => <MindmapRenderer payload={payload} />}
      </JsonArtifactBoundary>
    );
  }

  if (kind === "quiz") {
    return (
      <JsonArtifactBoundary<QuizArtifactPayload> artifact={artifact}>
        {(payload) => <QuizArtifactPreview payload={payload} />}
      </JsonArtifactBoundary>
    );
  }

  if (kind === "podcast") {
    return <PodcastAudioPreview artifact={artifact} />;
  }

  if (kind === "transcript") {
    return (
      <TextArtifactBoundary artifact={artifact}>
        {(text) => <TextArtifactPreview title={artifact.filename} text={text} />}
      </TextArtifactBoundary>
    );
  }

  if (kind === "json") {
    return (
      <JsonArtifactBoundary<unknown> artifact={artifact}>
        {(payload) => <JsonArtifactPreview payload={payload} />}
      </JsonArtifactBoundary>
    );
  }

  return <ArtifactFilePreview artifact={artifact} siblings={siblings} />;
}

function PodcastAudioPreview({ artifact }: { artifact: Artifact }) {
  return (
    <section className="rounded-lg border border-border bg-background p-4">
      <div className="mb-3 flex items-center gap-3">
        <span className="inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-md bg-primary/15 text-primary">
          <Mic2 className="h-4 w-4" />
        </span>
        <div className="min-w-0">
          <h3 className="truncate text-sm font-semibold">Two-host podcast</h3>
          <p className="mt-0.5 text-xs text-muted-foreground">Alex and Sam explain the selected course material.</p>
        </div>
      </div>
      <audio controls preload="metadata" className="w-full" src={artifactHref(artifact)}>
        Your browser does not support audio playback.
      </audio>
    </section>
  );
}

type QuizArtifactPayload = {
  title?: string;
  intro_message?: string;
  generation_run_id?: string;
  generation_mode?: string;
  questions?: Array<{
    id?: string;
    type?: string;
    bloom_level?: string;
    question?: string;
    options?: string[];
    correct_index?: number;
    answer?: boolean;
    explanation?: string;
    concept?: string;
  }>;
  bloom_distribution?: Record<string, number>;
};

function QuizArtifactPreview({ payload }: { payload: QuizArtifactPayload }) {
  const questions = Array.isArray(payload.questions) ? payload.questions : [];
  const [answers, setAnswers] = useState<Record<string, number>>({});
  const [submitted, setSubmitted] = useState(false);
  const questionKey = (question: NonNullable<QuizArtifactPayload["questions"]>[number], index: number) =>
    question.id || `question-${index}`;
  const answeredCount = questions.filter((question, index) => answers[questionKey(question, index)] !== undefined).length;
  const allAnswered = questions.length > 0 && answeredCount === questions.length;
  const correctCount = questions.reduce((score, question, index) => {
    const correctIndex = quizCorrectOptionIndex(question);
    return score + (correctIndex !== null && answers[questionKey(question, index)] === correctIndex ? 1 : 0);
  }, 0);
  const scorePercent = questions.length ? Math.round((correctCount / questions.length) * 100) : 0;

  useEffect(() => {
    setAnswers({});
    setSubmitted(false);
  }, [payload.generation_run_id]);

  return (
    <div className="flex flex-col gap-4">
      <div>
        <h3 className="text-lg font-semibold">{payload.title || "Quiz"}</h3>
        {payload.intro_message && (
          <div className="mt-2 text-muted-foreground">
            <QuizMarkdown content={payload.intro_message} />
          </div>
        )}
      </div>

      {Object.keys(payload.bloom_distribution ?? {}).length > 0 && (
        <div className="flex flex-wrap gap-2">
          {Object.entries(payload.bloom_distribution ?? {}).map(([level, count]) => (
            <Badge key={level} variant="muted">
              {level}: {count}
            </Badge>
          ))}
        </div>
      )}

      {submitted && (
        <section className="rounded-lg border border-primary/35 bg-primary/10 p-4 text-center">
          <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Final score</p>
          <p className="mt-1 text-3xl font-bold text-foreground">
            {correctCount}/{questions.length}
          </p>
          <p className="mt-1 text-sm font-medium text-primary">{scorePercent}%</p>
        </section>
      )}

      <div className="flex flex-col gap-3">
        {questions.map((question, index) => {
          const key = questionKey(question, index);
          const options = quizQuestionOptions(question);
          const selectedIndex = answers[key];
          const correctIndex = quizCorrectOptionIndex(question);
          const isCorrect = submitted && correctIndex !== null && selectedIndex === correctIndex;
          return (
            <article
              key={key}
              className={cn(
                "rounded-md border bg-background/50 p-3 transition-colors",
                !submitted && "border-border",
                submitted && isCorrect && "border-[hsl(var(--success))]/50 bg-[hsl(var(--success))]/5",
                submitted && !isCorrect && "border-[hsl(var(--danger))]/50 bg-[hsl(var(--danger))]/5",
              )}
            >
              <div className="flex flex-wrap items-center gap-2">
                <Badge variant="primary">Question {index + 1}</Badge>
                {question.bloom_level && <Badge variant="muted">{question.bloom_level}</Badge>}
                {question.concept && <Badge variant="muted">{question.concept}</Badge>}
                {submitted && <Badge variant={isCorrect ? "success" : "danger"}>{isCorrect ? "Correct" : "Incorrect"}</Badge>}
              </div>
              <div className="mt-3 font-medium text-foreground">
                <QuizMarkdown content={question.question || "Question"} />
              </div>
              <div className="mt-3 flex flex-col gap-2">
                {options.map((option, optionIndex) => {
                  const selected = selectedIndex === optionIndex;
                  const correct = correctIndex === optionIndex;
                  return (
                    <label
                      key={`${key}-${optionIndex}`}
                      className={cn(
                        "flex cursor-pointer items-start gap-3 rounded-md border border-border px-3 py-2.5 text-sm leading-5 transition-colors",
                        !submitted && selected && "border-primary/60 bg-primary/10 text-foreground",
                        !submitted && !selected && "hover:border-primary/35 hover:bg-primary/5",
                        submitted && correct && "border-[hsl(var(--success))]/60 bg-[hsl(var(--success))]/10 text-foreground",
                        submitted && selected && !correct && "border-[hsl(var(--danger))]/60 bg-[hsl(var(--danger))]/10 text-foreground",
                        submitted && "cursor-default",
                      )}
                    >
                      <input
                        type="radio"
                        name={`quiz-${payload.generation_run_id || "artifact"}-${key}`}
                        value={optionIndex}
                        checked={selected}
                        disabled={submitted}
                        onChange={() => setAnswers((current) => ({ ...current, [key]: optionIndex }))}
                        className="mt-0.5 h-4 w-4 shrink-0 accent-primary"
                      />
                      <span className="min-w-0 flex-1">
                        <QuizMarkdown content={option} />
                      </span>
                      {submitted && correct && <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-[hsl(var(--success))]" />}
                      {submitted && selected && !correct && <X className="mt-0.5 h-4 w-4 shrink-0 text-[hsl(var(--danger))]" />}
                    </label>
                  );
                })}
              </div>
              {submitted && !isCorrect && correctIndex !== null && options[correctIndex] && (
                <div className="mt-3 text-sm font-medium text-[hsl(var(--success))]">
                  <span>Correct answer:</span>
                  <QuizMarkdown content={options[correctIndex]} />
                </div>
              )}
              {submitted && question.explanation && (
                <div className="mt-2 text-xs leading-5 text-muted-foreground">
                  <QuizMarkdown content={question.explanation} />
                </div>
              )}
            </article>
          );
        })}
      </div>

      {questions.length > 0 && (
        <section className="sticky bottom-0 rounded-lg border border-border bg-surface/95 p-4 shadow-lg backdrop-blur">
          {!submitted ? (
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div className="min-w-0 flex-1">
                <div className="mb-2 flex items-center justify-between gap-3 text-xs text-muted-foreground">
                  <span>Answered {answeredCount} of {questions.length}</span>
                  {!allAnswered && <span>Answer every question to submit.</span>}
                </div>
                <ProgressBar percent={(answeredCount / questions.length) * 100} />
              </div>
              <Button type="button" disabled={!allAnswered} onClick={() => setSubmitted(true)} className="sm:ml-4">
                Submit quiz
              </Button>
            </div>
          ) : (
            <div className="flex flex-wrap items-center justify-between gap-3">
              <p className="text-sm font-medium">
                Score: {correctCount}/{questions.length} ({scorePercent}%)
              </p>
              <Button
                type="button"
                variant="secondary"
                onClick={() => {
                  setAnswers({});
                  setSubmitted(false);
                }}
              >
                <RotateCcw className="h-4 w-4" />
                Try again
              </Button>
            </div>
          )}
        </section>
      )}
    </div>
  );
}

function QuizMarkdown({ content }: { content: string }) {
  return (
    <AssistantMarkdown
      content={content}
      className="!max-w-none !text-sm [&_.katex-display]:!my-2 [&_.katex-display]:!py-1 [&_p]:!mb-0 [&_p]:!leading-6"
    />
  );
}

function quizQuestionOptions(question: NonNullable<QuizArtifactPayload["questions"]>[number]): string[] {
  if (Array.isArray(question.options) && question.options.length > 0) return question.options;
  if (typeof question.answer === "boolean") return ["True", "False"];
  return [];
}

function quizCorrectOptionIndex(question: NonNullable<QuizArtifactPayload["questions"]>[number]): number | null {
  if (typeof question.correct_index === "number" && Number.isInteger(question.correct_index)) {
    return question.correct_index;
  }
  if (typeof question.answer === "boolean") return question.answer ? 0 : 1;
  return null;
}

function TextArtifactBoundary({
  artifact,
  children,
}: {
  artifact: Artifact;
  children: (payload: string) => React.ReactNode;
}) {
  const [payload, setPayload] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const url = artifactHref(artifact);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    setPayload(null);

    fetch(url, { credentials: "omit" })
      .then(async (response) => {
        if (!response.ok) throw new Error(`Artifact fetch failed (${response.status})`);
        return response.text();
      })
      .then((data) => {
        if (!cancelled) setPayload(data);
      })
      .catch((err) => {
        if (!cancelled) setError((err as Error).message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [url]);

  if (loading) return <div className="text-xs text-muted-foreground">Loading artifact...</div>;
  if (error) return <div className="text-xs text-[hsl(var(--danger))]">Couldn't load artifact: {error}</div>;
  return <>{children(payload ?? "")}</>;
}

function TextArtifactPreview({ title, text }: { title: string; text: string }) {
  return (
    <div className="flex flex-col gap-3">
      <h3 className="text-sm font-semibold">{title}</h3>
      <pre className="max-h-[70vh] overflow-auto whitespace-pre-wrap rounded-md border border-border bg-background p-4 text-sm leading-6 text-foreground">
        {text}
      </pre>
    </div>
  );
}

function JsonArtifactPreview({ payload }: { payload: unknown }) {
  return (
    <pre className="max-h-[70vh] overflow-auto whitespace-pre-wrap rounded-md border border-border bg-background p-4 text-xs leading-5 text-foreground">
      {JSON.stringify(payload, null, 2)}
    </pre>
  );
}

function ArtifactFilePreview({ artifact }: { artifact: Artifact; siblings?: Artifact[] }) {
  const url = artifactHref(artifact);
  const kind = normalizeArtifactKind(artifact);
  const canFrame = kind === "html" || kind === "report" || artifact.filename.toLowerCase().endsWith(".pdf");

  if (canFrame) {
    return (
      <iframe
        title={artifact.filename}
        src={url}
        className="h-[72vh] w-full rounded-md border border-border bg-background"
        sandbox="allow-scripts allow-same-origin"
      />
    );
  }

  return (
    <div className="rounded-md border border-border bg-background p-4">
      <div className="flex items-start gap-3">
        <FileText className="mt-0.5 h-5 w-5 shrink-0 text-primary" />
        <div className="min-w-0 flex-1">
          <h3 className="truncate text-sm font-semibold">{artifact.filename}</h3>
          <p className="mt-1 text-xs leading-5 text-muted-foreground">This file type is ready to download.</p>
        </div>
      </div>
    </div>
  );
}

function JsonArtifactBoundary<T>({
  artifact,
  children,
}: {
  artifact: Artifact;
  children: (payload: T) => React.ReactNode;
}) {
  const [payload, setPayload] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const url = artifactHref(artifact);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    setPayload(null);

    fetch(url, { credentials: "omit" })
      .then(async (response) => {
        if (!response.ok) throw new Error(`Artifact fetch failed (${response.status})`);
        return (await response.json()) as T;
      })
      .then((data) => {
        if (!cancelled) setPayload(data);
      })
      .catch((err) => {
        if (!cancelled) setError((err as Error).message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [url]);

  if (loading) return <div className="text-xs text-muted-foreground">Loading artifact...</div>;
  if (error) return <div className="text-xs text-[hsl(var(--danger))]">Couldn't load artifact: {error}</div>;
  if (!payload) return null;
  return <>{children(payload)}</>;
}

function artifactGroupMeta(outputType: string | null): {
  Icon: React.ComponentType<{ className?: string }>;
  label: string;
} {
  if (outputType === "mindmap") return { Icon: Network, label: "Mind map" };
  if (outputType === "podcast" || outputType === "transcript") return { Icon: Mic2, label: "Podcast" };
  if (outputType === "quiz") return { Icon: FileText, label: "Quiz" };
  if (outputType === "presentation") return { Icon: Presentation, label: "Presentation" };
  if (outputType === "chart") return { Icon: BarChart3, label: "Diagram" };
  if (outputType === "report") return { Icon: ScrollText, label: "Report" };
  if (outputType === "html") return { Icon: Eye, label: "Web preview" };
  if (outputType === "json") return { Icon: FileText, label: "Data file" };
  return { Icon: Eye, label: "Generated item" };
}

function findDownloadArtifact(artifacts: Artifact[]): Artifact | null {
  return (
    artifacts.find((artifact) => normalizeArtifactKind(artifact) === "podcast") ??
    artifacts.find((artifact) => normalizeArtifactKind(artifact) === "html") ??
    artifacts[0] ??
    null
  );
}

function previewArtifactsForGroup(artifacts: Artifact[]): Artifact[] {
  const mindmapArtifact = artifacts.find((artifact) => normalizeArtifactKind(artifact) === "mindmap");
  if (mindmapArtifact) return [mindmapArtifact];
  return [...artifacts].sort((left, right) => {
    const priority = (artifact: Artifact) => {
      const kind = normalizeArtifactKind(artifact);
      if (kind === "podcast") return 0;
      if (kind === "transcript") return 1;
      return 2;
    };
    return priority(left) - priority(right);
  });
}

function artifactHref(artifact: Artifact): string {
  return artifact.key ? api.artifactUrl(artifact.key) : artifact.url || "#";
}

function artifactId(artifact: Artifact): string {
  return artifact.key ?? artifact.url ?? `${artifact.type}:${artifact.filename}`;
}

function normalizeArtifactKind(artifact: Artifact): string {
  const type = artifact.type.toLowerCase();
  const name = artifact.filename.toLowerCase();
  if (type === "mindmap") return "mindmap";
  if (type === "html" || name.endsWith(".html")) return "html";
  if (type === "quiz") return "quiz";
  if (type === "transcript" || type === "text" || name.endsWith(".txt") || name.endsWith(".md")) return "transcript";
  if (type === "json" || name.endsWith(".json")) return "json";
  if (type === "podcast" || type === "audio") return "podcast";
  if (type === "presentation" || type === "pptx" || name.endsWith(".pptx")) return "presentation";
  if (type === "chart" || type === "diagram") return "chart";
  if (type === "report" || name.endsWith(".pdf")) return "report";
  return type || "file";
}

function SettingsPage({
  providers,
  parser,
  courseBuilderSettings,
  generatorSettings,
  retrieval,
  providerForm,
  editingProviderId,
  parserKey,
  retrievalBusy,
  courseBuilderSettingsBusy,
  courseBuilderSettingsError,
  generatorSettingsBusy,
  generatorSettingsError,
  canRebuildIndexes,
  theme,
  onBack,
  onProviderFormChange,
  onEditProvider,
  onCancelProviderEdit,
  onSetDefaultProvider,
  onDeleteProvider,
  onParserKeyChange,
  onThemeChange,
  onSaveProvider,
  onSaveParserKey,
  onClearParserKey,
  onTestProvider,
  onSetParserMode,
  onSetSequentialCourseUnlocking,
  onSetPodcastAudioEnabled,
  onRebuildIndexes,
}: {
  providers: ProviderRead[];
  parser: ParserSettings | null;
  courseBuilderSettings: CourseBuilderSettings | null;
  generatorSettings: GeneratorSettings | null;
  retrieval: RetrievalSettings | null;
  providerForm: ProviderForm;
  editingProviderId: string | null;
  parserKey: string;
  retrievalBusy: boolean;
  courseBuilderSettingsBusy: boolean;
  courseBuilderSettingsError: string;
  generatorSettingsBusy: boolean;
  generatorSettingsError: string;
  canRebuildIndexes: boolean;
  theme: Theme;
  onBack: () => void;
  onProviderFormChange: (next: ProviderForm) => void;
  onEditProvider: (provider: ProviderRead) => void;
  onCancelProviderEdit: () => void;
  onSetDefaultProvider: (providerId: string) => void;
  onDeleteProvider: (providerId: string) => void;
  onParserKeyChange: (value: string) => void;
  onThemeChange: (theme: Theme) => void;
  onSaveProvider: (event: FormEvent) => void;
  onSaveParserKey: (event: FormEvent) => void;
  onClearParserKey: () => void;
  onTestProvider: (providerId: string) => void;
  onSetParserMode: (useLocalParsersOnly: boolean) => void;
  onSetSequentialCourseUnlocking: (enabled: boolean) => void;
  onSetPodcastAudioEnabled: (enabled: boolean) => void;
  onRebuildIndexes: () => void;
}) {
  const [providerToDelete, setProviderToDelete] = useState<ProviderRead | null>(null);
  const editingProvider = providers.find((provider) => provider.id === editingProviderId);
  const defaultProvider = providers.find((provider) => provider.is_default_chat);

  return (
    <main className="min-h-dvh bg-background text-foreground">
      <header className="app-chrome app-pane sticky top-0 z-10 border-b border-border">
        <div className="mx-auto flex max-w-4xl items-center justify-between px-4 py-3 sm:px-6">
          <div className="flex items-center gap-3">
            <a
              href="/"
              className="flex h-9 w-9 items-center justify-center rounded-md bg-primary/15 text-primary transition-colors hover:bg-primary/25 active:bg-primary/20"
              aria-label="Back to conversations"
              title="Back to conversations"
              onClick={(event) => {
                event.preventDefault();
                onBack();
              }}
            >
              <ArrowLeft className="h-5 w-5" />
            </a>
            <div>
              <h1 className="text-lg font-semibold">Settings</h1>
              <p className="hidden text-xs text-muted-foreground sm:block">Global runtime configuration.</p>
            </div>
          </div>
          <div className="flex h-9 w-9 items-center justify-center rounded-md bg-muted text-muted-foreground">
            <GraduationCap className="h-5 w-5" />
          </div>
        </div>
      </header>

      <div className="mx-auto flex max-w-4xl flex-col gap-4 px-4 py-6 sm:px-6">
        <section className="rounded-md border border-border bg-surface">
          <header className="app-chrome flex items-center justify-between gap-3 border-b border-border px-5 py-3">
            <div className="flex items-center gap-2">
              <Palette className="h-4 w-4 text-primary" />
              <h2 className="text-sm font-semibold">Appearance</h2>
            </div>
            <StatusPill active>{theme === "dark" ? "Dark" : "Light"}</StatusPill>
          </header>
          <div className="px-5 py-4">
            <div className="grid max-w-md grid-cols-2 gap-2 rounded-md border border-border bg-background p-1">
              {(["dark", "light"] as Theme[]).map((option) => {
                const active = theme === option;
                return (
                  <button
                    key={option}
                    type="button"
                    onClick={() => onThemeChange(option)}
                    className={cn(
                      "app-chrome flex h-10 items-center justify-center gap-2 rounded-sm px-3 text-sm font-medium transition-colors",
                      "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                      active ? "bg-primary text-primary-foreground shadow-sm" : "text-muted-foreground hover:bg-muted hover:text-foreground",
                    )}
                    aria-pressed={active}
                  >
                    {option === "dark" ? <Moon className="h-4 w-4" /> : <Sun className="h-4 w-4" />}
                    {option === "dark" ? "Dark" : "Light"}
                  </button>
                );
              })}
            </div>
          </div>
        </section>

        <section className="rounded-md border border-border bg-surface">
          <header className="app-chrome flex items-center justify-between gap-3 border-b border-border px-5 py-3">
            <div className="flex items-center gap-2">
              <Mic2 className="h-4 w-4 text-primary" />
              <h2 className="text-sm font-semibold">Podcast generation</h2>
            </div>
            <StatusPill active={Boolean(generatorSettings?.podcast_audio_enabled)}>
              {generatorSettings
                ? generatorSettings.podcast_audio_enabled
                  ? "Audio + transcript"
                  : "Transcript only"
                : "Loading"}
            </StatusPill>
          </header>
          <div className="px-5 py-4">
            <label className="flex cursor-pointer items-start justify-between gap-4 rounded-md border border-border bg-background px-3 py-3">
              <span className="min-w-0">
                <span className="block text-sm font-medium">Generate an MP3 podcast</span>
                <span className="mt-1 block text-xs leading-5 text-muted-foreground">
                  When enabled, TeacherLM loads the local voice model and creates an MP3 plus the transcript. Disable it on lower-powered computers to generate only the transcript without loading or running the voice model.
                </span>
              </span>
              <input
                type="checkbox"
                checked={generatorSettings?.podcast_audio_enabled ?? true}
                disabled={!generatorSettings || generatorSettingsBusy}
                onChange={(event) => onSetPodcastAudioEnabled(event.target.checked)}
                className="mt-0.5 h-4 w-4 shrink-0 accent-primary"
                aria-label="Generate podcast MP3 audio"
              />
            </label>
            {generatorSettingsError && (
              <p className="mt-2 text-xs text-[hsl(var(--danger))]" role="alert">
                {generatorSettingsError}
              </p>
            )}
          </div>
        </section>

        <section className="rounded-md border border-border bg-surface">
          <header className="app-chrome flex items-center justify-between gap-3 border-b border-border px-5 py-3">
            <div className="flex items-center gap-2">
              <BookOpen className="h-4 w-4 text-primary" />
              <h2 className="text-sm font-semibold">Course progression</h2>
            </div>
            <StatusPill active={Boolean(courseBuilderSettings?.sequential_unlocking_enabled)}>
              {courseBuilderSettings?.sequential_unlocking_enabled ? "Sequential" : "Open"}
            </StatusPill>
          </header>
          <div className="px-5 py-4">
            <label className="flex cursor-pointer items-start justify-between gap-4 rounded-md border border-border bg-background px-3 py-3">
              <span className="min-w-0">
                <span className="block text-sm font-medium">Lock course content until previous work is passed</span>
                <span className="mt-1 block text-xs leading-5 text-muted-foreground">
                  When enabled, subchapters unlock in order and the next chapter opens after its previous chapter quiz is passed. When disabled, every generated chapter and subchapter is available; assessments keep their prerequisites.
                </span>
              </span>
              <input
                type="checkbox"
                checked={courseBuilderSettings?.sequential_unlocking_enabled ?? true}
                disabled={!courseBuilderSettings || courseBuilderSettingsBusy}
                onChange={(event) => onSetSequentialCourseUnlocking(event.target.checked)}
                className="mt-0.5 h-4 w-4 shrink-0 accent-primary"
              />
            </label>
            {courseBuilderSettingsError && (
              <p className="mt-2 text-xs text-[hsl(var(--danger))]" role="alert">
                {courseBuilderSettingsError}
              </p>
            )}
          </div>
        </section>

        <section className="rounded-md border border-border bg-surface">
          <header className="app-chrome flex items-center justify-between gap-3 border-b border-border px-5 py-3">
            <div className="flex items-center gap-2">
              <Network className="h-4 w-4 text-primary" />
              <h2 className="text-sm font-semibold">Course search</h2>
            </div>
            <StatusPill active={Boolean(retrieval?.index_status?.ready)}>
              {retrieval?.index_status?.ready ? "Ready" : retrieval ? "Needs refresh" : "Loading"}
            </StatusPill>
          </header>
          <div className="flex flex-col gap-4 px-5 py-4">
            <div className="grid gap-2 text-xs text-muted-foreground sm:grid-cols-5">
              <Metric label="Passages" value={retrieval?.index_status?.chunk_count ?? 0} />
              <Metric label="Ready" value={retrieval?.index_status?.embedded_chunk_count ?? 0} />
              <Metric label="Refresh" value={retrieval?.index_status?.stale_chunk_count ?? 0} />
              <Metric label="Study links" value={retrieval?.index_status?.graph_node_count ?? 0} />
              <Metric label="Connections" value={retrieval?.index_status?.graph_edge_count ?? 0} />
            </div>

            <div className="flex flex-wrap items-center justify-end gap-2">
              <Button type="button" variant="secondary" onClick={onRebuildIndexes} disabled={!canRebuildIndexes || retrievalBusy}>
                {retrievalBusy ? <Loader2 className="h-4 w-4 animate-spin" /> : <RotateCcw className="h-4 w-4" />}
                Refresh course search
              </Button>
            </div>
          </div>
        </section>

        <section className="rounded-md border border-border bg-surface">
          <header className="app-chrome flex items-center justify-between gap-3 border-b border-border px-5 py-3">
            <div className="flex items-center gap-2">
              <Bot className="h-4 w-4 text-primary" />
              <h2 className="text-sm font-semibold">Chat models</h2>
            </div>
            <StatusPill active={Boolean(defaultProvider)}>{defaultProvider?.display_name ?? "No default"}</StatusPill>
          </header>
          <div className="flex flex-col gap-5 px-5 py-4">
            <div className="flex flex-col gap-2">
              {providers.map((provider) => (
                <div
                  key={provider.id}
                  className={cn(
                    "grid grid-cols-[minmax(0,1fr)_auto] items-center gap-3 rounded-md border border-border bg-background px-3 py-3 md:grid-cols-[minmax(0,1fr)_auto_auto]",
                    provider.is_default_chat && "border-primary/35 bg-primary/5",
                  )}
                >
                  <div className="min-w-0">
                    <div className="flex min-w-0 items-center gap-2">
                      <strong className="truncate text-sm">{provider.display_name}</strong>
                      {provider.is_default_chat && <Badge variant="primary">Default</Badge>}
                    </div>
                    <div className="mt-1 truncate text-xs text-muted-foreground">
                      {provider.provider_type} / {provider.model_name}
                    </div>
                    <div className="truncate text-[11px] text-muted-foreground">
                      {provider.base_url}
                      {provider.api_key_set ? " / key saved" : ""}
                    </div>
                  </div>
                  <div className="flex flex-wrap items-center justify-end gap-1.5">
                    {!provider.is_default_chat && (
                      <Button type="button" variant="secondary" size="sm" onClick={() => onSetDefaultProvider(provider.id)}>
                        Use for chat
                      </Button>
                    )}
                    <Button type="button" variant="ghost" size="icon" title="Test connection" aria-label={`Test ${provider.display_name}`} onClick={() => onTestProvider(provider.id)}>
                      <KeyRound className="h-4 w-4" />
                    </Button>
                    <Button type="button" variant="secondary" size="sm" onClick={() => onEditProvider(provider)}>
                      Edit
                    </Button>
                    <Button
                      type="button"
                      variant="ghost"
                      size="icon"
                      title={providers.length <= 1 ? "Keep at least one chat model" : "Delete model"}
                      aria-label={`Delete ${provider.display_name}`}
                      disabled={providers.length <= 1}
                      onClick={() => setProviderToDelete(provider)}
                      className="hover:text-[hsl(var(--danger))]"
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                  <ProviderStatusBadge status={provider.status} />
                </div>
              ))}
            </div>

            {providerToDelete && (
              <div className="flex items-center justify-between gap-3 rounded-md border border-[hsl(var(--danger)/0.28)] bg-[hsl(var(--danger)/0.08)] px-3 py-3 text-sm">
                <span>Delete {providerToDelete.display_name}?</span>
                <div className="flex shrink-0 items-center gap-2">
                  <Button type="button" variant="secondary" size="sm" onClick={() => setProviderToDelete(null)}>
                    Cancel
                  </Button>
                  <Button
                    type="button"
                    variant="danger"
                    size="sm"
                    onClick={() => {
                      onDeleteProvider(providerToDelete.id);
                      setProviderToDelete(null);
                    }}
                  >
                    Delete
                  </Button>
                </div>
              </div>
            )}

            <form className="flex flex-col gap-4" onSubmit={onSaveProvider}>
              <h3 className="text-sm font-semibold">{editingProvider ? "Edit model" : "Add model"}</h3>
              <div className="grid gap-4 md:grid-cols-2">
                <Field label="Provider name">
                  <Input value={providerForm.display_name} onChange={(event) => onProviderFormChange({ ...providerForm, display_name: event.target.value })} placeholder="Provider name" />
                </Field>
                <Field label="Provider">
                  <select
                    value={providerForm.provider_type}
                    onChange={(event) => onProviderFormChange({ ...providerForm, provider_type: event.target.value })}
                    className="h-9 rounded-md border border-border bg-background px-3 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                  >
                    <option value="ollama">Ollama</option>
                    <option value="openai">OpenAI</option>
                    <option value="anthropic">Anthropic</option>
                    <option value="openai_compatible">OpenAI-compatible</option>
                    <option value="anthropic_compatible">Anthropic-compatible</option>
                  </select>
                </Field>
                <Field label={<span className="flex items-center gap-1.5"><Server className="h-3.5 w-3.5" />API link</span>}>
                  <Input value={providerForm.base_url} onChange={(event) => onProviderFormChange({ ...providerForm, base_url: event.target.value })} placeholder="Base URL" />
                </Field>
                <Field label="Model">
                  <Input value={providerForm.model_name} onChange={(event) => onProviderFormChange({ ...providerForm, model_name: event.target.value })} placeholder="Model" />
                </Field>
              </div>
              <Field label={<span className="flex items-center gap-1.5"><KeyRound className="h-3.5 w-3.5" />API key</span>}>
                <Input
                  type="password"
                  value={providerForm.api_key}
                  onChange={(event) => onProviderFormChange({ ...providerForm, api_key: event.target.value })}
                  placeholder={editingProvider?.api_key_set ? "New API key (leave blank to keep current)" : "API key"}
                  autoComplete="off"
                />
              </Field>
              <label className="flex items-center justify-between gap-4 rounded-md border border-border bg-background px-3 py-3">
                <span className="text-sm font-medium">Use this model for chat</span>
                <input
                  type="checkbox"
                  checked={providerForm.is_default_chat}
                  onChange={(event) => onProviderFormChange({ ...providerForm, is_default_chat: event.target.checked })}
                  className="h-4 w-4 accent-primary"
                />
              </label>
              <div className="flex flex-wrap items-center justify-end gap-2">
                <Button type="submit" variant="primary">
                  <Save className="h-4 w-4" />
                  {editingProvider ? "Save changes" : "Save model"}
                </Button>
                {editingProvider && (
                  <Button type="button" variant="secondary" onClick={onCancelProviderEdit}>
                    Cancel
                  </Button>
                )}
              </div>
            </form>
          </div>
        </section>

        <section className="rounded-md border border-border bg-surface">
          <header className="app-chrome flex items-center justify-between gap-3 border-b border-border px-5 py-3">
            <div className="flex items-center gap-2">
              <UploadCloud className="h-4 w-4 text-primary" />
              <h2 className="text-sm font-semibold">LlamaCloud parser</h2>
            </div>
            <StatusPill active={Boolean(parser?.llama_cloud_api_key_set)}>
              {parser?.llama_cloud_api_key_set ? "Key saved" : "No DB key"}
            </StatusPill>
          </header>
          <div className="flex flex-col gap-4 px-5 py-4">
            <div className="grid max-w-md grid-cols-2 gap-2 rounded-md border border-border bg-background p-1">
              <button
                type="button"
                onClick={() => onSetParserMode(true)}
                className={cn(
                  "app-chrome flex h-10 items-center justify-center rounded-sm px-3 text-sm font-medium transition-colors",
                  parser?.use_local_parsers_only ? "bg-primary text-primary-foreground shadow-sm" : "text-muted-foreground hover:bg-muted hover:text-foreground",
                )}
              >
                Local
              </button>
              <button
                type="button"
                onClick={() => onSetParserMode(false)}
                disabled={!parser?.llama_cloud_api_key_set}
                className={cn(
                  "app-chrome flex h-10 items-center justify-center rounded-sm px-3 text-sm font-medium transition-colors",
                  !parser?.use_local_parsers_only ? "bg-primary text-primary-foreground shadow-sm" : "text-muted-foreground hover:bg-muted hover:text-foreground",
                  !parser?.llama_cloud_api_key_set && "cursor-not-allowed opacity-50",
                )}
              >
                LlamaParse
              </button>
            </div>

            <form className="flex flex-col gap-3" onSubmit={onSaveParserKey}>
              <Field
                label={
                  <span className="flex items-center justify-between gap-3">
                    <span className="flex items-center gap-1.5">
                      <KeyRound className="h-3.5 w-3.5" />
                      LlamaParse key
                    </span>
                    <span>{parser?.llama_cloud_api_key_set ? "Configured" : "Empty"}</span>
                  </span>
                }
              >
                <div className="flex gap-2">
                  <Input
                    type="password"
                    value={parserKey}
                    onChange={(event) => onParserKeyChange(event.target.value)}
                    placeholder="llx-..."
                    autoComplete="off"
                  />
                  <Button type="submit" variant="primary" size="icon" title="Save LlamaParse key" aria-label="Save LlamaParse key" disabled={!parserKey.trim()}>
                    <Save className="h-4 w-4" />
                  </Button>
                </div>
              </Field>
            </form>
            <div className="flex justify-end">
              <Button type="button" variant="secondary" onClick={onClearParserKey} disabled={!parser?.llama_cloud_api_key_set}>
                Remove LlamaParse key
              </Button>
            </div>
          </div>
        </section>
      </div>
    </main>
  );
}

function DeleteConversationDialog({
  conversation,
  pending,
  onCancel,
  onConfirm,
}: {
  conversation: Conversation;
  pending: boolean;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-background/60 px-4 backdrop-blur-sm" role="presentation" onClick={pending ? undefined : onCancel}>
      <section
        className="w-full max-w-lg overflow-hidden rounded-lg border border-border bg-surface shadow-2xl"
        role="dialog"
        aria-modal="true"
        aria-label="Delete conversation"
        onClick={(event) => event.stopPropagation()}
      >
        <header className="flex items-start justify-between gap-4 border-b border-border px-5 py-4">
          <div>
            <h2 className="text-base font-semibold">Delete conversation?</h2>
            <p className="mt-1 text-sm leading-5 text-muted-foreground">
              "{conversation.title}" and all of its files, messages, and generated items will be removed. This can't be undone.
            </p>
          </div>
          <Button variant="ghost" size="icon" title="Close" aria-label="Close delete dialog" disabled={pending} onClick={onCancel}>
            <X className="h-4 w-4" />
          </Button>
        </header>
        <div className="flex justify-end gap-2 px-5 py-4">
          <Button variant="secondary" onClick={onCancel} disabled={pending}>
            Cancel
          </Button>
          <Button variant="danger" onClick={onConfirm} disabled={pending}>
            {pending ? <Loader2 className="h-4 w-4 animate-spin" /> : <Trash2 className="h-4 w-4" />}
            Delete
          </Button>
        </div>
      </section>
    </div>
  );
}

function Field({
  label,
  children,
}: {
  label: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <label className="flex flex-col gap-1.5">
      <span className="text-xs font-medium text-muted-foreground">{label}</span>
      {children}
    </label>
  );
}

function Metric({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="rounded-md border border-border bg-background px-3 py-2">
      <div className="text-[11px] text-muted-foreground">{label}</div>
      <div className="mt-1 text-sm font-semibold text-foreground">{value}</div>
    </div>
  );
}

function Input({
  className,
  type = "text",
  ...props
}: React.InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      type={type}
      className={cn(
        "flex h-9 w-full rounded-md border border-border bg-surface px-3 py-1 text-sm shadow-sm shadow-black/5 dark:bg-background dark:shadow-none",
        "placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
        "disabled:cursor-not-allowed disabled:opacity-50",
        className,
      )}
      {...props}
    />
  );
}

function Button({
  className,
  variant = "primary",
  size = "md",
  type = "button",
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: ButtonVariant;
  size?: ButtonSize;
}) {
  const variants: Record<ButtonVariant, string> = {
    primary: "bg-primary text-primary-foreground shadow-sm hover:bg-primary/90 active:bg-primary/85",
    secondary: "border border-border bg-surface text-surface-foreground shadow-sm shadow-black/5 hover:bg-muted active:bg-muted/80 dark:shadow-none",
    ghost: "text-foreground hover:bg-muted active:bg-muted/80",
    danger: "bg-danger text-white hover:bg-danger/90 active:bg-danger/80",
    link: "text-primary underline-offset-4 hover:underline",
  };
  const sizes: Record<ButtonSize, string> = {
    sm: "h-8 px-3",
    md: "h-9 px-4",
    lg: "h-10 px-5",
    icon: "h-9 w-9",
  };

  return (
    <button
      type={type}
      className={cn(
        "inline-flex select-none items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium transition-colors active:translate-y-px",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:pointer-events-none disabled:translate-y-0 disabled:opacity-50",
        variants[variant],
        sizes[size],
        className,
      )}
      {...props}
    />
  );
}

function Badge({
  variant = "default",
  className,
  ...props
}: React.HTMLAttributes<HTMLSpanElement> & {
  variant?: "default" | "muted" | "primary" | "success" | "warning" | "danger";
}) {
  const variants = {
    default: "border-border bg-surface text-surface-foreground",
    muted: "border-transparent bg-muted text-muted-foreground",
    primary: "border-transparent bg-primary/15 text-primary",
    success: "border-transparent bg-[hsl(var(--success)/0.18)] text-[hsl(var(--success))]",
    warning: "border-transparent bg-[hsl(var(--warning)/0.18)] text-[hsl(var(--warning))]",
    danger: "border-transparent bg-[hsl(var(--danger)/0.18)] text-[hsl(var(--danger))]",
  };
  return (
    <span
      className={cn("inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] font-medium", variants[variant], className)}
      {...props}
    />
  );
}

function StatusPill({
  active,
  children,
}: {
  active: boolean;
  children: React.ReactNode;
}) {
  return (
    <span
      className={cn(
        "rounded-md border px-2 py-1 text-[11px] font-medium",
        active ? "border-primary/30 bg-primary/10 text-primary" : "border-border bg-muted text-muted-foreground",
      )}
    >
      {children}
    </span>
  );
}

function FileStatusBadge({ status, error }: { status: string; error?: string | null }) {
  const labels: Record<string, string> = {
    uploaded: "Queued",
    parsing: "Parsing...",
    chunking: "Chunking...",
    extracting_concepts: "Extracting concepts...",
    planning_course: "Planning course structure...",
    building_course: "Building course...",
    embedding: "Embedding...",
    ready: "Ready",
    failed: "Failed",
  };
  if (status === "ready" || status === "ok") {
    return (
      <Badge variant="success" title="Ready to reference">
        <CheckCircle2 className="h-3 w-3" />
        {labels.ready}
      </Badge>
    );
  }
  if (status === "error" || status === "failed" || status === "unreachable") {
    return (
      <Badge variant="danger" title={error ?? status}>
        <AlertCircle className="h-3 w-3" />
        {labels.failed}
      </Badge>
    );
  }
  if (status === "queued" || status === "uploaded" || status === "unknown") {
    return <Badge variant="muted">{status === "unknown" ? "Not tested" : labels.uploaded}</Badge>;
  }
  return (
    <Badge variant="primary">
      <Loader2 className="h-3 w-3 animate-spin" />
      {labels[status] ?? "Processing..."}
    </Badge>
  );
}

function ProviderStatusBadge({ status }: { status: string }) {
  return <FileStatusBadge status={status === "ok" ? "ready" : status} />;
}

function StateCard({
  icon,
  title,
  body,
}: {
  icon: React.ReactNode;
  title: string;
  body: string;
}) {
  return (
    <section className="rounded-md border border-border bg-surface p-3">
      <div className="flex items-start gap-2">
        {icon}
        <div className="min-w-0">
          <h3 className="text-sm font-semibold">{title}</h3>
          <p className="mt-1 text-xs leading-5 text-muted-foreground">{body}</p>
        </div>
      </div>
    </section>
  );
}

function ProgressBar({ percent }: { percent: number }) {
  return (
    <div className="h-2 overflow-hidden rounded-full bg-muted">
      <div className="h-full bg-primary transition-all" style={{ width: `${Math.max(0, Math.min(100, percent))}%` }} />
    </div>
  );
}

function ArtifactLink({ artifact }: { artifact: Artifact }) {
  const [open, setOpen] = useState(false);
  const kind = normalizeArtifactKind(artifact);
  const { Icon, label } = artifactGroupMeta(kind);
  const group: ArtifactGroup = {
    id: artifactId(artifact),
    outputType: kind,
    createdAt: artifact.created_at ?? undefined,
    artifacts: [artifact],
  };

  return (
    <>
      <button
        type="button"
        className="flex min-h-10 w-full items-center gap-2 rounded-md border border-border bg-surface p-2 text-left text-foreground transition-colors hover:bg-muted focus:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        onClick={() => setOpen(true)}
      >
        <Icon className="h-4 w-4 shrink-0 text-primary" />
        <span className="min-w-0 flex-1 truncate text-sm font-medium">{artifact.filename || label}</span>
        <Badge variant="muted">{artifact.type}</Badge>
      </button>

      {open && <ArtifactModal group={group} title={label} Icon={Icon} onClose={() => setOpen(false)} />}
    </>
  );
}

function formatRelativeTime(value: string): string {
  const timestamp = new Date(value).getTime();
  if (Number.isNaN(timestamp)) return "recently";
  const seconds = Math.max(0, Math.floor((Date.now() - timestamp) / 1000));
  if (seconds < 60) return "just now";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  if (days < 30) return `${days}d ago`;
  return new Date(value).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}

function conversationPath(conversationId: string): string {
  return `/c/${encodeURIComponent(conversationId)}`;
}

function routeFromLocation(): AppRoute {
  const pathname = window.location.pathname;
  const conversationMatch = pathname.match(/^\/c\/([^/]+)\/?$/);
  if (conversationMatch?.[1]) {
    return { kind: "conversation", conversationId: decodeURIComponent(conversationMatch[1]) };
  }
  if (/^\/settings\/?$/.test(pathname)) {
    return { kind: "settings" };
  }
  return { kind: "home" };
}

function pathForRoute(route: AppRoute): string {
  if (route.kind === "conversation") return conversationPath(route.conversationId);
  if (route.kind === "settings") return "/settings";
  return "/";
}

function buildHint(learner: LearnerState | null): string | null {
  if (!learner) return null;
  const strong = learner.understood_concepts?.[0];
  const weak = learner.struggling_concepts?.[0];
  if (strong && weak) return `You're strong on ${strong}; let's review ${weak}.`;
  if (strong) return `Great progress on ${strong}.`;
  if (weak) return `Let's revisit ${weak}.`;
  return null;
}

function optimisticMessage(conversationId: string, role: string, content: string, outputType: string): Message {
  return {
    id: `optimistic-${role}-${Date.now()}`,
    conversation_id: conversationId,
    role,
    content,
    output_type: outputType,
    artifacts: [],
    sources: [],
    metadata: {},
    created_at: new Date().toISOString(),
  };
}

function cn(...classes: Array<string | false | null | undefined>): string {
  return classes.filter(Boolean).join(" ");
}

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException
    ? error.name === "AbortError"
    : error instanceof Error && error.name === "AbortError";
}

function readSavedTheme(): Theme {
  try {
    const storedUi = window.localStorage.getItem("teacherlm-ui");
    if (storedUi) {
      const parsed = JSON.parse(storedUi) as { state?: { theme?: unknown } };
      if (parsed.state?.theme === "light" || parsed.state?.theme === "dark") {
        return parsed.state.theme;
      }
    }
    const legacy = window.localStorage.getItem("teacherlm-theme");
    if (legacy === "light" || legacy === "dark") return legacy;
  } catch {
    return "dark";
  }
  return "dark";
}

function applyTheme(theme: Theme) {
  document.documentElement.classList.toggle("dark", theme === "dark");
  document.documentElement.style.colorScheme = theme;
  window.localStorage.setItem("teacherlm-theme", theme);
  window.localStorage.setItem("teacherlm-ui", JSON.stringify({ state: { theme }, version: 1 }));
}

async function copyToClipboard(value: string): Promise<boolean> {
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(value);
      return true;
    }
  } catch {
    // Fall through to the legacy copy path.
  }

  const textarea = document.createElement("textarea");
  textarea.value = value;
  textarea.setAttribute("readonly", "");
  textarea.style.position = "fixed";
  textarea.style.left = "-9999px";
  textarea.style.top = "0";
  document.body.appendChild(textarea);
  textarea.select();
  try {
    return document.execCommand("copy");
  } catch {
    return false;
  } finally {
    document.body.removeChild(textarea);
  }
}
