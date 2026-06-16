# Platform Frontend

`platform/frontend` is the student workspace. It is a Next.js application that lets a student manage conversations, upload course files, chat with the tutor, generate study outputs, work through a generated course, and track learning progress.

The frontend is not a static marketing site. Its first job is to be the actual study tool.

## Stack

| Tool | Use |
| --- | --- |
| Next.js 14 | App router, pages, build, and server runtime. |
| React 18 | UI components and stateful interactions. |
| TypeScript | Type-safe API models and UI props. |
| TanStack React Query v5 | Server-state fetching, caching, invalidation, and mutations. |
| Zustand v5 | Local UI/session stores. |
| Tailwind CSS | Utility styling. |
| MUI | Theme integration and baseline. |
| Radix Dialog/Tooltip | Accessible modal and tooltip primitives. |
| lucide-react | Icons. |
| sonner | Toast notifications. |
| react-markdown | Markdown rendering. |
| remark/rehype math and KaTeX | Math rendering. |
| markmap | Mindmap visualization. |
| mermaid and svg-pan-zoom | Diagram/chart rendering and zooming. |
| react-pdf | PDF preview for downloadable file artifacts. |
| recharts | Charting support. |

## Top-Level Files

| Path | Purpose |
| --- | --- |
| `package.json` | Declares frontend dependencies and scripts. |
| `package-lock.json` | Locked npm dependency tree. |
| `Dockerfile` | Builds Next.js standalone image on Node 20 Alpine. |
| `next.config.mjs` | Next.js configuration. |
| `tailwind.config.ts` | Tailwind theme/content configuration. |
| `postcss.config.js` | PostCSS/Tailwind processing. |
| `tsconfig.json` | TypeScript compiler options. |
| `public/pdf.worker.min.mjs` | PDF.js worker used by PDF previews. |
| `CLAUDE.md` | Local assistant guidance for frontend work. |

## App Routes

| Path | Route | Purpose |
| --- | --- | --- |
| `app/layout.tsx` | Global layout | Adds metadata, global CSS, theme bootstrap script, providers, and KaTeX CSS. |
| `app/providers.tsx` | Provider wrapper | Creates React Query client, MUI theme from CSS variables, baseline, and toast provider. |
| `app/globals.css` | Global styles | Theme variables, Tailwind layers, layout-level styling, markdown/math support, and app polish. |
| `app/page.tsx` | `/` | Conversation list and entry point. Lets the student create, open, and delete conversations and reach settings. |
| `app/c/[id]/page.tsx` | `/c/:id` | Conversation workspace route. Renders `Workspace`. |
| `app/settings/page.tsx` | `/settings` | Runtime settings and preferences page. Handles appearance, LLM provider/model/API key settings, parser key, and forced language. |

## API Client And Types

### `lib/types.ts`

Mirrors backend Pydantic schemas.

Major type groups:

- Conversations.
- Messages.
- Artifacts.
- Sources.
- Uploaded files.
- Generator registry entries.
- Learner state.
- Knowledge checks.
- Review tests.
- Course player.
- Knowledge graph.
- Course builder.
- SSE events.
- Runtime settings.
- Health/readiness.

This file is the frontend's contract with `platform/backend/schemas/`.

### `lib/api.ts`

Defines the API client.

Important details:

- `API_BASE_URL` comes from `NEXT_PUBLIC_API_BASE_URL`.
- In the browser, localhost-like API URLs are normalized to the current host where useful, which helps LAN/mobile use.
- `apiFetch` wraps fetch, JSON parsing, and error throwing.
- `ApiError` carries status and response details.

API namespaces:

- `conversations`
- `runtimeSettings`
- `knowledgeChecks`
- `reviewTests`
- `coursePlayer`
- `courseBuilder`
- `knowledgeGraph`
- `messages`
- `files`
- `generators`
- `health`

### `lib/sse.ts`

Implements manual POST-SSE parsing.

Why manual parsing exists:

- Browser `EventSource` only supports GET.
- Chat and generation need POST bodies.
- The backend streams Server-Sent Event formatted responses over a fetch body.

What it does:

- Uses `fetch`.
- Reads the response stream with `TextDecoderStream`.
- Splits SSE event blocks.
- Parses event names and data fields.
- Parses JSON event payloads.
- Calls handlers for chunk, token, source, artifact, progress, done, and error-like events.

### `lib/utils.ts`

Small utilities:

- `cn`: class name composition.
- `formatRelativeTime`: human-readable timestamps.

## Stores

Local state stores live under `stores/`.

### `conversationStore.ts`

Owns active conversation and streaming assistant state.

Important state:

- Active conversation ID.
- Temporary streaming assistant message.
- Streaming status.
- Abort controllers.
- Stream errors.

Important actions:

- Start a stream.
- Append text chunks.
- Add sources.
- Add artifacts.
- Merge metadata.
- End stream.
- Abort stream.
- Convert temporary streaming output into a message-like object.

Artifact deduplication happens here using keys such as artifact key, URL, and type.

### `progressStore.ts`

Local optimistic learner progress mirror.

Important behavior:

- Mirrors server learner state.
- Applies optimistic updates from streaming/generation events.
- Uses constants such as demonstrated step, struggle decay, and mastery thresholds.
- Produces hints for chat UI based on current progress.

The backend remains canonical; this store makes the UI feel immediate.

### `settingsStore.ts`

Persists user settings.

Important state:

- Theme-related preferences.
- Forced language.
- LLM provider choice.
- Provider base URLs.
- Default model selection.
- Runtime override choices.

Provider defaults include:

- OpenAI: `gpt-4.1-mini`
- Anthropic: `claude-sonnet-4-5`
- Local/Ollama: project default model.

### `uiStore.ts`

UI state:

- Theme.
- Sidebar collapsed state.
- Source file selection by conversation.
- Generator dialog open state.
- Active output type.

It also applies the theme class to the document.

## Hooks

Hooks live under `hooks/`.

| File | Purpose |
| --- | --- |
| `useChatStream.ts` | Shared stream runner for chat and generation. Handles optimistic user message, POST-SSE events, aborts, errors, artifacts, sources, learner updates, and query invalidation. |
| `useConversations.ts` | Conversation list/read/create/update/delete mutations and learner-state sync into progress store. |
| `useMessages.ts` | Loads messages for a conversation. |
| `useFiles.ts` | Lists files, uploads files, retries failed files, deletes files, and polls while ingestion is active. Sends forced language/LLM options on upload and retry. |
| `useGenerators.ts` | Loads generator registry entries for output type controls. |
| `useSourceFileSelection.ts` | Keeps selected ready file IDs valid. Defaults all ready files selected, forces the only ready file selected, and prevents deselecting the last selected source. |
| `useCourseBuilder.ts` | Loads and mutates generated course builder state. Polls running jobs and consumes coursebuilder progress EventSource stream. |
| `useCoursePlayer.ts` | Loads and mutates older course player state. |
| `useKnowledgeChecks.ts` | Starts/submits knowledge checks and invalidates learner progress. |
| `useReviewTests.ts` | Reads review due status and starts/submits/snoozes/dismisses review tests. |

## Workspace Layout

Workspace components live under `components/workspace/`.

### `Workspace.tsx`

Main study layout.

Responsibilities:

- Full-height workspace.
- Sources panel on the left.
- Main course/chat split in the center.
- Progress/generated-output panel on the right.
- Resizable course/chat area.
- Mobile drawers and mobile course/chat toggle.
- Editable conversation title.
- Generator dialog integration.
- Review test dialog integration.

### `SourcesPanel.tsx`

Wraps:

- File uploader.
- File list.
- Source file selection context.

### `CoursePanel.tsx`

Shows generated course content through `CourseBuilderPanel`.

### `ChatPanel.tsx`

Shows tutor chat and generation controls.

Important behavior:

- Loads learner state and files.
- Reads selected source file IDs.
- Disables chat/generation until at least one ready selected file exists.
- Shows hints derived from learner progress.

### `ProgressPanel.tsx`

Shows learner progress and generated artifacts.

Behavior:

- Collects persisted message artifacts.
- Includes streaming artifacts.
- Deduplicates artifacts.
- Shows quiz, podcast, and mindmap artifacts as modal buttons.
- Shows other downloadable/renderable artifacts inline where appropriate.

## File UI

File components live under `components/files/`.

| File | Purpose |
| --- | --- |
| `FileUploader.tsx` | Upload/dropzone UI. Sends selected files to backend and includes runtime language/LLM options. |
| `FileList.tsx` | Lists uploaded files, selected source files, retry/delete actions, and file statuses. |
| `FileStatusBadge.tsx` | Small status label for pending, processing, ready, failed, and related file states. |

The file UI is tied directly to source selection. Ready files become retrieval sources; non-ready files do not.

## Chat UI

Chat components live under `components/chat/`.

### `ChatInput.tsx`

Responsibilities:

- Autosizing textarea.
- Submit on Enter.
- Stop streaming button.
- Sends forced language and runtime options.
- Sends selected `source_file_ids`.

### `MessageList.tsx`

Responsibilities:

- Merges persisted messages with temporary streaming assistant output.
- Keeps the list scrolled to the latest message.
- Handles empty/loading states.

### `MessageBubble.tsx`

Large markdown and message rendering component.

Responsibilities:

- Render user and assistant bubbles.
- Render assistant markdown.
- Render sources/citations toggle.
- Render copy actions.
- Decide whether artifacts appear inline or in side panels.
- Repair common malformed math/Markdown patterns before render.
- Support GFM, math, KaTeX, code highlighting, tables, and source-rich answers.

It exports `AssistantMarkdown`, which course builder components reuse for rich educational block content.

### `OutputTypeButtons.tsx`

Displays enabled generator output types from the registry.

### `GeneratorDialog.tsx`

Collects generation options.

Current behavior:

- Shows enabled output types.
- Quiz exposes question count, difficulty, and question types.
- Quiz, mindmap, and podcast do not require a freeform topic in the current UI.
- Sends options through the generation stream hook.

## Artifact UI

Artifact components live under `components/artifacts/`.

| File | Purpose |
| --- | --- |
| `ArtifactRenderer.tsx` | Normalizes artifact type and fetches artifact JSON when needed. Routes to quiz, podcast, mindmap, chart, or file rendering. |
| `ArtifactModalButton.tsx` | Compact button that opens an artifact in a modal. Used for side-rail generated outputs. |
| `QuizRenderer.tsx` | Interactive quiz artifact UI. Can submit quiz attempts to backend so learner state updates. |
| `PodcastPlayer.tsx` | Audio player and transcript display for podcast artifacts. |
| `MindmapRenderer.tsx` | Dynamic markmap renderer with branch coloring, expand/collapse, and click-to-ask-teacher behavior. |
| `ChartRenderer.tsx` | Mermaid/chart renderer with zoom/pan support. |
| `FileDownload.tsx` | Download UI and PDF preview using react-pdf. |

## Course Builder UI

Course builder components live under `components/coursebuilder/`.

| File | Purpose |
| --- | --- |
| `CourseBuilderPanel.tsx` | Main generated course UI. Handles no-files, pending, running, failed, and ready states. Shows progress events, generated course header, chapter accordions, active content, rebuild button, and quiz flows. |
| `LessonBlockRenderer.tsx` | Renders lesson blocks such as markdown, tables, equations, charts, examples, and citations. Reuses assistant markdown rendering. |
| `Quiz.tsx` | Handles course builder chapter quiz display and submission. |
| `CitationList.tsx` | Displays source citations for generated course blocks. |
| `ChapterList.tsx` | Chapter list/accordion support component. |

Course builder is the main generated course surface in the current UI.

## Course Player UI

`components/course/CoursePlayer.tsx` renders the older/adaptive course player.

It supports:

- Chapter list.
- Lesson blocks.
- Locked/unlocked state.
- Chapter quiz submission.
- Knowledge graph and remediation hints.

It remains part of the frontend because backend routes and schemas still support it.

## Progress And Review UI

| Path | Purpose |
| --- | --- |
| `components/progress/LearnerProgress.tsx` | Learner progress visualization. Shows phase/objective/concept progress, weakest areas, mastery bars, known/struggling concepts, and knowledge check entry points. |
| `components/review/ReviewTestDialog.tsx` | Modal for due review tests. Supports start, answer, submit, snooze, and dismiss flows. |

`LearnerProgress` can synthesize display rows from legacy known/struggling concepts if richer phase data is not available.

## UI Primitives

Small reusable UI components live under `components/ui/`.

| File | Purpose |
| --- | --- |
| `Badge.tsx` | Status/category label. |
| `Button.tsx` | Styled button variants. |
| `Card.tsx` | Card container. |
| `Dialog.tsx` | Dialog wrapper. |
| `Input.tsx` | Styled input. |
| `Tooltip.tsx` | Tooltip wrapper. |

## Main Frontend Data Flow

Chat:

1. Student types in `ChatInput`.
2. `useChatStream` sends POST-SSE request to backend.
3. Backend streams events.
4. `lib/sse.ts` parses event blocks.
5. `conversationStore` appends streamed text, sources, artifacts, and metadata.
6. React Query invalidates messages, conversations, learner state, and review status on completion.

Upload:

1. Student uploads in `FileUploader`.
2. `useFiles` posts file to backend.
3. Backend enqueues ingestion.
4. `useFiles` polls while statuses are pending/processing.
5. Ready files become selectable sources.

Generation:

1. Student picks an output type.
2. `GeneratorDialog` collects options.
3. `useChatStream` sends generation request.
4. Backend dispatches to generator.
5. Streaming events update chat and side artifacts.

Course builder:

1. Frontend reads course builder status.
2. If needed, student starts generation/rebuild.
3. Hook watches progress events.
4. Course chapters, lessons, blocks, citations, and quizzes render in `CourseBuilderPanel`.

## Frontend Responsibility Boundary

The frontend should:

- Display backend and generator state clearly.
- Keep source selection valid.
- Stream responses live.
- Render artifacts faithfully.
- Provide optimistic feedback without replacing backend authority.
- Send runtime settings, language, and source-file selections with requests.

The frontend should not:

- Perform retrieval itself.
- Decide factual grounding.
- Mutate learner state without backend confirmation beyond temporary optimistic UI.
- Expose disabled generators as usable actions.
