# TeacherLM Frontend

The frontend is a Next.js workspace UI for uploading course files, chatting with the teacher, launching generators, viewing generated courses, tracking learning progress, and rendering artifacts.

It talks only to the backend API. It does not call generator services directly.

## Runtime Role

Default port: `3000`

Main surfaces:

- source sidebar for file upload, status, retry, delete, and source-file selection,
- course pane for generated CourseBuilder lessons and quizzes,
- chat pane for teacher chat and output-type buttons,
- generated-items/progress sidebar for learner progress, review prompts, artifacts, and sources,
- settings page for forced language and runtime LLM/parser configuration.

## Technologies And Why

| Technology | Why |
| --- | --- |
| Next.js 14 | App Router, production standalone output, server/client split. |
| React 18 | Component UI and streaming state updates. |
| TypeScript | Mirrors backend Pydantic schemas in `lib/types.ts`. |
| React Query | Server-state fetching, caching, invalidation, artifact JSON loading. |
| Zustand | Local UI, streaming, progress, source selection, and persisted theme/language state. |
| Tailwind CSS | Local design system and responsive workspace layout. |
| Radix Dialog/Tooltip/Slot | Accessible dialog and primitive behavior. |
| lucide-react | Primary icon set. |
| MUI icons/material + Emotion | Installed UI/icon dependencies available to the app. |
| react-dropzone | File upload drag/drop. |
| sonner | Toast notifications. |
| react-markdown, remark-gfm, remark-math, rehype-katex, katex | Markdown, tables, math, and formatted teacher responses. |
| markmap-lib / markmap-view | Interactive mind map rendering. |
| mermaid | Diagram rendering for chart artifacts. |
| svg-pan-zoom | Pan/zoom controls for rendered diagrams. |
| react-pdf / pdfjs-dist | PDF viewing support; `public/pdf.worker.min.mjs` provides the worker. |
| recharts | Chart-capable dependency for visual data components. |
| class-variance-authority, clsx, tailwind-merge | Small UI styling helpers. |

## API Layer

`lib/api.ts` defines typed wrappers around backend routes:

- conversations,
- runtime settings,
- knowledge checks,
- review tests,
- course player,
- CourseBuilder,
- knowledge graph,
- messages,
- files,
- generators,
- health.

`NEXT_PUBLIC_API_BASE_URL` is inlined at build time. In the browser, `resolveApiBaseUrl()` rewrites a configured `localhost` host to the current page hostname when the app is opened from another device on the LAN.

## SSE Streaming

Chat and generation endpoints are POST requests. Native `EventSource` only supports GET, so the frontend uses `fetch()` and parses `text/event-stream` manually in `lib/sse.ts`.

`ssePost()`:

1. sends JSON to `/api/conversations/{id}/chat` or `/generate`,
2. reads the response body with `TextDecoderStream`,
3. buffers until blank-line SSE event boundaries,
4. parses `event:` and `data:` lines,
5. JSON-decodes data when possible,
6. yields typed `SseEvent` objects.

`hooks/useChatStream.ts` handles:

- optimistic user messages for chat,
- streaming assistant text,
- source merging,
- artifact accumulation,
- generator/output metadata,
- learner state updates on `done`,
- stream abort controllers,
- React Query cache invalidation after streams finish.

## State Stores

| Store | Purpose |
| --- | --- |
| `conversationStore.ts` | Active conversation, streaming assistant message, abort controller, stream errors, artifacts, sources. |
| `progressStore.ts` | Local mirror of learner state plus optimistic update logic. |
| `settingsStore.ts` | Persisted forced language and provider UI constants. |
| `uiStore.ts` | Theme, side-panel collapse state, source-file selection by conversation, generator dialog state. |

Theme and forced language are persisted in local storage. Source-file selection is conversation-local UI state.

## Workspace Layout

`components/workspace/Workspace.tsx` builds the main screen:

- top bar with source/generated panel toggles, editable conversation title, settings link, and mobile course/chat switch,
- left Sources panel,
- center course pane and chat pane,
- draggable desktop resize handle between course and chat,
- right generated/progress panel,
- mobile drawers for sources and generated items.

The desktop layout keeps course and chat visible side by side. Mobile switches the main area between course and chat while side panels become overlays.

## Source-File Selection

The Sources panel tracks ready files and stores selected `source_file_ids` in `uiStore`.

Generation dialogs and chat send those IDs to the backend. The backend enforces the filter during retrieval. CourseBuilder is the exception: it intentionally uses every ready file in the conversation.

## Generator UI

The frontend lists enabled generators from `/api/generators`.

Implemented output buttons:

- teacher chat (`text`) through the chat input,
- quiz,
- podcast,
- mind map.

Registered but disabled output types may still exist in local UI constants for future work:

- report,
- presentation,
- chart/diagram.

`GeneratorDialog.tsx` prepares generator options:

- quiz exposes question count, difficulty, and one quiz type (`multiple_choice` or `true_false`),
- mind map defaults to `llm_refine: true`, `max_nodes: 110`, and `size: "standard"`,
- podcast fields are currently hidden in the dialog, so UI-triggered podcasts use generator defaults plus selected sources and forced language,
- topic input is hidden for quiz, podcast, and mind map so they operate over selected course files broadly,
- forced language from Settings is merged into generator options unless a caller explicitly supplies `options.language`.

## Artifact Rendering

`components/artifacts/ArtifactRenderer.tsx` dispatches by artifact type:

- `quiz`: fetch JSON and render with `QuizRenderer`,
- `mindmap`: fetch JSON and render with `MindmapRenderer`,
- `audio` / `podcast`: render `PodcastPlayer` and attach transcript sibling when present,
- `chart` / `diagram` / `mermaid`: fetch metadata and render Mermaid with pan/zoom,
- `pdf`, `pptx`, and unknown files: render download controls,
- `transcript`: consumed as a podcast sibling and not shown alone.

Artifact JSON is loaded through React Query with infinite stale time because generated artifact URLs point to immutable content for the message.

## Quiz Rendering And Mastery Feedback

`QuizRenderer` supports:

- MCQ,
- true/false,
- fill-blank payloads if a generator returns them,
- local scoring fallback,
- backend quiz-attempt submission when a conversation ID is present,
- learner-state update through the backend response,
- retry/reset UI.

The current quiz generator advertises MCQ and true/false only.

## Mind Map Rendering

`MindmapRenderer`:

- dynamically imports Markmap libraries on the client,
- collapses nodes below the root initially,
- colors each main branch and descendants consistently,
- injects safety CSS so Markmap links stay unfilled and labels remain readable,
- offers expand/collapse sizing,
- refits after toggles,
- lets students click a leaf node to ask the teacher for a grounded explanation,
- sends that follow-up through the same chat stream with forced language options.

## Podcast Rendering

`PodcastPlayer`:

- renders an HTML audio element,
- offers download,
- shows duration when metadata is available,
- can show/hide transcript text loaded from the transcript artifact.

## Diagram Rendering

`ChartRenderer`:

- renders Mermaid code client-side,
- switches Mermaid theme with app theme,
- uses strict Mermaid security mode,
- wires svg-pan-zoom controls,
- shows raw Mermaid code if rendering fails.

The chart generator is currently disabled in the registry, but the renderer is present.

## Runtime Settings UI

The frontend settings page works with `/api/settings/runtime`.

Supported provider labels:

- Ollama,
- OpenAI,
- Anthropic,
- OpenAI-compatible.

Provider defaults are UI conveniences. The backend is authoritative and encrypts stored API keys.

## Local Development

From this directory:

```bash
npm install
npm run dev
```

Build and check:

```bash
npm run build
npm run lint
npm run typecheck
```

Docker builds use `next.config.mjs` standalone output and the multi-stage `Dockerfile`. `NEXT_PUBLIC_API_BASE_URL` must be provided as a build arg because Next.js inlines it during build.
