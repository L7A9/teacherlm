# Platform Frontend — CLAUDE.md

## Stack: Next.js 14 App Router + TailwindCSS + TanStack Query v5 + Zustand

## UX Vision
Three-panel NotebookLM-style layout:
- LEFT: Sources (uploaded files)
- CENTER: Chat with teacher + output buttons above input
- RIGHT: Learning progress + generated artifacts

## Critical UX Rules
- User NEVER sees "agent" or "generator" terminology
- Output types are shown as BUTTONS above chat input: 
  💬 Chat (default)  📝 Quiz  📄 Report  📊 Diagram  🗺️ Mind map  
  🎙️ Podcast  📑 Presentation
- Clicking a button opens an options dialog, then streams result into chat
- Typing in chat → always routes to Teacher (chat output)
- Chat shows learner progress subtly: "💡 You're strong on X, 
  let's review Y" (from learner_state)

## Module Map
frontend/
├── app/
│   ├── layout.tsx
│   ├── page.tsx                   # conversation list
│   └── c/[id]/page.tsx            # workspace
├── components/
│   ├── workspace/
│   │   ├── Workspace.tsx
│   │   ├── SourcesPanel.tsx
│   │   ├── ChatPanel.tsx
│   │   └── ProgressPanel.tsx      # right panel: artifacts + learner state
│   ├── chat/
│   │   ├── MessageList.tsx
│   │   ├── MessageBubble.tsx
│   │   ├── ChatInput.tsx
│   │   ├── OutputTypeButtons.tsx  # row of buttons above input
│   │   └── GeneratorDialog.tsx    # options modal per output type
│   ├── artifacts/
│   │   ├── ArtifactRenderer.tsx   # switch on type
│   │   ├── QuizRenderer.tsx
│   │   ├── ChartRenderer.tsx      # mermaid.js client-side
│   │   ├── PodcastPlayer.tsx
│   │   └── FileDownload.tsx       # generic for PDF/PPTX
│   ├── files/
│   │   ├── FileUploader.tsx
│   │   ├── FileList.tsx
│   │   └── FileStatusBadge.tsx
│   ├── progress/
│   │   └── LearnerProgress.tsx    # visual of concepts learned/struggling
│   └── ui/                         # primitives
├── lib/
│   ├── api.ts
│   ├── sse.ts
│   └── types.ts
├── stores/
│   ├── conversationStore.ts
│   ├── progressStore.ts
│   └── uiStore.ts
└── hooks/*.ts