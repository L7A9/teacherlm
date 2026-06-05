import { create } from "zustand";

import type { OutputType, UUID } from "@/lib/types";

export type Theme = "dark" | "light";

interface UiState {
  theme: Theme;
  sourcesCollapsed: boolean;
  progressCollapsed: boolean;
  sourceFileSelectionByConversation: Record<UUID, string[]>;

  generatorDialog: {
    open: boolean;
    outputType: OutputType | null;
  };

  setTheme: (theme: Theme) => void;
  setSourcesCollapsed: (collapsed: boolean) => void;
  setProgressCollapsed: (collapsed: boolean) => void;
  toggleSources: () => void;
  toggleProgress: () => void;
  setSourceFileSelection: (conversationId: UUID, sourceFileIds: string[]) => void;
  openGeneratorDialog: (outputType: OutputType) => void;
  closeGeneratorDialog: () => void;
}

export const useUiStore = create<UiState>((set) => ({
  theme: "dark",
  sourcesCollapsed: false,
  progressCollapsed: false,
  sourceFileSelectionByConversation: {},
  generatorDialog: { open: false, outputType: null },

  setTheme: (theme) => {
    set({ theme });
    if (typeof document !== "undefined") {
      document.documentElement.classList.toggle("dark", theme === "dark");
    }
  },

  setSourcesCollapsed: (collapsed) =>
    set({ sourcesCollapsed: collapsed }),

  setProgressCollapsed: (collapsed) =>
    set({ progressCollapsed: collapsed }),

  toggleSources: () =>
    set((s) => ({ sourcesCollapsed: !s.sourcesCollapsed })),

  toggleProgress: () =>
    set((s) => ({ progressCollapsed: !s.progressCollapsed })),

  setSourceFileSelection: (conversationId, sourceFileIds) =>
    set((s) => ({
      sourceFileSelectionByConversation: {
        ...s.sourceFileSelectionByConversation,
        [conversationId]: sourceFileIds,
      },
    })),

  openGeneratorDialog: (outputType) =>
    set({ generatorDialog: { open: true, outputType } }),

  closeGeneratorDialog: () =>
    set({ generatorDialog: { open: false, outputType: null } }),
}));
