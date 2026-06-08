import { create } from "zustand";
import { persist } from "zustand/middleware";

import type { OutputType, UUID } from "@/lib/types";

export type Theme = "dark" | "light";

export const THEME_OPTIONS: ReadonlyArray<{ value: Theme; label: string }> = [
  { value: "dark", label: "Dark" },
  { value: "light", label: "Light" },
] as const;

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

export function applyTheme(theme: Theme) {
  if (typeof document === "undefined") return;
  document.documentElement.classList.toggle("dark", theme === "dark");
  document.documentElement.style.colorScheme = theme;
}

function safeTheme(value: unknown): Theme {
  return value === "light" ? "light" : "dark";
}

export const useUiStore = create<UiState>()(
  persist(
    (set) => ({
      theme: "dark",
      sourcesCollapsed: false,
      progressCollapsed: false,
      sourceFileSelectionByConversation: {},
      generatorDialog: { open: false, outputType: null },

      setTheme: (theme) => {
        const next = safeTheme(theme);
        set({ theme: next });
        applyTheme(next);
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
    }),
    {
      name: "teacherlm-ui",
      version: 1,
      partialize: (state) => ({ theme: state.theme }),
      migrate: (persisted) => ({
        theme:
          typeof persisted === "object" &&
          persisted !== null &&
          "theme" in persisted
            ? safeTheme((persisted as { theme?: unknown }).theme)
            : "dark",
      }),
      onRehydrateStorage: () => (state) => {
        applyTheme(safeTheme(state?.theme));
      },
    },
  ),
);
