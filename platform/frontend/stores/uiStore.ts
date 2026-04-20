import { create } from "zustand";

import type { OutputType } from "@/lib/types";

export type Theme = "dark" | "light";

interface UiState {
  theme: Theme;
  sourcesCollapsed: boolean;
  progressCollapsed: boolean;

  generatorDialog: {
    open: boolean;
    outputType: OutputType | null;
  };

  setTheme: (theme: Theme) => void;
  toggleSources: () => void;
  toggleProgress: () => void;
  openGeneratorDialog: (outputType: OutputType) => void;
  closeGeneratorDialog: () => void;
}

export const useUiStore = create<UiState>((set) => ({
  theme: "dark",
  sourcesCollapsed: false,
  progressCollapsed: false,
  generatorDialog: { open: false, outputType: null },

  setTheme: (theme) => {
    set({ theme });
    if (typeof document !== "undefined") {
      document.documentElement.classList.toggle("dark", theme === "dark");
    }
  },

  toggleSources: () =>
    set((s) => ({ sourcesCollapsed: !s.sourcesCollapsed })),

  toggleProgress: () =>
    set((s) => ({ progressCollapsed: !s.progressCollapsed })),

  openGeneratorDialog: (outputType) =>
    set({ generatorDialog: { open: true, outputType } }),

  closeGeneratorDialog: () =>
    set({ generatorDialog: { open: false, outputType: null } }),
}));
