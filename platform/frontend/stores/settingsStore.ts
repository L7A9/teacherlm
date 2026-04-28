import { create } from "zustand";
import { persist } from "zustand/middleware";

// User-facing labels for the language picker on the settings page. The
// backend's podcast_gen + frontend GeneratorDialog had this list inline;
// it now lives here so adding a language updates both surfaces at once.
export const LANGUAGE_OPTIONS: ReadonlyArray<{ value: string; label: string }> = [
  { value: "en-us", label: "English (US)" },
  { value: "en-gb", label: "English (UK)" },
  { value: "fr-fr", label: "Français" },
  { value: "es", label: "Español" },
  { value: "it", label: "Italiano" },
  { value: "pt-br", label: "Português (BR)" },
  { value: "de", label: "Deutsch" },
  { value: "ja", label: "日本語" },
  { value: "cmn", label: "中文 (普通话)" },
  { value: "hi", label: "हिन्दी" },
] as const;

export function languageLabel(value: string): string {
  return LANGUAGE_OPTIONS.find((o) => o.value === value)?.label ?? value;
}

interface SettingsState {
  // Forced language for every generator + chat reply. `null` means the
  // user hasn't picked one — generators behave as before.
  forcedLanguage: string | null;
  setForcedLanguage: (value: string | null) => void;
}

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set) => ({
      forcedLanguage: null,
      setForcedLanguage: (forcedLanguage) => set({ forcedLanguage }),
    }),
    {
      name: "teacherlm-settings",
      version: 1,
    },
  ),
);
