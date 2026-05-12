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

export type LlmProvider =
  | "ollama"
  | "openai"
  | "anthropic"
  | "openai_compatible";

export const LLM_PROVIDER_OPTIONS: ReadonlyArray<{
  value: LlmProvider;
  label: string;
}> = [
  { value: "ollama", label: "Ollama (local / Ollama cloud)" },
  { value: "openai", label: "OpenAI API" },
  { value: "anthropic", label: "Anthropic Claude API" },
  { value: "openai_compatible", label: "OpenAI-compatible provider" },
] as const;

export interface ModelSettings {
  enabled: boolean;
  provider: LlmProvider;
  model: string;
  baseUrl: string;
  apiKey: string;
}

export const DEFAULT_MODEL_SETTINGS: ModelSettings = {
  enabled: false,
  provider: "ollama",
  model: "llama3.1:8b-instruct-q4_K_M",
  baseUrl: "http://host.docker.internal:11434",
  apiKey: "",
};

export function defaultBaseUrlForProvider(provider: LlmProvider): string {
  if (provider === "ollama") return DEFAULT_MODEL_SETTINGS.baseUrl;
  if (provider === "anthropic") return "https://api.anthropic.com";
  return "https://api.openai.com/v1";
}

export function defaultModelForProvider(provider: LlmProvider): string {
  if (provider === "openai") return "gpt-4.1-mini";
  if (provider === "anthropic") return "claude-sonnet-4-5";
  return DEFAULT_MODEL_SETTINGS.model;
}

export function providerLabel(provider: LlmProvider): string {
  return (
    LLM_PROVIDER_OPTIONS.find((option) => option.value === provider)?.label ??
    provider
  );
}

export function providerRequiresApiKey(provider: LlmProvider): boolean {
  return provider !== "ollama";
}

export function modelSettingsToOptions(
  modelSettings: ModelSettings,
): Record<string, unknown> {
  if (!modelSettings.enabled || !modelSettings.model.trim()) return {};
  return {
    llm: {
      enabled: true,
      provider: modelSettings.provider,
      model: modelSettings.model.trim(),
      base_url:
        modelSettings.baseUrl.trim() ||
        defaultBaseUrlForProvider(modelSettings.provider),
      ...(providerRequiresApiKey(modelSettings.provider) &&
      modelSettings.apiKey.trim()
        ? { api_key: modelSettings.apiKey.trim() }
        : {}),
    },
  };
}

interface SettingsState {
  // Forced language for every generator + chat reply. `null` means the
  // user hasn't picked one — generators behave as before.
  forcedLanguage: string | null;
  modelSettings: ModelSettings;
  setForcedLanguage: (value: string | null) => void;
  setModelSettings: (value: Partial<ModelSettings>) => void;
  resetModelSettings: () => void;
}

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set) => ({
      forcedLanguage: null,
      modelSettings: DEFAULT_MODEL_SETTINGS,
      setForcedLanguage: (forcedLanguage) => set({ forcedLanguage }),
      setModelSettings: (value) =>
        set((state) => ({
          modelSettings: { ...state.modelSettings, ...value },
        })),
      resetModelSettings: () => set({ modelSettings: DEFAULT_MODEL_SETTINGS }),
    }),
    {
      name: "teacherlm-settings",
      version: 1,
    },
  ),
);
