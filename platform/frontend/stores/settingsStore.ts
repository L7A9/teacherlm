import { create } from "zustand";
import { persist } from "zustand/middleware";

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
  { value: "ollama", label: "Ollama" },
  { value: "openai", label: "OpenAI" },
  { value: "anthropic", label: "Anthropic" },
  { value: "openai_compatible", label: "OpenAI-compatible" },
] as const;

export const DEFAULT_OLLAMA_MODEL = "llama3.1:8b-instruct-q4_K_M";
export const DEFAULT_OLLAMA_BASE_URL = "http://host.docker.internal:11434";

export function defaultBaseUrlForProvider(provider: LlmProvider): string {
  if (provider === "ollama") return DEFAULT_OLLAMA_BASE_URL;
  if (provider === "anthropic") return "https://api.anthropic.com";
  return "https://api.openai.com/v1";
}

export function defaultModelForProvider(provider: LlmProvider): string {
  if (provider === "openai") return "gpt-4.1-mini";
  if (provider === "anthropic") return "claude-sonnet-4-5";
  return DEFAULT_OLLAMA_MODEL;
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

export function forcedLanguageToOptions(
  forcedLanguage: string | null,
): Record<string, unknown> {
  return forcedLanguage ? { language: forcedLanguage } : {};
}

interface SettingsState {
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
      version: 2,
      partialize: (state) => ({ forcedLanguage: state.forcedLanguage }),
      migrate: (persisted) => ({
        forcedLanguage:
          typeof persisted === "object" &&
          persisted !== null &&
          "forcedLanguage" in persisted
            ? (persisted as { forcedLanguage?: string | null }).forcedLanguage ?? null
            : null,
      }),
    },
  ),
);
