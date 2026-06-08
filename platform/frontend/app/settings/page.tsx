"use client";

import { useEffect, useState } from "react";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { toast } from "sonner";

import {
  ArrowLeft,
  Bot,
  GraduationCap,
  KeyRound,
  Languages,
  Moon,
  Palette,
  Save,
  Server,
  Sun,
  Trash2,
  UploadCloud,
} from "lucide-react";

import { Button } from "@/components/ui/Button";
import { Input, Label } from "@/components/ui/Input";
import { runtimeSettingsApi } from "@/lib/api";
import type { RuntimeSettingsResponse, RuntimeSettingsUpdate } from "@/lib/types";
import { cn } from "@/lib/utils";
import {
  LANGUAGE_OPTIONS,
  LLM_PROVIDER_OPTIONS,
  defaultBaseUrlForProvider,
  defaultModelForProvider,
  providerLabel,
  providerRequiresApiKey,
  type LlmProvider,
  useSettingsStore,
} from "@/stores/settingsStore";
import {
  THEME_OPTIONS,
  type Theme,
  useUiStore,
} from "@/stores/uiStore";

const AUTO_VALUE = "__auto__";
const SAVED_SECRET_MASK = "saved-key-configured";

export default function SettingsPage() {
  const qc = useQueryClient();
  const forcedLanguage = useSettingsStore((s) => s.forcedLanguage);
  const setForcedLanguage = useSettingsStore((s) => s.setForcedLanguage);
  const theme = useUiStore((s) => s.theme);
  const setTheme = useUiStore((s) => s.setTheme);

  const [hydrated, setHydrated] = useState(false);
  const [llmEnabled, setLlmEnabled] = useState(false);
  const [llmProvider, setLlmProvider] = useState<LlmProvider>("ollama");
  const [llmModel, setLlmModel] = useState("");
  const [llmApiLink, setLlmApiLink] = useState("");
  const [llmApiKey, setLlmApiKey] = useState("");
  const [parserApiKey, setParserApiKey] = useState("");

  useEffect(() => setHydrated(true), []);

  const settingsQuery = useQuery({
    queryKey: ["runtime-settings"],
    queryFn: runtimeSettingsApi.get,
  });

  useEffect(() => {
    const data = settingsQuery.data;
    if (!data) return;
    setLlmEnabled(data.llm.enabled);
    setLlmProvider(data.llm.provider);
    setLlmModel(data.llm.model || defaultModelForProvider(data.llm.provider));
    setLlmApiLink(data.llm.api_link || defaultBaseUrlForProvider(data.llm.provider));
    setLlmApiKey("");
    setParserApiKey("");
  }, [settingsQuery.data]);

  const updateSettings = useMutation({
    mutationFn: (body: RuntimeSettingsUpdate) => runtimeSettingsApi.update(body),
    onSuccess: (data) => {
      qc.setQueryData<RuntimeSettingsResponse>(["runtime-settings"], data);
      toast.success("Settings saved");
    },
    onError: (err) => {
      toast.error(`Settings failed: ${(err as Error).message}`);
    },
  });

  const loading = settingsQuery.isLoading || !hydrated;
  const saving = updateSettings.isPending;
  const llmKeySet = Boolean(settingsQuery.data?.llm.api_key_set);
  const parserKeySet = Boolean(settingsQuery.data?.parser.api_key_set);

  const saveLlm = () => {
    const llm: NonNullable<RuntimeSettingsUpdate["llm"]> = {
      enabled: llmEnabled,
      provider: llmProvider,
      model: llmModel.trim(),
      api_link: llmApiLink.trim(),
    };
    if (llmApiKey.trim()) llm.api_key = llmApiKey.trim();
    updateSettings.mutate(
      { llm },
      {
        onSuccess: () => setLlmApiKey(""),
      },
    );
  };

  const saveParserKey = () => {
    const key = parserApiKey.trim();
    if (!key) return;
    updateSettings.mutate(
      { parser: { api_key: key } },
      {
        onSuccess: () => setParserApiKey(""),
      },
    );
  };

  const clearLlmKey = () => {
    updateSettings.mutate({ llm: { api_key: null } });
  };

  const clearParserKey = () => {
    updateSettings.mutate({ parser: { api_key: null } });
  };

  return (
    <main className="min-h-dvh bg-background text-foreground">
      <header className="app-chrome app-pane sticky top-0 z-10 border-b border-border">
        <div className="mx-auto flex max-w-4xl items-center justify-between px-4 py-3 sm:px-6">
          <div className="flex items-center gap-3">
            <Link
              href="/"
              className="flex h-9 w-9 items-center justify-center rounded-md bg-primary/15 text-primary transition-colors hover:bg-primary/25 active:bg-primary/20"
              aria-label="Back to conversations"
              title="Back to conversations"
            >
              <ArrowLeft className="h-5 w-5" />
            </Link>
            <div>
              <h1 className="text-lg font-semibold">Settings</h1>
              <p className="hidden text-xs text-muted-foreground sm:block">
                Global runtime configuration.
              </p>
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
            <StatusPill active={hydrated}>
              {hydrated ? themeLabel(theme) : "Loading"}
            </StatusPill>
          </header>
          <div className="px-5 py-4">
            <div className="grid max-w-md grid-cols-2 gap-2 rounded-md border border-border bg-background p-1">
              {THEME_OPTIONS.map((option) => {
                const active = hydrated && theme === option.value;
                return (
                  <button
                    key={option.value}
                    type="button"
                    disabled={!hydrated}
                    onClick={() => setTheme(option.value)}
                    className={cn(
                      "app-chrome flex h-10 items-center justify-center gap-2 rounded-sm px-3 text-sm font-medium transition-colors",
                      "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                      active
                        ? "bg-primary text-primary-foreground shadow-sm"
                        : "text-muted-foreground hover:bg-muted hover:text-foreground",
                      !hydrated && "cursor-not-allowed opacity-50",
                    )}
                    aria-pressed={active}
                  >
                    <ThemeIcon theme={option.value} />
                    {option.label}
                  </button>
                );
              })}
            </div>
          </div>
        </section>

        <section className="rounded-md border border-border bg-surface">
          <header className="app-chrome flex items-center justify-between gap-3 border-b border-border px-5 py-3">
            <div className="flex items-center gap-2">
              <Bot className="h-4 w-4 text-primary" />
              <h2 className="text-sm font-semibold">Model provider</h2>
            </div>
            <StatusPill active={llmEnabled}>
              {llmEnabled ? providerLabel(llmProvider) : "Project defaults"}
            </StatusPill>
          </header>
          <div className="flex flex-col gap-5 px-5 py-4">
            <label className="flex items-center justify-between gap-4 rounded-md border border-border bg-background px-3 py-3">
              <span className="text-sm font-medium">Backend LLM profile</span>
              <input
                type="checkbox"
                checked={llmEnabled}
                disabled={loading || saving}
                onChange={(e) => setLlmEnabled(e.target.checked)}
                className="h-4 w-4 accent-primary"
                aria-label="Enable backend LLM profile"
              />
            </label>

            <div className="grid gap-4 md:grid-cols-2">
              <div className="flex flex-col gap-1.5">
                <Label htmlFor="llm-provider">Provider</Label>
                <select
                  id="llm-provider"
                  value={llmProvider}
                  disabled={loading || saving || !llmEnabled}
                  onChange={(e) => {
                    const provider = e.target.value as LlmProvider;
                    setLlmProvider(provider);
                    setLlmModel(defaultModelForProvider(provider));
                    setLlmApiLink(defaultBaseUrlForProvider(provider));
                  }}
                  className={cn(
                    "h-9 rounded-md border border-border bg-background px-3 text-sm",
                    "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                  )}
                >
                  {LLM_PROVIDER_OPTIONS.map((opt) => (
                    <option key={opt.value} value={opt.value}>
                      {opt.label}
                    </option>
                  ))}
                </select>
              </div>

              <div className="flex flex-col gap-1.5">
                <Label htmlFor="llm-model">Model</Label>
                <Input
                  id="llm-model"
                  value={llmModel}
                  disabled={loading || saving || !llmEnabled}
                  placeholder={defaultModelForProvider(llmProvider)}
                  onChange={(e) => setLlmModel(e.target.value)}
                />
              </div>
            </div>

            <div className="flex flex-col gap-1.5">
              <Label htmlFor="llm-api-link" className="flex items-center gap-1.5">
                <Server className="h-3.5 w-3.5" />
                API link
              </Label>
              <Input
                id="llm-api-link"
                value={llmApiLink}
                disabled={loading || saving || !llmEnabled}
                placeholder={defaultBaseUrlForProvider(llmProvider)}
                onChange={(e) => setLlmApiLink(e.target.value)}
              />
            </div>

            <SecretField
              id="llm-api-key"
              label="API key"
              icon={<KeyRound className="h-3.5 w-3.5" />}
              value={llmApiKey}
              keySet={llmKeySet}
              disabled={loading || saving || !llmEnabled}
              placeholder={providerRequiresApiKey(llmProvider) ? "Paste provider API key" : "Optional"}
              onChange={setLlmApiKey}
              onClear={clearLlmKey}
              canClear={llmKeySet}
            />

            <div className="flex flex-wrap items-center justify-end gap-2">
              <Button
                type="button"
                variant="primary"
                onClick={saveLlm}
                disabled={loading || saving}
              >
                <Save className="h-4 w-4" />
                Save model
              </Button>
            </div>
          </div>
        </section>

        <section className="rounded-md border border-border bg-surface">
          <header className="app-chrome flex items-center justify-between gap-3 border-b border-border px-5 py-3">
            <div className="flex items-center gap-2">
              <UploadCloud className="h-4 w-4 text-primary" />
              <h2 className="text-sm font-semibold">LlamaCloud parser</h2>
            </div>
            <StatusPill active={parserKeySet}>
              {parserKeySet ? "Key saved" : "No DB key"}
            </StatusPill>
          </header>
          <div className="flex flex-col gap-4 px-5 py-4">
            <SecretField
              id="parser-api-key"
              label="API key"
              icon={<KeyRound className="h-3.5 w-3.5" />}
              value={parserApiKey}
              keySet={parserKeySet}
              disabled={loading || saving}
              placeholder="Paste LlamaCloud API key"
              onChange={setParserApiKey}
              onClear={clearParserKey}
              canClear={parserKeySet}
            />
            <div className="flex flex-wrap items-center justify-end gap-2">
              <Button
                type="button"
                variant="primary"
                onClick={saveParserKey}
                disabled={loading || saving || !parserApiKey.trim()}
              >
                <Save className="h-4 w-4" />
                Save parser key
              </Button>
            </div>
          </div>
        </section>

        <section className="rounded-md border border-border bg-surface">
          <header className="app-chrome flex items-center gap-2 border-b border-border px-5 py-3">
            <Languages className="h-4 w-4 text-primary" />
            <h2 className="text-sm font-semibold">Default language</h2>
          </header>
          <div className="flex flex-col gap-4 px-5 py-4">
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="forced-language">Language</Label>
              <select
                id="forced-language"
                value={hydrated ? forcedLanguage ?? AUTO_VALUE : AUTO_VALUE}
                disabled={!hydrated}
                onChange={(e) =>
                  setForcedLanguage(
                    e.target.value === AUTO_VALUE ? null : e.target.value,
                  )
                }
                className={cn(
                  "h-9 w-full max-w-xs rounded-md border border-border bg-background px-3 text-sm",
                  "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                )}
              >
                <option value={AUTO_VALUE}>Auto</option>
                {LANGUAGE_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
            </div>

            {hydrated && forcedLanguage && (
              <div className="flex items-center justify-between rounded-md border border-primary/30 bg-primary/10 px-3 py-2 text-xs">
                <span>
                  Active:{" "}
                  <strong>
                    {LANGUAGE_OPTIONS.find((o) => o.value === forcedLanguage)
                      ?.label ?? forcedLanguage}
                  </strong>
                </span>
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={() => setForcedLanguage(null)}
                >
                  Reset
                </Button>
              </div>
            )}
          </div>
        </section>
      </div>
    </main>
  );
}

function ThemeIcon({ theme }: { theme: Theme }) {
  if (theme === "light") return <Sun className="h-4 w-4" />;
  return <Moon className="h-4 w-4" />;
}

function themeLabel(theme: Theme) {
  return THEME_OPTIONS.find((option) => option.value === theme)?.label ?? theme;
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
        active
          ? "border-primary/30 bg-primary/10 text-primary"
          : "border-border bg-muted text-muted-foreground",
      )}
    >
      {children}
    </span>
  );
}

function SecretField({
  id,
  label,
  icon,
  value,
  keySet,
  disabled,
  placeholder,
  canClear,
  onChange,
  onClear,
}: {
  id: string;
  label: string;
  icon: React.ReactNode;
  value: string;
  keySet: boolean;
  disabled: boolean;
  placeholder: string;
  canClear: boolean;
  onChange: (value: string) => void;
  onClear: () => void;
}) {
  const [editingSavedKey, setEditingSavedKey] = useState(false);
  const showingSavedMask = keySet && !value && !editingSavedKey;
  const displayValue = showingSavedMask ? SAVED_SECRET_MASK : value;

  useEffect(() => {
    if (!keySet || value) setEditingSavedKey(false);
  }, [keySet, value]);

  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center justify-between gap-3">
        <Label htmlFor={id} className="flex items-center gap-1.5">
          {icon}
          {label}
        </Label>
        <span className="text-[11px] text-muted-foreground">
          {keySet ? "Configured" : "Empty"}
        </span>
      </div>
      <div className="flex gap-2">
        <Input
          id={id}
          type="password"
          value={displayValue}
          disabled={disabled}
          placeholder={editingSavedKey ? "Enter replacement key" : placeholder}
          autoComplete="off"
          title={showingSavedMask ? "A key is saved. Focus to enter a replacement." : undefined}
          onFocus={() => {
            if (keySet && !value) setEditingSavedKey(true);
          }}
          onBlur={() => {
            if (keySet && !value) setEditingSavedKey(false);
          }}
          onChange={(e) => onChange(e.target.value)}
        />
        <Button
          type="button"
          variant="secondary"
          size="icon"
          onClick={onClear}
          disabled={disabled || !canClear}
          aria-label={`Clear ${label}`}
          title={`Clear ${label}`}
        >
          <Trash2 className="h-4 w-4" />
        </Button>
      </div>
    </div>
  );
}
