"use client";

import { useEffect, useState } from "react";

import Link from "next/link";

import {
  ArrowLeft,
  Bot,
  GraduationCap,
  KeyRound,
  Languages,
  RotateCcw,
  Server,
} from "lucide-react";

import { Button } from "@/components/ui/Button";
import { Input, Label } from "@/components/ui/Input";
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

const AUTO_VALUE = "__auto__";

export default function SettingsPage() {
  const forcedLanguage = useSettingsStore((s) => s.forcedLanguage);
  const setForcedLanguage = useSettingsStore((s) => s.setForcedLanguage);
  const modelSettings = useSettingsStore((s) => s.modelSettings);
  const setModelSettings = useSettingsStore((s) => s.setModelSettings);
  const resetModelSettings = useSettingsStore((s) => s.resetModelSettings);

  // Avoid SSR / persist hydration mismatch — render the bound value only
  // after the persisted store has rehydrated on the client.
  const [hydrated, setHydrated] = useState(false);
  useEffect(() => setHydrated(true), []);

  return (
    <main className="min-h-dvh bg-background text-foreground">
      <header className="border-b border-border">
        <div className="mx-auto flex max-w-3xl items-center justify-between px-6 py-5">
          <div className="flex items-center gap-3">
            <Link
              href="/"
              className="flex h-9 w-9 items-center justify-center rounded-md bg-primary/15 text-primary transition-colors hover:bg-primary/25"
              aria-label="Back to conversations"
              title="Back to conversations"
            >
              <ArrowLeft className="h-5 w-5" />
            </Link>
            <div>
              <h1 className="text-lg font-semibold">Settings</h1>
              <p className="text-xs text-muted-foreground">
                Preferences applied to every conversation on this device.
              </p>
            </div>
          </div>
          <div className="flex h-9 w-9 items-center justify-center rounded-md bg-muted text-muted-foreground">
            <GraduationCap className="h-5 w-5" />
          </div>
        </div>
      </header>

      <div className="mx-auto flex max-w-3xl flex-col gap-6 px-6 py-8">
        <section className="rounded-lg border border-border bg-surface">
          <header className="flex items-center gap-2 border-b border-border px-5 py-4">
            <Bot className="h-4 w-4 text-primary" />
            <h2 className="text-sm font-semibold">Model provider</h2>
          </header>
          <div className="flex flex-col gap-5 px-5 py-4">
            <div className="flex items-start justify-between gap-4 rounded-md border border-border bg-background px-3 py-3">
              <div>
                <p className="text-sm font-medium">Use custom model settings</p>
                <p className="mt-1 text-xs text-muted-foreground">
                  When off, TeacherLM uses the models configured in Docker/env
                  exactly like before.
                </p>
              </div>
              <input
                type="checkbox"
                checked={hydrated ? modelSettings.enabled : false}
                disabled={!hydrated}
                onChange={(e) =>
                  setModelSettings({ enabled: e.target.checked })
                }
                className="mt-1 h-4 w-4 accent-primary"
                aria-label="Use custom model settings"
              />
            </div>

            <div className="grid gap-4 md:grid-cols-2">
              <div className="flex flex-col gap-1.5">
                <Label htmlFor="llm-provider">Provider</Label>
                <select
                  id="llm-provider"
                  value={hydrated ? modelSettings.provider : "ollama"}
                  disabled={!hydrated || !modelSettings.enabled}
                  onChange={(e) => {
                    const provider = e.target.value as LlmProvider;
                    setModelSettings({
                      provider,
                      baseUrl: defaultBaseUrlForProvider(provider),
                      model: defaultModelForProvider(provider),
                      apiKey:
                        provider === "ollama" ? "" : modelSettings.apiKey,
                    });
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
                <Label htmlFor="llm-model">Model name</Label>
                <Input
                  id="llm-model"
                  value={hydrated ? modelSettings.model : ""}
                  disabled={!hydrated || !modelSettings.enabled}
                  placeholder={defaultModelForProvider(modelSettings.provider)}
                  onChange={(e) => setModelSettings({ model: e.target.value })}
                />
              </div>
            </div>

            <div className="flex flex-col gap-1.5">
              <Label htmlFor="llm-base-url" className="flex items-center gap-1.5">
                <Server className="h-3.5 w-3.5" />
                API link
              </Label>
              <Input
                id="llm-base-url"
                value={hydrated ? modelSettings.baseUrl : ""}
                disabled={!hydrated || !modelSettings.enabled}
                placeholder={
                  modelSettings.provider === "ollama"
                    ? "http://host.docker.internal:11434"
                    : defaultBaseUrlForProvider(modelSettings.provider)
                }
                onChange={(e) =>
                  setModelSettings({ baseUrl: e.target.value })
                }
              />
              <p className="text-[11px] text-muted-foreground">
                Ollama uses your local or Ollama-cloud access. OpenAI and
                OpenAI-compatible providers use <code>/chat/completions</code>.
                Anthropic uses its native <code>/v1/messages</code> API.
              </p>
            </div>

            {providerRequiresApiKey(modelSettings.provider) && (
              <div className="flex flex-col gap-1.5">
                <Label htmlFor="llm-api-key" className="flex items-center gap-1.5">
                  <KeyRound className="h-3.5 w-3.5" />
                  API key
                </Label>
                <Input
                  id="llm-api-key"
                  type="password"
                  value={hydrated ? modelSettings.apiKey : ""}
                  disabled={!hydrated || !modelSettings.enabled}
                  placeholder="sk-..."
                  autoComplete="off"
                  onChange={(e) =>
                    setModelSettings({ apiKey: e.target.value })
                  }
                />
                <p className="text-[11px] text-muted-foreground">
                  Use an API key for this provider. ChatGPT Plus and Claude
                  Pro app subscriptions are not API keys; their separate API
                  accounts are required.
                </p>
              </div>
            )}

            <div className="flex items-center justify-between rounded-md border border-border bg-muted px-3 py-2 text-xs text-muted-foreground">
              <span>
                Active:{" "}
                <strong className="text-foreground">
                  {hydrated && modelSettings.enabled
                    ? `${providerLabel(modelSettings.provider)} / ${modelSettings.model || "no model"}`
                    : "Project defaults"}
                </strong>
              </span>
              <Button
                type="button"
                variant="ghost"
                size="sm"
                onClick={resetModelSettings}
                disabled={!hydrated}
              >
                <RotateCcw className="h-3.5 w-3.5" />
                Reset
              </Button>
            </div>
          </div>
        </section>

        <section className="rounded-lg border border-border bg-surface">
          <header className="flex items-center gap-2 border-b border-border px-5 py-4">
            <Languages className="h-4 w-4 text-primary" />
            <h2 className="text-sm font-semibold">Default language</h2>
          </header>
          <div className="flex flex-col gap-3 px-5 py-4">
            <p className="text-sm text-muted-foreground">
              When set, every generator (podcast, quiz, mind map…) and
              the teacher's chat replies use this language — no need to
              pick it each time. Pick <em>Auto</em> to let each generator
              decide based on the source files.
            </p>

            <div className="flex flex-col gap-1.5">
              <label
                htmlFor="forced-language"
                className="text-xs font-medium text-muted-foreground"
              >
                Language
              </label>
              <select
                id="forced-language"
                value={
                  hydrated
                    ? forcedLanguage ?? AUTO_VALUE
                    : AUTO_VALUE
                }
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
                <option value={AUTO_VALUE}>
                  Auto (let each generator decide)
                </option>
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
                  Currently forcing every generator to use{" "}
                  <strong>
                    {LANGUAGE_OPTIONS.find((o) => o.value === forcedLanguage)
                      ?.label ?? forcedLanguage}
                  </strong>
                  .
                </span>
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={() => setForcedLanguage(null)}
                >
                  Reset to Auto
                </Button>
              </div>
            )}
          </div>
        </section>
      </div>
    </main>
  );
}
