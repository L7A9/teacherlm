import { afterEach, describe, expect, it, vi } from "vitest";

import { api, streamChat } from "./api";

const originalFetch = globalThis.fetch;

afterEach(() => {
  globalThis.fetch = originalFetch;
  vi.restoreAllMocks();
});

describe("streamChat cancellation", () => {
  it("aborts the active streaming request through its AbortSignal", async () => {
    globalThis.fetch = vi.fn((_input: RequestInfo | URL, init?: RequestInit) => (
      new Promise<Response>((_resolve, reject) => {
        init?.signal?.addEventListener("abort", () => {
          reject(new DOMException("The operation was aborted", "AbortError"));
        }, { once: true });
      })
    ));
    const controller = new AbortController();
    const pending = streamChat("conversation-1", "Explain matrices", [], () => undefined, controller.signal);

    controller.abort();

    await expect(pending).rejects.toMatchObject({ name: "AbortError" });
  });
});

describe("CourseBuilder profiles", () => {
  it("requests the improved profile only for an explicit quality rebuild", async () => {
    const fetchMock = vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) => new Response(JSON.stringify({ status: "building", chapters: [] }), {
      status: 202,
      headers: { "Content-Type": "application/json" },
    }));
    globalThis.fetch = fetchMock;

    await api.rebuildCoursebuilder("conversation-1", true);

    expect(String(fetchMock.mock.calls[0]?.[0])).toContain("improved_quality=true");
  });
});

describe("generator settings", () => {
  it("persists transcript-only podcast mode through the settings endpoint", async () => {
    const fetchMock = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => new Response(JSON.stringify({ podcast_audio_enabled: false }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    }));
    globalThis.fetch = fetchMock;

    const updated = await api.updateGeneratorSettings({ podcast_audio_enabled: false });

    expect(updated.podcast_audio_enabled).toBe(false);
    expect(String(fetchMock.mock.calls[0]?.[0])).toContain("/settings/generators");
    expect(fetchMock.mock.calls[0]?.[1]).toMatchObject({
      method: "PATCH",
      body: JSON.stringify({ podcast_audio_enabled: false }),
    });
  });
});

describe("first-run setup", () => {
  it("starts model provisioning through the setup endpoint", async () => {
    const payload = {
      ready: false,
      running: true,
      progress: 0.2,
      message: "Downloading",
      error: null,
      active_component: "chat_model",
      components: [],
    };
    const fetchMock = vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) => new Response(JSON.stringify(payload), {
      status: 202,
      headers: { "Content-Type": "application/json" },
    }));
    globalThis.fetch = fetchMock;

    const result = await api.startSetup();

    expect(result.running).toBe(true);
    expect(String(fetchMock.mock.calls[0]?.[0])).toContain("/setup");
    expect(fetchMock.mock.calls[0]?.[1]).toMatchObject({ method: "POST" });
  });
});
