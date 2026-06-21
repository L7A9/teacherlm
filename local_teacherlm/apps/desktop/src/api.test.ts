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
