import { API_BASE_URL, ApiError } from "./api";
import type { SseEvent } from "./types";

// Our SSE endpoints are POST (chat + generate), so the native EventSource API
// (GET-only) can't reach them. Stream the response body with fetch and parse
// the text/event-stream framing by hand.

export interface SsePostOptions<B> {
  path: string;
  body: B;
  signal?: AbortSignal;
  headers?: Record<string, string>;
}

export async function* ssePost<B>({
  path,
  body,
  signal,
  headers,
}: SsePostOptions<B>): AsyncGenerator<SseEvent, void, void> {
  const url = path.startsWith("/") ? `${API_BASE_URL}${path}` : `${API_BASE_URL}/${path}`;

  const response = await fetch(url, {
    method: "POST",
    headers: {
      Accept: "text/event-stream",
      "Content-Type": "application/json",
      ...headers,
    },
    body: JSON.stringify(body),
    credentials: "include",
    signal,
  });

  if (!response.ok) {
    const text = await response.text().catch(() => "");
    throw new ApiError(response.status, text || response.statusText, text);
  }
  if (!response.body) {
    throw new ApiError(500, "response body is empty", null);
  }

  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();

  // SSE events are separated by blank lines. Buffer across reads until we see
  // `\n\n` (or `\r\n\r\n`), then emit one event per block.
  let buffer = "";
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += value;

      let sepIdx: number;
      // eslint-disable-next-line no-cond-assign
      while ((sepIdx = findEventBoundary(buffer)) !== -1) {
        const rawEvent = buffer.slice(0, sepIdx);
        buffer = buffer.slice(sepIdx + boundaryLength(buffer, sepIdx));
        const parsed = parseEventBlock(rawEvent);
        if (parsed) yield parsed;
      }
    }
    const tail = buffer.trim();
    if (tail) {
      const parsed = parseEventBlock(tail);
      if (parsed) yield parsed;
    }
  } finally {
    reader.releaseLock();
  }
}

function findEventBoundary(buf: string): number {
  const a = buf.indexOf("\n\n");
  const b = buf.indexOf("\r\n\r\n");
  if (a === -1) return b;
  if (b === -1) return a;
  return Math.min(a, b);
}

function boundaryLength(buf: string, idx: number): number {
  return buf.startsWith("\r\n\r\n", idx) ? 4 : 2;
}

function parseEventBlock(raw: string): SseEvent | null {
  if (!raw.trim()) return null;

  let eventName = "message";
  const dataLines: string[] = [];

  for (const line of raw.split(/\r?\n/)) {
    if (!line || line.startsWith(":")) continue;
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);

    if (field === "event") eventName = value;
    else if (field === "data") dataLines.push(value);
  }

  if (dataLines.length === 0) return null;
  const rawData = dataLines.join("\n");
  let data: unknown = rawData;
  try {
    data = JSON.parse(rawData);
  } catch {
    // leave as raw string
  }
  return { event: eventName, data };
}
