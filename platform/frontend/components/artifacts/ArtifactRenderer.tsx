"use client";

import { useQuery } from "@tanstack/react-query";

import { ChartRenderer } from "@/components/artifacts/ChartRenderer";
import { FileDownload } from "@/components/artifacts/FileDownload";
import { MindmapRenderer } from "@/components/artifacts/MindmapRenderer";
import { PodcastPlayer } from "@/components/artifacts/PodcastPlayer";
import { QuizRenderer } from "@/components/artifacts/QuizRenderer";
import type {
  Artifact,
  ChartArtifactMetadata,
  MindmapPayload,
  PodcastArtifactMetadata,
  QuizPayload,
  UUID,
} from "@/lib/types";

interface Props {
  artifact: Artifact;
  siblings?: Artifact[];
  conversationId?: UUID;
}

export function ArtifactRenderer({
  artifact,
  siblings = [],
  conversationId,
}: Props) {
  const kind = normalizeKind(artifact);

  switch (kind) {
    case "quiz":
      return (
        <JsonBoundary<QuizPayload> url={artifact.url}>
          {(payload) => <QuizRenderer payload={payload} />}
        </JsonBoundary>
      );

    case "chart":
      return (
        <JsonBoundary<ChartArtifactMetadata> url={artifact.url}>
          {(metadata) => <ChartRenderer metadata={metadata} />}
        </JsonBoundary>
      );

    case "mindmap":
      return (
        <JsonBoundary<MindmapPayload> url={artifact.url}>
          {(payload) => (
            <MindmapRenderer
              payload={payload}
              conversationId={conversationId}
            />
          )}
        </JsonBoundary>
      );

    case "podcast": {
      const transcript = findSibling(siblings, "transcript");
      return (
        <PodcastMaybeWithTranscript
          artifact={artifact}
          transcriptUrl={transcript?.url}
        />
      );
    }

    case "pdf":
    case "pptx":
    case "file":
      return <FileDownload artifact={artifact} />;

    case "transcript":
      // consumed by PodcastPlayer above — don't render standalone
      return null;

    default:
      return <FileDownload artifact={artifact} />;
  }
}

// ---------- helpers ----------

type Kind =
  | "quiz"
  | "chart"
  | "mindmap"
  | "podcast"
  | "transcript"
  | "pdf"
  | "pptx"
  | "file";

function normalizeKind(a: Artifact): Kind {
  const t = a.type.toLowerCase();
  const name = (a.filename ?? "").toLowerCase();
  // Structured JSON payloads use exact-match types so non-JSON sibling
  // exports fall through to FileDownload instead of trying to parse them.
  if (t === "quiz") return "quiz";
  if (t === "mindmap") return "mindmap";
  if (t === "chart" || t === "diagram" || t === "mermaid") return "chart";
  if (t === "podcast" || t === "audio") return "podcast";
  if (t === "transcript") return "transcript";
  if (t === "pdf" || name.endsWith(".pdf")) return "pdf";
  if (t === "pptx" || t === "presentation" || name.endsWith(".pptx")) {
    return "pptx";
  }
  return "file";
}

function findSibling(siblings: Artifact[], kind: Kind): Artifact | undefined {
  return siblings.find((s) => normalizeKind(s) === kind);
}

async function fetchJson(url: string): Promise<unknown> {
  const response = await fetch(url, { credentials: "omit" });
  if (!response.ok) {
    throw new Error(`Artifact fetch failed (${response.status})`);
  }
  return response.json();
}

function JsonBoundary<T>({
  url,
  children,
}: {
  url: string;
  children: (payload: T) => React.ReactNode;
}) {
  const { data, isLoading, error } = useQuery({
    queryKey: ["artifact-json", url],
    queryFn: () => fetchJson(url),
    staleTime: Infinity,
  });

  if (isLoading) {
    return (
      <div className="text-xs text-muted-foreground">Loading artifact…</div>
    );
  }
  if (error) {
    return (
      <div className="text-xs text-[hsl(var(--danger))]">
        Couldn't load artifact: {(error as Error).message}
      </div>
    );
  }
  if (data === undefined) return null;
  return <>{children(data as T)}</>;
}

function PodcastMaybeWithTranscript({
  artifact,
  transcriptUrl,
}: {
  artifact: Artifact;
  transcriptUrl?: string;
}) {
  const { data } = useQuery({
    queryKey: ["podcast-transcript", transcriptUrl],
    queryFn: async (): Promise<PodcastArtifactMetadata> => {
      if (!transcriptUrl) return {};
      const res = await fetch(transcriptUrl, { credentials: "omit" });
      if (!res.ok) return {};
      const ctype = res.headers.get("content-type") ?? "";
      if (ctype.includes("application/json")) {
        return (await res.json()) as PodcastArtifactMetadata;
      }
      return { transcript: await res.text() };
    },
    enabled: Boolean(transcriptUrl),
    staleTime: Infinity,
  });

  return <PodcastPlayer artifact={artifact} metadata={data} />;
}
