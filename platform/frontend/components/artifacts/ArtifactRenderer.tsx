"use client";

import { useQuery } from "@tanstack/react-query";

import { ChartRenderer } from "@/components/artifacts/ChartRenderer";
import { FileDownload } from "@/components/artifacts/FileDownload";
import { FlashcardRenderer } from "@/components/artifacts/FlashcardRenderer";
import { PodcastPlayer } from "@/components/artifacts/PodcastPlayer";
import { QuizRenderer } from "@/components/artifacts/QuizRenderer";
import type {
  Artifact,
  ChartArtifactMetadata,
  FlashcardPayload,
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

    case "flashcards":
      return (
        <JsonBoundary<FlashcardPayload> url={artifact.url}>
          {(payload) => (
            <FlashcardRenderer payload={payload} conversationId={conversationId} />
          )}
        </JsonBoundary>
      );

    case "chart":
      return (
        <JsonBoundary<ChartArtifactMetadata> url={artifact.url}>
          {(metadata) => <ChartRenderer metadata={metadata} />}
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
  | "flashcards"
  | "chart"
  | "podcast"
  | "transcript"
  | "pdf"
  | "pptx"
  | "file";

function normalizeKind(a: Artifact): Kind {
  const t = a.type.toLowerCase();
  if (t.includes("quiz")) return "quiz";
  if (t.includes("flashcard")) return "flashcards";
  if (t.includes("chart") || t.includes("diagram") || t.includes("mermaid")) {
    return "chart";
  }
  if (t.includes("podcast") || t.includes("audio")) return "podcast";
  if (t.includes("transcript")) return "transcript";
  const name = (a.filename ?? "").toLowerCase();
  if (t.includes("pdf") || name.endsWith(".pdf")) return "pdf";
  if (
    t.includes("pptx") ||
    t.includes("presentation") ||
    name.endsWith(".pptx")
  ) {
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
