"use client";

import { useState } from "react";

import { Download, FileText, Headphones } from "lucide-react";

import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import type { Artifact, PodcastArtifactMetadata } from "@/lib/types";

interface Props {
  artifact: Artifact;
  metadata?: PodcastArtifactMetadata;
}

export function PodcastPlayer({ artifact, metadata }: Props) {
  const [showTranscript, setShowTranscript] = useState(false);
  const transcript = metadata?.transcript?.trim();
  const duration = metadata?.duration_seconds;

  return (
    <div className="flex flex-col gap-3 rounded-lg border border-border bg-surface p-4">
      <header className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-2">
          <div className="flex h-10 w-10 items-center justify-center rounded-md bg-primary/15 text-primary">
            <Headphones className="h-5 w-5" />
          </div>
          <div>
            <h3 className="text-sm font-semibold">
              {artifact.filename ?? "Podcast"}
            </h3>
            {duration != null && (
              <p className="text-[11px] text-muted-foreground">
                {formatDuration(duration)}
              </p>
            )}
          </div>
        </div>

        <Button variant="secondary" size="sm" asChild>
          <a
            href={artifact.url}
            download={artifact.filename ?? "podcast.mp3"}
            aria-label="Download podcast"
          >
            <Download className="h-4 w-4" />
            Download
          </a>
        </Button>
      </header>

      <audio controls preload="metadata" className="w-full">
        <source src={artifact.url} />
        Your browser doesn't support the audio element.
      </audio>

      {transcript && (
        <div className="flex flex-col gap-2">
          <button
            type="button"
            onClick={() => setShowTranscript((v) => !v)}
            className="inline-flex w-fit items-center gap-1.5 text-xs font-medium text-muted-foreground hover:text-foreground"
          >
            <FileText className="h-3.5 w-3.5" />
            {showTranscript ? "Hide transcript" : "Show transcript"}
            <Badge variant="muted">{transcript.split(/\s+/).length} words</Badge>
          </button>
          {showTranscript && (
            <pre className="max-h-72 overflow-y-auto whitespace-pre-wrap rounded-md bg-muted p-3 text-xs leading-relaxed text-muted-foreground">
              {transcript}
            </pre>
          )}
        </div>
      )}
    </div>
  );
}

function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds <= 0) return "";
  const mins = Math.floor(seconds / 60);
  const secs = Math.round(seconds % 60)
    .toString()
    .padStart(2, "0");
  return `${mins}:${secs}`;
}
