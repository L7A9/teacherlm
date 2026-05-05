"use client";

import { useState } from "react";

import { Eye, FileText, Mic2, Network } from "lucide-react";

import { ArtifactRenderer } from "@/components/artifacts/ArtifactRenderer";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/Dialog";
import type { Artifact, UUID } from "@/lib/types";
import { cn } from "@/lib/utils";

interface Props {
  outputType: string;
  artifacts: Artifact[];
  conversationId: UUID;
  createdAt: string;
}

interface Meta {
  Icon: React.ComponentType<{ className?: string }>;
  label: string;
}

const META: Record<string, Meta> = {
  quiz: { Icon: FileText, label: "Quiz" },
  podcast: { Icon: Mic2, label: "Podcast" },
  mindmap: { Icon: Network, label: "Mind map" },
};

const FALLBACK: Meta = { Icon: Eye, label: "Generated item" };

export function ArtifactModalButton({
  outputType,
  artifacts,
  conversationId,
  createdAt,
}: Props) {
  const [open, setOpen] = useState(false);
  const { Icon, label } = META[outputType] ?? FALLBACK;

  const date = new Date(createdAt).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        className={cn(
          "flex w-full items-center gap-3 rounded-lg border border-border bg-surface p-3 text-left",
          "transition-colors hover:border-primary/40 hover:bg-primary/5",
          "focus:outline-none focus-visible:ring-2 focus-visible:ring-ring",
        )}
      >
        <span className="inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-md bg-primary/15 text-primary">
          <Icon className="h-4 w-4" />
        </span>
        <span className="flex flex-1 flex-col gap-0.5 min-w-0">
          <span className="text-sm font-medium">{label}</span>
          <span className="truncate text-[11px] text-muted-foreground">
            Generated {date} · click to open
          </span>
        </span>
      </button>

      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent
          className={cn(
            "w-[min(96vw,72rem)] max-w-none max-h-[90vh] overflow-y-auto",
          )}
        >
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Icon className="h-4 w-4 text-primary" />
              {label}
            </DialogTitle>
          </DialogHeader>

          <div className="flex flex-col gap-3">
            {artifacts.map((a, idx) => (
              <ArtifactRenderer
                key={`${a.url}-${idx}`}
                artifact={a}
                siblings={artifacts.filter((_, i) => i !== idx)}
                conversationId={conversationId}
              />
            ))}
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}
