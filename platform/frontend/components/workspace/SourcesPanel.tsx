"use client";

import { FolderOpen } from "lucide-react";

import { FileList } from "@/components/files/FileList";
import { FileUploader } from "@/components/files/FileUploader";
import type { UUID } from "@/lib/types";
import { cn } from "@/lib/utils";

interface Props {
  conversationId: UUID;
  className?: string;
}

export function SourcesPanel({ conversationId, className }: Props) {
  return (
    <aside
      className={cn(
        "flex h-full flex-col border-r border-border bg-background",
        className,
      )}
      aria-label="Sources"
    >
      <header className="flex items-center gap-2 border-b border-border px-4 py-3">
        <FolderOpen className="h-4 w-4 text-primary" />
        <h2 className="text-sm font-semibold">Sources</h2>
      </header>

      <div className="px-4 py-3">
        <FileUploader conversationId={conversationId} />
      </div>

      <div className="flex-1 overflow-y-auto px-2 pb-4">
        <FileList conversationId={conversationId} />
      </div>
    </aside>
  );
}
