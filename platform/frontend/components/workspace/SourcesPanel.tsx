"use client";

import { FolderOpen, X } from "lucide-react";

import { FileList } from "@/components/files/FileList";
import { FileUploader } from "@/components/files/FileUploader";
import { Button } from "@/components/ui/Button";
import type { UUID } from "@/lib/types";
import { cn } from "@/lib/utils";

interface Props {
  conversationId: UUID;
  className?: string;
  onClose?: () => void;
}

export function SourcesPanel({ conversationId, className, onClose }: Props) {
  return (
    <aside
      className={cn(
        "app-pane flex h-full min-h-0 flex-col overflow-hidden border-r border-border",
        className,
      )}
      aria-label="Sources"
    >
      <header className="app-chrome flex h-11 items-center justify-between gap-2 border-b border-border px-4">
        <div className="flex min-w-0 items-center gap-2">
          <FolderOpen className="h-4 w-4 text-primary" />
          <h2 className="truncate text-sm font-semibold">Sources</h2>
        </div>
        {onClose && (
          <Button
            type="button"
            variant="ghost"
            size="icon"
            className="h-8 w-8 lg:hidden"
            onClick={onClose}
            aria-label="Close sources"
            title="Close"
          >
            <X className="h-4 w-4" />
          </Button>
        )}
      </header>

      <div className="px-4 py-3 app-chrome">
        <FileUploader conversationId={conversationId} />
      </div>

      <div className="flex-1 overflow-y-auto px-2 pb-4">
        <FileList conversationId={conversationId} />
      </div>
    </aside>
  );
}
