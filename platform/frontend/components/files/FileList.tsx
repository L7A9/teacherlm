"use client";

import { FileText, RotateCcw, Trash2 } from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/Button";
import { FileStatusBadge } from "@/components/files/FileStatusBadge";
import { useDeleteFile, useFiles, useRetryFile } from "@/hooks/useFiles";
import type { UUID } from "@/lib/types";
import { formatRelativeTime } from "@/lib/utils";

interface Props {
  conversationId: UUID;
}

export function FileList({ conversationId }: Props) {
  const { data, isLoading, error } = useFiles(conversationId);
  const remove = useDeleteFile(conversationId);
  const retry = useRetryFile(conversationId);

  if (isLoading) {
    return <div className="text-xs text-muted-foreground">Loading files…</div>;
  }
  if (error) {
    return <div className="text-xs text-danger">Failed to load files.</div>;
  }
  const items = data?.items ?? [];
  if (items.length === 0) {
    return (
      <div className="text-xs text-muted-foreground">
        No files yet. Upload a document to start teaching.
      </div>
    );
  }

  return (
    <ul className="flex flex-col gap-1.5">
      {items.map((file) => (
        <li
          key={file.id}
          className="group flex items-center gap-2 rounded-md border border-border bg-surface px-2.5 py-2"
        >
          <FileText className="h-4 w-4 shrink-0 text-muted-foreground" />
          <div className="min-w-0 flex-1">
            <div
              className="truncate text-sm font-medium"
              title={file.filename}
            >
              {file.filename}
            </div>
            <div className="flex items-center gap-2 text-[11px] text-muted-foreground">
              <span>{formatRelativeTime(file.created_at)}</span>
              {file.status === "ready" && file.chunk_count > 0 && (
                <span>· {file.chunk_count} chunks</span>
              )}
            </div>
          </div>
          <FileStatusBadge status={file.status} error={file.error} />
          {file.status === "failed" && (
            <Button
              variant="ghost"
              size="icon"
              className="transition-opacity"
              aria-label={`Retry ${file.filename}`}
              title="Retry from the beginning"
              disabled={retry.isPending}
              onClick={() =>
                retry.mutate(file.id, {
                  onSuccess: () => toast.success(`Retrying ${file.filename}`),
                  onError: (err) => toast.error(`Retry failed: ${err.message}`),
                })
              }
            >
              <RotateCcw className="h-4 w-4" />
            </Button>
          )}
          <Button
            variant="ghost"
            size="icon"
            className="opacity-0 transition-opacity group-hover:opacity-100"
            aria-label={`Delete ${file.filename}`}
            onClick={() =>
              remove.mutate(file.id, {
                onSuccess: () => toast.success(`Removed ${file.filename}`),
                onError: (err) => toast.error(`Delete failed: ${err.message}`),
              })
            }
          >
            <Trash2 className="h-4 w-4" />
          </Button>
        </li>
      ))}
    </ul>
  );
}
