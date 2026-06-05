"use client";

import { useState } from "react";

import { FileText, Loader2, RotateCcw, Trash2 } from "lucide-react";
import { toast } from "sonner";

import { FileStatusBadge } from "@/components/files/FileStatusBadge";
import { Button } from "@/components/ui/Button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/Dialog";
import { useDeleteFile, useFiles, useRetryFile } from "@/hooks/useFiles";
import { isUsableSourceFile, useSourceFileSelection } from "@/hooks/useSourceFileSelection";
import type { UploadedFile, UUID } from "@/lib/types";
import { cn, formatRelativeTime } from "@/lib/utils";

interface Props {
  conversationId: UUID;
}

export function FileList({ conversationId }: Props) {
  const { data, isLoading, error } = useFiles(conversationId);
  const remove = useDeleteFile(conversationId);
  const retry = useRetryFile(conversationId);
  const [fileToDelete, setFileToDelete] = useState<UploadedFile | null>(null);
  const items = data?.items ?? [];
  const {
    activeCount,
    readyFiles,
    selectedSet,
    showFileCheckboxes,
    toggleFile,
  } = useSourceFileSelection(conversationId, items);

  const handleDelete = async () => {
    if (!fileToDelete) return;
    try {
      await remove.mutateAsync(fileToDelete.id);
      toast.success(`Removed ${fileToDelete.filename}`);
      setFileToDelete(null);
    } catch (err) {
      toast.error(`Delete failed: ${(err as Error).message}`);
    }
  };

  if (isLoading) {
    return <div className="text-xs text-muted-foreground">Loading files...</div>;
  }
  if (error) {
    return <div className="text-xs text-danger">Failed to load files.</div>;
  }
  if (items.length === 0) {
    return (
      <div className="text-xs text-muted-foreground">
        No files yet. Upload a document to start teaching.
      </div>
    );
  }

  return (
    <>
      {showFileCheckboxes && (
        <div className="mb-2 px-1 text-[11px] text-muted-foreground">
          Using {activeCount}/{readyFiles.length} ready files
        </div>
      )}
      <ul className="flex flex-col gap-1.5">
        {items.map((file) => {
          const selectable = showFileCheckboxes && isUsableSourceFile(file);
          const checked = selectedSet.has(file.file_id);
          const locked = selectable && checked && activeCount <= 1;
          return (
            <li
              key={file.id}
              className={cn(
                "group flex items-center gap-2 rounded-md border border-border bg-surface px-2.5 py-2",
                selectable && checked && "border-primary/35 bg-primary/5",
              )}
            >
              {selectable ? (
                <input
                  type="checkbox"
                  checked={checked}
                  disabled={locked}
                  onChange={() => toggleFile(file.file_id)}
                  className="h-4 w-4 shrink-0 rounded border-border accent-primary"
                  aria-label={`Use ${file.filename}`}
                  title={locked ? "At least one source file must stay active" : `Use ${file.filename}`}
                />
              ) : (
                <FileText className="h-4 w-4 shrink-0 text-muted-foreground" />
              )}
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
                    <span>- {file.chunk_count} chunks</span>
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
                className="opacity-0 transition-opacity group-hover:opacity-100 focus-visible:opacity-100"
                aria-label={`Delete ${file.filename}`}
                title="Delete file"
                onClick={() => setFileToDelete(file)}
              >
                <Trash2 className="h-4 w-4" />
              </Button>
            </li>
          );
        })}
      </ul>

      <Dialog
        open={Boolean(fileToDelete)}
        onOpenChange={(open) => {
          if (!open && !remove.isPending) {
            setFileToDelete(null);
          }
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete file?</DialogTitle>
            <DialogDescription>
              {fileToDelete ? (
                <>
                  "{fileToDelete.filename}" will be removed from this conversation.
                  This can't be undone.
                </>
              ) : (
                "This file will be removed from this conversation."
              )}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button
              variant="secondary"
              onClick={() => setFileToDelete(null)}
              disabled={remove.isPending}
            >
              Cancel
            </Button>
            <Button
              variant="danger"
              onClick={handleDelete}
              disabled={remove.isPending || !fileToDelete}
            >
              {remove.isPending ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Trash2 className="h-4 w-4" />
              )}
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
