"use client";

import { useCallback } from "react";

import { UploadCloud } from "lucide-react";
import { useDropzone, type FileRejection } from "react-dropzone";
import { toast } from "sonner";

import { useUploadFile } from "@/hooks/useFiles";
import type { UUID } from "@/lib/types";
import { cn } from "@/lib/utils";

interface Props {
  conversationId: UUID;
  className?: string;
}

const ACCEPTED = {
  "application/pdf": [".pdf"],
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [".docx"],
  "application/vnd.openxmlformats-officedocument.presentationml.presentation": [".pptx"],
  "text/plain": [".txt"],
  "text/markdown": [".md"],
  "text/html": [".html", ".htm"],
};

const MAX_SIZE_MB = 50;

export function FileUploader({ conversationId, className }: Props) {
  const upload = useUploadFile(conversationId);

  const onDrop = useCallback(
    (accepted: File[], rejected: FileRejection[]) => {
      for (const reject of rejected) {
        const first = reject.errors[0]?.message ?? "Rejected";
        toast.error(`${reject.file.name}: ${first}`);
      }
      for (const file of accepted) {
        upload.mutate(file, {
          onSuccess: () => toast.success(`Uploaded ${file.name}`),
          onError: (err) => toast.error(`Upload failed: ${err.message}`),
        });
      }
    },
    [upload],
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: ACCEPTED,
    maxSize: MAX_SIZE_MB * 1024 * 1024,
    disabled: upload.isPending,
  });

  return (
    <div
      {...getRootProps()}
      className={cn(
        "flex flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed px-4 py-6 text-center cursor-pointer transition-colors",
        "border-border bg-surface hover:border-primary/60 hover:bg-primary/5",
        isDragActive && "border-primary bg-primary/10",
        upload.isPending && "opacity-60 cursor-not-allowed",
        className,
      )}
    >
      <input {...getInputProps()} />
      <UploadCloud className="h-6 w-6 text-muted-foreground" />
      <div className="text-sm font-medium">
        {isDragActive ? "Drop to upload" : "Drag files here or click to browse"}
      </div>
      <div className="text-[11px] text-muted-foreground">
        PDF, DOCX, PPTX, TXT, MD, HTML · up to {MAX_SIZE_MB} MB
      </div>
    </div>
  );
}
