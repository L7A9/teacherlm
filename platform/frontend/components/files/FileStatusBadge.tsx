import { AlertCircle, CheckCircle2, Loader2, Upload } from "lucide-react";

import { Badge } from "@/components/ui/Badge";
import type { FileStatus } from "@/lib/types";

interface Props {
  status: FileStatus;
  error?: string | null;
}

const LABELS: Record<FileStatus, string> = {
  uploaded: "Queued",
  parsing: "Parsing…",
  chunking: "Chunking…",
  embedding: "Embedding…",
  ready: "Ready",
  failed: "Failed",
};

export function FileStatusBadge({ status, error }: Props) {
  if (status === "ready") {
    return (
      <Badge variant="success" title="Ready to reference">
        <CheckCircle2 className="h-3 w-3" />
        {LABELS[status]}
      </Badge>
    );
  }
  if (status === "failed") {
    return (
      <Badge variant="danger" title={error ?? undefined}>
        <AlertCircle className="h-3 w-3" />
        {LABELS[status]}
      </Badge>
    );
  }
  if (status === "uploaded") {
    return (
      <Badge variant="muted">
        <Upload className="h-3 w-3" />
        {LABELS[status]}
      </Badge>
    );
  }
  return (
    <Badge variant="primary">
      <Loader2 className="h-3 w-3 animate-spin" />
      {LABELS[status]}
    </Badge>
  );
}
