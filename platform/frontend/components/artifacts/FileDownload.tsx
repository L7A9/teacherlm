"use client";

import { useEffect, useState } from "react";

import {
  ChevronLeft,
  ChevronRight,
  Download,
  Eye,
  EyeOff,
  FileText,
  Presentation,
} from "lucide-react";

import { Button } from "@/components/ui/Button";
import type { Artifact } from "@/lib/types";
import { cn } from "@/lib/utils";

interface Props {
  artifact: Artifact;
  className?: string;
}

export function FileDownload({ artifact, className }: Props) {
  const kind = detectKind(artifact);
  const [preview, setPreview] = useState(false);

  return (
    <div
      className={cn(
        "flex flex-col gap-3 rounded-lg border border-border bg-surface p-4",
        className,
      )}
    >
      <header className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-3">
          <div
            className={cn(
              "flex h-10 w-10 items-center justify-center rounded-md",
              kind === "pdf"
                ? "bg-[hsl(var(--danger)/0.15)] text-[hsl(var(--danger))]"
                : kind === "pptx"
                  ? "bg-[hsl(var(--warning)/0.15)] text-[hsl(var(--warning))]"
                  : "bg-muted text-muted-foreground",
            )}
          >
            {kind === "pptx" ? (
              <Presentation className="h-5 w-5" />
            ) : (
              <FileText className="h-5 w-5" />
            )}
          </div>
          <div className="min-w-0">
            <div className="truncate text-sm font-semibold" title={artifact.filename ?? undefined}>
              {artifact.filename ?? artifact.type}
            </div>
            <div className="text-[11px] uppercase tracking-wide text-muted-foreground">
              {kind}
            </div>
          </div>
        </div>

        <div className="flex items-center gap-1.5">
          {kind === "pdf" && (
            <Button
              variant="secondary"
              size="sm"
              onClick={() => setPreview((v) => !v)}
            >
              {preview ? (
                <>
                  <EyeOff className="h-4 w-4" /> Hide preview
                </>
              ) : (
                <>
                  <Eye className="h-4 w-4" /> Preview
                </>
              )}
            </Button>
          )}
          <Button variant="primary" size="sm" asChild>
            <a href={artifact.url} download={artifact.filename ?? undefined}>
              <Download className="h-4 w-4" />
              Download
            </a>
          </Button>
        </div>
      </header>

      {kind === "pdf" && preview && <PdfPreview url={artifact.url} />}
    </div>
  );
}

function PdfPreview({ url }: { url: string }) {
  const [mod, setMod] = useState<typeof import("react-pdf") | null>(null);
  const [numPages, setNumPages] = useState(0);
  const [page, setPage] = useState(1);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const rp = await import("react-pdf");
        rp.pdfjs.GlobalWorkerOptions.workerSrc = "/pdf.worker.min.mjs";
        if (!cancelled) setMod(rp);
      } catch (err) {
        if (!cancelled) setError((err as Error).message);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  if (error) {
    return (
      <div className="text-xs text-[hsl(var(--danger))]">
        Couldn't load PDF viewer: {error}
      </div>
    );
  }
  if (!mod) {
    return (
      <div className="text-xs text-muted-foreground">Loading preview…</div>
    );
  }

  const { Document, Page } = mod;

  return (
    <div className="flex flex-col items-center gap-2">
      <div className="w-full overflow-auto rounded-md border border-border bg-background p-2">
        <Document
          file={url}
          onLoadSuccess={({ numPages: n }) => setNumPages(n)}
          onLoadError={(err) => setError(err.message)}
          loading={
            <div className="text-xs text-muted-foreground">Loading PDF…</div>
          }
        >
          <Page
            pageNumber={page}
            width={560}
            renderAnnotationLayer={false}
            renderTextLayer={false}
          />
        </Document>
      </div>

      {numPages > 1 && (
        <div className="flex items-center gap-2 text-xs">
          <Button
            variant="ghost"
            size="icon"
            aria-label="Previous page"
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            disabled={page <= 1}
          >
            <ChevronLeft className="h-4 w-4" />
          </Button>
          <span className="tabular-nums text-muted-foreground">
            Page {page} / {numPages}
          </span>
          <Button
            variant="ghost"
            size="icon"
            aria-label="Next page"
            onClick={() => setPage((p) => Math.min(numPages, p + 1))}
            disabled={page >= numPages}
          >
            <ChevronRight className="h-4 w-4" />
          </Button>
        </div>
      )}
    </div>
  );
}

function detectKind(artifact: Artifact): "pdf" | "pptx" | "file" {
  const name = (artifact.filename ?? "").toLowerCase();
  const type = artifact.type.toLowerCase();
  if (name.endsWith(".pdf") || type.includes("pdf")) return "pdf";
  if (
    name.endsWith(".pptx") ||
    type.includes("pptx") ||
    type.includes("presentation")
  )
    return "pptx";
  return "file";
}
