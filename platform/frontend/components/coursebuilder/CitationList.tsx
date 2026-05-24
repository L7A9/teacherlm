"use client";

import { FileText } from "lucide-react";

import type { CourseBuilderCitation } from "@/lib/types";

interface Props {
  citations: CourseBuilderCitation[];
}

export function CourseBuilderCitationList({ citations }: Props) {
  if (citations.length === 0) return null;
  return (
    <details className="mt-2 rounded-md border border-border bg-background/60 px-2 py-1">
      <summary className="cursor-pointer text-[11px] font-medium text-muted-foreground">
        Sources
      </summary>
      <ul className="mt-2 flex flex-col gap-2">
        {citations.map((citation) => (
          <li key={citation.chunk_id} className="text-[11px] leading-4 text-muted-foreground">
            <div className="flex items-center gap-1 font-medium text-foreground">
              <FileText className="h-3 w-3" />
              <span className="truncate">{citation.source || "Uploaded file"}</span>
            </div>
            {(citation.section || citation.page_start) && (
              <div className="mt-0.5">
                {citation.section}
                {citation.page_start ? ` p.${citation.page_start}` : ""}
              </div>
            )}
            {citation.snippet && (
              <p className="mt-1 line-clamp-3">{citation.snippet}</p>
            )}
          </li>
        ))}
      </ul>
    </details>
  );
}
