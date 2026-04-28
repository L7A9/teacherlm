"use client";

import { useEffect, useRef, useState } from "react";

import { Maximize2, Minimize2 } from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/Button";
import { useChatStream } from "@/hooks/useChatStream";
import type { MindmapPayload, UUID } from "@/lib/types";
import { cn } from "@/lib/utils";

// Bright palette designed for dark backgrounds (Tailwind -400 variants).
// Each main branch and all its descendants share one color.
const BRANCH_COLORS = [
  "#60a5fa", // blue-400
  "#4ade80", // green-400
  "#f472b6", // pink-400
  "#fb923c", // orange-400
  "#a78bfa", // violet-400
  "#38bdf8", // sky-400
  "#34d399", // emerald-400
  "#facc15", // yellow-400
  "#e879f9", // fuchsia-400
  "#2dd4bf", // teal-400
];

// Safety-net: ensure markmap link paths are never filled.
// markmap injects its own <style> into the SVG, but inject this in <head> too.
const SAFETY_CSS = `.markmap-link{fill:none}`;
let safetyInjected = false;
function injectSafetyCss() {
  if (safetyInjected || typeof document === "undefined") return;
  safetyInjected = true;
  const el = document.createElement("style");
  el.textContent = SAFETY_CSS;
  document.head.appendChild(el);
}

// Walk the markmap tree and mark every node that has children as folded.
// Markmap's built-in click-on-circle toggles `payload.fold` per node, so
// the user gets progressive disclosure: click the root's circle to reveal
// its branches, click a branch's circle to reveal its subtopics, etc.
type MMNode = {
  payload?: { fold?: number; [k: string]: unknown };
  children?: MMNode[];
};
function collapseAllChildren(node: MMNode | undefined): void {
  if (!node?.children?.length) return;
  node.payload = { ...(node.payload ?? {}), fold: 1 };
  for (const child of node.children) {
    collapseAllChildren(child);
  }
}

interface Props {
  payload: MindmapPayload;
  conversationId?: UUID;
  className?: string;
}

interface MarkmapInstance {
  destroy?: () => void;
  fit?: () => Promise<unknown> | void;
}

export function MindmapRenderer({ payload, conversationId, className }: Props) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const mmRef = useRef<MarkmapInstance | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(false);
  const sendChat = useChatStream();

  useEffect(() => {
    injectSafetyCss();
  }, []);

  // Build / rebuild the markmap when the markdown changes.
  useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return;

    let cancelled = false;

    (async () => {
      try {
        const [{ Transformer }, { Markmap }] = await Promise.all([
          import("markmap-lib"),
          import("markmap-view"),
        ]);
        if (cancelled) return;

        const transformer = new Transformer();
        const { root } = transformer.transform(payload.markdown);
        collapseAllChildren(root as unknown as MMNode);

        try {
          mmRef.current?.destroy?.();
        } catch {
          // ignore
        }

        mmRef.current = Markmap.create(
          svg,
          {
            duration: 400,
            maxWidth: 260,
            spacingHorizontal: 80,
            spacingVertical: 6,
            paddingX: 12,
            autoFit: true,
            // Override markmap's built-in dark text (#333) for our dark background.
            // The style option is appended after the global CSS so variables override.
            style: () =>
              `.markmap {
                --markmap-text-color: #f1f5f9;
                --markmap-circle-open-bg: #1e293b;
                --markmap-font: 400 13px/1.5 system-ui, -apple-system, sans-serif;
              }`,
            // Color each branch (and all its descendants) consistently.
            // node.state.path is "rootId.branchId.subtopicId..." — use segment [1].
            color: (node: any) => {
              const parts = String(node?.state?.path ?? "").split(".");
              const branchIdx = parseInt(parts[1] ?? "") || 0;
              return (
                BRANCH_COLORS[Math.abs(branchIdx) % BRANCH_COLORS.length] ??
                BRANCH_COLORS[0]!
              );
            },
          },
          root,
        ) as MarkmapInstance;
      } catch (err) {
        if (!cancelled) setError((err as Error).message);
      }
    })();

    return () => {
      cancelled = true;
      try {
        mmRef.current?.destroy?.();
      } catch {
        // ignore
      }
      mmRef.current = null;
    };
  }, [payload.markdown]);

  // Refit after expand/collapse.
  useEffect(() => {
    const id = window.setTimeout(() => {
      try {
        void mmRef.current?.fit?.();
      } catch {
        // ignore
      }
    }, 80);
    return () => window.clearTimeout(id);
  }, [expanded]);

  // Click delegation: clicking a node label asks the teacher to explain it.
  useEffect(() => {
    const svg = svgRef.current;
    if (!svg || !conversationId) return;

    const onClick = (e: MouseEvent) => {
      const target = e.target as Element | null;

      // Markmap renders a small circle on every node with children; clicking
      // it toggles fold. Don't intercept those — let markmap handle expand
      // / collapse. Only label / foreignObject clicks ask the teacher.
      if (target?.closest("circle")) return;

      const node = target?.closest("g.markmap-node");
      if (!node) return;

      const text = (
        node.querySelector("foreignObject")?.textContent ??
        node.querySelector("text")?.textContent ??
        ""
      ).trim();
      if (!text) return;

      e.preventDefault();
      e.stopPropagation();

      toast(`Asking your teacher about "${text}"…`);
      void sendChat(conversationId, {
        user_message:
          `Explain "${text}" in detail, using the uploaded materials. ` +
          `Cite the relevant sources.`,
      });
    };

    svg.addEventListener("click", onClick, true);
    return () => svg.removeEventListener("click", onClick, true);
  }, [conversationId, sendChat]);

  if (error) {
    return (
      <div className="rounded-md border border-danger/40 bg-danger/10 p-3 text-xs text-[hsl(var(--danger))]">
        Couldn't render mind map: {error}
      </div>
    );
  }

  return (
    <div className={cn("flex flex-col gap-2", className)}>
      <div className="flex items-center justify-between gap-2">
        <span className="text-[11px] text-muted-foreground">
          {payload.central_topic ? `${payload.central_topic} · ` : ""}
          Click a node's circle to expand its children. Click a label to
          ask your teacher to explain it.
        </span>
        <Button
          variant="secondary"
          size="icon"
          aria-label={expanded ? "Collapse" : "Expand"}
          onClick={() => setExpanded((v) => !v)}
        >
          {expanded ? (
            <Minimize2 className="h-4 w-4" />
          ) : (
            <Maximize2 className="h-4 w-4" />
          )}
        </Button>
      </div>

      <div
        className={cn(
          "w-full overflow-hidden rounded-lg border border-border bg-[#0f172a]",
          expanded ? "h-[80vh]" : "h-[520px]",
        )}
      >
        <svg ref={svgRef} className="h-full w-full" />
      </div>
    </div>
  );
}
