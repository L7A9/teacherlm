"use client";

import { useEffect, useId, useRef, useState } from "react";

import { Minus, Plus, RotateCcw } from "lucide-react";

import { Button } from "@/components/ui/Button";
import type { ChartArtifactMetadata } from "@/lib/types";
import { cn } from "@/lib/utils";

type PanZoomInstance = {
  zoomIn: () => void;
  zoomOut: () => void;
  resetZoom: () => void;
  resetPan: () => void;
  destroy: () => void;
};

interface Props {
  metadata: ChartArtifactMetadata;
  className?: string;
}

export function ChartRenderer({ metadata, className }: Props) {
  const uid = useId().replace(/:/g, "-");
  const containerRef = useRef<HTMLDivElement | null>(null);
  const panZoomRef = useRef<PanZoomInstance | null>(null);
  const [svg, setSvg] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const code = metadata.mermaid_code?.trim();

  useEffect(() => {
    if (!code) {
      setSvg(null);
      setError(null);
      return;
    }

    let cancelled = false;
    setError(null);

    (async () => {
      try {
        const mermaid = (await import("mermaid")).default;
        mermaid.initialize({
          startOnLoad: false,
          theme: "dark",
          securityLevel: "strict",
          fontFamily: "var(--font-sans)",
        });
        const { svg: rendered } = await mermaid.render(
          `mermaid-${uid}`,
          code,
        );
        if (!cancelled) setSvg(rendered);
      } catch (err) {
        if (!cancelled) setError((err as Error).message);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [code, uid]);

  // Wire svg-pan-zoom once the SVG is in the DOM.
  useEffect(() => {
    if (!svg || !containerRef.current) return;
    const svgEl = containerRef.current.querySelector<SVGSVGElement>("svg");
    if (!svgEl) return;

    svgEl.setAttribute("width", "100%");
    svgEl.setAttribute("height", "100%");
    svgEl.removeAttribute("style");

    let disposed = false;

    (async () => {
      const panZoomModule = await import("svg-pan-zoom");
      if (disposed) return;
      const svgPanZoom =
        (panZoomModule as unknown as { default?: typeof panZoomModule })
          .default ?? panZoomModule;
      panZoomRef.current = (svgPanZoom as unknown as (
        el: SVGElement,
        opts?: Record<string, unknown>,
      ) => PanZoomInstance)(svgEl, {
        zoomEnabled: true,
        controlIconsEnabled: false,
        fit: true,
        center: true,
        minZoom: 0.4,
        maxZoom: 6,
      });
    })();

    return () => {
      disposed = true;
      panZoomRef.current?.destroy();
      panZoomRef.current = null;
    };
  }, [svg]);

  if (!code) {
    return (
      <div className="text-xs text-muted-foreground">
        No diagram code was provided.
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col gap-2 rounded-md border border-danger/40 bg-danger/10 p-3 text-xs">
        <div className="font-medium text-[hsl(var(--danger))]">
          Couldn't render diagram: {error}
        </div>
        <pre className="overflow-x-auto whitespace-pre-wrap text-[11px] text-muted-foreground">
          {code}
        </pre>
      </div>
    );
  }

  return (
    <div className={cn("flex flex-col gap-2", className)}>
      <div className="flex items-center justify-end gap-1.5">
        <Button
          variant="secondary"
          size="icon"
          aria-label="Zoom out"
          onClick={() => panZoomRef.current?.zoomOut()}
        >
          <Minus className="h-4 w-4" />
        </Button>
        <Button
          variant="secondary"
          size="icon"
          aria-label="Zoom in"
          onClick={() => panZoomRef.current?.zoomIn()}
        >
          <Plus className="h-4 w-4" />
        </Button>
        <Button
          variant="secondary"
          size="icon"
          aria-label="Reset view"
          onClick={() => {
            panZoomRef.current?.resetZoom();
            panZoomRef.current?.resetPan();
          }}
        >
          <RotateCcw className="h-4 w-4" />
        </Button>
      </div>

      <div
        ref={containerRef}
        className="h-[420px] w-full overflow-hidden rounded-lg border border-border bg-background"
        dangerouslySetInnerHTML={svg ? { __html: svg } : undefined}
      >
        {svg ? undefined : (
          <div className="flex h-full items-center justify-center text-xs text-muted-foreground">
            Rendering diagram…
          </div>
        )}
      </div>
    </div>
  );
}
