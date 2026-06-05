"use client";

import { useEffect, useRef, useState } from "react";

import { Maximize2, Minimize2 } from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/Button";
import { useChatStream } from "@/hooks/useChatStream";
import type { MindmapPayload, UUID } from "@/lib/types";
import { cn } from "@/lib/utils";
import {
  forcedLanguageToOptions,
  useSettingsStore,
} from "@/stores/settingsStore";

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
  "#f87171", // red-400
  "#c084fc", // purple-400
];
const ROOT_COLOR = "#cbd5e1";

// Safety-net: ensure markmap link paths are never filled.
// markmap injects its own <style> into the SVG, but inject this in <head> too.
const SAFETY_CSS = `
  .markmap-link{fill:none!important}
  .markmap-node{color:var(--branch-color,#cbd5e1)!important}
  .markmap-node>line,
  .markmap-node>circle{stroke:currentColor!important}
  .markmap-foreign,
  .markmap-foreign *{
    color:inherit!important;
    border-color:currentColor!important;
    text-decoration-color:currentColor!important;
  }
`;
let safetyInjected = false;
function injectSafetyCss() {
  if (safetyInjected || typeof document === "undefined") return;
  safetyInjected = true;
  const el = document.createElement("style");
  el.textContent = SAFETY_CSS;
  document.head.appendChild(el);
}

type MMNode = {
  payload?: { fold?: number; [k: string]: unknown };
  children?: MMNode[];
};

function collapseInitialTree(node: MMNode | undefined, depth = 0): void {
  if (!node?.children?.length) return;
  node.payload = { ...(node.payload ?? {}) };
  if (depth === 0) {
    delete node.payload.fold;
  } else {
    node.payload.fold = 1;
  }
  for (const child of node.children) {
    collapseInitialTree(child, depth + 1);
  }
}

function nodeExplanationPrompt(label: string, language: string | null): string {
  switch (language) {
    case "fr-fr":
      return `Explique "${label}" en detail avec les supports importes. Cite les sources pertinentes.`;
    case "es":
      return `Explica "${label}" en detalle usando los materiales subidos. Cita las fuentes relevantes.`;
    case "it":
      return `Spiega "${label}" in dettaglio usando i materiali caricati. Cita le fonti pertinenti.`;
    case "pt-br":
      return `Explique "${label}" em detalhe usando os materiais enviados. Cite as fontes relevantes.`;
    case "de":
      return `Erklaere "${label}" im Detail anhand der hochgeladenen Materialien. Zitiere die relevanten Quellen.`;
    case "ja":
      return `アップロードされた資料を使って「${label}」を詳しく説明してください。関連する出典も示してください。`;
    case "cmn":
      return `请根据上传的资料详细解释“${label}”，并引用相关来源。`;
    case "hi":
      return `अपलोड की गई सामग्री का उपयोग करके "${label}" को विस्तार से समझाइए। संबंधित स्रोत भी बताइए।`;
    default:
      return (
        `Explain "${label}" in detail, using the uploaded materials. ` +
        `Cite the relevant sources.`
      );
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

function fitMindmap(mm: MarkmapInstance | null): void {
  try {
    void mm?.fit?.();
  } catch {
    // ignore
  }
}

function applyVisibleBranchColors(svg: SVGSVGElement): void {
  const branchNodes = Array.from(
    svg.querySelectorAll<SVGGElement>('g.markmap-node[data-depth="2"]'),
  );
  const colorByBranchPath = new Map<string, string>();
  branchNodes.forEach((node, index) => {
    const path = node.getAttribute("data-path");
    if (path) {
      colorByBranchPath.set(path, BRANCH_COLORS[index % BRANCH_COLORS.length]!);
    }
  });

  const resolveColor = (path: string | null, depth: string | null): string => {
    if (!path || depth === "1") return ROOT_COLOR;
    const branchPath = path.split(".").slice(0, 2).join(".");
    return (
      colorByBranchPath.get(branchPath) ??
      BRANCH_COLORS[fallbackColorIndex(branchPath) % BRANCH_COLORS.length]!
    );
  };

  for (const node of Array.from(svg.querySelectorAll<SVGGElement>("g.markmap-node"))) {
    const color = resolveColor(
      node.getAttribute("data-path"),
      node.getAttribute("data-depth"),
    );
    node.style.setProperty("--branch-color", color);
    node.style.setProperty("color", color, "important");
    for (const labelPart of Array.from(
      node.querySelectorAll<HTMLElement>("foreignObject, foreignObject *"),
    )) {
      labelPart.style.setProperty("color", color, "important");
      labelPart.style.setProperty("border-color", color, "important");
      labelPart.style.setProperty("text-decoration-color", color, "important");
    }
    node.querySelector<SVGTextElement>("text")?.style.setProperty(
      "fill",
      color,
      "important",
    );
    for (const svgPart of Array.from(
      node.querySelectorAll<SVGElement>("circle,path,line,polyline"),
    )) {
      svgPart.setAttribute("stroke", color);
      svgPart.style.setProperty("stroke", color, "important");
      if (svgPart.tagName.toLowerCase() === "circle") {
        svgPart.setAttribute("fill", color);
        svgPart.style.setProperty("fill", color, "important");
      }
    }
  }

  for (const link of Array.from(svg.querySelectorAll<SVGPathElement>("path.markmap-link"))) {
    const color = resolveColor(
      link.getAttribute("data-path"),
      link.getAttribute("data-depth"),
    );
    link.setAttribute("stroke", color);
    link.setAttribute("fill", "none");
    link.style.setProperty("stroke", color, "important");
    link.style.setProperty("fill", "none", "important");
  }
}

function refreshMindmapView(
  mm: MarkmapInstance | null,
  svg: SVGSVGElement | null,
  delay = 80,
): void {
  if (!svg) return;
  window.setTimeout(() => {
    fitMindmap(mm);
    applyVisibleBranchColors(svg);
    window.setTimeout(() => applyVisibleBranchColors(svg), 420);
  }, delay);
}

function fallbackColorIndex(value: string): number {
  let hash = 0;
  for (let index = 0; index < value.length; index += 1) {
    hash = (hash * 31 + value.charCodeAt(index)) >>> 0;
  }
  return hash;
}

function branchColor(node: any): string {
  let branch = node;
  while (branch?.parent?.parent) {
    branch = branch.parent;
  }
  if (!branch?.parent) return ROOT_COLOR;

  const siblings = Array.isArray(branch.parent.children) ? branch.parent.children : [];
  const siblingIndex = siblings.indexOf(branch);
  const colorIndex =
    siblingIndex >= 0
      ? siblingIndex
      : fallbackColorIndex(String(branch?.state?.path ?? branch?.content ?? ""));
  return BRANCH_COLORS[colorIndex % BRANCH_COLORS.length] ?? BRANCH_COLORS[0]!;
}

export function MindmapRenderer({ payload, conversationId, className }: Props) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const mmRef = useRef<MarkmapInstance | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(false);
  const sendChat = useChatStream();
  const forcedLanguage = useSettingsStore((s) => s.forcedLanguage);

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
        collapseInitialTree(root as unknown as MMNode);

        try {
          mmRef.current?.destroy?.();
        } catch {
          // ignore
        }

        mmRef.current = Markmap.create(
          svg,
          {
            duration: 400,
            maxWidth: 300,
            spacingHorizontal: 110,
            spacingVertical: 12,
            paddingX: 16,
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
            color: branchColor,
          },
          root,
        ) as MarkmapInstance;
        refreshMindmapView(mmRef.current, svg);
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
      refreshMindmapView(mmRef.current, svgRef.current, 0);
    }, 80);
    return () => window.clearTimeout(id);
  }, [expanded]);

  // Click delegation: clicking a node label asks the teacher to explain it.
  useEffect(() => {
    const svg = svgRef.current;
    if (!svg || !conversationId) return;

    const onClick = (e: MouseEvent) => {
      const target = e.target as Element | null;

      // Markmap renders a small circle on every node with children. Clicking
      // the circle keeps native expand/collapse; clicking a node label with
      // children forwards to that same toggle.
      if (target?.closest("circle")) {
        refreshMindmapView(mmRef.current, svg, 450);
        return;
      }

      const node = target?.closest("g.markmap-node");
      if (!node) return;

      const text = (
        node.querySelector("foreignObject")?.textContent ??
        node.querySelector("text")?.textContent ??
        ""
      ).trim();
      if (!text) return;

      const toggle = node.querySelector("circle");
      if (toggle) {
        e.preventDefault();
        e.stopPropagation();
        toggle.dispatchEvent(
          new MouseEvent("click", {
            bubbles: true,
            cancelable: true,
            view: window,
          }),
        );
        refreshMindmapView(mmRef.current, svg, 450);
        return;
      }

      e.preventDefault();
      e.stopPropagation();

      toast(`Asking your teacher about "${text}"…`);
      const options = forcedLanguageToOptions(forcedLanguage);
      void sendChat(conversationId, {
        user_message: nodeExplanationPrompt(text, forcedLanguage),
        options,
      });
    };

    svg.addEventListener("click", onClick, true);
    return () => svg.removeEventListener("click", onClick, true);
  }, [conversationId, forcedLanguage, sendChat]);

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
        <span className="truncate text-xs font-medium text-muted-foreground">
          {payload.central_topic || "Mind map"}
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
          expanded ? "h-[80vh]" : "h-[640px] md:h-[700px]",
        )}
      >
        <svg ref={svgRef} className="h-full w-full" />
      </div>
    </div>
  );
}
