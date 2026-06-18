import { useEffect, useRef, useState } from "react";

import { Download, Maximize2, Minimize2 } from "lucide-react";

import type { MindmapNode, MindmapPayload } from "../types";

const BRANCH_COLORS = [
  "#60a5fa",
  "#4ade80",
  "#f472b6",
  "#fb923c",
  "#a78bfa",
  "#38bdf8",
  "#34d399",
  "#facc15",
  "#e879f9",
  "#2dd4bf",
  "#f87171",
  "#c084fc",
];
const ROOT_COLOR = "#cbd5e1";

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

type MarkmapTreeNode = {
  payload?: { fold?: number; [key: string]: unknown };
  children?: MarkmapTreeNode[];
};

interface MarkmapInstance {
  destroy?: () => void;
  fit?: () => Promise<unknown> | void;
}

export function MindmapRenderer({ payload, className }: { payload: MindmapPayload; className?: string }) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const mmRef = useRef<MarkmapInstance | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [exportError, setExportError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(false);
  const markdown = mindmapMarkdown(payload);

  useEffect(() => {
    injectSafetyCss();
  }, []);

  useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return;

    let cancelled = false;

    (async () => {
      try {
        const [{ Transformer }, { Markmap }] = await Promise.all([import("markmap-lib"), import("markmap-view")]);
        if (cancelled) return;

        const transformer = new Transformer();
        const { root } = transformer.transform(markdown);
        collapseInitialTree(root as MarkmapTreeNode);

        try {
          mmRef.current?.destroy?.();
        } catch {
          // ignore stale renderer cleanup failures
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
            style: () =>
              `.markmap {
                --markmap-text-color: #f1f5f9;
                --markmap-circle-open-bg: #1e293b;
                --markmap-font: 400 13px/1.5 system-ui, -apple-system, sans-serif;
              }`,
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
        // ignore cleanup failures
      }
      mmRef.current = null;
    };
  }, [markdown]);

  useEffect(() => {
    const id = window.setTimeout(() => refreshMindmapView(mmRef.current, svgRef.current, 0), 80);
    return () => window.clearTimeout(id);
  }, [expanded]);

  useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return;

    const onClick = (event: MouseEvent) => {
      const target = event.target as Element | null;
      if (target?.closest("circle")) {
        refreshMindmapView(mmRef.current, svg, 450);
        return;
      }

      const node = target?.closest("g.markmap-node");
      if (!node) return;

      const toggle = node.querySelector("circle");
      if (!toggle) return;

      event.preventDefault();
      event.stopPropagation();
      toggle.dispatchEvent(
        new MouseEvent("click", {
          bubbles: true,
          cancelable: true,
          view: window,
        }),
      );
      refreshMindmapView(mmRef.current, svg, 450);
    };

    svg.addEventListener("click", onClick, true);
    return () => svg.removeEventListener("click", onClick, true);
  }, []);

  if (error) {
    return (
      <div className="rounded-md border border-danger/40 bg-danger/10 p-3 text-xs text-[hsl(var(--danger))]">
        Couldn't render mind map: {error}
      </div>
    );
  }

  const downloadMindmap = async (format: "svg" | "png") => {
    setExportError(null);
    refreshMindmapView(mmRef.current, svgRef.current, 0);
    await waitForFrame();

    let exportView: ExpandedMindmapExport | null = null;
    try {
      exportView = await renderExpandedMindmapForExport(markdown);
      const svg = exportView.svg;
      if (format === "svg") {
        downloadBlob(
          new Blob([serializeMindmapSvg(svg)], { type: "image/svg+xml;charset=utf-8" }),
          mindmapDownloadName(payload, "svg"),
        );
        return;
      }
      await downloadMindmapPng(svg, mindmapDownloadName(payload, "png"));
    } catch (err) {
      setExportError((err as Error).message || "Could not export the mind map.");
    } finally {
      exportView?.cleanup();
    }
  };

  return (
    <div className={["flex flex-col gap-2", className].filter(Boolean).join(" ")}>
      <div className="flex items-center justify-between gap-2">
        <span className="truncate text-xs font-medium text-muted-foreground">{mindmapTitle(payload)}</span>
        <div className="flex shrink-0 items-center gap-2">
          <button
            type="button"
            className="app-chrome inline-flex h-9 items-center justify-center gap-2 rounded-md border border-border bg-surface px-3 text-xs font-medium text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
            aria-label="Download mind map as PNG"
            title="Download PNG"
            onClick={() => void downloadMindmap("png")}
          >
            <Download className="h-4 w-4" />
            PNG
          </button>
          <button
            type="button"
            className="app-chrome inline-flex h-9 items-center justify-center gap-2 rounded-md border border-border bg-surface px-3 text-xs font-medium text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
            aria-label="Download mind map as SVG"
            title="Download SVG"
            onClick={() => void downloadMindmap("svg")}
          >
            <Download className="h-4 w-4" />
            SVG
          </button>
          <button
            type="button"
            className="app-chrome inline-flex h-9 w-9 items-center justify-center rounded-md border border-border bg-surface text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
            aria-label={expanded ? "Collapse mind map" : "Expand mind map"}
            title={expanded ? "Collapse" : "Expand"}
            onClick={() => setExpanded((value) => !value)}
          >
            {expanded ? <Minimize2 className="h-4 w-4" /> : <Maximize2 className="h-4 w-4" />}
          </button>
        </div>
      </div>
      {exportError && <p className="text-xs text-[hsl(var(--danger))]">{exportError}</p>}

      <div
        className={[
          "w-full overflow-hidden rounded-lg border border-border bg-[#0f172a]",
          expanded ? "h-[80vh]" : "h-[640px] md:h-[700px]",
        ].join(" ")}
      >
        <svg ref={svgRef} className="h-full w-full" />
      </div>
    </div>
  );
}

type ExpandedMindmapExport = {
  svg: SVGSVGElement;
  cleanup: () => void;
};

async function renderExpandedMindmapForExport(markdown: string): Promise<ExpandedMindmapExport> {
  const [{ Transformer }, { Markmap }] = await Promise.all([import("markmap-lib"), import("markmap-view")]);
  const transformer = new Transformer();
  const { root } = transformer.transform(markdown);
  const nodeCount = countMarkdownNodes(markdown);
  const width = Math.min(3200, Math.max(1800, 1200 + nodeCount * 18));
  const height = Math.min(4200, Math.max(1400, 900 + nodeCount * 32));
  const container = document.createElement("div");
  container.setAttribute("aria-hidden", "true");
  container.style.position = "fixed";
  container.style.left = "-10000px";
  container.style.top = "0";
  container.style.width = `${width}px`;
  container.style.height = `${height}px`;
  container.style.opacity = "0";
  container.style.pointerEvents = "none";
  container.style.overflow = "hidden";

  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("width", String(width));
  svg.setAttribute("height", String(height));
  svg.style.width = `${width}px`;
  svg.style.height = `${height}px`;
  container.appendChild(svg);
  document.body.appendChild(container);

  let mm: MarkmapInstance | null = null;
  try {
    mm = Markmap.create(
      svg,
      {
        duration: 0,
        maxWidth: 300,
        spacingHorizontal: 120,
        spacingVertical: 14,
        paddingX: 24,
        autoFit: true,
        style: () =>
          `.markmap {
            --markmap-text-color: #f1f5f9;
            --markmap-circle-open-bg: #1e293b;
            --markmap-font: 400 13px/1.5 system-ui, -apple-system, sans-serif;
          }`,
        color: branchColor,
      },
      root,
    ) as MarkmapInstance;
    await settleExportMindmap(mm, svg);
  } catch (err) {
    container.remove();
    try {
      mm?.destroy?.();
    } catch {
      // ignore failed export cleanup
    }
    throw err;
  }

  return {
    svg,
    cleanup: () => {
      try {
        mm?.destroy?.();
      } catch {
        // ignore failed export cleanup
      }
      container.remove();
    },
  };
}

async function settleExportMindmap(mm: MarkmapInstance | null, svg: SVGSVGElement): Promise<void> {
  await waitForFrame();
  fitMindmap(mm);
  applyVisibleBranchColors(svg);
  await waitForMs(120);
  fitMindmap(mm);
  applyVisibleBranchColors(svg);
}

function countMarkdownNodes(markdown: string): number {
  return markdown
    .split(/\r?\n/)
    .filter((line) => /^\s*(?:#{1,6}\s+|- )/.test(line))
    .length;
}

function serializeMindmapSvg(svg: SVGSVGElement, options: { nativeText?: boolean } = {}): string {
  const clone = svg.cloneNode(true) as SVGSVGElement;
  if (options.nativeText) {
    replaceForeignObjectsWithNativeText(svg, clone);
  }
  const bounds = svg.getBoundingClientRect();
  const width = Math.max(800, Math.ceil(bounds.width || svg.clientWidth || 1200));
  const height = Math.max(600, Math.ceil(bounds.height || svg.clientHeight || 800));
  const viewBox = clone.getAttribute("viewBox") || `0 0 ${width} ${height}`;
  const [x = 0, y = 0, viewWidth = width, viewHeight = height] = viewBox.split(/\s+/).map((value) => Number(value));

  clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  clone.setAttribute("version", "1.1");
  clone.setAttribute("width", String(width));
  clone.setAttribute("height", String(height));
  clone.setAttribute("viewBox", viewBox);

  const background = document.createElementNS("http://www.w3.org/2000/svg", "rect");
  background.setAttribute("x", String(Number.isFinite(x) ? x : 0));
  background.setAttribute("y", String(Number.isFinite(y) ? y : 0));
  background.setAttribute("width", String(Number.isFinite(viewWidth) ? viewWidth : width));
  background.setAttribute("height", String(Number.isFinite(viewHeight) ? viewHeight : height));
  background.setAttribute("fill", "#0f172a");
  clone.insertBefore(background, clone.firstChild);

  const style = document.createElementNS("http://www.w3.org/2000/svg", "style");
  style.textContent = `
    ${SAFETY_CSS}
    .markmap{--markmap-text-color:#f1f5f9;--markmap-circle-open-bg:#1e293b;font:400 13px/1.5 system-ui,-apple-system,sans-serif}
    .markmap-foreign{font:400 13px/1.5 system-ui,-apple-system,sans-serif}
  `;
  clone.insertBefore(style, clone.firstChild);

  return `<?xml version="1.0" encoding="UTF-8"?>\n${new XMLSerializer().serializeToString(clone)}`;
}

async function downloadMindmapPng(svg: SVGSVGElement, filename: string): Promise<void> {
  const serialized = serializeMindmapSvg(svg, { nativeText: true });
  const svgUrl = URL.createObjectURL(new Blob([serialized], { type: "image/svg+xml;charset=utf-8" }));
  try {
    const image = new Image();
    image.crossOrigin = "anonymous";
    image.decoding = "async";
    image.src = svgUrl;
    await new Promise<void>((resolve, reject) => {
      image.onload = () => resolve();
      image.onerror = () => reject(new Error("Could not render the mind map image."));
    });

    const scale = 2;
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, image.naturalWidth || image.width) * scale;
    canvas.height = Math.max(1, image.naturalHeight || image.height) * scale;
    const context = canvas.getContext("2d");
    if (!context) throw new Error("Could not prepare image export.");
    context.fillStyle = "#0f172a";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.drawImage(image, 0, 0, canvas.width, canvas.height);

    const blob = await new Promise<Blob>((resolve, reject) => {
      canvas.toBlob((value) => (value ? resolve(value) : reject(new Error("Could not create the PNG file."))), "image/png");
    });
    downloadBlob(blob, filename);
  } finally {
    URL.revokeObjectURL(svgUrl);
  }
}

function replaceForeignObjectsWithNativeText(sourceSvg: SVGSVGElement, cloneSvg: SVGSVGElement): void {
  const sourceObjects = Array.from(sourceSvg.querySelectorAll<SVGForeignObjectElement>("foreignObject"));
  const cloneObjects = Array.from(cloneSvg.querySelectorAll<SVGForeignObjectElement>("foreignObject"));

  cloneObjects.forEach((foreignObject, index) => {
    const sourceObject = sourceObjects[index];
    const text = normalizeExportText(sourceObject?.textContent || foreignObject.textContent || "");
    if (!text) {
      foreignObject.remove();
      return;
    }

    const textElement = document.createElementNS("http://www.w3.org/2000/svg", "text");
    const x = readSvgNumber(foreignObject.getAttribute("x"), 0);
    const y = readSvgNumber(foreignObject.getAttribute("y"), 0);
    const width = Math.max(80, readSvgNumber(foreignObject.getAttribute("width"), 260));
    const fontSize = 13;
    const color = exportTextColor(sourceObject);
    const lines = wrapExportText(text, width, fontSize);

    textElement.setAttribute("x", String(x));
    textElement.setAttribute("y", String(y + 2));
    textElement.setAttribute("fill", color);
    textElement.setAttribute("font-family", "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif");
    textElement.setAttribute("font-size", String(fontSize));
    textElement.setAttribute("font-weight", "400");
    textElement.setAttribute("dominant-baseline", "text-before-edge");

    lines.forEach((line, lineIndex) => {
      const tspan = document.createElementNS("http://www.w3.org/2000/svg", "tspan");
      tspan.setAttribute("x", String(x));
      tspan.setAttribute("dy", lineIndex === 0 ? "0" : "1.35em");
      tspan.textContent = line;
      textElement.appendChild(tspan);
    });

    foreignObject.replaceWith(textElement);
  });
}

function normalizeExportText(value: string): string {
  return value.replace(/\s+/g, " ").trim();
}

function readSvgNumber(value: string | null, fallback: number): number {
  if (!value) return fallback;
  const parsed = Number.parseFloat(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function exportTextColor(sourceObject: SVGForeignObjectElement | undefined): string {
  if (!sourceObject) return "#f1f5f9";
  const node = sourceObject.closest("g.markmap-node") as SVGElement | null;
  const color = node ? window.getComputedStyle(node).color : window.getComputedStyle(sourceObject).color;
  return color && color !== "rgba(0, 0, 0, 0)" ? color : "#f1f5f9";
}

function wrapExportText(text: string, width: number, fontSize: number): string[] {
  const maxChars = Math.max(10, Math.floor(width / (fontSize * 0.58)));
  const words = text.split(/\s+/).filter(Boolean);
  const lines: string[] = [];
  let current = "";

  for (const word of words) {
    const candidate = current ? `${current} ${word}` : word;
    if (candidate.length <= maxChars) {
      current = candidate;
      continue;
    }
    if (current) lines.push(current);
    current = word;
  }
  if (current) lines.push(current);

  return lines.length > 0 ? lines.slice(0, 5) : [text.slice(0, maxChars)];
}

function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function waitForFrame(): Promise<void> {
  return new Promise((resolve) => window.requestAnimationFrame(() => resolve()));
}

function waitForMs(ms: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function mindmapDownloadName(payload: MindmapPayload, extension: "svg" | "png"): string {
  const title = mindmapTitle(payload)
    .toLowerCase()
    .replace(/[^a-z0-9\u00c0-\u00ff]+/gi, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 80);
  return `${title || "mindmap"}.${extension}`;
}

function injectSafetyCss() {
  if (safetyInjected || typeof document === "undefined") return;
  safetyInjected = true;
  const el = document.createElement("style");
  el.textContent = SAFETY_CSS;
  document.head.appendChild(el);
}

function collapseInitialTree(node: MarkmapTreeNode | undefined, depth = 0): void {
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

function fitMindmap(mm: MarkmapInstance | null): void {
  try {
    void mm?.fit?.();
  } catch {
    // ignore
  }
}

function refreshMindmapView(mm: MarkmapInstance | null, svg: SVGSVGElement | null, delay = 80): void {
  if (!svg) return;
  window.setTimeout(() => {
    fitMindmap(mm);
    applyVisibleBranchColors(svg);
    window.setTimeout(() => applyVisibleBranchColors(svg), 420);
  }, delay);
}

function applyVisibleBranchColors(svg: SVGSVGElement): void {
  const branchNodes = Array.from(svg.querySelectorAll<SVGGElement>('g.markmap-node[data-depth="2"]'));
  const colorByBranchPath = new Map<string, string>();
  branchNodes.forEach((node, index) => {
    const path = node.getAttribute("data-path");
    if (path) colorByBranchPath.set(path, BRANCH_COLORS[index % BRANCH_COLORS.length] ?? ROOT_COLOR);
  });

  const resolveColor = (path: string | null, depth: string | null): string => {
    if (!path || depth === "1") return ROOT_COLOR;
    const branchPath = path.split(".").slice(0, 2).join(".");
    return colorByBranchPath.get(branchPath) ?? BRANCH_COLORS[fallbackColorIndex(branchPath) % BRANCH_COLORS.length] ?? ROOT_COLOR;
  };

  for (const node of Array.from(svg.querySelectorAll<SVGGElement>("g.markmap-node"))) {
    const color = resolveColor(node.getAttribute("data-path"), node.getAttribute("data-depth"));
    node.style.setProperty("--branch-color", color);
    node.style.setProperty("color", color, "important");
    for (const labelPart of Array.from(node.querySelectorAll<HTMLElement>("foreignObject, foreignObject *"))) {
      labelPart.style.setProperty("color", color, "important");
      labelPart.style.setProperty("border-color", color, "important");
      labelPart.style.setProperty("text-decoration-color", color, "important");
    }
    node.querySelector<SVGTextElement>("text")?.style.setProperty("fill", color, "important");
    for (const svgPart of Array.from(node.querySelectorAll<SVGElement>("circle,path,line,polyline"))) {
      svgPart.setAttribute("stroke", color);
      svgPart.style.setProperty("stroke", color, "important");
      if (svgPart.tagName.toLowerCase() === "circle") {
        svgPart.setAttribute("fill", color);
        svgPart.style.setProperty("fill", color, "important");
      }
    }
  }

  for (const link of Array.from(svg.querySelectorAll<SVGPathElement>("path.markmap-link"))) {
    const color = resolveColor(link.getAttribute("data-path"), link.getAttribute("data-depth"));
    link.setAttribute("stroke", color);
    link.setAttribute("fill", "none");
    link.style.setProperty("stroke", color, "important");
    link.style.setProperty("fill", "none", "important");
  }
}

function branchColor(node: any): string {
  let branch = node;
  while (branch?.parent?.parent) {
    branch = branch.parent;
  }
  if (!branch?.parent) return ROOT_COLOR;

  const siblings = Array.isArray(branch.parent.children) ? branch.parent.children : [];
  const siblingIndex = siblings.indexOf(branch);
  const colorIndex = siblingIndex >= 0 ? siblingIndex : fallbackColorIndex(String(branch?.state?.path ?? branch?.content ?? ""));
  return BRANCH_COLORS[colorIndex % BRANCH_COLORS.length] ?? ROOT_COLOR;
}

function fallbackColorIndex(value: string): number {
  let hash = 0;
  for (let index = 0; index < value.length; index += 1) {
    hash = (hash * 31 + value.charCodeAt(index)) >>> 0;
  }
  return hash;
}

function mindmapTitle(payload: MindmapPayload): string {
  return payload.central_topic?.trim() || firstHeading(payload.markdown) || "Mind map";
}

function mindmapMarkdown(payload: MindmapPayload): string {
  if (payload.markdown?.trim()) return payload.markdown;
  const title = payload.central_topic?.trim() || "Mind map";
  const lines = [`# ${title}`];
  for (const branch of payload.branches ?? []) {
    appendNode(lines, branch, 2);
  }
  return lines.join("\n");
}

function appendNode(lines: string[], node: MindmapNode, depth: number): void {
  const text = String(node.text || "Topic").trim() || "Topic";
  if (depth <= 6) {
    lines.push(`${"#".repeat(depth)} ${text}`);
  } else {
    lines.push(`${"  ".repeat(depth - 2)}- ${text}`);
  }
  for (const child of node.children ?? []) {
    appendNode(lines, child, depth + 1);
  }
}

function firstHeading(markdown?: string): string | null {
  const line = markdown
    ?.split(/\r?\n/)
    .map((item) => item.trim())
    .find((item) => item.startsWith("# "));
  return line ? line.replace(/^#\s+/, "").trim() : null;
}
