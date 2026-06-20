import { useEffect, useState } from "react";

import { Check, Copy } from "lucide-react";
import ReactMarkdown, { type Components } from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark, oneLight } from "react-syntax-highlighter/dist/cjs/styles/prism";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import type { PluggableList } from "unified";

import { normalizeMathMarkdown } from "./mathMarkdown";

const markdownComponents: Components = {
  h1: ({ node, ...props }) => <h1 className="mb-4 mt-6 text-2xl font-bold text-foreground" {...props} />,
  h2: ({ node, ...props }) => (
    <h2 className="mb-3 mt-5 border-b border-border pb-1.5 text-xl font-semibold text-foreground" {...props} />
  ),
  h3: ({ node, ...props }) => <h3 className="mb-2 mt-4 text-lg font-semibold text-foreground" {...props} />,
  h4: ({ node, ...props }) => <h4 className="mb-1.5 mt-3 text-base font-semibold text-foreground" {...props} />,
  h5: ({ node, ...props }) => (
    <h5 className="mb-1 mt-3 text-sm font-semibold uppercase tracking-wide text-muted-foreground" {...props} />
  ),
  h6: ({ node, ...props }) => (
    <h6 className="mb-1 mt-3 text-xs font-semibold uppercase tracking-wide text-muted-foreground" {...props} />
  ),
  p: ({ node, ...props }) => <p className="mb-4 leading-7 text-surface-foreground last:mb-0" {...props} />,
  ul: ({ node, ...props }) => (
    <ul
      className="mb-4 ml-6 list-disc space-y-2 text-surface-foreground marker:text-muted-foreground last:mb-0"
      {...props}
    />
  ),
  ol: ({ node, ...props }) => (
    <ol
      className="mb-4 ml-6 list-decimal space-y-2 text-surface-foreground marker:text-muted-foreground last:mb-0"
      {...props}
    />
  ),
  li: ({ node, ...props }) => <li className="leading-7" {...props} />,
  code: ({ node, className, children, ...props }) => {
    const match = /language-(\w+)/.exec(className ?? "");
    const codeText = String(children).replace(/\n$/, "");
    const language = match?.[1];
    if (language) {
      return <CodeBlock language={language} code={codeText} />;
    }
    return (
      <code className="rounded bg-muted px-1.5 py-0.5 font-mono text-[0.85em] text-primary" {...props}>
        {children}
      </code>
    );
  },
  pre: ({ children }) => <>{children}</>,
  blockquote: ({ node, ...props }) => (
    <blockquote className="my-4 border-l-4 border-accent bg-accent/10 px-4 py-2 not-italic text-surface-foreground" {...props} />
  ),
  table: ({ node, ...props }) => (
    <div className="my-4 overflow-x-auto">
      <table className="min-w-full border-collapse border border-border text-left" {...props} />
    </div>
  ),
  thead: ({ node, ...props }) => <thead className="bg-muted" {...props} />,
  th: ({ node, ...props }) => (
    <th className="border border-border px-4 py-2 text-left font-semibold text-foreground" {...props} />
  ),
  td: ({ node, ...props }) => (
    <td className="border border-border px-4 py-2 align-top text-surface-foreground" {...props} />
  ),
  tr: ({ node, ...props }) => <tr className="even:bg-muted/50" {...props} />,
  a: ({ node, ...props }) => (
    <a className="text-primary underline-offset-2 hover:underline" target="_blank" rel="noreferrer" {...props} />
  ),
  hr: ({ node, ...props }) => <hr className="my-6 border-border" {...props} />,
  strong: ({ node, ...props }) => <strong className="font-semibold text-foreground" {...props} />,
  em: ({ node, ...props }) => <em className="italic text-surface-foreground" {...props} />,
};

const markdownRemarkPlugins: PluggableList = [remarkGfm, remarkMath];
const markdownRehypePlugins: PluggableList = [[rehypeKatex, { strict: false, throwOnError: false }]];

export function AssistantMarkdown({
  content,
  className = "",
  variant = "assistant",
}: {
  content: string;
  className?: string;
  variant?: "assistant" | "user";
}) {
  const normalized = normalizeMathMarkdown(content);
  const variantClass = variant === "user"
    ? "math-markdown-user text-primary-foreground [&_p]:whitespace-pre-wrap"
    : "math-markdown-assistant";

  return (
    <div className={`math-markdown prose prose-slate min-w-0 max-w-none dark:prose-invert ${variantClass} ${className}`.trim()}>
      <ReactMarkdown
        remarkPlugins={markdownRemarkPlugins}
        rehypePlugins={markdownRehypePlugins}
        components={markdownComponents}
      >
        {normalized}
      </ReactMarkdown>
    </div>
  );
}

function repairDamagedLatexCommands(content: string): string {
  let repaired = content
    .replace(/\f(?=rac\b)/g, "\\f")
    .replace(/\t(?=ext\b)/g, "\\t")
    .replace(/\r(?=oot\b)/g, "\\r")
    .replace(/\\root\{2\}\{/g, "\\sqrt{")
    .replace(/(^|[^\\A-Za-z])oot\{2\}\{/g, (_match: string, prefix: string) => `${prefix}\\sqrt{`)
    .replace(/(^|[^\\A-Za-z])root\{2\}\{/g, (_match: string, prefix: string) => `${prefix}\\sqrt{`);

  const commands: Record<string, string> = {
    rac: "frac",
    ext: "text",
    hat: "hat",
    sqrt: "sqrt",
    sum: "sum",
  };

  for (const [damaged, command] of Object.entries(commands)) {
    repaired = repaired.replace(
      new RegExp(`(^|[^\\\\A-Za-z])${damaged}(?=\\s*\\{|_)`, "g"),
      (_match: string, prefix: string) => `${prefix}\\${command}`,
    );
  }

  return repaired.replace(/\\text\{\s*sum\s*\}(?=_|\s*_\{)/g, "\\sum");
}

function normalizeLatexDelimiters(content: string): string {
  return content
    .replace(/\\\[([\s\S]*?)\\\]/g, (_match, math: string) => {
      const trimmed = math.trim();
      return trimmed ? `$$\n${trimmed}\n$$` : "";
    })
    .replace(/\\\(([\s\S]*?)\\\)/g, (_match, math: string) => {
      const trimmed = math.trim();
      return trimmed ? `$${trimmed}$` : "";
    });
}

function normalizeLeakedLatexMatrices(content: string): string {
  const lines = content.split(/\r?\n/);
  const out: string[] = [];
  let inFence = false;
  let inMathBlock = false;

  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed.startsWith("```")) {
      inFence = !inFence;
      out.push(line);
      continue;
    }
    if (trimmed === "$$") {
      inMathBlock = !inMathBlock;
      out.push(line);
      continue;
    }
    if (!inFence && !inMathBlock && containsLatexMatrix(trimmed)) {
      const parts = splitMatrixLine(line);
      if (parts) {
        if (parts.before) out.push(parts.before);
        out.push("$$", repairLatexMatrix(parts.matrix), "$$");
        if (parts.after) out.push(parts.after);
        continue;
      }
    }
    out.push(line);
  }

  return out.join("\n");
}

function containsLatexMatrix(line: string): boolean {
  return /\\begin\{[a-z]*matrix\}/.test(line) && /\\end\{[a-z]*matrix\}/.test(line);
}

function splitMatrixLine(line: string): { before: string; matrix: string; after: string } | null {
  const begin = line.search(/\\begin\{[a-z]*matrix\}/);
  const endMatch = line.match(/\\end\{[a-z]*matrix\}/);
  if (begin < 0 || !endMatch || endMatch.index === undefined) return null;

  const leading = line.slice(0, begin);
  const assignment = leading.match(/([A-Za-z][A-Za-z0-9_{}^\\]*\s*=\s*)$/);
  const start = assignment?.index === undefined ? begin : assignment.index;
  const end = endMatch.index + endMatch[0].length;
  return {
    before: line.slice(0, start).trim(),
    matrix: line.slice(start, end).trim(),
    after: line.slice(end).trim(),
  };
}

function repairLatexMatrix(raw: string): string {
  const match = raw.match(/^(.*?)\\begin\{([a-z]*matrix)\}([\s\S]*?)\\end\{\2\}(.*)$/);
  if (!match) return raw;

  const [, prefix = "", env = "pmatrix", body = "", suffix = ""] = match;
  const repairedBody = repairLatexMatrixBody(body);
  const repairedPrefix = repairLatexExpression(prefix);
  const repairedSuffix = repairLatexExpression(suffix);
  return `${repairedPrefix}\\begin{${env}}${repairedBody}\\end{${env}}${repairedSuffix}`.trim();
}

function repairLatexMatrixBody(body: string): string {
  let repaired = body
    .replace(/\\dots\s+dots\b/g, "\\dots")
    .replace(/\bdots\s+dots\b/g, "\\dots")
    .replace(/\bdots\b/g, "\\dots")
    .replace(/\\vdots\s+\\vdots\b/g, "\\vdots");

  repaired = repaired.replace(/((?:[A-Za-z]+_\{?\d+\}?)|(?:\d+))\s+_\{?\d+\}?/g, "$1");
  repaired = repaired.replace(/_([A-Za-z0-9]+)/g, "_{$1}");
  repaired = repaired.replace(/\s+(?=(?:u_\{?\d+\}?|u_m|\\vdots)\s*&)/g, " \\\\ ");
  repaired = repaired.replace(/\s*&\s*/g, " & ");
  repaired = repaired.replace(/\s*\\\\\s*/g, " \\\\ ");
  return repaired.replace(/\s+/g, " ").trim();
}

function repairLatexExpression(value: string): string {
  return value
    .replace(/\\dots\s+dots\b/g, "\\dots")
    .replace(/\bdots\b/g, "\\dots")
    .replace(/_([A-Za-z0-9]+)/g, "_{$1}")
    .replace(/\s+/g, " ");
}

function normalizeDisplayMath(content: string): string {
  const lines = content.split(/\r?\n/);
  const out: string[] = [];
  let inFence = false;
  let inMathBlock = false;

  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed.startsWith("```")) {
      inFence = !inFence;
      out.push(line);
      continue;
    }
    if (trimmed === "$$") {
      inMathBlock = !inMathBlock;
      out.push(line);
      continue;
    }
    if (!inFence && !inMathBlock && isFormulaLine(trimmed)) {
      out.push("$$", toLatexFormula(trimmed), "$$");
      continue;
    }
    out.push(line);
  }

  return out.join("\n");
}

function degradeMalformedTables(content: string): string {
  const lines = content.split(/\r?\n/);
  const out: string[] = [];

  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i] ?? "";
    const next = lines[i + 1] ?? "";
    if (!isTableRow(line) || !isSeparatorRow(next)) {
      out.push(line);
      continue;
    }

    const block: string[] = [line, next];
    let j = i + 2;
    for (; j < lines.length; j += 1) {
      const candidate = lines[j] ?? "";
      if (!candidate.trim()) break;
      block.push(candidate);
    }

    if (!isMalformedTableBlock(block)) {
      out.push(...block);
      i = j - 1;
      continue;
    }

    out.push(...renderMalformedTableAsSections(block));
    i = j - 1;
  }

  return out.join("\n");
}

function repairMalformedMathFences(content: string): string {
  const lines = content.split(/\r?\n/);
  const out: string[] = [];
  let inFence = false;
  let inMathBlock = false;

  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed.startsWith("```")) {
      inFence = !inFence;
      out.push(line);
      continue;
    }
    if (inFence) {
      out.push(line);
      continue;
    }
    if (trimmed === "$$") {
      inMathBlock = !inMathBlock;
      out.push(line);
      continue;
    }
    if (inMathBlock && trimmed.startsWith("$$")) {
      const trailing = trimmed.slice(2).trim();
      inMathBlock = false;
      out.push("$$");
      if (trailing) out.push(trailing);
      continue;
    }

    const inlineDisplayMath = splitInlineDisplayMath(line);
    if (inlineDisplayMath) {
      out.push(...inlineDisplayMath);
      continue;
    }

    out.push(line);
  }

  if (inMathBlock) out.push("$$");
  return out.join("\n");
}

function isTableRow(line: string): boolean {
  const trimmed = line.trim();
  return trimmed.startsWith("|") && trimmed.includes("|", 1);
}

function isSeparatorRow(line: string): boolean {
  return /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(line);
}

function isMalformedTableBlock(lines: string[]): boolean {
  return lines.some((line, index) => {
    const trimmed = line.trim();
    if (!trimmed) return false;
    if (index === 1 && isSeparatorRow(line)) return false;
    return trimmed.includes("$$") || !isTableRow(line);
  });
}

function renderMalformedTableAsSections(lines: string[]): string[] {
  const headers = splitTableRow(lines[0] ?? "");
  const leftHeader = cleanCell(headers[1] ?? "Option A");
  const rightHeader = cleanCell(headers[2] ?? "Option B");
  const rendered: string[] = [];
  let current: string[] | null = null;

  const flush = () => {
    if (!current) return;
    const [label, left, right] = current;
    const title = cleanCell(label ?? "");
    if (title) rendered.push(`### ${title}`);
    if (left?.trim()) {
      rendered.push(`**${leftHeader}:**`, "", cleanCell(left));
    }
    if (right?.trim()) {
      rendered.push("", `**${rightHeader}:**`, "", cleanCell(right));
    }
    rendered.push("");
    current = null;
  };

  for (const line of lines.slice(2)) {
    const trimmed = line.trim();
    if (!trimmed || isSeparatorRow(line)) continue;

    if (isTableRow(line)) {
      const cells = splitTableRow(line);
      if (cells.length >= 2 && cells[0]?.trim()) {
        flush();
        current = [cells[0] ?? "", cells[1] ?? "", cells.slice(2).join(" | ")];
      } else if (current) {
        current[2] = [current[2], cells.join(" | ")].filter(Boolean).join("\n");
      }
      continue;
    }

    if (current) {
      const continuationTarget = !current[2]?.trim() && !trimmed.startsWith("|") ? 1 : 2;
      current[continuationTarget] = [current[continuationTarget], line].filter(Boolean).join("\n");
    } else {
      rendered.push(line);
    }
  }

  flush();
  return rendered.length ? rendered : lines;
}

function splitTableRow(line: string): string[] {
  const trimmed = line.trim().replace(/^\|/, "").replace(/\|$/, "");
  return trimmed.split("|").map((cell) => cell.trim());
}

function cleanCell(value: string): string {
  const stripped = value.trim();
  if (stripped.includes("\n") || stripped.includes("$$")) {
    return stripped
      .split(/\r?\n/)
      .map(cleanTableContinuationLine)
      .join("\n")
      .replace(/^\*\*(.+)\*\*$/, "$1")
      .trim();
  }
  return cleanTableContinuationLine(stripped)
    .replace(/\s+/g, " ")
    .replace(/^\*\*(.+)\*\*$/, "$1")
    .trim();
}

function cleanTableContinuationLine(line: string): string {
  return line
    .trim()
    .replace(/^\|\s*/, "")
    .replace(/\s*\|$/, "")
    .replace(/\s+\|\s+/g, "\n\n")
    .trim();
}

function splitInlineDisplayMath(line: string): string[] | null {
  const first = line.indexOf("$$");
  if (first === -1) return null;
  const second = line.indexOf("$$", first + 2);
  if (second === -1) return null;

  const before = line.slice(0, first).trim();
  const math = line.slice(first + 2, second).trim();
  const after = line.slice(second + 2).trim();

  if (!math) return null;
  return [...(before ? [before] : []), "$$", math, "$$", ...(after ? [after] : [])];
}

function isFormulaLine(line: string): boolean {
  if (!line || line.includes("$") || line.endsWith(":") || line.length > 220) {
    return false;
  }
  if (isBareLatexFormulaLine(line)) return true;
  if (!/[=\u2248\u2264\u2265]/.test(line)) return false;
  if (!/[\u221a\u2211\u03a3\u220f\u03c0\u03bc\u03c3\u03bb\u0394\u03b4\u03b8\u03b1\u03b2\u03b3\u03a9\u03c9^_]/.test(line)) {
    return false;
  }
  const words = line.match(/[A-Za-z\u00c0-\u00ff]{3,}/g) ?? [];
  const proseWords = words.filter((word) => {
    if (/^[A-Z]{2,}$/.test(word)) return false;
    return !["sqrt", "sum", "min", "max", "log", "sin", "cos", "tan"].includes(word.toLowerCase());
  });
  return proseWords.length <= 2;
}

function isBareLatexFormulaLine(line: string): boolean {
  if (!/\\(?:frac|sum|sqrt|hat|bar|vec|left|right|lVert|rVert|begin|end)\b|\\\|/.test(line)) {
    return false;
  }
  if (/[.!?;:]$/.test(line)) return false;
  const words = line.match(/[A-Za-z\u00c0-\u00ff]{3,}/g) ?? [];
  const proseWords = words.filter((word) => {
    const normalized = word.toLowerCase();
    return ![
      "frac",
      "sum",
      "sqrt",
      "hat",
      "bar",
      "vec",
      "left",
      "right",
      "begin",
      "end",
      "pmatrix",
      "bmatrix",
      "matrix",
      "operatorname",
    ].includes(normalized);
  });
  return proseWords.length <= 2;
}

function toLatexFormula(line: string): string {
  let expr = line
    .replace(/\byi\b/g, "y_i")
    .replace(/\b\u0177i\b/g, "\\hat{y}_i")
    .replace(/\u0177/g, "\\hat{y}")
    .replace(/([A-Za-z])\u0302/g, "\\hat{$1}")
    .replace(/\*/g, "\\cdot ")
    .replace(/\u2264/g, "\\le ")
    .replace(/\u2265/g, "\\ge ")
    .replace(/\u2248/g, "\\approx ")
    .replace(/\u03a3/g, "\\Sigma")
    .replace(/\u03c3/g, "\\sigma")
    .replace(/\u03c0/g, "\\pi")
    .replace(/\u03bb/g, "\\lambda")
    .replace(/\u03b8/g, "\\theta")
    .replace(/\u03bc/g, "\\mu")
    .replace(/\u0394/g, "\\Delta")
    .replace(/\u03b4/g, "\\delta")
    .replace(/\u03a9/g, "\\Omega")
    .replace(/\u03c9/g, "\\omega");

  expr = replaceSqrtGroups(expr);
  expr = expr.replace(/\u2211/g, "\\sum ");
  expr = expr.replace(/^([A-Z]{2,})\s*=/, "\\operatorname{$1} =");
  expr = expr.replace(/\^([A-Za-z0-9]+)/g, "^{$1}");
  expr = expr.replace(/_([A-Za-z0-9]+)/g, "_{$1}");
  return repairDanglingLatex(expr.replace(/\s+/g, " ").trim());
}

function repairDanglingLatex(expr: string): string {
  if (!/\\+$/.test(expr)) return expr;
  const normDelimiterCount = expr.match(/\\\|/g)?.length ?? 0;
  if (normDelimiterCount % 2 === 1) {
    return expr.replace(/\\+$/, "\\|");
  }
  return expr.replace(/\\+$/, "").trim();
}

function replaceSqrtGroups(input: string): string {
  let out = "";
  for (let i = 0; i < input.length; i += 1) {
    if (input[i] !== "\u221a" || input[i + 1] !== "(") {
      out += input[i];
      continue;
    }
    const close = findMatchingParen(input, i + 1);
    if (close === -1) {
      out += "\\sqrt";
      continue;
    }
    const inner = input.slice(i + 2, close);
    out += `\\sqrt{${formatSqrtInner(inner)}}`;
    i = close;
  }
  return out;
}

function findMatchingParen(input: string, openIndex: number): number {
  let depth = 0;
  for (let i = openIndex; i < input.length; i += 1) {
    if (input[i] === "(") depth += 1;
    else if (input[i] === ")") {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

function formatSqrtInner(inner: string): string {
  const normalized = inner
    .replace(/\byi\b/g, "y_i")
    .replace(/\b\u0177i\b/g, "\\hat{y}_i")
    .replace(/\u0177/g, "\\hat{y}")
    .replace(/([A-Za-z])\u0302/g, "\\hat{$1}")
    .replace(/\u2211/g, "\\sum ")
    .replace(/\*/g, "\\cdot ");

  const leadingAverage = normalized.match(/^\s*1\s*\/\s*([A-Za-z][A-Za-z0-9]*)\s*(?:\\cdot\s*)?(.+)$/);
  if (leadingAverage) {
    const [, denominator, rest] = leadingAverage;
    if (denominator && rest) {
      return `\\frac{1}{${denominator}} ${rest.trim()}`;
    }
  }

  const fraction = splitTopLevel(normalized, "/");
  if (fraction) {
    return `\\frac{${fraction.left.trim()}}{${fraction.right.trim()}}`;
  }
  return normalized;
}

function splitTopLevel(input: string, separator: string): { left: string; right: string } | null {
  let depth = 0;
  for (let i = 0; i < input.length; i += 1) {
    if (input[i] === "(") depth += 1;
    else if (input[i] === ")") depth -= 1;
    else if (input[i] === separator && depth === 0) {
      return {
        left: input.slice(0, i),
        right: input.slice(i + 1),
      };
    }
  }
  return null;
}

function CodeBlock({ language, code }: { language: string; code: string }) {
  const [copied, setCopied] = useState(false);
  const isDark = useIsDarkTheme();

  const onCopy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard may be unavailable in some desktop WebView contexts.
    }
  };

  return (
    <div className="group relative my-4 overflow-hidden rounded-lg border border-border bg-background">
      <div className="flex items-center justify-between border-b border-border bg-muted px-3 py-1.5 text-[11px] text-muted-foreground">
        <span className="font-mono uppercase tracking-wide">{language}</span>
        <button
          type="button"
          onClick={onCopy}
          className="app-chrome inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-muted-foreground transition-colors hover:bg-background hover:text-foreground active:bg-background/80"
          aria-label={copied ? "Copied" : "Copy code"}
        >
          {copied ? <Check className="h-3 w-3" /> : <Copy className="h-3 w-3" />}
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <SyntaxHighlighter
        language={language}
        style={isDark ? oneDark : oneLight}
        PreTag="div"
        customStyle={{
          margin: 0,
          padding: "0.875rem 1rem",
          background: "transparent",
          fontSize: "0.85em",
          lineHeight: 1.6,
        }}
        codeTagProps={{
          style: { fontFamily: "var(--font-mono)" },
        }}
      >
        {code}
      </SyntaxHighlighter>
    </div>
  );
}

function useIsDarkTheme(): boolean {
  const [isDark, setIsDark] = useState(() => document.documentElement.classList.contains("dark"));

  useEffect(() => {
    const update = () => setIsDark(document.documentElement.classList.contains("dark"));
    const observer = new MutationObserver(update);
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
    update();
    return () => observer.disconnect();
  }, []);

  return isDark;
}
