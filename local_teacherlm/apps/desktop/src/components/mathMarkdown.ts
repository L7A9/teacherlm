const DISPLAY_ENVIRONMENTS = [
  "matrix",
  "pmatrix",
  "bmatrix",
  "Bmatrix",
  "vmatrix",
  "Vmatrix",
  "smallmatrix",
  "array",
  "cases",
  "aligned",
  "alignedat",
  "gathered",
] as const;

type MarkdownSegment = { protected: boolean; value: string };

export function normalizeMathMarkdown(content: string): string {
  return splitFencedCode(content)
    .map((segment) => segment.protected ? segment.value : normalizeUnfencedMarkdown(segment.value))
    .join("");
}

function normalizeUnfencedMarkdown(content: string): string {
  const inlineCode: string[] = [];
  const protectedContent = content.replace(/(`+)([\s\S]*?)\1/g, (match) => {
    const token = `\uE000TEACHERLM_CODE_${inlineCode.length}\uE001`;
    inlineCode.push(match);
    return token;
  });

  let normalized = normalizeExtractedSourceMarkup(protectedContent);
  normalized = repairOrphanStrongMarkers(normalized);
  normalized = repairDamagedLatexCommands(normalized);
  normalized = normalizeLatexDelimiters(normalized);
  normalized = wrapBareMathEnvironments(normalized);
  normalized = canonicalizeDisplayMath(normalized);
  normalized = closeStreamingDisplayMath(normalized);
  normalized = normalizeBareFormulaLines(normalized);
  normalized = degradeMalformedTables(normalized);

  return normalized.replace(/\uE000TEACHERLM_CODE_(\d+)\uE001/g, (_match, index: string) => (
    inlineCode[Number(index)] ?? ""
  ));
}

export function normalizeExtractedSourceMarkup(content: string): string {
  let normalized = content
    .replace(
      /<\s*(?:page_number|page_header|page_footer|page_break)\b[^>]*>[\s\S]*?<\s*\/\s*(?:page_number|page_header|page_footer|page_break)\s*>/gi,
      "\n",
    )
    .replace(/<\s*(?:page_number|page_header|page_footer|page_break)\b[^>]*\/\s*>/gi, "\n")
    .replace(/&nbsp;?/gi, " ");

  normalized = normalized.replace(/<table\b[^>]*>[\s\S]*?<\/table\s*>/gi, (table) => {
    const markdown = htmlTableToMarkdown(table);
    return markdown ? `\n\n${markdown}\n\n` : "\n";
  });
  return normalized.replace(/\n{3,}/g, "\n\n");
}

function repairOrphanStrongMarkers(content: string): string {
  const repaired = content.replace(
    /(^|\n+)(?!\*\*)([\p{Lu}\p{Lt}][^*\n]{2,90})\*\*(?=\s|$)/gu,
    (_match, prefix: string, label: string) => `${prefix}**${label}**`,
  );
  return repaired.replace(
    /(^|[.!?]\s+|\n+)([\p{Lu}\p{Lt}][^*\n]{2,90}[?:])\*\*(?=\s|$)/gu,
    (_match, prefix: string, label: string) => `${prefix}**${label}**`,
  ).replace(
    /(?<!\*)\b([\p{Lu}\p{Lt}][\p{L}\p{N}\s'’()@%+./-]{1,60}[?:])\*\*(?=\s|$)/gu,
    "$1",
  );
}

function htmlTableToMarkdown(table: string): string {
  const rows = [...table.matchAll(/<tr\b[^>]*>([\s\S]*?)<\/tr\s*>/gi)]
    .map((row) => [...String(row[1] ?? "").matchAll(/<t([hd])\b[^>]*>([\s\S]*?)<\/t\1\s*>/gi)]
      .map((cell) => ({ header: String(cell[1]).toLowerCase() === "h", value: cleanHtmlCell(String(cell[2] ?? "")) })))
    .filter((row) => row.length > 0);
  if (!rows.length) return "";

  const width = Math.max(...rows.map((row) => row.length));
  const values = rows.map((row) => [
    ...row.map((cell) => cell.value),
    ...Array.from({ length: Math.max(0, width - row.length) }, () => ""),
  ]);
  if (!rows[0]?.every((cell) => cell.header)) {
    values.unshift(Array.from({ length: width }, (_, index) => `Column ${index + 1}`));
  }
  const [headers = [], ...body] = values;
  return [
    markdownTableRow(headers),
    markdownTableRow(Array.from({ length: width }, () => "---")),
    ...body.map(markdownTableRow),
  ].join("\n");
}

function cleanHtmlCell(value: string): string {
  return decodeHtmlEntities(
    value
      .replace(/<br\s*\/?>/gi, " ")
      .replace(/<[^>]+>/g, " "),
  )
    .replace(/\s+/g, " ")
    .trim()
    .replace(/\|/g, "\\|");
}

function decodeHtmlEntities(value: string): string {
  const named: Record<string, string> = {
    amp: "&",
    apos: "'",
    gt: ">",
    lt: "<",
    nbsp: " ",
    quot: '"',
  };
  return value.replace(/&(#x[0-9a-f]+|#\d+|[a-z]+);/gi, (entity, code: string) => {
    if (code[0] !== "#") return named[code.toLowerCase()] ?? entity;
    const numeric = code[1]?.toLowerCase() === "x"
      ? Number.parseInt(code.slice(2), 16)
      : Number.parseInt(code.slice(1), 10);
    return Number.isFinite(numeric) && numeric >= 0 && numeric <= 0x10FFFF
      ? String.fromCodePoint(numeric)
      : entity;
  });
}

function markdownTableRow(cells: string[]): string {
  return `| ${cells.join(" | ")} |`;
}

function splitFencedCode(content: string): MarkdownSegment[] {
  const lines = content.match(/[^\n]*\n|[^\n]+$/g) ?? [];
  const segments: MarkdownSegment[] = [];
  let buffer = "";
  let fence: { marker: string; length: number } | null = null;
  let bufferProtected = false;

  const flush = () => {
    if (!buffer) return;
    segments.push({ protected: bufferProtected, value: buffer });
    buffer = "";
  };

  for (const line of lines) {
    const fenceMatch = line.match(/^ {0,3}(`{3,}|~{3,})/);
    if (!fence && fenceMatch) {
      flush();
      fence = { marker: fenceMatch[1][0] ?? "`", length: fenceMatch[1].length };
      bufferProtected = true;
      buffer += line;
      continue;
    }

    if (fence) {
      buffer += line;
      const closing = new RegExp(`^ {0,3}${escapeRegex(fence.marker)}{${fence.length},}\\s*$`);
      if (closing.test(line.trimEnd())) {
        flush();
        fence = null;
        bufferProtected = false;
      }
      continue;
    }

    bufferProtected = false;
    buffer += line;
  }

  flush();
  return segments;
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
    .replace(/\\\[([\s\S]*?)\\\]/g, (_match, math: string) => displayBlock(math))
    .replace(/\\\(([\s\S]*?)\\\)/g, (_match, math: string) => {
      const trimmed = math.trim();
      return trimmed ? `$${trimmed}$` : "";
    });
}

function wrapBareMathEnvironments(content: string): string {
  const environments = DISPLAY_ENVIRONMENTS.join("|");
  const pattern = new RegExp(`\\\\begin\\{(${environments})\\}([\\s\\S]*?)\\\\end\\{\\1\\}`, "g");
  const matches = [...content.matchAll(pattern)];
  let normalized = content;

  for (const match of matches.reverse()) {
    if (match.index === undefined || isInsideMath(content, match.index)) continue;
    const environment = match[1] ?? "matrix";
    const whole = match[0];
    const body = match[2] ?? "";
    const lineStart = content.lastIndexOf("\n", match.index - 1) + 1;
    const leading = content.slice(lineStart, match.index);
    const assignment = leading.match(/(?:[A-Za-z][A-Za-z0-9_{}^]*|\\[A-Za-z]+(?:\{[^}]*\})?)\s*=\s*$/);
    const start = assignment?.index === undefined ? match.index : lineStart + assignment.index;
    const expression = `${content.slice(start, match.index)}\\begin{${environment}}${repairEnvironmentBody(body)}\\end{${environment}}`;
    const end = match.index + whole.length;
    normalized = `${normalized.slice(0, start)}${displayBlock(expression)}${normalized.slice(end)}`;
  }

  return normalized;
}

function repairEnvironmentBody(body: string): string {
  const repaired = body
    .replace(/\\dots\s+dots\b/g, "\\dots")
    .replace(/\bdots\s+dots\b/g, "\\dots")
    .replace(/(^|\s)dots(?=\s|&|$)/g, "$1\\dots")
    .replace(/_([A-Za-z0-9]+)/g, "_{$1}");
  const rows = repaired
    .split(/\r?\n/)
    .map((row) => row.trim())
    .filter(Boolean);

  if (rows.length <= 1) {
    return repaired.replace(/\s*\\\\\s*/g, " \\\\ ").replace(/\s*&\s*/g, " & ").trim();
  }

  return rows
    .map((row) => row.replace(/\\+\s*$/, "").trim())
    .join(" \\\\ ")
    .replace(/\s*&\s*/g, " & ")
    .trim();
}

function canonicalizeDisplayMath(content: string): string {
  return content.replace(/\$\$([\s\S]*?)\$\$/g, (_match, math: string) => displayBlock(math));
}

function closeStreamingDisplayMath(content: string): string {
  const delimiters = content.match(/(?<!\\)\$\$/g)?.length ?? 0;
  if (delimiters % 2 === 0) return content;
  const opener = content.lastIndexOf("$$");
  const before = content.slice(0, opener).trimEnd();
  const math = content.slice(opener + 2).trim();
  return `${before ? `${before}\n` : ""}$$\n${math}\n$$`;
}

function normalizeBareFormulaLines(content: string): string {
  const lines = content.split(/\r?\n/);
  const out: string[] = [];
  let inMathBlock = false;

  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed === "$$") {
      inMathBlock = !inMathBlock;
      out.push(line);
      continue;
    }
    if (!inMathBlock && isFormulaLine(trimmed)) {
      out.push("$$", toLatexFormula(trimmed), "$$");
      continue;
    }
    out.push(line);
  }

  return out.join("\n");
}

function isFormulaLine(line: string): boolean {
  if (
    !line
    || line.includes("$")
    || line.endsWith(":")
    || line.length > 260
    || /^(?:\||#{1,6}\s|[-*+]\s|\d+[.)]\s|>)/.test(line)
  ) {
    return false;
  }
  if (isBareLatexFormulaLine(line)) return true;
  if (!/[=≈≤≥]/.test(line)) return false;
  if (!/[√∑Σ∏πμσλΔδθαβγΩω^_]/.test(line)) return false;
  const words = line.match(/[A-Za-zÀ-ÿ]{3,}/g) ?? [];
  const proseWords = words.filter((word) => {
    if (/^[A-Z]{2,}$/.test(word)) return false;
    return !["sqrt", "sum", "min", "max", "log", "sin", "cos", "tan"].includes(word.toLowerCase());
  });
  return proseWords.length <= 2;
}

function isBareLatexFormulaLine(line: string): boolean {
  if (!/\\(?:frac|sum|sqrt|hat|bar|vec|left|right|lVert|rVert|operatorname)\b|\\\|/.test(line)) {
    return false;
  }
  if (/[.!?;:]$/.test(line)) return false;
  const words = line.match(/[A-Za-zÀ-ÿ]{3,}/g) ?? [];
  const proseWords = words.filter((word) => ![
    "frac", "sum", "sqrt", "hat", "bar", "vec", "left", "right", "operatorname",
  ].includes(word.toLowerCase()));
  return proseWords.length <= 2;
}

function toLatexFormula(line: string): string {
  let expression = line
    .replace(/\byi\b/g, "y_i")
    .replace(/\bŷi\b/g, "\\hat{y}_i")
    .replace(/ŷ/g, "\\hat{y}")
    .replace(/([A-Za-z])̂/g, "\\hat{$1}")
    .replace(/\*/g, "\\cdot ")
    .replace(/≤/g, "\\le ")
    .replace(/≥/g, "\\ge ")
    .replace(/≈/g, "\\approx ")
    .replace(/Σ/g, "\\Sigma")
    .replace(/σ/g, "\\sigma")
    .replace(/π/g, "\\pi")
    .replace(/λ/g, "\\lambda")
    .replace(/θ/g, "\\theta")
    .replace(/μ/g, "\\mu")
    .replace(/Δ/g, "\\Delta")
    .replace(/δ/g, "\\delta")
    .replace(/Ω/g, "\\Omega")
    .replace(/ω/g, "\\omega");

  expression = replaceSqrtGroups(expression);
  expression = expression.replace(/∑/g, "\\sum ");
  expression = expression.replace(/^([A-Z]{2,})\s*=/, "\\operatorname{$1} =");
  expression = expression.replace(/\^([A-Za-z0-9]+)/g, "^{$1}");
  expression = expression.replace(/_([A-Za-z0-9]+)/g, "_{$1}");
  return repairDanglingLatex(expression.replace(/\s+/g, " ").trim());
}

function replaceSqrtGroups(input: string): string {
  let out = "";
  for (let index = 0; index < input.length; index += 1) {
    if (input[index] !== "√" || input[index + 1] !== "(") {
      out += input[index];
      continue;
    }
    const close = findMatchingParen(input, index + 1);
    if (close === -1) {
      out += "\\sqrt";
      continue;
    }
    out += `\\sqrt{${input.slice(index + 2, close)}}`;
    index = close;
  }
  return out;
}

function findMatchingParen(input: string, openIndex: number): number {
  let depth = 0;
  for (let index = openIndex; index < input.length; index += 1) {
    if (input[index] === "(") depth += 1;
    else if (input[index] === ")") {
      depth -= 1;
      if (depth === 0) return index;
    }
  }
  return -1;
}

function repairDanglingLatex(expression: string): string {
  if (!/\\+$/.test(expression)) return expression;
  const normDelimiterCount = expression.match(/\\\|/g)?.length ?? 0;
  return normDelimiterCount % 2 === 1
    ? expression.replace(/\\+$/, "\\|")
    : expression.replace(/\\+$/, "").trim();
}

function degradeMalformedTables(content: string): string {
  const lines = content.split(/\r?\n/);
  const out: string[] = [];

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index] ?? "";
    const next = lines[index + 1] ?? "";
    if (!isTableRow(line) || !isSeparatorRow(next)) {
      out.push(line);
      continue;
    }

    const block = [line, next];
    let end = index + 2;
    for (; end < lines.length; end += 1) {
      const candidate = lines[end] ?? "";
      if (!candidate.trim()) break;
      block.push(candidate);
    }

    if (!block.some((candidate, row) => row !== 1 && candidate.includes("$$"))) {
      out.push(...block);
      index = end - 1;
      continue;
    }

    out.push(...renderMalformedTableAsSections(block));
    index = end - 1;
  }

  return out.join("\n");
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
    if (cleanCell(label ?? "")) rendered.push(`### ${cleanCell(label ?? "")}`);
    if (left?.trim()) rendered.push(`**${leftHeader}:**`, "", cleanCell(left));
    if (right?.trim()) rendered.push("", `**${rightHeader}:**`, "", cleanCell(right));
    rendered.push("");
    current = null;
  };

  for (const line of lines.slice(2)) {
    if (!line.trim() || isSeparatorRow(line)) continue;
    if (isTableRow(line)) {
      const cells = splitTableRow(line);
      if (cells[0]?.trim()) {
        flush();
        current = [cells[0] ?? "", cells[1] ?? "", cells.slice(2).join(" | ")];
      } else if (current) {
        current[2] = [current[2], cells.join(" | ")].filter(Boolean).join("\n");
      }
    } else if (current) {
      current[2] = [current[2], line].filter(Boolean).join("\n");
    } else {
      rendered.push(line);
    }
  }

  flush();
  return rendered.length ? rendered : lines;
}

function isTableRow(line: string): boolean {
  const trimmed = line.trim();
  return trimmed.startsWith("|") && trimmed.includes("|", 1);
}

function isSeparatorRow(line: string): boolean {
  return /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(line);
}

function splitTableRow(line: string): string[] {
  return line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((cell) => cell.trim());
}

function cleanCell(value: string): string {
  return value
    .trim()
    .replace(/^\|\s*/, "")
    .replace(/\s*\|$/, "")
    .replace(/^\*\*(.+)\*\*$/, "$1")
    .trim();
}

function displayBlock(math: string): string {
  const trimmed = math.trim();
  return trimmed ? `\n$$\n${trimmed}\n$$\n` : "";
}

function isInsideMath(content: string, end: number): boolean {
  let inline = false;
  let display = false;
  for (let index = 0; index < end; index += 1) {
    if (content[index] !== "$" || isEscaped(content, index)) continue;
    if (content[index + 1] === "$") {
      display = !display;
      index += 1;
    } else if (!display) {
      inline = !inline;
    }
  }
  return inline || display;
}

function isEscaped(content: string, index: number): boolean {
  let backslashes = 0;
  for (let cursor = index - 1; cursor >= 0 && content[cursor] === "\\"; cursor -= 1) backslashes += 1;
  return backslashes % 2 === 1;
}

function escapeRegex(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
