import type { CourseLessonBlock } from "../types";


const PROSE_BLOCK_TYPES = new Set(["markdown", "definition", "example", "procedure", "warning", "summary"]);

export function dedupeCourseBlocks(blocks: CourseLessonBlock[]): CourseLessonBlock[] {
  const kept: CourseLessonBlock[] = [];
  const exactSeen = new Set<string>();

  for (const originalBlock of blocks) {
    const block = cleanStructuredBlock(originalBlock);
    if (!block) continue;
    if (!block.content.trim()) continue;
    const comparable = comparableContent(block.content);
    const exactKey = `${block.block_type}:${comparable}`;
    if (exactSeen.has(exactKey)) continue;
    exactSeen.add(exactKey);

    if (
      PROSE_BLOCK_TYPES.has(block.block_type)
      && kept.some((previous) => (
        PROSE_BLOCK_TYPES.has(previous.block_type)
        && isRedundantProse(block.content, previous.content)
      ))
    ) {
      continue;
    }
    kept.push(block);
  }
  return mergeAdjacentMathBlocks(kept);
}

export function shouldShowLessonSummary(summary: string, blocks: CourseLessonBlock[]): boolean {
  const content = summary.trim();
  if (!content) return false;
  return !blocks.some((block) => (
    PROSE_BLOCK_TYPES.has(block.block_type)
    && isRedundantProse(content, block.content)
  ));
}

export function cleanCourseCitationSnippet(value: string): string {
  return value
    .replace(
      /<\s*(?:page_number|page_header|page_footer|page_break)\b[^>]*>[\s\S]*?<\s*\/\s*(?:page_number|page_header|page_footer|page_break)\s*>/gi,
      " ",
    )
    .replace(/<\/(?:td|th)\s*>/gi, " · ")
    .replace(/<\/tr\s*>/gi, " — ")
    .replace(/<[^>]+>/g, " ")
    .replace(/&nbsp;?/gi, " ")
    .replace(/&amp;/gi, "&")
    .replace(/&lt;/gi, "<")
    .replace(/&gt;/gi, ">")
    .replace(/&quot;/gi, '"')
    .replace(/&#39;|&apos;/gi, "'")
    .replace(/\s+/g, " ")
    .replace(/(?:\s*[·—]\s*){2,}/g, " · ")
    .trim();
}

function isRedundantProse(candidate: string, existing: string): boolean {
  const candidateText = comparableContent(candidate);
  const existingText = comparableContent(existing);
  if (!candidateText || !existingText) return false;
  if (candidateText === existingText) return true;
  if (candidateText.length > existingText.length * 1.15) return false;
  if (existingText.includes(candidateText)) return true;

  const candidateTokens = new Set(candidateText.split(" ").filter(Boolean));
  const existingTokens = new Set(existingText.split(" ").filter(Boolean));
  if (Math.min(candidateTokens.size, existingTokens.size) < 8) return false;
  let overlap = 0;
  for (const token of candidateTokens) {
    if (existingTokens.has(token)) overlap += 1;
  }
  return overlap / candidateTokens.size >= 0.92;
}

function comparableContent(value: string): string {
  return value
    .replace(
      /<\s*(?:page_number|page_header|page_footer|page_break)\b[^>]*>[\s\S]*?<\s*\/\s*(?:page_number|page_header|page_footer|page_break)\s*>/gi,
      " ",
    )
    .replace(/<[^>]+>/g, " ")
    .replace(/[`*_~#$|\\()[\]{}<>:;,.!?=+\-/]+/g, " ")
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLocaleLowerCase()
    .replace(/[^\p{L}\p{N}]+/gu, " ")
    .trim();
}

function cleanStructuredBlock(block: CourseLessonBlock): CourseLessonBlock | null {
  if (block.block_type !== "timeline" || !block.data_json?.events) return block;
  const events = block.data_json.events.filter((event) => (
    !/<\/?(?:table|tr|td|th)\b/i.test(event.description)
  ));
  if (!events.length) return null;
  if (events.length === block.data_json.events.length) return block;
  return {
    ...block,
    content: events.map((event) => `- **${event.date}** — ${event.description}`).join("\n"),
    data_json: { ...block.data_json, events },
  };
}

function mergeAdjacentMathBlocks(blocks: CourseLessonBlock[]): CourseLessonBlock[] {
  const merged: CourseLessonBlock[] = [];
  for (const block of blocks) {
    const previous = merged[merged.length - 1];
    if (!previous || !canMergeMathBlocks(previous, block)) {
      merged.push(block);
      continue;
    }

    const citations = [...previous.citations];
    const citationIds = new Set(citations.map((citation) => citation.chunk_id));
    for (const citation of block.citations) {
      if (!citationIds.has(citation.chunk_id)) citations.push(citation);
    }
    merged[merged.length - 1] = {
      ...previous,
      title: mathGroupTitle(previous.block_type),
      content: `${previous.content.trim()}\n\n${block.content.trim()}`,
      source_chunk_ids: [...new Set([...previous.source_chunk_ids, ...block.source_chunk_ids])],
      citations,
    };
  }
  return merged;
}

function canMergeMathBlocks(left: CourseLessonBlock, right: CourseLessonBlock): boolean {
  if (left.block_type !== right.block_type || !["equation", "matrix", "chemical_equation"].includes(left.block_type)) {
    return false;
  }
  const genericTitle = /^(?:source )?(?:equation|matrix|chemical equation)s?$/i;
  return left.title.trim().toLocaleLowerCase() === right.title.trim().toLocaleLowerCase()
    || (genericTitle.test(left.title.trim()) && genericTitle.test(right.title.trim()));
}

function mathGroupTitle(blockType: string): string {
  if (blockType === "matrix") return "Matrices";
  if (blockType === "chemical_equation") return "Chemical equations";
  return "Equations";
}
