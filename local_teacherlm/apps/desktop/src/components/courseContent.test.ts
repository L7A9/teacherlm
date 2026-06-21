import { describe, expect, it } from "vitest";

import type { CourseLessonBlock } from "../types";
import { cleanCourseCitationSnippet, dedupeCourseBlocks, shouldShowLessonSummary } from "./courseContent";


function block(id: string, blockType: string, content: string): CourseLessonBlock {
  return {
    id,
    block_type: blockType,
    title: id,
    content,
    source_chunk_ids: ["chunk-1"],
    citations: [],
  };
}

describe("course content cleanup", () => {
  it("removes repeated prose while preserving structured blocks", () => {
    const explanation = "A ranking is evaluated from the ordered relevance values. The score is then normalized against the ideal ranking.";
    const blocks = dedupeCourseBlocks([
      block("explanation", "markdown", explanation),
      block("example", "example", explanation),
      block("equation", "equation", "$$nDCG = DCG / IDCG$$"),
      block("summary", "summary", "The score is then normalized against the ideal ranking."),
    ]);

    expect(blocks.map((item) => item.id)).toEqual(["explanation", "equation"]);
  });

  it("hides a lesson summary already stated by the first teaching block", () => {
    const summary = "The source introduces a general method for comparing ordered results.";
    const blocks = [
      block("lesson", "markdown", `${summary} It then develops the method with supported details and constraints.`),
    ];

    expect(shouldShowLessonSummary(summary, blocks)).toBe(false);
  });

  it("keeps a genuinely distinct example", () => {
    const blocks = dedupeCourseBlocks([
      block("definition", "definition", "The source defines the general mechanism and its operating conditions in several connected sentences."),
      block("example", "example", "A separate worked case applies the mechanism to concrete source values and interprets the result."),
    ]);

    expect(blocks).toHaveLength(2);
  });

  it("groups adjacent generic equations into one readable section", () => {
    const blocks = dedupeCourseBlocks([
      { ...block("equation-1", "equation", "$$A = B + C$$"), title: "Equation" },
      { ...block("equation-2", "equation", "$$D = A / N$$"), title: "Equation" },
    ]);

    expect(blocks).toHaveLength(1);
    expect(blocks[0]?.title).toBe("Equations");
    expect(blocks[0]?.content).toContain("A = B + C");
    expect(blocks[0]?.content).toContain("D = A / N");
  });

  it("drops timeline events accidentally extracted from table markup", () => {
    const timeline: CourseLessonBlock = {
      ...block("timeline", "timeline", "- **100** — <tr><td>Rank 100</td></tr>"),
      data_json: { events: [{ date: "100", description: "<tr><td>Rank 100</td></tr>" }] },
    };

    expect(dedupeCourseBlocks([timeline])).toEqual([]);
  });

  it("removes parser tags from citation previews", () => {
    const snippet = "<page_number>8</page_number><table><tr><th>Item</th><th>Value</th></tr><tr><td>A</td><td>5</td></tr></table>";

    expect(cleanCourseCitationSnippet(snippet)).toBe("Item · Value · A · 5 ·");
  });
});
