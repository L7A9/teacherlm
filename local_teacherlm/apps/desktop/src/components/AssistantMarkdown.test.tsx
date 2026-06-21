import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { AssistantMarkdown } from "./AssistantMarkdown";
import { normalizeMathMarkdown } from "./mathMarkdown";

describe("normalizeMathMarkdown", () => {
  it("normalizes standard inline and display LaTeX delimiters", () => {
    const normalized = normalizeMathMarkdown("Inline \\(x^2 + 1\\) and display \\[E = mc^2\\].");

    expect(normalized).toContain("$x^2 + 1$");
    expect(normalized).toContain("$$\nE = mc^2\n$$");
  });

  it("wraps and repairs a bare multiline matrix", () => {
    const normalized = normalizeMathMarkdown(String.raw`A = \begin{bmatrix}
1 & 2
3 & 4
\end{bmatrix}`);

    expect(normalized).toContain("$$");
    expect(normalized).toContain(String.raw`A = \begin{bmatrix}1 & 2 \\ 3 & 4\end{bmatrix}`);
  });

  it("closes an unfinished display block while a response is streaming", () => {
    const normalized = normalizeMathMarkdown("The result is $$x^2 + y^2");

    expect(normalized.trimEnd().endsWith("$$")).toBe(true);
    expect(normalized.match(/\$\$/g)).toHaveLength(2);
    expect(normalized).toContain("The result is\n$$\nx^2 + y^2\n$$");
  });

  it("does not rewrite inline or fenced code", () => {
    const markdown = String.raw`Keep \`\\(not math\\)\` literal.

\`\`\`tex
\[also not rendered\]
\begin{pmatrix}1 & 0 \\ 0 & 1\end{pmatrix}
\`\`\``;

    expect(normalizeMathMarkdown(markdown)).toBe(markdown);
  });

  it("converts parser HTML tables and removes page markers", () => {
    const normalized = normalizeMathMarkdown(`Before
<table><tr><th>Item</th><th>Value</th></tr><tr><td>Alpha</td><td>5</td></tr></table>
<page_number>8</page_number>
After`);

    expect(normalized).toContain("| Item | Value |");
    expect(normalized).toContain("| Alpha | 5 |");
    expect(normalized).not.toContain("page_number");
    expect(normalized).not.toContain("<table>");
  });

  it("repairs bold labels damaged while stripping source bullets", () => {
    const normalized = normalizeMathMarkdown(
      "En contexte : choisir la bonne mesure** Objectif :** Le résultat est solide. Pourquoi ce résultat ?** La source explique la raison.",
    );

    expect(normalized).toBe(
      "**En contexte : choisir la bonne mesure** Objectif : Le résultat est solide. **Pourquoi ce résultat ?** La source explique la raison.",
    );
  });
});

describe("AssistantMarkdown", () => {
  it("renders equations and matrices through KaTeX", () => {
    const html = renderToStaticMarkup(
      <AssistantMarkdown content={String.raw`\[A = \begin{pmatrix}1 & 0 \\ 0 & 1\end{pmatrix}\]`} />,
    );

    expect(html).toContain("katex-display");
    expect(html).toContain("mtable");
  });

  it("uses readable renderer variants for both chat roles", () => {
    const assistant = renderToStaticMarkup(<AssistantMarkdown content="$x+y$" />);
    const user = renderToStaticMarkup(<AssistantMarkdown content="$x+y$" variant="user" />);

    expect(assistant).toContain("math-markdown-assistant");
    expect(user).toContain("math-markdown-user");
    expect(assistant).toContain("class=\"katex\"");
    expect(user).toContain("class=\"katex\"");
  });

  it("renders extracted source tables as semantic tables", () => {
    const html = renderToStaticMarkup(
      <AssistantMarkdown content="<table><tr><th>Term</th><th>Meaning</th></tr><tr><td>A</td><td>General value</td></tr></table>" />,
    );

    expect(html).toContain("<table");
    expect(html).toContain("<th");
    expect(html).not.toContain("&lt;table&gt;");
  });
});
