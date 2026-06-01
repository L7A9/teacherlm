import json
import re
import unicodedata
from collections.abc import AsyncIterator
from dataclasses import dataclass

from teacherlm_core.llm.language import set_current_language
from teacherlm_core.llm.runtime import set_current_llm_options
from teacherlm_core.schemas.chunk import Chunk
from teacherlm_core.schemas.generator_io import GeneratorInput, LearnerUpdates

from .config import get_settings
from .schemas import ResponseMode
from .services.confidence_scorer import compute as compute_confidence
from .services.learner_analyzer import extract_learner_updates
from .services.llm_service import build_chat_system_prompt, get_llm_service
from .services.query_analyzer import analyze as analyze_query
from .services.response_mode import select_mode

_MODE_PROMPT_FILE: dict[ResponseMode, str] = {
    "explain": "mode_explain.txt",
    "guide": "mode_guide.txt",
    "quiz_back": "mode_quiz_back.txt",
    "affirm": "mode_affirm.txt",
}


@dataclass(slots=True)
class FormulaCard:
    formula: str
    source: str
    definitions: list[str]


@dataclass(slots=True)
class CourseTopic:
    title: str
    source: str
    details: list[str]


def _format_chunks(chunks: list[Chunk]) -> str:
    if not chunks:
        return "(no context chunks available)"
    return "\n\n".join(
        _format_chunk(i + 1, c)
        for i, c in enumerate(chunks)
    )


def _format_chunk(index: int, chunk: Chunk) -> str:
    formulas = _extract_formula_snippets(chunk.text)
    formula_block = (
        "\nKey formulas found in this chunk:\n"
        + "\n".join(f"- {formula}" for formula in formulas)
        if formulas
        else ""
    )
    return (
        f"[{index}] source={chunk.source} score={chunk.score:.3f}"
        f"{formula_block}\n{chunk.text}"
    )


def _extract_formula_snippets(text: str, *, limit: int = 4) -> list[str]:
    snippets: list[str] = []

    for match in re.finditer(r"\$\$(.+?)\$\$", text, flags=re.DOTALL):
        formula = " ".join(match.group(1).split())
        if formula:
            snippets.append(f"$$ {formula} $$")

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or "$$" in line:
            continue
        if not _looks_like_formula_line(line):
            continue
        snippets.append(line)

    seen: set[str] = set()
    out: list[str] = []
    for snippet in snippets:
        key = re.sub(r"\s+", " ", snippet).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key[:500])
        if len(out) >= limit:
            break
    return out


_MATH_QUERY_EXPR_RE = re.compile(
    r"(\b[A-Za-z]\w*\s*(?:=|\+|\*|/|\^)\s*[A-Za-z0-9\\$]|\b\w+\s*[_^]\s*\w+|\\(?:frac|sum|sqrt|hat|bar|vec|int|prod)\b)"
)
_FORMULA_WORD_RE = re.compile(
    r"\b(equation|formula|formule|calculate|calculer|compute|derive|symbol|math|Ã©quation|معادلة|صيغة|احسب)\b",
    re.IGNORECASE,
)
_CODE_OR_MARKUP_RE = re.compile(
    r"(<\w+|</\w+|android:|xmlns:|=\s*[\"']|;\s*$|\b(?:new|class|return|public|private|protected|const|let|var)\b)",
    re.IGNORECASE,
)


def _looks_like_formula_line(line: str) -> bool:
    if _CODE_OR_MARKUP_RE.search(line):
        return False
    if re.search(r"\\(?:frac|sum|sqrt|hat|bar|vec|int|prod)\b|[∑√∫±×÷≤≥≈∞]", line):
        return True
    if re.search(r"\b(?:RMSE|MSE|MAE|DCG|nDCG|TF-IDF)\b", line):
        return bool(re.search(r"[=^_]", line))
    if "$" in line:
        return bool(re.search(r"[=^_]|\\", line))
    return bool(
        re.search(
            r"\b[A-Za-z]\w*\s*=\s*[-+*/^().,\w\s]+$|"
            r"\b[A-Za-z]\w*\s*[_^]\s*[A-Za-z0-9]",
            line,
        )
    )


def _is_formula_question(message: str) -> bool:
    return bool(_FORMULA_WORD_RE.search(message) or _MATH_QUERY_EXPR_RE.search(message))


_QUERY_EVIDENCE_STOPWORDS = {
    "about",
    "and",
    "are",
    "can",
    "course",
    "define",
    "describe",
    "explain",
    "file",
    "files",
    "for",
    "from",
    "give",
    "how",
    "its",
    "lesson",
    "material",
    "materials",
    "me",
    "please",
    "show",
    "summarize",
    "teach",
    "tell",
    "the",
    "this",
    "to",
    "uploaded",
    "want",
    "what",
    "why",
    "you",
    "calculate",
    "compute",
    "derive",
    "derivation",
    "equation",
    "equations",
    "formula",
    "formulas",
    "math",
    "symbol",
    "symbols",
}


def _query_evidence_terms(message: str) -> set[str]:
    terms: set[str] = set()
    for raw in re.findall(r"[A-Za-z][A-Za-z0-9_@+-]*", message):
        term = raw.casefold()
        if len(term) < 3 or term in _QUERY_EVIDENCE_STOPWORDS:
            continue
        terms.add(term)
    return terms


def _has_context_evidence(message: str, chunks: list[Chunk]) -> bool:
    """Detect direct source mentions when reranker logits are negative."""

    terms = _query_evidence_terms(message)
    if not terms or not chunks:
        return False

    haystack_parts: list[str] = []
    for chunk in chunks:
        haystack_parts.extend(
            [
                chunk.text,
                chunk.source,
                str(chunk.metadata.get("heading_path", "")),
                " ".join(str(item) for item in chunk.metadata.get("key_concepts", []) or []),
            ]
        )
    haystack = " ".join(haystack_parts).casefold()
    if not haystack.strip():
        return False

    hits = 0
    for term in terms:
        pattern = rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])"
        if re.search(pattern, haystack):
            hits += 1

    required = 1 if len(terms) == 1 else min(2, len(terms))
    return hits >= required


def _is_formula_only_question(message: str) -> bool:
    if not _is_formula_question(message):
        return False
    normalized = message.casefold()
    direct_formula = bool(
        re.search(
            r"\b(?:what|give|show|write|provide|list)\b.*\b(?:formula|formulas|equation|equations|symbol|symbols)\b|"
            r"\b(?:formula|formulas|equation|equations)\s+for\b",
            normalized,
        )
        or _MATH_QUERY_EXPR_RE.search(message)
    )
    asks_for_explanation = bool(
        re.search(r"\b(?:explain|teach|describe|overview|definition|define)\b", normalized)
    )
    return direct_formula and not asks_for_explanation


def _is_course_overview_question(message: str) -> bool:
    normalized = message.casefold()
    normalized = re.sub(r"[^\w\sà-ÿ]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return bool(
        re.search(
            r"\b(what (?:is |s )?(?:this|that) course about|"
            r"what (?:are )?(?:these|those) (?:documents|files|materials) about|"
            r"explain this course|explain the course|course overview|"
            r"summarize this course|summarise this course|summarize the course|summarise the course|"
            r"starting today|what should i study first|where should i start|"
            r"beginner roadmap|prepare me for the exam|teach me this course|"
            r"de quoi parle|résume.*cours|resume.*cours|explique.*cours)\b",
            normalized,
        )
    )


def _norm_label(value: str) -> str:
    value = unicodedata.normalize("NFKD", value.casefold())
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = re.sub(r"[^a-z0-9à-ÿ]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _is_generic_course_label(value: str) -> bool:
    normalized = _norm_label(value)
    return normalized in {
        "",
        "course",
        "cours",
        "module",
        "section",
        "introduction",
        "conclusion",
    }


def _is_generic_course_label(value: str) -> bool:
    normalized = _norm_label(value)
    return normalized in {
        "",
        "course",
        "cours",
        "module",
        "section",
        "introduction",
        "conclusion",
        "overview",
        "outline",
        "agenda",
        "summary",
        "resume",
        "plan",
    }


def _is_noisy_course_label(value: str) -> bool:
    if _is_artifact_label(value):
        return True
    normalized = _norm_label(value)
    if not normalized or _is_generic_course_label(normalized):
        return True
    if len(normalized) > 120:
        return True
    if re.search(r"\b(page|slide|logo|navigation|footer|copyright|thank you|questions?)\b", normalized):
        return True
    if re.search(r"\b(plan de|table des mati|table of contents|contents|agenda)\b", normalized):
        return True
    if re.search(r"\b(university|universite|ecole|school|college|faculty|faculte|professor|enseignant)\b", normalized):
        return True
    if normalized.startswith(("master ", "degree ", "program ")):
        return True
    if re.search(r"\b(people who bought|recommended to|read by her)\b", normalized):
        return True
    if re.fullmatch(r"[\d\W_]+", normalized):
        return True
    return False


def _clean_heading(raw: str) -> str:
    heading = _strip_artifacts(raw).lstrip("-•0123456789. ").strip()
    if ">" in heading:
        parts = [part.strip() for part in heading.split(">") if part.strip()]
        for part in reversed(parts):
            if not _is_noisy_course_label(part):
                return part
        return parts[-1] if parts else heading
    return heading


def _strip_artifacts(value: str) -> str:
    text = re.sub(
        r"</?(?:th|td|tr|table|tbody|thead|mark|b)\b[^>]*>?",
        " ",
        str(value),
        flags=re.IGNORECASE,
    )
    text = text.replace("</th", " ").replace("</td", " ").replace("</tr", " ")
    text = text.replace("<th", " ").replace("<td", " ").replace("<tr", " ")
    text = text.replace("**", "").replace("__", "")
    return re.sub(r"\s+", " ", text).strip(" -:;,.")


def _ascii_norm(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or "").casefold())
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def _is_artifact_label(value: str) -> bool:
    text = str(value or "").strip()
    folded = _ascii_norm(text)
    if not folded:
        return True
    if re.search(r"</?(?:th|td|tr|table|tbody|thead|mark|b)\b|</(?:th|td|tr|mark|b)?$", folded):
        return True
    if any(token in text for token in ("\\begin", "\\end", "$", "\\vec", "\\frac", "_{", "^", "{", "}")):
        return True
    if re.search(r"\b(?:section path|section summary|source file|key details|formal pieces)\b", folded):
        return True
    if re.fullmatch(r"(?:ratings|analysis of ratings|input from user|output to user)", folded):
        return True
    if re.search(r"^\s*(?:doc|position|rang|rank|groupe|group|facteur|factor|systeme|system|etape|step)\s+\w+", folded):
        return True
    if re.search(r"^\s*\d+(?:\.\d+)*\s*[:.)-]?\s+", folded):
        return True
    if re.search(r"\b(?:sim\(|num\s*:|den\s*:|calculer|note\s+\d|classement parfait)\b", folded):
        return True
    return False


def _extract_course_topics(chunks: list[Chunk], *, limit: int = 6) -> list[CourseTopic]:
    module_chunks = [
        c
        for c in chunks
        if c.metadata.get("context_type") == "mindmap_module_pack"
        or re.search(r"(?im)^Module\s+\d+\s*:", c.text)
    ]
    outline_text = "\n\n".join(c.text for c in module_chunks)
    if not outline_text:
        outline_text = "\n\n".join(c.text for c in chunks if c.source in {"course_outline", "course"})
    if not outline_text:
        outline_text = "\n\n".join(c.text for c in chunks)

    module_matches = list(re.finditer(r"(?im)^Module\s+\d+\s*:\s*(.+?)\s*$", outline_text))
    topics: list[CourseTopic] = []
    seen_titles: set[str] = set()

    for index, match in enumerate(module_matches):
        start = match.end()
        end = module_matches[index + 1].start() if index + 1 < len(module_matches) else len(outline_text)
        block = outline_text[start:end]
        source_match = re.search(r"(?im)^Source file:\s*(.+?)\s*$", block)
        source = source_match.group(1).strip() if source_match else "course materials"
        headings = [
            _clean_heading(line)
            for line in block.splitlines()
            if line.strip().startswith(("-", "•"))
        ]
        headings = [
            heading
            for heading in headings
            if heading and not _is_noisy_course_label(heading)
        ]
        if not headings:
            module_title = _clean_heading(match.group(1))
            if module_title and not _is_noisy_course_label(module_title):
                headings = [module_title]

        module_title = _clean_heading(match.group(1))
        title = module_title if module_title and not _is_noisy_course_label(module_title) else ""
        if not title:
            title = headings[0] if headings else module_title
        if not title or _is_noisy_course_label(title):
            title = source.removesuffix(".pdf")
        key = _norm_label(title)
        if key in seen_titles:
            for heading in headings[1:]:
                heading_key = _norm_label(heading)
                if heading_key not in seen_titles:
                    title = heading
                    key = heading_key
                    break
        if key in seen_titles:
            continue
        seen_titles.add(key)
        details = headings[:4] if _norm_label(title) == _norm_label(module_title) else headings[1:4]
        topics.append(CourseTopic(title=title, source=source, details=details))
        if len(topics) >= limit:
            break

    if topics:
        return topics

    fallback_headings: list[CourseTopic] = []
    for chunk in chunks:
        for raw_line in chunk.text.splitlines():
            line = _clean_heading(raw_line)
            if not line or _is_noisy_course_label(line) or len(line) > 90:
                continue
            key = _norm_label(line)
            if key in seen_titles:
                continue
            seen_titles.add(key)
            fallback_headings.append(CourseTopic(title=line, source=chunk.source, details=[]))
            if len(fallback_headings) >= limit:
                return fallback_headings
    return fallback_headings


def _infer_course_title(chunks: list[Chunk], topics: list[CourseTopic]) -> str:
    for chunk in chunks:
        if chunk.source in {"course_outline", "course"}:
            match = re.search(r"(?im)^(?:Course|Central topic|Topic):\s*(.+?)\s*$", chunk.text)
            if match and not _is_generic_course_label(match.group(1)):
                return match.group(1).strip()
    return topics[0].title if topics else "this course"


def _infer_course_title(chunks: list[Chunk], topics: list[CourseTopic]) -> str:
    candidates: list[str] = []
    metadata_titles: list[str] = []
    for chunk in chunks:
        if chunk.source in {"course_outline", "course"}:
            match = re.search(r"(?im)^(?:Course|Central topic|Topic):\s*(.+?)\s*$", chunk.text)
            if match and not _is_noisy_course_label(match.group(1)):
                candidates.append(_clean_heading(match.group(1)))
        title = str(chunk.metadata.get("document_title") or "").strip()
        if title:
            metadata_titles.append(title)
            candidates.append(title)
        for raw_line in chunk.text.splitlines()[:12]:
            line = _clean_heading(raw_line)
            if line and not _is_noisy_course_label(line):
                candidates.append(line)
    for topic in topics:
        candidates.extend([topic.title, *topic.details])

    metadata_counts: dict[str, tuple[str, int]] = {}
    for title in metadata_titles:
        label = _clean_heading(title)
        if _is_noisy_course_label(label):
            continue
        key = _norm_label(label)
        original, count = metadata_counts.get(key, (label, 0))
        metadata_counts[key] = (original, count + 1)
    repeated_metadata = sorted(metadata_counts.values(), key=lambda item: item[1], reverse=True)
    if repeated_metadata and repeated_metadata[0][1] >= 2:
        return repeated_metadata[0][0]
    if topics and not _is_noisy_course_label(topics[0].title):
        return topics[0].title

    counts: dict[str, tuple[str, int]] = {}
    for candidate in candidates:
        label = _clean_heading(candidate)
        if _is_noisy_course_label(label):
            continue
        key = _norm_label(label)
        original, count = counts.get(key, (label, 0))
        counts[key] = (original, count + 1)
    ranked = sorted(counts.values(), key=lambda item: (item[1], len(item[0])), reverse=True)
    if ranked:
        return ranked[0][0]
    return topics[0].title if topics else "this course"


def _collect_overview_concepts(topics: list[CourseTopic], chunks: list[Chunk], *, limit: int = 12) -> list[str]:
    concepts: list[str] = []
    for topic in topics:
        concepts.extend([topic.title, *topic.details])
    for chunk in chunks:
        raw = chunk.metadata.get("key_concepts") or []
        if isinstance(raw, list):
            concepts.extend(str(item) for item in raw)
        for line in chunk.text.splitlines():
            match = re.match(r"\s*(?:Key concepts|concepts):\s*(.+)$", line, flags=re.IGNORECASE)
            if match:
                concepts.extend(part.strip() for part in match.group(1).split(","))

    out: list[str] = []
    seen: set[str] = set()
    for concept in concepts:
        label = _clean_heading(str(concept))
        key = _norm_label(label)
        if _is_noisy_course_label(label) or _is_artifact_label(label) or key in seen:
            continue
        seen.add(key)
        out.append(label)
        if len(out) >= limit:
            break
    return out


def _collect_overview_formulas(chunks: list[Chunk], *, limit: int = 5) -> list[str]:
    formulas: list[str] = []
    for chunk in chunks:
        formulas.extend(_extract_formula_snippets(chunk.text, limit=2))
        if len(formulas) >= limit:
            break
    out: list[str] = []
    seen: set[str] = set()
    for formula in formulas:
        key = re.sub(r"\s+", " ", formula)
        if key in seen:
            continue
        seen.add(key)
        out.append(formula)
        if len(out) >= limit:
            break
    return out


def _course_overview_response(chunks: list[Chunk]) -> str | None:
    topics = _extract_course_topics(chunks)
    if not topics:
        return None

    title = _infer_course_title(chunks, topics)
    concepts = _collect_overview_concepts(topics, chunks)
    lines = [
        f"Here is the big picture: this course is about **{title}**, based on the uploaded files.",
        "",
        "## Main path through the course",
        "",
    ]
    for index, topic in enumerate(topics, start=1):
        detail = ""
        if topic.details:
            detail = ": " + ", ".join(topic.details)
        lines.append(f"{index}. **{topic.title}**{detail}.")

    source_names = []
    for topic in topics:
        if topic.source not in source_names:
            source_names.append(topic.source)
    if concepts:
        lines.extend(["", "## Key ideas to learn", "", ", ".join(concepts[:12]) + "."])

    first_steps = [topic.title for topic in topics[:3]]
    if first_steps:
        lines.extend(
            [
                "",
                "## What to study first",
                "",
                "Start with " + " -> ".join(first_steps) + ". After each part, ask me for a tiny example or a quiz so you can check understanding.",
            ]
        )
    lines.extend(
        [
            "",
            "By the end, you should be able to explain the main concepts, connect the modules together, and use the formulas, procedures, examples, or cases that appear in the files.",
            "",
            f"Sources used: {', '.join(source_names[:6])}.",
        ]
    )
    return "\n".join(lines)


def _formula_cards(chunks: list[Chunk], query: str) -> list[FormulaCard]:
    query_tokens = {
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_@]*", query.casefold())
        if len(token) > 2
    }
    cards: list[FormulaCard] = []
    for chunk in chunks:
        formulas = _extract_formula_snippets(chunk.text)
        definitions = _extract_symbol_definitions(chunk.text)
        for formula in formulas:
            formula_key = formula.casefold()
            score = sum(1 for token in query_tokens if token in formula_key)
            if query_tokens and score == 0 and not any(
                token in chunk.text.casefold() for token in query_tokens
            ):
                continue
            cards.append(FormulaCard(formula=formula, source=chunk.source, definitions=definitions))
    return cards


def _extract_symbol_definitions(text: str, *, limit: int = 6) -> list[str]:
    definitions: list[str] = []
    pattern = re.compile(
        r"(?:^|\s)[\-•]\s*(\$[^$]+\$)\s*=\s*(.+?)(?=\s+[\-•]\s*\$|\n|$)",
        flags=re.DOTALL,
    )
    for match in pattern.finditer(text):
        symbol = " ".join(match.group(1).split())
        meaning = " ".join(match.group(2).split()).strip(" .;")
        if symbol and meaning:
            definitions.append(f"{symbol}: {meaning}")

    seen: set[str] = set()
    out: list[str] = []
    for item in definitions:
        if item in seen:
            continue
        seen.add(item)
        out.append(item[:240])
        if len(out) >= limit:
            break
    return out


def _formula_response(cards: list[FormulaCard]) -> str:
    primary = cards[0]
    lines = [
        "Yes — your uploaded material gives the formula directly.",
        "",
        "## Formula",
        primary.formula,
        f"[source: {primary.source}]",
    ]
    if primary.definitions:
        lines.extend(
            [
                "",
                "## Symbols",
                *[f"- `{definition.split(':', 1)[0].strip('$')}` means {definition.split(':', 1)[1].strip()} [source: {primary.source}]" for definition in primary.definitions],
            ]
        )
    lines.extend(
        [
            "",
            "In plain language: use this formula in the context shown by the source chunk. Ask me for a worked example if you want to practice the same notation step by step.",
        ]
    )
    return "\n".join(lines)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def run(inp: GeneratorInput) -> AsyncIterator[str]:
    settings = get_settings()
    options = dict(inp.options or {})
    set_current_llm_options(options)
    llm = get_llm_service()

    set_current_language(options.get("language"))

    history = inp.chat_history or []
    learner = inp.learner_state

    analysis = await analyze_query(
        user_message=inp.user_message,
        chat_history=history,
        learner_state=learner.model_dump(),
        llm=llm,
    )
    mode = select_mode(analysis, learner)

    yield _sse(
        "analysis",
        {
            "intent": analysis.intent,
            "confusion_level": analysis.confusion_level,
            "targets_concept": analysis.targets_concept,
            "mode": mode,
        },
    )

    ranked_chunks = list(inp.context_chunks or [])[: settings.max_context_chunks]

    yield _sse(
        "sources",
        {
            "sources": [
                {
                    "text": c.text,
                    "source": c.source,
                    "score": c.score,
                    "chunk_id": c.chunk_id,
                }
                for c in ranked_chunks
            ]
        },
    )

    is_course_overview = _is_course_overview_question(inp.user_message)
    top_score = ranked_chunks[0].score if ranked_chunks else float("-inf")
    context_has_evidence = _has_context_evidence(inp.user_message, ranked_chunks)
    if (
        top_score < settings.min_relevance_score
        and not context_has_evidence
        and not (is_course_overview and ranked_chunks)
    ):
        refusal = (
            "That question doesn't appear to be covered in the course materials you've "
            "uploaded, so I can't answer it from your sources. I stay grounded in your "
            "uploaded files — want to explore a topic from them instead?"
        )
        yield _sse("token", {"delta": refusal})
        yield _sse(
            "done",
            {
                "response": refusal,
                "generator_id": settings.generator_id,
                "output_type": settings.output_type,
                "artifacts": [],
                "sources": [],
                "learner_updates": LearnerUpdates().model_dump(),
                "metadata": {
                    "mode": "refuse",
                    "analysis": analysis.model_dump(),
                    "confidence": 1.0,
                    "context_ranker": "backend",
                    "refused_reason": "off_topic",
                    "top_score": top_score,
                },
            },
        )
        return

    formula_cards = _formula_cards(ranked_chunks, inp.user_message)
    if _is_formula_only_question(inp.user_message) and formula_cards:
        response = _formula_response(formula_cards)
        yield _sse("token", {"delta": response})
        yield _sse(
            "done",
            {
                "response": response,
                "generator_id": settings.generator_id,
                "output_type": settings.output_type,
                "artifacts": [],
                "sources": [c.model_dump() for c in ranked_chunks],
                "learner_updates": LearnerUpdates(
                    concepts_covered=[formula_cards[0].formula]
                ).model_dump(),
                "metadata": {
                    "mode": mode,
                    "analysis": analysis.model_dump(),
                    "confidence": {
                        "groundedness": 1.0,
                        "coverage": 1.0,
                        "overall": 1.0,
                        "label": "high",
                        "chunks_used": len(ranked_chunks),
                    },
                    "context_ranker": "backend",
                    "formula_answer": True,
                },
            },
        )
        return

    overview_response = _course_overview_response(ranked_chunks) if is_course_overview else None
    if overview_response:
        yield _sse("token", {"delta": overview_response})
        yield _sse(
            "done",
            {
                "response": overview_response,
                "generator_id": settings.generator_id,
                "output_type": settings.output_type,
                "artifacts": [],
                "sources": [c.model_dump() for c in ranked_chunks],
                "learner_updates": LearnerUpdates(
                    concepts_covered=[topic.title for topic in _extract_course_topics(ranked_chunks)]
                ).model_dump(),
                "metadata": {
                    "mode": mode,
                    "analysis": analysis.model_dump(),
                    "confidence": {
                        "groundedness": 1.0,
                        "coverage": 0.9,
                        "overall": 0.95,
                        "label": "high",
                        "chunks_used": len(ranked_chunks),
                    },
                    "context_ranker": "backend",
                    "course_overview_answer": True,
                },
            },
        )
        return

    system = build_chat_system_prompt(
        _MODE_PROMPT_FILE[mode],
        context=_format_chunks(ranked_chunks),
        understood_concepts=", ".join(learner.understood_concepts) or "(none yet)",
        struggling_concepts=", ".join(learner.struggling_concepts) or "(none)",
        user_message=inp.user_message,
    )

    response_parts: list[str] = []
    async for delta in llm.stream_reply(
        system=system,
        chat_history=history,
        user_message=inp.user_message,
    ):
        response_parts.append(delta)
        yield _sse("token", {"delta": delta})

    full_response = "".join(response_parts)

    confidence = await compute_confidence(
        response=full_response,
        chunks=ranked_chunks,
        query=inp.user_message,
    )

    extraction = await extract_learner_updates(
        user_message=inp.user_message,
        assistant_response=full_response,
        llm=llm,
        known_concepts=[item.model_dump() for item in inp.learner_state.known_concepts],
    )
    learner_updates = LearnerUpdates(
        concepts_covered=extraction.covered,
        concepts_demonstrated=extraction.demonstrated_understanding,
        concepts_struggled=extraction.showed_confusion,
    )

    yield _sse(
        "done",
        {
            "response": full_response,
            "generator_id": settings.generator_id,
            "output_type": settings.output_type,
            "artifacts": [],
            "sources": [c.model_dump() for c in ranked_chunks],
            "learner_updates": learner_updates.model_dump(),
            "metadata": {
                "mode": mode,
                "analysis": analysis.model_dump(),
                "confidence": confidence,
                "context_ranker": "backend",
                "llm_fallback": llm.last_chat_used_fallback,
                "llm_fallback_reason": llm.last_chat_fallback_reason,
            },
        },
    )
