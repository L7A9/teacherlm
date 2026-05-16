import json
import re
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
        if not re.search(r"[=∑√\\]|\\frac|\\sum|\\sqrt|\\hat|RMSE|MSE|MAE|DCG|nDCG", line):
            continue
        if not re.search(r"\$|\\|[=∑√^_]", line):
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


def _is_formula_question(message: str) -> bool:
    return bool(
        re.search(
            r"\b(equation|formula|formule|calculate|calculer|math|rmse|mse|mae|dcg|ndcg)\b",
            message,
            flags=re.IGNORECASE,
        )
    )


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
            r"starting today|"
            r"de quoi parle|résume.*cours|resume.*cours|explique.*cours)\b",
            normalized,
        )
    )


def _norm_label(value: str) -> str:
    value = value.casefold()
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
        "developpement mobile",
        "développement mobile",
        "developpement mobile sous android",
        "développement mobile sous android",
        "lst sitd",
        "introduction",
        "conclusion",
        "qcm",
    }


def _clean_heading(raw: str) -> str:
    heading = raw.strip().lstrip("-•0123456789. ").strip()
    if ">" in heading:
        parts = [part.strip() for part in heading.split(">") if part.strip()]
        for part in reversed(parts):
            if not _is_generic_course_label(part):
                return part
        return parts[-1] if parts else heading
    return heading


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
            if heading and not _is_generic_course_label(heading)
        ]
        if not headings:
            module_title = _clean_heading(match.group(1))
            if module_title and not _is_generic_course_label(module_title):
                headings = [module_title]

        title = headings[0] if headings else _clean_heading(match.group(1))
        if not title or _is_generic_course_label(title):
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
        topics.append(CourseTopic(title=title, source=source, details=headings[1:4]))
        if len(topics) >= limit:
            break

    if topics:
        return topics

    fallback_headings: list[CourseTopic] = []
    for chunk in chunks:
        for raw_line in chunk.text.splitlines():
            line = _clean_heading(raw_line)
            if not line or _is_generic_course_label(line) or len(line) > 90:
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
    topic_text = " ".join(
        part
        for topic in topics
        for part in [topic.title, *topic.details]
    )
    topic_text_normalized = _norm_label(topic_text)
    if (
        "android" in topic_text_normalized
        or "androidmanifest" in topic_text_normalized
        or "sqlite" in topic_text_normalized
        or "shared preferences" in topic_text_normalized
    ):
        return "Android mobile application development"
    if "recommand" in topic_text_normalized or "recommend" in topic_text_normalized:
        return "recommendation systems"
    return topics[0].title if topics else "this course"


def _course_overview_response(chunks: list[Chunk]) -> str | None:
    topics = _extract_course_topics(chunks)
    if not topics:
        return None

    title = _infer_course_title(chunks, topics)
    lines = [
        f"This course is about **{title}**.",
        "",
        "If you're starting today, think of it as a path through these main parts:",
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
    lines.extend(
        [
            "",
            "A good first goal is to understand the big picture, then go module by module and connect each concept to a small example from the course files.",
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
            "In plain language: this formula measures how far the predicted values are from the real values. In your recommendation-system materials, it is used for rating prediction, but the course also warns that rating-error metrics alone can miss ranking quality in recommendation lists.",
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
    if top_score < settings.min_relevance_score and not (is_course_overview and ranked_chunks):
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
    if _is_formula_question(inp.user_message) and formula_cards:
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
            },
        },
    )
