from __future__ import annotations

import html
import json
import re
import unicodedata
from collections import Counter
from collections.abc import AsyncIterator
from typing import Any

from teacherlm_core.llm.providers import LLMMessage, complete_text
from teacherlm_core.schemas import (
    Chunk,
    GeneratorInput,
    GeneratorManifest,
    GeneratorOutput,
    LearnerUpdates,
)

from local_api.db import get_store
from local_api.services.artifacts import get_artifact_service
from local_api.services.settings import get_settings_service


GeneratorEvent = dict[str, Any]

_BLOOM = ("remember", "understand", "apply", "analyze")
_QUESTION_WORDS = {"what", "how", "why", "explain", "describe", "define", "teach", "summarize", "compare"}
_STOPWORDS = {
    "about",
    "after",
    "also",
    "because",
    "course",
    "does",
    "each",
    "file",
    "from",
    "have",
    "into",
    "lesson",
    "material",
    "more",
    "should",
    "source",
    "student",
    "teacher",
    "their",
    "there",
    "these",
    "this",
    "those",
    "through",
    "upload",
    "using",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
}

_QUIZ_ADMIN_TERMS = {
    "abdelaaziz",
    "author",
    "auteur",
    "contents",
    "course",
    "document",
    "ecole",
    "ens",
    "file",
    "guide",
    "hessane",
    "lecture",
    "master",
    "meknes",
    "moulay",
    "novembre",
    "prof",
    "professeur",
    "resume",
    "school",
    "seance",
    "semaine",
    "students",
    "table",
    "universite",
}

_QUIZ_GENERIC_LABELS = {
    "conclusion",
    "introduction",
    "plan",
    "plan de la seance",
    "resume",
    "table des matieres",
}

_QUIZ_WEAK_SINGLE_TERMS = {
    "calculer",
    "comment",
    "cours",
    "crucial",
    "etape",
    "expose",
    "guide",
    "choix",
    "informationnelle",
    "intelligents",
    "less",
    "mesurez",
    "message",
    "normale",
    "notre",
    "objectif",
    "optimize",
    "page",
    "page_number",
    "position",
    "principe",
    "optimisez",
    "recommande",
    "recommandee",
    "recommandé",
    "recommandée",
    "repository",
    "resume",
    "table",
    "surcharge",
    "taux",
    "voici",
    "vivons",
}

_QUIZ_ALLOWED_CATEGORIES = {
    "definition",
    "relationship",
    "mechanism",
    "causality",
    "application",
    "classification",
}

_QUIZ_BANNED_QUESTION_RE = re.compile(
    r"\b("
    r"according to|author|instructor|professor|professeur|auteur|"
    r"this course|the course|course material|the reading|the lecture|the document|the file|"
    r"chapter|page|slide|mentioned|stated|wrote|published|semester|year|version|"
    r"which statement|which of the following is mentioned|who wrote|what year|in what year"
    r")\b",
    re.I,
)

_QUIZ_EXCLUSION_RE = re.compile(
    r"(\b[A-Z]{2,4}\s?\d{2,4}\b|https?://|www\.|[\w.-]+\.(?:pdf|pptx?|docx?|txt|md)\b|"
    r"\b(?:semester|syllabus|grading|deadline|due date|prerequisite|office hours|bibliography|references|uploaded)\b)",
    re.I,
)

_QUIZ_NOISE_LINE_RE = re.compile(
    r"(\b(?:universit[ée]|ecole|école|school|faculty|department|master|semester|syllabus|"
    r"prof\.?|professeur|instructor|author|auteur|office hours|grading|deadline|due date|"
    r"bibliography|references|table des mati[eè]res|page\s*\d+)\b|"
    r"\b\d{4}\b|[\w.-]+\.(?:pdf|pptx?|docx?|txt|md)\b|https?://)",
    re.I,
)

_QUIZ_TECHNICAL_TERM_RE = re.compile(
    r"\b(?:"
    r"RMSE|nDCG|NDCG|DCGk|DCG|SVD|TF-IDF|CBF|CF|NCF|Top-N|"
    r"Pr[ée]cision@k|Rappel@k|Recall@k|"
    r"filtrage collaboratif|filtrage bas[ée] sur le contenu|syst[èe]me de recommandation|"
    r"surcharge informationnelle|m[ée]trique(?:s)? d[’']?[ée]valuation|"
    r"m[ée]trique(?:s)? de classement|feedback explicite|feedback implicite|"
    r"taux de compl[ée]tude|"
    r"similarit[ée] cosinus|factorisation de matrices|mod[èe]les latents|"
    r"diversit[ée]|s[ée]rendipit[ée]|sur-apprentissage|overfitting|underfitting|"
    r"gradient descent|descente de gradient|activation function|fonction d[’']activation|"
    r"neural network|r[ée]seau neuronal|apprentissage supervis[ée]|apprentissage non supervis[ée]"
    r")\b",
    re.I,
)


class GeneratorService:
    def list_manifests(self, *, only_enabled: bool = False) -> list[GeneratorManifest]:
        rows = get_store().query("SELECT * FROM generator_registry ORDER BY id ASC")
        manifests = [GeneratorManifest.model_validate(json.loads(row["manifest_json"])) for row in rows]
        if only_enabled:
            manifests = [manifest for manifest in manifests if manifest.enabled]
        return manifests

    def manifest_for_output(self, output_type: str) -> GeneratorManifest:
        for manifest in self.list_manifests(only_enabled=True):
            if manifest.output_type == output_type:
                return manifest
        raise KeyError(f"no enabled generator for output type {output_type}")

    def chat_default(self) -> GeneratorManifest:
        for manifest in self.list_manifests(only_enabled=True):
            if manifest.is_chat_default:
                return manifest
        return self.manifest_for_output("text")

    async def run(
        self,
        manifest: GeneratorManifest,
        payload: GeneratorInput,
    ) -> AsyncIterator[GeneratorEvent]:
        if manifest.transport != "local_inprocess":
            yield _event(
                "error",
                {
                    "message": f"{manifest.display_name} is registered as {manifest.transport}; MCP execution is not enabled yet.",
                    "generator_id": manifest.generator_id,
                },
            )
            return

        match manifest.generator_id:
            case "teacher_gen":
                async for event in _teacher(payload):
                    yield event
            case "quiz_gen":
                async for event in _quiz(payload):
                    yield event
            case "mindmap_gen":
                async for event in _mindmap(payload):
                    yield event
            case "podcast_gen":
                async for event in _podcast(payload):
                    yield event
            case _:
                yield _event("error", {"message": f"unknown in-process generator {manifest.generator_id}"})


async def _teacher(payload: GeneratorInput) -> AsyncIterator[GeneratorEvent]:
    chunks = payload.context_chunks
    mode = _teacher_mode(payload.user_message, payload.learner_state.struggling_concepts)
    analysis = {
        "intent": _intent(payload.user_message),
        "confusion_level": _confusion_level(payload.user_message),
        "targets_concept": _target_concept(payload.user_message, chunks),
        "mode": mode,
    }
    yield _event("analysis", analysis)
    yield _event("sources", [chunk.model_dump() for chunk in chunks])

    if not chunks:
        response = (
            "That is not covered by the course material I have for this conversation yet. "
            "Upload or select the relevant source file and I will ground the explanation in it."
        )
        output = GeneratorOutput(
            response=response,
            generator_id="teacher_gen",
            output_type="text",
            sources=[],
            learner_updates=LearnerUpdates(concepts_struggled=_concepts_from_text(payload.user_message)),
            metadata={"mode": "refuse", "analysis": analysis, "confidence": _confidence([], grounded=False)},
        )
        yield _event("token", response)
        yield _event("done", output.model_dump())
        return

    formula_cards = _formula_cards(chunks, payload.user_message)
    if _is_formula_question(payload.user_message) and formula_cards:
        response = _formula_response(formula_cards[0])
    elif _is_course_overview_question(payload.user_message):
        response = _course_overview_response(chunks)
    else:
        response = await _llm_or_fallback(
            system=_teacher_system_prompt(mode, chunks, payload),
            user=payload.user_message,
            fallback=lambda: _teacher_fallback_response(payload.user_message, chunks, mode),
            history=payload.chat_history,
        )

    concepts = _concepts_from_chunks(chunks) or _concepts_from_text(payload.user_message)
    output = GeneratorOutput(
        response=response,
        generator_id="teacher_gen",
        output_type="text",
        sources=chunks,
        learner_updates=LearnerUpdates(concepts_covered=concepts[:8]),
        metadata={
            "mode": mode,
            "analysis": analysis,
            "confidence": _confidence(chunks, grounded=True),
            "context_ranker": "local_platform_like",
        },
    )
    yield _event("token", response)
    yield _event("done", output.model_dump())


async def _quiz(payload: GeneratorInput) -> AsyncIterator[GeneratorEvent]:
    chunks = payload.context_chunks
    yield _event("progress", {"stage": "extracting_concepts", "chunks": len(chunks)})
    knowledge_items = _quiz_knowledge_items(chunks)
    concepts = _dedupe(item["concept"] for item in knowledge_items if item.get("concept")) or _quiz_concepts_from_chunks(chunks) or _concepts_from_text(payload.user_message)
    yield _event("progress", {"stage": "concepts_extracted", "count": len(concepts)})

    target = _question_count(payload.options)
    kinds = _question_kinds(payload.options)
    plan = _quiz_plan(concepts, target, kinds)
    yield _event("progress", {"stage": "planned", "total": len(plan), "counts": dict(Counter(slot["kind"] for slot in plan))})

    questions, dropped = _build_quiz_questions(plan, chunks, knowledge_items)
    bloom_counts = dict(Counter(q["bloom_level"] for q in questions))
    yield _event("progress", {"stage": "validated", "kept": len(questions), "dropped": dropped, "top_up": 0})

    if not questions:
        response = "I found the sources, but I could not make grounded quiz questions from the selected material."
        output = GeneratorOutput(
            response=response,
            generator_id="quiz_gen",
            output_type="quiz",
            sources=chunks,
            metadata={"reason": "no_questions_after_validation", "plan": {"slots": plan}, "dropped_questions": dropped},
        )
        yield _event("token", response)
        yield _event("done", output.model_dump())
        return

    intro = _quiz_intro(payload.learner_state.struggling_concepts, payload.learner_state.understood_concepts, bloom_counts)
    quiz_data = {
        "title": _quiz_title(payload.options, concepts),
        "intro_message": intro,
        "questions": questions,
        "bloom_distribution": bloom_counts,
    }
    artifact = get_artifact_service().create_artifact(
        payload.conversation_id,
        "quiz",
        "quiz.json",
        quiz_data,
        mime_type="application/json",
    )
    response = f"{intro}\n\nSee the quiz below."
    output = GeneratorOutput(
        response=response,
        generator_id="quiz_gen",
        output_type="quiz",
        artifacts=[artifact],
        sources=chunks,
        learner_updates=LearnerUpdates(concepts_covered=sorted({q["concept"] for q in questions if q.get("concept")})),
        metadata={
            "quiz_data": quiz_data,
            "plan": {"slots": plan, "total": len(plan), "counts": dict(Counter(slot["kind"] for slot in plan))},
            "bloom_distribution": bloom_counts,
            "dropped_questions": dropped,
            "top_up_questions": 0,
        },
    )
    yield _event("artifact", artifact.model_dump())
    yield _event("token", response)
    yield _event("done", output.model_dump())


async def _mindmap(payload: GeneratorInput) -> AsyncIterator[GeneratorEvent]:
    chunks = payload.context_chunks
    yield _event("progress", {"stage": "starting", "chunks": len(chunks)})
    mindmap = _build_mindmap(chunks, payload.user_message, payload.options)
    yield _event(
        "progress",
        {
            "stage": "hierarchy_built",
            "node_count": _count_mindmap_nodes(mindmap),
            "branches": [branch["text"] for branch in mindmap["branches"]],
        },
    )
    markdown = _mindmap_markdown(mindmap)
    mindmap["markdown"] = markdown
    json_artifact = get_artifact_service().create_artifact(
        payload.conversation_id,
        "mindmap",
        "mindmap.json",
        mindmap,
        mime_type="application/json",
    )
    html_artifact = get_artifact_service().create_artifact(
        payload.conversation_id,
        "html",
        "mindmap.html",
        _mindmap_html(mindmap),
        mime_type="text/html",
    )
    response = (
        f"I built a grounded mind map of '{mindmap['central_topic']}' from your materials. "
        f"It covers {len(mindmap['branches'])} main themes and {_count_mindmap_nodes(mindmap)} concepts."
    )
    output = GeneratorOutput(
        response=response,
        generator_id="mindmap_gen",
        output_type="mindmap",
        artifacts=[json_artifact, html_artifact],
        sources=chunks[:10],
        learner_updates=LearnerUpdates(concepts_covered=_mindmap_labels(mindmap)),
        metadata={
            "markdown": markdown,
            "node_count": _count_mindmap_nodes(mindmap),
            "depth": _mindmap_depth(mindmap),
            "central_topic": mindmap["central_topic"],
            "main_branches": [branch["text"] for branch in mindmap["branches"]],
        },
    )
    yield _event("artifact", json_artifact.model_dump())
    yield _event("artifact", html_artifact.model_dump())
    yield _event("token", response)
    yield _event("done", output.model_dump())


async def _podcast(payload: GeneratorInput) -> AsyncIterator[GeneratorEvent]:
    chunks = [chunk for chunk in payload.context_chunks if chunk.text.strip()]
    duration = str(payload.options.get("duration") or "short")
    language = str(payload.options.get("language") or "en")
    yield _event(
        "progress",
        {"stage": "starting", "chunks": len(chunks), "duration": duration, "language": language, "backend": "transcript"},
    )

    if not chunks:
        response = "I could not pull enough material from your sources to script a podcast."
        output = GeneratorOutput(
            response=response,
            generator_id="podcast_gen",
            output_type="podcast",
            sources=[],
            metadata={"reason": "no_context"},
        )
        yield _event("token", response)
        yield _event("done", output.model_dump())
        return

    yield _event("progress", {"stage": "extracting_arc"})
    arc = _narrative_arc(chunks, payload.user_message)
    yield _event("progress", {"stage": "arc_ready", "title": arc["title"], "key_points": len(arc["key_points"])})
    yield _event("progress", {"stage": "scripting", "duration": duration})
    script = _podcast_script(arc, chunks, payload.options)
    transcript = _podcast_transcript(script)
    yield _event("progress", {"stage": "scripted", "segments": len(script["segments"]), "word_count": len(transcript.split())})
    artifact = get_artifact_service().create_artifact(
        payload.conversation_id,
        "transcript",
        "podcast_transcript.txt",
        transcript,
        mime_type="text/plain",
    )
    response = (
        f"I drafted a {len(transcript.split())}-word two-host podcast script for \"{script['title']}\". "
        "Local audio synthesis is not configured here, so I attached the transcript."
    )
    output = GeneratorOutput(
        response=response,
        generator_id="podcast_gen",
        output_type="podcast",
        artifacts=[artifact],
        sources=chunks,
        learner_updates=LearnerUpdates(concepts_covered=arc["key_points"]),
        metadata={
            "podcast": {
                "title": script["title"],
                "summary": script["summary"],
                "duration_ms": 0,
                "word_count": len(transcript.split()),
                "segment_count": len(script["segments"]),
                "transcript": transcript,
                "used_fallback_tts": False,
                "tts_skipped": True,
            },
            "narrative_arc": arc,
            "duration_choice": duration,
            "language": language,
            "voices": {"backend": "transcript", "single_voice": True},
        },
    )
    yield _event("artifact", artifact.model_dump())
    yield _event("token", response)
    yield _event("done", output.model_dump())


async def _llm_or_fallback(
    *,
    system: str,
    user: str,
    fallback: Any,
    history: list[dict] | None = None,
) -> str:
    provider = get_settings_service().get_default_chat_provider_config()
    if provider is None:
        return fallback()
    messages = [LLMMessage(role="system", content=system)]
    for item in (history or [])[-8:]:
        role = str(item.get("role") or "")
        if role in {"user", "assistant"} and item.get("content"):
            messages.append(LLMMessage(role=role, content=str(item["content"])[:3000]))
    messages.append(LLMMessage(role="user", content=user))
    try:
        return await complete_text(provider, messages, temperature=0.2)
    except Exception:
        return fallback()


def _teacher_system_prompt(mode: str, chunks: list[Chunk], payload: GeneratorInput) -> str:
    context = "\n\n".join(_format_chunk(index + 1, chunk) for index, chunk in enumerate(chunks[:8]))
    understood = ", ".join(payload.learner_state.understood_concepts) or "(none yet)"
    struggling = ", ".join(payload.learner_state.struggling_concepts) or "(none yet)"
    return (
        "You are TeacherLM, a warm encouraging teacher for students. Answer only from the uploaded course context. "
        "If the answer is not in the context, say that clearly. Use concise markdown, explain step by step, "
        "and cite source names/chunk ids in-line when making claims.\n\n"
        f"Response mode: {mode}\nUnderstood concepts: {understood}\nStruggling concepts: {struggling}\n\n"
        f"COURSE CONTEXT:\n{context}"
    )


def _format_chunk(index: int, chunk: Chunk) -> str:
    heading = chunk.metadata.get("heading_path") or chunk.metadata.get("section_title") or ""
    concepts = ", ".join(str(item) for item in chunk.metadata.get("key_concepts", [])[:6])
    return (
        f"[{index}] source={chunk.source} chunk_id={chunk.chunk_id} score={chunk.score:.3f}\n"
        f"section={heading}\nconcepts={concepts}\n{chunk.text[:2500]}"
    )


def _teacher_fallback_response(question: str, chunks: list[Chunk], mode: str) -> str:
    concepts = _concepts_from_chunks(chunks)
    lines = [
        f"Question: {question.strip()}" if question.strip() else "Here is the grounded explanation.",
        "",
        f"I will use a **{mode}** approach and stay inside the uploaded sources.",
        "",
    ]
    if concepts:
        lines.extend(["Key ideas: " + ", ".join(concepts[:6]) + ".", ""])
    for index, chunk in enumerate(chunks[:4], start=1):
        heading = chunk.metadata.get("heading_path") or chunk.metadata.get("section_title") or chunk.source
        excerpt = " ".join(chunk.text.split())[:360]
        lines.append(f"{index}. **{heading}**: {excerpt} [source: {chunk.source}, chunk: {chunk.chunk_id}]")
    lines.extend(["", "Try explaining the first point back to me in your own words, and I will check it against the sources."])
    return "\n".join(lines)


def _teacher_mode(message: str, struggling: list[str]) -> str:
    normalized = message.casefold()
    if any(word in normalized for word in ("quiz me", "test me", "ask me")):
        return "quiz_back"
    if any(word in normalized for word in ("confused", "stuck", "don't understand", "dont understand", "help")) or struggling:
        return "guide"
    if any(word in normalized for word in ("yes", "i understand", "got it")) and len(normalized.split()) <= 8:
        return "affirm"
    return "explain"


def _intent(message: str) -> str:
    lowered = message.casefold()
    if _is_course_overview_question(message):
        return "course_overview"
    if _is_formula_question(message):
        return "formula_lookup"
    if any(word in lowered for word in ("quiz", "test", "practice")):
        return "practice"
    if any(word in lowered for word in ("confused", "stuck", "help")):
        return "remediation"
    return "explain"


def _confusion_level(message: str) -> float:
    lowered = message.casefold()
    score = 0.0
    for token in ("confused", "stuck", "lost", "hard", "don't understand", "dont understand"):
        if token in lowered:
            score += 0.25
    return min(1.0, score)


def _target_concept(message: str, chunks: list[Chunk]) -> str | None:
    query_terms = set(_tokens(message))
    for concept in _concepts_from_chunks(chunks):
        if set(_tokens(concept)) & query_terms:
            return concept
    return None


def _confidence(chunks: list[Chunk], *, grounded: bool) -> dict[str, Any]:
    if not grounded:
        return {"groundedness": 0.0, "coverage": 0.0, "overall": 0.0, "label": "none", "chunks_used": 0}
    coverage = min(1.0, len(chunks) / 6)
    groundedness = 0.8 if chunks else 0.0
    overall = round((coverage + groundedness) / 2, 3)
    label = "high" if overall >= 0.75 else "medium" if overall >= 0.45 else "low"
    return {"groundedness": groundedness, "coverage": coverage, "overall": overall, "label": label, "chunks_used": len(chunks)}


def _is_formula_question(message: str) -> bool:
    return bool(
        re.search(r"\b(formula|equation|calculate|compute|derive|symbol|math|formule|equation)\b", message, re.I)
        or re.search(r"\b[A-Za-z]\w*\s*(?:=|\+|\*|/|\^)\s*[A-Za-z0-9\\$]", message)
    )


def _formula_cards(chunks: list[Chunk], query: str) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    query_terms = set(_tokens(query))
    for chunk in chunks:
        formulas = _extract_formulas(chunk.text)
        if not formulas:
            continue
        definitions = _symbol_definitions(chunk.text)
        for formula in formulas:
            haystack = f"{formula} {chunk.text}".casefold()
            if query_terms and not any(term in haystack for term in query_terms):
                continue
            cards.append({"formula": formula, "source": chunk.source, "chunk_id": chunk.chunk_id, "definitions": definitions})
    return cards


def _formula_response(card: dict[str, Any]) -> str:
    lines = ["Yes, your uploaded material gives this formula directly.", "", "## Formula", card["formula"], f"[source: {card['source']}, chunk: {card['chunk_id']}]"]
    if card["definitions"]:
        lines.extend(["", "## Symbols"])
        lines.extend(f"- {item}" for item in card["definitions"][:6])
    lines.extend(["", "In plain language: use the formula in the context shown by that source chunk."])
    return "\n".join(lines)


def _extract_formulas(text: str, limit: int = 5) -> list[str]:
    out: list[str] = []
    out.extend(f"$$ {' '.join(match.split())} $$" for match in re.findall(r"\$\$(.+?)\$\$", text, re.S))
    for line in text.splitlines():
        stripped = line.strip()
        if re.search(r"\\(?:frac|sum|sqrt|int|prod)|[∑√∫±×÷≤≥≈∞]", stripped) or re.search(r"\b[A-Za-z]\w*\s*=\s*[-+*/^().,\w\s]+$", stripped):
            out.append(stripped)
    return _dedupe(out)[:limit]


def _symbol_definitions(text: str) -> list[str]:
    out: list[str] = []
    for match in re.finditer(r"(?:^|\n)\s*[-*]\s*(\$?[^:=\n]{1,20}\$?)\s*[:=]\s*(.+)", text):
        out.append(f"{match.group(1).strip()}: {match.group(2).strip()[:180]}")
    return _dedupe(out)


def _is_course_overview_question(message: str) -> bool:
    normalized = re.sub(r"[^\w\s]", " ", message.casefold())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return bool(
        re.search(
            r"\b(what is this course about|what are these files about|course overview|summarize this course|summarise this course|"
            r"explain this course|teach me this course|where should i start|what should i study first|prepare me for the exam|"
            r"de quoi parle|resume.*cours|explique.*cours)\b",
            normalized,
        )
    )


def _course_overview_response(chunks: list[Chunk]) -> str:
    groups = _group_chunks_by_section(chunks)
    title = groups[0]["title"] if groups else "this course"
    concepts = _concepts_from_chunks(chunks)
    lines = [f"Here is the big picture: this course is about **{title}**, based on the uploaded files.", "", "## Main path through the course", ""]
    for index, group in enumerate(groups[:8], start=1):
        details = ", ".join(group["concepts"][:4])
        suffix = f": {details}" if details else ""
        lines.append(f"{index}. **{group['title']}**{suffix}.")
    if concepts:
        lines.extend(["", "## Key ideas to learn", "", ", ".join(concepts[:12]) + "."])
    if groups:
        lines.extend(["", "## What to study first", "", "Start with " + " -> ".join(group["title"] for group in groups[:3]) + "."])
    lines.extend(["", "Sources used: " + ", ".join(_dedupe(chunk.source for chunk in chunks)[:8]) + "."])
    return "\n".join(lines)


def _question_count(options: dict[str, Any]) -> int:
    raw = options.get("question_count") or options.get("n_questions") or options.get("count") or 5
    try:
        return max(1, min(int(raw), 20))
    except (TypeError, ValueError):
        return 5


def _question_kinds(options: dict[str, Any]) -> list[str]:
    raw = options.get("question_types") or options.get("question_type") or options.get("types") or options.get("kinds") or ["mcq"]
    if isinstance(raw, str):
        raw = [raw]
    out = []
    for item in raw:
        key = str(item).strip().lower().replace("-", "_")
        if key in {"mcq", "multiple_choice", "multi_choice"}:
            out.append("mcq")
        elif key in {"true_false", "truefalse", "tf"}:
            out.append("true_false")
    return out or ["mcq"]


def _quiz_concepts_from_chunks(chunks: list[Chunk]) -> list[str]:
    section_concepts: list[str] = []
    keyword_concepts: list[str] = []
    for chunk in chunks:
        metadata = chunk.metadata
        for label in _metadata_labels(metadata, "section_title", "heading_path"):
            if _is_assessable_concept(label):
                section_concepts.append(label)
        for concept in metadata.get("key_concepts", []):
            label = _clean_label(str(concept))
            if _is_assessable_concept(label):
                keyword_concepts.append(label)

    concepts = [*section_concepts, *keyword_concepts]
    if not concepts:
        for chunk in chunks:
            concepts.extend(label for label in _important_phrases(chunk.text) if _is_assessable_concept(label))
    return _dedupe(concepts)[:24]


def _metadata_labels(metadata: dict[str, Any], *keys: str) -> list[str]:
    labels: list[str] = []
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, list):
            labels.extend(str(item) for item in value)
        elif value:
            raw = str(value)
            if ">" in raw:
                labels.extend(part.strip() for part in raw.split(">") if part.strip())
            else:
                labels.append(raw)
    return labels


def _is_assessable_concept(label: str) -> bool:
    clean = _normalize_quiz_concept(label)
    if not clean:
        return False
    if re.search(r"\.(?:pdf|pptx?|docx?|txt|md)\b", clean.casefold()):
        return False
    normalized = _quiz_norm(clean)
    if normalized.startswith(("guide complet", "course title", "titre du cours")):
        return False
    if normalized in {"systemes de recommandation et blockchain", "systèmes de recommandation et blockchain"}:
        return False
    if normalized in _QUIZ_GENERIC_LABELS:
        return False
    if any(generic in normalized for generic in _QUIZ_GENERIC_LABELS) and len(normalized.split()) <= 3:
        return False
    tokens = [token for token in re.findall(r"[a-z0-9]+", normalized) if token not in _STOPWORDS]
    if not tokens:
        return False
    if len(tokens) > 6:
        return False
    if normalized.startswith(("ce guide", "this guide", "message pour", "peut ")):
        return False
    if normalized.startswith(("when ", "lorsque ", "problem when", "en education", "en éducation")):
        return False
    if len(tokens) == 1 and tokens[0] in _QUIZ_WEAK_SINGLE_TERMS:
        return False
    admin_hits = sum(token in _QUIZ_ADMIN_TERMS for token in tokens)
    if admin_hits and admin_hits >= max(1, len(tokens) - 1):
        return False
    if len(tokens) <= 2 and admin_hits:
        return False
    return True


def _quiz_plan(concepts: list[str], total: int, kinds: list[str]) -> list[dict[str, str]]:
    concepts = concepts or ["the selected material"]
    return [
        {
            "concept": concepts[index % len(concepts)],
            "bloom_level": _BLOOM[index % len(_BLOOM)],
            "kind": kinds[index % len(kinds)],
        }
        for index in range(total)
    ]


def _quiz_knowledge_items(chunks: list[Chunk]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for chunk in chunks:
        clean_text = _strip_quiz_noise(chunk.text)
        for statement in _statement_candidates(clean_text):
            item = _knowledge_item_from_statement(statement, chunk)
            if item is not None:
                items.append(item)
    return _dedupe_quiz_items(items)


def _knowledge_item_from_statement(statement: str, chunk: Chunk) -> dict[str, Any] | None:
    if _contains_quiz_exclusion(statement):
        return None
    category = _quiz_category(statement)
    if category not in _QUIZ_ALLOWED_CATEGORIES:
        return None
    concept = _quiz_concept_from_statement(statement, chunk)
    if not concept:
        return None
    related = _related_quiz_concept(statement, concept)
    answer = _clean_quiz_statement(statement)
    if not _is_quiz_answer_option(answer):
        return None
    return {
        "category": category,
        "concept": concept,
        "related_concept": related,
        "statement": answer,
        "answer": answer,
        "source_chunk_id": chunk.chunk_id,
        "source": chunk.source,
        "metadata": chunk.metadata,
    }


def _quiz_category(statement: str) -> str:
    normalized = _quiz_norm(statement)
    if re.search(r"\$\$|\\(?:sum|frac|sqrt)|\b[A-Za-z]\w*\s*=", statement):
        return "mechanism"
    if re.search(r"\b(?:types?|cat[ée]gories?|taxonomy|taxonomie|classes?|approches?)\b", normalized):
        return "classification"
    if re.search(r"\b(?:because|cause|causes|caused|why|therefore|explains why|entra[iî]ne|provoque|cause|car|parce que|afin de|permet de)\b", normalized):
        return "causality"
    if re.search(r"\b(?:step|steps|process|processus|m[ée]canisme|calcul|formula|formule|fonctionne|optimise|applique|utilise|mesure)\b", normalized):
        return "mechanism"
    if re.search(r"\b(?:compare|diff[ée]rence|differs?|versus|vs|whereas|while|alors que|contrairement|relation|relates?|hybride|combine)\b", normalized):
        return "relationship"
    if re.search(r"\b(?:scenario|sc[ée]nario|example|exemple|case|cas|given|lorsque|si vous|when)\b", normalized):
        return "application"
    if re.search(r"\b(?:is|are|means|refers to|represents|defined as|est|sont|d[ée]signe|correspond|se d[ée]finit)\b", normalized):
        return "definition"
    return "definition"


def _build_quiz_questions(
    plan: list[dict[str, str]],
    chunks: list[Chunk],
    knowledge_items: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    questions: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    items = knowledge_items or _quiz_knowledge_items(chunks)
    used_item_keys: set[str] = set()
    used_question_keys: set[str] = set()
    for index, slot in enumerate(plan):
        question = _next_valid_quiz_question(items, slot, index, used_item_keys, used_question_keys)
        if question is None:
            dropped.append({"slot": slot, "reason": "no_valid_knowledge_item"})
            continue
        question["id"] = f"q{len(questions) + 1}"
        questions.append(question)
    return questions, dropped


def _next_valid_quiz_question(
    items: list[dict[str, Any]],
    slot: dict[str, str],
    index: int,
    used_item_keys: set[str],
    used_question_keys: set[str],
) -> dict[str, Any] | None:
    for item in _rank_quiz_items(items, slot):
        key = _norm(f"{item.get('concept')} {item.get('answer')}")
        if key in used_item_keys:
            continue
        question = _quiz_question_from_item(item, slot["kind"], index, items)
        if question is None:
            continue
        question_key = _norm(str(question.get("question", "")))
        if not question_key or question_key in used_question_keys:
            continue
        used_item_keys.add(key)
        used_question_keys.add(question_key)
        return question
    return None


def _rank_quiz_items(items: list[dict[str, Any]], slot: dict[str, str]) -> list[dict[str, Any]]:
    slot_terms = set(_tokens(slot.get("concept", "")))
    target_bloom = slot.get("bloom_level", "understand")
    target_index = _BLOOM.index(target_bloom) if target_bloom in _BLOOM else 1
    return sorted(
        items,
        key=lambda item: (
            -sum(term in _quiz_norm(str(item.get("concept", ""))) for term in slot_terms),
            abs(_BLOOM.index(_bloom_for_quiz_category(str(item.get("category", "definition")))) - target_index),
            str(item.get("concept", "")),
        ),
    )


def _quiz_question_from_item(
    item: dict[str, Any],
    kind: str,
    index: int,
    items: list[dict[str, Any]],
) -> dict[str, Any] | None:
    bloom_level = _bloom_for_quiz_category(str(item.get("category", "definition")))
    concept = _quiz_display_concept(str(item.get("concept", "")))
    if kind == "true_false":
        answer = index % 2 == 0
        statement = str(item["statement"]) if answer else _false_statement_for_item(item)
        question = {
            "id": "",
            "type": "true_false",
            "category": item["category"],
            "bloom_level": bloom_level,
            "question": f"True or false: {statement}",
            "options": ["True", "False"],
            "answer": answer,
            "explanation": _quiz_explanation(str(item["statement"]), item),
            "concept": concept,
            "source_chunk_id": item["source_chunk_id"],
        }
        return question if _validate_quiz_question(question) else None

    distractors = _distractors_for_item(item, items)
    options, correct_index = _ordered_options(str(item["answer"]), distractors, index)
    question = {
        "id": "",
        "type": "mcq",
        "category": item["category"],
        "bloom_level": bloom_level,
        "question": _quiz_question_stem_for_item(item),
        "options": options,
        "correct_index": correct_index,
        "explanation": _quiz_explanation(str(item["statement"]), item),
        "concept": concept,
        "source_chunk_id": item["source_chunk_id"],
    }
    return question if _validate_quiz_question(question) else None


def _strip_quiz_noise(text: str) -> str:
    lines: list[str] = []
    skipping_tail = False
    for raw_line in str(text or "").splitlines():
        line = " ".join(raw_line.split()).strip()
        if not line:
            continue
        normalized = _quiz_norm(line)
        if re.fullmatch(r"\d{1,4}", normalized):
            continue
        if re.match(r"^(?:references|bibliography|bibliographie|webographie)\b", normalized):
            skipping_tail = True
            continue
        if skipping_tail:
            continue
        if _QUIZ_NOISE_LINE_RE.search(line) and len(line.split()) <= 18:
            continue
        lines.append(line)
    return "\n".join(lines)


def _contains_quiz_exclusion(text: str) -> bool:
    return bool(_QUIZ_EXCLUSION_RE.search(str(text or "")))


def _quiz_concept_from_statement(statement: str, chunk: Chunk) -> str:
    candidates: list[str] = []
    for match in _QUIZ_TECHNICAL_TERM_RE.finditer(statement):
        candidates.append(match.group(0))
    candidates.extend(_pattern_concepts_from_statement(statement))
    for concept in chunk.metadata.get("key_concepts", []):
        label = _normalize_quiz_concept(str(concept))
        if _is_quiz_key_concept_candidate(label) and _concept_matches_statement(label, statement):
            candidates.append(label)
    candidates.extend(_metadata_labels(chunk.metadata, "section_title", "heading_path"))

    for candidate in candidates:
        concept = _normalize_quiz_concept(candidate)
        if _is_assessable_concept(concept):
            return concept
    return ""


def _concept_matches_statement(concept: str, statement: str) -> bool:
    concept_terms = set(_tokens(concept))
    statement_terms = set(_tokens(statement))
    if not concept_terms:
        return False
    return bool(concept_terms & statement_terms) or _quiz_norm(concept) in _quiz_norm(statement)


def _is_quiz_key_concept_candidate(label: str) -> bool:
    if not _is_assessable_concept(label):
        return False
    tokens = _tokens(label)
    if len(tokens) > 1:
        return True
    clean = _quiz_display_concept(label)
    if _QUIZ_TECHNICAL_TERM_RE.fullmatch(clean):
        return True
    return bool(re.search(r"[A-Z].*[A-Z]|\d|@|-", clean))


def _pattern_concepts_from_statement(statement: str) -> list[str]:
    text = _clean_quiz_statement(statement)
    patterns = [
        r"\b(?:choix de la bonne|bonne|mauvaise)\s+([^.;:]{4,70})",
        r"\b(?:si vous mesurez|mesurer|mesure)\s+([A-Za-z0-9@À-ÿ_\-]{2,40})",
        r"\b(?:le|la|les|un|une|des)\s+([^.;:]{4,80}?)\s+(?:est|sont|désigne|correspond|permet|mesure|combine|optimise)\b",
        r"\b([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9@_\- ]{3,70}?)\s+(?:is|are|means|refers to|represents|allows|measures)\b",
    ]
    out: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.I):
            out.append(match.group(1))
    return out


def _related_quiz_concept(statement: str, concept: str) -> str:
    terms = [term for term in _technical_terms(statement) if _norm(term) != _norm(concept)]
    if terms:
        return terms[0]
    for candidate in _pattern_concepts_from_statement(statement):
        normalized = _normalize_quiz_concept(candidate)
        if normalized and _norm(normalized) != _norm(concept) and _is_assessable_concept(normalized):
            return normalized
    return ""


def _technical_terms(text: str) -> list[str]:
    out = [match.group(0) for match in _QUIZ_TECHNICAL_TERM_RE.finditer(str(text or ""))]
    for match in re.finditer(r"\b[A-Z][A-Za-z0-9]*(?:-[A-Z][A-Za-z0-9]*)?(?:@[a-z])?\b", str(text or "")):
        value = match.group(0)
        looks_technical = bool(re.search(r"[A-Z].*[A-Z]|\d|@|-", value))
        if looks_technical and len(value) > 1 and _is_assessable_concept(value):
            out.append(value)
    return _dedupe(out)


def _normalize_quiz_concept(label: str) -> str:
    clean = _clean_label(str(label or ""))
    clean = re.sub(r"\bDCGk\b", "DCG", clean)
    clean = re.sub(r"\(\s*\)", "", clean)
    clean = re.sub(r"^\d+(?:\.\d+)*\s*", "", clean)
    clean = re.sub(r"^(?:semaine|week)\s+\d+\s*[:\-]\s*", "", clean, flags=re.I)
    clean = re.sub(r"^(?:le|la)\s+probl[èe]me\s*[:\-]\s*", "", clean, flags=re.I)
    clean = re.sub(r"^qu[’']?est[- ]ce qu[’']?(?:un|une)\s+", "", clean, flags=re.I)
    clean = clean.strip(" ?:-")
    return re.sub(r"\s+", " ", clean)


def _dedupe_quiz_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for item in items:
        key = _norm(f"{item.get('category')} {item.get('concept')} {item.get('answer')}")
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _distractors_for_item(item: dict[str, Any], items: list[dict[str, Any]]) -> list[str]:
    correct = str(item.get("answer", ""))
    concept_terms = set(_tokens(str(item.get("concept", ""))))
    same_chunk: list[str] = []
    same_source: list[str] = []
    same_category: list[str] = []
    related: list[str] = []
    fallback: list[str] = []
    for other in items:
        answer = str(other.get("answer", ""))
        if _norm(answer) == _norm(correct) or not _is_quiz_answer_option(answer):
            continue
        other_terms = set(_tokens(f"{other.get('concept', '')} {answer}"))
        if other.get("source_chunk_id") == item.get("source_chunk_id"):
            same_chunk.append(answer)
        elif other.get("source") == item.get("source"):
            same_source.append(answer)
        elif concept_terms & other_terms:
            related.append(answer)
        elif other.get("category") == item.get("category"):
            same_category.append(answer)
        else:
            fallback.append(answer)
    distractors = _dedupe([*same_chunk, *same_source, *related, *same_category, *_plausible_misconceptions(item), *fallback])
    return [option for option in distractors if _norm(option) != _norm(correct)][:3]


def _plausible_misconceptions(item: dict[str, Any]) -> list[str]:
    concept = _quiz_display_concept(str(item.get("concept", "")))
    match str(item.get("category", "definition")):
        case "relationship":
            return [
                f"{concept} removes the need to compare alternatives or assumptions.",
                f"{concept} treats all related variables as interchangeable.",
                f"{concept} has no effect on the interpretation of outcomes.",
            ]
        case "mechanism":
            return [
                f"{concept} works automatically without inputs, constraints, or evaluation.",
                f"{concept} skips intermediate steps and directly guarantees the final result.",
                f"{concept} depends only on labels rather than the underlying data or process.",
            ]
        case "causality":
            return [
                f"{concept} occurs randomly and is not linked to any underlying condition.",
                f"{concept} is caused only by naming conventions.",
                f"{concept} has the same effect regardless of context.",
            ]
        case "classification":
            return [
                f"{concept} belongs to every category equally.",
                f"{concept} is classified only by its filename or position.",
                f"{concept} has no category because it is just an identifier.",
            ]
        case "application":
            return [
                f"{concept} should be applied without checking assumptions or trade-offs.",
                f"{concept} is only useful when no decision has to be made.",
                f"{concept} guarantees the best outcome in every scenario.",
            ]
        case _:
            return [
                f"{concept} is only a label and has no operational meaning.",
                f"{concept} means the same thing as every related term.",
                f"{concept} is defined by where it appears rather than what it explains.",
            ]


def _bloom_for_quiz_category(category: str) -> str:
    match category:
        case "definition":
            return "remember"
        case "classification" | "relationship" | "mechanism":
            return "understand"
        case "application":
            return "apply"
        case "causality":
            return "analyze"
        case _:
            return "understand"


def _quiz_question_stem_for_item(item: dict[str, Any]) -> str:
    concept = _quiz_display_concept(str(item.get("concept", "")))
    related = _quiz_display_concept(str(item.get("related_concept", "")))
    match str(item.get("category", "definition")):
        case "definition":
            return f"What best defines {concept}?"
        case "relationship":
            if related and _norm(related) != _norm(concept):
                return f"How does {concept} relate to {related}?"
            return f"What relationship is central to {concept}?"
        case "mechanism":
            return f"What happens when {concept} is applied?"
        case "causality":
            return f"Why does {concept} occur or matter?"
        case "application":
            return f"A system must make a decision involving {concept}. Which principle best applies?"
        case "classification":
            return f"Which category best describes {concept}?"
        case _:
            return f"What best explains {concept}?"


def _false_statement_for_item(item: dict[str, Any]) -> str:
    concept = _quiz_display_concept(str(item.get("concept", "")))
    match str(item.get("category", "definition")):
        case "relationship":
            return f"{concept} has no relationship to any other concept or decision."
        case "mechanism":
            return f"{concept} works without inputs, intermediate steps, or constraints."
        case "causality":
            return f"{concept} occurs without any cause, condition, or trade-off."
        case "classification":
            return f"{concept} belongs to every category in exactly the same way."
        case "application":
            return f"{concept} should be applied without checking context or assumptions."
        case _:
            return f"{concept} is only a name and has no conceptual meaning."


def _validate_quiz_question(question: dict[str, Any]) -> bool:
    text = str(question.get("question", ""))
    if not text or _BANNED_QUESTION(text):
        return False
    if question.get("category") not in _QUIZ_ALLOWED_CATEGORIES:
        return False
    if question.get("type") == "mcq":
        options = question.get("options", [])
        if not isinstance(options, list) or len(options) != 4:
            return False
        if len({_norm(option) for option in options}) != 4:
            return False
        if any(not _is_quiz_answer_option(str(option)) for option in options):
            return False
    if question.get("type") == "true_false":
        return "True or false:" in text and not _contains_quiz_exclusion(text)
    return question.get("type") == "mcq"


def _BANNED_QUESTION(text: str) -> bool:
    return bool(_QUIZ_BANNED_QUESTION_RE.search(text) or _contains_quiz_exclusion(text))


def _is_quiz_answer_option(option: str) -> bool:
    clean = _clean_quiz_statement(option)
    if len(clean) < 18:
        return False
    if _looks_like_table_or_figure(clean):
        return False
    if _contains_quiz_exclusion(clean):
        return False
    if _QUIZ_NOISE_LINE_RE.search(clean) and len(clean.split()) <= 18:
        return False
    if re.search(r"\bposition\b.*\b(?:note|relevance|pertinence)\b", clean, re.I):
        return False
    if re.match(r"^(?:voici|here are|here is)\b", clean, re.I):
        return False
    if re.search(r"\b(?:ce guide|this guide|guide d[’']?[ée]valuation|message pour)\b", clean, re.I):
        return False
    if re.search(r"\b(?:according to|author|instructor|professor|professeur|chapter|page|slide|document|file|uploaded)\b", clean, re.I):
        return False
    return True


def _looks_like_table_or_figure(text: str) -> bool:
    clean = _clean_quiz_statement(text)
    if re.search(r"^\[?image\b|^figure\b|^table\b", clean, re.I):
        return True
    if "|" in clean:
        return True
    if re.search(r"^(?:source|arrows?|a large arrow)\b|\b(?:icon|labeled|labelled)\b", clean, re.I):
        return True
    if re.search(r"\b(?:cas d[’']usage|m[ée]trique recommand[ée]e|vraie note|position\s*\(?i\)?)\b", clean, re.I):
        return True
    if re.search(r"\)[A-Za-zÀ-ÿ]", clean):
        return True
    camel_boundaries = len(re.findall(r"[a-zà-ÿ][A-ZÀ-Ý]", clean))
    if camel_boundaries >= 2 and not re.search(r"\b(?:nDCG|DCG|TF-IDF|Top-N)\b", clean):
        return True
    if len(clean.split()) <= 4 and len(clean) > 35 and not re.search(r"[.;:]", clean):
        return True
    return False


def _best_chunk_for_concept(concept: str, chunks: list[Chunk]) -> Chunk | None:
    if not chunks:
        return None
    terms = set(_tokens(concept))
    ranked = sorted(
        chunks,
        key=lambda chunk: (
            -sum(term in _searchable_text(chunk).casefold() for term in terms),
            -len(chunk.text),
        ),
    )
    return ranked[0]


def _best_statement(chunk: Chunk, concept: str) -> str:
    candidates = _statement_candidates(chunk.text)
    if not candidates:
        fallback = " ".join(chunk.text.split())[:220]
        return fallback if _is_quiz_worthy_statement(fallback) else ""
    terms = set(_tokens(concept))
    return max(candidates, key=lambda item: sum(term in item.casefold() for term in terms))


def _statement_candidates(text: str) -> list[str]:
    out: list[str] = []
    for piece in re.split(r"(?<=[.!?])\s+|\n+", text):
        cleaned = " ".join(piece.split()).strip(" -*#:;")
        cleaned = _clean_quiz_statement(cleaned)
        if len(cleaned) < 24 or cleaned.endswith("?"):
            continue
        if len(cleaned) > 180:
            cleaned = cleaned[:177].rsplit(" ", 1)[0].rstrip(",;:")
        if not cleaned.endswith((".", "!")):
            cleaned += "."
        if _is_quiz_worthy_statement(cleaned):
            out.append(cleaned)
    return _dedupe(out)


def _distractors(chunks: list[Chunk], source_chunk_id: str, correct: str, concept: str) -> list[str]:
    seen = {_norm(correct)}
    out: list[str] = []
    for chunk in [c for c in chunks if c.chunk_id != source_chunk_id] + [c for c in chunks if c.chunk_id == source_chunk_id]:
        for statement in _statement_candidates(chunk.text):
            key = _norm(statement)
            if key in seen:
                continue
            seen.add(key)
            out.append(statement)
            if len(out) >= 3:
                return out
    generic = _generic_quiz_distractors(concept)
    return [*out, *generic][:3]


def _is_quiz_worthy_statement(statement: str) -> bool:
    normalized = _quiz_norm(statement)
    if len(normalized) < 35:
        return False
    if _looks_like_table_or_figure(statement):
        return False
    if normalized.startswith(("master ", "universite ", "ecole ", "ecole normale ", "universite moulay ")):
        return False
    if normalized.startswith(("voici ", "here are ", "here is ")):
        return False
    if re.search(r"\b(?:pr|prof|professeur)\.?\s+[a-z]", normalized):
        return False
    if normalized.startswith(("plan de la seance", "resume", "conclusion")) and len(normalized.split()) < 18:
        return False
    tokens = [token for token in re.findall(r"[a-z0-9]+", normalized) if token not in _STOPWORDS]
    if not tokens:
        return False
    admin_hits = sum(token in _QUIZ_ADMIN_TERMS for token in tokens)
    if admin_hits >= max(2, len(tokens) // 2):
        return False
    return sum(token not in _QUIZ_ADMIN_TERMS for token in tokens) >= 3


def _quiz_question_stem(concept: str, bloom_level: str) -> str:
    display = _quiz_display_concept(concept)
    if _looks_like_section_title(display):
        match bloom_level:
            case "apply":
                return f"Which course idea is applied in {display}?"
            case "analyze":
                return f"What relationship or consequence is emphasized in {display}?"
            case "understand":
                return f"What is the main idea in {display}?"
            case _:
                return f"Which statement belongs to {display}?"
    match bloom_level:
        case "apply":
            return f"How is {display} used in the course material?"
        case "analyze":
            return f"What important relationship or consequence is linked to {display}?"
        case "understand":
            return f"What does the course material explain about {display}?"
        case _:
            return f"Which idea is directly associated with {display}?"


def _looks_like_section_title(concept: str) -> bool:
    normalized = _quiz_norm(concept)
    return (
        ":" in concept
        or "?" in concept
        or bool(re.match(r"^(?:le|la|les|qu|types?|taxonomie|semaine|guide|systemes?)\b", normalized))
        or len(normalized.split()) >= 4
    )


def _ordered_options(correct: str, distractors: list[str], index: int) -> tuple[list[str], int]:
    options = [correct, *distractors]
    while len(options) < 4:
        options.append(f"This choice ignores the key condition needed to answer item {index + 1}.")
    options = options[:4]
    shift = index % 4
    ordered = options[shift:] + options[:shift]
    return ordered, ordered.index(correct)


def _quiz_explanation(statement: str, item_or_chunk: Any) -> str:
    return f"Key idea: {_clean_quiz_statement(statement)}"


def _false_statement_for_concept(concept: str) -> str:
    display = _quiz_display_concept(concept)
    return (
        f"{display} is presented only as document metadata, not as a method, metric, "
        "learning idea, or course concept."
    )


def _generic_quiz_distractors(concept: str) -> list[str]:
    display = _quiz_display_concept(concept)
    return [
        f"{display} is mainly a file-management detail and does not affect the course topic.",
        f"{display} removes the need to compare methods, metrics, or assumptions.",
        f"{display} is unrelated to learners, items, recommendations, evaluation, or decisions.",
    ]


def _clean_quiz_statement(statement: str) -> str:
    clean = html.unescape(str(statement or ""))
    clean = re.sub(r"</?[^>]+>", "", clean)
    clean = re.sub(r"[*_`]+", "", clean)
    clean = re.sub(r"\s+([:;,.!?])", r"\1", clean)
    clean = re.sub(r"\s+", " ", clean)
    return clean.strip(" -:;,.#")


def _quiz_display_concept(concept: str) -> str:
    clean = re.sub(r"\(\s*\)", "", str(concept or ""))
    clean = re.sub(r"^concept\s+cl[ée]\s*:\s*", "", clean, flags=re.I)
    match = re.match(r"^exemple de calcul\s+(.+)$", clean, flags=re.I)
    if match:
        clean = f"{match.group(1).strip()} calculation"
    clean = re.sub(r"\s+", " ", clean).strip(" -:;,.#")
    return clean or "the selected course concept"


def _quiz_norm(text: str) -> str:
    normalized = re.sub(r"[_\-.]+", " ", _strip_accents(str(text or "")))
    return re.sub(r"\s+", " ", normalized.casefold()).strip()


def _quiz_intro(struggling: list[str], understood: list[str], bloom_counts: dict[str, int]) -> str:
    if struggling:
        return f"Let's check the concepts that may need practice, especially {', '.join(struggling[:3])}."
    if understood:
        return f"Let's reinforce what you already started learning: {', '.join(understood[:3])}."
    mix = ", ".join(f"{key}: {value}" for key, value in bloom_counts.items())
    return f"Let's test what you know with a source-grounded quiz ({mix})."


def _quiz_title(options: dict[str, Any], concepts: list[str]) -> str:
    topic = str(options.get("topic") or "").strip()
    if topic:
        return f"Quiz: {topic}"
    if concepts:
        return f"Quiz: {concepts[0]}"
    return "Quiz"


def _build_mindmap(chunks: list[Chunk], prompt: str, options: dict[str, Any]) -> dict[str, Any]:
    max_nodes = int(options.get("max_nodes") or 110)
    mindmap = _mindmap_from_module_packs(chunks, max_nodes=max_nodes)
    if mindmap is None:
        mindmap = _mindmap_from_heading_chunks(chunks, prompt, max_nodes=max_nodes)
    if mindmap is None:
        mindmap = _mindmap_from_concepts(chunks, prompt, max_nodes=max_nodes)
    return _balance_mindmap(_refine_mindmap(mindmap), max_nodes=max_nodes)


def _mindmap_from_module_packs(chunks: list[Chunk], *, max_nodes: int) -> dict[str, Any] | None:
    modules = [
        chunk
        for chunk in chunks
        if chunk.metadata.get("context_type") == "mindmap_module_pack" and chunk.text.strip()
    ]
    if not modules:
        return None
    modules.sort(
        key=lambda chunk: (
            int(chunk.metadata.get("document_order", 0) or 0),
            str(chunk.metadata.get("source_filename", chunk.source)),
        )
    )
    main_modules = [
        module
        for module in modules
        if str(module.metadata.get("document_role", "main")).casefold() != "supporting"
    ] or modules

    branch_budget = min(7, len(main_modules))
    leaf_budget = max(4, (max_nodes - 1 - branch_budget) // max(1, branch_budget))
    branches: list[dict[str, Any]] = []
    module_titles: list[str] = []
    for module in main_modules[:branch_budget]:
        title = _best_mindmap_module_title(module)
        module_titles.append(title)
        children = _mindmap_module_study_nodes(module.text, title, max_children=leaf_budget)
        branch_label = _short_mindmap_label(_compact_mindmap_module_title(title), max_len=80)
        if _is_noisy_mindmap_label(branch_label):
            continue
        branches.append({"text": branch_label, "children": children})

    _merge_supporting_mindmap_modules(branches, modules, branch_budget=branch_budget)

    if len(branches) < 3:
        headings = _dedupe(
            heading
            for module in modules
            for heading in _clean_mindmap_heading_labels(_module_major_headings(module.text), "")
        )
        branches = [{"text": _short_mindmap_label(heading), "children": []} for heading in headings[:10]]
    if len(branches) < 3:
        return None
    central_topic = _infer_mindmap_central_topic(main_modules, module_titles)
    _remove_mindmap_root_repeats(branches, central_topic)
    return {
        "central_topic": central_topic,
        "branches": branches,
    }


def _mindmap_from_heading_chunks(chunks: list[Chunk], prompt: str, *, max_nodes: int) -> dict[str, Any] | None:
    paths: list[list[str]] = []
    details: list[tuple[str, str]] = []
    for chunk in chunks:
        heading = chunk.metadata.get("heading_path_list") or chunk.metadata.get("heading_path") or chunk.metadata.get("section_title")
        if isinstance(heading, list):
            raw_parts = [str(part) for part in heading]
        else:
            raw_parts = re.split(r"\s*>\s*", str(heading or ""))
        parts = [
            _clean_mindmap_label(part)
            for part in raw_parts
            if _clean_mindmap_label(part) and not _is_noisy_mindmap_label(_clean_mindmap_label(part))
        ]
        if parts:
            paths.append(parts[:4])
            details.append((" > ".join(parts), _first_sentence(chunk.text)))

    if not paths:
        return None
    central = _central_topic_from_paths(paths) or _central_topic(chunks, prompt)
    central_key = _mindmap_norm(central)
    trimmed_paths = [[part for part in path if _mindmap_norm(part) != central_key] for path in paths]
    branches = _nodes_from_heading_paths(trimmed_paths, max_children=8)
    _attach_mindmap_details(branches, details)
    branches = [branch for branch in branches if not _is_noisy_mindmap_label(branch["text"])]
    if len(branches) < 3:
        return None
    return {"central_topic": central, "branches": branches[:8]}


def _mindmap_from_concepts(chunks: list[Chunk], prompt: str, *, max_nodes: int) -> dict[str, Any]:
    groups = _group_chunks_by_section(chunks)
    central = _central_topic(chunks, prompt)
    branches: list[dict[str, Any]] = []
    for group in groups[:8]:
        title = _short_mindmap_label(group["title"])
        if _is_noisy_mindmap_label(title):
            continue
        labels = [
            _short_mindmap_label(label)
            for label in [*group["concepts"], *_important_phrases(group["text"])]
            if not _is_noisy_mindmap_label(label)
        ]
        children = [{"text": label, "children": []} for label in _dedupe(labels)[:6]]
        branches.append({"text": title, "children": children})
        if _node_count({"text": central, "children": branches}) >= max_nodes:
            break
    if not branches:
        branches = [
            {"text": concept, "children": []}
            for concept in (_concepts_from_chunks(chunks) or ["Course overview"])[:6]
            if not _is_noisy_mindmap_label(concept)
        ]
    return {"central_topic": central, "branches": branches or [{"text": "Course overview", "children": []}]}


def _best_mindmap_module_title(module: Chunk) -> str:
    metadata_title = str(module.metadata.get("document_title") or "").strip()
    text_title = _module_title(module.text)
    title = metadata_title if metadata_title else text_title
    if _is_wrapper_mindmap_title(title) or _is_generic_mindmap_title(title):
        title = _discover_mindmap_course_title(module.text) or title
    if _is_wrapper_mindmap_title(title) or _is_generic_mindmap_title(title):
        title = _distinctive_mindmap_module_title(module.text) or title
    return title or module.source


def _module_title(text: str) -> str:
    match = re.search(r"(?m)^Module\s+\d+\s*:\s*(.+)$", text)
    return _clean_mindmap_label(match.group(1)) if match else ""


def _module_major_headings(text: str) -> list[str]:
    headings: list[str] = []
    in_headings = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line == "Major headings:":
            in_headings = True
            continue
        if in_headings and line == "Study outline details:":
            break
        if in_headings and line.startswith("- "):
            label = _clean_mindmap_label(line[2:])
            if label and not _is_noisy_mindmap_label(label):
                headings.append(label)
    return _dedupe(headings)


def _mindmap_module_details(text: str) -> list[tuple[str, str]]:
    details: list[tuple[str, str]] = []
    in_details = False
    current_label = ""
    current_detail: list[str] = []

    def flush() -> None:
        nonlocal current_label, current_detail
        if current_label:
            detail = _clean_mindmap_detail(" ".join(current_detail))
            if detail:
                details.append((current_label, detail))
        current_label = ""
        current_detail = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if stripped == "Study outline details:":
            in_details = True
            continue
        if not in_details:
            continue
        if stripped.startswith("- "):
            flush()
            body = stripped[2:]
            label, _, detail = body.partition(":")
            current_label = _clean_mindmap_label(label)
            current_detail = [detail.strip()] if detail.strip() else []
            continue
        if current_label and stripped:
            current_detail.append(stripped)
    flush()
    return [(label, detail) for label, detail in details if label and not _is_noisy_mindmap_label(label)]


def _mindmap_module_study_nodes(text: str, module_title: str, *, max_children: int) -> list[dict[str, Any]]:
    paths = _module_heading_paths(text, module_title)
    details = _mindmap_module_details(text)
    children = _nodes_from_heading_paths(paths, max_children=max_children)
    if children:
        _attach_mindmap_details(children, details)
        return children[:max_children]

    labels = _clean_mindmap_heading_labels(_module_major_headings(text), module_title)
    out: list[dict[str, Any]] = []
    used_details: set[str] = set()
    for label in labels:
        if len(out) >= max_children:
            break
        child_details = _matching_mindmap_details(label, details, used_details, limit=2)
        out.append(
            {
                "text": _short_mindmap_label(label),
                "children": [{"text": _short_mindmap_label(detail), "children": []} for detail in child_details],
            }
        )
    if len(out) < max_children:
        for detail_label, detail in details:
            label = _short_mindmap_label(detail_label or detail)
            key = _mindmap_norm(label)
            if key in used_details or _is_noisy_mindmap_label(label):
                continue
            out.append({"text": label, "children": []})
            used_details.add(key)
            if len(out) >= max_children:
                break
    return out


def _module_heading_paths(text: str, module_title: str) -> list[list[str]]:
    module_keys = {_mindmap_norm(module_title), _mindmap_norm(_compact_mindmap_module_title(module_title))}
    paths: list[list[str]] = []
    for heading in _module_major_headings(text):
        parts: list[str] = []
        for raw_part in re.split(r"\s*>\s*", heading):
            label = _clean_mindmap_label(raw_part)
            key = _mindmap_norm(label)
            if not label or key in module_keys or key in _GENERIC_MINDMAP_BRANCHES:
                continue
            if _is_noisy_mindmap_label(label) or _is_wrapper_mindmap_title(label):
                continue
            parts.append(label)
        if parts:
            paths.append(parts[:4])
    return _dedupe_paths(paths)


def _nodes_from_heading_paths(paths: list[list[str]], *, max_children: int) -> list[dict[str, Any]]:
    roots: list[dict[str, Any]] = []

    def add_child(children: list[dict[str, Any]], label: str, *, limit: int) -> dict[str, Any] | None:
        key = _mindmap_norm(label)
        for child in children:
            if _mindmap_norm(child["text"]) == key:
                return child
        if len(children) >= limit:
            return None
        node = {"text": _short_mindmap_label(label), "children": []}
        children.append(node)
        return node

    for path in paths:
        siblings = roots
        for depth, label in enumerate(path):
            node = add_child(siblings, label, limit=max_children if depth == 0 else 7)
            if node is None:
                break
            siblings = node["children"]
    return roots


def _attach_mindmap_details(nodes: list[dict[str, Any]], details: list[tuple[str, str]]) -> None:
    if not details:
        return
    tree_nodes = _iter_mindmap_nodes(nodes)
    used: set[str] = set()
    for label, detail in details:
        detail_labels = _detail_nodes_from_summary(detail)
        if not detail_labels:
            detail_labels = [label]
        label_tokens = _mindmap_tokens(f"{label} {detail}")
        best_node: dict[str, Any] | None = None
        best_score = 0
        for node in tree_nodes:
            if len(node.get("children", [])) >= 7:
                continue
            score = len(_mindmap_tokens(node["text"]) & label_tokens)
            if score > best_score:
                best_node = node
                best_score = score
        if best_node is None or best_score <= 0:
            continue
        existing = {_mindmap_norm(child["text"]) for child in best_node.get("children", [])}
        for detail_label in detail_labels[:2]:
            clean = _short_mindmap_label(detail_label)
            key = _mindmap_norm(clean)
            if key in used or key in existing or _is_noisy_mindmap_label(clean):
                continue
            best_node.setdefault("children", []).append({"text": clean, "children": []})
            existing.add(key)
            used.add(key)


def _detail_nodes_from_summary(summary: str) -> list[str]:
    labels: list[str] = []
    for match in re.finditer(r"\*\*([^*:]{3,70})\*\*\s*:?", summary):
        labels.append(match.group(1))
    for raw_line in re.split(r"[\n;]", summary):
        line = _clean_mindmap_label(raw_line)
        if not line or _is_noisy_mindmap_label(line):
            continue
        if ":" in line:
            prefix = _clean_mindmap_label(line.split(":", 1)[0])
            if prefix and not _is_noisy_mindmap_label(prefix):
                labels.append(prefix)
        elif re.match(r"^(?:Objectif|Principe|Etape|Avantage|Inconvenient|Solution|Formule)\b", _strip_accents(line), re.IGNORECASE):
            labels.append(line)
        if len(labels) >= 4:
            break
    return _dedupe(_short_mindmap_label(label) for label in labels if label)[:4]


def _iter_mindmap_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    def walk(node: dict[str, Any]) -> None:
        out.append(node)
        for child in node.get("children", []):
            walk(child)

    for node in nodes:
        walk(node)
    return out


def _matching_mindmap_details(label: str, details: list[tuple[str, str]], used: set[str], *, limit: int) -> list[str]:
    label_tokens = _mindmap_tokens(label)
    matches: list[str] = []
    for detail_label, detail in details:
        key = _mindmap_norm(detail_label)
        if key in used or _is_noisy_mindmap_label(detail_label):
            continue
        if label_tokens & _mindmap_tokens(f"{detail_label} {detail}"):
            matches.extend(_detail_nodes_from_summary(detail)[:1] or [detail_label])
            used.add(key)
        if len(matches) >= limit:
            break
    return matches[:limit]


def _merge_supporting_mindmap_modules(branches: list[dict[str, Any]], modules: list[Chunk], *, branch_budget: int) -> None:
    supporting = [
        module
        for module in modules[branch_budget:]
        if str(module.metadata.get("document_role", "main")).casefold() == "supporting"
    ]
    for module in supporting:
        labels = _clean_mindmap_heading_labels(_module_major_headings(module.text), _best_mindmap_module_title(module))
        target = _best_matching_mindmap_branch(branches, labels)
        if target is None:
            continue
        existing = {_mindmap_norm(child["text"]) for child in target.get("children", [])}
        for label in labels:
            key = _mindmap_norm(label)
            if key in existing or _is_noisy_mindmap_label(label):
                continue
            target.setdefault("children", []).append({"text": _short_mindmap_label(label), "children": []})
            existing.add(key)
            if len(target["children"]) >= 9:
                break


def _best_matching_mindmap_branch(branches: list[dict[str, Any]], labels: list[str]) -> dict[str, Any] | None:
    if not branches:
        return None
    label_tokens = _mindmap_tokens(" ".join(labels))
    best = branches[-1]
    best_score = -1
    for branch in branches:
        score = len(label_tokens & _mindmap_tokens(" ".join(_mindmap_node_texts(branch))))
        if score > best_score:
            best = branch
            best_score = score
    return best


def _clean_mindmap_heading_labels(headings: list[str], module_title: str) -> list[str]:
    module_keys = {_mindmap_norm(module_title), _mindmap_norm(_compact_mindmap_module_title(module_title))}
    labels: list[str] = []
    for heading in headings:
        for raw_part in re.split(r"\s*>\s*", heading):
            label = _clean_mindmap_label(raw_part)
            key = _mindmap_norm(label)
            if not label or key in module_keys or key in _GENERIC_MINDMAP_BRANCHES:
                continue
            if _is_noisy_mindmap_label(label) or _is_wrapper_mindmap_title(label):
                continue
            labels.append(label)
    return _dedupe(labels)


def _infer_mindmap_central_topic(modules: list[Chunk], titles: list[str]) -> str:
    root_modules: dict[str, set[str]] = {}
    root_originals: dict[str, str] = {}
    for module_index, module in enumerate(modules):
        for heading in _module_major_headings(module.text):
            parts = [_clean_mindmap_label(part) for part in re.split(r"\s*>\s*", heading)]
            parts = [part for part in parts if part and not _is_noisy_mindmap_label(part)]
            if not parts:
                continue
            root = parts[0]
            key = _mindmap_norm(root)
            if _is_generic_mindmap_root_label(root) or _is_wrapper_mindmap_title(root):
                continue
            root_modules.setdefault(key, set()).add(str(module.metadata.get("source_file_id") or module_index))
            root_originals.setdefault(key, root)
    ranked = sorted(root_modules.items(), key=lambda item: (-len(item[1]), len(root_originals.get(item[0], ""))))
    for key, module_ids in ranked:
        if len(module_ids) >= 2:
            return _short_mindmap_label(root_originals[key], max_len=60)

    cleaned = [_compact_mindmap_module_title(title) for title in titles if title.strip()]
    repeated = _repeated_mindmap_tokens(cleaned)
    if repeated:
        phrase = _phrase_around_repeated_token(cleaned[0], repeated)
        if phrase:
            return _short_mindmap_label(phrase, max_len=60)
    return _short_mindmap_label(cleaned[0], max_len=60) if cleaned else "Course map"


def _remove_mindmap_root_repeats(branches: list[dict[str, Any]], central_topic: str) -> None:
    central_key = _mindmap_norm(central_topic)

    def clean_children(children: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for child in children:
            if _mindmap_norm(child.get("text", "")) == central_key:
                out.extend(clean_children(child.get("children", [])))
                continue
            child["children"] = clean_children(child.get("children", []))
            out.append(child)
        return out

    for branch in branches:
        branch["children"] = clean_children(branch.get("children", []))


def _central_topic_from_paths(paths: list[list[str]]) -> str | None:
    counts: Counter[str] = Counter()
    originals: dict[str, str] = {}
    for path in paths:
        if not path:
            continue
        root = path[0]
        key = _mindmap_norm(root)
        if _is_generic_mindmap_root_label(root) or _is_noisy_mindmap_label(root):
            continue
        counts[key] += 1
        originals.setdefault(key, root)
    for key, count in counts.most_common():
        if count >= 2:
            return _short_mindmap_label(originals[key], max_len=60)
    return None


def _discover_mindmap_course_title(text: str) -> str:
    for heading in _module_major_headings(text):
        parts = [_clean_mindmap_label(part) for part in re.split(r"\s*>\s*", heading)]
        for part in parts:
            if re.search(r"\b(?:semaine|week|lecture|chapter|chapitre|module)\s+\d+", _strip_accents(part), re.IGNORECASE):
                if not _is_noisy_mindmap_label(part):
                    return part
    return ""


def _distinctive_mindmap_module_title(text: str) -> str:
    for heading in _module_major_headings(text):
        parts = [_clean_mindmap_label(part) for part in re.split(r"\s*>\s*", heading)]
        for part in reversed(parts):
            key = _mindmap_norm(part)
            if not key or key in _GENERIC_MINDMAP_BRANCHES or key in _GENERIC_MINDMAP_ROOTS:
                continue
            if _is_generic_mindmap_title(part) or _is_wrapper_mindmap_title(part) or _is_noisy_mindmap_label(part):
                continue
            return part
    return ""


def _compact_mindmap_module_title(title: str) -> str:
    title = _clean_mindmap_label(title)
    title = re.sub(
        r"^(?:semaine|week|lecture|lesson|chapter|chapitre|module|unit|cours)\s+\d+\s*[:\-\u2013\u2014]?\s*",
        "",
        title,
        flags=re.IGNORECASE,
    )
    return title.strip(" -:;") or "Module"


def _is_generic_mindmap_title(title: str) -> bool:
    return _mindmap_norm(title) in {
        "plan de la seance",
        "outline",
        "agenda",
        "introduction",
        "conclusion",
        "course",
        "cours",
        "module",
        "lecture",
    }


def _is_wrapper_mindmap_title(title: str) -> bool:
    key = _mindmap_norm(title)
    if not key:
        return True
    if key.endswith("pdf") or key.endswith("pptx") or key.endswith("docx"):
        return True
    return bool(
        re.fullmatch(
            r"(?:lecture|week|semaine|chapter|chapitre|module|cours)\s*\d+"
            r"(?:\s+(?:v\d+|organized|clean|cleaned|slides|presentation))*",
            key,
        )
    )


def _refine_mindmap(mindmap: dict[str, Any]) -> dict[str, Any]:
    branches: list[dict[str, Any]] = []
    seen_signatures: list[set[str]] = []
    central_key = _mindmap_norm(mindmap.get("central_topic", ""))
    for branch in mindmap.get("branches", []):
        label = _short_mindmap_label(str(branch.get("text", "")))
        if _is_noisy_mindmap_label(label) or _mindmap_norm(label) in _GENERIC_MINDMAP_BRANCHES:
            continue
        branch["text"] = label
        branch["children"] = _refine_mindmap_children(
            branch.get("children", []),
            {_mindmap_norm(label), central_key},
        )
        signature = _mindmap_tokens(" ".join(_mindmap_node_texts(branch)))
        if any(_signature_overlap(signature, prior) >= 0.6 for prior in seen_signatures):
            continue
        branches.append(branch)
        seen_signatures.append(signature)
    if len(branches) >= 3:
        mindmap["branches"] = branches
    mindmap["central_topic"] = _short_mindmap_label(str(mindmap.get("central_topic") or "Course map"), max_len=60)
    return mindmap


def _refine_mindmap_children(
    children: list[dict[str, Any]],
    ancestor_keys: set[str] | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    ancestor_keys = set(ancestor_keys or set())
    for child in children:
        label = _short_mindmap_label(str(child.get("text", "")))
        key = _mindmap_norm(label)
        if not key or key in seen or _is_noisy_mindmap_label(label):
            continue
        refined_children = _refine_mindmap_children(child.get("children", []), ancestor_keys | {key})
        if key in ancestor_keys:
            out.extend(refined_children)
            continue
        seen.add(key)
        out.append({"text": label, "children": refined_children})
    return out


def _balance_mindmap(mindmap: dict[str, Any], *, max_nodes: int) -> dict[str, Any]:
    for branch in mindmap.get("branches", []):
        _promote_single_child_chains(branch)
        _cap_mindmap_children(branch, limit=7)
    mindmap["branches"] = mindmap.get("branches", [])[:7]
    while _count_mindmap_nodes(mindmap) > max_nodes and _trim_one_mindmap_leaf(mindmap):
        pass
    return mindmap


def _promote_single_child_chains(node: dict[str, Any]) -> None:
    for child in node.get("children", []):
        _promote_single_child_chains(child)
    children = node.get("children", [])
    if len(children) == 1 and children[0].get("children"):
        node["children"] = children[0]["children"]


def _cap_mindmap_children(node: dict[str, Any], *, limit: int) -> None:
    node["children"] = node.get("children", [])[:limit]
    for child in node["children"]:
        _cap_mindmap_children(child, limit=limit)


def _trim_one_mindmap_leaf(mindmap: dict[str, Any]) -> bool:
    leaves: list[tuple[int, dict[str, Any], dict[str, Any]]] = []

    def walk(parent: dict[str, Any], depth: int) -> None:
        for child in parent.get("children", []):
            if child.get("children"):
                walk(child, depth + 1)
            else:
                leaves.append((depth + 1, parent, child))

    root = {"children": mindmap.get("branches", [])}
    walk(root, 0)
    if not leaves:
        return False
    leaves.sort(key=lambda item: (-item[0], -len(str(item[2].get("text", "")))))
    _, parent, leaf = leaves[0]
    try:
        parent["children"].remove(leaf)
        return True
    except ValueError:
        return False


def _count_mindmap_nodes(mindmap: dict[str, Any]) -> int:
    return _node_count({"text": mindmap.get("central_topic", "Course map"), "children": mindmap.get("branches", [])})


def _mindmap_node_texts(node: dict[str, Any]) -> list[str]:
    values = [str(node.get("text", ""))]
    for child in node.get("children", []):
        values.extend(_mindmap_node_texts(child))
    return values


def _signature_overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, min(len(a), len(b)))


def _repeated_mindmap_tokens(labels: list[str]) -> set[str]:
    counts: Counter[str] = Counter()
    for label in labels:
        counts.update(_mindmap_tokens(label))
    return {token for token, count in counts.items() if count >= 2}


def _phrase_around_repeated_token(title: str, repeated: set[str]) -> str:
    words = re.findall(r"[\w\u00c0-\u00ff]+", title)
    norm_words = [_mindmap_norm(word) for word in words]
    for index, token in enumerate(norm_words):
        if token not in repeated:
            continue
        start = max(0, index - 2)
        end = min(len(words), index + 3)
        phrase = " ".join(words[start:end]).strip()
        if len(_mindmap_tokens(phrase)) >= 2:
            return phrase
    return ""


def _dedupe_paths(paths: list[list[str]]) -> list[list[str]]:
    seen: set[tuple[str, ...]] = set()
    out: list[list[str]] = []
    for path in paths:
        key = tuple(_mindmap_norm(part) for part in path if _mindmap_norm(part))
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _clean_mindmap_detail(text: str) -> str:
    text = html.unescape(str(text or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    text = _strip_mindmap_media_markers(text)
    text = re.sub(r"\${1,2}.*?\${1,2}", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" -:;")
    if len(text) > 260:
        text = text[:260].rsplit(" ", 1)[0].strip(" -:;")
    return _repair_mindmap_label_punctuation(text)


def _clean_mindmap_label(label: str) -> str:
    label = html.unescape(str(label or ""))
    label = re.sub(r"<[^>]+>", " ", label)
    label = _strip_mindmap_media_markers(label)
    label = re.sub(r"\${1,2}.*?\${1,2}", " ", label)
    label = re.sub(r"[*_`#]+", "", label)
    label = re.sub(r"\s+", " ", label)
    label = re.sub(r"^(?:\d+\.\s*)+", "", label)
    label = re.sub(r"\s+\d{1,3}$", "", label)
    label = label.strip(" -:;,.\u2022\u25b6")
    return _repair_mindmap_label_punctuation(label)


def _strip_mindmap_media_markers(text: str) -> str:
    text = re.sub(r"!\[[^\]]*]\([^)]+\)", " ", text)
    text = re.sub(r"\[\s*Image[^\]]*\]?", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bImage\s*\d*\b", " ", text, flags=re.IGNORECASE)
    return text


def _repair_mindmap_label_punctuation(label: str) -> str:
    label = str(label or "").strip(" -:;,.\u2022\u25b6")
    label = re.sub(r"\(\s*\)", "", label)
    label = re.sub(r"\[\s*]", "", label)
    label = re.sub(r"\s+", " ", label).strip(" -:;,.\u2022\u25b6")
    if label.count("(") > label.count(")"):
        open_index = label.rfind("(")
        if open_index >= max(0, len(label) - 24):
            label = label[:open_index].strip(" -:;,.")
    if label.count("[") > label.count("]"):
        open_index = label.rfind("[")
        if open_index >= max(0, len(label) - 24):
            label = label[:open_index].strip(" -:;,.")
    return label


def _short_mindmap_label(label: str, *, max_len: int = 80) -> str:
    if ">" in str(label or ""):
        parts = [part.strip() for part in str(label).split(">") if part.strip()]
        label = parts[-1] if parts else label
    label = _clean_mindmap_label(label)
    if len(label) <= max_len:
        return label
    cut = label[:max_len].rsplit(" ", 1)[0].strip(" -:;")
    return _repair_mindmap_label_punctuation(cut or label[:max_len].strip(" -:;"))


def _is_noisy_mindmap_label(label: str) -> bool:
    key = _mindmap_norm(label)
    if not key or len(key) < 4:
        return True
    noisy_exact = {
        "abdelaaziz",
        "hessane",
        "master",
        "normale",
        "novembre",
        "moulay ismail",
        "ecole normale superieure",
        "universite moulay ismail",
        "plan de la seance",
        "table des matieres",
        "references et lectures recommandees",
        "references",
        "source material",
        "source plan item",
        "key details",
        "resume",
        "ordre",
        "liste ordonnee",
        "scenario",
        "etape",
        "people who bought",
        "recommended to user",
        "similar users",
        "recommended system",
        "read by user",
        "read by both users",
    }
    if key in noisy_exact:
        return True
    if _is_generic_mindmap_root_label(label):
        return True
    if re.search(r"\ba\s*verifier\b", key):
        return True
    if re.search(r"(?:^|\s)image(?:\s|$)", key):
        return True
    if re.search(r"\b(slide|logo|layout|attribution|copyright|navigation|footer|figure|toolbar)\b", key):
        return True
    if re.fullmatch(r"(?:pr|prof|dr)\s+.+", key):
        return True
    tokens = _mindmap_tokens(label)
    if len(tokens) == 1 and next(iter(tokens)) in {"alice", "bob", "carole", "david", "novembre"}:
        return True
    return len(tokens) < 1


def _mindmap_tokens(text: str) -> set[str]:
    stop = {
        "les",
        "des",
        "dans",
        "pour",
        "avec",
        "une",
        "un",
        "du",
        "de",
        "la",
        "le",
        "et",
        "en",
        "au",
        "aux",
        "sur",
        "par",
        "the",
        "and",
        "for",
        "with",
        "from",
        "into",
        "this",
        "that",
        "cours",
        "lecture",
    }
    return {
        token
        for token in re.findall(r"[\w\u00c0-\u00ff]+", _mindmap_norm(text))
        if len(token) > 2 and token not in stop and not token.isdigit()
    }


def _mindmap_norm(text: str) -> str:
    normalized = re.sub(r"[_\-.]+", " ", _strip_accents(str(text or "")))
    return re.sub(r"\s+", " ", normalized.casefold()).strip()


def _is_generic_mindmap_root_label(label: str) -> bool:
    key = _mindmap_norm(label)
    if key in _GENERIC_MINDMAP_ROOTS:
        return True
    return bool(
        re.search(
            r"\b(?:conclusion|prochaines etapes|synthese|recapitulatif|plan de la seance)\b",
            key,
        )
    )


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text or ""))
    return "".join(char for char in normalized if not unicodedata.combining(char))


_GENERIC_MINDMAP_ROOTS = {
    "course",
    "cours",
    "overview",
    "outline",
    "introduction",
    "conclusion",
    "conclusion et prochaines etapes",
    "synthese",
    "module",
    "lecture",
}


_GENERIC_MINDMAP_BRANCHES = {
    "concepts",
    "applications",
    "autres",
    "other",
    "overview",
    "introduction",
    "conclusion",
    "module",
    "lecture",
    "organized",
    "source material",
}


def _central_topic(chunks: list[Chunk], prompt: str) -> str:
    topic = str(prompt or "").strip()
    if topic and not topic.lower().startswith("generate"):
        return topic[:90]
    groups = _group_chunks_by_section(chunks)
    if groups:
        return groups[0]["title"]
    concepts = _concepts_from_chunks(chunks)
    return concepts[0] if concepts else "Course map"


def _group_chunks_by_section(chunks: list[Chunk]) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        title = str(chunk.metadata.get("section_title") or chunk.metadata.get("heading_path") or chunk.source).split(">")[-1].strip()
        title = _clean_label(title) or chunk.source
        group = groups.setdefault(title.casefold(), {"title": title, "chunks": [], "concepts": [], "text": ""})
        group["chunks"].append(chunk)
        group["text"] += "\n\n" + chunk.text
        for concept in chunk.metadata.get("key_concepts", []):
            label = _clean_label(str(concept))
            if label and label not in group["concepts"]:
                group["concepts"].append(label)
    return list(groups.values())


def _mindmap_markdown(mindmap: dict[str, Any]) -> str:
    lines = [f"# {mindmap['central_topic']}", ""]

    def add_node(node: dict[str, Any], depth: int) -> None:
        lines.append(f"{'  ' * depth}- {node.get('text', 'Topic')}")
        for child in node.get("children", []):
            add_node(child, depth + 1)

    for branch in mindmap["branches"]:
        add_node(branch, 0)
    return "\n".join(lines).strip() + "\n"


def _mindmap_html(mindmap: dict[str, Any]) -> str:
    def render(node: dict[str, Any]) -> str:
        children = node.get("children") or []
        if not children:
            return f"<li>{html.escape(str(node.get('text', 'Topic')))}</li>"
        return f"<li>{html.escape(str(node.get('text', 'Topic')))}<ul>{''.join(render(child) for child in children)}</ul></li>"

    root = {"text": mindmap["central_topic"], "children": mindmap["branches"]}
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>TeacherLM Mind Map</title>"
        "<style>body{font-family:system-ui;margin:32px;line-height:1.5;color:#17202a}li{margin:6px 0}</style>"
        f"</head><body><ul>{render(root)}</ul></body></html>"
    )


def _count_mindmap_nodes(mindmap: dict[str, Any]) -> int:
    return _node_count({"text": mindmap["central_topic"], "children": mindmap["branches"]})


def _node_count(node: dict[str, Any]) -> int:
    return 1 + sum(_node_count(child) for child in node.get("children", []))


def _mindmap_depth(mindmap: dict[str, Any]) -> int:
    def depth(node: dict[str, Any]) -> int:
        children = node.get("children") or []
        return 1 if not children else 1 + max(depth(child) for child in children)

    return depth({"text": mindmap["central_topic"], "children": mindmap["branches"]})


def _mindmap_labels(mindmap: dict[str, Any]) -> list[str]:
    labels = [mindmap["central_topic"]]

    def walk(node: dict[str, Any]) -> None:
        labels.append(str(node.get("text", "")))
        for child in node.get("children", []):
            walk(child)

    for branch in mindmap["branches"]:
        walk(branch)
    return _dedupe(label for label in labels if label)


def _narrative_arc(chunks: list[Chunk], prompt: str) -> dict[str, Any]:
    concepts = _concepts_from_chunks(chunks)
    title = _central_topic(chunks, prompt)
    key_points = concepts[:8] or [group["title"] for group in _group_chunks_by_section(chunks)[:6]]
    intro = _first_chunk_with(chunks, ("introduction", "overview", "definition")) or chunks[0]
    conclusion = _first_chunk_with(chunks, ("conclusion", "summary", "takeaway")) or chunks[-1]
    return {
        "title": title,
        "summary": _first_sentence(intro.text),
        "key_points": key_points,
        "arc": [
            {"role": "intro", "source_chunk_id": intro.chunk_id, "source": intro.source},
            *[
                {"role": "key_point", "label": point, "source_chunk_id": (_best_chunk_for_concept(point, chunks) or intro).chunk_id}
                for point in key_points[:6]
            ],
            {"role": "wrap_up", "source_chunk_id": conclusion.chunk_id, "source": conclusion.source},
        ],
    }


def _podcast_script(arc: dict[str, Any], chunks: list[Chunk], options: dict[str, Any]) -> dict[str, Any]:
    host_a = str(options.get("host_a_name") or "Host A")
    host_b = str(options.get("host_b_name") or "Host B")
    segments = [
        {"speaker": host_a, "text": f"Today we are studying {arc['title']} from the uploaded course materials."},
        {"speaker": host_b, "text": f"The big idea is: {arc['summary']}"},
    ]
    for index, point in enumerate(arc["key_points"][:6], start=1):
        chunk = _best_chunk_for_concept(point, chunks) or chunks[min(index - 1, len(chunks) - 1)]
        excerpt = " ".join(chunk.text.split())[:260]
        segments.append({"speaker": host_a if index % 2 else host_b, "text": f"Point {index}: {point}."})
        segments.append({"speaker": host_b if index % 2 else host_a, "text": f"Source detail from {chunk.source}: {excerpt}"})
    segments.append({"speaker": host_a, "text": "Pause here and try to explain one key point in your own words before moving on."})
    return {"title": arc["title"], "summary": arc["summary"], "segments": segments}


def _podcast_transcript(script: dict[str, Any]) -> str:
    lines = [script["title"], "", script["summary"], ""]
    lines.extend(f"{segment['speaker']}: {segment['text']}" for segment in script["segments"])
    return "\n".join(lines)


def _first_chunk_with(chunks: list[Chunk], needles: tuple[str, ...]) -> Chunk | None:
    for chunk in chunks:
        haystack = f"{chunk.metadata.get('heading_path', '')} {chunk.text}".casefold()
        if any(needle in haystack for needle in needles):
            return chunk
    return None


def _concepts_from_chunks(chunks: list[Chunk]) -> list[str]:
    concepts: list[str] = []
    for chunk in chunks:
        for concept in chunk.metadata.get("key_concepts", []):
            label = _clean_label(str(concept))
            if label:
                concepts.append(label)
    if not concepts:
        for chunk in chunks:
            concepts.extend(_important_phrases(chunk.text))
    return _dedupe(concepts)[:24]


def _concepts_from_text(text: str) -> list[str]:
    return _important_phrases(text)[:8]


def _important_phrases(text: str) -> list[str]:
    words = [word for word in re.findall(r"\b[A-Za-z][A-Za-z0-9_-]{4,}\b", text) if word.casefold() not in _STOPWORDS]
    counts = Counter(word[:1].upper() + word[1:] for word in words)
    return [word for word, _ in counts.most_common(12)]


def _searchable_text(chunk: Chunk) -> str:
    metadata = chunk.metadata
    values = [chunk.text, str(metadata.get("section_title", "")), str(metadata.get("heading_path", ""))]
    values.extend(str(item) for item in metadata.get("key_concepts", []))
    values.extend(str(item) for item in metadata.get("generated_questions", []))
    return "\n".join(values)


def _tokens(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", text.casefold()) if len(token) > 2 and token not in _STOPWORDS]


def _clean_label(value: str) -> str:
    label = re.sub(r"\s+", " ", str(value or "")).strip(" -:;,.#*")
    if not label or len(label) > 120:
        return ""
    if re.fullmatch(r"[\d\W_]+", label):
        return ""
    return label


def _first_sentence(text: str) -> str:
    clean = " ".join(str(text or "").split())
    return re.split(r"(?<=[.!?])\s+", clean, maxsplit=1)[0][:500] if clean else ""


def _norm(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").casefold()))


def _dedupe(values: Any) -> list[Any]:
    seen: set[str] = set()
    out: list[Any] = []
    for value in values:
        key = _norm(str(value))
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _event(name: str, data: Any) -> GeneratorEvent:
    return {"event": name, "data": data}


_generator_service: GeneratorService | None = None


def get_generator_service() -> GeneratorService:
    global _generator_service
    if _generator_service is None:
        _generator_service = GeneratorService()
    return _generator_service
