from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from local_api.db import get_store, utc_now


MAX_CHAPTERS = 10
MAX_LESSONS_PER_CHAPTER = 8
MAX_BLOCKS_PER_LESSON = 4
MAX_QUIZ_QUESTIONS = 5


class LocalCourseBuilderService:
    def get_or_build(self, conversation_id: str) -> dict[str, Any]:
        files = get_store().list_files(conversation_id)
        if not files:
            return {"chapters": [], "status": "empty"}
        pending = [file for file in files if file["status"] != "ready"]
        if pending:
            return {
                "chapters": [],
                "status": "waiting_for_files",
                "files_total": len(files),
                "files_pending": len(pending),
            }

        chunks = get_store().list_chunks(conversation_id)
        if not chunks:
            return {"chapters": [], "status": "empty", "files_total": len(files), "files_pending": 0}

        existing = get_store().one(
            "SELECT payload_json FROM coursebuilder_courses WHERE conversation_id = ? ORDER BY updated_at DESC LIMIT 1",
            (conversation_id,),
        )
        if existing is not None:
            payload = json.loads(existing["payload_json"])
            if payload.get("metadata", {}).get("chunk_count") == len(chunks):
                return payload

        return self.rebuild(conversation_id)

    def rebuild(self, conversation_id: str) -> dict[str, Any]:
        chunks = get_store().list_chunks(conversation_id)
        payload = _build_course(conversation_id, chunks)
        get_store().execute(
            """
            INSERT INTO coursebuilder_courses (id, conversation_id, payload_json, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
              payload_json = excluded.payload_json,
              updated_at = excluded.updated_at
            """,
            (
                f"coursebuilder_{conversation_id}",
                conversation_id,
                json.dumps(payload, ensure_ascii=False),
                utc_now(),
            ),
        )
        return payload


def _build_course(conversation_id: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    groups = _chapter_groups(chunks)
    chapters = []
    for chapter_index, group in enumerate(groups[:MAX_CHAPTERS]):
        lessons = _lessons_for_group(conversation_id, chapter_index, group)
        source_chunk_ids = _dedupe(chunk["id"] for chunk in group["chunks"])[:12]
        chapters.append(
            {
                "id": _stable_id(conversation_id, "chapter", chapter_index, group["title"]),
                "title": group["title"],
                "description": _first_sentence(" ".join(chunk["text"] for chunk in group["chunks"][:2])),
                "order_index": chapter_index,
                "summary": _summary(" ".join(chunk["text"] for chunk in group["chunks"][:3]), 420),
                "source_chunk_ids": source_chunk_ids,
                "citations": _citations(group["chunks"], source_chunk_ids),
                "is_locked": False,
                "unlock_rule": "open",
                "lessons": lessons,
                "quiz": _chapter_quiz(group["title"], group["chunks"]),
            }
        )

    title = _course_title(chapters, chunks)
    return {
        "id": f"coursebuilder_{conversation_id}",
        "conversation_id": conversation_id,
        "status": "ready" if chapters else "empty",
        "title": title,
        "description": "A structured course generated from the uploaded documents.",
        "learning_objectives": [f"Understand {chapter['title']}" for chapter in chapters[:6]],
        "prerequisites": [],
        "language": "auto",
        "chapters": chapters,
        "metadata": {
            "context_pack_version": "local-coursebuilder-v1",
            "chunk_count": len(chunks),
            "chapter_count": len(chapters),
            "lesson_count": sum(len(chapter["lessons"]) for chapter in chapters),
            "source_file_count": len({chunk["source_file_id"] for chunk in chunks}),
        },
    }


def _chapter_groups(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        path = metadata.get("heading_path_list") or []
        if isinstance(path, list) and path:
            title = str(path[0])
        else:
            title = str(metadata.get("section_title") or metadata.get("heading_path") or chunk["source_filename"])
            title = title.split(">")[0]
        title = _clean_title(title) or chunk["source_filename"]
        key = f"{chunk['source_file_id']}::{title.casefold()}"
        group = grouped.setdefault(key, {"title": title, "chunks": []})
        group["chunks"].append(chunk)

    groups = list(grouped.values())
    if len(groups) <= MAX_CHAPTERS:
        return groups

    by_file: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        key = chunk["source_file_id"]
        group = by_file.setdefault(key, {"title": chunk["source_filename"], "chunks": []})
        group["chunks"].append(chunk)
    return list(by_file.values())


def _lessons_for_group(conversation_id: str, chapter_index: int, group: dict[str, Any]) -> list[dict[str, Any]]:
    section_groups: dict[str, list[dict[str, Any]]] = {}
    for chunk in group["chunks"]:
        metadata = chunk.get("metadata", {})
        title = str(metadata.get("section_title") or metadata.get("heading_path") or group["title"]).split(">")[-1]
        title = _clean_title(title) or group["title"]
        section_groups.setdefault(title.casefold(), []).append(chunk)

    lessons = []
    for lesson_index, (section_key, section_chunks) in enumerate(section_groups.items()):
        if lesson_index >= MAX_LESSONS_PER_CHAPTER:
            break
        title = _clean_title(section_chunks[0].get("metadata", {}).get("section_title") or section_key) or group["title"]
        source_chunk_ids = _dedupe(chunk["id"] for chunk in section_chunks)[:8]
        blocks = _lesson_blocks(title, section_chunks)
        lessons.append(
            {
                "id": _stable_id(conversation_id, "lesson", chapter_index, lesson_index, title),
                "title": title,
                "order_index": lesson_index,
                "summary": _summary(" ".join(chunk["text"] for chunk in section_chunks[:2]), 320),
                "learning_objectives": [f"Explain {title}", f"Connect {title} to the source material"],
                "source_chunk_ids": source_chunk_ids,
                "citations": _citations(section_chunks, source_chunk_ids),
                "blocks": blocks,
                "quiz": _chapter_quiz(title, section_chunks),
            }
        )
    return lessons or [
        {
            "id": _stable_id(conversation_id, "lesson", chapter_index, 0, group["title"]),
            "title": group["title"],
            "order_index": 0,
            "summary": _summary(" ".join(chunk["text"] for chunk in group["chunks"][:2]), 320),
            "learning_objectives": [f"Understand {group['title']}"],
            "source_chunk_ids": _dedupe(chunk["id"] for chunk in group["chunks"])[:8],
            "citations": _citations(group["chunks"], None),
            "blocks": _lesson_blocks(group["title"], group["chunks"]),
            "quiz": _chapter_quiz(group["title"], group["chunks"]),
        }
    ]


def _lesson_blocks(title: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    source_ids = _dedupe(chunk["id"] for chunk in chunks)[:4]
    explanation = _source_paragraphs(chunks, sentence_limit=9)
    example = _source_paragraphs(_example_first(chunks), sentence_limit=5)
    takeaway = _source_paragraphs(chunks, sentence_limit=4)
    blocks = [
        {
            "id": _stable_id(title, "block", "explanation"),
            "block_type": "explanation",
            "title": title,
            "content": explanation,
            "data_json": {},
            "source_chunk_ids": source_ids,
            "citations": _citations(chunks, source_ids),
        },
        {
            "id": _stable_id(title, "block", "example"),
            "block_type": "example",
            "title": "Source-grounded example",
            "content": example or explanation,
            "data_json": {},
            "source_chunk_ids": source_ids,
            "citations": _citations(chunks, source_ids),
        },
        {
            "id": _stable_id(title, "block", "summary"),
            "block_type": "summary",
            "title": "Key takeaway",
            "content": takeaway or explanation,
            "data_json": {},
            "source_chunk_ids": source_ids,
            "citations": _citations(chunks, source_ids),
        },
    ]
    return [block for block in blocks if block["content"]][:MAX_BLOCKS_PER_LESSON]


def _chapter_quiz(title: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    questions = []
    snippets = []
    for chunk in chunks:
        sentence = _first_sentence(chunk["text"])
        if sentence:
            snippets.append((sentence, chunk))
        if len(snippets) >= MAX_QUIZ_QUESTIONS:
            break
    if not snippets and chunks:
        snippets = [(chunks[0]["text"][:180], chunks[0])]
    for index, (snippet, chunk) in enumerate(snippets[:MAX_QUIZ_QUESTIONS]):
        questions.append(
            {
                "id": _stable_id(title, "quiz-question", index),
                "prompt": f"Which statement is supported by the sources for {title}?",
                "options": [
                    snippet[:180],
                    "A point not supported by the uploaded documents.",
                    "A summary from an unrelated chapter.",
                    "An invented claim with no cited source.",
                ],
                "correct_index": 0,
                "explanation": "The correct option is directly drawn from the cited source chunk.",
                "source_chunk_ids": [chunk["id"]],
                "citations": _citations([chunk], [chunk["id"]]),
            }
        )
    return {"id": _stable_id(title, "quiz"), "questions": questions, "pass_score": 0.7}


def _citations(chunks: list[dict[str, Any]], source_chunk_ids: list[str] | None) -> list[dict[str, Any]]:
    wanted = set(source_chunk_ids or [])
    selected = [chunk for chunk in chunks if not wanted or chunk["id"] in wanted] or chunks[:2]
    citations = []
    seen = set()
    for chunk in selected[:4]:
        if chunk["id"] in seen:
            continue
        seen.add(chunk["id"])
        metadata = chunk.get("metadata", {})
        citations.append(
            {
                "chunk_id": chunk["id"],
                "source": chunk["source_filename"],
                "section": metadata.get("heading_path") or metadata.get("section_title") or "",
                "snippet": " ".join(chunk["text"].split())[:360],
            }
        )
    return citations


def _source_paragraphs(chunks: list[dict[str, Any]], *, sentence_limit: int, max_chars: int = 2200) -> str:
    sentences = []
    seen = set()
    for chunk in chunks[:6]:
        for sentence in _source_sentences(chunk["text"]):
            key = sentence.casefold()
            if key in seen:
                continue
            seen.add(key)
            sentences.append(sentence)
            if len(sentences) >= sentence_limit:
                break
        if len(sentences) >= sentence_limit:
            break
    paragraphs = []
    for index in range(0, len(sentences), 3):
        paragraph = " ".join(sentences[index : index + 3]).strip()
        if paragraph:
            paragraphs.append(paragraph)
    return "\n\n".join(paragraphs)[:max_chars].strip()


def _source_sentences(text: str) -> list[str]:
    clean = re.sub(r"\s+", " ", str(text or "").replace("\r", "\n")).strip()
    pieces = re.split(r"(?<=[.!?。！？])\s+|\s+[•*-]\s+", clean)
    out = []
    for piece in pieces:
        sentence = piece.strip(" -•*\t")
        if len(sentence) >= 25 and len(re.findall(r"\w+", sentence)) >= 5:
            out.append(sentence[:500])
    return out or ([clean[:900]] if clean else [])


def _example_first(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    terms = ("example", "exemple", "par exemple", "e.g.", "application", "case study", "cas ")
    return [
        chunk
        for _, chunk in sorted(
            enumerate(chunks),
            key=lambda item: (0 if any(term in item[1]["text"].casefold() for term in terms) else 1, item[0]),
        )
    ]


def _course_title(chapters: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> str:
    metadata_titles = [
        str(chunk.get("metadata", {}).get("document_title") or "").strip()
        for chunk in chunks
        if chunk.get("metadata", {}).get("document_title")
    ]
    if metadata_titles:
        return Counter(metadata_titles).most_common(1)[0][0]
    return chapters[0]["title"] if chapters else "Generated Course"


def _summary(text: str, max_chars: int) -> str:
    clean = " ".join(str(text or "").split())
    if len(clean) <= max_chars:
        return clean
    return clean[:max_chars].rsplit(" ", 1)[0].strip()


def _first_sentence(text: str) -> str:
    clean = " ".join(str(text or "").split())
    return re.split(r"(?<=[.!?])\s+", clean, maxsplit=1)[0][:500] if clean else ""


def _clean_title(value: Any) -> str:
    title = re.sub(r"\s+", " ", str(value or "")).strip(" -:;,.#*")
    if not title or len(title) > 140 or re.fullmatch(r"[\d\W_]+", title):
        return ""
    return title


def _dedupe(values: Any) -> list[Any]:
    out = []
    seen = set()
    for value in values:
        key = str(value).casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _stable_id(*parts: Any) -> str:
    raw = "::".join(str(part) for part in parts)
    compact = re.sub(r"[^a-zA-Z0-9]+", "_", raw).strip("_").lower()
    return compact[:96] or "coursebuilder_item"


_coursebuilder_service: LocalCourseBuilderService | None = None


def get_coursebuilder_service() -> LocalCourseBuilderService:
    global _coursebuilder_service
    if _coursebuilder_service is None:
        _coursebuilder_service = LocalCourseBuilderService()
    return _coursebuilder_service
