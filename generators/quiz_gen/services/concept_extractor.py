import logging
import re

from teacherlm_core.schemas.chunk import Chunk

from ..schemas import ConceptCard, ExtractedConcepts
from .llm_service import LLMService, build_system_prompt


logger = logging.getLogger(__name__)


# Reject concept names that are boilerplate metadata — authors, affiliations,
# schools, copyright, etc. The prompt already tells the LLM to skip these, but
# smaller models leak them anyway.
_BOILERPLATE_TERMS = re.compile(
    r"\b(author|authors|supervisor|university|college|institute|department|"
    r"faculty|school|copyright|acknowledg(e)?ments?|bibliography|references|"
    r"table of contents|arxiv|doi|page\s+\d+)\b",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")
_DEFINITION_RE = re.compile(r"^\s*(?:[-*]\s*)?([^:\n]{2,90})\s*:\s+(.{12,})$")
_COURSE_CONCEPT_HINTS = re.compile(
    r"\b(concept|method|model|algorithm|approach|system|metric|measure|"
    r"formula|definition|principle|process|classification|theory|law|"
    r"cause|effect|impact|function|structure|property|relationship|"
    r"methode|mod[eè]le|algorithme|approche|syst[eè]me|m[eé]trique|"
    r"mesure|formule|d[eé]finition|principe|processus|classification|"
    r"th[eé]orie|loi|cause|effet|impact|fonction|structure|propri[eé]t[eé]|"
    r"relation|filtrage|apprentissage|recommandation|similarit[eé]|"
    r"learning|filtering|factorization|decomposition|regression)\b|"
    r"\b(?-i:[A-Z]{2,})\b",
    re.IGNORECASE,
)


def _is_proper_noun_only(name: str) -> bool:
    """Title Case Name With No Lowercase Words — almost always a person or org."""
    words = _WORD_RE.findall(name)
    if not words:
        return False
    if len(words) == 1 and words[0][0].islower():
        return False
    significant = [w for w in words if len(w) > 2]
    if not significant:
        return False
    return all(w[0].isupper() for w in significant)


def _is_boilerplate_concept(card: ConceptCard) -> bool:
    haystack = f"{card.name}\n{card.description or ''}"
    if _BOILERPLATE_TERMS.search(haystack):
        return True
    if _is_proper_noun_only(card.name) and not _COURSE_CONCEPT_HINTS.search(haystack):
        return True
    return False


def _format_chunks(chunks: list[Chunk]) -> str:
    if not chunks:
        return "(no context chunks available)"
    return "\n\n".join(
        f"[chunk_id={c.chunk_id} source={c.source}]\n{c.text}" for c in chunks
    )


def _clean_card(card: ConceptCard, valid_ids: set[str]) -> ConceptCard | None:
    """Keep the card if it has a usable name. Filter chunk_ids to known ones,
    but DON'T drop the card if none survive — the question generator falls back
    to the top-scoring chunk, which is still a valid source.
    """
    if not card.name or not card.name.strip():
        return None
    if _is_boilerplate_concept(card):
        logger.debug("dropping boilerplate concept: %r", card.name)
        return None
    kept = [cid for cid in card.source_chunk_ids if cid in valid_ids]
    return card.model_copy(update={"source_chunk_ids": kept})


def _clean_list(cards: list[ConceptCard], valid_ids: set[str]) -> list[ConceptCard]:
    out: list[ConceptCard] = []
    for card in cards:
        cleaned = _clean_card(card, valid_ids)
        if cleaned is not None:
            out.append(cleaned)
    return out


async def extract_concepts(
    chunks: list[Chunk],
    llm: LLMService,
) -> ExtractedConcepts:
    """Extract concepts grouped by Bloom's level via ollama format=ExtractedConcepts."""
    if not chunks:
        return ExtractedConcepts()

    system = build_system_prompt(
        "concept_extraction.txt",
        context=_format_chunks(chunks),
    )
    user = "Extract the testable concepts from the chunks above, grouped by Bloom's level."

    try:
        result = await llm.extract_structured(
            system=system,
            user_message=user,
            schema=ExtractedConcepts,
        )
    except Exception:
        logger.exception("concept extraction failed for %d chunks", len(chunks))
        return _fallback_concepts(chunks)

    valid_ids = {c.chunk_id for c in chunks}
    cleaned = ExtractedConcepts(
        remember=_clean_list(result.remember, valid_ids),
        understand=_clean_list(result.understand, valid_ids),
        apply=_clean_list(result.apply, valid_ids),
        analyze=_clean_list(result.analyze, valid_ids),
    )
    total = sum(
        len(getattr(cleaned, lvl)) for lvl in ("remember", "understand", "apply", "analyze")
    )
    logger.info(
        "extracted %d concepts from %d chunks (raw: r=%d u=%d a=%d an=%d)",
        total,
        len(chunks),
        len(result.remember),
        len(result.understand),
        len(result.apply),
        len(result.analyze),
    )
    return cleaned if total else _fallback_concepts(chunks)


def _fallback_concepts(chunks: list[Chunk]) -> ExtractedConcepts:
    cards: list[ConceptCard] = []
    seen: set[str] = set()
    for chunk in chunks:
        candidates: list[tuple[str, str]] = []
        raw_concepts = chunk.metadata.get("key_concepts") or []
        if isinstance(raw_concepts, list):
            candidates.extend((str(item), "Course key concept from metadata.") for item in raw_concepts)
        heading = str(chunk.metadata.get("section_title") or chunk.metadata.get("heading_path") or "").strip()
        if heading:
            candidates.append((heading.split(">")[-1].strip(), "Section heading from the uploaded material."))
        for line in chunk.text.splitlines():
            definition = _DEFINITION_RE.match(line.strip())
            if definition:
                candidates.append((definition.group(1).strip(" -*"), definition.group(2).strip()[:240]))

        for name, description in candidates:
            name = _clean_name(name)
            key = name.casefold()
            if not name or key in seen:
                continue
            card = ConceptCard(
                name=name,
                bloom_level="understand",
                description=description,
                source_chunk_ids=[chunk.chunk_id],
            )
            if _is_boilerplate_concept(card):
                continue
            seen.add(key)
            cards.append(card)
            if len(cards) >= 40:
                break
        if len(cards) >= 40:
            break
    return ExtractedConcepts(
        remember=cards[::4],
        understand=cards[1::4] or cards[:10],
        apply=cards[2::4],
        analyze=cards[3::4],
    )


def _clean_name(name: str) -> str:
    name = re.sub(r"\s+", " ", name).strip(" -*#:;")
    if ">" in name:
        parts = [part.strip() for part in name.split(">") if part.strip()]
        name = parts[-1] if parts else name
    return name[:90]
