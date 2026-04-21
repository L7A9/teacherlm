from __future__ import annotations

import re
from functools import lru_cache

import spacy
from teacherlm_core.schemas.chunk import Chunk

from ..config import get_settings
from ..schemas import MinedConcept


# "X is Y", "X are Y", "X refers to Y", "X means Y", "X is defined as Y"
_DEFINITION_RE = re.compile(
    r"""
    ^\s*
    (?P<term>[A-Z][\w\-\s]{2,60}?)
    \s+(?:is|are|refers\s+to|means|is\s+defined\s+as)\s+
    (?P<def>.+?)\s*[.!?]\s*$
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Markup that shouldn't appear in a flashcard front/back. Tables, code fences,
# and display-math blocks produce unreadable cards (HTML tags verbatim, multi-line
# equations), and the miner's sentence splitter doesn't recognise them as
# non-prose — so we filter at the sentence level before they become context.
_HTML_TAG_RE = re.compile(r"<\s*(table|tr|td|th|tbody|thead|ul|ol|li|pre|code|img)\b", re.IGNORECASE)
_DISPLAY_MATH_RE = re.compile(r"\$\$")
_MARKDOWN_TABLE_RE = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)
_MAX_SENTENCE_CHARS = 400  # anything longer is almost certainly a swallowed table/list


def _is_prose_sentence(text: str) -> bool:
    """Reject sentences that are actually tables, code blocks, or display math
    collapsed into a single 'sentence' by the splitter."""
    if len(text) > _MAX_SENTENCE_CHARS:
        return False
    if _HTML_TAG_RE.search(text):
        return False
    if _DISPLAY_MATH_RE.search(text):
        return False
    if _MARKDOWN_TABLE_RE.search(text):
        return False
    # Markup-heavy lines: >30% non-letter chars usually means equation/table soup.
    letters = sum(c.isalpha() or c.isspace() for c in text)
    if text and letters / len(text) < 0.6:
        return False
    return True


# Front-matter / back-matter junk that commonly appears in uploaded course PDFs
# and slide decks: cover pages, author blocks, copyright notices, page footers,
# acknowledgments. These sentences aren't teaching anything, so we skip mining
# them entirely.
_BOILERPLATE_PATTERNS = (
    re.compile(r"©|copyright|all rights reserved", re.IGNORECASE),
    re.compile(r"\b(university|college|institute|school|faculty|department)\b", re.IGNORECASE),
    re.compile(r"\b(author|authors|supervisor|presented by|prepared by|submitted by)\b", re.IGNORECASE),
    re.compile(r"\backnowledg(e)?ments?\b", re.IGNORECASE),
    re.compile(r"\b(table of contents|references|bibliography)\b", re.IGNORECASE),
    re.compile(r"\bpage\s+\d+\b", re.IGNORECASE),
    re.compile(r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", re.IGNORECASE),  # emails
    re.compile(r"https?://\S+", re.IGNORECASE),  # URLs
    re.compile(r"\barxiv\b", re.IGNORECASE),
    re.compile(r"\bfigure\s+\d+|fig\.\s*\d+", re.IGNORECASE),  # figure captions w/o context
)


def _is_boilerplate(text: str) -> bool:
    return any(p.search(text) for p in _BOILERPLATE_PATTERNS)


# A candidate concept name is "proper-noun-only" if every significant token
# starts uppercase — almost always a person, organization, or title rather
# than a teachable concept.
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")


def _is_proper_noun_phrase(name: str) -> bool:
    words = _WORD_RE.findall(name)
    if len(words) < 1:
        return False
    # Single lowercase word like "photosynthesis" is fine.
    if len(words) == 1 and words[0][0].islower():
        return False
    significant = [w for w in words if len(w) > 2]
    if not significant:
        return False
    return all(w[0].isupper() for w in significant)


@lru_cache
def _nlp():
    settings = get_settings()
    try:
        return spacy.load(settings.spacy_model, disable=["lemmatizer"])
    except OSError as exc:
        raise RuntimeError(
            f"spaCy model '{settings.spacy_model}' is not installed. "
            f"Install with: python -m spacy download {settings.spacy_model}"
        ) from exc


def _normalize(term: str) -> str:
    return " ".join(term.strip().split())


def _acceptable(term: str) -> bool:
    settings = get_settings()
    if not term:
        return False
    if len(term) < settings.min_concept_chars or len(term) > settings.max_concept_chars:
        return False
    # Reject pure pronouns / stopword-only phrases.
    letters = sum(c.isalpha() for c in term)
    return letters >= 2


def _extract_from_sentence(sent, keep_types: tuple[str, ...]) -> list[str]:
    """Return candidate concept names from a single spaCy sentence."""
    out: list[str] = []
    for ent in sent.ents:
        if ent.label_ in keep_types:
            out.append(_normalize(ent.text))
    for chunk in sent.noun_chunks:
        # Strip leading determiners ("the photosynthesis" → "photosynthesis").
        tokens = [t for t in chunk if not t.is_stop and not t.is_punct]
        if not tokens:
            continue
        phrase = _normalize(" ".join(t.text for t in tokens))
        out.append(phrase)
    return [o for o in out if _acceptable(o)]


def _match_definition(sentence_text: str) -> tuple[str, str] | None:
    m = _DEFINITION_RE.match(sentence_text)
    if not m:
        return None
    term = _normalize(m.group("term"))
    definition = _normalize(m.group("def"))
    if not _acceptable(term) or not definition:
        return None
    return term, definition


def mine_concepts(chunks: list[Chunk]) -> list[MinedConcept]:
    """Extract candidate concepts from chunks.

    Strategy:
      1. spaCy NER + noun chunks per sentence → candidate names.
      2. Regex scan for "X is Y" style definitions → enriches the definition
         field when present (used by basic_card_gen for grounding).

    Concepts are keyed case-insensitively; repeat occurrences bump the counter
    and, if a definition appears in any occurrence, it's attached to the entry.
    """
    settings = get_settings()
    if not chunks:
        return []

    nlp = _nlp()
    by_key: dict[str, MinedConcept] = {}

    for chunk in chunks:
        doc = nlp(chunk.text)
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if not sent_text or not _is_prose_sentence(sent_text):
                continue
            if _is_boilerplate(sent_text):
                continue

            definition_pair = _match_definition(sent_text)
            names = [
                n for n in _extract_from_sentence(sent, settings.ner_keep_types)
                if not _is_proper_noun_phrase(n)
            ]

            if definition_pair:
                term, definition = definition_pair
                if _is_proper_noun_phrase(term):
                    # "John Smith is the author..." — not a teachable concept.
                    definition_pair = None
                    definition = None
                elif term not in (n for n in names):
                    names.append(term)
            else:
                definition = None

            for name in names:
                key = name.lower()
                existing = by_key.get(key)
                if existing is None:
                    by_key[key] = MinedConcept(
                        name=name,
                        context_sentence=sent_text,
                        definition=definition if definition_pair and definition_pair[0].lower() == key else None,
                        source_chunk_id=chunk.chunk_id,
                        occurrences=1,
                    )
                else:
                    update: dict = {"occurrences": existing.occurrences + 1}
                    if existing.definition is None and definition_pair and definition_pair[0].lower() == key:
                        update["definition"] = definition
                        update["context_sentence"] = sent_text
                    by_key[key] = existing.model_copy(update=update)

    return list(by_key.values())
