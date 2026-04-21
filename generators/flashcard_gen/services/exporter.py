from __future__ import annotations

import csv
import io
import logging
import tempfile
from pathlib import Path

import genanki
from teacherlm_core.schemas.generator_io import GeneratorArtifact

from ..schemas import BasicCard, ClozeCard, FlashcardDeck
from .artifact_store import ArtifactStore


logger = logging.getLogger(__name__)


# Stable model IDs so re-exported decks merge cleanly inside Anki.
_ANKI_BASIC_MODEL_ID = 1607392319
_ANKI_CLOZE_MODEL_ID = 1607392320


def _basic_model() -> genanki.Model:
    return genanki.Model(
        _ANKI_BASIC_MODEL_ID,
        "TeacherLM Basic",
        fields=[{"name": "Front"}, {"name": "Back"}],
        templates=[
            {
                "name": "Card 1",
                "qfmt": "{{Front}}",
                "afmt": '{{FrontSide}}<hr id="answer">{{Back}}',
            }
        ],
    )


def _cloze_model() -> genanki.Model:
    return genanki.Model(
        _ANKI_CLOZE_MODEL_ID,
        "TeacherLM Cloze",
        fields=[{"name": "Text"}],
        templates=[
            {
                "name": "Cloze",
                "qfmt": "{{cloze:Text}}",
                "afmt": "{{cloze:Text}}",
            }
        ],
        model_type=genanki.Model.CLOZE,
    )


def _json_payload(deck: FlashcardDeck) -> bytes:
    return deck.model_dump_json(indent=2).encode("utf-8")


def _csv_payload(deck: FlashcardDeck) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["type", "front_or_text", "back_or_answer", "concept", "source_chunk_id"])
    for card in deck.cards:
        if isinstance(card, BasicCard):
            writer.writerow(["basic", card.front, card.back, card.concept, card.source_chunk_id])
        elif isinstance(card, ClozeCard):
            writer.writerow(["cloze", card.text, card.answer, card.concept, card.source_chunk_id])
    return buf.getvalue().encode("utf-8")


def _apkg_payload(deck: FlashcardDeck) -> bytes:
    # Deck ID derived from title hash so re-exports of the same deck update
    # cards in place rather than creating a parallel deck tree.
    deck_id = abs(hash(deck.title)) % (10**10) + 1
    anki_deck = genanki.Deck(deck_id, deck.title)
    basic_model = _basic_model()
    cloze_model = _cloze_model()

    for card in deck.cards:
        if isinstance(card, BasicCard):
            anki_deck.add_note(
                genanki.Note(model=basic_model, fields=[card.front, card.back])
            )
        elif isinstance(card, ClozeCard):
            anki_deck.add_note(
                genanki.Note(model=cloze_model, fields=[card.text])
            )

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "deck.apkg"
        genanki.Package(anki_deck).write_to_file(str(path))
        return path.read_bytes()


async def export_deck(
    deck: FlashcardDeck,
    *,
    conversation_id: str,
    store: ArtifactStore,
) -> list[GeneratorArtifact]:
    """Upload JSON + CSV + APKG to MinIO. Failures per-format are logged but
    don't abort the others; returned list only includes successful uploads."""
    artifacts: list[GeneratorArtifact] = []
    targets = [
        ("flashcards.json", _json_payload(deck), "application/json", "flashcards"),
        ("flashcards.csv", _csv_payload(deck), "text/csv", "flashcards_csv"),
        (
            "flashcards.apkg",
            _apkg_payload(deck),
            "application/octet-stream",
            "flashcards_apkg",
        ),
    ]
    for filename, payload, content_type, artifact_type in targets:
        try:
            _, url = await store.save(
                conversation_id=conversation_id,
                filename=filename,
                payload=payload,
                content_type=content_type,
            )
            artifacts.append(
                GeneratorArtifact(type=artifact_type, url=url, filename=filename)
            )
        except Exception:
            logger.exception("failed to upload %s", filename)
    return artifacts
