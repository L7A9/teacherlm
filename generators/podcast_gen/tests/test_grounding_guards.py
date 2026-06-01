from __future__ import annotations

import sys
import unittest
from pathlib import Path


GENERATORS_DIR = Path(__file__).resolve().parents[2]
CORE_DIR = Path(__file__).resolve().parents[3] / "packages" / "teacherlm_core"
for path in (GENERATORS_DIR, CORE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from teacherlm_core.schemas.chunk import Chunk  # noqa: E402

from podcast_gen.schemas import NarrativeArc, PodcastScript, Segment  # noqa: E402
from podcast_gen.services.grounding_guard import (  # noqa: E402
    deterministic_script_from_arc,
    script_claims_no_materials,
    usable_context_chunks,
)
from podcast_gen.services.narrative_extractor import (  # noqa: E402
    extract_narrative_arc,
    format_context_for_speech,
)


class FakeLLM:
    async def extract_structured(self, *, system, user_message, schema):  # noqa: ANN001, ANN202
        return schema(
            title="No Materials Available for Podcast",
            intro="There are no uploaded course materials to work with yet.",
            key_points=["No source excerpts were available."],
            conclusion="Upload more files first.",
            sources=[],
        )


class PromptLeakLLM:
    async def extract_structured(self, *, system, user_message, schema):  # noqa: ANN001, ANN202
        return schema(
            title="Les bases de la planification de contenu",
            intro="Nous allons planifier le podcast.",
            key_points=["Choisir une structure de podcast."],
            conclusion="La planification est importante.",
            sources=[],
        )


class PodcastGroundingGuardTests(unittest.IsolatedAsyncioTestCase):
    async def test_bad_no_materials_arc_is_replaced_when_chunks_exist(self) -> None:
        chunks = [
            Chunk(
                text=(
                    "Android Intents allow one activity to request an action from "
                    "another activity. They can also carry values in a Bundle."
                ),
                source="Cours 3.pdf",
                score=1,
                chunk_id="intent-1",
                metadata={"heading_path": "Communication > Utilisation des Intents"},
            )
        ]

        arc = await extract_narrative_arc(
            chunks,
            topic_focus="",
            language_hint="Write in English.",
            llm=FakeLLM(),  # type: ignore[arg-type]
        )

        self.assertNotIn("No Materials", arc.title)
        self.assertIn("Intents", arc.title)
        self.assertTrue(arc.key_points)
        self.assertEqual(arc.sources, ["Cours 3.pdf"])

    async def test_prompt_leak_title_is_replaced_with_source_topic(self) -> None:
        chunks = [
            Chunk(
                text="La classe Bundle permet de transmettre des donnees entre deux activites Android.",
                source="Cours 3.pdf",
                score=1,
                chunk_id="bundle-1",
                metadata={"heading_path": "Communication entre les activites > Bundle"},
            )
        ]

        arc = await extract_narrative_arc(
            chunks,
            topic_focus="",
            language_hint="Ecris en francais.",
            llm=PromptLeakLLM(),  # type: ignore[arg-type]
        )

        self.assertNotIn("planification de contenu", arc.title.lower())
        self.assertIn("Bundle", arc.title)

    def test_context_format_skips_tiny_noise_but_keeps_real_material(self) -> None:
        text = format_context_for_speech(
            [
                Chunk(text="22", source="Cours 3.pdf", score=1, chunk_id="page", metadata={}),
                Chunk(
                    text="A Bundle stores values that can be passed between Android activities.",
                    source="Cours 3.pdf",
                    score=1,
                    chunk_id="bundle",
                    metadata={},
                ),
            ]
        )

        self.assertNotIn("\n22\n", text)
        self.assertIn("Bundle stores values", text)

    def test_script_no_materials_guard_and_deterministic_fallback(self) -> None:
        bad = PodcastScript(
            title="No Materials Available for Podcast",
            summary="No uploaded materials.",
            segments=[Segment(speaker="host_a", text="No materials are available.")],
        )
        self.assertTrue(script_claims_no_materials(bad))

        arc = NarrativeArc(
            title="Android Intents",
            intro="The material explains how activities communicate.",
            key_points=["Intent objects launch activities", "Bundles pass data"],
            conclusion="Use intents and bundles together for navigation with data.",
            sources=["Cours 3.pdf"],
        )
        fallback = deterministic_script_from_arc(arc)
        self.assertFalse(script_claims_no_materials(fallback))
        self.assertGreaterEqual(len(fallback.segments), 6)

    def test_usable_context_chunks_filters_empty_and_tiny_chunks(self) -> None:
        chunks = [
            Chunk(text="22", source="Cours 3.pdf", score=1, chunk_id="page", metadata={}),
            Chunk(
                text="Activities move through lifecycle callbacks such as onCreate and onPause.",
                source="Cours 2.pdf",
                score=1,
                chunk_id="life",
                metadata={},
            ),
        ]

        self.assertEqual([chunk.chunk_id for chunk in usable_context_chunks(chunks)], ["life"])


if __name__ == "__main__":
    unittest.main()
