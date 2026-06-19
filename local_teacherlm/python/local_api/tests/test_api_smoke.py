from pathlib import Path
import asyncio
import sys
import types
import json
import re

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "teacherlm_core"))
sys.path.insert(0, str(ROOT / "local_api"))


def _client(monkeypatch, tmp_path):
    monkeypatch.setenv("TEACHERLM_APP_DATA_DIR", str(tmp_path))
    _install_fake_fastembed(monkeypatch)
    from fastapi.testclient import TestClient

    from local_api.config import get_settings
    from local_api.db import get_store
    from local_api.main import create_app
    import local_api.services.generators as generators_module
    import local_api.services.ingestion as ingestion_module
    import local_api.services.knowledge_graph as graph_module
    import local_api.services.podcast_audio as podcast_audio_module
    import local_api.services.retrieval as retrieval_module
    import local_api.services.secrets as secrets_module
    import local_api.services.settings as settings_module
    import local_api.services.vector_service as vector_module

    get_settings.cache_clear()
    store = get_store()
    if store._conn is not None:
        store._conn.close()
        store._conn = None
    store.settings = get_settings()
    secrets_module._secret_box = None
    settings_module._settings_service = None
    ingestion_module._ingestion_service = None
    vector_module._vector_service = None
    graph_module._knowledge_graph_service = None
    retrieval_module._retrieval_service = None
    generators_module._generator_service = None
    podcast_audio_module.get_podcast_audio_service.cache_clear()

    async def offline_generator_llm(*_args, **_kwargs):
        raise RuntimeError("LLM disabled in API smoke tests")

    monkeypatch.setattr(generators_module, "complete_text", offline_generator_llm)

    class OfflinePodcastAudio:
        async def synthesize(self, *_args, **_kwargs):
            raise podcast_audio_module.PodcastAudioError("tts_unavailable", "TTS disabled in API smoke tests")

    monkeypatch.setattr(generators_module, "get_podcast_audio_service", lambda: OfflinePodcastAudio())
    store.initialize()
    return TestClient(create_app())


def _install_fake_fastembed(monkeypatch) -> None:
    class FakeTextEmbedding:
        def __init__(self, model_name: str = "") -> None:
            self.model_name = model_name

        def embed(self, texts):
            return self._vectors(texts)

        def passage_embed(self, texts):
            return self._vectors(texts)

        def query_embed(self, texts):
            return self._vectors(texts)

        def _vectors(self, texts):
            for text in texts:
                vector = [0.0] * 1024
                for token in re.findall(r"\w+", str(text).casefold()):
                    vector[hash(token) % 1024] += 1.0
                norm = sum(value * value for value in vector) ** 0.5 or 1.0
                yield [value / norm for value in vector]

    class FakeTextCrossEncoder:
        def __init__(self, model_name: str = "") -> None:
            self.model_name = model_name

        def rerank(self, query, documents):
            query_terms = set(re.findall(r"\w+", str(query).casefold()))
            scores = []
            for document in documents:
                doc_terms = set(re.findall(r"\w+", str(document).casefold()))
                scores.append(float(len(query_terms & doc_terms)))
            return scores

    fastembed = types.ModuleType("fastembed")
    fastembed.TextEmbedding = FakeTextEmbedding
    rerank_module = types.ModuleType("fastembed.rerank")
    cross_encoder = types.ModuleType("fastembed.rerank.cross_encoder")
    cross_encoder.TextCrossEncoder = FakeTextCrossEncoder
    monkeypatch.setitem(sys.modules, "fastembed", fastembed)
    monkeypatch.setitem(sys.modules, "fastembed.rerank", rerank_module)
    monkeypatch.setitem(sys.modules, "fastembed.rerank.cross_encoder", cross_encoder)


def test_quiz_repairs_wrapped_answer_sentences() -> None:
    from local_api.services import generators

    text = '''The jam study demonstrated the phenomenon known as "choice overload
problem" when humans are faced with choices, so less is often better.
▶ A separate complete idea remains a separate answer option.'''

    candidates = generators._statement_candidates(text)

    assert 'The jam study demonstrated the phenomenon known as "choice overload problem" when humans are faced with choices, so less is often better.' in candidates
    assert not any(candidate.startswith("problem\"") for candidate in candidates)
    assert all(not generators._has_unbalanced_quiz_delimiters(candidate) for candidate in candidates)


def test_quiz_rejects_incomplete_and_overlapping_options() -> None:
    from local_api.services import generators

    assert not generators._is_quiz_answer_option('problem" when humans are faced with choices, less is better')
    question = {
        "type": "mcq",
        "category": "definition",
        "question": "What best defines hybrid search?",
        "options": [
            "Hybrid search combines lexical and semantic evidence.",
            "Hybrid search combines lexical and semantic evidence to improve retrieval.",
            "Hybrid search relies only on document filenames and locations.",
            "Hybrid search removes the need to evaluate retrieved results.",
        ],
    }
    assert not generators._validate_quiz_question(question)


def test_quiz_calls_structured_llm_with_full_chunks_graph_type_and_count(monkeypatch) -> None:
    from teacherlm_core.llm.providers import LLMProviderConfig
    from teacherlm_core.schemas import GeneratorInput, LearnerState
    import local_api.services.generators as generators

    provider = LLMProviderConfig(
        provider_id="quiz-provider",
        display_name="Quiz provider",
        provider_type="ollama",
        model_name="quiz-model",
    )
    monkeypatch.setattr(
        generators,
        "get_settings_service",
        lambda: types.SimpleNamespace(get_default_chat_provider_config=lambda: provider),
    )
    calls: list[dict] = []

    async def fake_complete_text(_provider, messages, *, json_schema=None, temperature=0.2):
        calls.append({"messages": messages, "json_schema": json_schema, "temperature": temperature})
        return json.dumps(
            {
                "title": f"Fresh Search Quiz {len(calls)}",
                "intro_message": "Apply the selected search concepts.",
                "questions": [
                    {
                        "type": "mcq",
                        "category": "definition",
                        "bloom_level": "remember",
                        "question": "What evidence does dense retrieval use for semantic matching?",
                        "options": [
                            "Dense retrieval uses vector similarity to represent semantic meaning.",
                            "Dense retrieval uses only alphabetical ordering of unrelated labels.",
                            "Dense retrieval removes all semantic information before matching.",
                            "Dense retrieval depends exclusively on arbitrary visual decoration.",
                        ],
                        "correct_index": 0,
                        "explanation": "Dense retrieval represents semantic meaning through vector similarity.",
                        "concept": "Dense retrieval",
                        "source_chunk_id": "dense-chunk",
                    },
                    {
                        "type": "mcq",
                        "category": "mechanism",
                        "bloom_level": "understand",
                        "question": "What does reciprocal rank fusion combine?",
                        "options": [
                            "Reciprocal rank fusion combines multiple ranked result lists.",
                            "Reciprocal rank fusion discards every available ranking signal.",
                            "Reciprocal rank fusion sorts only by source filename length.",
                            "Reciprocal rank fusion creates results without any ranked inputs.",
                        ],
                        "correct_index": 0,
                        "explanation": "Reciprocal rank fusion combines ranked result lists.",
                        "concept": "Reciprocal rank fusion",
                        "source_chunk_id": "fusion-chunk",
                    },
                ],
            }
        )

    monkeypatch.setattr(generators, "complete_text", fake_complete_text)
    source_chunks = [
        generators.Chunk(
            text="Dense retrieval uses vector similarity to represent semantic meaning.",
            source="search.md",
            score=1.0,
            chunk_id="dense-chunk",
            metadata={"quiz_full_context": True},
        ),
        generators.Chunk(
            text="Reciprocal rank fusion combines multiple ranked result lists.",
            source="search.md",
            score=1.0,
            chunk_id="fusion-chunk",
            metadata={"quiz_full_context": True},
        ),
    ]
    graph_chunk = generators.Chunk(
        text="Dense Retrieval --supports--> Reciprocal Rank Fusion",
        source="knowledge_graph",
        score=1.0,
        chunk_id="quiz-graph:test",
        metadata={
            "context_type": "quiz_graph_context",
            "graph_complete": True,
            "source_file_ids": ["file-search"],
            "source_chunk_ids": ["dense-chunk", "fusion-chunk"],
            "graph_nodes": [
                {"id": "dense", "label": "Dense Retrieval", "node_type": "concept"},
                {"id": "fusion", "label": "Reciprocal Rank Fusion", "node_type": "concept"},
            ],
            "graph_edges": [
                {
                    "source_node_id": "dense",
                    "target_node_id": "fusion",
                    "source_label": "Dense Retrieval",
                    "target_label": "Reciprocal Rank Fusion",
                    "relation_type": "supports",
                }
            ],
        },
    )

    def payload(run_id: str) -> GeneratorInput:
        return GeneratorInput(
            conversation_id="quiz-conversation",
            user_message="Focus on search mechanisms",
            context_chunks=[*source_chunks, graph_chunk],
            learner_state=LearnerState(conversation_id="quiz-conversation"),
            chat_history=[],
            options={"generation_run_id": run_id, "question_type": "mcq", "question_count": 2},
        )

    first, first_meta = asyncio.run(
        generators._build_fresh_quiz(
            payload("quiz-run-one"),
            source_chunks=source_chunks,
            graph_chunks=[graph_chunk],
            question_type="mcq",
            question_count=2,
        )
    )
    second, second_meta = asyncio.run(
        generators._build_fresh_quiz(
            payload("quiz-run-two"),
            source_chunks=source_chunks,
            graph_chunks=[graph_chunk],
            question_type="mcq",
            question_count=2,
        )
    )

    assert len(calls) == 2
    assert all(call["json_schema"] for call in calls)
    assert calls[0]["temperature"] == 0.7
    first_prompt = calls[0]["messages"][-1].content
    assert "quiz-run-one" in first_prompt
    assert "Focus on search mechanisms" in first_prompt
    assert "Question type: mcq" in first_prompt
    assert "Number of questions: 2" in first_prompt
    assert source_chunks[0].text in first_prompt
    assert source_chunks[1].text in first_prompt
    assert "Dense Retrieval" in first_prompt and "supports" in first_prompt
    assert first["title"] != second["title"]
    assert first_meta["backend"] == "llm_structured_fresh_generation"
    assert second_meta["backend"] == "llm_structured_fresh_generation"


def test_quiz_returns_grounded_fallback_when_generation_and_repair_are_invalid(monkeypatch) -> None:
    from teacherlm_core.llm.providers import LLMProviderConfig
    from teacherlm_core.schemas import GeneratorInput, LearnerState
    import local_api.services.generators as generators

    provider = LLMProviderConfig(
        provider_id="quiz-provider",
        display_name="Quiz provider",
        provider_type="ollama",
        model_name="quiz-model",
    )
    monkeypatch.setattr(
        generators,
        "get_settings_service",
        lambda: types.SimpleNamespace(get_default_chat_provider_config=lambda: provider),
    )
    calls: list[dict] = []

    async def invalid_complete_text(_provider, messages, *, json_schema=None, temperature=0.2):
        calls.append({"messages": messages, "json_schema": json_schema, "temperature": temperature})
        return "not valid quiz json"

    monkeypatch.setattr(generators, "complete_text", invalid_complete_text)
    source_chunks = [
        generators.Chunk(
            text=(
                "Dense retrieval uses vector similarity to represent semantic meaning and identify relevant "
                "documents for a search query."
            ),
            source="search.md",
            score=1.0,
            chunk_id="dense-chunk",
            metadata={"section_title": "Dense retrieval"},
        ),
        generators.Chunk(
            text=(
                "Reciprocal rank fusion combines multiple ranked result lists to improve the final ordering of "
                "retrieved documents."
            ),
            source="search.md",
            score=1.0,
            chunk_id="fusion-chunk",
            metadata={"section_title": "Reciprocal rank fusion"},
        ),
    ]
    payload = GeneratorInput(
        conversation_id="quiz-fallback",
        user_message="Test search concepts",
        context_chunks=source_chunks,
        learner_state=LearnerState(conversation_id="quiz-fallback"),
        chat_history=[],
        options={"generation_run_id": "fallback-run", "question_type": "mcq", "question_count": 2},
    )

    quiz, metadata = asyncio.run(
        generators._build_fresh_quiz(
            payload,
            source_chunks=source_chunks,
            graph_chunks=[],
            question_type="mcq",
            question_count=2,
        )
    )

    assert len(calls) == 2
    assert calls[1]["temperature"] == 0.45
    assert source_chunks[0].text not in calls[1]["messages"][-1].content
    assert metadata["backend"] == "deterministic_grounded_fallback"
    assert metadata["reason"] == "structured_generation_failed"
    assert metadata["repair_attempted"] is True
    assert len(quiz["questions"]) == 2
    assert {question["source_chunk_id"] for question in quiz["questions"]} == {"dense-chunk", "fusion-chunk"}
    assert all(len(question["options"]) == 4 for question in quiz["questions"])


def test_quiz_accepts_mistral_nested_option_objects_without_repair() -> None:
    import local_api.services.generators as generators

    source_chunk = generators.Chunk(
        text=(
            "Item-based collaborative filtering finds items similar to the target item that the user already rated, "
            "then predicts a score with a weighted average of those neighboring item ratings."
        ),
        source="collaborative-filtering.pdf",
        score=1.0,
        chunk_id="chunk-item-cf",
        metadata={"section_title": "Item-Based Collaborative Filtering"},
    )
    raw = json.dumps(
        {
            "quiz": {
                "metadata": {"question_type": "mcq", "language": "en"},
                "questions": [
                    {
                        "question_text": "How does item-based collaborative filtering predict a missing score?",
                        "options": [
                            {"text": "RMSE", "is_correct": False},
                            {
                                "text": "It uses a weighted average of ratings for similar neighboring items.",
                                "is_correct": True,
                                "source_chunk_id": "chunk-invented-by-model",
                            },
                            {"text": "It discards every rating before making the prediction.", "is_correct": False},
                            {"text": "It predicts scores from visual formatting alone.", "is_correct": False},
                        ],
                        "bloom_level": "application",
                    }
                ],
            }
        }
    )

    quiz = generators._validated_llm_quiz(
        raw,
        source_chunks=[source_chunk],
        question_type="mcq",
        question_count=1,
    )

    assert quiz["questions"][0]["correct_index"] == 1
    assert quiz["questions"][0]["source_chunk_id"] == "chunk-item-cf"
    assert quiz["questions"][0]["bloom_level"] == "apply"


def test_app_factory_smoke(monkeypatch, tmp_path) -> None:
    client = _client(monkeypatch, tmp_path)
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["ok"] is True
    created = client.post("/api/conversations", json={"title": "Algebra"}).json()
    assert created["title"] == "Algebra"
    updated = client.patch(f"/api/conversations/{created['id']}", json={"title": "Linear Algebra"}).json()
    assert updated["title"] == "Linear Algebra"

    second = client.post("/api/conversations", json={"title": "Biology"}).json()
    upload = client.post(
        f"/api/conversations/{created['id']}/files",
        files={"file": ("lesson.txt", b"# Vectors\n\nVectors have magnitude and direction.", "text/plain")},
    )
    assert upload.status_code == 200
    assert upload.json()["status"] == "uploaded"
    assert upload.json()["conversation_id"] == created["id"]

    first_files = client.get(f"/api/conversations/{created['id']}/files").json()["files"]
    second_files = client.get(f"/api/conversations/{second['id']}/files").json()["files"]
    assert [file["filename"] for file in first_files] == ["lesson.txt"]
    assert first_files[0]["status"] == "ready"
    assert first_files[0]["chunk_count"] > 0
    assert first_files[0]["parser_used"] == "local:text"
    assert second_files == []

    deleted = client.delete(f"/api/conversations/{created['id']}")
    assert deleted.status_code == 204
    assert client.get(f"/api/conversations/{created['id']}").status_code == 404
    assert client.get(f"/api/conversations/{created['id']}/files").status_code == 404

    model_a = client.post(
        "/api/settings/llm-providers",
        json={
            "display_name": "Model A",
            "provider_type": "ollama",
            "base_url": "http://localhost:11434",
            "model_name": "llama3.2",
            "is_default_chat": True,
        },
    ).json()
    model_b = client.post(
        "/api/settings/llm-providers",
        json={
            "display_name": "Model B",
            "provider_type": "openai_compatible",
            "base_url": "http://localhost:1234/v1",
            "model_name": "student-model",
            "api_key": "test-key",
        },
    ).json()
    providers = client.get("/api/settings/llm-providers").json()["providers"]
    assert [provider["id"] for provider in providers if provider["is_default_chat"]] == [model_a["id"]]

    updated_b = client.patch(
        f"/api/settings/llm-providers/{model_b['id']}",
        json={"is_default_chat": True},
    ).json()
    assert updated_b["is_default_chat"] is True
    providers = client.get("/api/settings/llm-providers").json()["providers"]
    assert [provider["id"] for provider in providers if provider["is_default_chat"]] == [model_b["id"]]

    assert client.delete(f"/api/settings/llm-providers/{model_b['id']}").status_code == 200
    providers = client.get("/api/settings/llm-providers").json()["providers"]
    defaults = [provider["id"] for provider in providers if provider["is_default_chat"]]
    assert len(defaults) == 1
    assert defaults[0] != model_b["id"]

    parser = client.patch("/api/settings/parse", json={"llama_cloud_api_key": "llx-test", "use_local_parsers_only": False}).json()
    assert parser["llama_cloud_api_key_set"] is True
    assert parser["use_local_parsers_only"] is False
    parser = client.patch("/api/settings/parse", json={"clear_llama_cloud_api_key": True}).json()
    assert parser["llama_cloud_api_key_set"] is False
    assert parser["use_local_parsers_only"] is True


def test_upload_field_delete_and_retry(monkeypatch, tmp_path) -> None:
    client = _client(monkeypatch, tmp_path)
    conversation = client.post("/api/conversations", json={"title": "Files"}).json()
    upload = client.post(
        f"/api/conversations/{conversation['id']}/files",
        files={"upload": ("bad.pdf", b"not really a pdf", "application/pdf")},
    )
    assert upload.status_code == 200
    file_id = upload.json()["id"]

    failed = client.get(f"/api/conversations/{conversation['id']}/files/{file_id}").json()
    assert failed["status"] == "failed"
    assert "PDF local parser failed" in failed["error"]

    retry = client.post(f"/api/conversations/{conversation['id']}/files/{file_id}/retry")
    assert retry.status_code == 200
    retried = client.get(f"/api/conversations/{conversation['id']}/files/{file_id}").json()
    assert retried["status"] == "failed"
    assert retried["chunk_count"] == 0

    delete = client.delete(f"/api/conversations/{conversation['id']}/files/{file_id}")
    assert delete.status_code == 204
    assert client.get(f"/api/conversations/{conversation['id']}/files/{file_id}").status_code == 404


def test_llama_cloud_parser_mode_calls_llama_cloud(monkeypatch, tmp_path) -> None:
    calls = []

    class FakeParsing:
        async def parse(self, **kwargs):
            calls.append(kwargs)
            return types.SimpleNamespace(
                markdown_full="# Cloud Lesson\n\nCloud parsed markdown from LlamaCloud.",
                job=types.SimpleNamespace(id="job-123"),
            )

    class FakeLlamaCloud:
        def __init__(self, *, api_key: str, base_url: str | None = None) -> None:
            self.api_key = api_key
            self.base_url = base_url
            self.parsing = FakeParsing()

    monkeypatch.setitem(sys.modules, "llama_cloud", types.SimpleNamespace(AsyncLlamaCloud=FakeLlamaCloud))

    client = _client(monkeypatch, tmp_path)
    parser = client.patch(
        "/api/settings/parse",
        json={"llama_cloud_api_key": "llx-test", "use_local_parsers_only": False},
    ).json()
    assert parser["status"] == "llamaparse_configured"

    conversation = client.post("/api/conversations", json={"title": "Cloud"}).json()
    upload = client.post(
        f"/api/conversations/{conversation['id']}/files",
        files={"upload": ("lesson.txt", b"plain text still goes through cloud mode", "text/plain")},
    )
    assert upload.status_code == 200
    file_id = upload.json()["id"]
    stored = client.get(f"/api/conversations/{conversation['id']}/files/{file_id}").json()
    assert stored["status"] == "ready"

    assert stored["parser_used"] == "llamacloud:job-123"
    assert stored["chunk_count"] > 0
    assert calls
    assert calls[0]["tier"] == "cost_effective"
    assert calls[0]["version"] == "latest"
    assert calls[0]["expand"] == ["markdown"]
    assert calls[0]["upload_file"][0] == "lesson.txt"


def test_local_vectors_graph_settings_and_rebuild(monkeypatch, tmp_path) -> None:
    client = _client(monkeypatch, tmp_path)
    conversation = client.post("/api/conversations", json={"title": "RAG"}).json()
    upload = client.post(
        f"/api/conversations/{conversation['id']}/files",
        files={
            "upload": (
                "rag.md",
                b"""# Retrieval

HyDE creates a hypothetical answer to improve retrieval.

## Graph Search

Graph search connects concepts, chunks, formulas, and examples.

## Reranking

The cross encoder reranker scores candidate chunks after RRF fusion.
""",
                "text/markdown",
            )
        },
    )
    assert upload.status_code == 200
    assert client.get(f"/api/conversations/{conversation['id']}/files/{upload.json()['id']}").json()["status"] == "ready"

    from local_api.db import get_store

    chunks = get_store().list_chunks(conversation["id"])
    assert chunks
    assert all(chunk["embedding"] for chunk in chunks)
    assert {chunk["metadata"]["embedding_model"] for chunk in chunks} == {"intfloat/multilingual-e5-large"}
    assert {chunk["metadata"]["embedding_dim"] for chunk in chunks} == {1024}
    assert all(chunk["metadata"]["generated_questions"] for chunk in chunks)

    graph = client.get(f"/api/conversations/{conversation['id']}/knowledge-graph").json()
    assert graph["node_count"] > 0
    assert graph["edge_count"] > 0
    assert {"course", "file", "section", "chunk", "concept"} <= {node["node_type"] for node in graph["nodes"]}
    assert all("source_chunk_ids" in edge for edge in graph["edges"])

    settings = client.get("/api/settings/retrieval").json()
    assert settings["embedding_model"] == "intfloat/multilingual-e5-large"
    assert settings["retrieval_rerank_enabled"] is True
    assert settings["retrieval_graph_enabled"] is True
    assert settings["index_status"]["embedded_chunk_count"] >= len(chunks)

    updated = client.patch(
        "/api/settings/retrieval",
        json={"retrieval_hyde_enabled": False, "retrieval_graph_enabled": False},
    ).json()
    assert updated["retrieval_hyde_enabled"] is False
    assert updated["retrieval_graph_enabled"] is False

    rebuilt = client.post(f"/api/conversations/{conversation['id']}/indexes/rebuild").json()
    assert rebuilt["ok"] is True
    assert rebuilt["index_status"]["stale_chunk_count"] == 0
    assert rebuilt["index_status"]["graph_node_count"] > 0


def test_retrieval_uses_vector_hyde_graph_and_reranker(monkeypatch, tmp_path) -> None:
    client = _client(monkeypatch, tmp_path)
    conversation = client.post("/api/conversations", json={"title": "Hybrid"}).json()
    client.post(
        f"/api/conversations/{conversation['id']}/files",
        files={
            "upload": (
                "hybrid.md",
                b"""# Search

Dense semantic vectors help find meaning when the student's words differ from the source.

## HyDE

HyDE writes a hypothetical course excerpt for retrieval only and never becomes a cited source.

## Graph

Knowledge graph neighbors connect Graph Search to related chunks and concepts.
""",
                "text/markdown",
            )
        },
    )

    from local_api.services.retrieval import get_retrieval_service

    hits = asyncio.run(
        get_retrieval_service().retrieve_for(
            conversation_id=conversation["id"],
            user_message="How does hypothetical document expansion help semantic search?",
            output_type="text",
            options={"hyde_text": "HyDE hypothetical course excerpt semantic vectors retrieval"},
        )
    )
    assert hits
    via = {
        item
        for hit in hits
        for item in (hit.metadata.get("retrieval_via") if isinstance(hit.metadata.get("retrieval_via"), list) else [hit.metadata.get("retrieval_via")])
    }
    assert "dense_vector" in via or "hyde_dense" in via or "knowledge_graph" in via
    assert any(hit.metadata.get("retrieval_score_type") == "reranker" for hit in hits)


def test_platform_like_generators_and_coursebuilder(monkeypatch, tmp_path) -> None:
    client = _client(monkeypatch, tmp_path)
    conversation = client.post("/api/conversations", json={"title": "Platform parity"}).json()
    upload = client.post(
            f"/api/conversations/{conversation['id']}/files",
        files={
            "upload": (
                "course.md",
                b"""# Retrieval Systems

Course code: TLM-101
Author: Dr. Jane Smith
Semester: 2025
Institution: Example University

Retrieval augmented generation uses indexed source chunks to answer questions with citations.

## Hybrid Search

Hybrid search combines exact lexical matching with semantic similarity. It helps students find precise definitions and broader related ideas.

## Evaluation

Grounded answers should cite the uploaded chunks. A quiz should test concepts with options supported by those chunks.

## Podcast Review

A good podcast follows a narrative arc: introduction, key points, examples, and wrap-up.
""",
                "text/markdown",
            )
        },
    )
    assert upload.status_code == 200
    file_id = upload.json()["id"]
    stored = client.get(f"/api/conversations/{conversation['id']}/files/{file_id}").json()
    assert stored["status"] == "ready"

    from local_api.db import get_store
    import local_api.services.generators as generators_module

    course_chunks = get_store().list_chunks(conversation["id"], source_file_ids=[file_id])

    def chunk_id_for(text: str) -> str:
        return next(row["id"] for row in course_chunks if text.casefold() in row["text"].casefold())

    quiz_llm_calls: list[dict] = []

    async def fake_quiz_llm(_provider, messages, *, json_schema=None, temperature=0.2):
        quiz_llm_calls.append({"messages": messages, "json_schema": json_schema, "temperature": temperature})
        return json.dumps(
            {
                "title": f"Fresh Retrieval Quiz {len(quiz_llm_calls)}",
                "intro_message": "Test your understanding of the selected retrieval material.",
                "questions": [
                    {
                        "type": "mcq",
                        "category": "mechanism",
                        "bloom_level": "understand",
                        "question": "How does retrieval augmented generation support grounded answers?",
                        "options": [
                            "It uses indexed source chunks to answer questions with citations.",
                            "It removes all source evidence before producing an answer.",
                            "It ranks responses only by their visual formatting choices.",
                            "It replaces retrieval with an unrelated random selection process.",
                        ],
                        "correct_index": 0,
                        "explanation": "Retrieval augmented generation uses indexed source chunks for cited answers.",
                        "concept": "Retrieval augmented generation",
                        "source_chunk_id": chunk_id_for("Retrieval augmented generation"),
                    },
                    {
                        "type": "mcq",
                        "category": "relationship",
                        "bloom_level": "understand",
                        "question": "Which combination correctly describes hybrid search?",
                        "options": [
                            "It combines exact lexical matching with semantic similarity.",
                            "It combines color matching with chronological sorting alone.",
                            "It ignores both precise terms and broader related meanings.",
                            "It relies entirely on an unranked collection of unrelated items.",
                        ],
                        "correct_index": 0,
                        "explanation": "Hybrid search combines lexical matching and semantic similarity.",
                        "concept": "Hybrid search",
                        "source_chunk_id": chunk_id_for("Hybrid search combines"),
                    },
                    {
                        "type": "mcq",
                        "category": "application",
                        "bloom_level": "apply",
                        "question": "What should a grounded answer provide for its claims?",
                        "options": [
                            "It should cite the source chunks that support the answer.",
                            "It should hide every piece of supporting course evidence.",
                            "It should replace evidence with unsupported personal guesses.",
                            "It should choose claims solely because they sound confident.",
                        ],
                        "correct_index": 0,
                        "explanation": "Grounded answers cite the source chunks supporting their claims.",
                        "concept": "Grounded answers",
                        "source_chunk_id": chunk_id_for("Grounded answers"),
                    },
                    {
                        "type": "mcq",
                        "category": "classification",
                        "bloom_level": "remember",
                        "question": "Which sequence belongs to a strong podcast narrative arc?",
                        "options": [
                            "Introduction, key points, examples, and a concluding wrap-up.",
                            "A silent opening followed by disconnected labels and no ending.",
                            "Only a timestamp, an empty audio segment, and unrelated markers.",
                            "Random fragments presented without examples or a coherent progression.",
                        ],
                        "correct_index": 0,
                        "explanation": "A good podcast uses an introduction, key points, examples, and wrap-up.",
                        "concept": "Podcast narrative arc",
                        "source_chunk_id": chunk_id_for("A good podcast"),
                    },
                ],
            }
        )

    monkeypatch.setattr(generators_module, "complete_text", fake_quiz_llm)
    provider = client.post(
        "/api/settings/llm-providers",
        json={
            "display_name": "Offline structured quiz model",
            "provider_type": "ollama",
            "model_name": "test-quiz-model",
            "is_default_chat": True,
        },
    )
    assert provider.status_code == 200

    course = client.get(f"/api/conversations/{conversation['id']}/coursebuilder").json()
    assert course["status"] == "ready"
    assert course["chapters"]
    assert course["chapters"][0]["lessons"]
    assert course["chapters"][0]["lessons"][0]["blocks"]
    assert course["chapters"][0]["lessons"][0]["citations"]

    quiz_request = {
        "output_type": "quiz",
        "prompt": "Generate quiz",
        "source_file_ids": [file_id],
        "options": {"question_count": 4},
    }
    quiz_done = _done_event(
        client.post(f"/api/conversations/{conversation['id']}/generate", json=quiz_request).text
    )
    assert quiz_done["output_type"] == "quiz"
    quiz_data = quiz_done["metadata"]["quiz_data"]
    assert len(quiz_data["questions"]) == 4
    assert {question["source_chunk_id"] for question in quiz_data["questions"]}
    assert all(question["type"] == "mcq" for question in quiz_data["questions"])
    assert all(len(question["options"]) == 4 for question in quiz_data["questions"])
    banned_quiz_terms = (
        "according to",
        "author",
        "chapter",
        "course code",
        "document",
        "dr. jane",
        "mentioned",
        "page",
        "semester",
        "the lecture",
        "this course",
        "uploaded",
        "which statement",
    )
    quiz_text = "\n".join(
        "\n".join([question["question"], *question.get("options", [])])
        for question in quiz_data["questions"]
    ).casefold()
    assert not any(term in quiz_text for term in banned_quiz_terms)
    assert {question.get("category") for question in quiz_data["questions"]} <= {
        "definition",
        "relationship",
        "mechanism",
        "causality",
        "application",
        "classification",
    }
    assert "bloom_distribution" in quiz_done["metadata"]
    second_quiz_done = _done_event(
        client.post(f"/api/conversations/{conversation['id']}/generate", json=quiz_request).text
    )
    assert quiz_done["metadata"]["generation_mode"] == "fresh_quiz"
    assert quiz_done["metadata"]["fresh_generation"] is True
    assert quiz_done["metadata"]["rebuild_from_scratch"] is True
    assert quiz_done["metadata"]["retrieval_mode"] == "full_selected_files_with_graph"
    assert quiz_done["metadata"]["source_file_ids"] == [file_id]
    assert quiz_done["metadata"]["source_chunk_count"] > 0
    assert quiz_done["metadata"]["graph_search_used"] is True
    assert quiz_done["metadata"]["graph_search"]["complete"] is True
    assert quiz_done["metadata"]["graph_node_count"] > 0
    assert quiz_done["metadata"]["synthesis"]["backend"] == "llm_structured_fresh_generation"
    assert quiz_done["metadata"]["synthesis"]["question_type_sent"] == "mcq"
    assert quiz_done["metadata"]["synthesis"]["question_count_sent"] == 4
    assert len(quiz_llm_calls) == 2
    assert all(call["json_schema"] for call in quiz_llm_calls)
    assert quiz_done["metadata"]["generation_run_id"] != second_quiz_done["metadata"]["generation_run_id"]
    assert quiz_done["artifacts"][0]["key"] != second_quiz_done["artifacts"][0]["key"]

    mindmap_done = _done_event(
        client.post(
            f"/api/conversations/{conversation['id']}/generate",
            json={"output_type": "mindmap", "prompt": "Generate mindmap", "source_file_ids": [file_id], "options": {}},
        ).text
    )
    assert mindmap_done["output_type"] == "mindmap"
    assert mindmap_done["metadata"]["node_count"] >= 3
    assert mindmap_done["metadata"]["central_topic"]
    assert len(mindmap_done["artifacts"]) >= 2

    podcast_done = _done_event(
        client.post(
            f"/api/conversations/{conversation['id']}/generate",
            json={"output_type": "podcast", "prompt": "Generate podcast", "source_file_ids": [file_id], "options": {}},
        ).text
    )
    assert podcast_done["output_type"] == "podcast"
    assert podcast_done["metadata"]["narrative_arc"]["key_points"]
    assert podcast_done["metadata"]["podcast"]["transcript"]
    assert podcast_done["metadata"]["podcast"]["tts_skipped"] is True
    assert podcast_done["metadata"]["podcast"]["audio_status"] == "failed"
    assert podcast_done["metadata"]["podcast"]["audio_error_code"] == "tts_unavailable"
    assert [artifact["type"] for artifact in podcast_done["artifacts"]] == ["transcript"]


def test_mindmap_rebuilds_from_all_selected_file_chunks(monkeypatch, tmp_path) -> None:
    client = _client(monkeypatch, tmp_path)
    conversation = client.post("/api/conversations", json={"title": "Mind map full context"}).json()
    first = client.post(
        f"/api/conversations/{conversation['id']}/files",
        files={
            "upload": (
                "first.md",
                b"""# Search Foundations

## Dense Retrieval
Dense retrieval represents meaning with vectors.

## Sparse Retrieval
Sparse retrieval uses exact lexical evidence.

## Fusion
Reciprocal rank fusion combines ranked result lists.
""",
                "text/markdown",
            )
        },
    ).json()
    second = client.post(
        f"/api/conversations/{conversation['id']}/files",
        files={
            "upload": (
                "second.md",
                b"""# Evaluation

## Precision
Precision measures relevant retrieved items.

## Recall
Recall measures how many relevant items were found.
""",
                "text/markdown",
            )
        },
    ).json()

    from local_api.db import get_store
    from local_api.services.retrieval import get_retrieval_service

    retrieval_settings = client.patch(
        "/api/settings/retrieval",
        json={"retrieval_graph_enabled": False},
    ).json()
    assert retrieval_settings["retrieval_graph_enabled"] is False

    selected_rows = get_store().list_chunks(conversation["id"], source_file_ids=[first["id"]])
    assert len(selected_rows) >= 3
    selected_context = asyncio.run(
        get_retrieval_service().retrieve_for(
            conversation_id=conversation["id"],
            user_message="Generate mindmap",
            output_type="mindmap",
            source_file_ids=[first["id"]],
        )
    )
    full_context = [chunk for chunk in selected_context if chunk.metadata.get("mindmap_full_context") is True]
    graph_context = [
        chunk for chunk in selected_context if chunk.metadata.get("context_type") == "mindmap_graph_context"
    ]
    assert [chunk.chunk_id for chunk in full_context] == [row["id"] for row in selected_rows]
    assert {chunk.metadata.get("source_file_id") for chunk in full_context} == {first["id"]}
    assert second["id"] not in {chunk.metadata.get("source_file_id") for chunk in full_context}
    assert graph_context
    assert graph_context[0].metadata["retrieval_via"] == "knowledge_graph"
    assert graph_context[0].metadata["retrieval_mode"] == "graph_search"
    assert graph_context[0].metadata["graph_complete"] is True
    assert graph_context[0].metadata["graph_node_count"] > 0
    assert graph_context[0].metadata["source_file_ids"] == [first["id"]]
    assert set(graph_context[0].metadata["source_chunk_ids"]) == {row["id"] for row in selected_rows}
    assert {"course", "file", "section", "chunk"} <= {
        node["node_type"] for node in graph_context[0].metadata["graph_nodes"]
    }
    assert any(edge["relation_type"] == "part_of" for edge in graph_context[0].metadata["graph_edges"])
    assert second["id"] not in {
        str(node.get("metadata", {}).get("source_file_id") or "")
        for node in graph_context[0].metadata["graph_nodes"]
    }

    selected_quiz_context = asyncio.run(
        get_retrieval_service().retrieve_for(
            conversation_id=conversation["id"],
            user_message="Generate quiz",
            output_type="quiz",
            source_file_ids=[first["id"]],
        )
    )
    quiz_full_context = [
        chunk for chunk in selected_quiz_context if chunk.metadata.get("quiz_full_context") is True
        and chunk.metadata.get("context_type") != "quiz_graph_context"
    ]
    quiz_graph_context = [
        chunk for chunk in selected_quiz_context if chunk.metadata.get("context_type") == "quiz_graph_context"
    ]
    assert [chunk.chunk_id for chunk in quiz_full_context] == [row["id"] for row in selected_rows]
    assert {chunk.metadata.get("source_file_id") for chunk in quiz_full_context} == {first["id"]}
    assert second["id"] not in {chunk.metadata.get("source_file_id") for chunk in quiz_full_context}
    assert quiz_graph_context
    assert quiz_graph_context[0].metadata["graph_complete"] is True
    assert quiz_graph_context[0].metadata["source_file_ids"] == [first["id"]]
    assert set(quiz_graph_context[0].metadata["source_chunk_ids"]) == {row["id"] for row in selected_rows}
    assert second["id"] not in {
        str(node.get("metadata", {}).get("source_file_id") or "")
        for node in quiz_graph_context[0].metadata["graph_nodes"]
    }

    empty_scope_error = _error_event(
        client.post(
            f"/api/conversations/{conversation['id']}/generate",
            json={"output_type": "mindmap", "prompt": "Generate mindmap", "source_file_ids": [], "options": {}},
        ).text
    )
    assert "Select at least one ready source file" in empty_scope_error["message"]

    empty_podcast_error = _error_event(
        client.post(
            f"/api/conversations/{conversation['id']}/generate",
            json={"output_type": "podcast", "prompt": "Generate podcast", "source_file_ids": [], "options": {}},
        ).text
    )
    assert "Select at least one ready source file" in empty_podcast_error["message"]

    request = {
        "output_type": "mindmap",
        "prompt": "Generate mindmap",
        "source_file_ids": [first["id"]],
        "options": {},
    }
    first_run = _done_event(client.post(f"/api/conversations/{conversation['id']}/generate", json=request).text)
    second_run = _done_event(client.post(f"/api/conversations/{conversation['id']}/generate", json=request).text)
    assert first_run["metadata"]["generation_mode"] == "full_rebuild"
    assert first_run["metadata"]["source_chunk_count"] == len(selected_rows)
    assert first_run["metadata"]["source_file_count"] == 1
    assert first_run["metadata"]["source_file_ids"] == [first["id"]]
    assert first_run["metadata"]["rebuild_from_scratch"] is True
    assert first_run["metadata"]["generation_run_id"]
    assert first_run["metadata"]["generation_run_id"] != second_run["metadata"]["generation_run_id"]
    assert first_run["metadata"]["graph_search_used"] is True
    assert first_run["metadata"]["graph_search"]["complete"] is True
    assert first_run["metadata"]["graph_node_count"] > 0
    assert first_run["metadata"]["central_topic"] == "Search Foundations"
    assert {"Dense Retrieval", "Sparse Retrieval", "Fusion"} <= set(first_run["metadata"]["main_branches"])
    assert first_run["artifacts"][0]["key"] != second_run["artifacts"][0]["key"]


def test_mindmap_graph_context_enriches_chunk_hierarchy() -> None:
    from teacherlm_core.schemas.chunk import Chunk
    from local_api.services.generators import _build_mindmap

    source_chunks = [
        Chunk(
            text=text,
            source="search.md",
            score=1.0,
            chunk_id=chunk_id,
            metadata={
                "heading_path_list": ["Search Foundations", heading],
                "mindmap_full_context": True,
            },
        )
        for chunk_id, heading, text in [
            ("dense", "Dense Retrieval", "Dense retrieval compares vector representations."),
            ("sparse", "Sparse Retrieval", "Sparse retrieval matches lexical evidence."),
            ("fusion", "Fusion", "Fusion combines ranked retrieval results."),
        ]
    ]
    graph_context = Chunk(
        text="Knowledge graph context",
        source="knowledge_graph",
        score=1.0,
        chunk_id="mindmap-graph:test",
        metadata={
            "context_type": "mindmap_graph_context",
            "graph_nodes": [
                {
                    "id": "vector-similarity",
                    "label": "Vector Similarity",
                    "node_type": "concept",
                    "description": "Compares dense vector representations.",
                    "source_chunk_ids": ["dense"],
                },
                {
                    "id": "sparse-retrieval",
                    "label": "Sparse Retrieval",
                    "node_type": "concept",
                    "description": "",
                    "source_chunk_ids": ["sparse"],
                },
            ],
            "graph_edges": [
                {
                    "source_node_id": "vector-similarity",
                    "target_node_id": "sparse-retrieval",
                    "relation_type": "contrasts_with",
                    "confidence": 0.95,
                }
            ],
            "graph_node_count": 2,
            "graph_edge_count": 1,
        },
    )

    mindmap = _build_mindmap([*source_chunks, graph_context], "Generate mindmap", {})
    markdown = json.dumps(mindmap)
    assert "Vector Similarity" in markdown
    assert "Contrasts with: Sparse Retrieval" in markdown


def test_mindmap_fresh_rebuild_calls_structured_model_every_time(monkeypatch) -> None:
    from teacherlm_core.llm.providers import LLMProviderConfig
    from teacherlm_core.schemas import GeneratorInput, LearnerState
    import local_api.services.generators as generators_module

    provider = LLMProviderConfig(
        provider_id="provider-test",
        display_name="Test provider",
        provider_type="ollama",
        model_name="test-model",
    )
    monkeypatch.setattr(
        generators_module,
        "get_settings_service",
        lambda: types.SimpleNamespace(get_default_chat_provider_config=lambda: provider),
    )
    calls: list[dict] = []

    async def fake_complete_text(_provider, messages, *, json_schema=None, temperature=0.2):
        calls.append({"messages": messages, "json_schema": json_schema, "temperature": temperature})
        run_number = len(calls)
        return json.dumps(
            {
                "central_topic": "Conceptual Retrieval Map" if run_number == 1 else "Applied Retrieval Map",
                "branches": [
                    {
                        "text": "Dense Retrieval",
                        "children": [
                            {"text": "Vector Similarity", "children": []},
                            {"text": "Semantic Matching", "children": []},
                        ],
                    },
                    {
                        "text": "Sparse Retrieval",
                        "children": [
                            {"text": "Lexical Evidence", "children": []},
                            {"text": "Exact Terms", "children": []},
                        ],
                    },
                    {
                        "text": "Fusion",
                        "children": [
                            {"text": "Ranked Lists", "children": []},
                            {"text": "Combined Results", "children": []},
                        ],
                    },
                ],
            }
        )

    monkeypatch.setattr(generators_module, "complete_text", fake_complete_text)
    chunks = [
        generators_module.Chunk(
            text=text,
            source="search.md",
            score=1.0,
            chunk_id=chunk_id,
            metadata={
                "heading_path_list": ["Search Foundations", heading],
                "mindmap_full_context": True,
            },
        )
        for chunk_id, heading, text in [
            ("dense", "Dense Retrieval", "Dense retrieval uses vector similarity and semantic matching."),
            ("sparse", "Sparse Retrieval", "Sparse retrieval uses lexical evidence and exact terms."),
            ("fusion", "Fusion", "Fusion combines ranked result lists."),
        ]
    ]

    def payload(run_id: str) -> GeneratorInput:
        return GeneratorInput(
            conversation_id="conversation-test",
            user_message="Generate mindmap",
            context_chunks=chunks,
            learner_state=LearnerState(conversation_id="conversation-test"),
            chat_history=[],
            options={"generation_run_id": run_id, "max_nodes": 40},
        )

    first, first_meta = asyncio.run(generators_module._build_fresh_mindmap(payload("run-one")))
    second, second_meta = asyncio.run(generators_module._build_fresh_mindmap(payload("run-two")))

    assert len(calls) == 2
    assert calls[0]["json_schema"]
    assert calls[0]["temperature"] == 0.7
    assert "run-one" in calls[0]["messages"][-1].content
    assert "run-two" in calls[1]["messages"][-1].content
    assert first["central_topic"] != second["central_topic"]
    assert first_meta["backend"] == "llm_structured_fresh_rebuild"
    assert second_meta["backend"] == "llm_structured_fresh_rebuild"


def test_mindmap_rejects_low_coverage_output_when_sources_support_more() -> None:
    import local_api.services.generators as generators

    raw = json.dumps(
        {
            "central_topic": "Recommendation Systems",
            "branches": [
                {
                    "text": f"Branch {branch_index}",
                    "children": [
                        {"text": f"Concept {branch_index}.{child_index}", "children": []}
                        for child_index in range(1, 4)
                    ],
                }
                for branch_index in range(1, 6)
            ],
        }
    )

    compact = generators._validated_llm_mindmap(raw, max_nodes=110, min_nodes=8)
    assert generators._count_mindmap_nodes(compact) == 21
    with pytest.raises(ValueError, match="too shallow"):
        generators._validated_llm_mindmap(raw, max_nodes=110, min_nodes=35)


def _done_event(raw_sse: str) -> dict:
    for block in raw_sse.split("\n\n"):
        lines = block.splitlines()
        if "event: done" not in lines:
            continue
        data_line = next(line for line in lines if line.startswith("data: "))
        return json.loads(data_line.removeprefix("data: "))
    raise AssertionError(f"no done event in SSE:\n{raw_sse}")


def _error_event(raw_sse: str) -> dict:
    for block in raw_sse.split("\n\n"):
        lines = block.splitlines()
        if "event: error" not in lines:
            continue
        data_line = next(line for line in lines if line.startswith("data: "))
        return json.loads(data_line.removeprefix("data: "))
    raise AssertionError(f"no error event in SSE:\n{raw_sse}")
