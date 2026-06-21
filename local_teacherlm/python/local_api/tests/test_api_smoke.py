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
    import local_api.services.coursebuilder as coursebuilder_module
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
    coursebuilder_module._coursebuilder_service = None

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

    class FakeAsyncClient:
        def __init__(self, *, api_key: str, base_url: str | None = None) -> None:
            self.api_key = api_key
            self.base_url = base_url
            self.parsing = FakeParsing()

        async def __aenter__(self):
            return self

        async def __aexit__(self, _exc_type, _exc, _traceback):
            return False

    monkeypatch.setitem(sys.modules, "llama_cloud", types.SimpleNamespace(AsyncClient=FakeAsyncClient))

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


def test_coursebuilder_progression_setting_switches_lock_policy_without_losing_progress(monkeypatch, tmp_path) -> None:
    client = _client(monkeypatch, tmp_path)
    from local_api.services.coursebuilder import get_coursebuilder_service

    course = {
        "id": "course_progression_test",
        "conversation_id": "conversation_progression_test",
        "status": "ready",
        "source_fingerprint": "progression-test",
        "title": "Progression test",
        "chapters": [
            {
                "id": "chapter_1",
                "title": "Chapter 1",
                "generation_status": "ready",
                "content_fingerprint": "chapter-1",
                "lessons": [
                    {"id": "lesson_1_1", "generation_status": "ready"},
                    {"id": "lesson_1_2", "generation_status": "ready"},
                ],
                "quiz": {"id": "quiz_1", "questions": [], "pass_score": 0.7},
            },
            {
                "id": "chapter_2",
                "title": "Chapter 2",
                "generation_status": "ready",
                "content_fingerprint": "chapter-2",
                "lessons": [
                    {"id": "lesson_2_1", "generation_status": "ready"},
                    {"id": "lesson_2_2", "generation_status": "ready"},
                ],
                "quiz": {"id": "quiz_2", "questions": [], "pass_score": 0.7},
            },
        ],
        "final_quiz": {"id": "quiz_final", "questions": [], "pass_score": 0.7},
        "metadata": {},
    }
    service = get_coursebuilder_service()
    service._save_course(course, build_id="build_progression_test", quality_mode="fallback")

    settings = client.get("/api/settings/coursebuilder")
    assert settings.status_code == 200
    assert settings.json() == {"sequential_unlocking_enabled": True}

    strict = service._public_course(course)
    assert strict["chapters"][0]["is_locked"] is False
    assert strict["chapters"][0]["lessons"][0]["is_locked"] is False
    assert strict["chapters"][0]["lessons"][1]["is_locked"] is True
    assert strict["chapters"][1]["is_locked"] is True

    opened = client.patch(
        "/api/settings/coursebuilder",
        json={"sequential_unlocking_enabled": False},
    )
    assert opened.status_code == 200
    assert opened.json() == {"sequential_unlocking_enabled": False}

    free = service._public_course(course)
    assert all(not chapter["is_locked"] for chapter in free["chapters"])
    assert all(
        not lesson["is_locked"]
        for chapter in free["chapters"]
        for lesson in chapter["lessons"]
    )
    assert all(chapter["quiz"]["is_locked"] for chapter in free["chapters"])
    assert free["final_quiz"]["is_locked"] is True

    progress = service._load_progress(course["conversation_id"], course)
    progress["completed_lesson_ids"] = ["lesson_2_1", "lesson_2_2"]
    service._save_progress(course["conversation_id"], course, progress)
    free_with_progress = service._public_course(course)
    assert free_with_progress["chapters"][1]["quiz"]["is_locked"] is False

    relocked = client.patch(
        "/api/settings/coursebuilder",
        json={"sequential_unlocking_enabled": True},
    )
    assert relocked.status_code == 200
    strict_with_progress = service._public_course(course)
    assert strict_with_progress["chapters"][1]["is_locked"] is False
    assert all(not lesson["is_locked"] for lesson in strict_with_progress["chapters"][1]["lessons"])
    assert strict_with_progress["chapters"][1]["quiz"]["is_locked"] is True

    progress["passed_quiz_ids"] = ["quiz_1", "quiz_2"]
    service._save_progress(course["conversation_id"], course, progress)
    completed = service._public_course(course)
    assert completed["final_quiz"]["is_locked"] is False
    assert client.get("/api/settings/coursebuilder").json() == {"sequential_unlocking_enabled": True}


def test_coursebuilder_rebuild_reconciles_progress_by_stable_item_fingerprint(monkeypatch, tmp_path) -> None:
    _client(monkeypatch, tmp_path)
    from local_api.services.coursebuilder import get_coursebuilder_service

    service = get_coursebuilder_service()
    old_course = {
        "id": "course_progress_rebuild",
        "conversation_id": "conversation_progress_rebuild",
        "source_fingerprint": "source-old",
        "status": "ready",
        "chapters": [
            {
                "id": "chapter-stable",
                "content_fingerprint": "chapter-old",
                "lessons": [
                    {"id": "lesson-stable", "content_fingerprint": "lesson-fingerprint-stable"},
                    {"id": "lesson-changed", "content_fingerprint": "lesson-fingerprint-old"},
                ],
                "quiz": {
                    "id": "quiz-stable",
                    "questions": [{"prompt": "Stable question", "source_chunk_ids": ["chunk-stable"]}],
                },
            }
        ],
        "final_quiz": None,
    }
    progress = service._load_progress(old_course["conversation_id"], old_course)
    progress.update(
        {
            "completed_lesson_ids": ["lesson-stable", "lesson-changed"],
            "passed_quiz_ids": ["quiz-stable"],
            "quiz_scores": {"quiz-stable": 1.0},
            "quiz_attempt_counts": {"quiz-stable": 2},
        }
    )
    service._save_progress(old_course["conversation_id"], old_course, progress)

    rebuilt = json.loads(json.dumps(old_course))
    rebuilt["source_fingerprint"] = "source-new"
    rebuilt["chapters"][0]["content_fingerprint"] = "chapter-new"
    rebuilt["chapters"][0]["lessons"][1]["content_fingerprint"] = "lesson-fingerprint-new"
    service._reconcile_progress(rebuilt)
    reconciled = service._load_progress(rebuilt["conversation_id"], rebuilt)

    assert reconciled["completed_lesson_ids"] == ["lesson-stable"]
    assert reconciled["passed_quiz_ids"] == ["quiz-stable"]
    assert reconciled["quiz_scores"] == {"quiz-stable": 1.0}
    assert reconciled["quiz_attempt_counts"] == {"quiz-stable": 2}


def test_manual_coursebuilder_rebuild_forces_a_fresh_structural_plan(monkeypatch, tmp_path) -> None:
    _client(monkeypatch, tmp_path)
    import local_api.services.coursebuilder as coursebuilder

    service = coursebuilder.get_coursebuilder_service()
    chunks = [
        {
            "id": "chunk-manual-rebuild",
            "source_file_id": "file-manual-rebuild",
            "source_filename": "manual.md",
            "chunk_index": 0,
            "text": "A grounded lesson explains the concept and its supported relationships in sufficient detail.",
            "metadata": {"heading_path": "Chapter > Lesson", "heading_path_list": ["Chapter", "Lesson"]},
        }
    ]
    files = [{"id": "file-manual-rebuild", "status": "ready", "created_at": "2026-01-01"}]
    fingerprint = coursebuilder._source_fingerprint(files, chunks)
    outline = coursebuilder.CourseOutline(
        title="Manual rebuild",
        chapters=[
            coursebuilder.OutlineChapter(
                title="Chapter",
                source_chunk_ids=["chunk-manual-rebuild"],
                lessons=[coursebuilder.OutlineLesson(title="Lesson", source_chunk_ids=["chunk-manual-rebuild"])],
            )
        ],
    )
    seen_force: list[bool] = []
    seen_improved_quality: list[bool] = []

    async def prepared(
        _conversation_id: str,
        *,
        force: bool = False,
        improved_quality: bool = False,
    ):
        seen_force.append(force)
        seen_improved_quality.append(improved_quality)
        return {
            "id": "courseplan_conversation-manual-rebuild",
            "plan_id": "plan-manual-rebuild",
            "conversation_id": "conversation-manual-rebuild",
            "contract_version": coursebuilder.COURSE_PLAN_CONTRACT_VERSION,
            "source_fingerprint": fingerprint,
            "status": "draft",
            "outline": outline.model_dump(mode="json"),
            "metadata": {"quality_mode": "fallback", "structure_mode": "source_exact"},
        }

    class OfflineSettings:
        def get_default_chat_provider_config(self):
            return None

        def get_coursebuilder_settings(self):
            return types.SimpleNamespace(sequential_unlocking_enabled=True)

    monkeypatch.setattr(service, "_ready_material", lambda _conversation_id: (files, chunks, fingerprint))
    monkeypatch.setattr(service, "prepare_plan_async", prepared)
    monkeypatch.setattr(coursebuilder, "_course_graph", lambda _conversation_id: {"nodes": [], "edges": []})
    monkeypatch.setattr(coursebuilder, "get_settings_service", lambda: OfflineSettings())

    result = asyncio.run(
        service.rebuild_async(
            "conversation-manual-rebuild",
            force=True,
            improved_quality=True,
        )
    )

    assert seen_force == [True]
    assert seen_improved_quality == [True]
    assert result["status"] == "ready"
    assert result["chapters"][0]["title"] == "Chapter"


def test_coursebuilder_stop_cancels_active_work_and_persists_stopped_state(monkeypatch, tmp_path) -> None:
    client = _client(monkeypatch, tmp_path)
    import local_api.services.coursebuilder as coursebuilder

    conversation = client.post("/api/conversations", json={"title": "Stop build"}).json()
    service = coursebuilder.get_coursebuilder_service()
    chunks = [
        {
            "id": "chunk-stop-build",
            "conversation_id": conversation["id"],
            "source_file_id": "file-stop-build",
            "source_filename": "course.md",
            "chunk_index": 0,
            "text": "Grounded source material for a cancellable course build.",
            "metadata": {"heading_path_list": ["Chapter", "Lesson"], "section_title": "Lesson"},
        }
    ]
    payload = coursebuilder._build_fallback_course(conversation["id"], chunks, "fingerprint-stop")
    payload["status"] = "building"
    payload["metadata"]["stage"] = "generating_chapter"
    service._save_course(payload, build_id="build-stop", quality_mode="fallback")
    started = asyncio.Event()

    async def slow_build(
        _conversation_id: str,
        *,
        force: bool,
        improved_quality: bool,
        cancel_event: asyncio.Event,
    ):
        del force, improved_quality, cancel_event
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(service, "_rebuild_async_impl", slow_build)

    async def scenario() -> tuple[dict, dict]:
        cancel_event = service.begin_build(conversation["id"], force=False)
        task = asyncio.create_task(
            service.rebuild_async(conversation["id"], force=False, _cancel_event=cancel_event)
        )
        await started.wait()
        stopped = service.stop_build(conversation["id"])
        result = await asyncio.wait_for(task, timeout=2)
        return stopped, result

    stopped, result = asyncio.run(scenario())

    assert stopped["status"] == "stopped"
    assert result["status"] == "stopped"
    assert result["metadata"]["stopped_from_stage"] == "generating_chapter"
    assert result["chapters"]
    assert client.post(f"/api/conversations/{conversation['id']}/coursebuilder/stop").json()["status"] == "stopped"
    stored = service._private_course(conversation["id"])
    assert stored["status"] == "stopped"

    service.begin_build(conversation["id"], force=False)
    queued = service.mark_build_queued(conversation["id"], resuming=True)
    assert queued["status"] == "building"
    stopped_before_task_start = service.stop_build(conversation["id"])
    assert stopped_before_task_start["status"] == "stopped"
    assert stopped_before_task_start["metadata"]["stopped_from_stage"] == "generating_chapter"


def test_coursebuilder_resume_keeps_matching_completed_chapter_prefix() -> None:
    import local_api.services.coursebuilder as coursebuilder

    chunks = [
        {
            "id": "chunk-resume",
            "conversation_id": "conversation-resume",
            "source_file_id": "file-resume",
            "source_filename": "resume.md",
            "chunk_index": 0,
            "text": "A detailed grounded explanation supports resuming this completed chapter.",
            "metadata": {"heading_path_list": ["Chapter", "Lesson"], "section_title": "Lesson"},
        }
    ]
    course = coursebuilder._build_fallback_course("conversation-resume", chunks, "fingerprint")
    outline = coursebuilder._outline_from_course(course)

    resumed = coursebuilder._resumable_chapter_prefix(
        "conversation-resume",
        outline,
        course["chapters"],
    )

    assert [chapter["id"] for chapter in resumed] == [course["chapters"][0]["id"]]
    changed = json.loads(json.dumps(course["chapters"]))
    changed[0]["content_fingerprint"] = "changed"
    assert coursebuilder._resumable_chapter_prefix("conversation-resume", outline, changed) == []


def test_coursebuilder_mastery_gates_final_quiz_and_rich_blocks(monkeypatch, tmp_path) -> None:
    client = _client(monkeypatch, tmp_path)
    conversation = client.post("/api/conversations", json={"title": "Cumulative science history"}).json()
    uploads = [
        (
            "foundations.md",
            b"""# Mathematical Foundations

## Variables and Equations
An equation relates quantities and provides the foundation for later calculations.

$$E = mc^2$$

## Matrices
Matrices organize values so transformations can be calculated consistently.

$$A = \\begin{pmatrix}1 & 0 \\\\ 0 & 1\\end{pmatrix}$$

| Symbol | Meaning |
| --- | --- |
| E | Energy |
| m | Mass |

In 1905 Einstein published work connecting mass and energy.
""",
        ),
        (
            "applications.md",
            b"""# Scientific Applications

## Chemical Reactions
Balanced reactions preserve the number of atoms on both sides.

2H2 + O2 -> 2H2O

## Historical Development
In 1945 scientific institutions entered a new period of international collaboration.
Applications depend on the mathematical and scientific foundations introduced earlier.
""",
        ),
    ]
    for filename, body in uploads:
        response = client.post(
            f"/api/conversations/{conversation['id']}/files",
            files={"upload": (filename, body, "text/markdown")},
        )
        assert response.status_code == 200

    course_url = f"/api/conversations/{conversation['id']}/coursebuilder"
    course = client.get(course_url).json()
    assert course["status"] == "ready"
    assert len(course["chapters"]) == 2
    assert course["chapters"][0]["is_locked"] is False
    assert course["chapters"][1]["is_locked"] is True

    block_types = {
        block["block_type"]
        for chapter in course["chapters"]
        for lesson in chapter["lessons"]
        for block in lesson["blocks"]
    }
    assert {"equation", "matrix", "chemical_equation", "table", "timeline"} <= block_types

    locked_quiz = course["chapters"][0]["quiz"]
    locked_submit = client.post(
        f"{course_url}/quizzes/{locked_quiz['id']}/submit",
        json={"answers": []},
    )
    assert locked_submit.status_code == 409

    from local_api.db import get_store

    for chapter_index in range(len(course["chapters"])):
        course = client.get(course_url).json()
        chapter = course["chapters"][chapter_index]
        assert chapter["is_locked"] is False
        for lesson in chapter["lessons"]:
            refreshed = client.get(course_url).json()
            current_lesson = next(
                item
                for item in refreshed["chapters"][chapter_index]["lessons"]
                if item["id"] == lesson["id"]
            )
            assert current_lesson["is_locked"] is False
            completed = client.post(f"{course_url}/lessons/{lesson['id']}/complete")
            assert completed.status_code == 200

        public_course = client.get(course_url).json()
        public_quiz = public_course["chapters"][chapter_index]["quiz"]
        assert public_quiz["is_locked"] is False
        assert all("correct_option_id" not in question for question in public_quiz["questions"])
        assert 4 <= len(public_quiz["questions"]) <= 10

        private_row = get_store().one(
            "SELECT payload_json FROM coursebuilder_courses WHERE conversation_id = ?",
            (conversation["id"],),
        )
        private_course = json.loads(private_row["payload_json"])
        private_quiz = private_course["chapters"][chapter_index]["quiz"]
        correct_by_question = {
            question["id"]: question["correct_option_id"]
            for question in private_quiz["questions"]
        }
        submitted = client.post(
            f"{course_url}/quizzes/{public_quiz['id']}/submit",
            json={
                "answers": [
                    {"question_id": question["id"], "option_id": correct_by_question[question["id"]]}
                    for question in public_quiz["questions"]
                ]
            },
        )
        assert submitted.status_code == 200
        assert submitted.json()["passed"] is True

    course = client.get(course_url).json()
    final_quiz = course["final_quiz"]
    assert final_quiz["is_locked"] is False
    assert len(final_quiz["questions"]) == 10

    private_row = get_store().one(
        "SELECT payload_json FROM coursebuilder_courses WHERE conversation_id = ?",
        (conversation["id"],),
    )
    private_final = json.loads(private_row["payload_json"])["final_quiz"]
    private_questions = {question["id"]: question for question in private_final["questions"]}
    wrong_answers = []
    for question in final_quiz["questions"]:
        private_question = private_questions[question["id"]]
        wrong_option = next(
            option["id"]
            for option in private_question["options"]
            if option["id"] != private_question["correct_option_id"]
        )
        wrong_answers.append({"question_id": question["id"], "option_id": wrong_option})
    failed = client.post(
        f"{course_url}/quizzes/{final_quiz['id']}/submit",
        json={"answers": wrong_answers},
    ).json()
    assert failed["passed"] is False
    assert failed["review_lesson_ids"]

    retry_course = client.get(course_url).json()
    retry_quiz = retry_course["final_quiz"]
    passed = client.post(
        f"{course_url}/quizzes/{retry_quiz['id']}/submit",
        json={
            "answers": [
                {
                    "question_id": question["id"],
                    "option_id": private_questions[question["id"]]["correct_option_id"],
                }
                for question in retry_quiz["questions"]
            ]
        },
    ).json()
    assert passed["passed"] is True
    assert passed["course"]["progress"]["course_completed"] is True


def test_coursebuilder_structured_synthesis_keeps_every_block_grounded(monkeypatch) -> None:
    from teacherlm_core.llm.providers import LLMProviderConfig
    import local_api.services.coursebuilder as coursebuilder

    chunks = [
        {
            "id": "chunk-foundation",
            "conversation_id": "conv-synthesis",
            "source_file_id": "file-foundation",
            "source_filename": "foundation.md",
            "chunk_index": 0,
            "text": "Vector representations are the foundation for semantic retrieval.",
            "metadata": {
                "heading_path": "Retrieval > Vector Foundations",
                "heading_path_list": ["Retrieval", "Vector Foundations"],
                "section_title": "Vector Foundations",
                "key_concepts": ["Vector representations"],
            },
        },
        {
            "id": "chunk-application",
            "conversation_id": "conv-synthesis",
            "source_file_id": "file-application",
            "source_filename": "application.md",
            "chunk_index": 1,
            "text": "Semantic retrieval applies vector similarity to rank relevant passages.",
            "metadata": {
                "heading_path": "Retrieval > Semantic Retrieval",
                "heading_path_list": ["Retrieval", "Semantic Retrieval"],
                "section_title": "Semantic Retrieval",
                "key_concepts": ["Semantic retrieval"],
            },
        },
    ]
    fallback = coursebuilder._build_fallback_course("conv-synthesis", chunks, "fingerprint")
    provider = LLMProviderConfig(provider_id="test", display_name="Test", provider_type="ollama", model_name="test")

    async def fake_complete(_provider, messages, *, json_schema=None, temperature=0.2):
        properties = (json_schema or {}).get("properties", {})
        if "chapters" in properties:
            return json.dumps(
                {
                    "title": "Retrieval Foundations",
                    "description": "A cumulative retrieval course.",
                    "learning_objectives": ["Connect vector foundations to semantic retrieval"],
                    "chapters": [
                        {
                            "title": "Vector Foundations",
                            "description": "Build the representation foundation.",
                            "learning_objectives": ["Explain vector representations"],
                            "source_chunk_ids": ["chunk-foundation", "chunk-application"],
                            "lessons": [
                                {
                                    "title": "From vectors to retrieval",
                                    "summary": "Vectors prepare semantic retrieval.",
                                    "learning_objectives": ["Connect the two ideas"],
                                    "source_chunk_ids": ["chunk-foundation", "chunk-application"],
                                }
                            ],
                        }
                    ],
                }
            )
        if "lessons" in properties:
            return json.dumps(
                {
                    "summary": "Vector representations support semantic retrieval.",
                    "lessons": [
                        {
                            "title": "From vectors to retrieval",
                            "summary": "Vectors prepare semantic retrieval.",
                            "learning_objectives": ["Connect the two ideas"],
                            "source_chunk_ids": ["chunk-foundation", "chunk-application"],
                            "blocks": [
                                {
                                    "block_type": "markdown",
                                    "title": "Foundation",
                                    "content": "Vector representations provide the basis used by semantic retrieval.",
                                    "source_chunk_ids": ["chunk-foundation", "chunk-application"],
                                }
                            ],
                        }
                    ],
                }
            )
        count_match = re.search(r"Create exactly (\d+)", messages[-1].content)
        count = int(count_match.group(1)) if count_match else 4
        return json.dumps(
            {
                "questions": [
                    {
                        "prompt": f"How do vector foundations support semantic retrieval in case {index + 1}?",
                        "options": [
                            "They provide representations used for similarity ranking.",
                            "They remove the need for ranking.",
                            "They make course evidence unnecessary.",
                            "They replace passages with unrelated labels.",
                        ],
                        "correct_index": 0,
                        "explanation": "The supplied evidence connects vectors to similarity-based retrieval.",
                        "source_chunk_id": "chunk-application",
                    }
                    for index in range(count)
                ]
            }
        )

    monkeypatch.setattr(coursebuilder, "complete_text", fake_complete)
    outline = asyncio.run(coursebuilder._build_outline_with_llm(provider, chunks, fallback))
    chapter = asyncio.run(
        coursebuilder._build_chapter_with_llm(
            provider,
            "conv-synthesis",
            0,
            outline.chapters[0],
            chunks,
            "",
        )
    )
    assert chapter["lessons"][0]["blocks"]
    assert chapter["lessons"][0]["blocks"][0]["source_chunk_ids"] == ["chunk-foundation", "chunk-application"]
    assert [lesson["lesson_stage"] for lesson in chapter["lessons"]] == ["introduction", "content", "conclusion"]
    assert len(chapter["quiz"]["questions"]) == 5
    assert all(question["source_chunk_ids"] == ["chunk-application"] for question in chapter["quiz"]["questions"])


def test_coursebuilder_uses_domain_specific_architectures() -> None:
    import local_api.services.coursebuilder as coursebuilder

    def chunks_for(course_title: str, sections: list[tuple[str, str]]) -> list[dict]:
        return [
            {
                "id": f"{course_title}-{index}",
                "conversation_id": "conv-domain",
                "source_file_id": f"file-{course_title}",
                "source_filename": f"{course_title}.md",
                "chunk_index": index,
                "text": text,
                "metadata": {
                    "heading_path": f"{course_title} > {title}",
                    "heading_path_list": [course_title, title],
                    "section_title": title,
                    "key_concepts": [],
                    "equation_count": int("=" in text),
                },
            }
            for index, (title, text) in enumerate(sections)
        ]

    recommendation = chunks_for(
        "Recommendation Systems",
        [
            ("Evaluation Metrics", "Precision, recall, RMSE and MAE evaluate recommender quality."),
            ("Hybrid Recommenders", "Hybrid recommenders combine collaborative and content-based components."),
            ("Collaborative Filtering", "Collaborative filtering learns from user-item interactions."),
            ("Introduction", "What is a recommendation system and what problem does it solve?"),
            ("Deep Learning Recommenders", "Neural recommenders learn advanced representations."),
            ("Fundamentals", "Users, items, ratings and feedback form the prerequisites."),
            ("Matrix Equations", "The user-item matrix and similarity equations formalize CF."),
            ("Content-Based Filtering", "Content-based filtering uses item features."),
        ],
    )
    history = chunks_for(
        "Revolution History",
        [
            ("Consequences", "The aftermath and legacy followed the revolution."),
            ("Events of 1791", "In 1791 central events changed the political order."),
            ("Background", "The social and economic context developed before 1789."),
            ("Causes", "Long-term causes and immediate factors led to the revolution in 1789."),
        ],
    )
    chemistry = chunks_for(
        "General Chemistry",
        [
            ("Laboratory Safety", "Chemical laboratory safety covers hazards, handling and disposal."),
            ("Reaction Mechanisms", "Reaction mechanisms and kinetics explain transformation pathways."),
            ("Atomic Structure", "Atoms, orbitals, molecules and chemical bonds establish structure."),
            ("Balanced Reactions", "Chemical reactions conserve atoms: 2H2 + O2 -> 2H2O."),
        ],
    )
    physics = chunks_for(
        "Classical Physics",
        [
            ("Experiments", "Experiments measure force, motion and uncertainty."),
            ("Derivation", "A derivation uses calculus to obtain the motion equation."),
            ("Physical Quantities", "Units, vectors and measurement are physics foundations."),
            ("Newton's Laws", "Newton's law relates force and acceleration: F = ma."),
        ],
    )

    cases = [
        (
            recommendation,
            "conceptual",
            ["Introduction", "Fundamentals", "Collaborative Filtering", "Matrix Equations", "Content-Based Filtering", "Hybrid Recommenders", "Deep Learning Recommenders", "Evaluation Metrics"],
        ),
        (history, "historical", ["Background", "Causes", "Events of 1791", "Consequences"]),
        (chemistry, "chemistry", ["Atomic Structure", "Balanced Reactions", "Reaction Mechanisms", "Laboratory Safety"]),
        (physics, "physics", ["Physical Quantities", "Newton's Laws", "Derivation", "Experiments"]),
    ]
    for chunks, expected_architecture, expected_titles in cases:
        architecture = coursebuilder._infer_course_architecture(chunks)
        groups = coursebuilder._chapter_groups(chunks, architecture=architecture)
        assert architecture == expected_architecture
        assert [group["title"] for group in groups] == expected_titles
        assert {chunk["id"] for group in groups for chunk in group["chunks"]} == {chunk["id"] for chunk in chunks}


def test_coursebuilder_has_no_chapter_cap_and_every_chapter_has_a_subchapter_arc() -> None:
    import local_api.services.coursebuilder as coursebuilder

    chunks = [
        {
            "id": f"chunk-{index}",
            "conversation_id": "conv-unbounded",
            "source_file_id": "file-unbounded",
            "source_filename": "large-course.md",
            "chunk_index": index,
            "text": f"Topic {index + 1} contains distinct grounded material and prepares the following topic.",
            "metadata": {
                "heading_path": f"Large Course > Topic {index + 1}",
                "heading_path_list": ["Large Course", f"Topic {index + 1}"],
                "section_title": f"Topic {index + 1}",
                "key_concepts": [f"Topic {index + 1}"],
            },
        }
        for index in range(15)
    ]

    course = coursebuilder._build_fallback_course("conv-unbounded", chunks, "fingerprint")

    assert len(course["chapters"]) == 15
    assert [chapter["title"] for chapter in course["chapters"]] == [f"Topic {index + 1}" for index in range(15)]
    for chapter in course["chapters"]:
        stages = [lesson["lesson_stage"] for lesson in chapter["lessons"]]
        assert len(stages) >= 3
        assert stages[0] == "introduction"
        assert stages[-1] == "conclusion"
        assert "content" in stages[1:-1]
        content_ids = {
            chunk_id
            for lesson in chapter["lessons"]
            if lesson["lesson_stage"] == "content"
            for chunk_id in lesson["source_chunk_ids"]
        }
        assert content_ids == set(chapter["source_chunk_ids"])


def test_coursebuilder_uses_multi_file_units_as_chapters_and_sections_as_subchapters() -> None:
    import local_api.services.coursebuilder as coursebuilder

    chunks = []
    for file_index, (filename, unit_title, sections) in enumerate(
        [
            ("lecture-1.pdf", "Week 1: Foundations", ["Definitions", "Feedback data"]),
            ("lecture-2.pdf", "Week 2: Collaborative Filtering", ["Neighborhoods", "Matrix factorization"]),
            ("lecture-3.pdf", "Week 3: Evaluation", ["Precision and recall", "nDCG"]),
        ]
    ):
        for section_index, section in enumerate(sections):
            chunks.append(
                {
                    "id": f"chunk-{file_index}-{section_index}",
                    "conversation_id": "conv-units",
                    "source_file_id": f"file-{file_index}",
                    "source_filename": filename,
                    "chunk_index": section_index,
                    "text": " ".join([f"{section} contains detailed grounded teaching material."] * 10),
                    "metadata": {
                        "heading_path": f"{unit_title} > {section}",
                        "heading_path_list": [unit_title, section],
                        "section_title": section,
                        "key_concepts": [section],
                    },
                }
            )

    course = coursebuilder._build_fallback_course("conv-units", chunks, "fingerprint")

    assert len(course["chapters"]) == 3
    assert {chapter["title"] for chapter in course["chapters"]} == {
        "Week 1: Foundations",
        "Week 2: Collaborative Filtering",
        "Week 3: Evaluation",
    }
    assert all(len(chapter["lessons"]) == 4 for chapter in course["chapters"])
    assert {
        lesson["title"]
        for chapter in course["chapters"]
        for lesson in chapter["lessons"]
        if lesson["lesson_stage"] == "content"
    } == {"Definitions", "Feedback data", "Neighborhoods", "Matrix factorization", "Precision and recall", "nDCG"}


def test_coursebuilder_fallback_outline_clamps_long_lesson_summaries() -> None:
    import local_api.services.coursebuilder as coursebuilder

    outline = coursebuilder._outline_from_course(
        {
            "title": "Long source course",
            "metadata": {"architecture_type": "conceptual"},
            "chapters": [
                {
                    "title": "Long chapter",
                    "source_chunk_ids": ["chunk-long"],
                    "lessons": [
                        {
                            "title": "Long lesson",
                            "summary": "grounded " * 100,
                            "source_chunk_ids": ["chunk-long"],
                        }
                    ],
                }
            ],
        }
    )

    assert len(outline.chapters[0].lessons[0].summary) == 500


def test_coursebuilder_graph_planning_repairs_omitted_chunk_coverage(monkeypatch) -> None:
    from teacherlm_core.llm.providers import LLMProviderConfig
    import local_api.services.coursebuilder as coursebuilder

    chunks = [
        {
            "id": chunk_id,
            "conversation_id": "conv-graph-course",
            "source_file_id": "file-course",
            "source_filename": "course.md",
            "chunk_index": index,
            "text": text,
            "metadata": {
                "heading_path": f"Recommendation Systems > {title}",
                "heading_path_list": ["Recommendation Systems", title],
                "section_title": title,
                "key_concepts": [title],
            },
        }
        for index, (chunk_id, title, text) in enumerate(
            [
                ("chunk-cf", "Collaborative Filtering", "Collaborative filtering learns from user-item interactions."),
                ("chunk-cbf", "Content-Based Filtering", "Content-based filtering uses item features."),
                ("chunk-hybrid", "Hybrid Recommenders", "Hybrid recommenders combine CF and CBF."),
            ]
        )
    ]
    graph = {
        "nodes": [
            {"id": "node-cf", "node_type": "concept", "label": "Collaborative Filtering", "source_chunk_ids": ["chunk-cf"]},
            {"id": "node-cbf", "node_type": "concept", "label": "Content-Based Filtering", "source_chunk_ids": ["chunk-cbf"]},
            {"id": "node-hybrid", "node_type": "concept", "label": "Hybrid Recommenders", "source_chunk_ids": ["chunk-hybrid"]},
        ],
        "edges": [
            {
                "source_node_id": "node-hybrid",
                "target_node_id": "node-cf",
                "relation_type": "requires",
                "confidence": 0.95,
                "source_chunk_ids": ["chunk-hybrid", "chunk-cf"],
            },
            {
                "source_node_id": "node-hybrid",
                "target_node_id": "node-cbf",
                "relation_type": "requires",
                "confidence": 0.95,
                "source_chunk_ids": ["chunk-hybrid", "chunk-cbf"],
            },
        ],
    }
    fallback = coursebuilder._build_fallback_course("conv-graph-course", chunks, "fingerprint", graph=graph)
    provider = LLMProviderConfig(provider_id="test", display_name="Test", provider_type="ollama", model_name="test")
    prompts: list[str] = []

    async def incomplete_outline(_provider, messages, *, json_schema=None, temperature=0.2):
        prompts.append(messages[-1].content)
        return json.dumps(
            {
                "title": "Recommendation Systems",
                "description": "A prerequisite-driven recommender course.",
                "architecture_type": "conceptual",
                "architecture_rationale": "Components precede their hybrid.",
                "learning_objectives": ["Connect CF, CBF and hybrid recommenders"],
                "chapters": [
                    {
                        "title": "Filtering Foundations",
                        "description": "Learn the component recommenders.",
                        "pedagogical_role": "standard_method",
                        "source_chunk_ids": ["chunk-cf", "chunk-cbf"],
                        "lessons": [
                            {
                                "title": "CF and CBF",
                                "source_chunk_ids": ["chunk-cf", "chunk-cbf"],
                                "pedagogical_role": "standard_method",
                            }
                        ],
                    }
                ],
            }
        )

    monkeypatch.setattr(coursebuilder, "complete_text", incomplete_outline)
    outline = asyncio.run(coursebuilder._build_outline_with_llm(provider, chunks, fallback, graph=graph))
    covered = {
        chunk_id
        for chapter in outline.chapters
        for lesson in chapter.lessons
        for chunk_id in lesson.source_chunk_ids
    }
    assert covered == {"chunk-cf", "chunk-cbf", "chunk-hybrid"}
    assert all(chunk["id"] in prompts[0] for chunk in chunks)
    assert "Hybrid Recommenders --requires--> Collaborative Filtering" in prompts[0]


def test_course_plan_graph_validation_repairs_prerequisite_order() -> None:
    import local_api.services.coursebuilder as coursebuilder

    chunks = [
        {
            "id": chunk_id,
            "source_file_id": "file-course",
            "source_filename": "course.md",
            "chunk_index": index,
            "text": text,
            "metadata": {"heading_path_list": ["Recommendation Systems", title]},
        }
        for index, (chunk_id, title, text) in enumerate(
            [
                ("chunk-hybrid", "Hybrid", "Hybrid recommenders combine component methods."),
                ("chunk-cf", "Collaborative Filtering", "Collaborative filtering is a component method."),
            ]
        )
    ]
    outline = coursebuilder.CourseOutline.model_validate(
        {
            "title": "Recommendation Systems",
            "architecture_type": "conceptual",
            "chapters": [
                {
                    "title": "Hybrid Recommenders",
                    "source_chunk_ids": ["chunk-hybrid"],
                    "lessons": [{"title": "Hybrid", "source_chunk_ids": ["chunk-hybrid"]}],
                },
                {
                    "title": "Collaborative Filtering",
                    "source_chunk_ids": ["chunk-cf"],
                    "lessons": [{"title": "CF", "source_chunk_ids": ["chunk-cf"]}],
                },
            ],
        }
    )
    graph = {
        "nodes": [
            {"id": "hybrid", "node_type": "concept", "source_chunk_ids": ["chunk-hybrid"]},
            {"id": "cf", "node_type": "concept", "source_chunk_ids": ["chunk-cf"]},
        ],
        "edges": [
            {
                "source_node_id": "hybrid",
                "target_node_id": "cf",
                "relation_type": "requires",
                "confidence": 0.95,
            }
        ],
    }

    validated = coursebuilder._validate_plan_with_graph(outline, chunks, graph)

    assert [chapter.title for chapter in validated.chapters] == ["Collaborative Filtering", "Hybrid Recommenders"]
    assert "Collaborative Filtering" in validated.chapters[1].prerequisite_chapter_titles


def test_course_plan_persists_source_queries_from_headings_concepts_and_graph() -> None:
    import local_api.services.coursebuilder as coursebuilder

    chunks = [
        {
            "id": "chunk-svd",
            "source_file_id": "file-course",
            "source_filename": "recommenders.md",
            "chunk_index": 0,
            "text": "Singular value decomposition factorizes the user-item interaction matrix.",
            "metadata": {
                "heading_path": "Collaborative Filtering > Matrix Factorization",
                "heading_path_list": ["Collaborative Filtering", "Matrix Factorization"],
                "key_concepts": ["SVD", "latent factors"],
            },
        }
    ]
    outline = coursebuilder.CourseOutline.model_validate(
        {
            "title": "Recommendation Systems",
            "chapters": [
                {
                    "title": "Collaborative Filtering",
                    "source_chunk_ids": ["chunk-svd"],
                    "lessons": [
                        {
                            "title": "Matrix factorization",
                            "source_chunk_ids": ["chunk-svd"],
                            "source_queries": ["original source phrase"],
                        }
                    ],
                }
            ],
        }
    )
    graph = {
        "nodes": [
            {
                "id": "concept-svd",
                "node_type": "concept",
                "label": "Low-rank approximation",
                "source_chunk_ids": ["chunk-svd"],
            }
        ],
        "edges": [],
    }

    enriched = coursebuilder._ensure_outline_source_queries(outline, chunks, graph)
    queries = enriched.chapters[0].lessons[0].source_queries

    assert "original source phrase" in queries
    assert "matrix factorization" in {query.casefold() for query in queries}
    assert "SVD" in queries
    assert "Low-rank approximation" in queries


def test_coursebuilder_lesson_retrieval_stays_in_planned_scope(monkeypatch) -> None:
    import local_api.services.coursebuilder as coursebuilder

    detailed_svd = " ".join(
        ["SVD decomposes the interaction matrix into grounded latent user and item factors."] * 24
    )
    chunks = [
        {
            "id": "chunk-plan",
            "source_file_id": "file-course",
            "source_filename": "course.md",
            "chunk_index": 0,
            "text": "Source plan item: Matrix factorization",
            "metadata": {"heading_path_list": ["CF", "Outline"]},
        },
        {
            "id": "chunk-svd",
            "source_file_id": "file-course",
            "source_filename": "course.md",
            "chunk_index": 1,
            "text": detailed_svd,
            "metadata": {
                "heading_path_list": ["CF", "Matrix factorization"],
                "key_concepts": ["SVD"],
            },
        },
        {
            "id": "chunk-cbf",
            "source_file_id": "file-course",
            "source_filename": "course.md",
            "chunk_index": 2,
            "text": " ".join(["Content-based filtering uses item feature profiles."] * 24),
            "metadata": {"heading_path_list": ["CBF", "Feature profiles"]},
        },
    ]
    chapter = coursebuilder.OutlineChapter.model_validate(
        {
            "title": "Filtering Methods",
            "source_chunk_ids": [chunk["id"] for chunk in chunks],
            "source_queries": ["filtering methods"],
            "lessons": [
                {
                    "title": "Matrix factorization",
                    "source_chunk_ids": ["chunk-plan", "chunk-svd"],
                    "source_queries": ["SVD latent factors"],
                }
            ],
        }
    )

    class FakeRetrieval:
        async def retrieve_for(self, **_kwargs):
            return [
                types.SimpleNamespace(chunk_id="chunk-cbf"),
                types.SimpleNamespace(chunk_id="chunk-plan"),
                types.SimpleNamespace(chunk_id="chunk-svd"),
            ]

    monkeypatch.setattr(coursebuilder, "get_retrieval_service", lambda: FakeRetrieval())
    selected, count = asyncio.run(
        coursebuilder._retrieve_lesson_evidence(
            "conv-course",
            chapter,
            chapter.lessons[0],
            chunks,
        )
    )

    assert count == 1
    assert [chunk["id"] for chunk in selected] == ["chunk-svd"]


def test_coursebuilder_rejects_thin_rich_blocks_and_marks_unsupported_content() -> None:
    import local_api.services.coursebuilder as coursebuilder

    assert coursebuilder._is_thin_teaching_block("definition", "Matrix factorization.", "Matrix factorization")
    block = coursebuilder._block(
        "Thin lesson",
        0,
        "warning",
        "Insufficient source material",
        coursebuilder._insufficient_source_message(),
        [],
    )
    assert block["validation_status"] == "insufficient_source_material"
    assert block["citations"] == []
    assert coursebuilder._looks_chemical("2H2 + O2 -> 2H2O")
    assert not coursebuilder._looks_chemical("Frontend sends the user ID -> API returns JSON recommendations")


def test_course_plan_is_generated_from_parser_markdown_before_chunking(monkeypatch, tmp_path) -> None:
    client = _client(monkeypatch, tmp_path)
    conversation = client.post("/api/conversations", json={"title": "Two-stage planning"}).json()

    import local_api.services.coursebuilder as coursebuilder
    from local_api.db import get_store
    from local_api.services.vector_service import get_vector_service

    events: list[str] = []
    outline_calls = 0
    content_calls = 0

    async def planning_llm(_provider, messages, *, json_schema=None, temperature=0.2):
        nonlocal content_calls, outline_calls
        properties = (json_schema or {}).get("properties", {})
        if "chapters" not in properties:
            content_calls += 1
            raise RuntimeError("chapter and quiz synthesis disabled for lifecycle test")
        outline_calls += 1
        events.append("plan_llm")
        files = get_store().list_files(conversation["id"])
        assert files and all(file["status"] == "planning_course" for file in files)
        assert get_store().list_chunks(conversation["id"]) == []
        prompt = messages[-1].content
        assert "PARSER MARKDOWN SOURCES" in prompt
        assert "## Fundamentals" in prompt
        assert "## Hybrid Recommenders" in prompt
        lesson_titles = [
            "Fundamentals",
            "Collaborative Filtering",
            "Evaluation Metrics",
            "Hybrid Recommenders",
            "Deep Learning",
        ] if "## Hybrid Recommenders" in prompt else [
            "Fundamentals",
            "Collaborative Filtering",
            "Evaluation Metrics",
        ]
        return json.dumps(
            {
                "title": "Recommendation Systems",
                "description": "A cumulative plan prepared from parser Markdown before chunking.",
                "architecture_type": "conceptual",
                "architecture_rationale": "Foundations precede methods and evaluation.",
                "chapters": [
                    {
                        "title": "Foundations and Methods",
                        "pedagogical_role": "foundation",
                        "lessons": [
                            {
                                "title": title,
                                "pedagogical_role": "definition",
                            }
                            for title in lesson_titles
                        ],
                    }
                ],
            }
        )

    monkeypatch.setattr(coursebuilder, "complete_text", planning_llm)
    vector_service = get_vector_service()
    original_embed = vector_service.embed_chunks

    async def tracked_embed(chunks):
        events.append("embedding")
        plan_row = get_store().one(
            "SELECT status, payload_json FROM coursebuilder_plans WHERE conversation_id = ?",
            (conversation["id"],),
        )
        assert plan_row is not None
        assert plan_row["status"] == "draft"
        assert json.loads(plan_row["payload_json"])["outline"]["chapters"]
        return await original_embed(chunks)

    monkeypatch.setattr(vector_service, "embed_chunks", tracked_embed)
    upload = client.post(
        f"/api/conversations/{conversation['id']}/files/batch",
        files=[
            (
                "uploads",
                (
                "recommendation.md",
                b"""# Recommendation Systems

## Fundamentals
Users, items, ratings, and feedback define the recommendation problem.

## Collaborative Filtering
Collaborative filtering learns from user-item interactions.

## Evaluation Metrics
Precision, recall, RMSE, and MAE evaluate recommendation quality.
""",
                "text/markdown",
                ),
            ),
            (
                "uploads",
                (
                    "advanced.md",
                    b"""# Recommendation Systems

## Hybrid Recommenders
Hybrid recommenders combine collaborative and content-based components.

## Deep Learning
Neural recommenders learn advanced user and item representations.
""",
                    "text/markdown",
                ),
            ),
        ],
    )
    assert upload.status_code == 200
    assert len(upload.json()["files"]) == 2
    for uploaded_file in upload.json()["files"]:
        stored = client.get(f"/api/conversations/{conversation['id']}/files/{uploaded_file['id']}").json()
        assert stored["status"] == "ready"
    assert events == ["plan_llm", "embedding", "embedding"]
    assert outline_calls == 1
    assert content_calls > 0

    plan = client.get(f"/api/conversations/{conversation['id']}/coursebuilder/plan").json()
    assert plan["status"] == "validated"
    assert plan["metadata"]["stage"] == "validated_with_knowledge_graph"
    assert plan["metadata"]["chunk_coverage_ratio"] == 1.0
    assert plan["metadata"]["planning_basis"] == "parser_markdown"
    assert plan["metadata"]["structure_mode"] == "markdown_llm"
    assert plan["metadata"]["evidence_bound_after_chunking"] is True
    assert [item["title"] for item in plan["chapters"][0]["subchapters"]] == [
        "Fundamentals",
        "Collaborative Filtering",
        "Evaluation Metrics",
        "Hybrid Recommenders",
        "Deep Learning",
    ]
    assert all(item["lesson_stage"] == "content" for item in plan["chapters"][0]["subchapters"])
    assert plan["chapters"][0]["source_queries"]
    assert all(item["source_queries"] for item in plan["chapters"][0]["subchapters"])
    assert "blocks" not in json.dumps(plan["outline"])
    assert "final_quiz" not in json.dumps(plan["outline"])

    course = client.get(f"/api/conversations/{conversation['id']}/coursebuilder").json()
    assert course["status"] == "ready"
    assert course["metadata"]["build_profile"] == "standard"
    assert course["metadata"]["quality_pipeline_version"] == coursebuilder.QUALITY_PIPELINE_VERSION
    assert course["metadata"]["plan_id"] == plan["plan_id"]
    assert course["chapters"][0]["title"] == plan["chapters"][0]["title"]
    assert [lesson["title"] for lesson in course["chapters"][0]["lessons"]] == [
        item["title"] for item in plan["chapters"][0]["subchapters"]
    ]

    deleted = client.delete(
        f"/api/conversations/{conversation['id']}/files/{upload.json()['files'][1]['id']}"
    )
    assert deleted.status_code == 204
    replanned = client.get(f"/api/conversations/{conversation['id']}/coursebuilder/plan").json()
    assert replanned["status"] == "validated"
    assert replanned["plan_id"] != plan["plan_id"]
    assert replanned["metadata"]["chunk_coverage_ratio"] == 1.0


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
