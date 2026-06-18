from pathlib import Path
import asyncio
import sys
import types
import json
import re

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
    import local_api.services.ingestion as ingestion_module
    import local_api.services.knowledge_graph as graph_module
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

    course = client.get(f"/api/conversations/{conversation['id']}/coursebuilder").json()
    assert course["status"] == "ready"
    assert course["chapters"]
    assert course["chapters"][0]["lessons"]
    assert course["chapters"][0]["lessons"][0]["blocks"]
    assert course["chapters"][0]["lessons"][0]["citations"]

    quiz_done = _done_event(
        client.post(
            f"/api/conversations/{conversation['id']}/generate",
            json={"output_type": "quiz", "prompt": "Generate quiz", "source_file_ids": [], "options": {"question_count": 4}},
        ).text
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

    mindmap_done = _done_event(
        client.post(
            f"/api/conversations/{conversation['id']}/generate",
            json={"output_type": "mindmap", "prompt": "Generate mindmap", "source_file_ids": [], "options": {}},
        ).text
    )
    assert mindmap_done["output_type"] == "mindmap"
    assert mindmap_done["metadata"]["node_count"] >= 3
    assert mindmap_done["metadata"]["central_topic"]
    assert len(mindmap_done["artifacts"]) >= 2

    podcast_done = _done_event(
        client.post(
            f"/api/conversations/{conversation['id']}/generate",
            json={"output_type": "podcast", "prompt": "Generate podcast", "source_file_ids": [], "options": {}},
        ).text
    )
    assert podcast_done["output_type"] == "podcast"
    assert podcast_done["metadata"]["narrative_arc"]["key_points"]
    assert podcast_done["metadata"]["podcast"]["transcript"]
    assert podcast_done["metadata"]["podcast"]["tts_skipped"] is True


def _done_event(raw_sse: str) -> dict:
    for block in raw_sse.split("\n\n"):
        lines = block.splitlines()
        if "event: done" not in lines:
            continue
        data_line = next(line for line in lines if line.startswith("data: "))
        return json.loads(data_line.removeprefix("data: "))
    raise AssertionError(f"no done event in SSE:\n{raw_sse}")
