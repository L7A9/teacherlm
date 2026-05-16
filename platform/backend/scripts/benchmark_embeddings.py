from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any


BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parents[1]
for path in (BACKEND_DIR, REPO_ROOT / "packages" / "teacherlm_core"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


MODEL_DIMS = {
    "BAAI/bge-m3": 1024,
    "intfloat/multilingual-e5-large": 1024,
    "BAAI/bge-small-en-v1.5": 384,
}


async def _benchmark_model(model_name: str, texts: list[str]) -> dict[str, Any]:
    from fastembed import TextEmbedding

    started = time.perf_counter()
    embedder = await asyncio.to_thread(TextEmbedding, model_name)
    load_ms = (time.perf_counter() - started) * 1000.0

    started = time.perf_counter()

    def _embed() -> list[list[float]]:
        embed = getattr(embedder, "passage_embed", embedder.embed)
        return [list(vec) for vec in embed(texts)]

    vectors = await asyncio.to_thread(_embed)
    embed_ms = (time.perf_counter() - started) * 1000.0
    dim = len(vectors[0]) if vectors else MODEL_DIMS.get(model_name)
    return {
        "model_name": model_name,
        "dimension": dim,
        "text_count": len(texts),
        "load_ms": round(load_ms, 2),
        "embed_ms": round(embed_ms, 2),
        "texts_per_second": round(len(texts) / max(embed_ms / 1000.0, 0.001), 2),
    }


def _load_texts(path: Path, limit: int) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        rows = data.get("texts") or data.get("chunks") or []
    else:
        rows = data
    texts = []
    for item in rows:
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, dict):
            texts.append(str(item.get("text") or item.get("preview") or ""))
    return [text for text in texts if text.strip()][:limit]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark fastembed-compatible embedding models.")
    parser.add_argument("texts_json", type=Path, help="JSON with texts or chunks to embed")
    parser.add_argument("--models", default="BAAI/bge-m3,intfloat/multilingual-e5-large,BAAI/bge-small-en-v1.5")
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--out", type=Path)
    return parser


async def _amain() -> None:
    args = _parser().parse_args()
    texts = _load_texts(args.texts_json, args.limit)
    models = [model.strip() for model in args.models.split(",") if model.strip()]
    report = {"text_count": len(texts), "models": []}
    for model in models:
        report["models"].append(await _benchmark_model(model, texts))
    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    asyncio.run(_amain())
