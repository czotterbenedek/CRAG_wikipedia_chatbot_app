from typing import Any

import faiss
import numpy as np
from pydantic import BaseModel, ConfigDict

from src.data_preprocessing.loader import (
    ChunkBundle,
    build_chunk_cache_path,
    build_embeddings_cache_path,
    load_cached_embeddings,
    load_chunk_bundle,
    resolve_raw_data_path,
)
from src.models import ProjectSettings, RetrievedChunk


class VectorStore(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    bundle: ChunkBundle
    index: faiss.Index
    embeddings: np.ndarray
    embeddings_cache_path: str


class RetrieverOutput(BaseModel):
    retrieved_chunks: list[RetrievedChunk]
    retrieved_context: str


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def build_vector_store(settings: ProjectSettings) -> VectorStore:
    source_path = resolve_raw_data_path(settings.project_root, settings.global_settings.raw_corpus_file)
    bundle = load_chunk_bundle(
        build_chunk_cache_path(
            settings.chunk_cache_dir_path,
            source_path,
            settings.global_settings.chunk_size,
            settings.global_settings.chunk_overlap,
        )
    )
    cache_path = build_embeddings_cache_path(
        settings.embeddings_cache_dir_path,
        bundle.source_path,
        settings.global_settings.chunk_size,
        settings.global_settings.chunk_overlap,
    )
    cached_embeddings = load_cached_embeddings(cache_path)
    if cached_embeddings is None:
        raise FileNotFoundError(
            f"Missing embeddings cache at {cache_path}. Run prepare_data.py first or let the app refresh the caches."
        )

    embeddings = cached_embeddings
    normalized = _normalize_rows(embeddings)
    index = faiss.IndexFlatIP(normalized.shape[1])
    index.add(normalized)
    return VectorStore(bundle=bundle, index=index, embeddings=normalized, embeddings_cache_path=str(cache_path))


def retrieve_chunks(query: str, settings: ProjectSettings, store: VectorStore) -> RetrieverOutput:
    import ollama

    query_embedding = np.asarray([ollama.embeddings(model=settings.embedding_model, prompt=query)["embedding"]], dtype=np.float32)
    query_embedding = _normalize_rows(query_embedding)
    scores, indices = store.index.search(query_embedding, settings.top_k)
    retrieved_chunks: list[RetrievedChunk] = []

    for score, index in zip(scores[0], indices[0], strict=False):
        if index < 0:
            continue
        chunk = store.bundle.chunks[int(index)]
        retrieved_chunks.append(RetrievedChunk(chunk_id=chunk.chunk_id, source=chunk.source, score=float(score), text=chunk.text))

    retrieved_context = "\n\n".join(
        f"<chunk id=\"{chunk.chunk_id}\" source=\"{chunk.source}\" score=\"{chunk.score:.3f}\">{chunk.text}</chunk>"
        for chunk in retrieved_chunks
    )
    return RetrieverOutput(retrieved_chunks=retrieved_chunks, retrieved_context=retrieved_context)


def retriever_node(state: dict[str, Any], settings: ProjectSettings, store: VectorStore) -> dict[str, Any]:
    result = retrieve_chunks(state.get("question", ""), settings, store)
    return {**state, **result.model_dump()}
