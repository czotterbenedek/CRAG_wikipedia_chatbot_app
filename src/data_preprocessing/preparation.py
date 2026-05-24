from pathlib import Path

import numpy as np
import ollama
from tqdm import tqdm

from src.data_preprocessing.loader import (
    build_chunk_cache_path,
    build_embeddings_cache_path,
    load_raw_chunk_bundle,
    resolve_raw_data_path,
    save_cached_embeddings,
    save_chunk_bundle,
)
from src.models import ProjectSettings


def _embed_texts(texts: list[str], model_name: str) -> np.ndarray:
    embeddings = [
        ollama.embeddings(model=model_name, prompt=text)["embedding"]
        for text in tqdm(texts, desc="Embedding texts", unit="text")
    ]
    return np.asarray(embeddings, dtype=np.float32)


def _is_cache_stale(source_path: Path, cache_path: Path) -> bool:
    if not cache_path.exists():
        return True
    return source_path.stat().st_mtime > cache_path.stat().st_mtime


def prepare_data_assets(settings: ProjectSettings) -> Path:
    bundle = load_raw_chunk_bundle(
        settings.project_root,
        settings.global_settings.raw_corpus_file,
        settings.global_settings.chunk_size,
        settings.global_settings.chunk_overlap,
    )
    chunk_cache_path = build_chunk_cache_path(
        settings.chunk_cache_dir_path,
        bundle.source_path,
        settings.global_settings.chunk_size,
        settings.global_settings.chunk_overlap,
    )
    embeddings_cache_path = build_embeddings_cache_path(
        settings.embeddings_cache_dir_path,
        bundle.source_path,
        settings.global_settings.chunk_size,
        settings.global_settings.chunk_overlap,
    )

    save_chunk_bundle(chunk_cache_path, bundle)
    embeddings = _embed_texts([chunk.text for chunk in bundle.chunks], settings.embedding_model)
    save_cached_embeddings(embeddings_cache_path, embeddings, str(bundle.source_path), len(bundle.chunks))
    return bundle.source_path


def ensure_data_assets(settings: ProjectSettings) -> Path:
    source_path = resolve_raw_data_path(settings.project_root, settings.global_settings.raw_corpus_file)
    chunk_cache_path = build_chunk_cache_path(
        settings.chunk_cache_dir_path,
        source_path,
        settings.global_settings.chunk_size,
        settings.global_settings.chunk_overlap,
    )
    embeddings_cache_path = build_embeddings_cache_path(
        settings.embeddings_cache_dir_path,
        source_path,
        settings.global_settings.chunk_size,
        settings.global_settings.chunk_overlap,
    )
    if _is_cache_stale(source_path, chunk_cache_path) or _is_cache_stale(source_path, embeddings_cache_path):
        return prepare_data_assets(settings)
    return source_path