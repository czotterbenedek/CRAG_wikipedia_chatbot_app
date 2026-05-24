import json
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from tqdm import tqdm


class ChunkRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    chunk_id: int = Field(ge=0)
    source: str
    text: str
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)


class ChunkBundle(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    source_path: Path
    chunks: list[ChunkRecord]


def resolve_raw_data_path(project_root: Path, raw_corpus_file: str) -> Path:
    configured_path = project_root / raw_corpus_file
    if configured_path.exists():
        return configured_path
    fallback_path = project_root / "data" / "raw" / "AllCombined.txt"
    if fallback_path.exists():
        return fallback_path
    raise FileNotFoundError("Could not find the configured raw corpus or data/raw/AllCombined.txt.")


def build_embeddings_cache_path(cache_dir: Path, source_path: Path, chunk_size: int, chunk_overlap: int) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{source_path.stem}_chunk{chunk_size}_overlap{chunk_overlap}.npz"


def build_chunk_cache_path(cache_dir: Path, source_path: Path, chunk_size: int, chunk_overlap: int) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{source_path.stem}_chunk{chunk_size}_overlap{chunk_overlap}.json"


def save_chunk_bundle(cache_path: Path, bundle: ChunkBundle) -> None:
    payload = {
        "source_path": str(bundle.source_path),
        "chunks": [chunk.model_dump() for chunk in bundle.chunks],
    }
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_chunk_bundle(cache_path: Path) -> ChunkBundle:
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    return ChunkBundle(
        source_path=Path(payload["source_path"]),
        chunks=[ChunkRecord.model_validate(chunk) for chunk in payload["chunks"]],
    )


def load_cached_embeddings(cache_path: Path) -> np.ndarray | None:
    try:
        cached = np.load(cache_path, allow_pickle=False)
        return cached["embeddings"].astype(np.float32)
    except FileNotFoundError:
        return None


def save_cached_embeddings(cache_path: Path, embeddings: np.ndarray, source_path: str, chunk_count: int) -> None:
    np.savez_compressed(
        cache_path,
        embeddings=embeddings.astype(np.float32),
        source_path=source_path,
        chunk_count=chunk_count,
    )


def _chunk_text(text: str, *, chunk_size: int, chunk_overlap: int, source_name: str) -> list[ChunkRecord]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return []

    chunks: list[ChunkRecord] = []
    start = 0
    chunk_id = 0
    step = max(1, chunk_size - chunk_overlap)
    estimated_total = max(1, (len(cleaned) + step - 1) // step)

    with tqdm(total=estimated_total, desc="Chunking corpus", unit="chunk") as progress:
        while start < len(cleaned):
            end = min(len(cleaned), start + chunk_size)
            chunk = cleaned[start:end].strip()
            if chunk:
                chunks.append(
                    ChunkRecord(
                        chunk_id=chunk_id,
                        source=source_name,
                        text=chunk,
                        start_char=start,
                        end_char=end,
                    )
                )
                chunk_id += 1
                progress.update(1)
            if end >= len(cleaned):
                break
            start = end - chunk_overlap if chunk_overlap > 0 else end

    return chunks


def load_raw_chunk_bundle(
    project_root: Path,
    raw_corpus_file: str,
    chunk_size: int,
    chunk_overlap: int,
) -> ChunkBundle:
    source_path = resolve_raw_data_path(project_root, raw_corpus_file)
    raw_text = source_path.read_text(encoding="utf-8", errors="ignore")
    chunks = _chunk_text(raw_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap, source_name=source_path.name)
    if not chunks:
        raise ValueError("The raw corpus is empty after preprocessing.")
    return ChunkBundle(source_path=source_path, chunks=chunks)