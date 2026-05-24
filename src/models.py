from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class GlobalSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    embedding_model: str = "nomic-embed-text:latest"
    ollama_model: str = "llama3.2:3b"
    raw_corpus_file: str = "data/raw/AllCombined.txt"
    embeddings_cache_dir: str = "data/embeddings"
    chunk_size: int = Field(default=1500, ge=1)
    chunk_overlap: int = Field(default=220, ge=0)
    top_k: int = Field(default=4, ge=1)
    history_max_turns: int = Field(default=8, ge=1)
    history_similarity_threshold: float = Field(default=0.6, ge=0.0, le=1.0)


class SupervisorSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str = "llama3.2:3b"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    allowed_topics: str = "Wikipedia-style factual questions grounded in the chunked corpus."
    system_prompt: str


class SummarySettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str = "llama3.2:3b"
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    system_prompt: str


class CorrectiveSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str = "llama3.2:3b"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    system_prompt: str


class FallbackSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str = "llama3.2:3b"
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    system_prompt: str


class ProjectSettings(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    project_root: Path
    global_settings: GlobalSettings
    supervisor: SupervisorSettings
    summary: SummarySettings
    corrective: CorrectiveSettings
    fallback: FallbackSettings

    @property
    def raw_corpus_path(self) -> Path:
        return self.project_root / self.global_settings.raw_corpus_file

    @property
    def embeddings_cache_dir_path(self) -> Path:
        return self.project_root / self.global_settings.embeddings_cache_dir

    @property
    def chunk_cache_dir_path(self) -> Path:
        return self.project_root / "data" / "chunks"

    @property
    def embedding_model(self) -> str:
        return self.global_settings.embedding_model

    @property
    def ollama_model(self) -> str:
        return self.global_settings.ollama_model

    @property
    def top_k(self) -> int:
        return self.global_settings.top_k

    @property
    def chunk_size(self) -> int:
        return self.global_settings.chunk_size

    @property
    def chunk_overlap(self) -> int:
        return self.global_settings.chunk_overlap

    @property
    def history_max_turns(self) -> int:
        return self.global_settings.history_max_turns

    @property
    def history_similarity_threshold(self) -> float:
        return self.global_settings.history_similarity_threshold


class RetrievedChunk(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    chunk_id: int
    source: str
    score: float
    text: str


class SupervisorDecision(BaseModel):
    action: Literal["retrieve", "clarify", "reject"]
    message: str = ""


class RetrievalRelevanceDecision(BaseModel):
    is_relevant: bool
    message: str = ""
