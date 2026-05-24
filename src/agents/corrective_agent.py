from typing import Any

import ollama
from pydantic import BaseModel, ConfigDict

from src.models import ProjectSettings, RetrievalRelevanceDecision


class _ChunkPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    chunk_id: int
    source: str
    score: float
    text: str


class _FallbackAnswer(BaseModel):
    model_config = ConfigDict(extra="ignore")

    answer: str


def _chat(model: str, system_prompt: str, user_prompt: str, temperature: float, *, schema: dict[str, Any]) -> str:
    response = ollama.chat(
        model=model,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        format=schema,
        options={"temperature": temperature},
    )
    return response["message"]["content"].strip()


def grade_retrieved_chunks(state: dict[str, Any], settings: ProjectSettings) -> dict[str, Any]:
    question = state.get("question", "").strip()
    retrieved_chunks = state.get("retrieved_chunks", [])

    if not question or not retrieved_chunks:
        decision = RetrievalRelevanceDecision(is_relevant=False, message="No retrieved chunks are available for grading.")
        return {**state, "retrieved_chunks_relevant": decision.is_relevant, "retrieval_grade_message": decision.message}

    chunks = [_ChunkPayload.model_validate(chunk) for chunk in retrieved_chunks]
    chunk_block = "\n\n".join(
        f"<chunk id=\"{chunk.chunk_id}\" source=\"{chunk.source}\" score=\"{chunk.score:.3f}\">{chunk.text}</chunk>"
        for chunk in chunks
    )
    user_prompt = f"""
<user_question>
{question}
</user_question>

<retrieved_chunks>
{chunk_block}
</retrieved_chunks>

Decide whether the retrieved chunks are relevant enough to answer the question.
Return JSON only with keys is_relevant and message.
""".strip()
    raw_response = _chat(
        settings.corrective.model,
        settings.corrective.system_prompt,
        user_prompt,
        settings.corrective.temperature,
        schema=RetrievalRelevanceDecision.model_json_schema(),
    )
    try:
        decision = RetrievalRelevanceDecision.model_validate_json(raw_response)
    except Exception:
        decision = RetrievalRelevanceDecision(is_relevant=False, message="Could not validate chunk relevance.")

    return {**state, "retrieved_chunks_relevant": decision.is_relevant, "retrieval_grade_message": decision.message}


def fallback_answer_node(state: dict[str, Any], settings: ProjectSettings) -> dict[str, Any]:
    question = state.get("question", "")
    user_prompt = f"""
<user_question>
{question}
</user_question>

The retrieved chunks were not relevant enough to answer this question.
Respond with one concise paragraph that begins with:
"The query cannot be answered based on the chunks. Based on my knowledge, the answer is that ..."
Then provide the best answer you can from general knowledge.
""".strip()
    raw_response = _chat(
        settings.fallback.model,
        settings.fallback.system_prompt,
        user_prompt,
        settings.fallback.temperature,
        schema=_FallbackAnswer.model_json_schema(),
    )
    try:
        fallback_text = _FallbackAnswer.model_validate_json(raw_response).answer
    except Exception:
        fallback_text = raw_response
    return {
        **state,
        "answer": fallback_text,
        "retrieved_chunks": [],
        "retrieved_context": "",
        "retrieved_chunks_relevant": False,
    }