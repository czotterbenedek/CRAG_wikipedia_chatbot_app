from typing import Any

import numpy as np
import ollama


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def _embed_text(text: str, model_name: str) -> np.ndarray:
    embedding = ollama.embeddings(model=model_name, prompt=text)["embedding"]
    return _normalize(np.asarray(embedding, dtype=np.float32))


def _turn_text(turn: dict[str, str]) -> str:
    role = turn.get("role", "user")
    content = turn.get("content", "").strip()
    return f"{role}: {content}".strip()


def select_relevant_history(
    chat_history: list[dict[str, str]],
    question: str,
    model_name: str,
    *,
    max_turns: int = 8,
    similarity_threshold: float = 0.6,
) -> list[dict[str, str]]:
    if not chat_history:
        return []

    cleaned_question = question.strip()
    if not cleaned_question:
        return []

    candidates = chat_history[-max_turns:]
    question_embedding = _embed_text(cleaned_question, model_name)
    selected_indices: set[int] = set()

    for index, turn in enumerate(candidates):
        role = turn.get("role", "")
        content = turn.get("content", "").strip()
        if role not in {"user", "assistant"} or not content:
            continue

        if index == len(candidates) - 1 and role == "user" and content == cleaned_question:
            continue

        similarity = float(np.dot(question_embedding, _embed_text(_turn_text(turn), model_name)))
        if similarity >= similarity_threshold:
            selected_indices.add(index)

    if not selected_indices:
        return []

    # Keep the user turn that led into any selected assistant response.
    for index in list(selected_indices):
        if candidates[index].get("role") == "assistant" and index > 0 and candidates[index - 1].get("role") == "user":
            selected_indices.add(index - 1)

    return [turn for index, turn in enumerate(candidates) if index in selected_indices]


def build_relevant_history_block(
    chat_history: list[dict[str, str]],
    question: str,
    model_name: str,
    *,
    max_turns: int = 8,
    similarity_threshold: float = 0.6,
) -> str:
    relevant_history = select_relevant_history(
        chat_history,
        question,
        model_name,
        max_turns=max_turns,
        similarity_threshold=similarity_threshold,
    )
    if not relevant_history:
        return '<turn role="system">No prior history.</turn>'
    return "\n".join(f"<turn role=\"{turn['role']}\">{turn['content']}</turn>" for turn in relevant_history)