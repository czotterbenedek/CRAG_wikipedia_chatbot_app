from typing import Any

import ollama

from src.conversation_memory import build_relevant_history_block
from src.models import ProjectSettings, SupervisorDecision


def _chat(model: str, system_prompt: str, user_prompt: str, temperature: float) -> str:
    response = ollama.chat(
        model=model,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        format=SupervisorDecision.model_json_schema(),
        options={"temperature": temperature},
    )
    return response["message"]["content"].strip()


def decide_route(question: str, chat_history: list[dict[str, str]], settings: ProjectSettings) -> SupervisorDecision:
    history_block = build_relevant_history_block(
        chat_history,
        question,
        settings.embedding_model,
        max_turns=settings.history_max_turns,
        similarity_threshold=settings.history_similarity_threshold,
    )
    user_prompt = f"""
<conversation_history>
{history_block if history_block else '<turn role="system">No prior history.</turn>'}
</conversation_history>

<user_question>
{question}
</user_question>

Return JSON only with keys action and message.
""".strip()
    raw_response = _chat(settings.supervisor.model, settings.supervisor.system_prompt, user_prompt, settings.supervisor.temperature)
    try:
        decision = SupervisorDecision.model_validate_json(raw_response)
    except Exception:
        decision = SupervisorDecision(action="retrieve", message="")
    if decision.action == "clarify" and not decision.message:
        return SupervisorDecision(action="clarify", message="Please clarify your Wikipedia-related question so I can answer it accurately.")
    if decision.action == "reject" and not decision.message:
        return SupervisorDecision(action="reject", message="I can only answer basic Wikipedia-related questions grounded in the corpus.")
    return decision


def supervisor_node(state: dict[str, Any], settings: ProjectSettings) -> dict[str, Any]:
    decision = decide_route(state.get("question", ""), state.get("chat_history", []), settings)
    return {**state, "supervisor_decision": decision.action, "supervisor_message": decision.message}
