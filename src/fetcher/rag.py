"""Answer a question from retrieved chunks with an LLM, then check its shape and sources."""

import os
from collections.abc import Callable
from typing import Any, cast

from pydantic import BaseModel

from fetcher.search import Hit

Messages = list[dict[str, str]]
LLM = Callable[[Messages], str]  # chat messages in, the model's reply text out

SYSTEM_PROMPT = """You answer questions using only the context passages you are given.
Each passage starts with its id in square brackets.
Reply with JSON only, in exactly this shape:
{"answer": "...", "sources": ["<id of every passage you used>"], "confident": true}
If the passages do not contain the answer, say so in "answer", leave "sources" empty
and set "confident" to false. Never use outside knowledge."""


class Answer(BaseModel):
    """The structured output we expect back. Pydantic rejects anything else."""

    answer: str
    sources: list[str]  # passage ids, so every claim can be traced to a fetched page
    confident: bool  # false when the context did not contain the answer


def build_messages(question: str, hits: list[Hit]) -> Messages:
    passages = "\n\n".join(f"[{hit.chunk.id}] {hit.chunk.text}" for hit in hits)
    user = f"Context passages:\n\n{passages}\n\nQuestion: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def parse_answer(raw: str, hits: list[Hit]) -> Answer:
    """Validate the JSON, then make sure it only cites passages it was actually given."""
    answer = Answer.model_validate_json(raw)  # raises ValidationError (a ValueError) if malformed
    given = {hit.chunk.id for hit in hits}
    made_up = [source for source in answer.sources if source not in given]
    if made_up:
        raise ValueError(f"cited passages that were not in the context: {made_up}")
    return answer


def ask(question: str, hits: list[Hit], llm: LLM, max_attempts: int = 2) -> Answer:
    """Ask the LLM. If the reply is invalid, tell it what was wrong and let it try again."""
    messages = build_messages(question, hits)
    error: ValueError | None = None
    for _ in range(max_attempts):
        raw = llm(messages)
        try:
            return parse_answer(raw, hits)
        except ValueError as e:
            error = e
            messages = [
                *messages,
                {"role": "assistant", "content": raw},
                {"role": "user", "content": f"Invalid reply ({e}). Reply again, JSON only."},
            ]
    raise RuntimeError(f"no valid answer after {max_attempts} attempts: {error}")


def openai_compatible_llm() -> LLM:
    """Any OpenAI-compatible chat API: OpenAI, Groq, Gemini, a local Ollama, and others.

    Set LLM_API_KEY, LLM_MODEL and (for anything but OpenAI) LLM_BASE_URL.
    """
    from openai import OpenAI  # imported here so the fetcher works without an API key

    model = os.environ.get("LLM_MODEL")
    if not model:
        raise RuntimeError("set LLM_MODEL (and LLM_API_KEY, LLM_BASE_URL) first, see README")
    client = OpenAI(api_key=os.environ.get("LLM_API_KEY"), base_url=os.environ.get("LLM_BASE_URL"))

    def call(messages: Messages) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=cast(Any, messages),
            response_format={"type": "json_object"},  # ask the API itself for valid JSON
            temperature=0,
        )
        return response.choices[0].message.content or ""

    return call
