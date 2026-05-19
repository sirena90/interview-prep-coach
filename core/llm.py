"""Thin wrapper around the Anthropic Messages API.

The whole module exists so that every agent in the system can call the LLM
with one line:

    result = call_llm(system=SYSTEM_PROMPT, user=user_msg, schema=ScoreReport)

It handles:
  - sending the prompt to Anthropic's Messages API (sync, Streamlit-friendly)
  - stripping ```json ... ``` fences if the model adds them
  - parsing JSON
  - validating the result against the requested Pydantic schema
  - one repair retry if the first response is malformed
"""
from __future__ import annotations

import json
import os
from typing import Callable, Type, TypeVar

from anthropic import Anthropic
from pydantic import BaseModel, ValidationError


T = TypeVar("T", bound=BaseModel)


# ---- Configuration ---------------------------------------------------------

DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.0  # deterministic by default; agents can override

# Single shared client (created on first use). Streamlit reruns the script on
# every interaction, but module state survives.
_client: Anthropic | None = None


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Add it to your .env file "
                "(see .env.example for the template)."
            )
        _client = Anthropic(api_key=api_key)
    return _client


# ---- Helpers ---------------------------------------------------------------

def _strip_code_fences(text: str) -> str:
    """Strip ```json ... ``` wrappers if the LLM added them."""
    t = text.strip()
    if t.startswith("```"):
        # Drop opening fence (with optional language label like ```json)
        t = t.split("\n", 1)[1] if "\n" in t else t
        # Drop closing fence
        if t.endswith("```"):
            t = t[:-3]
    return t.strip()


def _call_anthropic(
    system: str,
    user: str,
    model: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """One synchronous call to the Messages API. Returns raw text."""
    client = _get_client()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    # Anthropic returns a list of content blocks; we only consume text blocks.
    parts = [b.text for b in response.content if getattr(b, "type", None) == "text"]
    return "\n".join(parts).strip()


# ---- Public API ------------------------------------------------------------

def call_llm(
    *,
    system: str,
    user: str,
    schema: Type[T],
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> T:
    """Send a prompt and get back a validated Pydantic object.

    Args:
        system:       System prompt (instructions for the model).
        user:         User message (the task / data the model works on).
        schema:       Pydantic model class describing the expected response.
        model:        Anthropic model name (default: latest Sonnet).
        max_tokens:   Output token cap (default 1024).
        temperature:  Sampling temperature (default 0 for determinism).

    Returns:
        An instance of `schema` populated from the model's JSON response.

    Raises:
        RuntimeError on missing API key.
        ValidationError / json.JSONDecodeError if repair retry also fails.
    """
    raw = _call_anthropic(system, user, model, max_tokens, temperature)

    # Happy path: parse and validate
    try:
        data = json.loads(_strip_code_fences(raw))
        return schema.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as first_error:
        pass  # fall through to repair attempt

    # Repair attempt: tell the model exactly what was wrong and re-ask.
    repair_user = (
        f"{user}\n\n"
        "Your previous response was not valid JSON for the required schema.\n"
        f"Error: {first_error}\n\n"
        "Schema you must match (JSON Schema):\n"
        f"{json.dumps(schema.model_json_schema(), indent=2)}\n\n"
        "Your previous response was:\n"
        f"{raw}\n\n"
        "Reply with ONLY valid JSON matching the schema. "
        "No prose before or after, no code fences."
    )
    raw2 = _call_anthropic(system, repair_user, model, max_tokens, temperature)
    data = json.loads(_strip_code_fences(raw2))  # if this fails, bubble up
    return schema.model_validate(data)

# ---- Tool use API --------------------------------------------------------

def call_llm_with_tools(
    *,
    system: str,
    user: str,
    schema: Type[T],
    tools: list[dict],
    tool_executor: Callable[[str, dict], str],
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    max_iterations: int = 8,
) -> T:
    """Run a tool-use loop with the LLM and return a validated Pydantic object.

    The model is given a set of tools it can call. When the model emits a
    tool_use block, we execute the tool via `tool_executor(name, input_dict)`
    and feed the result back. The loop ends when the model emits a final
    text answer (no more tool_use) or when max_iterations is hit.

    Args:
        system, user:      prompts as in call_llm
        schema:            Pydantic model class for the final answer
        tools:             Anthropic tool specifications (list of dicts)
        tool_executor:     function(tool_name, tool_input_dict) -> result string
        max_iterations:    safety cap on tool-call rounds

    Returns:
        An instance of `schema` populated from the model's final JSON output.
    """
    client = _get_client()
    messages: list[dict] = [{"role": "user", "content": user}]

    for _ in range(max_iterations):
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            tools=tools,
            messages=messages,
        )

        if response.stop_reason == "tool_use":
            # Execute every tool_use block in the response, collect results
            tool_results = []
            for block in response.content:
                if getattr(block, "type", None) == "tool_use":
                    try:
                        result = tool_executor(block.name, dict(block.input))
                    except Exception as e:
                        result = f"Tool execution error: {e}"
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            # Append assistant's tool_use message and our tool_result reply
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
            continue

        # Final answer — extract text, parse JSON, validate against schema
        text_parts = [
            b.text for b in response.content
            if getattr(b, "type", None) == "text"
        ]
        raw = "\n".join(text_parts).strip()
        cleaned = _strip_code_fences(raw)
        data = json.loads(cleaned)
        return schema.model_validate(data)

    raise RuntimeError(
        f"Tool-use loop exceeded {max_iterations} iterations without producing a final answer."
    )

