"""Provider-agnostic wrapper around chat LLMs (Anthropic / OpenAI / Mistral).

Every agent calls the LLM with one line:

    result = call_llm(system=SYSTEM_PROMPT, user=user_msg, schema=ScoreReport)

The provider is auto-detected from whichever API key is in .env (see
core.config.settings.active_provider); model and defaults also come from
settings. Switching LLMs is just a matter of which key .env contains.

It handles:
  - routing the call to the configured provider
  - stripping ```json ... ``` fences if the model adds them
  - parsing JSON and validating against the requested Pydantic schema
  - one repair retry if the first response is malformed

Observability:
  - every call is traced to LangSmith when LANGSMITH_API_KEY is configured
    (each provider client is wrapped, and call_llm is @traceable)
  - every call also appends one line to logs/llm_calls.jsonl as an offline
    fallback — provider, model, latency, token usage, status
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Type, TypeVar

from anthropic import Anthropic, RateLimitError as AnthropicRateLimitError
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError as OpenAIRateLimitError
from pydantic import BaseModel, ValidationError

from core.config import settings

T = TypeVar("T", bound=BaseModel)

# Load .env, then disable LangSmith tracing if it is switched on without a key.
# Otherwise langsmith floods the console with 401 errors on every LLM call.
load_dotenv()
if (os.getenv("LANGSMITH_TRACING", "").strip().lower() in ("true", "1", "yes")
        and not os.getenv("LANGSMITH_API_KEY", "").strip()):
    os.environ["LANGSMITH_TRACING"] = "false"


# ---- Optional LangSmith tracing -------------------------------------------
# LangSmith degrades to a no-op on its own when no LANGSMITH_API_KEY is set,
# so there is nothing to configure here. The try/except only guards the case
# where the package isn't installed at all.
try:
    from langsmith import traceable
    from langsmith.wrappers import wrap_anthropic, wrap_openai
except ImportError:  # pragma: no cover - langsmith is a declared dependency
    def traceable(*d_args, **d_kwargs):
        """No-op stand-in: supports both @traceable and @traceable(...)."""
        if len(d_args) == 1 and callable(d_args[0]) and not d_kwargs:
            return d_args[0]
        return lambda fn: fn

    def wrap_anthropic(client):
        return client

    def wrap_openai(client):
        return client

    # Don't fail silently: if tracing was asked for but the package is absent,
    # say so on stderr so it isn't a mystery why no traces appear.
    if os.getenv("LANGSMITH_TRACING", "").strip().lower() in ("true", "1", "yes"):
        import sys
        print(
            "WARNING: LANGSMITH_TRACING is enabled but the 'langsmith' package "
            "is not installed — tracing is OFF. Run: pip install langsmith",
            file=sys.stderr,
        )


# ---- Configuration ---------------------------------------------------------

# Offline trace log — one JSON line per call. Survives without LangSmith.
_LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "llm_calls.jsonl"

# Lazily-created provider clients (Streamlit reruns the script; module state
# survives). Mistral talks the OpenAI-compatible API, so it reuses OpenAI().
_anthropic_client: Anthropic | None = None
_openai_client: OpenAI | None = None
_mistral_client: OpenAI | None = None


# ---- Provider clients ------------------------------------------------------
# Each client is wrapped for LangSmith tracing; the wrappers are no-ops when
# LangSmith is not configured.

def _get_anthropic() -> Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        if not settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set in your .env file.")
        _anthropic_client = wrap_anthropic(Anthropic(api_key=settings.anthropic_api_key))
    return _anthropic_client


def _get_openai() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set in your .env file.")
        _openai_client = wrap_openai(OpenAI(api_key=settings.openai_api_key))
    return _openai_client


def _get_mistral() -> OpenAI:
    global _mistral_client
    if _mistral_client is None:
        if not settings.mistral_api_key:
            raise RuntimeError("MISTRAL_API_KEY is not set in your .env file.")
        _mistral_client = wrap_openai(OpenAI(
            api_key=settings.mistral_api_key,
            base_url="https://api.mistral.ai/v1",
        ))
    return _mistral_client


# ---- Raw provider calls — each returns (text, usage) -----------------------

def _call_anthropic(system, user, model, max_tokens, temperature):
    response = _get_anthropic().messages.create(
        model=model, max_tokens=max_tokens, temperature=temperature,
        system=system, messages=[{"role": "user", "content": user}],
    )
    text = "\n".join(
        b.text for b in response.content if getattr(b, "type", None) == "text"
    ).strip()
    return text, getattr(response, "usage", None)


def _call_openai_compatible(client, system, user, model, max_tokens, temperature):
    response = client.chat.completions.create(
        model=model, max_tokens=max_tokens, temperature=temperature,
        # JSON mode: forces the response to be a parseable JSON object.
        # (Requires the word "json" in the prompt — the system prompts have it.)
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    text = (response.choices[0].message.content or "").strip()
    return text, getattr(response, "usage", None)


# Provider -> callable(system, user, model, max_tokens, temperature) -> (text, usage)
_DISPATCH: dict[str, Callable] = {
    "anthropic": lambda s, u, m, mt, t: _call_anthropic(s, u, m, mt, t),
    "openai":    lambda s, u, m, mt, t: _call_openai_compatible(_get_openai(), s, u, m, mt, t),
    "mistral":   lambda s, u, m, mt, t: _call_openai_compatible(_get_mistral(), s, u, m, mt, t),
}


# ---- Rate-limit handling ---------------------------------------------------
# A rate-limited provider (common on free tiers like Mistral) raises a
# RateLimitError. We retry with exponential backoff and let the UI show a
# "waiting / retrying" message through an optional notifier callback.

_RATE_LIMIT_ERRORS = (AnthropicRateLimitError, OpenAIRateLimitError)
_MAX_RETRIES = 4
_BASE_DELAY_SECONDS = 2.0   # backoff grows 2 -> 4 -> 8 -> 16
_MAX_DELAY_SECONDS = 20.0

# Optional UI hook: app.py registers a callback (e.g. st.toast) so the user
# sees each retry. Stays None for tests and scripts — then retries are silent.
_retry_notifier: "Callable[[str], None] | None" = None


def set_retry_notifier(notifier: "Callable[[str], None] | None") -> None:
    """Register a callback invoked with a human-readable message before each
    rate-limit retry. Pass None to clear it."""
    global _retry_notifier
    _retry_notifier = notifier


def _notify_retry(message: str) -> None:
    if _retry_notifier is not None:
        try:
            _retry_notifier(message)
        except Exception:
            pass  # a broken notifier must never break the LLM call


def _retry_delay(exc: Exception, attempt: int) -> float:
    """Seconds to wait before the next attempt: honour the provider's
    Retry-After header when present, else exponential backoff."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        raw = headers.get("retry-after")
        if raw:
            try:
                return min(float(raw), _MAX_DELAY_SECONDS)
            except (ValueError, TypeError):
                pass
    return min(_BASE_DELAY_SECONDS * (2 ** attempt), _MAX_DELAY_SECONDS)


def _retry_on_rate_limit(make_request):
    """Run make_request() — a zero-arg provider call — retrying rate-limit
    errors with exponential backoff. Used by every provider call: plain
    completions and tool-use loops alike.

    Raises RuntimeError with a clear message if the provider keeps rate-
    limiting after _MAX_RETRIES attempts.
    """
    for attempt in range(_MAX_RETRIES + 1):
        try:
            return make_request()
        except _RATE_LIMIT_ERRORS as exc:
            if attempt == _MAX_RETRIES:
                raise RuntimeError(
                    "The LLM provider is rate-limiting requests and did not "
                    "recover after several retries. Wait a minute and try "
                    "again, or switch provider in .env."
                ) from exc
            delay = _retry_delay(exc, attempt)
            _notify_retry(
                f"Rate limited by the LLM provider — waiting {delay:.0f}s, "
                f"then retrying (attempt {attempt + 1} of {_MAX_RETRIES})..."
            )
            time.sleep(delay)


def _call_with_retry(fn, system, user, model, max_tokens, temperature):
    """Rate-limit-retried wrapper for a plain provider call."""
    return _retry_on_rate_limit(
        lambda: fn(system, user, model, max_tokens, temperature)
    )


# ---- Helpers ---------------------------------------------------------------

def _strip_fences(text: str) -> str:
    """Strip ```json ... ``` wrappers if the LLM added them."""
    t = text.strip()
    if t.startswith("```"):
        # Drop opening fence (with optional language label like ```json)
        t = t.split("\n", 1)[1] if "\n" in t else t
        if t.endswith("```"):
            t = t[:-3]
    return t.strip()


def _usage_tokens(usage) -> tuple[int, int]:
    """Normalise a provider usage object to (input_tokens, output_tokens).

    Anthropic uses input_tokens / output_tokens; OpenAI (and Mistral via the
    OpenAI API) use prompt_tokens / completion_tokens.
    """
    if usage is None:
        return 0, 0
    inp = getattr(usage, "input_tokens", None)
    if inp is None:
        inp = getattr(usage, "prompt_tokens", 0)
    out = getattr(usage, "output_tokens", None)
    if out is None:
        out = getattr(usage, "completion_tokens", 0)
    return inp or 0, out or 0


def _log_call(record: dict) -> None:
    """Append one JSON line describing an LLM call.

    Wrapped in a bare except on purpose: observability must never be able to
    break the app. A failed log write is silently dropped.
    """
    try:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except Exception:
        pass


# ---- Public API ------------------------------------------------------------

def call_llm(
    *,
    system: str,
    user: str,
    schema: Type[T],
    model: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> T:
    """Send a prompt and get back a validated Pydantic object.

    The provider is auto-detected from the API key in .env; model and the
    defaults come from core.config.settings unless overridden by the caller.

    Raises:
        RuntimeError if no API key is configured.
        ValidationError / json.JSONDecodeError if the repair retry also fails.
    """
    # Provider is auto-detected from the API key present in .env.
    provider = settings.active_provider
    fn = _DISPATCH.get(provider)
    if fn is None:
        raise ValueError(
            f"Unknown provider '{provider}'. Choose from: {list(_DISPATCH)}"
        )

    model = model or settings.active_model
    max_tokens = max_tokens or settings.default_max_tokens
    temperature = temperature if temperature is not None else settings.default_temperature

    started = time.perf_counter()
    status = "ok"
    input_tokens = 0
    output_tokens = 0

    def _add(usage) -> None:
        nonlocal input_tokens, output_tokens
        i, o = _usage_tokens(usage)
        input_tokens += i
        output_tokens += o

    try:
        raw, usage = _call_with_retry(fn, system, user, model, max_tokens, temperature)
        _add(usage)

        # Happy path: parse and validate.
        try:
            return schema.model_validate(json.loads(_strip_fences(raw)))
        except (json.JSONDecodeError, ValidationError) as exc:
            parse_error = exc  # bind it — the `as` name is cleared after the block
            status = "repaired"

        # Repair attempt: tell the model exactly what was wrong and re-ask.
        repair_prompt = (
            f"{user}\n\n"
            "Your previous response did not match the required schema.\n"
            f"Error: {parse_error}\n\n"
            "JSON schema the object must satisfy:\n"
            f"{json.dumps(schema.model_json_schema(), indent=2)}\n\n"
            "Your previous response was:\n"
            f"{raw}\n\n"
            "Reply with ONLY a JSON object that has a real value for every "
            "required field. Do NOT return the schema definition itself. "
            "No prose, no code fences."
        )
        raw2, usage2 = _call_with_retry(
            fn, system, repair_prompt, model, max_tokens, temperature
        )
        _add(usage2)
        return schema.model_validate(json.loads(_strip_fences(raw2)))
    except Exception:
        status = "failed"
        raise
    finally:
        _log_call({
            "ts": datetime.now(timezone.utc).isoformat(),
            "provider": provider,
            "model": model,
            "schema": schema.__name__,
            "temperature": temperature,
            "status": status,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        })

# ---- Tool use API ----------------------------------------------------------
# Provider-agnostic tool use. Tools are declared in ONE neutral format:
#     {"name": str, "description": str, "parameters": <JSON Schema dict>}
# A per-provider adapter translates that into the provider's own tool format
# and runs the tool-call loop. Adding a provider = add one adapter + one
# entry in _TOOL_LOOPS; callers and tool specs never change.


def _to_anthropic_tools(tools: list[dict]) -> list[dict]:
    """Neutral tool specs -> Anthropic `tools` format."""
    return [
        {
            "name": t["name"],
            "description": t["description"],
            "input_schema": t["parameters"],
        }
        for t in tools
    ]


def _to_openai_tools(tools: list[dict]) -> list[dict]:
    """Neutral tool specs -> OpenAI `tools` format (also used for Mistral)."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            },
        }
        for t in tools
    ]


def _tool_loop_anthropic(
    client, system, user, schema, tools, tool_executor,
    model, max_tokens, temperature, max_iterations,
):
    """Tool-use loop on Anthropic's Messages API."""
    anthropic_tools = _to_anthropic_tools(tools)
    messages: list = [{"role": "user", "content": user}]

    for _ in range(max_iterations):
        response = _retry_on_rate_limit(lambda: client.messages.create(
            model=model, max_tokens=max_tokens, temperature=temperature,
            system=system, tools=anthropic_tools, messages=messages,
        ))
        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if getattr(block, "type", None) == "tool_use":
                    try:
                        result = tool_executor(block.name, dict(block.input))
                    except Exception as exc:
                        result = f"Tool execution error: {exc}"
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
            continue

        text = "\n".join(
            b.text for b in response.content
            if getattr(b, "type", None) == "text"
        ).strip()
        return schema.model_validate(json.loads(_strip_fences(text)))

    raise RuntimeError(
        f"Tool-use loop exceeded {max_iterations} iterations without a final answer."
    )


def _tool_loop_openai(
    client, system, user, schema, tools, tool_executor,
    model, max_tokens, temperature, max_iterations,
):
    """Tool-use loop on the OpenAI-compatible chat API (OpenAI and Mistral)."""
    openai_tools = _to_openai_tools(tools)
    messages: list = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    for _ in range(max_iterations):
        response = _retry_on_rate_limit(lambda: client.chat.completions.create(
            model=model, max_tokens=max_tokens, temperature=temperature,
            tools=openai_tools, messages=messages,
        ))
        message = response.choices[0].message

        if message.tool_calls:
            messages.append(message)  # the assistant turn that requested tools
            for call in message.tool_calls:
                try:
                    args = json.loads(call.function.arguments or "{}")
                    result = tool_executor(call.function.name, args)
                except Exception as exc:
                    result = f"Tool execution error: {exc}"
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result,
                })
            continue

        text = (message.content or "").strip()
        return schema.model_validate(json.loads(_strip_fences(text)))

    raise RuntimeError(
        f"Tool-use loop exceeded {max_iterations} iterations without a final answer."
    )


# provider -> adapter that runs the tool-use loop for that provider
_TOOL_LOOPS = {
    "anthropic": lambda *a: _tool_loop_anthropic(_get_anthropic(), *a),
    "openai":    lambda *a: _tool_loop_openai(_get_openai(), *a),
    "mistral":   lambda *a: _tool_loop_openai(_get_mistral(), *a),
}


@traceable(name="call_llm_with_tools")
def call_llm_with_tools(
    *,
    system: str,
    user: str,
    schema: Type[T],
    tools: list[dict],
    tool_executor: Callable[[str, dict], str],
    model: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    max_iterations: int = 8,
) -> T:
    """Run a provider-agnostic tool-use loop and return a validated object.

    The model is offered `tools` it can call; when it does, the call is run
    via `tool_executor(name, args_dict)` and the result fed back. The loop
    ends when the model returns a final answer or `max_iterations` is hit.

    `tools` uses ONE neutral format, translated per provider:
        {"name": str, "description": str, "parameters": <JSON Schema dict>}

    Provider, model, and defaults come from core.config.settings unless
    overridden. Works with whichever provider .env selects (Anthropic /
    OpenAI / Mistral) — the matching adapter is chosen automatically.
    """
    model = model or settings.active_model
    max_tokens = max_tokens or settings.default_max_tokens
    temperature = temperature if temperature is not None else settings.default_temperature

    provider = settings.active_provider
    loop = _TOOL_LOOPS.get(provider)
    if loop is None:
        raise ValueError(
            f"Unknown provider '{provider}'. Choose from: {list(_TOOL_LOOPS)}"
        )
    return loop(system, user, schema, tools, tool_executor,
                model, max_tokens, temperature, max_iterations)

