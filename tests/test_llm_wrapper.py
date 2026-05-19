"""Layer 1 — deterministic unit tests for the LLM wrapper + config.

Covers the pure / I/O parts that need no real API call: provider
auto-detection, fence stripping, token-usage normalisation, the JSONL
observability log, and provider routing.
"""
import json

import pytest
from pydantic import ValidationError

import core.llm as llm_mod
from core.config import Settings, settings
from core.llm import _strip_fences, _usage_tokens
from core.models import DimensionScore


class TestProviderDetection:
    """config.Settings auto-detects the provider from the API key in .env."""

    @staticmethod
    def _settings(**keys):
        # _env_file=None ignores the real .env; explicit kwargs take priority.
        base = {"anthropic_api_key": "", "openai_api_key": "", "mistral_api_key": ""}
        base.update(keys)
        return Settings(_env_file=None, **base)

    def test_detects_anthropic_from_its_key(self):
        s = self._settings(anthropic_api_key="sk-ant-x")
        assert s.active_provider == "anthropic"
        assert s.active_model == s.anthropic_default_model
        assert s.active_api_key == "sk-ant-x"

    def test_detects_openai_from_its_key(self):
        assert self._settings(openai_api_key="sk-x").active_provider == "openai"

    def test_detects_mistral_from_its_key(self):
        assert self._settings(mistral_api_key="m-x").active_provider == "mistral"

    def test_priority_breaks_ties_when_several_keys_present(self):
        s = self._settings(anthropic_api_key="a", openai_api_key="o", mistral_api_key="m")
        assert s.active_provider == "anthropic"

    def test_no_key_set_raises(self):
        with pytest.raises(ValidationError):
            self._settings()


class TestStripFences:
    def test_plain_json_untouched(self):
        assert _strip_fences('{"a": 1}') == '{"a": 1}'

    def test_strips_json_labelled_fence(self):
        assert _strip_fences('```json\n{"a": 1}\n```') == '{"a": 1}'

    def test_strips_bare_fence(self):
        assert _strip_fences('```\n{"a": 1}\n```') == '{"a": 1}'

    def test_strips_surrounding_whitespace(self):
        assert _strip_fences('  \n{"a": 1}\n  ') == '{"a": 1}'


class TestUsageTokens:
    """Anthropic and OpenAI report token usage under different field names."""

    def test_none_usage_is_zero(self):
        assert _usage_tokens(None) == (0, 0)

    def test_anthropic_style_field_names(self):
        class U:
            input_tokens = 10
            output_tokens = 20
        assert _usage_tokens(U()) == (10, 20)

    def test_openai_style_field_names(self):
        class U:
            prompt_tokens = 5
            completion_tokens = 7
        assert _usage_tokens(U()) == (5, 7)


class TestProviderRouting:
    def test_dispatch_table_has_the_three_providers(self):
        assert set(llm_mod._DISPATCH) == {"anthropic", "openai", "mistral"}

    def test_call_llm_routes_to_the_active_provider(self, monkeypatch, tmp_path):
        # Fake the active provider's dispatch entry — no real API call.
        canned = '{"score": 4, "comment": "ok"}'
        monkeypatch.setattr(
            llm_mod, "_DISPATCH",
            {settings.active_provider: lambda *a: (canned, None)},
        )
        monkeypatch.setattr(llm_mod, "_LOG_PATH", tmp_path / "logs" / "calls.jsonl")

        result = llm_mod.call_llm(system="s", user="u", schema=DimensionScore)
        assert result.score == 4
        assert result.comment == "ok"


class TestLogCall:
    def test_writes_one_json_line(self, tmp_path, monkeypatch):
        log_path = tmp_path / "logs" / "llm_calls.jsonl"
        monkeypatch.setattr(llm_mod, "_LOG_PATH", log_path)

        llm_mod._log_call({"schema": "ScoreReport", "status": "ok"})

        lines = log_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0]) == {"schema": "ScoreReport", "status": "ok"}

    def test_appends_across_calls(self, tmp_path, monkeypatch):
        log_path = tmp_path / "logs" / "llm_calls.jsonl"
        monkeypatch.setattr(llm_mod, "_LOG_PATH", log_path)

        llm_mod._log_call({"n": 1})
        llm_mod._log_call({"n": 2})

        lines = log_path.read_text(encoding="utf-8").splitlines()
        assert [json.loads(line)["n"] for line in lines] == [1, 2]

    def test_never_raises_on_unwritable_path(self, monkeypatch):
        # A path whose parent cannot be created — _log_call must swallow it.
        monkeypatch.setattr(llm_mod, "_LOG_PATH", llm_mod.Path("\0bad/llm.jsonl"))
        llm_mod._log_call({"schema": "X"})  # must not raise


class TestCallLlmLogging:
    def test_successful_call_logs_provider_and_status(self, monkeypatch, tmp_path):
        log_path = tmp_path / "logs" / "calls.jsonl"
        monkeypatch.setattr(llm_mod, "_LOG_PATH", log_path)
        monkeypatch.setattr(
            llm_mod, "_DISPATCH",
            {settings.active_provider: lambda *a: ('{"score": 3, "comment": "x"}', None)},
        )

        llm_mod.call_llm(system="s", user="u", schema=DimensionScore)

        record = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])
        assert record["status"] == "ok"
        assert record["provider"] == settings.active_provider
        assert record["schema"] == "DimensionScore"


class TestRateLimitRetry:
    """call_llm retries rate-limited provider calls with backoff and tells the UI."""

    def test_notifier_receives_messages(self):
        seen = []
        llm_mod.set_retry_notifier(seen.append)
        try:
            llm_mod._notify_retry("hello")
        finally:
            llm_mod.set_retry_notifier(None)
        assert seen == ["hello"]

    def test_broken_notifier_never_raises(self):
        def boom(_msg):
            raise RuntimeError("bad notifier")
        llm_mod.set_retry_notifier(boom)
        try:
            llm_mod._notify_retry("x")  # must not raise
        finally:
            llm_mod.set_retry_notifier(None)

    def test_retry_delay_is_exponential_without_a_header(self):
        class NoResponse:
            response = None
        assert llm_mod._retry_delay(NoResponse(), 0) == 2.0
        assert llm_mod._retry_delay(NoResponse(), 1) == 4.0
        assert llm_mod._retry_delay(NoResponse(), 2) == 8.0

    def test_retry_delay_honours_retry_after_header(self):
        class Resp:
            headers = {"retry-after": "7"}
        class Exc:
            response = Resp()
        assert llm_mod._retry_delay(Exc(), 0) == 7.0

    def test_retry_delay_is_capped(self):
        class NoResponse:
            response = None
        assert llm_mod._retry_delay(NoResponse(), 10) == 20.0  # would be 2048 uncapped

    def test_call_with_retry_recovers_after_transient_limits(self, monkeypatch):
        monkeypatch.setattr(llm_mod.time, "sleep", lambda _s: None)

        class FakeRateLimit(Exception):
            pass
        monkeypatch.setattr(llm_mod, "_RATE_LIMIT_ERRORS", (FakeRateLimit,))

        attempts = {"n": 0}

        def fn(*_args):
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise FakeRateLimit()
            return ("recovered", None)

        text, _usage = llm_mod._call_with_retry(fn, "s", "u", "m", 10, 0.0)
        assert text == "recovered"
        assert attempts["n"] == 3

    def test_call_with_retry_gives_up_with_a_clear_error(self, monkeypatch):
        monkeypatch.setattr(llm_mod.time, "sleep", lambda _s: None)

        class FakeRateLimit(Exception):
            pass
        monkeypatch.setattr(llm_mod, "_RATE_LIMIT_ERRORS", (FakeRateLimit,))

        def always_limited(*_args):
            raise FakeRateLimit()

        with pytest.raises(RuntimeError, match="rate-limiting"):
            llm_mod._call_with_retry(always_limited, "s", "u", "m", 10, 0.0)


class TestToolUse:
    """Provider-agnostic tool use: one neutral spec, per-provider adapters."""

    _NEUTRAL = [{
        "name": "lookup",
        "description": "look something up",
        "parameters": {
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
        },
    }]

    def test_neutral_specs_translate_to_anthropic_format(self):
        out = llm_mod._to_anthropic_tools(self._NEUTRAL)
        assert out[0]["name"] == "lookup"
        assert out[0]["input_schema"] == self._NEUTRAL[0]["parameters"]
        assert "parameters" not in out[0]

    def test_neutral_specs_translate_to_openai_format(self):
        out = llm_mod._to_openai_tools(self._NEUTRAL)
        assert out[0]["type"] == "function"
        assert out[0]["function"]["name"] == "lookup"
        assert out[0]["function"]["parameters"] == self._NEUTRAL[0]["parameters"]

    def test_tool_loops_cover_all_three_providers(self):
        assert set(llm_mod._TOOL_LOOPS) == {"anthropic", "openai", "mistral"}

    def test_call_llm_with_tools_routes_to_the_active_provider(self, monkeypatch):
        seen = {}

        def fake_loop(system, user, schema, tools, tool_executor,
                      model, max_tokens, temperature, max_iterations):
            seen["ran"] = True
            return "RESULT"

        monkeypatch.setitem(llm_mod._TOOL_LOOPS, settings.active_provider, fake_loop)
        result = llm_mod.call_llm_with_tools(
            system="s", user="u", schema=DimensionScore,
            tools=self._NEUTRAL, tool_executor=lambda name, args: "x",
        )
        assert seen["ran"] is True
        assert result == "RESULT"
