"""Application settings — multi-provider LLM configuration.

The active LLM provider is NOT set explicitly. It is auto-detected from
whichever ``*_API_KEY`` is present in ``.env``:

    ANTHROPIC_API_KEY set  -> Anthropic
    OPENAI_API_KEY set     -> OpenAI
    MISTRAL_API_KEY set    -> Mistral

If several keys are present, ``_PROVIDER_PRIORITY`` breaks the tie.
``core/llm.py`` reads ``settings.active_provider`` to pick the right client.
"""
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).parent.parent

Provider = Literal["anthropic", "openai", "mistral"]

# Detection order used when more than one API key is present in .env.
_PROVIDER_PRIORITY: tuple[Provider, ...] = ("anthropic", "openai", "mistral")


class Settings(BaseSettings):
    # One API key per supported provider. Only the one you intend to use
    # needs a value; the active provider is detected from these.
    anthropic_api_key: str = Field(default="")
    openai_api_key: str = Field(default="")
    mistral_api_key: str = Field(default="")

    anthropic_default_model: str = Field(default="claude-sonnet-4-5-20250929")
    openai_default_model: str = Field(default="gpt-4o")
    mistral_default_model: str = Field(default="mistral-large-latest")

    default_max_tokens: int = Field(default=1024, gt=0)
    default_temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    raw_data_dir: Path = Field(default=BASE_DIR / "data" / "raw")
    processed_data_dir: Path = Field(default=BASE_DIR / "data" / "processed")

    model_config = {
        "env_file": str(BASE_DIR / ".env"),
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        # .env legitimately holds keys this model doesn't declare (e.g. the
        # LANGSMITH_* vars, read straight from the environment by langsmith).
        # Without this, any such key makes Settings() raise on startup.
        "extra": "ignore",
    }

    def _api_keys(self) -> dict[str, str]:
        return {
            "anthropic": self.anthropic_api_key,
            "openai": self.openai_api_key,
            "mistral": self.mistral_api_key,
        }

    @property
    def active_provider(self) -> Provider:
        """The LLM provider to use, detected from whichever API key is set."""
        keys = self._api_keys()
        for provider in _PROVIDER_PRIORITY:
            if keys[provider].strip():
                return provider
        # Unreachable in practice: _require_at_least_one_key validates first.
        raise RuntimeError(
            "No LLM API key found. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, "
            "or MISTRAL_API_KEY in .env."
        )

    @property
    def active_model(self) -> str:
        return {
            "anthropic": self.anthropic_default_model,
            "openai": self.openai_default_model,
            "mistral": self.mistral_default_model,
        }[self.active_provider]

    @property
    def active_api_key(self) -> str:
        return self._api_keys()[self.active_provider]

    @model_validator(mode="after")
    def _require_at_least_one_key(self) -> "Settings":
        if not any(k.strip() for k in self._api_keys().values()):
            raise ValueError(
                "No LLM API key set in .env — add one of ANTHROPIC_API_KEY, "
                "OPENAI_API_KEY, or MISTRAL_API_KEY."
            )
        return self

    @model_validator(mode="after")
    def _ensure_data_dirs(self) -> "Settings":
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        return self


settings = Settings()
