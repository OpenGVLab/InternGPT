"""
LLM provider factory for InternGPT.

Supports OpenAI (default) and MiniMax as LLM backends.
MiniMax models use the OpenAI-compatible chat completion API
(https://api.minimax.io/v1).

Usage:
    # Auto-detect from environment variables
    llm = create_llm()

    # Explicit provider selection
    llm = create_llm(provider="minimax")
"""

import os
import re
from typing import Any, List, Mapping, Optional

import openai

# Handle both old (0.x) and new (1.x+) openai SDK
_OPENAI_V1 = hasattr(openai, "__version__") and int(openai.__version__.split(".")[0]) >= 1

try:
    from langchain.llms.base import LLM
except ImportError:
    from langchain_core.language_models.llms import LLM

try:
    from langchain.llms.openai import OpenAI
except ImportError:
    from langchain_openai import OpenAI


# ---------------------------------------------------------------------------
# MiniMax-specific constants
# ---------------------------------------------------------------------------
MINIMAX_API_BASE = "https://api.minimax.io/v1"
MINIMAX_DEFAULT_MODEL = "MiniMax-M2.7"
MINIMAX_MODELS = [
    "MiniMax-M2.7",
    "MiniMax-M2.7-highspeed",
    "MiniMax-M2.5",
    "MiniMax-M2.5-highspeed",
]


def _chat_completion(model, messages, temperature, stop, api_key, api_base):
    """Call chat completion API, compatible with both old and new openai SDK."""
    if _OPENAI_V1:
        client = openai.OpenAI(api_key=api_key, base_url=api_base)
        kwargs = dict(model=model, messages=messages, temperature=temperature)
        if stop:
            kwargs["stop"] = stop
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""
    else:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stop=stop,
            api_key=api_key,
            api_base=api_base,
        )
        return response.choices[0].message.content or ""


class MiniMaxLLM(LLM):
    """LangChain LLM wrapper for MiniMax via OpenAI-compatible API.

    MiniMax exposes an OpenAI-compatible ``/v1/chat/completions`` endpoint.
    This class translates a plain-text prompt into a single-user-message chat
    request so that it can be used with LangChain agents that expect a
    ``BaseLLM`` interface.

    Parameters
    ----------
    model_name : str
        MiniMax model identifier (default ``MiniMax-M2.7``).
    temperature : float
        Sampling temperature.  MiniMax requires ``temperature`` in (0, 1],
        so values <= 0 are clamped to 0.01.
    api_key : str
        MiniMax API key.  Falls back to the ``MINIMAX_API_KEY`` env var.
    api_base : str
        Base URL for the MiniMax API (default ``https://api.minimax.io/v1``).
    """

    model_name: str = MINIMAX_DEFAULT_MODEL
    temperature: float = 0.01
    api_key: str = ""
    api_base: str = MINIMAX_API_BASE

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        if not self.api_key:
            self.api_key = os.environ.get("MINIMAX_API_KEY", "")
        # Clamp temperature to MiniMax's valid range (0, 1]
        if self.temperature <= 0:
            self.temperature = 0.01

    @property
    def _llm_type(self) -> str:
        return "minimax"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "api_base": self.api_base,
        }

    def _strip_think_tags(self, text: str) -> str:
        """Remove <think>...</think> blocks from model output."""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        """Call MiniMax chat completion API with a single user message."""
        content = _chat_completion(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            stop=stop,
            api_key=self.api_key,
            api_base=self.api_base,
        )
        return self._strip_think_tags(content)


# ---------------------------------------------------------------------------
# Provider detection helpers
# ---------------------------------------------------------------------------
SUPPORTED_PROVIDERS = ("openai", "minimax")


def detect_provider() -> str:
    """Detect the LLM provider from environment variables.

    Priority:
    1. ``LLM_PROVIDER`` env var (explicit).
    2. If ``MINIMAX_API_KEY`` is set and ``OPENAI_API_KEY`` is not → minimax.
    3. Default → openai.
    """
    explicit = os.environ.get("LLM_PROVIDER", "").lower().strip()
    if explicit in SUPPORTED_PROVIDERS:
        return explicit

    has_minimax = bool(os.environ.get("MINIMAX_API_KEY"))
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))

    if has_minimax and not has_openai:
        return "minimax"

    return "openai"


def create_llm(
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0,
) -> LLM:
    """Create an LLM instance for the given provider.

    Parameters
    ----------
    provider : str, optional
        ``"openai"`` or ``"minimax"``.  Auto-detected if *None*.
    api_key : str, optional
        API key override.  Otherwise read from the environment.
    model_name : str, optional
        Model name override.
    temperature : float
        Sampling temperature (default 0).

    Returns
    -------
    langchain.llms.base.LLM
        A LangChain-compatible LLM instance.
    """
    if provider is None:
        provider = detect_provider()

    provider = provider.lower().strip()

    if provider == "minimax":
        kwargs: dict = {"temperature": temperature}
        if api_key:
            kwargs["api_key"] = api_key
        if model_name:
            kwargs["model_name"] = model_name
        return MiniMaxLLM(**kwargs)

    # Default: OpenAI
    kwargs_openai: dict = {"temperature": temperature}
    if api_key:
        kwargs_openai["openai_api_key"] = api_key
    if model_name:
        kwargs_openai["model_name"] = model_name
    return OpenAI(**kwargs_openai)
