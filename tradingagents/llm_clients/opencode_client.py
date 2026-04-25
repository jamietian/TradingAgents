import os
from typing import Any, Optional

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from .base_client import BaseLLMClient, normalize_content
from .validators import validate_model


class NormalizedChatOpenAI(ChatOpenAI):
    def invoke(self, input, config=None, **kwargs):
        return normalize_content(super().invoke(input, config, **kwargs))


class NormalizedChatAnthropic(ChatAnthropic):
    def invoke(self, input, config=None, **kwargs):
        return normalize_content(super().invoke(input, config, **kwargs))


_PASSTHROUGH_OPENAI = (
    "timeout", "max_retries", "reasoning_effort",
    "api_key", "callbacks", "http_client", "http_async_client",
)
_PASSTHROUGH_ANTHROPIC = (
    "timeout", "max_retries", "api_key", "max_tokens",
    "callbacks", "http_client", "http_async_client", "effort",
)

_ANTHROPIC_MODEL_PREFIXES = ("claude-", "minimax-")
_BASE_URLS = {
    "opencode": "https://opencode.ai/zen/v1",
    "opencode-go": "https://opencode.ai/zen/go/v1",
}
_API_KEY_ENV = "OPENCODE_API_KEY"


class OpencodeClient(BaseLLMClient):
    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        provider: str = "opencode",
        **kwargs,
    ):
        super().__init__(model, base_url, **kwargs)
        self.provider = provider.lower()

    def _is_anthropic_model(self) -> bool:
        model_lower = self.model.lower()
        return any(model_lower.startswith(prefix) for prefix in _ANTHROPIC_MODEL_PREFIXES)

    def get_llm(self) -> Any:
        self.warn_if_unknown_model()
        if self._is_anthropic_model():
            return self._get_anthropic_llm()
        return self._get_openai_llm()

    def _get_openai_llm(self) -> Any:
        llm_kwargs = {"model": self.model}

        base_url = self.base_url or _BASE_URLS.get(self.provider)
        if base_url:
            llm_kwargs["base_url"] = base_url

        api_key = self.kwargs.get("api_key") or os.environ.get(_API_KEY_ENV)
        if api_key:
            llm_kwargs["api_key"] = api_key

        for key in _PASSTHROUGH_OPENAI:
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]

        if self.provider == "opencode":
            llm_kwargs["use_responses_api"] = True

        return NormalizedChatOpenAI(**llm_kwargs)

    def _get_anthropic_llm(self) -> Any:
        llm_kwargs = {"model": self.model}

        base_url = self.base_url or _BASE_URLS.get(self.provider)
        if base_url:
            llm_kwargs["base_url"] = base_url

        api_key = self.kwargs.get("api_key") or os.environ.get(_API_KEY_ENV)
        if api_key:
            llm_kwargs["api_key"] = api_key

        for key in _PASSTHROUGH_ANTHROPIC:
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]

        return NormalizedChatAnthropic(**llm_kwargs)

    def validate_model(self) -> bool:
        return validate_model(self.provider, self.model)
