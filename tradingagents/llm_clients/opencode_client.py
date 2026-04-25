import os
from typing import Any, Optional, Mapping

from langchain_openai import ChatOpenAI
from langchain_openai.chat_models.base import (
    _convert_message_to_dict as _orig_convert,
    _convert_dict_to_message as _orig_dict_to_msg,
)
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage

from .base_client import BaseLLMClient, normalize_content
from .validators import validate_model


def _convert_message_to_dict_patched(message, api="chat/completions"):
    message_dict = _orig_convert(message, api=api)
    if isinstance(message, AIMessage):
        reasoning = message.additional_kwargs.get("reasoning_content")
        if reasoning is not None:
            message_dict["reasoning_content"] = reasoning
    return message_dict


def _convert_dict_to_message_patched(_dict: Mapping[str, Any]):
    message = _orig_dict_to_msg(_dict)
    if isinstance(message, AIMessage):
        reasoning = _dict.get("reasoning_content")
        if reasoning is not None:
            message.additional_kwargs["reasoning_content"] = reasoning
    return message


class NormalizedChatOpenAI(ChatOpenAI):
    def invoke(self, input, config=None, **kwargs):
        return normalize_content(super().invoke(input, config, **kwargs))

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        from langchain_openai.chat_models import base as openai_base
        original = openai_base._convert_message_to_dict
        openai_base._convert_message_to_dict = _convert_message_to_dict_patched
        try:
            payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        finally:
            openai_base._convert_message_to_dict = original
        if self.model_name and self.model_name.lower().startswith("kimi-k2"):
            for msg in payload.get("messages", []):
                if msg.get("role") == "assistant" and msg.get("tool_calls") and "reasoning_content" not in msg:
                    msg["reasoning_content"] = ""
        return payload

    def _create_chat_result(self, response, generation_info=None):
        from langchain_openai.chat_models import base as openai_base
        original = openai_base._convert_dict_to_message
        openai_base._convert_dict_to_message = _convert_dict_to_message_patched
        try:
            return super()._create_chat_result(response, generation_info=generation_info)
        finally:
            openai_base._convert_dict_to_message = original


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

        if self.model.lower().startswith("kimi-k2"):
            llm_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

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
