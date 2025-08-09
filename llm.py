import json
import os
from abc import ABC, abstractmethod

from typing import Any
import backoff
from anthropic import AsyncAnthropic
from logger import get_logger
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from utils import is_token_limit_error

provider_args = {
    "openai": {"api_key": os.getenv("OPENAI_API_KEY")},
    "google": {
        "api_key": os.getenv("GOOGLE_API_KEY"),
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    },
    "anthropic": {
        "api_key": os.getenv("ANTHROPIC_API_KEY"),
        "base_url": "https://api.anthropic.com/v1/",
    },
    "together": {
        "api_key": os.getenv("TOGETHER_API_KEY"),
        "base_url": "https://api.together.xyz/v1/",
    },
    "fireworks": {
        "api_key": os.getenv("FIREWORKS_API_KEY"),
        "base_url": "https://api.fireworks.ai/v1/",
    },
    "mistralai": {
        "api_key": os.getenv("MISTRAL_API_KEY"),
        "base_url": "https://api.mistral.ai/v1/",
    },
    "grok": {
        "api_key": os.getenv("GROK_API_KEY"),
        "base_url": "https://api.x.ai/v1",
    },
    "cohere": {
        "api_key": os.getenv("COHERE_API_KEY"),
        "base_url": "https://api.cohere.ai/compatibility/v1/",
    },
}

llm_logger = get_logger(__name__)


class LLM(ABC):
    def __init__(self, provider: str, model_name: str):
        self.provider = provider
        self.model_name = model_name
        params = self.get_provider_args()
        if self.provider == "fireworks":
            self.model_name = "accounts/fireworks/models/" + self.model_name
        self.client = AsyncOpenAI(
            api_key=params["api_key"], base_url=params.get("base_url")
        )

    def get_provider_args(self) -> dict[str, Any]:
        params = provider_args[self.provider]
        return params

    def chat(self, conversation: list[dict[str, Any]]) -> Any:
        raise NotImplementedError

    @abstractmethod
    def parse_response(self, response: Any) -> str:
        pass

    @abstractmethod
    def append_tool_result(
        self,
        messages: list[dict[str, Any]],
        tool_content: Any,
        tool_result: str,
    ) -> list[dict[str, Any]]:
        pass

    @abstractmethod
    def get_tool_calls(self, response: Any) -> list[dict[str, Any]]:
        pass

    @abstractmethod
    def convert_usage(self, usage: Any) -> dict[str, Any]:
        pass


class GeneralLLM(LLM):
    def __init__(
        self,
        provider: str,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int = 8192,
    ):
        super().__init__(provider=provider, model_name=model_name)
        self.temperature = temperature
        self.max_tokens = max_tokens
        if provider == "anthropic":
            params = self.get_provider_args()
            self.anthropic_client = AsyncAnthropic(api_key=params["api_key"])

    async def safe_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] = [],
        ignore_token_error: bool = False,
    ) -> ChatCompletion:
        while True:
            try:
                return await self._retryable_chat(messages, tools)
            except Exception as e:
                error_str = str(e).lower()
                if is_token_limit_error(error_str):
                    if ignore_token_error:
                        raise Exception(f"Token limit error: {str(e)}")
                    if len(messages) > 2:
                        llm_logger.warning(
                            f"Too long, removing oldest message pair. {len(messages)}"
                        )
                        messages.pop(1)
                        continue
                    else:
                        raise Exception(
                            f"Cannot reduce message context any further: {str(e)}"
                        )
                raise

    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=None,
        max_value=150,
        max_time=1200,
        jitter=backoff.full_jitter,
        giveup=lambda e: is_token_limit_error(str(e).lower()),
    )
    async def _retryable_chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ChatCompletion:
        try:
            return await self.chat(messages, tools)
        except Exception as e:
            print(f"Error: {e}")
            raise

    async def _openai_responses_chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] = []
    ):
        input_payload = messages
        args = {
            "model": self.model_name,
            "input": input_payload,
        }
        if self.temperature is not None:
            args["temperature"] = self.temperature
        if self.max_tokens is not None:
            args["max_output_tokens"] = self.max_tokens
        if tools:
            args["tools"] = tools
        return await self.client.responses.create(**args)

    async def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] = []
    ) -> ChatCompletion:
        if self.provider == "anthropic":
            if self.model_name == "claude-3-7-sonnet-20250219-thinking":
                model_name = "claude-3-7-sonnet-20250219"
                if len(tools) > 0:
                    return await self.anthropic_client.messages.create(
                        model=model_name,
                        messages=messages,
                        temperature=1,
                        tools=tools,
                        thinking={"type": "enabled", "budget_tokens": 13384},
                        max_tokens=16384,
                    )
                else:
                    return await self.anthropic_client.messages.create(
                        model=model_name,
                        messages=messages,
                        temperature=1,
                        max_tokens=16384,
                        thinking={"type": "enabled", "budget_tokens": 13384},
                    )
            else:
                if len(tools) > 0:
                    return await self.anthropic_client.messages.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=self.temperature,
                        tools=tools,
                        max_tokens=8192,
                    )
                else:
                    return await self.anthropic_client.messages.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=8192,
                    )

        if self.provider == "openai":
            return await self._openai_responses_chat(messages, tools)

        if self.model_name in [
            "o3-mini-2025-01-31",
            "o1-2024-12-17",
            "o3-2025-04-16",
            "o4-mini-2025-04-16",
        ]:
            if len(tools) > 0:
                return await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=tools,
                )
            else:
                return await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                )

        if self.model_name == "grok-3-mini-fast-beta-high-reasoning":
            if len(tools) > 0:
                return await self.client.chat.completions.create(
                    model="grok-3-mini-fast-beta",
                    messages=messages,
                    temperature=self.temperature,
                    tools=tools,
                    max_tokens=self.max_tokens,
                    reasoning_effort="high",
                )
            else:
                return await self.client.chat.completions.create(
                    model="grok-3-mini-fast-beta",
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    reasoning_effort="high",
                )

        if self.model_name == "grok-3-mini-fast-beta-low-reasoning":
            if len(tools) > 0:
                return await self.client.chat.completions.create(
                    model="grok-3-mini-fast-beta",
                    messages=messages,
                    temperature=self.temperature,
                    tools=tools,
                    max_tokens=self.max_tokens,
                    reasoning_effort="low",
                )
            else:
                return await self.client.chat.completions.create(
                    model="grok-3-mini-fast-beta",
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    reasoning_effort="low",
                )

        if len(tools) > 0:
            return await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                tools=tools,
                max_tokens=self.max_tokens,
            )
        else:
            return await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

    def _responses_output_items(self, response: Any) -> list[Any]:
        items = []
        try:
            if hasattr(response, "output") and response.output:
                items = response.output
        except Exception:
            pass
        return items

    def get_tool_calls(self, response: Any) -> list[dict[str, Any]]:
        tools = []

        if self.provider == "anthropic":
            for content in response.content:
                if content.type == "tool_use":
                    tools.append(
                        {
                            "name": content.name,
                            "arguments": content.input,
                            "tool_content": content,
                        }
                    )
            return tools

        if self.provider == "openai":
            items = self._responses_output_items(response)
            for item in items:
                if getattr(item, "type", None) in ("tool_call", "function_call"):
                    name = getattr(item, "name", None) or getattr(
                        getattr(item, "function", None), "name", None
                    )
                    args_raw = getattr(item, "arguments", None)
                    try:
                        args = (
                            json.loads(args_raw)
                            if isinstance(args_raw, str)
                            else (args_raw or {})
                        )
                    except Exception:
                        args = {}
                    tools.append(
                        {
                            "name": name,
                            "arguments": args,
                            "tool_content": item,
                        }
                    )
            return tools

        for choice in response.choices:
            if choice.message.tool_calls:
                for tool_call in choice.message.tool_calls:
                    tools.append(
                        {
                            "name": tool_call.function.name,
                            "arguments": json.loads(tool_call.function.arguments),
                            "tool_content": tool_call,
                        }
                    )
        return tools

    def parse_response(self, response: Any) -> str:
        if self.provider == "anthropic":
            for content in response.content:
                if content.type == "text":
                    return content.text
            return ""

        if self.provider == "openai":
            if hasattr(response, "output_text") and response.output_text is not None:
                return response.output_text
            items = self._responses_output_items(response)
            texts = []
            for item in items:
                if getattr(item, "type", None) in ("message", "output_text", "text"):
                    text_val = getattr(item, "content", None) or getattr(
                        item, "text", None
                    )
                    if isinstance(text_val, str):
                        texts.append(text_val)
            return "\n".join(texts)

        return response.choices[0].message.content

    def append_tool_result(
        self,
        messages: list[dict[str, Any]],
        tool_content: Any,
        tool_result: str,
    ) -> list[dict[str, Any]]:
        if self.provider == "anthropic":
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_content.id,
                            "content": tool_result,
                        }
                    ],
                }
            )
        elif self.provider == "openai":
            messages.append(
                {
                    "role": "tool",
                    "content": tool_result if tool_result is not None else "",
                    "tool_call_id": getattr(tool_content, "id", None),
                }
            )
        else:
            messages.append(
                {
                    "role": "tool",
                    "content": tool_result if tool_result is not None else "",
                    "tool_call_id": tool_content.id,
                }
            )
        return messages

    def convert_usage(self, usage: Any) -> dict[str, Any]:
        if self.provider == "anthropic":
            return {
                "prompt_tokens": usage.input_tokens,
                "completion_tokens": usage.output_tokens,
                "total_tokens": usage.input_tokens + usage.output_tokens,
            }
        else:
            try:
                return {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                }
            except Exception:
                try:
                    return {
                        "prompt_tokens": getattr(usage, "input_tokens", 0),
                        "completion_tokens": getattr(usage, "output_tokens", 0),
                        "total_tokens": getattr(usage, "total_tokens", 0),
                    }
                except Exception:
                    return {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    }
