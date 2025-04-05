"""
Contains utilities to get intelligent responses from various AI providers
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from openai import OpenAI
from src.brainstorm.agents import Tool


class AIProvider(ABC):
    @abstractmethod
    def get_response(self, messages: List[Dict], **kwargs) -> str:
        pass

    @abstractmethod
    def get_response_from_tool(self, tool: Tool, messages: List[Dict], **kwargs) -> str:
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        pass


class OpenAIProvider(AIProvider):
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def get_response(self, messages: List[Dict], **kwargs) -> str:
        model = kwargs.pop('model', self.model)
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        )
        return response.choices[0].message.content

    def get_response_from_tool(self, tool: Tool, messages: List[Dict], **kwargs) -> str:
        return self.get_response(messages, **kwargs)

    @property
    def version(self) -> str:
        return f"OpenAI Provider (Model: {self.model})"


class OpenRouterProvider(AIProvider):
    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-4",
        site_url: Optional[str] = None,
        site_name: Optional[str] = None
    ):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        self.api_key = api_key
        self.model = model
        self.site_url = site_url
        self.site_name = site_name

    def get_response(self, messages: List[Dict], **kwargs) -> str:
        model = kwargs.pop('model', self.model)
        
        extra_headers = {
        }
        if self.site_url:
            extra_headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            extra_headers["X-Title"] = self.site_name

        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            extra_headers=extra_headers,
            **kwargs
        )
        return response.choices[0].message.content

    def get_response_from_tool(self, tool: Tool, messages: List[Dict], **kwargs) -> str:
        return self.get_response(messages, **kwargs)

    @property
    def version(self) -> str:
        return f"OpenRouter Provider (Model: {self.model})"


class AI:
    def __init__(self, provider: str = "openai", **kwargs):
        """
        Initialize AI with specified provider
        
        Args:
            provider: The provider to use ("openai" or "openrouter")
            **kwargs: Provider-specific arguments
                For OpenAI:
                    - api_key: Your OpenAI API key
                    - model: Model name (default: "gpt-4")
                For OpenRouter:
                    - api_key: Your OpenRouter API key
                    - model: Model name (default: "openai/gpt-4")
                    - site_url: Your site URL (optional)
                    - site_name: Your site name (optional)
        """
        if provider == "openai":
            self.provider = OpenAIProvider(**kwargs)
        elif provider == "openrouter":
            self.provider = OpenRouterProvider(**kwargs)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def get_response(self, messages: List[Dict], **kwargs) -> str:
        return self.provider.get_response(messages, **kwargs)

    def get_response_from_tool(self, tool: Tool, messages: List[Dict], **kwargs) -> str:
        return self.provider.get_response_from_tool(tool, messages, **kwargs)

    @property
    def version(self) -> str:
        return self.provider.version
