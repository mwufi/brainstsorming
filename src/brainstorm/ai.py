"""
Contains utilities to get intelligent responses from various AI providers
"""

import json
import os
from typing import List, Dict, Optional, Callable, Protocol
from openai import OpenAI
from src.brainstorm.tools import Tool


# Load models from JSON file
def load_models():
    models_path = os.path.join(os.path.dirname(__file__), "models.json")
    with open(models_path, "r") as f:
        return json.load(f)


# Get model info from the loaded models
def get_model_info(model_name: str, provider: str = None):
    models = load_models()
    
    # If provider is specified, look in that provider's models
    if provider and provider in models:
        if model_name in models[provider]:
            return models[provider][model_name]
    
    # Otherwise search in all providers
    for provider_models in models.values():
        if model_name in provider_models:
            return provider_models[model_name]
    
    return None


class ProviderConfig:
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: Optional[str] = None,
        extra_headers: Optional[Dict] = None,
        name: str = "Unknown Provider"
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.extra_headers = extra_headers or {}
        self.name = name
        
        # Validate model
        model_info = get_model_info(model)
        if not model_info:
            raise ValueError(f"Unknown model: {model}")


def create_openai_client(config: ProviderConfig) -> OpenAI:
    return OpenAI(
        api_key=config.api_key,
        base_url=config.base_url
    )


def handle_response(response) -> str:
    if hasattr(response, 'error') and response.error:
        error_msg = response.error.get('message', 'Unknown error occurred')
        error_code = response.error.get('code', 'unknown')
        raise Exception(f"AI Provider Error: {error_msg} (Code: {error_code})")
        
    if not response.choices:
        raise Exception("No response choices available")
        
    return response.choices[0].message.content


def create_provider(config: ProviderConfig) -> Dict:
    client = create_openai_client(config)
    
    def get_response(messages: List[Dict], **kwargs) -> str:
        model = kwargs.pop('model', config.model)
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            extra_headers=config.extra_headers,
            **kwargs
        )
        
        return handle_response(response)
    
    def get_response_from_tool(tool: Tool, messages: List[Dict], **kwargs) -> str:
        return get_response(messages, **kwargs)
    
    def get_version() -> str:
        return f"{config.name} (Model: {config.model})"
    
    return {
        'get_response': get_response,
        'get_response_from_tool': get_response_from_tool,
        'version': get_version
    }


def create_openai_provider(api_key: str, model: str = "gpt-4o") -> Dict:
    config = ProviderConfig(
        api_key=api_key,
        model=model,
        name="OpenAI Provider"
    )
    return create_provider(config)


def create_openrouter_provider(
    api_key: str,
    model: str = "openai/gpt-4o",
    site_url: Optional[str] = None,
    site_name: Optional[str] = None
) -> Dict:
    extra_headers = {}
    if site_url:
        extra_headers["HTTP-Referer"] = site_url
    if site_name:
        extra_headers["X-Title"] = site_name

    config = ProviderConfig(
        api_key=api_key,
        model=model,
        base_url="https://openrouter.ai/api/v1",
        extra_headers=extra_headers,
        name="OpenRouter Provider"
    )
    return create_provider(config)


class AI:
    def __init__(self, provider: str = "openai", **kwargs):
        """
        Initialize AI with specified provider
        
        Args:
            provider: The provider to use ("openai" or "openrouter")
            **kwargs: Provider-specific arguments
                For OpenAI:
                    - api_key: Your OpenAI API key
                    - model: Model name (default: "gpt-4o")
                For OpenRouter:
                    - api_key: Your OpenRouter API key
                    - model: Model name (default: "openai/gpt-4o")
                    - site_url: Your site URL (optional)
                    - site_name: Your site name (optional)
        """
        if provider == "openai":
            self.provider = create_openai_provider(**kwargs)
        elif provider == "openrouter":
            self.provider = create_openrouter_provider(**kwargs)
        else:
            raise ValueError(f"Unknown provider: {provider}")
            
        # Store model info for reference
        self.model_name = kwargs.get('model', self._get_default_model(provider))
        self.model_info = get_model_info(self.model_name, provider)

    def _get_default_model(self, provider: str) -> str:
        """Get the default model for a provider"""
        if provider == "openai":
            return "gpt-4o"
        elif provider == "openrouter":
            return "openai/gpt-4o"
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def get_response(self, messages: List[Dict], **kwargs) -> str:
        return self.provider['get_response'](messages, **kwargs)

    def get_response_from_tool(self, tool: Tool, messages: List[Dict], **kwargs) -> str:
        return self.provider['get_response_from_tool'](tool, messages, **kwargs)

    @property
    def version(self) -> str:
        return self.provider['version']()
        
    @property
    def model_description(self) -> str:
        """Get a description of the current model"""
        return self.model_info.get('description', "Unknown model") if self.model_info else "Unknown model"
        
    @property
    def max_tokens(self) -> Optional[int]:
        """Get the maximum tokens for the current model"""
        return self.model_info.get('max_tokens') if self.model_info else None
