"""
Contains utilities to get intelligent responses from various AI providers
"""

import json
import os
from typing import List, Dict, Optional, Callable, Protocol, Iterator, Union
from openai import OpenAI
from src.brainstorm.tools import Tool


# Default values
DEFAULT_MODELS = {
    "openai": "gpt-4o",
    "openrouter": "openai/gpt-4o"
}

DEFAULT_BASE_URLS = {
    "openrouter": "https://openrouter.ai/api/v1"
}


def load_models():
    """Load models from JSON file"""
    models_path = os.path.join(os.path.dirname(__file__), "models.json")
    with open(models_path, "r") as f:
        return json.load(f)


def get_model_info(model_name: str, provider: str = None):
    """Get model info from the loaded models"""
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
    """Configuration for an AI provider"""
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
    """Create an OpenAI client with the given configuration"""
    return OpenAI(
        api_key=config.api_key,
        base_url=config.base_url
    )


def handle_response(response) -> str:
    """Handle response from AI provider"""
    if hasattr(response, 'error') and response.error:
        error_msg = response.error.get('message', 'Unknown error occurred')
        error_code = response.error.get('code', 'unknown')
        raise Exception(f"AI Provider Error: {error_msg} (Code: {error_code})")
        
    if not response.choices:
        raise Exception("No response choices available")
        
    return response.choices[0].message.content


def handle_streaming_response(response) -> Iterator[str]:
    """Handle streaming response from AI provider"""
    for chunk in response:
        # Different providers may have different chunk formats
        # We'll handle the most common ones here
        
        # Handle OpenAI-style streaming format
        if hasattr(chunk, 'choices') and chunk.choices:
            choice = chunk.choices[0]
            
            # Handle the delta format (newer OpenAI API)
            if hasattr(choice, 'delta'):
                delta = choice.delta
                if hasattr(delta, 'content') and delta.content is not None:
                    yield delta.content
            
            # Handle the message format (some APIs)
            elif hasattr(choice, 'message'):
                message = choice.message
                if hasattr(message, 'content') and message.content is not None:
                    yield message.content
                    
            # Handle text format (older APIs)
            elif hasattr(choice, 'text'):
                if choice.text:
                    yield choice.text


def create_provider(config: ProviderConfig) -> Dict:
    """Create a provider from configuration"""
    client = create_openai_client(config)
    
    def get_response(messages: List[Dict], **kwargs) -> str:
        """Get a complete response from the provider"""
        model = kwargs.pop('model', config.model)
        stream_mode = kwargs.pop('stream', False)
        
        if stream_mode:
            return get_streaming_response(messages, model=model, **kwargs)
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            extra_headers=config.extra_headers,
            **kwargs
        )
        
        return handle_response(response)
    
    def get_streaming_response(messages: List[Dict], **kwargs) -> Iterator[str]:
        """Get a streaming response from the provider"""
        model = kwargs.pop('model', config.model)
        
        # Make sure we don't have stream in kwargs
        if 'stream' in kwargs:
            kwargs.pop('stream')
            
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            extra_headers=config.extra_headers,
            stream=True,
            **kwargs
        )
        
        return handle_streaming_response(response)
    
    def get_response_from_tool(tool: Tool, messages: List[Dict], **kwargs) -> Union[str, Iterator[str]]:
        """Get a response using a tool"""
        stream_mode = kwargs.get('stream', False)
        if stream_mode:
            return get_streaming_response(messages, **kwargs)
        return get_response(messages, **kwargs)
    
    def get_version() -> str:
        """Get the version of the provider"""
        return f"{config.name} (Model: {config.model})"
    
    return {
        'get_response': get_response,
        'get_streaming_response': get_streaming_response,
        'get_response_from_tool': get_response_from_tool,
        'version': get_version
    }


def create_openai_provider(api_key: str, model: str = None) -> Dict:
    """Create an OpenAI provider"""
    if model is None:
        model = DEFAULT_MODELS["openai"]
        
    config = ProviderConfig(
        api_key=api_key,
        model=model,
        name="OpenAI Provider"
    )
    return create_provider(config)


def create_openrouter_provider(
    api_key: str,
    model: str = None,
    site_url: Optional[str] = None,
    site_name: Optional[str] = None
) -> Dict:
    """Create an OpenRouter provider"""
    if model is None:
        model = DEFAULT_MODELS["openrouter"]
        
    extra_headers = {}
    if site_url:
        extra_headers["HTTP-Referer"] = site_url
    if site_name:
        extra_headers["X-Title"] = site_name

    config = ProviderConfig(
        api_key=api_key,
        model=model,
        base_url=DEFAULT_BASE_URLS["openrouter"],
        extra_headers=extra_headers,
        name="OpenRouter Provider"
    )
    return create_provider(config)


class AI:
    """Main AI class that handles interactions with AI providers"""
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
        # Use default model if not specified
        if 'model' not in kwargs:
            kwargs['model'] = DEFAULT_MODELS.get(provider)
            
        if provider == "openai":
            self.provider = create_openai_provider(**kwargs)
        elif provider == "openrouter":
            self.provider = create_openrouter_provider(**kwargs)
        else:
            raise ValueError(f"Unknown provider: {provider}")
            
        # Store model info for reference
        self.model_name = kwargs.get('model', DEFAULT_MODELS.get(provider))
        self.model_info = get_model_info(self.model_name, provider)

    def get_response(self, messages: List[Dict], **kwargs) -> Union[str, Iterator[str]]:
        """Get a response from the AI"""
        stream_mode = kwargs.get('stream', False)
        if stream_mode:
            return self.get_streaming_response(messages, **kwargs)
        return self.provider['get_response'](messages, **kwargs)
    
    def get_streaming_response(self, messages: List[Dict], **kwargs) -> Iterator[str]:
        """Get a streaming response from the AI"""
        # Ensure stream is set to True
        kwargs['stream'] = True
        return self.provider['get_streaming_response'](messages, **kwargs)

    def get_response_from_tool(self, tool: Tool, messages: List[Dict], **kwargs) -> Union[str, Iterator[str]]:
        """Get a response using a tool"""
        return self.provider['get_response_from_tool'](tool, messages, **kwargs)

    @property
    def version(self) -> str:
        """Get the version of the AI"""
        return self.provider['version']()
        
    @property
    def model_description(self) -> str:
        """Get a description of the current model"""
        return self.model_info.get('description', "Unknown model") if self.model_info else "Unknown model"
        
    @property
    def max_tokens(self) -> Optional[int]:
        """Get the maximum tokens for the current model"""
        return self.model_info.get('max_tokens') if self.model_info else None