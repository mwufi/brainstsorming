"""
Contains model definitions for various AI providers
"""

import json
import os
from enum import Enum, auto
from typing import Dict, Optional


class ModelCategory(Enum):
    """Categories of models based on their capabilities"""
    GENERAL = auto()
    CODE = auto()
    VISION = auto()
    LONG_CONTEXT = auto()
    FAST = auto()
    EXPERIMENTAL = auto()


class ModelInfo:
    """Information about a specific model"""
    def __init__(
        self,
        name: str,
        provider: str,
        category: ModelCategory,
        description: str,
        max_tokens: Optional[int] = None,
        is_experimental: bool = False
    ):
        self.name = name
        self.provider = provider
        self.category = ModelCategory[category.upper()] if isinstance(category, str) else category
        self.description = description
        self.max_tokens = max_tokens
        self.is_experimental = is_experimental

    def __str__(self) -> str:
        return f"{self.name} ({self.provider})"

    def __repr__(self) -> str:
        return self.__str__()


def load_models() -> Dict[str, Dict[str, ModelInfo]]:
    """Load models from models.json file"""
    models_path = os.path.join(os.path.dirname(__file__), "models.json")
    with open(models_path, "r") as f:
        data = json.load(f)
        
    models = {}
    for provider, provider_models in data.items():
        models[provider] = {}
        for model_name, model_data in provider_models.items():
            models[provider][model_name] = ModelInfo(
                name=model_name,
                provider=provider,
                category=model_data["category"],
                description=model_data["description"],
                max_tokens=model_data.get("max_tokens"),
                is_experimental=model_data.get("is_experimental", False)
            )
    return models


# Load models from JSON
ALL_MODELS = {}
PROVIDER_MODELS = load_models()
for provider_models in PROVIDER_MODELS.values():
    ALL_MODELS.update(provider_models)


def get_model_info(model_name: str) -> Optional[ModelInfo]:
    """Get information about a specific model"""
    return ALL_MODELS.get(model_name)


def get_models_by_provider(provider: str) -> Dict[str, ModelInfo]:
    """Get all models from a specific provider"""
    return PROVIDER_MODELS.get(provider, {})


def get_models_by_category(category: ModelCategory) -> Dict[str, ModelInfo]:
    """Get all models in a specific category"""
    return {name: model for name, model in ALL_MODELS.items() if model.category == category}