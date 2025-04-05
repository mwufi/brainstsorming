"""
Example script demonstrating how to use the model definitions
"""

import os
from dotenv import load_dotenv
from src.brainstorm.models import (
    get_model_info, 
    get_models_by_provider, 
    get_models_by_category,
    ModelCategory,
    load_models
)
from src.brainstorm.ai import AI

# Load environment variables
load_dotenv()

def print_model_info(model_name: str):
    """Print information about a specific model"""
    model_info = get_model_info(model_name)
    if model_info:
        print(f"Model: {model_info.name}")
        print(f"Provider: {model_info.provider}")
        print(f"Category: {model_info.category.name}")
        print(f"Description: {model_info.description}")
        print(f"Max Tokens: {model_info.max_tokens}")
        print(f"Experimental: {model_info.is_experimental}")
        print()
    else:
        print(f"Model {model_name} not found")
        print()

def main():
    # Print all available models
    print("=== All Available Models ===")
    models = load_models()
    for provider, provider_models in models.items():
        print(f"\n{provider.upper()} MODELS:")
        for model_name in provider_models:
            print_model_info(model_name)
    
    # Create an AI instance with Reka model
    print("=== Using AI with Reka Flash 3 ===")
    ai = AI(
        provider="openrouter",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model="openrouter/quasar-alpha"
    )
    
    print(f"Using model: {ai.model_info['name']}")
    print(f"Description: {ai.model_info['description']}")
    print(f"Max tokens: {ai.model_info['max_tokens']}")
    
    # Example response
    try:
        response = ai.get_response([
            {"role": "user", "content": "Tell me a short joke about programming"}
        ])
        print(f"\nResponse: {response}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()