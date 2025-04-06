
![Screenshot 1](docs/screenshot1.png)

# Brainstorm - AI Chat Framework

A flexible chat framework for building AI agents with support for OpenAI and OpenRouter providers.

## Features

- Support for both OpenAI and OpenRouter API providers
- Streaming responses for real-time text generation
- Easily extensible agent system
- Configurable models through JSON configuration
- Conversation memory management
- Rich text formatting for chat interface

## Installation

```sh
# Install dependencies using uv
uv sync
```

## How to Run

```sh
# Run with default agent (catgirl)
uv run -m examples.main

# Run with specific agent
uv run -m examples.main catgirl
uv run -m examples.main adventuremaster

# Run with streaming mode (real-time responses)
uv run -m examples.main catgirl --stream
```

## Examples

### Basic Streaming Example

You can run a simple streaming example to test the framework:

```sh
uv run -m examples.streaming_example
```

## Adding More Agents

You can add more system prompts in the `/agents` directory. Create a JSON file with the following structure:

```json
{
  "name": "YourAgent",
  "description": "A detailed description of your agent's personality and goals",
  "ai_config": {
    "provider": "openrouter",
    "model": "openai/gpt-4o"
  }
}
```

## Changing the Models

You can add any model on OpenRouter or OpenAI by adding it in `models.json` and then adding an agent config to use that model.

## API Usage

```python
from src.brainstorm.agents import Agent
from src.brainstorm.ai import AI

# Initialize the AI
ai = AI(
    provider="openrouter",
    api_key="your-api-key",
    model="openai/gpt-4o"
)

# Create an agent
agent = Agent(
    name="Assistant",
    description="a helpful assistant",
    tools=[],
    ai=ai
)

# Get a complete response
response = agent.run("Tell me about Python programming")
print(response)

# Or use streaming for real-time responses
def handle_stream(chunk):
    print(chunk, end="", flush=True)

agent.run(
    "Tell me about Python programming",
    stream=True,
    stream_handler=handle_stream
)
```