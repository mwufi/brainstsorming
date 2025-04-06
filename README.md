
![Screenshot 1](docs/screenshot1.png)

# How to run

```sh
# Install deps (openai)
uv sync

# Run agents
uv run -m examples.main catgirl
uv run -m examples.main adventuremaster
```

## Adding more agents

You can add more system prompts in the `/agents` directory

## Changing the models

You can add any model on OpenRouter by adding it in `models.json` and then adding an agent config to use that.