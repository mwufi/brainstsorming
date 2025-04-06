
# How to run

```
uv sync
uv run -m examples.main <agentName> (right now we only have 'catgirl' and 'adventuremaster')
```

## Adding more agents

You can add more system prompts in the `/agents` directory

## Changing the models

You can add any model on OpenRouter by adding it in `models.json` and then adding an agent config to use that.