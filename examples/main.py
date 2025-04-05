from src.brainstorm.agents import Agent, Tool
from src.brainstorm.ai import AI
import os
from dotenv import load_dotenv

load_dotenv()

def main():
    agent = Agent(
        name="Brainstorm Agent",
        description="A tool that can help you brainstorm ideas",
        tools=[],
    )
    print(agent)
    print("Hello from brainstorm!")

    ai = AI(provider="openrouter", api_key=os.getenv("OPENROUTER_API_KEY"))
    print(ai.get_response([{"role": "user", "content": "Tell me a joke!"}], model="openai/gpt-4o-mini"))

if __name__ == "__main__":
    main()
