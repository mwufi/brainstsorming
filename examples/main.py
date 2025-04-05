from src.brainstorm.agents import Agent, Tool
from src.brainstorm.ai import AI
import os
from dotenv import load_dotenv

load_dotenv()

def test_openrouter():
    from openai import OpenAI

    print(os.getenv("OPENROUTER_API_KEY"))
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    completion = client.chat.completions.create(
        extra_headers={
            "HTTP-Referer": "https://www.google.com", # Optional. Site URL for rankings on openrouter.ai.
            "X-Title": "Google", # Optional. Site title for rankings on openrouter.ai.
        },
        extra_body={},
        model="openai/gpt-4o-mini-search-preview",
        messages=[
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": "What is in this image?"
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                }
                }
            ]
            }
        ]
        )
    print(completion)

def main():
    agent = Agent(
        name="Brainstorm Agent",
        description="A tool that can help you brainstorm ideas",
        tools=[],
    )
    print(agent)
    print("Hello from brainstorm!")

    # ai = AI(api_key=os.getenv("OPENAI_API_KEY"))
    ai = AI(provider="openrouter", api_key=os.getenv("OPENROUTER_API_KEY"))
    print(ai.get_response([{"role": "user", "content": "Tell me a joke!"}], model="openai/gpt-4o-mini"))

    # test_openrouter()

if __name__ == "__main__":
    main()
