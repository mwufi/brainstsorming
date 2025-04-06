import json
import requests
from openai import OpenAI

import os
from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file
load_dotenv()


# Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = "google/gemini-2.0-flash-001"
openai_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# Initial message setup
task = "What are the titles of some James Joyce books?"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": task}
]

# Tool definition
def search_gutenberg_books(search_terms):
    search_query = " ".join(search_terms)
    url = "https://gutendex.com/books"
    response = requests.get(url, params={"search": search_query})
    simplified_results = []
    for book in response.json().get("results", []):
        simplified_results.append({
            "id": book.get("id"),
            "title": book.get("title"),
            "authors": book.get("authors")
        })
    logger.info(f"Found {len(simplified_results)} books", simplified_results)
    return simplified_results

# Tool specification
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_gutenberg_books",
            "description": "Search for books in the Project Gutenberg library based on specified search terms",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_terms": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of search terms to find books in the Gutenberg library"
                    }
                },
                "required": ["search_terms"]
            }
        }
    }
]

# Tool mapping
TOOL_MAPPING = {"search_gutenberg_books": search_gutenberg_books}

# Agentic loop functions
def call_llm(msgs):
    resp = openai_client.chat.completions.create(
        model=MODEL,
        tools=tools,
        messages=msgs
    )
    # Convert response message to dict and append
    message_dict = {
        "role": resp.choices[0].message.role,
        "content": resp.choices[0].message.content
    }
    if resp.choices[0].message.tool_calls:
        message_dict["tool_calls"] = [
            {
                "id": tc.id,
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments
                },
                "type": tc.type
            } for tc in resp.choices[0].message.tool_calls
        ]
    msgs.append(message_dict)
    return resp

def get_tool_response(response):
    tool_call = response.choices[0].message.tool_calls[0]
    tool_name = tool_call.function.name
    tool_args = json.loads(tool_call.function.arguments)
    tool_result = TOOL_MAPPING[tool_name](**tool_args)
    return {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "name": tool_name,
        "content": json.dumps(tool_result),
    }

# Main execution
def main():
    global messages
    while True:
        resp = call_llm(messages)
        if resp.choices[0].message.tool_calls:
            messages.append(get_tool_response(resp))
        else:
            break
    
    # Print the final response
    print("Here are some books by James Joyce:")
    final_content = messages[-1]["content"]
    if isinstance(final_content, str):
        print(final_content)
    else:
        print("Unexpected response format")

if __name__ == "__main__":
    main()