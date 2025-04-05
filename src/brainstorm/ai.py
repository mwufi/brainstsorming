"""
Contains utilities to get intelligent responses to a set of messages
"""

from openai import OpenAI
from src.brainstorm.agents import Tool


class AI:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def get_response(self, messages: list[dict]) -> str:
        response = self.client.chat.completions.create(model="gpt-4o-mini", messages=messages)
        return response.choices[0].message.content

    def get_response_from_tool(self, tool: Tool, messages: list[dict]) -> str:
        response = self.client.chat.completions.create(model="gpt-4o-mini", messages=messages)
        return response.choices[0].message.content
