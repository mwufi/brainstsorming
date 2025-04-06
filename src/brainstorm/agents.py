

import uuid
from src.brainstorm.ai import AI
from src.brainstorm.tools import Tool

class Message:
    def __init__(self, role: str, content: list[dict]):
        """Initialize a message with a list of content items.
        Each content item is a dict with either:
        - {"type": "text", "text": str} for text content
        - {"type": "image_url", "image_url": {"url": str}} for image URLs
        """
        self.role = role
        self.content = content

    def get_text_content(self) -> list[str]:
        """Get all text content from the message"""
        return [item["text"] for item in self.content if item["type"] == "text"]

    def get_image_urls(self) -> list[str]:
        """Get all image URLs from the message"""
        return [item["image_url"]["url"] for item in self.content if item["type"] == "image_url"]

    def __str__(self) -> str:
        texts = self.get_text_content()
        urls = self.get_image_urls()
        return f"Message(role={self.role}, texts={texts}, image_urls={urls})"

    def __repr__(self) -> str:
        return self.__str__()

class Conversation:
    def __init__(self, id: str, messages: list[Message]):
        self.id = id
        self.messages = messages

    def add_message(self, message: Message):
        self.messages.append(message)

    def get_messages(self) -> list[Message]:
        return self.messages

    def format_for_ai(self) -> list[dict]:
        return [{"role": message.role, "content": message.content} for message in self.messages]

class Agent:
    def __init__(self, name: str, description: str, tools: list[Tool], ai: AI = None):
        self.name = name
        self.description = description
        self.tools = tools
        self.ai = ai

        # memories (id -> Conversation)
        self.conversations = {}

    def run(self, user_input: str, conversation_id: str = None, model: str = None) -> list[Message]:
        if not model:
            model = "meta-llama/llama-4-maverick:free"

        if not conversation_id:
            conversation_id = str(uuid.uuid4())
            self.conversations[conversation_id] = Conversation(conversation_id, [])

        conversation = self.conversations[conversation_id]
        conversation.add_message(Message(role="user", content=[{"type": "text", "text": user_input}]))

        messages_for_ai = conversation.format_for_ai()
        messages_for_ai.insert(0, {"role": "system", "content": self.system_prompt})
        response = self.ai.get_response(messages_for_ai, model=model)
        conversation.add_message(Message(role="assistant", content=[{"type": "text", "text": response}]))

        return response

    @property
    def system_prompt(self) -> str:
        return f"You are {self.name}. Your goal is to be {self.description}"

    def __str__(self) -> str:
        return self.system_prompt

    def __repr__(self) -> str:
        return self.__str__()

