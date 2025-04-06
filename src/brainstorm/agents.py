import uuid
from typing import List, Dict, Optional, Iterator, Union, Callable
from src.brainstorm.ai import AI
from src.brainstorm.tools import Tool

class Message:
    def __init__(self, role: str, content: Union[List[Dict], str]):
        """Initialize a message with content.
        Content can be either:
        - A string (for simple text content)
        - A list of content items where each item is a dict with either:
          - {"type": "text", "text": str} for text content
          - {"type": "image_url", "image_url": {"url": str}} for image URLs
        """
        self.role = role
        
        # Handle string content by converting to proper format
        if isinstance(content, str):
            self.content = [{"type": "text", "text": content}]
        else:
            self.content = content

    def get_text_content(self) -> List[str]:
        """Get all text content from the message"""
        return [item["text"] for item in self.content if item["type"] == "text"]

    def get_image_urls(self) -> List[str]:
        """Get all image URLs from the message"""
        return [item["image_url"]["url"] for item in self.content if item["type"] == "image_url"]

    def __str__(self) -> str:
        texts = self.get_text_content()
        urls = self.get_image_urls()
        return f"Message(role={self.role}, texts={texts}, image_urls={urls})"

    def __repr__(self) -> str:
        return self.__str__()

class Conversation:
    def __init__(self, id: str, messages: List[Message] = None):
        self.id = id
        self.messages = messages or []

    def add_message(self, message: Message):
        """Add a message to the conversation"""
        self.messages.append(message)

    def get_messages(self) -> List[Message]:
        """Get all messages in the conversation"""
        return self.messages

    def format_for_ai(self) -> List[Dict]:
        """Format messages for AI API consumption"""
        formatted_messages = []
        
        for message in self.messages:
            # For text-only content with a single element, some providers expect a string
            # instead of a list of content items (especially older OpenAI versions)
            # However, for our current structure, we'll always use the content list format
            formatted_messages.append({
                "role": message.role,
                "content": message.content
            })
            
        return formatted_messages
        
    def clear(self):
        """Clear all messages from the conversation"""
        self.messages = []
        
    def get_last_user_message(self) -> Optional[Message]:
        """Get the last user message in the conversation"""
        for message in reversed(self.messages):
            if message.role == "user":
                return message
        return None
        
    def get_last_assistant_message(self) -> Optional[Message]:
        """Get the last assistant message in the conversation"""
        for message in reversed(self.messages):
            if message.role == "assistant":
                return message
        return None

class Agent:
    def __init__(
        self, 
        name: str, 
        description: str, 
        tools: List[Tool], 
        ai: AI = None,
        default_model: str = None
    ):
        self.name = name
        self.description = description
        self.tools = tools
        self.ai = ai
        self.default_model = default_model or "meta-llama/llama-4-maverick:free"

        # memories (id -> Conversation)
        self.conversations = {}
        
    def init_conversation(self) -> str:
        """Initialize a new conversation and return its ID"""
        conversation_id = str(uuid.uuid4())
        self.conversations[conversation_id] = Conversation(conversation_id, [])
        return conversation_id

    def run(
        self, 
        user_input: str, 
        conversation_id: str = None, 
        model: str = None,
        stream: bool = False,
        stream_handler: Optional[Callable[[str], None]] = None,
        **kwargs
    ) -> Union[str, Iterator[str]]:
        """
        Run the agent with user input
        
        Args:
            user_input: The user's input
            conversation_id: The conversation ID (will be created if not provided)
            model: The model to use (will use default if not provided)
            stream: Whether to stream the response
            stream_handler: A function to handle streamed chunks (if None, will return an iterator)
            **kwargs: Additional parameters to pass to the AI
            
        Returns:
            Either a complete response string, or an iterator of response chunks
        """
        if not model:
            model = self.default_model

        # Create or retrieve conversation
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
            self.conversations[conversation_id] = Conversation(conversation_id, [])

        conversation = self.conversations[conversation_id]
        
        # Add user message to conversation
        conversation.add_message(Message(
            role="user", 
            content=[{"type": "text", "text": user_input}]
        ))

        # Prepare messages for AI
        messages_for_ai = conversation.format_for_ai()
        
        # Add system message at the beginning
        system_msg = {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]}
        messages_for_ai.insert(0, system_msg)
        
        # Get response (streaming or complete)
        if stream:
            if stream_handler:
                # Stream with handler function
                response_chunks = self.ai.get_streaming_response(
                    messages_for_ai, 
                    model=model,
                    **kwargs
                )
                full_response = ""
                for chunk in response_chunks:
                    full_response += chunk
                    stream_handler(chunk)
                
                # Add response to conversation
                conversation.add_message(Message(
                    role="assistant", 
                    content=[{"type": "text", "text": full_response}]
                ))
                return full_response
            else:
                # For returning an iterator, we need to collect all chunks and add to history
                # Store the chunks while yielding them
                def collect_and_yield():
                    full_response = ""
                    response_chunks = self.ai.get_streaming_response(
                        messages_for_ai, 
                        model=model,
                        **kwargs
                    )
                    for chunk in response_chunks:
                        full_response += chunk
                        yield chunk
                        
                    # After all chunks are processed, add the full message to conversation
                    # This runs when the iterator is exhausted
                    conversation.add_message(Message(
                        role="assistant", 
                        content=[{"type": "text", "text": full_response}]
                    ))
                
                return collect_and_yield()
        else:
            # Get complete response
            response = self.ai.get_response(
                messages_for_ai, 
                model=model,
                **kwargs
            )
            
            # Add response to conversation
            conversation.add_message(Message(
                role="assistant", 
                content=[{"type": "text", "text": response}]
            ))
            return response

    @property
    def system_prompt(self) -> str:
        """Get the system prompt for the agent"""
        return f"You are {self.name}. Your goal is to be {self.description}"

    def __str__(self) -> str:
        return self.system_prompt

    def __repr__(self) -> str:
        return self.__str__()