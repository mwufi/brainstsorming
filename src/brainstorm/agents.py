

from typing import Callable
from src.brainstorm.helper import helper_function

class Tool:
    def __init__(self, name: str, description: str, function: Callable):
        self.name = name
        self.description = description
        self.function = function


class Agent:
    def __init__(self, name: str, description: str, tools: list[Tool]):
        self.name = name
        self.description = description
        self.tools = tools

        helper_function()

    def run(self, input: str) -> str:
        pass

    def __str__(self) -> str:
        return f"{self.name}: {self.description}"

    def __repr__(self) -> str:
        return self.__str__()

