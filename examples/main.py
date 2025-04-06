from src.brainstorm.agents import Agent, Tool
from src.brainstorm.ai import AI
import os
import re
from dotenv import load_dotenv
from rich import print
from rich.panel import Panel
from rich.console import Console
from rich.prompt import Prompt
from rich.text import Text
from rich.style import Style

load_dotenv()

HEADER = """
[bold magenta]🐱 Catgirl AI Chat[/bold magenta]
[dim]A friendly AI companion powered by OpenRouter[/dim]
"""

def format_response(text):
    """Format the response text to display italicized text in pink."""
    # Create a Rich Text object
    rich_text = Text()
    
    # Split the text by markdown-style italics
    parts = re.split(r'(\*[^*]+\*|_[^_]+_)', text)
    
    for part in parts:
        if part.startswith('*') and part.endswith('*'):
            # Remove the asterisks and style the content
            content = part[1:-1]
            rich_text.append(content, style=Style(italic=True, color="magenta"))
        elif part.startswith('_') and part.endswith('_'):
            # Remove the underscores and style the content
            content = part[1:-1]
            rich_text.append(content, style=Style(italic=True, color="magenta"))
        else:
            # Regular text without styling
            rich_text.append(part)
    
    return rich_text

def main():
    console = Console()
    console.print(Panel(HEADER, border_style="magenta"))

    ai = AI(provider="openrouter", api_key=os.getenv("OPENROUTER_API_KEY"))

    agent = Agent(
        name="Catgirl",
        description="A friendly and playful catgirl companion who loves to chat",
        tools=[],
        ai=ai
    )

    console.print("[bold]System:[/bold]", agent, style="cyan")
    console.print()

    while True:
        try:
            user_input = Prompt.ask("[bold cyan]You[/bold cyan]")
            if user_input.lower() in ["exit", "quit"]:
                break

            response = agent.run(user_input)
            formatted_response = format_response(response)
            console.print("[bold magenta]Catgirl[/bold magenta]:", formatted_response)
            console.print()

        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")

    console.print("\n[dim]Thanks for chatting! Nya~ 👋[/dim]")

if __name__ == "__main__":
    main()
