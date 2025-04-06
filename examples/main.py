from src.brainstorm.agents import Agent, Tool
from src.brainstorm.ai import AI
import os
import re
import sys
import json
from pathlib import Path
from dotenv import load_dotenv
from rich import print
from rich.panel import Panel
from rich.console import Console
from rich.prompt import Prompt
from rich.text import Text
from rich.style import Style

load_dotenv()

HEADER = """
[bold magenta]🤖 AI Chat[/bold magenta]
[dim]A friendly AI companion powered by OpenRouter[/dim]
"""

def load_agent_config(agent_name: str) -> dict:
    """Load agent configuration from JSON file."""
    config_path = Path(f"agents/{agent_name}.json")
    if not config_path.exists():
        raise FileNotFoundError(f"Agent config not found: {config_path}")
    
    with open(config_path) as f:
        return json.load(f)

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
    
    # Get agent name from command line or use default
    agent_name = sys.argv[1] if len(sys.argv) > 1 else "catgirl"
    
    try:
        # Load agent configuration
        config = load_agent_config(agent_name)
        
        # Initialize AI with config
        ai = AI(
            provider=config["ai_config"]["provider"], 
            api_key=os.getenv("OPENROUTER_API_KEY"),
            model=config["ai_config"]["model"]
        )

        agent = Agent(
            name=config["name"],
            description=config["description"],
            tools=[],
            ai=ai
        )

        # Update header with agent name
        header_text = f"\n[bold magenta]🤖 {config['name']} AI Chat[/bold magenta]\n"
        header_text += "[dim]A friendly AI companion powered by OpenRouter[/dim]\n"
        console.print(Panel(header_text, border_style="magenta"))

        console.print("[bold]System:[/bold]", agent, style="cyan")
        console.print()

        while True:
            try:
                user_input = Prompt.ask("[bold cyan]You[/bold cyan]")
                if user_input.lower() in ["exit", "quit"]:
                    break

                response = agent.run(user_input)
                formatted_response = format_response(response)
                console.print(f"[bold magenta]{config['name']}[/bold magenta]:", formatted_response)
                console.print()

            except KeyboardInterrupt:
                break
            except Exception as e:
                console.print(f"[bold red]Error:[/bold red] {str(e)}")

        console.print("\n[dim]Thanks for chatting! 👋[/dim]")
    
    except FileNotFoundError as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        console.print("[yellow]Available agents:[/yellow]")
        for file in Path("agents").glob("*.json"):
            console.print(f"  - {file.stem}")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")

if __name__ == "__main__":
    main()
