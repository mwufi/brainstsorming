from src.brainstorm.agents import Agent, Tool
from src.brainstorm.ai import AI
import os
import re
import sys
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv
from rich import print
from rich.panel import Panel
from rich.console import Console
from rich.prompt import Prompt
from rich.text import Text
from rich.style import Style
from rich.live import Live

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

def format_stream_chunk(chunk, buffer=""):
    """Format a single streaming chunk to handle italics."""
    # Combine the buffer with the current chunk
    full_text = buffer + chunk
    
    # Process the complete text with proper formatting
    return format_response(full_text), full_text

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Chat with an AI agent")
    parser.add_argument("agent", nargs="?", default="catgirl", help="Agent name to use")
    parser.add_argument("--stream", action="store_true", help="Use streaming mode")
    return parser.parse_args()

def main():
    args = parse_args()
    console = Console()
    
    try:
        # Load agent configuration
        config = load_agent_config(args.agent)
        
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
            ai=ai,
            default_model=config["ai_config"]["model"]
        )

        # Create a single conversation ID to use for the entire session
        conversation_id = agent.init_conversation()

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

                if args.stream:
                    # Print the agent name on its own line first
                    console.print(f"[bold magenta]{config['name']}[/bold magenta]:")
                    
                    # Initialize response buffer
                    buffer = ""
                    
                    # Create a Live display for formatted output
                    with Live("", refresh_per_second=10, console=console) as live_display:
                        # Stream handler function
                        def handle_stream(chunk):
                            nonlocal buffer
                            # Update buffer with new chunk
                            buffer += chunk
                            # Format the entire accumulated text with proper handling of formatting
                            formatted_text, _ = format_stream_chunk("", buffer)
                            # Update the live display with formatted text
                            live_display.update(formatted_text)
                        
                        # Run agent with streaming
                        agent.run(
                            user_input,
                            conversation_id=conversation_id,
                            stream=True,
                            stream_handler=handle_stream
                        )
                    
                    # Print a newline after streaming is done
                    console.print()
                else:
                    # Traditional non-streaming response
                    response = agent.run(user_input, conversation_id=conversation_id)
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
