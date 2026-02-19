#!/usr/bin/env python3
"""
AI Chatbot – CLI Interface
==========================
Commands:
  ingest <file>        Ingest a document into the knowledge base
  chat                 Interactive streaming chat session
  weather              Run weather analysis wizard
  sources              List indexed documents
  inspect <filename>   Show all chunks for a document
  delete <filename>    Remove a document from the index
  reindex <file>       Delete + re-ingest a document
  stats                Knowledge base statistics
  memory               Show USER_MEMORY.md and COMPANY_MEMORY.md
  history              View and export conversation history
  eval                 Run the automated evaluation harness
  sanity               Run end-to-end sanity check
"""
import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()


def get_bot(user_id: str = "default"):
    from app.chatbot import Chatbot
    return Chatbot(user_id=user_id)


# ── CLI group ─────────────────────────────────────────────────────────────

@click.group()
def cli():
    """🤖 AI Chatbot — RAG + Memory + Weather + File Management"""
    pass


# ── ingest ────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("file_path")
def ingest(file_path):
    """Ingest a document into the knowledge base."""
    bot = get_bot()
    console.print(f"[bold cyan]Ingesting:[/bold cyan] {file_path}")

    with console.status("Parsing, chunking, and indexing..."):
        result = bot.ingest(file_path)

    table = Table(show_header=False, box=box.SIMPLE)
    table.add_row("File",             result["file"])
    table.add_row("Total chars",      f"{result['total_chars']:,}")
    table.add_row("Total chunks",     str(result["total_chunks"]))
    table.add_row("New chunks added", str(result["new_chunks"]))
    console.print(table)
    console.print("[bold green]✓ Ingestion complete[/bold green]")


# ── sources ───────────────────────────────────────────────────────────────

@cli.command()
def sources():
    """List all indexed documents with chunk counts."""
    from app.rag.file_manager import list_sources
    srcs = list_sources()

    if not srcs:
        console.print("[yellow]No documents indexed. Use: python cli.py ingest <file>[/yellow]")
        return

    table = Table(title="Knowledge Base", box=box.ROUNDED)
    table.add_column("File",        style="cyan")
    table.add_column("Chunks",      justify="right")
    table.add_column("Total chars", justify="right")

    for s in srcs:
        table.add_row(s["source"], str(s["chunks"]), f"{s['total_chars']:,}")

    console.print(table)


# ── inspect ───────────────────────────────────────────────────────────────

@cli.command()
@click.argument("filename")
@click.option("--full", is_flag=True, help="Show full chunk text (default: truncated)")
def inspect(filename, full):
    """Show all indexed chunks for a document."""
    from app.rag.file_manager import inspect_source

    chunks = inspect_source(filename)
    if not chunks:
        console.print(f"[red]No chunks found for '{filename}'. Is it indexed?[/red]")
        return

    console.print(f"\n[bold]{filename}[/bold] — {len(chunks)} chunks\n")
    for c in chunks:
        text = c["text"] if full else (c["text"][:120] + "..." if len(c["text"]) > 120 else c["text"])
        console.print(Panel(
            text,
            title=f"[dim]chunk {c['chunk_index'] + 1}  |  {c['char_count']} chars[/dim]",
            border_style="dim",
        ))


# ── delete ────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("filename")
@click.option("--yes", is_flag=True, help="Skip confirmation prompt")
def delete(filename, yes):
    """Remove a document from the knowledge base."""
    from app.rag.file_manager import delete_source

    if not yes:
        click.confirm(f"Delete all chunks for '{filename}'?", abort=True)

    with console.status(f"Deleting '{filename}'..."):
        result = delete_source(filename)

    if "error" in result:
        console.print(f"[red]✗ {result['error']}[/red]")
    else:
        console.print(f"[green]✓ Deleted {result['deleted']} chunks for '{filename}'[/green]")


# ── reindex ───────────────────────────────────────────────────────────────

@cli.command()
@click.argument("file_path")
def reindex(file_path):
    """Delete and re-ingest a document (use after editing a file)."""
    from app.rag.file_manager import reindex_source

    console.print(f"[bold cyan]Re-indexing:[/bold cyan] {file_path}")
    with console.status("Deleting old chunks and re-ingesting..."):
        result = reindex_source(file_path)

    table = Table(show_header=False, box=box.SIMPLE)
    table.add_row("File",         result["source"])
    table.add_row("Deleted",      str(result["deleted"]))
    table.add_row("New chunks",   str(result["new_chunks"]))
    table.add_row("Total chars",  f"{result['total_chars']:,}")
    console.print(table)
    console.print("[green]✓ Re-index complete[/green]")


# ── stats ─────────────────────────────────────────────────────────────────

@cli.command()
def stats():
    """Show knowledge base statistics."""
    from app.rag.file_manager import chunk_stats

    s = chunk_stats()
    table = Table(title="Knowledge Base Stats", box=box.ROUNDED, show_header=False)
    table.add_row("Total documents", str(s["total_sources"]))
    table.add_row("Total chunks",    str(s["total_chunks"]))
    table.add_row("Total chars",     f"{s['total_chars']:,}")
    table.add_row("Avg chunk size",  f"{s['avg_chunk_chars']} chars")
    console.print(table)

    if s["sources"]:
        console.print("\n[dim]Sources: " + ", ".join(s["sources"]) + "[/dim]")


# ── chat (streaming) ──────────────────────────────────────────────────────

@cli.command()
@click.option("--user-id", default="default", help="User namespace for multi-user mode")
@click.option("--no-stream", is_flag=True, help="Disable streaming (use buffered output)")
def chat(user_id, no_stream):
    """Start an interactive streaming chat session."""
    bot = get_bot(user_id)

    mode = "buffered" if no_stream else "streaming"
    console.print(Panel(
        f"[bold cyan]AI Chatbot[/bold cyan] — RAG + Memory + Weather\n"
        f"User: [bold]{user_id}[/bold]  |  Mode: {mode}\n"
        "Commands: [bold]exit[/bold]  [bold]clear[/bold]  [bold]history[/bold]  [bold]weather[/bold]",
        title="Welcome",
    ))

    srcs = bot.list_sources()
    if srcs:
        console.print(f"[dim]Documents: {', '.join(srcs)}[/dim]\n")
    else:
        console.print("[yellow]⚠ No documents indexed.[/yellow]\n")

    while True:
        try:
            user_input = console.input("[bold green]You:[/bold green] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input:
            continue
        if user_input.lower() == "exit":
            break
        if user_input.lower() == "clear":
            bot.clear_history()
            console.print("[dim]History cleared.[/dim]")
            continue
        if user_input.lower() == "history":
            _show_history(bot)
            continue
        if user_input.lower() == "weather":
            _weather_wizard(bot)
            continue

        console.print()
        console.print("[bold blue]Assistant:[/bold blue] ", end="")

        if no_stream:
            with console.status("[dim]Thinking...[/dim]"):
                result = bot.chat(user_input)
            console.print(Markdown(result["answer"]))
            _print_turn_footer(result)
        else:
            # Streaming output
            final = None
            for event in bot.stream_chat(user_input):
                if event["type"] == "token":
                    print(event["content"], end="", flush=True)
                elif event["type"] == "done":
                    final = event
            print()  # newline after stream
            if final:
                _print_turn_footer(final)

        console.print()


def _print_turn_footer(result: dict):
    if result.get("citations"):
        console.print("[dim]📎 " + " | ".join(result["citations"]) + "[/dim]")
    if result.get("memory_written"):
        console.print(
            f"[dim]🧠 Memory → {result['memory_target']}: {result['memory_summary']}[/dim]"
        )
    if result.get("injection_detected"):
        console.print("[yellow]⚠ Prompt injection detected in document content[/yellow]")
    if result.get("hallucinated_citations"):
        console.print(f"[yellow]⚠ Removed hallucinated citations: {result['hallucinated_citations']}[/yellow]")


def _weather_wizard(bot):
    console.print("\n[bold yellow]🌤 Weather Analysis[/bold yellow]")
    location   = console.input("Location: ").strip()
    start_date = console.input("Start date (YYYY-MM-DD): ").strip()
    end_date   = console.input("End date   (YYYY-MM-DD): ").strip()

    with console.status("Analyzing..."):
        result = bot.analyze_weather(location, start_date, end_date)

    if result["success"]:
        console.print(Panel(result["output"], title=f"Analysis: {result['location']}"))
        console.print(Panel(Markdown(result["summary"]), title="Summary", border_style="blue"))
    else:
        console.print(f"[red]Error: {result.get('error', 'unknown')}[/red]")
    console.print()


# ── history ───────────────────────────────────────────────────────────────

@cli.command()
@click.option("--export", type=click.Path(), default=None, help="Save history to file")
@click.pass_context
def history(ctx, export):
    """View or export the current conversation history."""
    # Note: history is session-scoped; this shows an empty session
    # To use in an active chat, type 'history' inside the chat command
    console.print("[yellow]Tip: Type 'history' inside 'python cli.py chat' to see the live session history.[/yellow]")
    if export:
        console.print(f"[dim]Export path would be: {export}[/dim]")


def _show_history(bot):
    """Display history from within an active chat session."""
    turns = bot.get_history()
    if not turns:
        console.print("[dim]No history yet.[/dim]")
        return

    console.print(f"\n[bold]Conversation History ({len(turns) // 2} turns)[/bold]")
    for turn in turns:
        role_label = "[green]You[/green]" if turn["role"] == "user" else "[blue]Assistant[/blue]"
        snippet = turn["content"][:200] + ("..." if len(turn["content"]) > 200 else "")
        console.print(f"  {role_label}: {snippet}")
    console.print()

    if click.confirm("Export full history to markdown?", default=False):
        path = click.prompt("Save to", default="conversation_history.md")
        bot.export_history(path)
        console.print(f"[green]✓ Saved to {path}[/green]")


# ── memory ────────────────────────────────────────────────────────────────

@cli.command()
def memory():
    """Display current USER_MEMORY.md and COMPANY_MEMORY.md."""
    from app.config import USER_MEMORY_FILE, COMPANY_MEMORY_FILE
    for label, path in [("User Memory", USER_MEMORY_FILE), ("Company Memory", COMPANY_MEMORY_FILE)]:
        if path.exists():
            console.print(Panel(Markdown(path.read_text()), title=f"[bold]{label}[/bold]"))
        else:
            console.print(f"[yellow]{label}: not yet created.[/yellow]")


# ── weather (standalone) ──────────────────────────────────────────────────

@cli.command()
@click.option("--location",   default="London, UK")
@click.option("--start-date", default="2024-01-01")
@click.option("--end-date",   default="2024-01-31")
def weather(location, start_date, end_date):
    """Run a weather time-series analysis."""
    bot = get_bot()
    with console.status("Analyzing..."):
        result = bot.analyze_weather(location, start_date, end_date)

    if result["success"]:
        console.print(Panel(result["output"], title=f"Analysis: {result['location']}"))
        console.print(Panel(Markdown(result["summary"]), title="Summary", border_style="blue"))
    else:
        console.print(f"[red]✗ {result.get('error', 'failed')}[/red]")


# ── eval ──────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--report", is_flag=True, help="Save JSON report to artifacts/eval_report.json")
def eval(report):
    """Run the automated evaluation harness."""
    from scripts.eval_harness import run_eval
    result = run_eval(save_report=report)
    sys.exit(0 if result["failed"] == 0 else 1)


# ── sanity ────────────────────────────────────────────────────────────────

@cli.command()
def sanity():
    """Run end-to-end sanity check → artifacts/sanity_output.json."""
    from scripts.run_sanity import run_sanity_check
    run_sanity_check()


if __name__ == "__main__":
    cli()
