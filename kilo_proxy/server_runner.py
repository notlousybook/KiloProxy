"""Server runner module for daemon mode."""

import sys
import typer
import uvicorn
from rich.console import Console

console = Console()
app = typer.Typer()


@app.command()
def run(
    host: str = typer.Option("127.0.0.1", "--host", "-h"),
    port: int = typer.Option(5380, "--port", "-p"),
):
    """Run the server."""
    from kilo_proxy import __version__

    console.print(f"[green]Starting Kilo Proxy v{__version__} on {host}:{port}[/green]")

    uvicorn.run(
        "kilo_proxy.server:app",
        host=host,
        port=port,
        access_log=True,
    )


if __name__ == "__main__":
    app()
