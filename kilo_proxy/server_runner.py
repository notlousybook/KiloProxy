"""Server runner module for daemon mode."""

import logging
import sys
from pathlib import Path

import typer
import uvicorn
from rich.console import Console

console = Console()
app = typer.Typer()

LOG_DIR = Path.home() / ".kilo-proxy" / "logs"
LOG_FILE = LOG_DIR / "server.log"


def setup_logging():
    """Setup logging to file."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("kilo-proxy")


def get_uvicorn_config(host: str, port: int):
    """Get uvicorn config dict for reuse."""
    config = uvicorn.Config(
        "kilo_proxy.server:app",
        host=host,
        port=port,
        access_log=False,
        limit_concurrency=1000,
        limit_max_requests=None,
        timeout_keep_alive=300,
    )
    return config


@app.command()
def run(
    host: str = typer.Option("127.0.0.1", "--host", "-h"),
    port: int = typer.Option(5380, "--port", "-p"),
):
    """Run the server."""
    from kilo_proxy import __version__

    logger = setup_logging()
    logger.info(f"Starting Kilo Proxy v{__version__} on {host}:{port}")
    console.print(f"[green]Starting Kilo Proxy v{__version__} on {host}:{port}[/green]")

    try:
        import uvloop

        config = uvicorn.Config(
            "kilo_proxy.server:app",
            host=host,
            port=port,
            access_log=False,
            limit_concurrency=1000,
            limit_max_requests=None,
            timeout_keep_alive=300,
            loop="uvloop",
        )
    except ImportError:
        config = uvicorn.Config(
            "kilo_proxy.server:app",
            host=host,
            port=port,
            access_log=False,
            limit_concurrency=1000,
            limit_max_requests=None,
            timeout_keep_alive=300,
        )

    server = uvicorn.Server(config)
    server.run()


if __name__ == "__main__":
    app()
