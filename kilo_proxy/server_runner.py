"""Server runner module for daemon mode."""

import logging
import os
import sys
from pathlib import Path

import typer
import uvicorn
from rich.console import Console

console = Console()
app = typer.Typer()

LOG_DIR = Path.home() / ".kilo-proxy" / "logs"
LOG_FILE = LOG_DIR / "server.log"
PID_FILE = Path.home() / ".kilo-proxy" / "server.pid"


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


def write_pid():
    """Write PID file for the server process."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))


def remove_pid():
    """Remove PID file."""
    if PID_FILE.exists():
        PID_FILE.unlink()


@app.command()
def run(
    host: str = typer.Option("127.0.0.1", "--host", "-h"),
    port: int = typer.Option(5380, "--port", "-p"),
):
    """Run the server."""
    from kilo_proxy import __version__

    logger = setup_logging()
    
    # Write PID file so the server can be tracked
    write_pid()
    logger.info(f"Starting Kilo Proxy v{__version__} on {host}:{port} (PID: {os.getpid()})")
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
    try:
        server.run()
    finally:
        # Clean up PID file on exit
        remove_pid()


if __name__ == "__main__":
    app()
