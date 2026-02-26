"""CLI for kilo-proxy."""

import asyncio
import copy
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import List, Optional

import httpx
import typer
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

from kilo_proxy import __version__
from kilo_proxy.config import (
    Config,
    generate_kilo_session_id,
    get_config_path,
    get_pid_path,
    load_config,
    regenerate_session_id,
    save_config,
)
from kilo_proxy.ip_shuffler import get_shuffler

LOG_DIR = Path.home() / ".kilo-proxy" / "logs"
LOG_FILE = LOG_DIR / "server.log"

app = typer.Typer(name="kilo-proxy", help="Fully OpenAI-compatible API proxy for Kilo")
console = Console()


def is_free_model(model_id: str, model_name: str = "") -> bool:
    """Check if model is free.

    Free models are:
    1. Models with :free in ID
    2. Models without / in ID (stealth free models)
    3. Models with (free) in name
    """
    return ":free" in model_id or "/" not in model_id or "(free)" in model_name.lower()


def is_server_running(pid: int) -> bool:
    try:
        if sys.platform == "win32":
            import ctypes

            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(0x1000, False, pid)
            if handle:
                kernel32.CloseHandle(handle)
                return True
            return False
        else:
            os.kill(pid, 0)
            return True
    except (OSError, ProcessLookupError):
        return False


def get_server_pid() -> Optional[int]:
    pid_path = get_pid_path()
    if pid_path.exists():
        try:
            with open(pid_path, "r") as f:
                return int(f.read().strip())
        except (ValueError, IOError):
            return None
    return None


def write_pid(pid: int) -> None:
    pid_path = get_pid_path()
    with open(pid_path, "w") as f:
        f.write(str(pid))


def remove_pid() -> None:
    pid_path = get_pid_path()
    if pid_path.exists():
        pid_path.unlink()


@app.command()
def auth(
    token: Optional[str] = typer.Argument(
        None, help="Auth token (omit to use 'anonymous')"
    ),
):
    """Set or remove authentication token. Default is 'anonymous'."""
    config = load_config()
    if token is None or token == "":
        config.auth_token = "anonymous"
        console.print("[green]Auth token reset to 'anonymous'[/green]")
    else:
        config.auth_token = token
        console.print(f"[green]Auth token set to: {token[:8]}...[/green]")
    save_config(config)


@app.command()
def start(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(5380, "--port", "-p", help="Port to bind to"),
    daemon: bool = typer.Option(True, "--daemon/--no-daemon", help="Run as daemon"),
):
    """Start the proxy server in the background."""
    existing_pid = get_server_pid()
    if existing_pid and is_server_running(existing_pid):
        console.print(
            f"[yellow]Server already running with PID {existing_pid}[/yellow]"
        )
        return

    config = load_config()
    config.host = host
    config.port = port
    save_config(config)

    if sys.platform == "win32":
        if sys.executable.endswith("python.exe"):
            python_exe = sys.executable
        else:
            python_exe = sys.executable

        creationflags = (
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        )
        process = subprocess.Popen(
            [
                python_exe,
                "-m",
                "kilo_proxy.server_runner",
                "--host",
                host,
                "--port",
                str(port),
            ],
            creationflags=creationflags,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        write_pid(process.pid)
        console.print(
            f"[green]Server started on {host}:{port} with PID {process.pid}[/green]"
        )
    else:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "kilo_proxy.server_runner",
                "--host",
                host,
                "--port",
                str(port),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        write_pid(process.pid)
        console.print(
            f"[green]Server started on {host}:{port} with PID {process.pid}[/green]"
        )


@app.command()
def stop():
    """Stop the background proxy server."""
    pid = get_server_pid()
    if not pid:
        console.print("[yellow]No server PID found[/yellow]")
        return

    if not is_server_running(pid):
        remove_pid()
        console.print(
            "[yellow]Server process not running, cleaned up PID file[/yellow]"
        )
        return

    try:
        if sys.platform == "win32":
            import ctypes

            kernel32 = ctypes.windll.kernel32
            PROCESS_TERMINATE = 0x0001
            handle = kernel32.OpenProcess(PROCESS_TERMINATE, False, pid)
            if handle:
                kernel32.TerminateProcess(handle, 0)
                kernel32.CloseHandle(handle)
        else:
            os.killpg(os.getpgid(pid), signal.SIGTERM)

        time.sleep(1)

        if is_server_running(pid):
            if sys.platform == "win32":
                import ctypes

                kernel32 = ctypes.windll.kernel32
                PROCESS_TERMINATE = 0x0001
                handle = kernel32.OpenProcess(PROCESS_TERMINATE, False, pid)
                if handle:
                    kernel32.TerminateProcess(handle, 1)
                    kernel32.CloseHandle(handle)
            else:
                os.killpg(os.getpgid(pid), signal.SIGKILL)

        remove_pid()
        console.print("[green]Server stopped[/green]")
    except Exception as e:
        console.print(f"[red]Error stopping server: {e}[/red]")


@app.command()
def restart():
    """Restart the proxy server."""
    pid = get_server_pid()
    was_running = pid and is_server_running(pid)

    if was_running:
        try:
            if sys.platform == "win32":
                import ctypes

                kernel32 = ctypes.windll.kernel32
                PROCESS_TERMINATE = 0x0001
                handle = kernel32.OpenProcess(PROCESS_TERMINATE, False, pid)
                if handle:
                    kernel32.TerminateProcess(handle, 0)
                    kernel32.CloseHandle(handle)
            else:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            time.sleep(2)
            if is_server_running(pid):
                if sys.platform == "win32":
                    import ctypes

                    kernel32 = ctypes.windll.kernel32
                    PROCESS_TERMINATE = 0x0001
                    handle = kernel32.OpenProcess(PROCESS_TERMINATE, False, pid)
                    if handle:
                        kernel32.TerminateProcess(handle, 1)
                        kernel32.CloseHandle(handle)
                else:
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
            remove_pid()
        except Exception as e:
            console.print(f"[yellow]Warning: {e}[/yellow]")

    time.sleep(1)

    config = load_config()
    host, port = config.host, config.port

    if sys.platform == "win32":
        if sys.executable.endswith("python.exe"):
            python_exe = sys.executable
        else:
            python_exe = sys.executable
        creationflags = (
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        )
        process = subprocess.Popen(
            [
                python_exe,
                "-m",
                "kilo_proxy.server_runner",
                "--host",
                host,
                "--port",
                str(port),
            ],
            creationflags=creationflags,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(3)
        if process.poll() is not None and process.returncode != 0:
            console.print(
                f"[red]Server failed to start (exit code: {process.returncode})[/red]"
            )
            return
        write_pid(process.pid)
        console.print(
            f"[green]Server restarted on {host}:{port} with PID {process.pid}[/green]"
        )
    else:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "kilo_proxy.server_runner",
                "--host",
                host,
                "--port",
                str(port),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        time.sleep(3)
        if process.poll() is not None and process.returncode != 0:
            console.print(
                f"[red]Server failed to start (exit code: {process.returncode})[/red]"
            )
            return
        write_pid(process.pid)
        console.print(
            f"[green]Server restarted on {host}:{port} with PID {process.pid}[/green]"
        )


@app.command()
def proxy(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(5380, "--port", "-p", help="Port to bind to"),
):
    """Run the proxy server in the foreground."""
    config = load_config()
    config.host = host
    config.port = port
    save_config(config)

    import uvicorn

    console.print(f"[green]Starting Kilo Proxy v{__version__} on {host}:{port}[/green]")
    console.print(f"[blue]OpenAI-compatible API: http://{host}:{port}/v1[/blue]")

    uvicorn.run(
        "kilo_proxy.server:app",
        host=host,
        port=port,
        access_log=True,
    )


@app.command()
def status():
    """Show server status and configuration."""
    config = load_config()
    pid = get_server_pid()

    table = Table(title="Kilo Proxy Status")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Version", __version__)
    table.add_row("Config Path", str(get_config_path()))
    table.add_row("Host", config.host)
    table.add_row("Port", str(config.port))
    table.add_row(
        "Auth Token",
        f"{config.auth_token[:8]}..."
        if len(config.auth_token) > 8
        else config.auth_token,
    )
    table.add_row("Session ID", config.session_id or "Not set")
    table.add_row(
        "Broke Mode", "[yellow]ON[/yellow]" if config.broke else "[dim]OFF[/dim]"
    )

    if pid and is_server_running(pid):
        table.add_row("Status", f"Running (PID: {pid})")
    elif pid:
        table.add_row("Status", "Stopped (stale PID file)")
        remove_pid()
    else:
        table.add_row("Status", "Not running")

    console.print(table)


@app.command()
def broke(
    mode: Optional[str] = typer.Argument(None, help="on/off/toggle"),
    list_models: bool = typer.Option(False, "--list", "-l", help="List free models"),
):
    """Toggle or set broke mode (free models only)."""
    config = load_config()

    if list_models:
        _list_free_models(config)
        return

    if mode == "on":
        config.broke = True
    elif mode == "off":
        config.broke = False
    elif mode == "toggle":
        config.broke = not config.broke
    else:
        config.broke = not config.broke

    save_config(config)
    status = "[green]ON[/green]" if config.broke else "[yellow]OFF[/yellow]"
    console.print(f"Broke mode: {status}")
    if config.broke:
        console.print("[dim]Only free models will be shown[/dim]")
    console.print(
        "[cyan]Note: Restart the server for changes to take effect: kilo-proxy restart[/cyan]"
    )


def _list_free_models(config: Config):
    """List all free models from Kilo API."""
    console.print("[yellow]Fetching free models...[/yellow]\n")

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(
                f"http://{config.host}:{config.port}/v1/models",
                headers={"Authorization": f"Bearer {config.auth_token}"},
            )

            if response.status_code != 200:
                console.print(
                    f"[red]Failed to fetch models: {response.status_code}[/red]"
                )
                return

            data = response.json()

            table = Table(title="Free Models")
            table.add_column("ID", style="cyan")
            table.add_column("Name", style="green")

            count = 0
            for model in data.get("data", []):
                model_id = model.get("id", "")
                model_name = model.get("name", "")
                if is_free_model(model_id, model_name):
                    table.add_row(model_id, model_name)
                    count += 1

            if count == 0:
                console.print("[yellow]No free models found[/yellow]")
            else:
                console.print(table)
                console.print(f"\n[green]Found {count} free models[/green]")

    except httpx.RequestError:
        console.print("[red]Proxy not running. Start with: kilo-proxy start[/red]")


@app.command()
def models():
    """List available models from Kilo API."""
    config = load_config()

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(
                f"http://{config.host}:{config.port}/v1/models",
                headers={"Authorization": f"Bearer {config.auth_token}"},
            )

            if response.status_code != 200:
                console.print(
                    f"[red]Failed to fetch models: {response.status_code}[/red]"
                )
                return

            data = response.json()

            models_list = data.get("data", [])

            if config.broke:
                models_list = [
                    m
                    for m in models_list
                    if is_free_model(m.get("id", ""), m.get("name", ""))
                ]
                title = "Available Free Models (Broke Mode)"
            else:
                title = "Available Models"

            table = Table(title=title)
            table.add_column("ID", style="cyan")
            table.add_column("Name", style="green")

            for model in models_list:
                model_id = model.get("id", "Unknown")
                model_name = model.get("name", model_id)
                table.add_row(model_id, model_name)

            console.print(table)
            console.print(f"\n[dim]Total: {len(models_list)} models[/dim]")

    except httpx.RequestError:
        console.print("[red]Proxy not running. Start with: kilo-proxy start[/red]")


@app.command()
def new_session():
    """Generate a new session ID."""
    session_id = regenerate_session_id()
    console.print(f"[green]New session ID: {session_id}[/green]")


@app.command("config")
def config_cmd(
    show: bool = typer.Option(False, "--show", "-s", help="Show current config"),
):
    """Interactive configuration wizard."""
    config = load_config()

    if show:
        console.print("[cyan]Current Configuration:[/cyan]")
        console.print_json(json.dumps(config.model_dump(), indent=2))
        return

    console.print("[bold cyan]Kilo Proxy Configuration Wizard[/bold cyan]\n")

    current_token = config.auth_token
    token_default = current_token if current_token != "anonymous" else ""
    token_input = Prompt.prompt("Auth token", default=token_default or "anonymous")
    config.auth_token = (
        token_input if token_input and token_input != "anonymous" else "anonymous"
    )

    config.host = Prompt.prompt("Host", default=config.host)
    config.port = int(Prompt.prompt("Port", default=str(config.port)))

    broke_input = Prompt.prompt(
        "Broke mode (free models only)", default="y" if config.broke else "n"
    ).lower()
    config.broke = broke_input in ("y", "yes", "true", "1")

    new_session = Prompt.prompt("Generate new session ID?", default="n").lower()
    if new_session in ("y", "yes", "true", "1"):
        config.session_id = generate_kilo_session_id()
        console.print(f"[dim]New session ID: {config.session_id}[/dim]")

    save_config(config)
    console.print(f"\n[green]Config saved to {get_config_path()}[/green]")
    console.print_json(json.dumps(config.model_dump(), indent=2))


@app.command("install")
def install_cmd(
    target: str = typer.Argument(..., help="What to install (opencode)"),
):
    """Install and configure integrations."""
    if target.lower() == "opencode":
        install_opencode()
    else:
        console.print(f"[red]Unknown install target: {target}[/red]")
        console.print("Available: opencode")


@app.command("opencode")
def opencode_cmd():
    """Start the proxy and launch opencode."""
    config = load_config()
    pid = get_server_pid()

    if not pid or not is_server_running(pid):
        console.print("[yellow]Proxy not running, starting in background...[/yellow]")

        host, port = config.host, config.port

        if sys.platform == "win32":
            python_exe = sys.executable
            creationflags = (
                subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
            )
            process = subprocess.Popen(
                [
                    python_exe,
                    "-m",
                    "kilo_proxy.server_runner",
                    "--host",
                    host,
                    "--port",
                    str(port),
                ],
                creationflags=creationflags,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            write_pid(process.pid)
            console.print(
                f"[green]Server started on {host}:{port} with PID {process.pid}[/green]"
            )
        else:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "kilo_proxy.server_runner",
                    "--host",
                    host,
                    "--port",
                    str(port),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            write_pid(process.pid)
            console.print(
                f"[green]Server started on {host}:{port} with PID {process.pid}[/green]"
            )

    default_opencode_path = Path.home() / ".config" / "opencode" / "opencode.json"
    if default_opencode_path.exists():
        console.print("[green]OpenCode config found, launching opencode...[/green]")
        subprocess.run(["opencode"])
    else:
        console.print(
            "[yellow]OpenCode not configured. Run: kilo-proxy install opencode[/yellow]"
        )


def fetch_models_from_proxy(config: Config) -> List[dict]:
    """Fetch models from running proxy."""
    with httpx.Client(timeout=30.0) as client:
        response = client.get(
            f"http://{config.host}:{config.port}/v1/models",
            headers={"Authorization": f"Bearer {config.auth_token}"},
        )
        data = response.json()
        return data.get("data", [])


def fetch_default_omo_config() -> dict:
    """Fetch default oh-my-opencode config from repo."""
    url = "https://raw.githubusercontent.com/code-yeongyu/oh-my-opencode/master/assets/default-config.json"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return json.loads(response.read().decode())
    except Exception:
        return {
            "$schema": "https://raw.githubusercontent.com/code-yeongyu/oh-my-opencode/master/assets/oh-my-opencode.schema.json",
            "agents": {},
            "categories": {},
        }


def extract_omo_models(omo_config: dict) -> List[str]:
    """Extract unique model references from oh-my-opencode config."""
    models = set()

    for agent_config in omo_config.get("agents", {}).values():
        if "model" in agent_config:
            models.add(agent_config["model"])

    for cat_config in omo_config.get("categories", {}).values():
        if "model" in cat_config:
            models.add(cat_config["model"])

    return sorted(list(models))


def apply_omo_mappings(omo_config: dict, mappings: dict) -> dict:
    """Apply model mappings to oh-my-opencode config."""
    config = copy.deepcopy(omo_config)

    for agent_name, agent_config in config.get("agents", {}).items():
        if "model" in agent_config:
            original = agent_config["model"]
            if original in mappings:
                agent_config["model"] = f"kilo-proxy/{mappings[original]}"

    for cat_name, cat_config in config.get("categories", {}).items():
        if "model" in cat_config:
            original = cat_config["model"]
            if original in mappings:
                cat_config["model"] = f"kilo-proxy/{mappings[original]}"

    return config


def interactive_select(prompt: str, options: List[str], default: int = 0) -> int:
    """Interactive selection. Returns selected index."""
    console.print(f"\n{prompt}")
    for i, opt in enumerate(options):
        marker = ">" if i == default else " "
        console.print(f"  {marker} {i + 1}. {opt}")

    choice = Prompt.ask(
        "  Select",
        default=str(default + 1),
    )
    try:
        return int(choice) - 1
    except ValueError:
        return default


def interactive_model_picker(
    models: List[dict], default: str = None, prompt: str = "Select model"
) -> str:
    """Interactive model picker. Returns selected model ID."""
    if not models:
        return default or ""

    page_size = 15
    total_pages = (len(models) + page_size - 1) // page_size
    current_page = 0

    default_idx = 0
    if default:
        for i, m in enumerate(models):
            if m["id"] == default:
                default_idx = i
                current_page = i // page_size
                break

    while True:
        start = current_page * page_size
        end = min(start + page_size, len(models))
        page_models = models[start:end]

        console.print(f"\n  {prompt} (Page {current_page + 1}/{total_pages}):")
        for i, m in enumerate(page_models):
            idx = start + i
            marker = "*" if m["id"] == default else " "
            console.print(f"    {marker} {idx + 1:3d}. {m['id'][:50]}")

        console.print(
            f"\n    [dim]n=next, p=prev, q=use default, 1-{len(models)}=select[/dim]"
        )

        choice = Prompt.ask("  Choice", default=str(default_idx + 1))

        if choice == "n" and current_page < total_pages - 1:
            current_page += 1
            continue
        elif choice == "p" and current_page > 0:
            current_page -= 1
            continue
        elif choice == "q":
            return default or models[0]["id"]
        elif choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                return models[idx]["id"]

        console.print("  [yellow]Invalid choice, try again[/yellow]")


def install_opencode():
    """Interactive OpenCode installation wizard."""
    console.print("[bold cyan]OpenCode Installation Wizard[/bold cyan]\n")

    config = load_config()

    # [1/7] Find OpenCode config
    console.print("[1/7] Finding OpenCode config...")

    default_opencode_path = Path.home() / ".config" / "opencode" / "opencode.json"
    opencode_path = None

    if default_opencode_path.exists():
        use_default = Prompt.ask(
            f"  Found: {default_opencode_path}\n  Use this config file?", default="y"
        ).lower() in ("y", "yes")
        if use_default:
            opencode_path = default_opencode_path
        else:
            custom_path = Prompt.ask("  Enter path to opencode.json")
            opencode_path = Path(custom_path).expanduser()
            if not opencode_path.exists():
                console.print(f"[red]  Config not found: {opencode_path}[/red]")
                return
    else:
        console.print("  [yellow]No default config found[/yellow]")
        custom_path = Prompt.ask(
            "  Enter path to opencode.json (or press Enter to create new)"
        )
        if custom_path:
            opencode_path = Path(custom_path).expanduser()
        else:
            opencode_path = default_opencode_path
            opencode_path.parent.mkdir(parents=True, exist_ok=True)

    console.print(f"  [green]Using: {opencode_path}[/green]\n")

    # [2/7] Broke mode
    console.print("[2/7] Broke mode")
    is_broke = Prompt.ask(
        "  Are you broke? (Only free models)", default="y" if config.broke else "n"
    ).lower() in ("y", "yes", "true", "1")
    config.broke = is_broke
    save_config(config)

    if is_broke:
        console.print("  [dim]Only free models will be available[/dim]\n")
    else:
        console.print("  [dim]All models will be available[/dim]\n")

    # [3/7] Fetch models from proxy
    console.print("[3/7] Fetching models...")

    try:
        models = fetch_models_from_proxy(config)
        if is_broke:
            models = [
                m for m in models if is_free_model(m.get("id", ""), m.get("name", ""))
            ]
            console.print(f"  [dim]Found {len(models)} free models[/dim]\n")
        else:
            console.print(f"  [dim]Found {len(models)} models[/dim]\n")
    except Exception as e:
        console.print(f"[red]  Error fetching models: {e}[/red]")
        console.print("[yellow]  Make sure proxy is running: kilo-proxy start[/yellow]")
        return

    if not models:
        console.print("[red]  No models found![/red]")
        return

    # Find default model
    default_model = None
    for m in models:
        if "minimax-m2.5:free" in m["id"]:
            default_model = m["id"]
            break
    if not default_model:
        default_model = models[0]["id"]

    # [4/7] Model configuration mode
    console.print("[4/7] Model Configuration")

    config_mode = interactive_select(
        "  Configure models:",
        ["Same model for all agents/categories", "Configure per agent/category"],
    )

    model_mappings = {}

    if config_mode == 0:
        console.print("\n  Select default model for all:")
        selected_model = interactive_model_picker(models, default=default_model)
        model_mappings["__default__"] = selected_model
        console.print(f"  [green]Selected: {selected_model}[/green]\n")

    # [5/7] Oh My OpenCode Support
    console.print("[5/7] Oh My OpenCode Support")

    install_omo = Prompt.ask(
        "  Install Oh My OpenCode support?", default="y"
    ).lower() in ("y", "yes", "true", "1")

    omo_path = opencode_path.parent / "oh-my-opencode.json"
    omo_config = {}

    if install_omo:
        if omo_path.exists():
            use_existing = Prompt.ask(
                f"  Found {omo_path}. Update it?", default="y"
            ).lower() in ("y", "yes")
            if use_existing:
                try:
                    with open(omo_path) as f:
                        omo_config = json.load(f)
                except Exception as e:
                    console.print(
                        f"  [yellow]Could not read existing config: {e}[/yellow]"
                    )
                    omo_config = fetch_default_omo_config()
            else:
                omo_config = fetch_default_omo_config()
        else:
            console.print("  [dim]Fetching default Oh My OpenCode config...[/dim]")
            omo_config = fetch_default_omo_config()

        unique_models = extract_omo_models(omo_config)

        if unique_models:
            console.print(
                f"\n  [dim]Mapping {len(unique_models)} Oh My OpenCode model references to Kilo...[/dim]\n"
            )

            omo_model_map = {}
            apply_to_all = None

            for omo_model in unique_models:
                if apply_to_all:
                    omo_model_map[omo_model] = apply_to_all
                    console.print(
                        f"    {omo_model} -> {apply_to_all} [dim](applied to all)[/dim]"
                    )
                else:
                    console.print(f"  Model mapping for '{omo_model}':")
                    selected = interactive_model_picker(
                        models, default=default_model, prompt="    Replace with"
                    )
                    omo_model_map[omo_model] = selected
                    console.print(f"    [green]{omo_model} -> {selected}[/green]")

                    remaining = len(unique_models) - len(omo_model_map)
                    if remaining > 0:
                        apply_all = Prompt.ask(
                            f"    Apply '{selected}' to all {remaining} remaining?",
                            default="n",
                        ).lower() in ("y", "yes")
                        if apply_all:
                            apply_to_all = selected

            omo_config = apply_omo_mappings(omo_config, omo_model_map)

        # Per-config if selected
        if config_mode == 1:
            console.print("\n  Configuring models per agent/category:")

            if "agents" in omo_config:
                for agent_name, agent_cfg in omo_config["agents"].items():
                    if "model" in agent_cfg:
                        current = agent_cfg["model"].replace("kilo-proxy/", "")
                        console.print(f"\n  Agent '{agent_name}':")
                        selected = interactive_model_picker(
                            models,
                            default=current
                            if current in [m["id"] for m in models]
                            else default_model,
                            prompt="    Select model",
                        )
                        agent_cfg["model"] = f"kilo-proxy/{selected}"

            if "categories" in omo_config:
                for cat_name, cat_cfg in omo_config["categories"].items():
                    if "model" in cat_cfg:
                        current = cat_cfg["model"].replace("kilo-proxy/", "")
                        console.print(f"\n  Category '{cat_name}':")
                        selected = interactive_model_picker(
                            models,
                            default=current
                            if current in [m["id"] for m in models]
                            else default_model,
                            prompt="    Select model",
                        )
                        cat_cfg["model"] = f"kilo-proxy/{selected}"

    console.print()

    # [6/7] Handle existing opencode.json
    console.print("[6/7] Writing Configuration")

    existing_config = {}
    merge_mode = 0

    if opencode_path.exists():
        try:
            with open(opencode_path) as f:
                existing_config = json.load(f)
        except Exception:
            existing_config = {}

        if existing_config:
            merge_mode = interactive_select(
                "  opencode.json already exists. How to handle?",
                [
                    "Merge (preserve existing, add/update kilo provider)",
                    "Overwrite (replace completely)",
                    "Cancel",
                ],
            )

            if merge_mode == 2:
                console.print("  [yellow]Cancelled[/yellow]")
                return
            elif merge_mode == 1:
                existing_config = {}

    # [7/7] Write configs

    kilo_provider = {
        "npm": "@ai-sdk/openai-compatible",
        "name": "Kilo-Proxy",
        "options": {
            "baseURL": f"http://{config.host}:{config.port}/v1",
            "apiKey": config.auth_token,
        },
        "models": {m["id"]: {"name": m.get("name", m["id"])} for m in models},
    }

    if "$schema" not in existing_config:
        existing_config["$schema"] = "https://opencode.ai/config.json"

    if "provider" not in existing_config:
        existing_config["provider"] = {}
    existing_config["provider"]["kilo-proxy"] = kilo_provider

    if install_omo:
        if "plugin" not in existing_config:
            existing_config["plugin"] = []
        if "oh-my-opencode" not in existing_config["plugin"]:
            existing_config["plugin"].append("oh-my-opencode")

    with open(opencode_path, "w") as f:
        json.dump(existing_config, f, indent=2)
    console.print(f"  [green]Written to {opencode_path}[/green]")

    if install_omo and omo_config:
        with open(omo_path, "w") as f:
            json.dump(omo_config, f, indent=2)
        console.print(f"  [green]Written to {omo_path}[/green]")

    console.print("\n[green]OpenCode installation complete![/green]")
    console.print("[dim]Run 'opencode' to start using Kilo models[/dim]")


@app.command("autolaunch")
def autolaunch_cmd():
    """Install the proxy to run on system boot."""
    config = load_config()

    if sys.platform == "win32":
        _install_windows_startup(config)
    elif sys.platform == "darwin":
        _install_macos_startup(config)
    else:
        _install_linux_startup(config)


@app.command("unautolaunch")
def unautolaunch_cmd():
    """Remove the proxy from system boot."""
    if sys.platform == "win32":
        _uninstall_windows_startup()
    elif sys.platform == "darwin":
        _uninstall_macos_startup()
    else:
        _uninstall_linux_startup()


def _install_windows_startup(config: Config):
    startup_folder = (
        Path(os.environ.get("APPDATA", ""))
        / "Microsoft"
        / "Windows"
        / "Start Menu"
        / "Programs"
        / "Startup"
    )
    if not startup_folder.exists():
        startup_folder.mkdir(parents=True, exist_ok=True)

    shortcut_path = startup_folder / "KiloProxy.lnk"

    python_exe = sys.executable
    python_dir = Path(python_exe).parent
    pythonw_exe = python_dir / "pythonw.exe"
    exe_to_use = str(pythonw_exe) if pythonw_exe.exists() else python_exe
    working_dir = str(Path(__file__).parent.parent)

    ps_script = f'''
$ws = New-Object -ComObject WScript.Shell
$s = $ws.CreateShortcut("{shortcut_path}")
$s.TargetPath = "{exe_to_use}"
$s.Arguments = "-m kilo_proxy.server_runner --host {config.host} --port {config.port}"
$s.WorkingDirectory = "{working_dir}"
$s.Description = "Kilo Proxy"
$s.Save()
'''

    try:
        result = subprocess.run(
            ["powershell", "-Command", ps_script],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0 and shortcut_path.exists():
            console.print(
                "[green]Autolaunch installed! Kilo Proxy will start on login.[/green]"
            )
        else:
            console.print(
                f"[red]Failed to create startup shortcut: {result.stderr}[/red]"
            )
    except Exception as e:
        console.print(f"[red]Error installing autolaunch: {e}[/red]")


def _uninstall_windows_startup():
    startup_folder = (
        Path(os.environ.get("APPDATA", ""))
        / "Microsoft"
        / "Windows"
        / "Start Menu"
        / "Programs"
        / "Startup"
    )
    shortcut_path = startup_folder / "KiloProxy.lnk"

    try:
        if shortcut_path.exists():
            shortcut_path.unlink()
            console.print("[green]Autolaunch removed![/green]")
        else:
            console.print("[yellow]No autolaunch shortcut found[/yellow]")
    except Exception as e:
        console.print(f"[red]Error removing autolaunch: {e}[/red]")


def _install_macos_startup(config: Config):
    launch_agents = Path.home() / "Library" / "LaunchAgents"
    launch_agents.mkdir(parents=True, exist_ok=True)

    plist_path = launch_agents / "com.kilo.proxy.plist"

    python_exe = sys.executable
    working_dir = str(Path(__file__).parent.parent)

    plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.kilo.proxy</string>
    <key>ProgramArguments</key>
    <array>
        <string>{python_exe}</string>
        <string>-m</string>
        <string>kilo_proxy.server_runner</string>
        <string>--host</string>
        <string>{config.host}</string>
        <string>--port</string>
        <string>{config.port}</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <false/>
    <key>WorkingDirectory</key>
    <string>{working_dir}</string>
    <key>StandardOutPath</key>
    <string>/tmp/kilo-proxy.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/kilo-proxy-error.log</string>
</dict>
</plist>"""

    try:
        plist_path.write_text(plist_content)
        subprocess.run(["launchctl", "load", str(plist_path)], check=True)
        console.print(
            "[green]Autolaunch installed! Kilo Proxy will start on login.[/green]"
        )
    except Exception as e:
        console.print(f"[red]Error installing autolaunch: {e}[/red]")


def _uninstall_macos_startup():
    plist_path = Path.home() / "Library" / "LaunchAgents" / "com.kilo.proxy.plist"

    try:
        if plist_path.exists():
            subprocess.run(
                ["launchctl", "unload", str(plist_path)], capture_output=True
            )
            plist_path.unlink()
        console.print("[green]Autolaunch removed![/green]")
    except Exception as e:
        console.print(f"[red]Error removing autolaunch: {e}[/red]")


def _install_linux_startup(config: Config):
    systemd_dir = Path.home() / ".config" / "systemd" / "user"
    systemd_dir.mkdir(parents=True, exist_ok=True)

    service_path = systemd_dir / "kilo-proxy.service"

    python_exe = sys.executable

    service_content = f"""[Unit]
Description=Kilo Proxy - OpenAI-compatible API proxy
After=network.target

[Service]
Type=simple
ExecStart={python_exe} -m kilo_proxy.server_runner --host {config.host} --port {config.port}
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
"""

    try:
        service_path.write_text(service_content)
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        subprocess.run(["systemctl", "--user", "enable", "kilo-proxy"], check=True)
        console.print(
            "[green]Autolaunch installed! Kilo Proxy will start on login.[/green]"
        )
    except Exception as e:
        console.print(f"[red]Error installing autolaunch: {e}[/red]")


def _uninstall_linux_startup():
    service_path = Path.home() / ".config" / "systemd" / "user" / "kilo-proxy.service"

    try:
        subprocess.run(
            ["systemctl", "--user", "disable", "kilo-proxy"], capture_output=True
        )
        if service_path.exists():
            service_path.unlink()
        subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True)
        console.print("[green]Autolaunch removed![/green]")
    except Exception as e:
        console.print(f"[red]Error removing autolaunch: {e}[/red]")


@app.command("config-show")
def config_show():
    """Show current configuration."""
    config = load_config()
    config_path = get_config_path()

    console.print(f"[cyan]Config file: {config_path}[/cyan]")
    console.print_json(json.dumps(config.model_dump(), indent=2))


@app.command("logs")
def logs_cmd(
    lines: int = typer.Option(100, "--lines", "-n", help="Number of lines to show"),
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow log output"),
    errors_only: bool = typer.Option(False, "--errors", "-e", help="Show errors only"),
):
    """Show server logs."""
    if not LOG_FILE.exists():
        console.print(
            "[yellow]No log file found. Server may not have started yet.[/yellow]"
        )
        return

    if errors_only:
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                content = f.read()
                error_lines = [
                    line
                    for line in content.splitlines()
                    if "ERROR" in line or "CRITICAL" in line
                ]
                if error_lines:
                    for line in error_lines[-lines:]:
                        console.print(line)
                else:
                    console.print("[yellow]No errors found[/yellow]")
        except Exception as e:
            console.print(f"[red]Error reading logs: {e}[/red]")
        return

    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            all_lines = f.readlines()
            for line in all_lines[-lines:]:
                console.print(line.rstrip())
    except Exception as e:
        console.print(f"[red]Error reading logs: {e}[/red]")


shuffle_app = typer.Typer(name="shuffle", help="IP shuffler management")
app.add_typer(shuffle_app, name="shuffle")


@shuffle_app.command("on")
def shuffle_on():
    """Enable IP shuffling."""
    shuffler = get_shuffler()
    asyncio.run(shuffler.set_enabled(True))
    console.print("[green]IP shuffling enabled[/green]")
    if not shuffler.get_proxy_list():
        console.print(
            "[yellow]No proxies configured. Add with: kilo-proxy shuffle add <url>[/yellow]"
        )


@shuffle_app.command("off")
def shuffle_off():
    """Disable IP shuffling."""
    shuffler = get_shuffler()
    asyncio.run(shuffler.set_enabled(False))
    console.print("[yellow]IP shuffling disabled[/yellow]")


@shuffle_app.command("add")
def shuffle_add(
    proxy_url: str = typer.Argument(
        ..., help="Proxy URL (e.g., http://ip:port or socks5://ip:port)"
    ),
):
    """Add a proxy to the rotation list."""
    shuffler = get_shuffler()
    asyncio.run(shuffler.add_proxy(proxy_url))
    console.print(f"[green]Added proxy: {proxy_url}[/green]")


@shuffle_app.command("remove")
def shuffle_remove(
    proxy_url: str = typer.Argument(..., help="Proxy URL to remove"),
):
    """Remove a proxy from the rotation list."""
    shuffler = get_shuffler()
    removed = asyncio.run(shuffler.remove_proxy(proxy_url))
    if removed:
        console.print(f"[green]Removed proxy: {proxy_url}[/green]")
    else:
        console.print(f"[yellow]Proxy not found: {proxy_url}[/yellow]")


@shuffle_app.command("clear")
def shuffle_clear():
    """Clear all proxies from the rotation list."""
    shuffler = get_shuffler()
    asyncio.run(shuffler.clear_proxies())
    console.print("[green]All proxies cleared[/green]")


@shuffle_app.command("load")
def shuffle_load(
    source: str = typer.Argument(
        ..., help="URL or file path to load proxies from (one per line)"
    ),
    timeout: int = typer.Option(
        30, "--timeout", "-t", help="Timeout in seconds for URLs"
    ),
):
    """Load proxies from a URL or file (auto-detected). Lines starting with # are comments."""
    shuffler = get_shuffler()
    is_url = source.startswith(("http://", "https://"))
    source_type = "URL" if is_url else "file"

    try:
        added = asyncio.run(shuffler.load_proxies(source, timeout))
        console.print(
            f"[green]Loaded {added} proxies from {source_type}: {source}[/green]"
        )
    except Exception as e:
        console.print(f"[red]Error loading proxies from {source_type}: {e}[/red]")


@shuffle_app.command("load-file", hidden=True)
def shuffle_load_file(
    file_path: str = typer.Argument(
        ..., help="Path to file containing proxies (one per line)"
    ),
):
    """Load proxies from a file (one proxy per line, lines starting with # are comments)."""
    shuffler = get_shuffler()

    try:
        added = asyncio.run(shuffler.load_proxies(file_path))
        console.print(f"[green]Loaded {added} proxies from {file_path}[/green]")
    except Exception as e:
        console.print(f"[red]Error loading proxies: {e}[/red]")


@shuffle_app.command("load-url", hidden=True)
def shuffle_load_url(
    url: str = typer.Argument(..., help="URL to fetch proxies from (one per line)"),
    timeout: int = typer.Option(30, "--timeout", "-t", help="Timeout in seconds"),
):
    """Load proxies from a URL (one proxy per line, lines starting with # are comments)."""
    shuffler = get_shuffler()

    try:
        added = asyncio.run(shuffler.load_proxies(url, timeout))
        console.print(f"[green]Loaded {added} proxies from {url}[/green]")
    except Exception as e:
        console.print(f"[red]Error loading proxies: {e}[/red]")


@shuffle_app.command("list")
def shuffle_list():
    """List all configured proxies."""
    shuffler = get_shuffler()
    proxies = shuffler.get_proxy_list()

    if not proxies:
        console.print("[yellow]No proxies configured[/yellow]")
        return

    table = Table(title="Configured Proxies")
    table.add_column("#", style="dim")
    table.add_column("Proxy URL", style="cyan")
    table.add_column("Status", style="green")

    current_idx = shuffler.get_current_index()
    for i, proxy in enumerate(proxies):
        status = "[yellow]* CURRENT[/yellow]" if i == current_idx else ""
        table.add_row(str(i + 1), proxy, status)

    console.print(table)
    console.print(f"\n[dim]Total: {len(proxies)} proxies[/dim]")


@shuffle_app.command("interval")
def shuffle_interval(
    seconds: int = typer.Argument(..., help="Shuffle interval in seconds (minimum 60)"),
):
    """Set the shuffle interval in seconds."""
    shuffler = get_shuffler()
    asyncio.run(shuffler.set_interval(seconds))
    actual = shuffler.get_interval()
    console.print(
        f"[green]Shuffle interval set to {actual} seconds ({actual // 60} minutes)[/green]"
    )


@shuffle_app.command("now")
def shuffle_now():
    """Force an immediate shuffle."""
    shuffler = get_shuffler()
    proxy, session_id = asyncio.run(shuffler.shuffle_now())
    if proxy:
        console.print(f"[green]Shuffled to proxy: {proxy}[/green]")
    else:
        console.print("[green]Session ID regenerated (no proxy configured)[/green]")
    console.print(f"[dim]New session ID: {session_id}[/dim]")


@shuffle_app.command("fix")
def shuffle_fix():
    """Fix and normalize existing proxy URLs in config."""
    shuffler = get_shuffler()
    proxies = shuffler.get_proxy_list()

    if not proxies:
        console.print("[yellow]No proxies to fix[/yellow]")
        return

    normalized = []
    for proxy in proxies:
        normalized.append(shuffler.normalize_proxy(proxy))

    asyncio.run(shuffler.clear_proxies())
    asyncio.run(shuffler.add_proxies(normalized))

    console.print(f"[green]Fixed {len(normalized)} proxies[/green]")
    for i, p in enumerate(normalized):
        console.print(f"  {i + 1}. {p}")


@shuffle_app.command("check")
def shuffle_check(
    timeout: int = typer.Option(
        30, "--timeout", "-t", help="Timeout in seconds for each proxy"
    ),
    remove: bool = typer.Option(
        False, "--remove", "-r", help="Automatically remove broken proxies"
    ),
):
    """Check all proxies and remove bad ones."""
    shuffler = get_shuffler()
    proxies = shuffler.get_proxy_list()

    if not proxies:
        console.print("[yellow]No proxies to check[/yellow]")
        return

    console.print(
        f"[cyan]Checking {len(proxies)} proxies (timeout: {timeout}s)...[/cyan]\n"
    )

    working, broken = asyncio.run(shuffler.check_proxies(timeout))

    console.print(f"[green]Working: {len(working)}[/green]")
    if working:
        for p in working:
            console.print(f"  ✓ {p}")

    console.print(f"\n[red]Broken: {len(broken)}[/red]")
    if broken:
        for p in broken:
            console.print(f"  ✗ {p}")

    if broken and remove:
        removed = asyncio.run(shuffler.remove_proxies(broken))
        console.print(f"\n[green]Removed {removed} broken proxies[/green]")
    elif broken:
        console.print(
            f"\n[yellow]Run with --remove to automatically remove broken proxies[/yellow]"
        )


@shuffle_app.command("status")
def shuffle_status():
    """Show IP shuffler status."""
    shuffler = get_shuffler()
    status = shuffler.get_status()

    table = Table(title="IP Shuffler Status")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row(
        "Enabled", "[green]Yes[/green]" if status["enabled"] else "[yellow]No[/yellow]"
    )
    table.add_row(
        "Running", "[green]Yes[/green]" if status["running"] else "[dim]No[/dim]"
    )
    table.add_row("Interval", f"{status['interval']}s ({status['interval'] // 60} min)")
    table.add_row("Proxy Count", str(status["proxy_count"]))
    table.add_row("Current Index", str(status["current_index"]))
    table.add_row("Current Proxy", status["current_proxy"] or "[dim]None[/dim]")
    table.add_row(
        "Current Session",
        status["current_session_id"][:16] + "..."
        if status["current_session_id"]
        else "[dim]None[/dim]",
    )

    if status["last_shuffle"] > 0:
        elapsed = time.time() - status["last_shuffle"]
        table.add_row("Last Shuffle", f"{int(elapsed)}s ago")
    else:
        table.add_row("Last Shuffle", "[dim]Never[/dim]")

    console.print(table)

    if status["proxy_list"]:
        console.print("\n[cyan]Proxy List:[/cyan]")
        for i, proxy in enumerate(status["proxy_list"]):
            marker = " *" if i == status["current_index"] else "  "
            console.print(f"  {marker}{i + 1}. {proxy}")


if __name__ == "__main__":
    app()
