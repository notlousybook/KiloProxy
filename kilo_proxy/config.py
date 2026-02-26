"""Configuration management for kilo-proxy."""

import json
import random
import string
from pathlib import Path

from typing import List

from pydantic import BaseModel


def generate_kilo_session_id() -> str:
    """Generate Kilo-style session ID: ses_xxxxxxxxxxxxxxxxxxxxxxxx"""
    chars = string.ascii_letters + string.digits
    return "ses_" + "".join(random.choices(chars, k=24))


class Config(BaseModel):
    auth_token: str = "anonymous"
    host: str = "127.0.0.1"
    port: int = 5380
    session_id: str = ""
    broke: bool = False
    ip_shuffle_enabled: bool = False
    ip_shuffle_interval: int = 900
    proxy_list: List[str] = []
    current_proxy_index: int = 0
    last_shuffle_time: float = 0.0

    def __init__(self, **data):
        super().__init__(**data)
        if not self.session_id:
            self.session_id = generate_kilo_session_id()


def get_config_dir() -> Path:
    home = Path.home()
    config_dir = home / ".kilo-proxy"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_config_path() -> Path:
    return get_config_dir() / "config.json"


def get_pid_path() -> Path:
    return get_config_dir() / "server.pid"


def load_config() -> Config:
    config_path = get_config_path()
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return Config(**data)
        except (json.JSONDecodeError, KeyError):
            return Config()
    return Config()


def save_config(config: Config) -> None:
    config_path = get_config_path()
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config.model_dump(), f, indent=2)


def get_session_id() -> str:
    config = load_config()
    return config.session_id


def regenerate_session_id() -> str:
    config = load_config()
    config.session_id = generate_kilo_session_id()
    save_config(config)
    return config.session_id
