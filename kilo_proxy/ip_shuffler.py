"""IP Shuffler module for rotating proxies and session IDs."""

import asyncio
import time
from pathlib import Path
from typing import List, Optional, Tuple

import httpx

from kilo_proxy.config import generate_kilo_session_id, load_config, save_config

_shuffler_instance: Optional["IPShuffler"] = None
_shuffler_task: Optional[asyncio.Task] = None


class IPShuffler:
    def __init__(self):
        self._current_proxy: Optional[str] = None
        self._current_session_id: str = ""
        self._enabled: bool = False
        self._interval: int = 900
        self._proxy_list: List[str] = []
        self._current_index: int = 0
        self._last_shuffle: float = 0.0
        self._lock = asyncio.Lock()
        self._running: bool = False
        self._initialized: bool = False

    def _initialize_sync(self):
        if self._initialized:
            return
        config = load_config()
        self._enabled = config.ip_shuffle_enabled
        self._interval = config.ip_shuffle_interval or 900
        self._proxy_list = config.proxy_list or []
        self._current_index = config.current_proxy_index or 0
        self._last_shuffle = config.last_shuffle_time or 0.0
        self._current_session_id = config.session_id or generate_kilo_session_id()

        if self._proxy_list:
            self._current_proxy = self._proxy_list[
                self._current_index % len(self._proxy_list)
            ]

        if self._enabled and self._proxy_list:
            self._running = True
        self._initialized = True

    async def initialize(self):
        self._initialize_sync()

    async def _save_state(self):
        config = load_config()
        config.ip_shuffle_enabled = self._enabled
        config.ip_shuffle_interval = self._interval
        config.proxy_list = self._proxy_list
        config.current_proxy_index = self._current_index
        config.last_shuffle_time = self._last_shuffle
        config.session_id = self._current_session_id
        save_config(config)

    async def shuffle_now(self) -> Tuple[Optional[str], str]:
        async with self._lock:
            if not self._proxy_list:
                self._current_proxy = None
                self._current_session_id = generate_kilo_session_id()
                self._last_shuffle = time.time()
                await self._save_state()
                return None, self._current_session_id

            self._current_index = (self._current_index + 1) % len(self._proxy_list)
            self._current_proxy = self._proxy_list[self._current_index]
            self._current_session_id = generate_kilo_session_id()
            self._last_shuffle = time.time()
            await self._save_state()
            return self._current_proxy, self._current_session_id

    async def _shuffle_loop(self):
        while self._running:
            try:
                await asyncio.sleep(self._interval)
                if self._enabled and self._proxy_list:
                    await self.shuffle_now()
            except asyncio.CancelledError:
                break
            except Exception:
                pass

    def start_background_task(self) -> asyncio.Task:
        if self._enabled and self._proxy_list and not self._shuffler_task_running():
            self._running = True
            global _shuffler_task
            _shuffler_task = asyncio.create_task(self._shuffle_loop())
            return _shuffler_task
        return None

    def _shuffler_task_running(self) -> bool:
        global _shuffler_task
        return _shuffler_task is not None and not _shuffler_task.done()

    async def stop(self):
        self._running = False
        global _shuffler_task
        if _shuffler_task and not _shuffler_task.done():
            _shuffler_task.cancel()
            try:
                await _shuffler_task
            except asyncio.CancelledError:
                pass
        _shuffler_task = None

    def get_current_proxy(self) -> Optional[str]:
        return self._current_proxy if self._enabled else None

    def get_current_session_id(self) -> str:
        return self._current_session_id

    def is_enabled(self) -> bool:
        return self._enabled

    def get_interval(self) -> int:
        return self._interval

    def get_proxy_list(self) -> List[str]:
        return self._proxy_list.copy()

    def get_current_index(self) -> int:
        return self._current_index

    def get_last_shuffle_time(self) -> float:
        return self._last_shuffle

    async def set_enabled(self, enabled: bool):
        self._enabled = enabled
        if enabled and self._proxy_list:
            self._running = True
            self.start_background_task()
        else:
            await self.stop()
        await self._save_state()

    async def set_interval(self, seconds: int):
        self._interval = max(60, seconds)
        await self._save_state()

    async def add_proxy(self, proxy_url: str):
        normalized = self.normalize_proxy(proxy_url)
        if normalized not in self._proxy_list:
            self._proxy_list.append(normalized)
            if self._enabled and len(self._proxy_list) == 1:
                self._current_index = 0
                self._current_proxy = proxy_url
            await self._save_state()

    async def add_proxies(self, proxy_urls: List[str]):
        added = 0
        for url in proxy_urls:
            if url not in self._proxy_list:
                self._proxy_list.append(url)
                added += 1
        if added > 0:
            if self._enabled and len(self._proxy_list) == added:
                self._current_index = 0
                self._current_proxy = self._proxy_list[0]
            await self._save_state()
        return added

    async def load_proxies(self, source: str, timeout: int = 30) -> int:
        if source.startswith(("http://", "https://")):
            content = await self._fetch_url(source, timeout)
        else:
            content = self._read_file(source)

        proxies = self._parse_proxies(content)
        added = await self.add_proxies(proxies)
        return added

    def _read_file(self, file_path: str) -> str:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    async def _fetch_url(self, url: str, timeout: int = 30) -> str:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
            response = await client.get(url)
            response.raise_for_status()
        return response.text

    def _parse_proxies(self, content: str) -> List[str]:
        proxies = []
        for line in content.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                proxies.append(self.normalize_proxy(line))
        return proxies

    def normalize_proxy(self, proxy: str) -> str:
        """Normalize proxy URL to proper format.

        Handles formats:
        - ip:port:username:password -> http://username:password@ip:port
        - username:password@ip:port -> http://username:password@ip:port
        - ip:port (no auth) -> http://ip:port
        - Already properly formatted -> returns as-is
        """
        proxy = proxy.strip()

        if proxy.startswith(("http://", "https://", "socks4://", "socks5://")):
            return proxy

        parts = proxy.split(":")
        if len(parts) == 4:
            ip, port, username, password = parts
            return f"http://{username}:{password}@{ip}:{port}"
        elif len(parts) == 2:
            return f"http://{proxy}"

        return proxy

    async def load_proxies_from_file(self, file_path: str) -> int:
        return await self.load_proxies(file_path)

    async def load_proxies_from_url(self, url: str, timeout: int = 30) -> int:
        return await self.load_proxies(url, timeout)

    async def check_proxies(self, timeout: int = 30) -> Tuple[List[str], List[str]]:
        """Check all proxies and return working and broken ones."""
        import asyncio

        working = []
        broken = []

        async def check_single_proxy(proxy: str) -> bool:
            try:
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(timeout, connect=10.0), proxy=proxy
                ) as client:
                    response = await client.get("https://httpbin.org/ip")
                    return response.status_code == 200
            except Exception:
                return False

        tasks = [check_single_proxy(p) for p in self._proxy_list]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for proxy, result in zip(self._proxy_list, results):
            if isinstance(result, bool) and result:
                working.append(proxy)
            else:
                broken.append(proxy)

        return working, broken

    async def remove_proxies(self, proxies: List[str]) -> int:
        """Remove multiple proxies from the list."""
        removed = 0
        for proxy in proxies:
            if await self.remove_proxy(proxy):
                removed += 1
        return removed
        if proxy_url in self._proxy_list:
            self._proxy_list.remove(proxy_url)
            if (
                self._current_index >= len(self._proxy_list)
                and len(self._proxy_list) > 0
            ):
                self._current_index = 0
            if self._proxy_list:
                self._current_proxy = self._proxy_list[self._current_index]
            else:
                self._current_proxy = None
            await self._save_state()
            return True
        return False

    async def clear_proxies(self):
        self._proxy_list = []
        self._current_proxy = None
        self._current_index = 0
        await self.stop()
        await self._save_state()

    def get_status(self) -> dict:
        return {
            "enabled": self._enabled,
            "interval": self._interval,
            "proxy_count": len(self._proxy_list),
            "proxy_list": self._proxy_list,
            "current_index": self._current_index,
            "current_proxy": self._current_proxy,
            "current_session_id": self._current_session_id,
            "last_shuffle": self._last_shuffle,
            "running": self._running,
        }


def get_shuffler() -> IPShuffler:
    global _shuffler_instance
    if _shuffler_instance is None:
        _shuffler_instance = IPShuffler()
    _shuffler_instance._initialize_sync()
    return _shuffler_instance


async def init_shuffler():
    shuffler = get_shuffler()
    await shuffler.initialize()
    if shuffler.is_enabled():
        shuffler.start_background_task()


async def shutdown_shuffler():
    shuffler = get_shuffler()
    await shuffler.stop()
