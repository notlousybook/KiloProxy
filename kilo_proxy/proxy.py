"""Core proxy logic for kilo-proxy."""

import json
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import httpx
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from kilo_proxy.config import load_config, generate_kilo_session_id
from kilo_proxy.ip_shuffler import get_shuffler

BASE_URL = "https://api.kilo.ai/api/openrouter"

DEFAULT_HEADERS = {
    "Content-Type": "application/json",
    "HTTP-Referer": "https://kilocode.ai",
    "User-Agent": "opencode-kilo-provider",
    "X-KILOCODE-EDITORNAME": "Kilo CLI",
    "X-KILOCODE-FEATURE": "cli",
    "X-Title": "Kilo Code",
}


def get_headers(
    auth_token: Optional[str] = None, extra_headers: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    config = load_config()
    token = auth_token or config.auth_token or "anonymous"
    shuffler = get_shuffler()
    if shuffler.is_enabled():
        session_id = shuffler.get_current_session_id()
    else:
        session_id = config.session_id or generate_kilo_session_id()
    headers = DEFAULT_HEADERS.copy()
    headers["Authorization"] = f"Bearer {token}"
    headers["X-KILOCODE-TASKID"] = session_id
    if extra_headers:
        headers.update(extra_headers)
    return headers


def transform_request_body(body: Dict[str, Any]) -> Dict[str, Any]:
    return body


def transform_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return messages


async def create_streaming_generator(
    url: str, headers: Dict[str, str], body: Dict[str, Any]
) -> AsyncIterator[bytes]:
    shuffler = get_shuffler()
    proxy = shuffler.get_current_proxy()
    if proxy:
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(300.0, connect=30.0), proxy=proxy
        )
    else:
        client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=30.0))
    try:
        async with client.stream("POST", url, headers=headers, json=body) as response:
            if response.status_code != 200:
                error_body = await response.aread()
                yield f"data: {json.dumps({'error': error_body.decode()})}\n\n".encode()
                return

            async for chunk in response.aiter_bytes(chunk_size=32):
                if chunk:
                    yield chunk
    except httpx.ProxyError as e:
        logger.error(f"Proxy error in streaming (check your proxy configuration): {e}")
        yield f"data: {json.dumps({'error': f'Proxy error: {str(e)}. Check your proxy configuration.'})}\n\n".encode()
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n".encode()
    finally:
        await client.aclose()


class ProxyClient:
    def __init__(
        self,
        auth_token: Optional[str] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ):
        self.auth_token = auth_token
        self.extra_headers = extra_headers
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        shuffler = get_shuffler()
        proxy = shuffler.get_current_proxy()
        if proxy:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(300.0, connect=30.0), proxy=proxy
            )
        else:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=30.0))
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()

    async def get_models(self) -> Dict[str, Any]:
        headers = get_headers(self.auth_token, self.extra_headers)
        url = f"{BASE_URL}/models"
        response = await self._client.get(url, headers=headers)
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Failed to fetch models: {response.text}",
            )
        return response.json()

    async def get_model(self, model_id: str) -> Dict[str, Any]:
        headers = get_headers(self.auth_token, self.extra_headers)
        url = f"{BASE_URL}/models/{model_id}"
        response = await self._client.get(url, headers=headers)
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Failed to fetch model: {response.text}",
            )
        return response.json()

    async def chat_completions(
        self, body: Dict[str, Any]
    ) -> Union[Dict[str, Any], StreamingResponse]:
        headers = get_headers(self.auth_token, self.extra_headers)
        url = f"{BASE_URL}/chat/completions"
        transformed_body = transform_request_body(body)

        stream = body.get("stream", False)

        if stream:
            return StreamingResponse(
                create_streaming_generator(url, headers, transformed_body),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            response = await self._client.post(
                url, headers=headers, json=transformed_body
            )
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Chat completion failed: {response.text}",
                )
            return response.json()

    async def completions(
        self, body: Dict[str, Any]
    ) -> Union[Dict[str, Any], StreamingResponse]:
        headers = get_headers(self.auth_token, self.extra_headers)
        url = f"{BASE_URL}/completions"
        transformed_body = transform_request_body(body)

        stream = body.get("stream", False)

        if stream:
            return StreamingResponse(
                create_streaming_generator(url, headers, transformed_body),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            response = await self._client.post(
                url, headers=headers, json=transformed_body
            )
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Completion failed: {response.text}",
                )
            return response.json()

    async def embeddings(self, body: Dict[str, Any]) -> Dict[str, Any]:
        headers = get_headers(self.auth_token, self.extra_headers)
        url = f"{BASE_URL}/embeddings"
        response = await self._client.post(url, headers=headers, json=body)
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Embedding failed: {response.text}",
            )
        return response.json()


async def proxy_models() -> Dict[str, Any]:
    async with ProxyClient() as client:
        return await client.get_models()


async def proxy_model(model_id: str) -> Dict[str, Any]:
    async with ProxyClient() as client:
        return await client.get_model(model_id)


async def proxy_chat_completions(
    body: Dict[str, Any],
    auth_token: Optional[str] = None,
    extra_headers: Optional[Dict[str, str]] = None,
) -> Union[Dict[str, Any], StreamingResponse]:
    stream = body.get("stream", False)
    headers = get_headers(auth_token, extra_headers)
    if stream:
        url = f"{BASE_URL}/chat/completions"
        transformed_body = transform_request_body(body)
        return StreamingResponse(
            create_streaming_generator(url, headers, transformed_body),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        try:
            async with ProxyClient(auth_token, extra_headers) as client:
                return await client.chat_completions(body)
        except httpx.ProxyError as e:
            logger.error(f"Proxy error (check your proxy configuration): {e}")
            raise HTTPException(
                status_code=502,
                detail=f"Proxy error: {str(e)}. Check your proxy configuration.",
            )
        except httpx.RequestError as e:
            logger.error(f"Request error: {e}")
            raise HTTPException(
                status_code=502,
                detail=f"Request to Kilo API failed: {str(e)}",
            )


async def proxy_completions(
    body: Dict[str, Any],
    auth_token: Optional[str] = None,
    extra_headers: Optional[Dict[str, str]] = None,
) -> Union[Dict[str, Any], StreamingResponse]:
    stream = body.get("stream", False)
    headers = get_headers(auth_token, extra_headers)
    if stream:
        url = f"{BASE_URL}/completions"
        transformed_body = transform_request_body(body)
        return StreamingResponse(
            create_streaming_generator(url, headers, transformed_body),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        async with ProxyClient(auth_token, extra_headers) as client:
            return await client.completions(body)


async def proxy_embeddings(
    body: Dict[str, Any],
    auth_token: Optional[str] = None,
    extra_headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    async with ProxyClient(auth_token, extra_headers) as client:
        return await client.embeddings(body)
