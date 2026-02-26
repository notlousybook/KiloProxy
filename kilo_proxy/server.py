"""FastAPI server for kilo-proxy."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from kilo_proxy import __version__
from kilo_proxy.config import load_config
from kilo_proxy.ip_shuffler import init_shuffler, shutdown_shuffler
from kilo_proxy.proxy import (
    ProxyClient,
    proxy_chat_completions,
    proxy_completions,
    proxy_embeddings,
)

LOG_DIR = Path.home() / ".kilo-proxy" / "logs"
LOG_FILE = LOG_DIR / "server.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("kilo-proxy")

app = FastAPI(
    title="Kilo Proxy",
    description="Fully OpenAI-compatible API proxy for Kilo",
    version=__version__,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    logger.info("Starting Kilo Proxy server")
    await init_shuffler()


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Kilo Proxy server")
    await shutdown_shuffler()


class ChatMessage(BaseModel):
    role: str
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


class StreamOptions(BaseModel):
    include_usage: Optional[bool] = None
    continuous_usage_stats: Optional[bool] = None


class ReasoningConfig(BaseModel):
    effort: Optional[str] = None
    max_tokens: Optional[int] = None


class ResponseFormat(BaseModel):
    type: str = "text"
    json_schema: Optional[Dict[str, Any]] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    functions: Optional[List[Dict[str, Any]]] = None
    function_call: Optional[Union[str, Dict[str, Any]]] = None
    response_format: Optional[ResponseFormat] = None
    user: Optional[str] = None
    n: Optional[int] = 1
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    seed: Optional[int] = None
    logit_bias: Optional[Dict[str, float]] = None
    reasoning: Optional[ReasoningConfig] = None
    include_reasoning: Optional[bool] = None

    model_config = {"extra": "allow"}


class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    n: Optional[int] = 1
    logprobs: Optional[int] = None
    echo: Optional[bool] = None
    seed: Optional[int] = None
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None

    model_config = {"extra": "allow"}


class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, List[str]]
    encoding_format: Optional[str] = "float"
    dimensions: Optional[int] = None
    user: Optional[str] = None

    model_config = {"extra": "allow"}


def get_auth_token(authorization: Optional[str] = None) -> Optional[str]:
    if authorization and authorization.startswith("Bearer "):
        return authorization[7:]
    return None


def get_extra_headers(request: Request) -> Dict[str, str]:
    """Extract extra headers like x-kilocode-mode from request."""
    extra = {}
    kilo_mode = request.headers.get("x-kilocode-mode")
    if kilo_mode:
        extra["x-kilocode-mode"] = kilo_mode
    return extra


def is_free_model(model_id: str, model_name: str = "") -> bool:
    """Check if model is free.

    Free models are:
    1. Models with :free in ID
    2. Models without / in ID (stealth free models)
    3. Models with (free) in name
    """
    return ":free" in model_id or "/" not in model_id or "(free)" in model_name.lower()


@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": __version__}


@app.get("/v1/models")
async def list_models(request: Request, authorization: Optional[str] = Header(None)):
    auth_token = get_auth_token(authorization)
    extra_headers = get_extra_headers(request)
    config = load_config()
    async with ProxyClient(auth_token, extra_headers) as client:
        result = await client.get_models()
        if config.broke and "data" in result:
            result["data"] = [
                m
                for m in result["data"]
                if is_free_model(m.get("id", ""), m.get("name", ""))
            ]
        return result


@app.get("/v1/models/{model_id}")
async def get_model(
    model_id: str, request: Request, authorization: Optional[str] = Header(None)
):
    auth_token = get_auth_token(authorization)
    extra_headers = get_extra_headers(request)
    async with ProxyClient(auth_token, extra_headers) as client:
        return await client.get_model(model_id)


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    raw_request: Request,
    authorization: Optional[str] = Header(None),
):
    auth_token = get_auth_token(authorization)
    extra_headers = get_extra_headers(raw_request)
    body = request.model_dump(exclude_none=True, exclude_unset=True)
    return await proxy_chat_completions(body, auth_token, extra_headers)


@app.post("/v1/completions")
async def completions(
    request: CompletionRequest,
    raw_request: Request,
    authorization: Optional[str] = Header(None),
):
    auth_token = get_auth_token(authorization)
    extra_headers = get_extra_headers(raw_request)
    body = request.model_dump(exclude_none=True, exclude_unset=True)
    return await proxy_completions(body, auth_token, extra_headers)


@app.post("/v1/embeddings")
async def embeddings(
    request: EmbeddingRequest,
    raw_request: Request,
    authorization: Optional[str] = Header(None),
):
    auth_token = get_auth_token(authorization)
    extra_headers = get_extra_headers(raw_request)
    body = request.model_dump(exclude_none=True, exclude_unset=True)
    return await proxy_embeddings(body, auth_token, extra_headers)


@app.get("/v1/engines")
async def list_engines(request: Request, authorization: Optional[str] = Header(None)):
    auth_token = get_auth_token(authorization)
    extra_headers = get_extra_headers(request)
    config = load_config()
    async with ProxyClient(auth_token, extra_headers) as client:
        models_data = await client.get_models()
        if "data" in models_data:
            if config.broke:
                models_data["data"] = [
                    m
                    for m in models_data["data"]
                    if is_free_model(m.get("id", ""), m.get("name", ""))
                ]
            return {"data": models_data["data"], "object": "list"}
        return models_data


@app.get("/v1/engines/{engine_id}")
async def get_engine(
    engine_id: str, request: Request, authorization: Optional[str] = Header(None)
):
    auth_token = get_auth_token(authorization)
    extra_headers = get_extra_headers(request)
    async with ProxyClient(auth_token, extra_headers) as client:
        return await client.get_model(engine_id)


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_catch_all(
    path: str,
    request: Request,
    authorization: Optional[str] = Header(None),
):
    logger.info(f"Proxy catch-all: {request.method} /v1/{path}")
    auth_token = get_auth_token(authorization)
    body = await request.body()
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)

    extra_headers = get_extra_headers(request)

    is_streaming = request.headers.get("accept") == "text/event-stream" or (
        body and b'"stream":true' in body
    )

    async with ProxyClient(auth_token, extra_headers) as client:
        url = f"https://api.kilo.ai/api/openrouter/{path}"
        method = request.method

        if body:
            try:
                body_json = json.loads(body)
            except json.JSONDecodeError:
                body_json = None
        else:
            body_json = None

        if is_streaming and method == "POST":
            from kilo_proxy.proxy import create_streaming_generator

            return StreamingResponse(
                create_streaming_generator(
                    url, client._client.headers, body_json or {}
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        try:
            if method == "GET":
                response = await client._client.get(url, headers=client._client.headers)
            elif method == "POST":
                response = await client._client.post(
                    url, headers=client._client.headers, json=body_json
                )
            elif method == "PUT":
                response = await client._client.put(
                    url, headers=client._client.headers, json=body_json
                )
            elif method == "DELETE":
                response = await client._client.delete(
                    url, headers=client._client.headers
                )
            elif method == "PATCH":
                response = await client._client.patch(
                    url, headers=client._client.headers, json=body_json
                )
            else:
                raise HTTPException(status_code=405, detail="Method not allowed")
        except Exception as e:
            logger.error(f"Proxy error: {e}")
            raise

        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            try:
                return JSONResponse(
                    content=response.json(), status_code=response.status_code
                )
            except Exception:
                return JSONResponse(
                    content={"error": response.text}, status_code=response.status_code
                )
        else:
            return JSONResponse(content=response.text, status_code=response.status_code)
