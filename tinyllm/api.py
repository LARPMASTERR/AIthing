from __future__ import annotations

import argparse
import asyncio
import os
import time
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from tinyllm.config import ProjectConfig
from tinyllm.engine import ChatEngine
from tinyllm.visualizer import embedding_layout


class APIMessage(BaseModel):
    role: str
    content: str

    @field_validator("role")
    @classmethod
    def validate_role(cls, value: str) -> str:
        if value not in {"system", "user", "assistant"}:
            raise ValueError("role must be system, user, or assistant")
        return value

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("content cannot be empty")
        return value


class ChatRequest(BaseModel):
    model: str = "tiny-convo-32m"
    messages: list[APIMessage] = Field(min_length=1)
    max_tokens: int = Field(default=128, ge=1, le=512)
    temperature: float = Field(default=0.8, ge=0, le=2)
    top_p: float = Field(default=0.9, gt=0, le=1)
    retrieval: bool = False
    stream: bool = False


def next_event(iterator):
    try:
        return next(iterator)
    except StopIteration:
        return None


def create_app(engine: ChatEngine, config: ProjectConfig | None = None) -> FastAPI:
    app = FastAPI(title="Tiny Conversational LLM", version="0.1.0")
    static_dir = Path(__file__).parent / "static"
    generation_lock = asyncio.Lock()
    allowed_origins = [origin.strip() for origin in os.getenv("TINYLLM_ALLOWED_ORIGINS", "").split(",") if origin.strip()]
    if allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    def visualizer() -> RedirectResponse:
        return RedirectResponse("/static/index.html")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "checkpoint": engine.metadata}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatRequest) -> dict:
        if request.stream:
            raise HTTPException(status_code=400, detail="streaming is not supported")
        async with generation_lock:
            text, pages = await asyncio.to_thread(
                engine.complete,
                [message.model_dump() for message in request.messages],
                request.max_tokens,
                request.temperature,
                request.top_p,
                request.retrieval,
            )
        prompt_tokens = sum(len(engine.tokenizer.encode(message.content)) for message in request.messages)
        completion_tokens = len(engine.tokenizer.encode(text))
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "sources": [{"title": page.title, "url": page.url} for page in pages],
        }

    @app.get("/visualizer/layout")
    async def visualizer_layout() -> FileResponse:
        if config is None:
            raise HTTPException(status_code=503, detail="visualizer layout is unavailable")
        path = await asyncio.to_thread(
            embedding_layout,
            engine.model,
            engine.tokenizer,
            engine.metadata,
            config.paths.tokenizer,
        )
        return FileResponse(path, media_type="application/json")

    @app.websocket("/ws/visualize")
    async def visualize_socket(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            request = ChatRequest.model_validate(await websocket.receive_json())
            async with generation_lock:
                iterator = engine.traced_events(
                    [message.model_dump() for message in request.messages],
                    request.max_tokens,
                    request.temperature,
                    request.top_p,
                    request.retrieval,
                )
                while event := await asyncio.to_thread(next_event, iterator):
                    await websocket.send_json(event)
        except WebSocketDisconnect:
            return
        except Exception as error:
            try:
                await websocket.send_json({"type": "error", "message": str(error)})
            except WebSocketDisconnect:
                pass
        finally:
            try:
                await websocket.close()
            except RuntimeError:
                pass

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve TinyLM's OpenAI-style API")
    parser.add_argument("--config", default="configs/quick.json")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    config = ProjectConfig.load(args.config)
    engine = ChatEngine.load(config)
    uvicorn.run(create_app(engine, config), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
