# coding=utf-8
# Copyright 2026 The RAON Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FastAPI integration for RAON realtime duplex demo."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import JSONResponse
from starlette.websockets import WebSocketDisconnect

from ..protocol.messages import Frame, MessageKind

logger = logging.getLogger(__name__)


class RealtimeRuntimeManager:
    """Owns realtime model/runtime and enforces one active session."""

    _MAX_COMPLETED_SESSIONS = 256

    def __init__(self, *, model_path: str, session_kwargs: dict[str, Any] | None = None) -> None:
        self.model_path = model_path
        self.session_kwargs = dict(session_kwargs or {})
        self._active_session: Any | None = None
        self._active_session_id: str | None = None
        self._active_session_connected = False
        self._completed_sessions: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    def active_session_id(self) -> str | None:
        with self._lock:
            return self._active_session_id

    def force_finish_active_session(self, reason: str = "force_restart") -> dict[str, Any] | None:
        """Force-close the currently active session, if any."""
        with self._lock:
            active_session_id = self._active_session_id
        if active_session_id is None:
            return None
        return self.finish_session(active_session_id, reason=reason)

    def _resolve_session_factory(self):
        # Deferred import to keep startup failures isolated to actual usage.
        from ..runtime import session as session_mod

        for name in ("create_session", "build_session", "make_session"):
            factory = getattr(session_mod, name, None)
            if callable(factory):
                return factory

        for name in (
            "RealtimeSession",
            "RaonRealtimeSession",
            "DuplexRealtimeSession",
            "LocalRealtimeSession",
            "Session",
        ):
            cls = getattr(session_mod, name, None)
            if cls is None:
                continue

            def _ctor(_cls=cls, **kwargs):
                return _cls(**kwargs)

            return _ctor

        raise RuntimeError("No session factory/class found in demo.realtime.runtime.session")

    def preload_runtime(self) -> None:
        """Eagerly initialize model runtime so first Start is not blocked by model load."""
        # Deferred import to keep module import lightweight.
        from ..runtime import session as session_mod

        get_runtime = getattr(session_mod, "get_runtime", None)
        if not callable(get_runtime):
            logger.warning("runtime preload skipped: session.get_runtime not found")
            return

        runtime_payload = dict((self.session_kwargs or {}).get("runtime") or {})
        try:
            get_runtime(
                model_path=self.model_path,
                dtype=str(runtime_payload.get("dtype", "bfloat16")),
                mem_fraction_static=float(runtime_payload.get("mem_fraction_static", 0.88)),
                disable_cuda_graph=bool(runtime_payload.get("disable_cuda_graph", False)),
                max_running_requests=runtime_payload.get("max_running_requests"),
                max_total_tokens=runtime_payload.get("max_total_tokens"),
                max_prefill_tokens=runtime_payload.get("max_prefill_tokens"),
                chunked_prefill_size=runtime_payload.get("chunked_prefill_size"),
                max_allocated_req_pool_indices=int(runtime_payload.get("max_allocated_req_pool_indices", 32)),
                gpu_id=int(runtime_payload.get("gpu_id", 0)),
                compile_audio_modules=str(runtime_payload.get("compile_audio_modules", True)).lower() not in ("false", "0", "no"),
                compile_max_sequence_length=int(runtime_payload.get("compile_max_sequence_length", 8192)),
            )
            logger.info("runtime preload complete model_path=%s", self.model_path)
        except Exception:
            logger.exception("runtime preload failed model_path=%s", self.model_path)
            raise

    def try_start_session(self, *, session_id: str, query: dict[str, Any] | None = None) -> Any | None:
        with self._lock:
            if self._active_session is not None:
                return None
            factory = self._resolve_session_factory()
            kwargs = dict(self.session_kwargs)
            kwargs.update(query or {})
            kwargs.setdefault("session_id", session_id)
            kwargs.setdefault("model_path", self.model_path)
            session = factory(**kwargs)
            self._active_session = session
            self._active_session_id = session_id
            self._active_session_connected = False
            return session

    def reserve_session(self, payload: dict[str, Any] | None = None) -> str:
        session_id = str((payload or {}).get("session_id") or uuid.uuid4())
        session = self.try_start_session(session_id=session_id, query=payload)
        if session is None:
            raise RuntimeError("session already active")
        return session_id

    def attach_session(self, *, session_id: str, query: dict[str, Any] | None = None) -> Any:
        with self._lock:
            if self._active_session is not None and self._active_session_id == session_id:
                if self._active_session_connected:
                    raise RuntimeError("session already connected")
                self._active_session_connected = True
                return self._active_session
            if self._active_session is not None:
                raise RuntimeError("session already active")

        session = self.try_start_session(session_id=session_id, query=query)
        if session is None:
            raise RuntimeError("session already active")
        with self._lock:
            self._active_session_connected = True
        return session

    async def _call_session(self, session: Any, names: tuple[str, ...], *args: Any, **kwargs: Any) -> Any:
        for name in names:
            fn = getattr(session, name, None)
            if callable(fn):
                result = fn(*args, **kwargs)
                if asyncio.iscoroutine(result):
                    return await result
                return result
        raise RuntimeError(f"Session does not implement any of {names}")

    async def on_start(self, session: Any) -> list[Frame]:
        result = await self._call_session(session, ("start", "on_start", "initialize"))
        if result is None:
            return [Frame.ready()]
        if isinstance(result, Frame):
            return [result]
        if isinstance(result, list):
            return result
        return [Frame.ready()]

    async def on_audio(self, session: Any, pcm: Any) -> list[Frame]:
        result = await self._call_session(
            session,
            ("handle_audio_frame", "feed_audio", "feed_and_decode", "on_audio"),
            pcm,
        )
        if result is None:
            return []
        if isinstance(result, Frame):
            return [result]
        if isinstance(result, list):
            return result
        return []

    async def on_close(self, session: Any, reason: str) -> list[Frame]:
        with contextlib.suppress(Exception):
            result = await self._call_session(
                session,
                ("request_close", "on_close", "finish"),
                reason,
            )
            if isinstance(result, Frame):
                return [result]
            if isinstance(result, list):
                return result
        return [Frame.close(reason=reason)]

    def finish_session(self, session_id: str, reason: str = "client_finish") -> dict[str, Any]:
        with self._lock:
            if session_id in self._completed_sessions:
                return self._completed_sessions[session_id]
            if self._active_session_id != session_id or self._active_session is None:
                raise KeyError(session_id)
            session = self._active_session
            self._active_session = None
            self._active_session_id = None
            self._active_session_connected = False

        payload: dict[str, Any] = {}
        try:
            finish_fn = getattr(session, "finish", None)
            if callable(finish_fn):
                result = finish_fn(reason)
                if isinstance(result, dict):
                    payload = result
        finally:
            meta_fn = getattr(session, "artifact_response", None)
            if callable(meta_fn):
                meta = meta_fn()
                if isinstance(meta, dict):
                    payload = meta
            close_fn = getattr(session, "close", None)
            if callable(close_fn):
                with contextlib.suppress(Exception):
                    close_fn()

        payload.setdefault("session_id", session_id)
        payload.setdefault("close_reason", reason)
        with self._lock:
            self._completed_sessions[session_id] = payload
            while len(self._completed_sessions) > self._MAX_COMPLETED_SESSIONS:
                oldest_session_id = next(iter(self._completed_sessions))
                self._completed_sessions.pop(oldest_session_id, None)
        return payload

    async def on_ping(self, session: Any, payload: bytes) -> list[Frame]:
        with contextlib.suppress(Exception):
            result = await self._call_session(session, ("on_ping",), payload)
            if isinstance(result, Frame):
                return [result]
            if isinstance(result, list):
                return result
        return [Frame(kind=MessageKind.PONG, payload=payload)]


_singleton_lock = threading.Lock()
_singleton_manager: RealtimeRuntimeManager | None = None


def get_runtime_manager(*, model_path: str, session_kwargs: dict[str, Any] | None = None) -> RealtimeRuntimeManager:
    """Return process-wide singleton runtime manager."""
    global _singleton_manager
    if _singleton_manager is not None:
        return _singleton_manager
    with _singleton_lock:
        if _singleton_manager is None:
            _singleton_manager = RealtimeRuntimeManager(model_path=model_path, session_kwargs=session_kwargs)
        return _singleton_manager


def mount_realtime_websocket(app: FastAPI, manager: RealtimeRuntimeManager, *, path: str = "/realtime/ws") -> None:
    """Mount websocket endpoint used by the browser realtime client."""

    @app.websocket(path)
    async def websocket_duplex(websocket: WebSocket) -> None:
        await websocket.accept()
        query = dict(websocket.query_params)
        session_id = str(query.get("session_id") or uuid.uuid4())
        try:
            session = manager.attach_session(session_id=session_id, query=query)
        except RuntimeError as exc:
            await websocket.send_bytes(Frame.error("session already active").encode())
            await websocket.send_bytes(Frame.close(reason=str(exc) or "busy").encode())
            await websocket.close()
            return

        logger.info("realtime session started id=%s", session_id)
        close_reason = "client_disconnect"
        try:
            for frame in await manager.on_start(session):
                await websocket.send_bytes(frame.encode())

            while True:
                raw = await websocket.receive_bytes()
                incoming = Frame.decode(raw)

                if incoming.kind == MessageKind.AUDIO:
                    outgoing = await manager.on_audio(session, incoming.audio_samples())
                elif incoming.kind == MessageKind.CLOSE:
                    close_reason = incoming.text_content() or "client_finish"
                    outgoing = await manager.on_close(session, close_reason)
                    for frame in outgoing:
                        await websocket.send_bytes(frame.encode())
                    break
                elif incoming.kind == MessageKind.PING:
                    outgoing = await manager.on_ping(session, incoming.payload)
                else:
                    outgoing = [Frame.error(f"unsupported frame kind: {int(incoming.kind)}")]

                for frame in outgoing:
                    await websocket.send_bytes(frame.encode())
        except WebSocketDisconnect:
            close_reason = "client_disconnect"
        except Exception as exc:  # pragma: no cover - runtime safety net
            close_reason = "internal_error"
            logger.exception("websocket session failed id=%s err=%s", session_id, exc)
            with contextlib.suppress(Exception):
                await websocket.send_bytes(Frame.error(str(exc)).encode())
                await websocket.send_bytes(Frame.close(reason=close_reason).encode())
        finally:
            with contextlib.suppress(Exception):
                if websocket.client_state.name == "CONNECTED":
                    await websocket.close()
            with contextlib.suppress(Exception):
                await manager.on_close(session, close_reason)
            with contextlib.suppress(Exception):
                manager.finish_session(session_id, reason=close_reason)
            logger.info("realtime session finished id=%s reason=%s", session_id, close_reason)


def create_fastapi_app(
    *,
    model_path: str,
    session_kwargs: dict[str, Any] | None = None,
    ws_path: str = "/realtime/ws",
) -> FastAPI:
    """Create FastAPI app with mounted realtime websocket endpoint."""
    app = FastAPI(title="RAON Realtime Duplex")
    manager = get_runtime_manager(model_path=model_path, session_kwargs=session_kwargs)
    mount_realtime_websocket(app, manager, path=ws_path)

    @app.on_event("startup")
    async def _startup_preload_runtime() -> None:
        manager.preload_runtime()

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "active_session_id": manager.active_session_id(),
            "ws_path": ws_path,
        }

    @app.post("/realtime/session/start")
    async def start_session(payload: dict[str, Any]) -> JSONResponse:
        force_restart = bool((payload or {}).get("force_restart", False))
        try:
            session_id = manager.reserve_session(payload)
        except RuntimeError as exc:
            if force_restart:
                with contextlib.suppress(Exception):
                    manager.force_finish_active_session(reason="force_restart")
                try:
                    session_id = manager.reserve_session(payload)
                except RuntimeError as retry_exc:
                    raise HTTPException(status_code=409, detail=str(retry_exc)) from retry_exc
            else:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
        return JSONResponse({"session_id": session_id, "ws_path": ws_path})

    @app.post("/realtime/session/finish")
    async def finish_session(payload: dict[str, Any]) -> JSONResponse:
        session_id = str(payload.get("session_id") or "").strip()
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id is required")
        reason = str(payload.get("reason") or "client_finish")
        try:
            result = manager.finish_session(session_id, reason=reason)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"unknown session_id: {session_id}") from exc
        return JSONResponse(result)

    return app


def mount_gradio_app(fastapi_app: FastAPI, blocks: Any, *, path: str = "/") -> FastAPI:
    """Mount Gradio Blocks app into an existing FastAPI app."""
    import gradio as gr

    return gr.mount_gradio_app(fastapi_app, blocks, path=path)
