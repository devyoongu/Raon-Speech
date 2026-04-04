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

"""Single-session realtime duplex orchestration."""

from __future__ import annotations

import contextlib
import inspect
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from .artifacts import SessionArtifacts
from .config import SessionConfig
from ..protocol.messages import Frame
from .prompt_map import resolve_prompt

if TYPE_CHECKING:
    from transformers import Qwen2TokenizerFast

    from ..backends.sglang_backend import SGLangRaonModel

logger = logging.getLogger(__name__)

INTERNAL_ERROR = "internal_error"
OVERLOADED_BACKLOG = "overloaded_backlog"
CLIENT_FINISH = "client_finish"

_FATAL_CUDA_ERROR_MARKERS = (
    "cuda error",
    "device-side assert",
    "acceleratorerror",
    "cublas",
    "cudnn",
    "illegal memory access",
    "operation not supported on global/shared address space",
)


def is_fatal_cuda_error(exc: BaseException) -> bool:
    """Return True when an exception indicates CUDA context corruption."""
    message = str(exc).lower()
    return any(marker in message for marker in _FATAL_CUDA_ERROR_MARKERS)


def _is_recoverable_decode_error(exc: BaseException) -> bool:
    if isinstance(exc, AssertionError):
        return True
    if isinstance(exc, RuntimeError):
        lowered = str(exc).lower()
        return "sil-no-audio" in lowered and "assert" in lowered
    return False


def _safe_int_attr(obj: object, name: str, default: int | None = None) -> int | None:
    value = getattr(obj, name, None)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _resolve_runtime_speaker_audio_path(path_value: str) -> Path:
    """Resolve speaker reference audio path for realtime runtime."""
    raw = Path(path_value).expanduser()
    if raw.is_absolute():
        return raw

    repo_root = Path(__file__).resolve().parents[3]
    candidates = [raw, repo_root / raw]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def _compute_runtime_speaker_embeds(model: Any, speaker_audio_path: str) -> tuple[Any, str]:
    """Load speaker reference audio and compute speaker embeddings."""
    import torch

    from raon.utils.audio_io import load_audio as _load_audio

    resolved_path = _resolve_runtime_speaker_audio_path(speaker_audio_path)
    cache_key = str(resolved_path)
    cache = getattr(model, "_realtime_speaker_embed_cache", None)
    if isinstance(cache, dict) and cache_key in cache:
        return cache[cache_key], cache_key

    speaker_audio, _ = _load_audio(
        resolved_path,
        target_sr=int(getattr(model, "sampling_rate", 24000)),
        mono=True,
        device=getattr(model, "device", "cuda"),
        dtype=getattr(model, "dtype", torch.float32),
    )
    with torch.inference_mode():
        speaker_embeds = model._compute_speaker_embeds(speaker_audio, None)

    if not isinstance(cache, dict):
        cache = {}
        setattr(model, "_realtime_speaker_embed_cache", cache)
    cache[cache_key] = speaker_embeds
    return speaker_embeds, cache_key


def _prepare_runtime_model(
    model: Any,
    *,
    compile_audio_modules: bool = True,
    compile_max_sequence_length: int = 8192,
) -> None:
    """Apply fd-demo runtime optimizations once per model instance."""
    with contextlib.suppress(Exception):
        import torch

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    prepared = bool(getattr(model, "_realtime_runtime_prepared", False))
    if not prepared:
        inner_model = None
        get_model = getattr(model, "get_model", None)
        if callable(get_model):
            try:
                inner_model = get_model()
            except Exception:
                inner_model = None

        # Align with fd-demo runtime defaults.
        with contextlib.suppress(Exception):
            model.use_duplex_end_pad = True
        if inner_model is not None:
            with contextlib.suppress(Exception):
                inner_model.use_duplex_end_pad = True

        if inner_model is not None and hasattr(inner_model, "_set_attention_implementation"):
            try:
                inner_model._set_attention_implementation("flash_attention_2")
                logger.info("realtime runtime: attention_implementation=flash_attention_2")
            except Exception:
                try:
                    inner_model._set_attention_implementation("sdpa")
                    logger.info("realtime runtime: attention_implementation=sdpa")
                except Exception as exc:
                    logger.warning("realtime runtime: failed to set attention implementation: %s", exc)

        setattr(model, "_realtime_runtime_prepared", True)

    if hasattr(model, "start_concurrent_audio_decoder"):
        model.start_concurrent_audio_decoder()

    if not compile_audio_modules:
        logger.info("realtime runtime: skipping compile_audio_modules; compile_audio_modules is disabled")
        return

    if not hasattr(model, "compile_audio_modules"):
        return

    if bool(getattr(model, "_realtime_audio_modules_compiled", False)):
        return

    try:
        logger.info(
            "realtime runtime: compiling audio modules (torch.compile), max_sequence_length=%d",
            compile_max_sequence_length,
        )
        model.compile_audio_modules(duplex=True, max_sequence_length=compile_max_sequence_length)
        setattr(model, "_realtime_audio_modules_compiled", True)
        logger.info("realtime runtime: compile_audio_modules completed")
    except Exception as exc:
        logger.warning("realtime runtime: compile_audio_modules failed (continuing without): %s", exc)


def _encode_single_token(tokenizer: object, token_text: str) -> int | None:
    try:
        token_ids = tokenizer.encode(token_text, add_special_tokens=False)
    except TypeError:
        try:
            token_ids = tokenizer.encode(token_text)
        except Exception:
            return None
    except Exception:
        return None
    if len(token_ids) != 1:
        return None
    return int(token_ids[0])


def _sanitize_prompt_tokens(
    tokens: list[object],
    *,
    text_vocab_size: int,
    fallback_token_id: int,
) -> tuple[list[int], int]:
    """Ensure prompt tokens stay in the model text vocab range."""
    if text_vocab_size <= 0:
        normalized = [int(tok) for tok in tokens] if tokens else [0]
        return normalized, 0

    safe_fallback = int(fallback_token_id)
    if safe_fallback < 0 or safe_fallback >= text_vocab_size:
        safe_fallback = max(0, text_vocab_size - 1)

    normalized: list[int] = []
    replaced = 0
    for raw in tokens:
        try:
            token = int(raw)
        except (TypeError, ValueError):
            token = safe_fallback
            replaced += 1
        if token < 0 or token >= text_vocab_size:
            token = safe_fallback
            replaced += 1
        normalized.append(token)

    if not normalized:
        normalized = [safe_fallback]
        replaced += 1

    return normalized, replaced


def _resolve_ignored_token_ids(
    model: Any,
    tokenizer: object,
) -> tuple[set[int], int | None, int | None]:
    epad_id = _encode_single_token(tokenizer, "<|fim_middle|>")
    if epad_id is None:
        epad_id = _safe_int_attr(model, "duplex_end_pad_token_id", 0) or 0

    dpad_id = _encode_single_token(tokenizer, "<|fim_prefix|>")
    if dpad_id is None:
        dpad_id = _safe_int_attr(model, "duplex_pad_token_id", 0) or 0

    sil_id = _encode_single_token(tokenizer, "<|audio_output_sil|>")
    if sil_id is None:
        sil_id = _safe_int_attr(model, "duplex_sil_token_id", None)

    ignored: set[int] = {epad_id, dpad_id}
    for attr in (
        "audio_output_token_id",
        "audio_input_token_id",
        "im_start_token_id",
        "audio_start_token_id",
        "speaker_token_id",
        "eos_token_id",
    ):
        val = _safe_int_attr(model, attr, None)
        if val is not None:
            ignored.add(val)

    if sil_id is not None:
        ignored.add(sil_id)

    return ignored, sil_id, _safe_int_attr(model, "audio_start_token_id", None)


def _decode_text_tokens(
    token_ids: list[int],
    tokenizer: object,
    *,
    text_vocab_size: int,
    ignored_token_ids: set[int],
    sil_token_id: int | None,
    audio_start_token_id: int | None,
) -> str:
    pieces: list[str] = []
    text_buffer: list[int] = []

    def _flush() -> None:
        if not text_buffer:
            return
        pieces.append(tokenizer.decode(text_buffer, skip_special_tokens=False))
        text_buffer.clear()

    for token_id in token_ids:
        if (
            sil_token_id is not None
            and token_id == sil_token_id
            or audio_start_token_id is not None
            and token_id == audio_start_token_id
        ):
            _flush()
            pieces.append("\n")
            continue
        if token_id in ignored_token_ids:
            _flush()
            continue
        if token_id < text_vocab_size:
            text_buffer.append(token_id)

    _flush()
    return "".join(pieces)


@dataclass(slots=True)
class RealtimeEvent:
    """Structured event emitted by one realtime decode step."""

    kind: Literal["text", "audio", "error"]
    text: str | None = None
    audio: np.ndarray | None = None


@dataclass(slots=True)
class StepResult:
    """One step execution result."""

    events: list[RealtimeEvent] = field(default_factory=list)
    close_requested_reason: str | None = None


@dataclass(slots=True)
class FeedAudioResult:
    dropped_bytes: int = 0
    dropped_frames: int = 0
    backlog_bytes: int = 0
    backlog_frames: float = 0.0
    time_behind_seconds: float = 0.0
    soft_backlog: bool = False
    hard_backlog: bool = False
    hard_action: str = "none"  # none | degrade | close


@dataclass
class SessionState:
    """Mutable state for one active realtime duplex session."""

    session_id: str
    config: SessionConfig

    raw_input_bytes: bytearray = field(default_factory=bytearray)
    decoding_state: Any | None = None
    prompt_token_ids: list[int] = field(default_factory=list)
    speaker_embeds: object | None = None
    last_sequence_len: int = 0
    ignored_text_token_ids: set[int] = field(default_factory=set)
    sil_token_id: int | None = None
    audio_start_token_id: int | None = None

    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)

    frames_in: int = 0
    frames_out: int = 0
    bytes_in: int = 0
    bytes_out: int = 0
    dropped_input_frames: int = 0
    dropped_input_bytes: int = 0
    backlog_soft_events: int = 0
    backlog_hard_events: int = 0
    max_time_behind_seconds: float = 0.0
    decode_errors: int = 0
    consecutive_decode_errors: int = 0
    decode_step_total_seconds: float = 0.0
    decode_step_max_seconds: float = 0.0
    close_requested_reason: str | None = None
    close_reason: str | None = None

    @property
    def idle_seconds(self) -> float:
        return time.time() - self.last_activity

    def touch(self) -> None:
        self.last_activity = time.time()


class RealtimeDuplexSession:
    """Single active session orchestration for duplex realtime decoding."""

    def __init__(
        self,
        *,
        model: SGLangRaonModel,
        tokenizer: Qwen2TokenizerFast,
        config: SessionConfig,
        speaker_embeds: object | None = None,
    ) -> None:
        config.validate()
        self._model = model
        self._tokenizer = tokenizer
        self._state = self._init_state(config, speaker_embeds=speaker_embeds)

    @property
    def state(self) -> SessionState:
        return self._state

    def _init_state(self, config: SessionConfig, speaker_embeds: object | None) -> SessionState:
        state = SessionState(session_id=config.session_id, config=config, speaker_embeds=speaker_embeds)
        prompt_text = resolve_prompt(
            config.prompt,
            config.prompt_role,
            persona=config.persona,
            persona_context=config.persona_context,
        )
        if prompt_text != config.prompt:
            logger.info("resolved prompt key=%s -> %r", config.prompt, prompt_text)

        messages = [{"role": config.prompt_role, "content": prompt_text}]
        text_or_tokens = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        if isinstance(text_or_tokens, list):
            prompt_tokens = text_or_tokens
        else:
            prompt_tokens = self._tokenizer.encode(text_or_tokens)

        text_vocab_size = int(getattr(self._model, "text_vocab_size", None) or getattr(self._model, "vocab_size", 0) or 0)
        fallback_token_id = int(
            getattr(
                self._model,
                "duplex_end_pad_token_id",
                max(0, text_vocab_size - 1),
            )
        )
        state.prompt_token_ids, replaced = _sanitize_prompt_tokens(
            list(prompt_tokens),
            text_vocab_size=text_vocab_size,
            fallback_token_id=fallback_token_id,
        )
        if replaced:
            logger.warning(
                "prompt_token_sanitized session_id=%s replaced=%d total=%d text_vocab_size=%d fallback=%d",
                state.session_id,
                replaced,
                len(state.prompt_token_ids),
                text_vocab_size,
                fallback_token_id,
            )

        ignored, sil_id, audio_start_id = _resolve_ignored_token_ids(self._model, self._tokenizer)
        state.ignored_text_token_ids = ignored
        state.sil_token_id = sil_id
        state.audio_start_token_id = audio_start_id
        return state

    def _ensure_decoding_state(self) -> None:
        import torch

        state = self._state
        if state.decoding_state is not None:
            return

        prompt_tensor = torch.tensor([state.prompt_token_ids], device=self._model.device, dtype=torch.long)
        attention_mask = torch.ones_like(prompt_tensor)
        sampling = state.config.sampling
        init_kwargs = dict(
            sequences=prompt_tensor,
            attention_mask=attention_mask,
            do_sample=sampling.do_sample,
            temperature=sampling.temperature,
            top_k=sampling.top_k,
            top_p=sampling.top_p,
            eos_penalty=sampling.eos_penalty,
            code_temperature=sampling.code_temperature,
            code_top_k=sampling.code_top_k,
            sil_penalty=sampling.sil_penalty,
            bc_penalty=sampling.bc_penalty,
            audio_encoder_chunk_frames=sampling.audio_encoder_chunk_frames,
            speaker_embeds=state.speaker_embeds,
            speak_first=state.config.speak_first,
        )
        accepted = set(inspect.signature(self._model.init_duplex_decoding_state).parameters.keys())
        init_kwargs = {k: v for k, v in init_kwargs.items() if k in accepted}
        with torch.inference_mode():
            state.decoding_state = self._model.init_duplex_decoding_state(**init_kwargs)
        state.last_sequence_len = int(state.decoding_state.sequences.shape[1])

    def _drop_oldest_aligned(self, target_drop_bytes: int) -> tuple[int, int]:
        state = self._state
        if target_drop_bytes <= 0 or not state.raw_input_bytes:
            return 0, 0
        frame_bytes = max(1, state.config.audio.frame_bytes)
        drop = min(target_drop_bytes, len(state.raw_input_bytes))
        drop -= drop % frame_bytes
        if drop <= 0:
            return 0, 0
        del state.raw_input_bytes[:drop]
        return drop, drop // frame_bytes

    def feed_audio(self, pcm_bytes: bytes) -> FeedAudioResult:
        state = self._state
        result = FeedAudioResult()
        state.raw_input_bytes.extend(pcm_bytes)
        state.bytes_in += len(pcm_bytes)
        state.touch()

        audio_cfg = state.config.audio
        max_bytes = audio_cfg.max_buffer_bytes
        if max_bytes > 0 and len(state.raw_input_bytes) > max_bytes:
            dropped, dropped_frames = self._drop_oldest_aligned(len(state.raw_input_bytes) - max_bytes)
            result.dropped_bytes += dropped
            result.dropped_frames += dropped_frames
            state.dropped_input_bytes += dropped
            state.dropped_input_frames += dropped_frames

        backlog_bytes = len(state.raw_input_bytes)
        backlog_seconds = backlog_bytes / max(1, audio_cfg.bytes_per_second)
        soft = audio_cfg.soft_backlog_seconds > 0 and backlog_seconds > audio_cfg.soft_backlog_seconds
        hard = audio_cfg.hard_backlog_seconds > 0 and backlog_seconds > audio_cfg.hard_backlog_seconds
        result.soft_backlog = soft
        result.hard_backlog = hard

        if soft:
            state.backlog_soft_events += 1
        if hard:
            state.backlog_hard_events += 1
            action = audio_cfg.hard_backlog_action.strip().lower()
            action = action if action in {"degrade", "close"} else "degrade"
            result.hard_action = action
            if action == "degrade":
                target_bytes = int(max(0.0, audio_cfg.degrade_target_seconds) * audio_cfg.bytes_per_second)
                if backlog_bytes > target_bytes:
                    dropped, dropped_frames = self._drop_oldest_aligned(backlog_bytes - target_bytes)
                    result.dropped_bytes += dropped
                    result.dropped_frames += dropped_frames
                    state.dropped_input_bytes += dropped
                    state.dropped_input_frames += dropped_frames

        result.backlog_bytes = len(state.raw_input_bytes)
        result.backlog_frames = result.backlog_bytes / max(1, audio_cfg.frame_bytes)
        result.time_behind_seconds = result.backlog_bytes / max(1, audio_cfg.bytes_per_second)
        if result.time_behind_seconds > state.max_time_behind_seconds:
            state.max_time_behind_seconds = result.time_behind_seconds

        if hard and result.hard_action == "close":
            state.close_requested_reason = OVERLOADED_BACKLOG
        return result

    def step(self) -> StepResult:
        import torch

        state = self._state
        audio_cfg = state.config.audio
        needed_bytes = audio_cfg.frame_bytes
        out = StepResult()
        if len(state.raw_input_bytes) < needed_bytes:
            return out
        if state.close_requested_reason:
            out.close_requested_reason = state.close_requested_reason
            return out

        self._ensure_decoding_state()

        chunk_bytes = bytes(state.raw_input_bytes[:needed_bytes])
        del state.raw_input_bytes[:needed_bytes]
        pcm_view = np.frombuffer(chunk_bytes, dtype=np.float32)
        pcm = pcm_view.copy()
        if len(pcm) < audio_cfg.frame_size:
            pcm = np.pad(pcm, (0, audio_cfg.frame_size - len(pcm)))

        if audio_cfg.mic_gain != 1.0:
            pcm = pcm * audio_cfg.mic_gain
        if audio_cfg.input_clip > 0:
            pcm = np.clip(pcm, -audio_cfg.input_clip, audio_cfg.input_clip)
        if audio_cfg.noise_gate > 0:
            rms = float(np.sqrt(np.mean(pcm * pcm))) if pcm.size else 0.0
            if rms < audio_cfg.noise_gate:
                pcm = np.zeros_like(pcm)

        state.frames_in += 1
        chunk_tensor = torch.from_numpy(pcm).to(device=self._model.device, dtype=self._model.dtype)[None, :]
        decode_started = time.perf_counter()
        try:
            with torch.inference_mode():
                state.decoding_state, audio_output = self._model.duplex_decoding_step(
                    state=state.decoding_state,
                    audio_input=chunk_tensor,
                )
            state.consecutive_decode_errors = 0
        except Exception as exc:
            if _is_recoverable_decode_error(exc):
                state.decode_errors += 1
                state.consecutive_decode_errors += 1
                if state.consecutive_decode_errors >= 3:
                    state.close_requested_reason = INTERNAL_ERROR
                    out.close_requested_reason = INTERNAL_ERROR
                out.events.append(RealtimeEvent(kind="error", text=f"decode error: {exc}"))
                out.events.append(RealtimeEvent(kind="audio", audio=np.zeros(audio_cfg.frame_size, dtype=np.float32)))
                return out
            raise
        finally:
            elapsed = max(0.0, time.perf_counter() - decode_started)
            state.decode_step_total_seconds += elapsed
            if elapsed > state.decode_step_max_seconds:
                state.decode_step_max_seconds = elapsed

        sequences = state.decoding_state.sequences
        new_tokens = sequences[0, state.last_sequence_len :].tolist()
        state.last_sequence_len = int(sequences.shape[1])
        text_delta = _decode_text_tokens(
            new_tokens,
            self._tokenizer,
            text_vocab_size=int(getattr(self._model, "text_vocab_size", None) or getattr(self._model, "vocab_size", 0)),
            ignored_token_ids=state.ignored_text_token_ids,
            sil_token_id=state.sil_token_id,
            audio_start_token_id=state.audio_start_token_id,
        )
        if text_delta:
            out.events.append(RealtimeEvent(kind="text", text=text_delta))

        output_np = audio_output[0].detach().float().cpu().numpy().astype(np.float32)
        if audio_cfg.output_gain != 1.0:
            output_np = output_np * audio_cfg.output_gain
        if audio_cfg.output_clip > 0:
            output_np = np.clip(output_np, -audio_cfg.output_clip, audio_cfg.output_clip)
        out.events.append(RealtimeEvent(kind="audio", audio=output_np))

        state.frames_out += 1
        state.bytes_out += len(output_np) * 4
        state.touch()
        return out

    def close(self, *, skip_gpu_cleanup: bool = False) -> None:
        state = self._state
        if state.decoding_state is not None:
            if not skip_gpu_cleanup:
                try:
                    import torch

                    with torch.inference_mode():
                        self._model.free_duplex_decoding_state(state.decoding_state)
                except Exception as exc:
                    if not is_fatal_cuda_error(exc):
                        logger.exception("session_close_error session_id=%s", state.session_id)
                    else:
                        logger.warning("session_close_fatal_cuda session_id=%s: %s", state.session_id, exc)
            state.decoding_state = None
        state.raw_input_bytes.clear()


_RUNTIME_LOCK = threading.Lock()
_RUNTIME_CACHE: dict[tuple[Any, ...], tuple[SGLangRaonModel, Qwen2TokenizerFast]] = {}


def _load_tokenizer_from_bundle(model_path: str) -> Qwen2TokenizerFast:
    from transformers import Qwen2TokenizerFast

    bundle_path = Path(model_path)
    for candidate in (bundle_path / "text_model", bundle_path / "duplex_model", bundle_path):
        if (candidate / "tokenizer.json").is_file():
            return Qwen2TokenizerFast.from_pretrained(str(candidate), local_files_only=True)
    raise FileNotFoundError(f"Could not locate tokenizer files under {bundle_path}")


def get_runtime(
    *,
    model_path: str,
    dtype: str = "bfloat16",
    mem_fraction_static: float = 0.88,
    disable_cuda_graph: bool = False,
    max_running_requests: int | None = None,
    max_total_tokens: int | None = None,
    max_prefill_tokens: int | None = None,
    chunked_prefill_size: int | None = None,
    max_allocated_req_pool_indices: int = 32,
    gpu_id: int = 0,
    compile_audio_modules: bool = True,
    compile_max_sequence_length: int = 8192,
) -> tuple[SGLangRaonModel, Qwen2TokenizerFast]:
    from transformers import Qwen2TokenizerFast

    from ..backends.sglang_backend import SGLangRaonModel

    key = (
        model_path,
        dtype,
        mem_fraction_static,
        disable_cuda_graph,
        max_running_requests,
        max_total_tokens,
        max_prefill_tokens,
        chunked_prefill_size,
        max_allocated_req_pool_indices,
        gpu_id,
        compile_audio_modules,
        compile_max_sequence_length,
    )
    with _RUNTIME_LOCK:
        cached = _RUNTIME_CACHE.get(key)
        if cached is not None:
            _prepare_runtime_model(
                cached[0],
                compile_audio_modules=compile_audio_modules,
                compile_max_sequence_length=compile_max_sequence_length,
            )
            return cached

        model = SGLangRaonModel(
            path=model_path,
            dtype=dtype,
            mem_fraction_static=mem_fraction_static,
            disable_cuda_graph=disable_cuda_graph,
            max_running_requests=max_running_requests,
            max_total_tokens=max_total_tokens,
            max_prefill_tokens=max_prefill_tokens,
            chunked_prefill_size=chunked_prefill_size,
            max_allocated_req_pool_indices=max_allocated_req_pool_indices,
            gpu_id=gpu_id,
        )
        tokenizer = getattr(model, "tokenizer", None)
        if not isinstance(tokenizer, Qwen2TokenizerFast):
            tokenizer = _load_tokenizer_from_bundle(model_path)
            model.tokenizer = tokenizer

        _prepare_runtime_model(
            model,
            compile_audio_modules=compile_audio_modules,
            compile_max_sequence_length=compile_max_sequence_length,
        )

        runtime = (model, tokenizer)
        _RUNTIME_CACHE[key] = runtime
        return runtime


class LocalRealtimeSession:
    """High-level session wrapper used by the FastAPI/WebSocket app."""

    def __init__(
        self,
        *,
        session_id: str,
        model_path: str,
        result_root: str = "./output/fd_gradio_demo",
        session: dict[str, Any] | None = None,
        runtime: dict[str, Any] | None = None,
        prompt: str | None = None,
        prompt_role: str = "system",
        temperature: float = 1.1,
        top_k: int = 100,
        top_p: float = 0.99,
        eos_penalty: float = 0.0,
        sil_penalty: float = 0.0,
        bc_penalty: float = 0.0,
        mic_gain: float = 1.0,
        noise_gate: float = 0.0,
        speaker_mode: str = "default",
        speaker_key: str | None = None,
        **_: Any,
    ) -> None:
        session_payload = dict(session or {})
        runtime_payload = dict(runtime or {})
        sampling_payload = dict(session_payload.get("sampling") or {})
        audio_payload = dict(session_payload.get("audio") or {})

        config = SessionConfig(
            session_id=session_id,
            prompt=str(session_payload.get("prompt", prompt or "eng:full_duplex:listen-first")),
            prompt_role=str(session_payload.get("prompt_role", prompt_role)),
            speaker_mode=str(session_payload.get("speaker_mode", speaker_mode)),
            speaker_key=session_payload.get("speaker_key", speaker_key),
        )
        config.sampling.temperature = float(sampling_payload.get("temperature", temperature))
        config.sampling.top_k = int(sampling_payload.get("top_k", top_k))
        config.sampling.top_p = float(sampling_payload.get("top_p", top_p))
        config.sampling.eos_penalty = float(sampling_payload.get("eos_penalty", eos_penalty))
        config.sampling.sil_penalty = float(sampling_payload.get("sil_penalty", sil_penalty))
        config.sampling.bc_penalty = float(sampling_payload.get("bc_penalty", bc_penalty))
        config.audio.mic_gain = float(audio_payload.get("mic_gain", audio_payload.get("input_gain", mic_gain)))
        config.audio.noise_gate = float(
            audio_payload.get("noise_gate", audio_payload.get("silence_rms_threshold", noise_gate))
        )

        model, tokenizer = get_runtime(
            model_path=model_path,
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

        speaker_audio_value = str(session_payload.get("speaker_audio", "") or "").strip()
        speaker_embeds = None
        resolved_speaker_audio_path: str | None = None
        if speaker_audio_value:
            try:
                speaker_embeds, resolved_speaker_audio_path = _compute_runtime_speaker_embeds(
                    model,
                    speaker_audio_value,
                )
                logger.info("realtime speaker reference loaded: %s", resolved_speaker_audio_path)
            except Exception as exc:
                logger.warning(
                    "realtime speaker reference load failed: %s (%s). Proceeding without speaker conditioning.",
                    speaker_audio_value,
                    exc,
                )

        self._model_path = model_path
        self._inner = RealtimeDuplexSession(
            model=model,
            tokenizer=tokenizer,
            config=config,
            speaker_embeds=speaker_embeds,
        )
        self._artifacts = SessionArtifacts(
            session_id=session_id,
            sample_rate=config.audio.sampling_rate,
            result_root=Path(result_root),
        )
        self._closed = False
        self._result_payload: dict[str, Any] | None = None
        self._session_params = {
            "prompt": config.prompt,
            "prompt_role": config.prompt_role,
            "speaker_mode": config.speaker_mode,
            "speaker_key": config.speaker_key,
            "speaker_audio": resolved_speaker_audio_path,
            "sampling": {
                "temperature": config.sampling.temperature,
                "top_k": config.sampling.top_k,
                "top_p": config.sampling.top_p,
                "eos_penalty": config.sampling.eos_penalty,
                "sil_penalty": config.sampling.sil_penalty,
                "bc_penalty": config.sampling.bc_penalty,
            },
            "audio": {
                "sampling_rate": config.audio.sampling_rate,
                "frame_size": config.audio.frame_size,
                "mic_gain": config.audio.mic_gain,
                "noise_gate": config.audio.noise_gate,
            },
        }

    @property
    def session_id(self) -> str:
        return self._inner.state.session_id

    def start(self) -> list[Frame]:
        self._artifacts.add_event("ready", {"session_id": self.session_id})
        return [Frame.ready()]

    def handle_audio_frame(self, pcm: np.ndarray) -> list[Frame]:
        if self._closed:
            return [Frame.close(self._inner.state.close_reason or CLIENT_FINISH)]

        pcm_np = np.asarray(pcm, dtype=np.float32).reshape(-1)
        self._artifacts.append_user_audio(pcm_np)
        self._inner.feed_audio(pcm_np.tobytes())

        frames: list[Frame] = []
        frame_bytes = self._inner.state.config.audio.frame_bytes
        while len(self._inner.state.raw_input_bytes) >= frame_bytes:
            result = self._inner.step()
            for event in result.events:
                if event.kind == "text" and event.text:
                    self._artifacts.append_text(event.text)
                    frames.append(Frame.text(event.text))
                elif event.kind == "audio" and event.audio is not None:
                    self._artifacts.append_assistant_audio(event.audio)
                    frames.append(Frame.audio(event.audio))
                elif event.kind == "error" and event.text:
                    frames.append(Frame.error(event.text))
            if result.close_requested_reason:
                self.finish(result.close_requested_reason)
                frames.append(Frame.close(result.close_requested_reason))
                break
            if not result.events:
                break
        return frames

    def request_close(self, reason: str) -> list[Frame]:
        self.finish(reason)
        return [Frame.close(reason)]

    def finish(self, reason: str = CLIENT_FINISH) -> dict[str, Any]:
        if self._result_payload is not None:
            return self._result_payload

        self._inner.state.close_reason = reason
        self._inner.close()
        state = self._inner.state
        frame_seconds = state.config.audio.frame_size / max(1, state.config.audio.sampling_rate)
        user_audio_seconds = float(state.frames_in) * frame_seconds
        decode_total = float(state.decode_step_total_seconds)
        runtime_stats = {
            "frames_in": int(state.frames_in),
            "frames_out": int(state.frames_out),
            "bytes_in": int(state.bytes_in),
            "bytes_out": int(state.bytes_out),
            "dropped_input_frames": int(state.dropped_input_frames),
            "dropped_input_bytes": int(state.dropped_input_bytes),
            "backlog_soft_events": int(state.backlog_soft_events),
            "backlog_hard_events": int(state.backlog_hard_events),
            "max_time_behind_seconds": float(state.max_time_behind_seconds),
            "decode_errors": int(state.decode_errors),
            "decode_step_total_seconds": decode_total,
            "decode_step_avg_ms": (decode_total / state.frames_in * 1000.0) if state.frames_in else 0.0,
            "decode_step_max_ms": float(state.decode_step_max_seconds * 1000.0),
            "user_audio_seconds_estimated": user_audio_seconds,
            "decode_rtf": (decode_total / user_audio_seconds) if user_audio_seconds > 0 else 0.0,
        }
        metadata = self._artifacts.flush(
            model_path=self._model_path,
            session_params=self._session_params,
            close_reason=reason,
            runtime_stats=runtime_stats,
            write_optional_bundle=True,
        )
        files = metadata.get("files", {})
        self._closed = True
        self._result_payload = {
            "session_id": self.session_id,
            "close_reason": reason,
            "metadata": metadata,
            "user_wav": files.get("user_wav"),
            "assistant_wav": files.get("assistant_wav"),
            "transcript_txt": files.get("transcript"),
            "metadata_json": files.get("metadata"),
            "events_jsonl": files.get("events_jsonl"),
            "conversation_stereo_wav": files.get("conversation_stereo_wav"),
            "session_bundle_zip": files.get("session_bundle_zip"),
        }
        return self._result_payload

    def artifact_response(self) -> dict[str, Any] | None:
        return self._result_payload

    def close(self) -> None:
        if not self._closed:
            self.finish(self._inner.state.close_reason or CLIENT_FINISH)


def create_session(**kwargs: Any) -> LocalRealtimeSession:
    """Factory used by the runtime manager to build the active session."""
    return LocalRealtimeSession(**kwargs)
