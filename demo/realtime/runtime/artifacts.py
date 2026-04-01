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

"""Artifact helpers for the realtime full-duplex Gradio demo."""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

DEFAULT_RESULT_ROOT = Path("/wbl-fast/usrs/hh/RAON-publish/result/fd_gradio_demo")


def utc_timestamp_for_path(now: datetime | None = None) -> str:
    """Return a UTC timestamp suitable for output directory names."""
    dt = now or datetime.now(UTC)
    return dt.strftime("%Y%m%d-%H%M%S")


def _as_float32_mono(samples: np.ndarray) -> np.ndarray:
    arr = np.asarray(samples, dtype=np.float32)
    return arr.reshape(-1)


def _concat_chunks(chunks: list[np.ndarray]) -> np.ndarray:
    if not chunks:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), _as_float32_mono(samples), sample_rate)


def _pad_to_same_length(lhs: np.ndarray, rhs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    max_len = max(lhs.shape[0], rhs.shape[0])
    if lhs.shape[0] < max_len:
        lhs = np.pad(lhs, (0, max_len - lhs.shape[0]))
    if rhs.shape[0] < max_len:
        rhs = np.pad(rhs, (0, max_len - rhs.shape[0]))
    return lhs, rhs


@dataclass(slots=True)
class SessionArtifacts:
    """Incremental collector and flusher for realtime session outputs."""

    session_id: str
    sample_rate: int = 24000
    result_root: Path = DEFAULT_RESULT_ROOT
    run_id: str | None = None
    started_at_utc: datetime = field(default_factory=lambda: datetime.now(UTC))

    user_audio_chunks: list[np.ndarray] = field(default_factory=list)
    assistant_audio_chunks: list[np.ndarray] = field(default_factory=list)
    text_deltas: list[str] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.result_root = Path(self.result_root)
        if self.run_id is None:
            self.run_id = utc_timestamp_for_path(self.started_at_utc)

    @property
    def output_dir(self) -> Path:
        return self.result_root / str(self.run_id)

    def append_user_audio(self, samples: np.ndarray) -> None:
        self.user_audio_chunks.append(_as_float32_mono(samples))

    def append_assistant_audio(self, samples: np.ndarray) -> None:
        self.assistant_audio_chunks.append(_as_float32_mono(samples))

    def append_text(self, delta: str) -> None:
        if delta:
            self.text_deltas.append(delta)

    def add_event(self, kind: str, payload: dict[str, Any] | None = None) -> None:
        self.events.append(
            {
                "ts_utc": datetime.now(UTC).isoformat(),
                "kind": kind,
                "payload": payload or {},
            }
        )

    def flush(
        self,
        *,
        model_path: str,
        session_params: dict[str, Any],
        close_reason: str | None = None,
        runtime_stats: dict[str, Any] | None = None,
        write_optional_bundle: bool = True,
    ) -> dict[str, Any]:
        """Write artifacts and return metadata including path manifest."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        finished_at = datetime.now(UTC)

        user = _concat_chunks(self.user_audio_chunks)
        assistant = _concat_chunks(self.assistant_audio_chunks)
        transcript = "".join(self.text_deltas)

        user_wav_path = self.output_dir / "user.wav"
        assistant_wav_path = self.output_dir / "assistant.wav"
        transcript_path = self.output_dir / "transcript.txt"
        metadata_path = self.output_dir / "metadata.json"
        events_path = self.output_dir / "events.jsonl"
        stereo_path = self.output_dir / "conversation_stereo.wav"
        bundle_path = self.output_dir / "session_bundle.zip"

        _write_wav(user_wav_path, user, self.sample_rate)
        _write_wav(assistant_wav_path, assistant, self.sample_rate)
        transcript_path.write_text(transcript, encoding="utf-8")

        if self.events:
            with events_path.open("w", encoding="utf-8") as f:
                for event in self.events:
                    f.write(json.dumps(event, ensure_ascii=False) + "\n")

        user_stereo, assistant_stereo = _pad_to_same_length(user, assistant)
        stereo = np.stack([user_stereo, assistant_stereo], axis=-1) if user_stereo.size else np.zeros((0, 2), np.float32)
        sf.write(str(stereo_path), stereo, self.sample_rate)

        duration_user = float(user.shape[0] / self.sample_rate)
        duration_assistant = float(assistant.shape[0] / self.sample_rate)
        duration_total = max(duration_user, duration_assistant)

        metadata: dict[str, Any] = {
            "session_id": self.session_id,
            "model_path": model_path,
            "session_params": session_params,
            "start_timestamp_utc": self.started_at_utc.isoformat(),
            "finish_timestamp_utc": finished_at.isoformat(),
            "durations_seconds": {
                "user": duration_user,
                "assistant": duration_assistant,
                "total": duration_total,
            },
            "sample_counts": {
                "user": int(user.shape[0]),
                "assistant": int(assistant.shape[0]),
            },
            "close_reason": close_reason or "",
            "runtime_stats": runtime_stats or {},
            "files": {
                "user_wav": str(user_wav_path),
                "assistant_wav": str(assistant_wav_path),
                "transcript": str(transcript_path),
                "metadata": str(metadata_path),
                "events_jsonl": str(events_path) if self.events else "",
                "conversation_stereo_wav": str(stereo_path),
                "session_bundle_zip": str(bundle_path) if write_optional_bundle else "",
            },
        }
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

        if write_optional_bundle:
            with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.write(user_wav_path, arcname="user.wav")
                zf.write(assistant_wav_path, arcname="assistant.wav")
                zf.write(transcript_path, arcname="transcript.txt")
                zf.write(metadata_path, arcname="metadata.json")
                zf.write(stereo_path, arcname="conversation_stereo.wav")
                if self.events:
                    zf.write(events_path, arcname="events.jsonl")

        return metadata


