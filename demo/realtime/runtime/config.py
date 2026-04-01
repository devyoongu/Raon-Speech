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

"""Configuration types for realtime full-duplex session runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class SamplingConfig:
    """Sampling parameters used for duplex decoding."""

    do_sample: bool = True
    temperature: float = 1.1
    top_k: int = 100
    top_p: float = 0.99
    eos_penalty: float = 0.0
    code_temperature: float = 0.8
    code_top_k: int = 10
    sil_penalty: float = 0.0
    bc_penalty: float = 0.0
    audio_encoder_chunk_frames: int = 100

    def validate(self) -> None:
        """Validate sampling knobs and hard-fail on unsupported combinations."""
        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")
        if self.top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {self.top_k}")
        if self.top_p <= 0 or self.top_p > 1:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.code_temperature <= 0:
            raise ValueError(f"code_temperature must be > 0, got {self.code_temperature}")
        if self.code_top_k < 0:
            raise ValueError(f"code_top_k must be >= 0, got {self.code_top_k}")
        if self.audio_encoder_chunk_frames <= 0:
            raise ValueError(f"audio_encoder_chunk_frames must be > 0, got {self.audio_encoder_chunk_frames}")


@dataclass(slots=True)
class AudioConfig:
    """Realtime audio transport and preprocessing knobs."""

    sampling_rate: int = 24000
    frame_size: int = 1920
    mic_gain: float = 1.0
    noise_gate: float = 0.0
    input_clip: float = 100.0
    output_gain: float = 1.0
    output_clip: float = 100.0
    max_raw_buffer_seconds: float = 10.0
    soft_backlog_seconds: float = 0.40
    hard_backlog_seconds: float = 1.20
    hard_backlog_action: str = "degrade"
    degrade_target_seconds: float = 0.30

    @property
    def frame_duration_ms(self) -> float:
        return self.frame_size / self.sampling_rate * 1000.0

    @property
    def frame_bytes(self) -> int:
        return self.frame_size * 4

    @property
    def bytes_per_second(self) -> int:
        return self.sampling_rate * 4

    @property
    def max_buffer_bytes(self) -> int:
        return int(self.max_raw_buffer_seconds * self.bytes_per_second)

    def validate(self) -> None:
        if self.sampling_rate <= 0:
            raise ValueError(f"sampling_rate must be > 0, got {self.sampling_rate}")
        if self.frame_size <= 0:
            raise ValueError(f"frame_size must be > 0, got {self.frame_size}")
        if self.mic_gain < 0:
            raise ValueError(f"mic_gain must be >= 0, got {self.mic_gain}")
        if self.noise_gate < 0:
            raise ValueError(f"noise_gate must be >= 0, got {self.noise_gate}")
        if self.max_raw_buffer_seconds <= 0:
            raise ValueError(f"max_raw_buffer_seconds must be > 0, got {self.max_raw_buffer_seconds}")
        if self.hard_backlog_action not in {"degrade", "close"}:
            raise ValueError(f"hard_backlog_action must be one of {{'degrade', 'close'}}, got {self.hard_backlog_action!r}")


@dataclass(slots=True)
class SessionConfig:
    """Configuration for one realtime full-duplex session."""

    session_id: str = ""
    prompt: str = "eng:full_duplex:listen-first"
    prompt_role: str = "system"
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    speaker_mode: str = "default"  # "none" | "default" | "recorded"
    speaker_key: str | None = None
    idle_timeout_seconds: float = 15.0
    speak_first: bool = False
    persona: str | None = None
    persona_context: str | None = None

    def validate(self) -> None:
        if self.prompt_role not in {"system", "user", "assistant"}:
            raise ValueError(f"prompt_role must be one of system/user/assistant, got {self.prompt_role!r}")
        if self.speaker_mode not in {"none", "default", "recorded"}:
            raise ValueError(f"speaker_mode must be one of {{'none', 'default', 'recorded'}}, got {self.speaker_mode!r}")
        if self.idle_timeout_seconds <= 0:
            raise ValueError(f"idle_timeout_seconds must be > 0, got {self.idle_timeout_seconds}")
        self.audio.validate()
        self.sampling.validate()

