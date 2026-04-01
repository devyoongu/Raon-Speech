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

"""Realtime binary message protocol for local duplex demo sessions."""

from __future__ import annotations

import enum
from dataclasses import dataclass

import numpy as np


class MessageKind(enum.IntEnum):
    """Message type indicators (first byte of wire frame)."""

    READY = 0x00
    AUDIO = 0x01
    TEXT = 0x02
    ERROR = 0x05
    CLOSE = 0x06
    PING = 0x07
    PONG = 0x08


@dataclass(slots=True)
class Frame:
    """Single realtime protocol frame."""

    kind: MessageKind
    payload: bytes

    def encode(self) -> bytes:
        """Encode frame to wire bytes."""
        return bytes([self.kind]) + self.payload

    @classmethod
    def decode(cls, data: bytes) -> Frame:
        """Decode frame from wire bytes."""
        if len(data) < 1:
            raise ValueError("Empty frame")
        return cls(kind=MessageKind(data[0]), payload=data[1:])

    @classmethod
    def ready(cls) -> Frame:
        return cls(kind=MessageKind.READY, payload=b"")

    @classmethod
    def audio(cls, pcm: np.ndarray) -> Frame:
        arr = np.asarray(pcm, dtype=np.float32).reshape(-1)
        return cls(kind=MessageKind.AUDIO, payload=arr.tobytes())

    @classmethod
    def text(cls, content: str) -> Frame:
        return cls(kind=MessageKind.TEXT, payload=content.encode("utf-8"))

    @classmethod
    def error(cls, message: str) -> Frame:
        return cls(kind=MessageKind.ERROR, payload=message.encode("utf-8"))

    @classmethod
    def close(cls, reason: str | None = None) -> Frame:
        payload = reason.encode("utf-8") if reason else b""
        return cls(kind=MessageKind.CLOSE, payload=payload)

    def audio_samples(self) -> np.ndarray:
        """Interpret payload as float32 PCM samples."""
        return np.frombuffer(self.payload, dtype=np.float32)

    def text_content(self) -> str:
        """Interpret payload as UTF-8 text."""
        return self.payload.decode("utf-8")
