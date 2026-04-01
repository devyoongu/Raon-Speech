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

"""Audio I/O utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch


def load_audio(
    path: str | Path,
    target_sr: int,
    *,
    mono: bool = True,
    channel: int | None = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, int]:
    """Load an audio file and resample to *target_sr*.

    Returns:
        Tuple of (waveform, sample_rate). waveform shape: [channels, samples].
    """
    data, sr = sf.read(str(path), dtype="float32")
    if data.ndim == 1:
        audio = torch.from_numpy(data)[None]  # [1, samples]
    else:
        audio = torch.from_numpy(data.T)  # [channels, samples]

    if channel is not None and audio.shape[0] > 1:
        audio = audio[channel : channel + 1]
    elif mono and audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    if sr != target_sr:
        import torchaudio.functional

        audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=target_sr)

    if dtype is not None:
        audio = audio.to(dtype=dtype)
    if device is not None:
        audio = audio.to(device=device)

    return audio, target_sr


def save_audio(
    audio: torch.Tensor | np.ndarray,
    sampling_rate: int,
    path: str | Path,
    length: int | None = None,
) -> None:
    """Save audio to a WAV file.

    Accepts a torch tensor or numpy array. Tensors are converted to float32
    numpy before writing. Multi-dimensional tensors are squeezed to 1-D.

    Args:
        audio: Waveform data — ``torch.Tensor`` or ``np.ndarray``.
        sampling_rate: Sampling rate in Hz.
        path: Destination file path.
        length: If provided, truncate to this many samples before saving.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(audio, torch.Tensor):
        if audio.ndim == 2:
            audio = audio[0]
        audio_np = audio.float().cpu().numpy()
    else:
        audio_np = np.asarray(audio, dtype=np.float32)
        if audio_np.ndim == 2:
            audio_np = audio_np[0]

    if length is not None:
        audio_np = audio_np[:length]

    sf.write(str(path), audio_np, sampling_rate)
