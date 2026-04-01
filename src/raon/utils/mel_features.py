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

"""Shared mel spectrogram feature extraction utilities."""

import torch
import torch.nn.functional as F


def compute_log_mel_spectrogram(
    audio: torch.Tensor,
    window: torch.Tensor,
    mel_filters: torch.Tensor,
    n_fft: int,
    hop_length: int,
) -> torch.Tensor:
    """Compute log-mel spectrogram from audio waveform.

    Mirrors ``WhisperFeatureExtractor._torch_extract_fbank_features``:
    runs a centered STFT, projects onto mel filterbanks, log-compresses,
    and applies global max normalization so that the dynamic range is
    clipped to 8 dB below the per-batch maximum.

    Args:
        audio: Waveform tensor. Shape: [batch, samples] or [samples].
        window: STFT window tensor of size ``n_fft``.
        mel_filters: Mel filterbank matrix. Shape: [n_mels, n_fft//2+1].
        n_fft: FFT size.
        hop_length: STFT hop length.

    Returns:
        Log-mel spectrogram. Shape: [batch, n_mels, time].
    """
    stft = torch.stft(audio, n_fft, hop_length, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2
    mel_spec = mel_filters.T @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    max_val = log_spec.max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
    log_spec = torch.maximum(log_spec, max_val - 8.0)
    log_spec = (log_spec + 4.0) / 4.0

    return log_spec
