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

import torch


def delay_audio_codes(
    delays: list[int],
    audio_codes: torch.Tensor,
    padding_value: int = 0,
) -> torch.Tensor:
    """Apply per-codebook delays to audio codes for training.

    Args:
        delays: List of delay values for each codebook
        audio_codes: Audio codes tensor, either (B, T, K) or (T, K)
        padding_value: Value to use for padding delayed positions

    Returns:
        Delayed audio codes with same shape as input
    """
    # Handle both 2D (T, K) and 3D (B, T, K) inputs
    squeeze_batch = False
    if audio_codes.dim() == 2:
        audio_codes = audio_codes.unsqueeze(0)  # (T, K) -> (1, T, K)
        squeeze_batch = True

    B, T, K = audio_codes.shape
    audio_codes_t = audio_codes.transpose(1, 2)  # (B, K, T)
    delayed = []
    for k, delay in enumerate(delays):
        if delay == 0:
            delayed.append(audio_codes_t[:, k])
        else:
            line = audio_codes_t[:, k].roll(delay, dims=1)
            line[:, :delay] = padding_value
            delayed.append(line)
    result = torch.stack(delayed, dim=1).transpose(1, 2)  # (B, T, K)

    if squeeze_batch:
        result = result.squeeze(0)  # (1, T, K) -> (T, K)

    return result


def undelay_audio_codes(
    delays: list[int],
    audio_codes: torch.Tensor,
    padding_value: int = 0,
) -> torch.Tensor:
    """Inverse of delay_audio_codes: shift codes back to original alignment."""
    if all(d == 0 for d in delays):
        return audio_codes
    squeeze_batch = False
    if audio_codes.dim() == 2:
        audio_codes = audio_codes.unsqueeze(0)
        squeeze_batch = True
    B, T, K = audio_codes.shape
    audio_codes_t = audio_codes.transpose(1, 2)
    undelayed = []
    for k, delay in enumerate(delays):
        if delay == 0:
            undelayed.append(audio_codes_t[:, k])
        else:
            line = audio_codes_t[:, k].roll(-delay, dims=1)
            line[:, -delay:] = padding_value
            undelayed.append(line)
    result = torch.stack(undelayed, dim=1).transpose(1, 2)
    if squeeze_batch:
        result = result.squeeze(0)
    return result


