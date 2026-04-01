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

from __future__ import annotations

from dataclasses import dataclass
from typing import NotRequired, TypedDict

import torch
from transformers.utils.generic import ModelOutput


class RaonInputs(TypedDict):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    audio_input: torch.Tensor | None
    audio_output: torch.Tensor | None
    speaker_encoder_audio: NotRequired[torch.Tensor | None]
    audio_input_lengths: torch.Tensor | None
    audio_output_lengths: torch.Tensor | None
    speaker_encoder_audio_lengths: NotRequired[torch.Tensor | None]
    labels: torch.Tensor
    sample_slot: NotRequired[int]


class DuplexInputs(TypedDict):
    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor
    audio_input: torch.Tensor
    audio_output: torch.Tensor
    audio_input_lengths: torch.Tensor
    audio_output_lengths: torch.Tensor
    speaker_encoder_audio: torch.Tensor | None
    speaker_encoder_audio_lengths: torch.Tensor | None
    sample_slot: int


@dataclass
class AudioEncoderOutput(ModelOutput):
    """Output of the audio encoder forward pass.

    Attributes:
        audio_embeds: Encoded audio representations. Shape: [batch, frames, hidden_size].
        audio_embeds_mask: Boolean mask indicating valid frames. Shape: [batch, frames].
    """

    audio_embeds: torch.Tensor | None = None
    audio_embeds_mask: torch.Tensor | None = None
    encoder_cache: tuple | None = None  # (encoder_past_key_values, conv_padding_cache) for streaming


@dataclass
class AudioTokenizerOutput(ModelOutput):
    """Output of the audio tokenizer (encoder + quantizer) forward pass.

    Attributes:
        audio_codes: Discrete codec codes per frame and codebook group.
            Shape: [batch, num_code_groups, num_frames].
        audio_codes_mask: Boolean mask indicating valid code frames. Shape: [batch, num_frames].
        mimi_features: Pre-quantization encoder features.
            Shape: [batch_size, num_frames, 512].
        encoder_cache: Streaming encoder cache tuple, if available.
    """

    audio_codes: torch.Tensor | None = None
    audio_codes_mask: torch.Tensor | None = None
    mimi_features: torch.Tensor | None = None  # [batch_size, num_frames, 512] pre-quantization features
    encoder_cache: tuple | None = None  # (encoder_past_key_values, conv_padding_cache) for streaming


@dataclass
class AudioDecoderOutput(ModelOutput):
    """Output of the audio decoder (codec decoder) forward pass.

    Attributes:
        audio: Reconstructed waveform. Shape: [batch, num_samples].
        decoder_cache: Streaming decoder cache tuple, if available.
    """

    audio: torch.Tensor
    decoder_cache: tuple | None = None  # (decoder_past_key_values, conv1d_padding_cache, convtranspose1d_padding_cache)
