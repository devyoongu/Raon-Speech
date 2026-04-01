# coding=utf-8
# Copyright 2026 The Qwen team, Alibaba Group, the HuggingFace Inc. team, and The RAON Authors. All rights reserved.
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
#
# Based on the HuggingFace Transformers Qwen3OmniMoe audio encoder implementation.
# Modified by The RAON Authors to support standalone attention modification and streaming.

"""AuT audio encoder with a duplicated attention submodule."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

import numpy as np
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from torch import nn
from torch.nn import functional as F
from transformers import WhisperFeatureExtractor
from transformers.activations import ACT2FN
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeAudioEncoderConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoder,
    Qwen3OmniMoePreTrainedModel,
    SinusoidsPositionEmbedding,
)

from .audio_tokenizer import CausalAudioEncoderOutput
from raon.utils.mel_features import compute_log_mel_spectrogram

_AUDIO_TOWER_PREFIX = "thinker.audio_tower."


@dataclass
class AuTStreamingState:
    """All mutable state needed for frame-by-frame streaming of the causal AuT encoder.

    A single instance is created via ``AuTEncoder.init_streaming_state`` (or
    ``AuTWrapper.init_streaming_state``) and passed into each successive
    ``forward`` call.  The caller must **not** modify the tensors in-place;
    updated state is returned from the forward methods.
    """

    stft_cache: torch.Tensor | None
    """Cached tail waveform samples for STFT overlap between chunks.
    Shape: [num_leftover_samples]. Dtype: float.  ``None`` before the first call."""

    running_max: float
    """Running maximum of per-frame log-mel maxima across all chunks so far.
    Used for causal running-max normalization in feature extraction."""

    conv_caches: list[torch.Tensor]
    """Per-Conv2d-layer border cache on the time axis (3 entries).
    Each tensor has shape [1, channels, freq_dim, 0..2]. Dtype: float."""

    kv_caches: list[tuple[torch.Tensor, torch.Tensor]]
    """Per-transformer-layer past key and value projections.
    Each tuple is (past_keys, past_values) with shape
    [1, num_heads, past_seq_len, head_dim]. Dtype: float.
    Empty (past_seq_len=0) before the first call."""

    num_frames_produced: int
    """Total number of output frames produced so far (positional embedding offset)."""


@dataclass
class AuTEncoderOutput:
    """Output of ``AuTEncoder.forward``.

    Contains the projected encoder hidden states and per-sample output frame
    counts.
    """

    last_hidden_state: torch.Tensor
    """Padded batched encoder output.
    Shape: [batch_size, max_output_frames, output_dim]. Dtype: float."""

    output_lens: list[int]
    """Per-sample output frame counts."""

    streaming_state: AuTStreamingState | None = None
    """Updated streaming state, present only when streaming mode is active."""


class AudioAttention(nn.Module):
    """Multi-headed attention for the AuT audio encoder.

    This is a standalone duplicate of the HuggingFace audio attention module that
    can be modified independently. The forward signature and weight layout are
    identical so that pretrained weights can be loaded directly.
    """

    def __init__(self, config: Qwen3OmniMoeAudioEncoderConfig) -> None:
        super().__init__()
        self.embed_dim: int = config.d_model
        self.num_heads: int = config.encoder_attention_heads
        self.dropout: float = config.attention_dropout
        self.head_dim: int = self.embed_dim // self.num_heads

        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.scaling: float = self.head_dim**-0.5
        self.attention_dropout: float = 0.0
        self.is_decoder: bool = False
        self.is_causal: bool = False
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal: bool = False,
    ) -> torch.Tensor:
        """Compute multi-headed self-attention.

        Args:
            hidden_states: Input tensor.
                Shape: [batch_size, seq_len, embed_dim]. Dtype: float.
            attention_mask: Per-token padding mask (True = valid).
                Shape: [batch_size, seq_len]. Dtype: bool.
            causal: If True, use causal (autoregressive) attention where each
                position can only attend to itself and earlier positions.

        Returns:
            Attention output. Shape: [batch_size, seq_len, embed_dim]. Dtype: float.
        """
        batch_size, seq_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        additive_mask = None
        if not causal and attention_mask is not None:
            key_padding = ~attention_mask.to(torch.bool)
            additive_mask = torch.zeros((batch_size, 1, 1, seq_len), device=hidden_states.device, dtype=hidden_states.dtype)
            additive_mask = additive_mask.masked_fill(
                key_padding.unsqueeze(1).unsqueeze(1), torch.finfo(hidden_states.dtype).min
            )

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=additive_mask,
            dropout_p=0.0 if not self.training else self.attention_dropout,
            is_causal=causal,
            scale=self.scaling,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_proj(attn_output)

    def forward_streaming(
        self,
        hidden_states: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Compute causal self-attention with a KV cache for streaming.

        Projects the new hidden states into Q/K/V, concatenates K/V with the
        cached past, runs scaled dot-product attention (new queries attend to
        all past + new keys), and returns the updated cache.

        Args:
            hidden_states: New input tokens for this streaming step.
                Shape: [new_seq_len, embed_dim]. Dtype: float.
            past_key_value: Tuple of (past_keys, past_values) from previous
                streaming steps.
                Each shape: [1, num_heads, past_seq_len, head_dim]. Dtype: float.

        Returns:
            A tuple of:
                - Attention output for the new tokens only.
                    Shape: [new_seq_len, embed_dim]. Dtype: float.
                - Updated (past_keys, past_values) including the new tokens.
                    Each shape: [1, num_heads, past_seq_len + new_seq_len, head_dim].
                    Dtype: float.
        """
        new_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).reshape(new_len, self.num_heads, -1)
        key_states = self.k_proj(hidden_states).reshape(new_len, self.num_heads, -1)
        value_states = self.v_proj(hidden_states).reshape(new_len, self.num_heads, -1)

        # [new_len, num_heads, head_dim] -> [1, num_heads, new_len, head_dim]
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        # Concatenate with past KV cache.
        past_keys, past_values = past_key_value
        key_states = torch.cat([past_keys, key_states], dim=2)
        value_states = torch.cat([past_values, value_states], dim=2)

        # Causal mask: new queries can attend to all past + current keys,
        # but not to future keys within the new chunk.
        total_len = key_states.shape[2]
        causal_mask = torch.full(
            (1, 1, new_len, total_len),
            torch.finfo(hidden_states.dtype).min,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        past_len = total_len - new_len
        # All new queries can attend to all past positions.
        causal_mask[:, :, :, :past_len] = 0.0
        # Within the new chunk, apply lower-triangular mask.
        new_block = (
            torch.triu(
                torch.ones(new_len, new_len, device=hidden_states.device, dtype=hidden_states.dtype),
                diagonal=1,
            )
            * torch.finfo(hidden_states.dtype).min
        )
        causal_mask[:, :, :, past_len:] = new_block

        # Use explicit matmul instead of scaled_dot_product_attention so that
        # the streaming (1-token-at-a-time) and non-streaming (full-sequence)
        # paths produce identical results on CPU.  SDPA dispatches to different
        # fused kernels depending on input shape, causing ~1e-3 diffs.
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        attn_weights = attn_weights + causal_mask
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.squeeze(0).transpose(0, 1).reshape(new_len, -1).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output, (key_states, value_states)


class AuTEncoderLayer(GradientCheckpointingLayer):
    """Single transformer encoder layer for the AuT audio encoder.

    Uses the local ``AudioAttention`` duplicate for self-attention while keeping
    the FFN, layer norms, and activation function from HuggingFace.
    """

    def __init__(self, config: Qwen3OmniMoeAudioEncoderConfig) -> None:
        super().__init__()
        self.embed_dim: int = config.d_model
        self.self_attn = AudioAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout: float = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout: float = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal: bool = False,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor]:
        """Run one encoder layer (pre-norm self-attention + FFN).

        Args:
            hidden_states: Batched input hidden states.
                Shape: [batch_size, seq_len, embed_dim]. Dtype: float.
            attention_mask: Per-token padding mask (True = valid).
                Shape: [batch_size, seq_len]. Dtype: bool.
            causal: If True, use causal attention.
            padding_mask: Per-token padding mask used to zero padded positions.
                Shape: [batch_size, seq_len]. Dtype: bool.

        Returns:
            Single-element tuple containing the output hidden states.
            Shape: [batch_size, seq_len, embed_dim]. Dtype: float.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal=causal,
        )
        hidden_states = residual + hidden_states
        if padding_mask is not None:
            hidden_states = hidden_states * padding_mask.unsqueeze(-1).to(hidden_states.dtype)
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states
        if padding_mask is not None:
            hidden_states = hidden_states * padding_mask.unsqueeze(-1).to(hidden_states.dtype)

        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        return outputs

    def forward_streaming(
        self,
        hidden_states: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Run one encoder layer in streaming mode with a KV cache.

        Args:
            hidden_states: New hidden states for this streaming step.
                Shape: [new_seq_len, embed_dim]. Dtype: float.
            past_key_value: Tuple of (past_keys, past_values) for this layer.
                Each shape: [1, num_heads, past_seq_len, head_dim]. Dtype: float.

        Returns:
            A tuple of:
                - Output hidden states for the new tokens.
                    Shape: [new_seq_len, embed_dim]. Dtype: float.
                - Updated (past_keys, past_values).
                    Each shape: [1, num_heads, past_seq_len + new_seq_len, head_dim].
                    Dtype: float.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, new_kv = self.self_attn.forward_streaming(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        return hidden_states, new_kv


class AuTEncoder(Qwen3OmniMoePreTrainedModel):
    """Audio Transformer encoder with a duplicated attention submodule.

    Architecturally identical to the HuggingFace audio encoder but uses
    ``AuTEncoderLayer`` (which contains the local ``AudioAttention`` duplicate)
    instead of the original encoder layer. All other components (convolutions,
    positional embeddings, projection layers) are unchanged.

    Pretrained weights from the HuggingFace audio encoder can be loaded directly
    because the state dict keys are identical (the attention weight names inside
    each layer match one-to-one).
    """

    config: Qwen3OmniMoeAudioEncoderConfig
    main_input_name = "input_features"
    _no_split_modules = ["AuTEncoderLayer"]
    _supports_sdpa = True

    def __init__(self, config: Qwen3OmniMoeAudioEncoderConfig) -> None:
        super().__init__(config)
        self.dropout: float = config.dropout

        embed_dim = config.d_model
        self.num_mel_bins: int = config.num_mel_bins
        self.max_source_positions: int = config.max_source_positions
        self.embed_scale: float = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.n_window: int = config.n_window
        self.positional_embedding = SinusoidsPositionEmbedding(self.max_source_positions, embed_dim)
        self.layers = nn.ModuleList([AuTEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.ln_post = nn.LayerNorm(config.d_model)
        self.gradient_checkpointing = False
        self.conv2d1 = nn.Conv2d(1, config.downsample_hidden_size, 3, 2, padding=1)
        self.conv2d2 = nn.Conv2d(config.downsample_hidden_size, config.downsample_hidden_size, 3, 2, padding=1)
        self.conv2d3 = nn.Conv2d(config.downsample_hidden_size, config.downsample_hidden_size, 3, 2, padding=1)
        self.conv_out = nn.Linear(
            config.downsample_hidden_size * ((((config.num_mel_bins + 1) // 2 + 1) // 2 + 1) // 2),
            config.d_model,
            bias=False,
        )
        self.proj1 = nn.Linear(config.d_model, config.d_model)
        self.act = ACT2FN[config.activation_function]
        self.proj2 = nn.Linear(config.d_model, config.output_dim)
        self.n_window_infer: int = self.config.n_window_infer
        self.conv_chunksize: int = self.config.conv_chunksize
        # Initialize weights and apply final processing.
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        """Return the first convolutional layer as the input embedding."""
        return self.conv2d1

    def set_input_embeddings(self, value: nn.Module) -> None:
        """Set the first convolutional layer."""
        self.conv2d1 = value

    def _prepare_batched_attention_mask(
        self,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Build a standard per-token padding mask for batched attention.

        Args:
            valid_mask: True at valid time positions and False at padded positions.
                Shape: [batch_size, seq_len]. Dtype: bool.

        Returns:
            Padding mask with True for valid tokens.
            Shape: [batch_size, seq_len]. Dtype: bool.
        """
        return valid_mask

    def _get_noncausal_chunk_length(self) -> int:
        """Return the expected non-causal attention chunk length after CNN.

        Returns:
            Chunk length in encoder frames. Dtype: int.
        """
        full_chunk_frames = self.n_window * 2
        frames_after_cnn = full_chunk_frames
        for _ in range(3):
            frames_after_cnn = (frames_after_cnn - 1) // 2 + 1
        scale = self.n_window_infer // (self.n_window * 2)
        return frames_after_cnn * scale

    def _apply_conv_stack(self, x: torch.Tensor) -> torch.Tensor:
        """Run the three Conv2d downsampling layers with GELU activations.

        Args:
            x: Input feature map.
                Shape: [batch_size, 1, num_mel_bins, num_frames]. Dtype: float.

        Returns:
            Downsampled feature map.
            Shape: [batch_size, hidden_channels, mel_bins_downsampled, frames_downsampled].
            Dtype: float.
        """
        x = F.gelu(self.conv2d1(x))
        x = F.gelu(self.conv2d2(x))
        x = F.gelu(self.conv2d3(x))
        return x

    def _apply_causal_conv_stack(self, x: torch.Tensor) -> torch.Tensor:
        """Run the three Conv2d downsampling layers with causal padding on the time axis.

        Each layer uses left-only padding along the time dimension (no future
        leakage) and symmetric padding along the frequency dimension. This
        ensures that each output time frame depends only on current and past
        input frames.

        Args:
            x: Input feature map.
                Shape: [batch_size, channels, num_mel_bins, num_frames]. Dtype: float.

        Returns:
            Downsampled feature map with causal time alignment.
            Shape: [batch_size, hidden_channels, mel_bins_downsampled, frames_downsampled].
            Dtype: float.
        """
        for conv in [self.conv2d1, self.conv2d2, self.conv2d3]:
            # Pad: (time_left, time_right, freq_left, freq_right)
            # Causal: left-pad time by kernel_size-1=2, no right-pad.
            # Symmetric: pad freq by 1 on each side (same as padding=1).
            x = F.pad(x, (2, 0, 1, 1))
            x = F.gelu(F.conv2d(x, conv.weight, conv.bias, stride=2))
        return x

    def _apply_causal_conv_stack_streaming(
        self,
        x: torch.Tensor,
        conv_caches: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run the causal Conv2d stack incrementally using cached border pixels.

        On each call the unconsumed tail frames (stored in ``conv_caches``) are
        prepended to the new input before padding and convolution, so that
        boundary frames are computed identically to the non-streaming path.

        The cache size varies depending on stride alignment: with kernel=3 and
        stride=2, the conv consumes frames in pairs.  If the combined input
        length is even, 1 frame is left over; if odd, 2 frames are left over.

        Args:
            x: New input feature map chunk.
                Shape: [1, channels, freq_dim, new_time_frames]. Dtype: float.
            conv_caches: List of 3 tensors, one per Conv2d layer.  Each holds
                the unconsumed tail frames from the previous call.
                Shape: [1, channels, freq_dim, 0..2]. Dtype: float.

        Returns:
            A tuple of:
                - Output feature map containing only the **new** output frames.
                    Shape: [1, hidden_channels, freq_downsampled, new_output_frames].
                    Dtype: float.
                - Updated ``conv_caches`` list (same structure, new tensors).
        """
        new_caches: list[torch.Tensor] = []
        for conv, cache in zip([self.conv2d1, self.conv2d2, self.conv2d3], conv_caches, strict=True):
            # Prepend unconsumed tail from the previous chunk.
            x_with_cache = torch.cat([cache, x], dim=3)
            combined_len = x_with_cache.shape[3]

            if combined_len < 3:
                # Not enough for even one conv output -- carry everything.
                new_caches.append(x_with_cache.clone())
                x = x_with_cache[:, :, :, :0]  # Empty time dim
                continue

            # Compute how many frames the conv will produce.
            num_outputs = (combined_len - 3) // 2 + 1
            # The next output would start at position 2 * num_outputs.
            # Everything from that position onward is unconsumed.
            unconsumed_start = 2 * num_outputs
            new_caches.append(x_with_cache[:, :, :, unconsumed_start:].clone())

            # Causal pad: freq symmetric (1,1), time already handled by cache.
            x_padded = F.pad(x_with_cache, (0, 0, 1, 1))
            x = F.gelu(F.conv2d(x_padded, conv.weight, conv.bias, stride=2))  # type: ignore
        return x, new_caches

    def _flatten_conv_output(self, conv_out_4d: torch.Tensor) -> torch.Tensor:
        """Flatten and project a 4D conv output to embedding space.

        Args:
            conv_out_4d: Output of the conv stack.
                Shape: [batch_size, hidden_channels, mel_bins_downsampled, frames_downsampled].
                Dtype: float.

        Returns:
            Projected embeddings.
            Shape: [batch_size, frames_downsampled, embed_dim]. Dtype: float.
        """
        batch_size, channels, freq, time = conv_out_4d.size()
        # [batch_size, channels, freq, time] -> [batch_size, time, channels * freq]
        return self.conv_out(conv_out_4d.permute(0, 3, 1, 2).contiguous().view(batch_size, time, channels * freq))

    def _downsample_chunked(
        self,
        input_features: torch.Tensor,
        feature_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """Downsample using the original chunked windowing strategy.

        Splits each sample's mel features into fixed-size ``n_window * 2``
        chunks, pads chunk tensors for Conv2d batching, and merges chunk
        outputs back per sample.

        This reproduces the exact behavior of the original HuggingFace
        audio encoder forward pass.

        Args:
            input_features: Padded batched mel spectrogram.
                Shape: [batch_size, num_mel_bins, max_frames]. Dtype: float.
            feature_lens: Per-sample mel frame counts.
                Shape: [batch_size]. Dtype: long.

        Returns:
            A tuple of:
                - hidden_states: Padded batched embeddings.
                    Shape: [batch_size, max_output_frames, embed_dim]. Dtype: float.
                - valid_mask: True at valid output positions.
                    Shape: [batch_size, max_output_frames]. Dtype: bool.
                - output_lens: Per-sample output frame counts.
        """
        output_lens: list[int] = []
        merged_sample_embeds: list[torch.Tensor] = []
        for sample_idx, sample_len_tensor in enumerate(feature_lens):
            sample_len = int(sample_len_tensor.item())
            chunk_count = math.ceil(sample_len / (self.n_window * 2))
            chunk_lengths = torch.tensor(
                [self.n_window * 2] * chunk_count,
                dtype=torch.long,
                device=input_features.device,
            )
            remainder = sample_len % (self.n_window * 2)
            if remainder != 0:
                chunk_lengths[-1] = remainder

            sample_feature = input_features[sample_idx, :, :sample_len]
            chunk_list = list(sample_feature.T.split(chunk_lengths.tolist(), dim=0))
            padded_feature = nn.utils.rnn.pad_sequence(chunk_list, batch_first=True).transpose(1, 2)
            padded_feature = padded_feature.unsqueeze(1)

            feature_lens_after_cnn = chunk_lengths
            for _ in range(3):
                feature_lens_after_cnn = (feature_lens_after_cnn - 1) // 2 + 1

            padded_embeds: list[torch.Tensor] = []
            for chunk in padded_feature.split(self.conv_chunksize, dim=0):
                padded_embeds.append(self._apply_conv_stack(chunk))
            padded_embed = torch.cat(padded_embeds, dim=0)
            padded_embed = self._flatten_conv_output(padded_embed)

            pos_embed_buffer: torch.Tensor = self.positional_embedding(padded_embed.shape[1])
            positional_embedding = pos_embed_buffer.unsqueeze(0).to(padded_embed.dtype)
            padded_embed = padded_embed + positional_embedding

            sample_chunk_embeds: list[torch.Tensor] = []
            for i in range(chunk_count):
                chunk_len = int(feature_lens_after_cnn[i].item())
                sample_chunk_embeds.append(padded_embed[i, :chunk_len])
            merged_embed = torch.cat(sample_chunk_embeds, dim=0)
            sample_len = int(merged_embed.shape[0])
            merged_sample_embeds.append(merged_embed)
            output_lens.append(sample_len)

        hidden_states = nn.utils.rnn.pad_sequence(merged_sample_embeds, batch_first=True)
        valid_mask = nn.utils.rnn.pad_sequence(
            [torch.ones(length, dtype=torch.bool, device=hidden_states.device) for length in output_lens],
            batch_first=True,
        )

        return hidden_states, valid_mask, output_lens

    def _downsample_causal(
        self,
        input_features: torch.Tensor,
        feature_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """Downsample without chunking using causal (left-only) time padding.

        Processes each sample's mel features independently through the causal
        conv stack (no windowed chunking), then pads outputs back to a batched
        tensor.

        Each Conv2d layer uses left-only padding along the time dimension,
        ensuring each output frame depends only on current and past input frames.

        Args:
            input_features: Padded batched mel spectrogram.
                Shape: [batch_size, num_mel_bins, max_frames]. Dtype: float.
            feature_lens: Per-sample mel frame counts.
                Shape: [batch_size]. Dtype: long.

        Returns:
            A tuple of:
                - hidden_states: Padded batched embeddings.
                    Shape: [batch_size, max_output_frames, embed_dim]. Dtype: float.
                - valid_mask: True at valid output positions.
                    Shape: [batch_size, max_output_frames]. Dtype: bool.
                - output_lens: Per-sample output frame counts.
        """
        all_embeds: list[torch.Tensor] = []
        output_lens: list[int] = []

        for sample_idx, sample_len_tensor in enumerate(feature_lens):
            sample_len = int(sample_len_tensor.item())
            # [num_frames, num_mel_bins] -> [1, 1, num_mel_bins, num_frames]
            x = input_features[sample_idx, :, :sample_len].unsqueeze(0).unsqueeze(0)
            conv_out = self._apply_causal_conv_stack(x)
            embed = self._flatten_conv_output(conv_out)  # [1, output_frames, embed_dim]

            pos_embed_buffer: torch.Tensor = self.positional_embedding(embed.shape[1])
            positional_embedding = pos_embed_buffer.unsqueeze(0).to(embed.dtype)
            embed = embed + positional_embedding

            output_frames = embed.shape[1]
            all_embeds.append(embed.squeeze(0))  # [output_frames, embed_dim]
            output_lens.append(output_frames)

        hidden_states = nn.utils.rnn.pad_sequence(all_embeds, batch_first=True)
        valid_mask = nn.utils.rnn.pad_sequence(
            [torch.ones(length, dtype=torch.bool, device=hidden_states.device) for length in output_lens],
            batch_first=True,
        )
        return hidden_states, valid_mask, output_lens

    def _downsample_causal_streaming(
        self,
        input_features: torch.Tensor,
        feature_lens: torch.Tensor,
        streaming_state: AuTStreamingState,
    ) -> tuple[torch.Tensor, list[int], AuTStreamingState]:
        """Downsample a single streaming chunk using cached conv border pixels.

        Processes the new mel frames through the causal conv stack (with cached
        border pixels from the previous call), flattens, and adds positional
        embeddings at the correct offset.

        Only supports ``batch_size=1`` (single-stream).

        Args:
            input_features: Packed mel spectrogram for this chunk (no batch dim).
                Shape: [num_mel_bins, new_frames]. Dtype: float.
            feature_lens: Frame count for this chunk.
                Shape: [1]. Dtype: long.
            streaming_state: Current streaming state with conv caches and
                positional offset.

        Returns:
            A tuple of:
                - hidden_states: Embeddings for the new output frames.
                    Shape: [new_output_frames, embed_dim]. Dtype: float.
                - output_lens: Single-element list with the number of new
                    output frames.
                - Updated ``AuTStreamingState``.
        """
        num_frames = int(feature_lens[0].item())
        # [num_mel_bins, num_frames] -> [1, 1, num_mel_bins, num_frames]
        x = input_features[:, :num_frames].unsqueeze(0).unsqueeze(0)

        conv_out, new_conv_caches = self._apply_causal_conv_stack_streaming(x, streaming_state.conv_caches)
        embed = self._flatten_conv_output(conv_out)  # [1, new_output_frames, embed_dim]

        new_output_frames = embed.shape[1]
        offset = streaming_state.num_frames_produced

        pos_embed_buffer: torch.Tensor = self.positional_embedding(offset + new_output_frames)
        positional_embedding = pos_embed_buffer[offset : offset + new_output_frames].unsqueeze(0).to(embed.dtype)
        embed = embed + positional_embedding

        hidden_states = embed.squeeze(0)  # [new_output_frames, embed_dim]

        new_state = AuTStreamingState(
            stft_cache=streaming_state.stft_cache,
            running_max=streaming_state.running_max,
            conv_caches=new_conv_caches,
            kv_caches=streaming_state.kv_caches,
            num_frames_produced=offset + new_output_frames,
        )

        return hidden_states, [new_output_frames], new_state

    def init_streaming_state(
        self,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> AuTStreamingState:
        """Create an initial (empty) streaming state for incremental encoding.

        The returned state should be passed to the first ``forward(...,
        streaming_state=...)`` call and then replaced with the state returned
        in the output on each subsequent call.

        Args:
            device: Device for the cache tensors.
            dtype: Dtype for the cache tensors.

        Returns:
            A fresh ``AuTStreamingState`` with zero-filled conv and KV caches.
        """
        num_mel_bins = self.num_mel_bins
        ds_hidden = self.conv2d1.out_channels  # downsample_hidden_size

        freq1 = (num_mel_bins + 1) // 2
        freq2 = (freq1 + 1) // 2
        freq3 = (freq2 + 1) // 2  # noqa: F841 — Kept for documentation

        conv_caches = [
            torch.zeros(1, 1, num_mel_bins, 2, device=device, dtype=dtype),
            torch.zeros(1, ds_hidden, freq1, 2, device=device, dtype=dtype),  # type: ignore
            torch.zeros(1, ds_hidden, freq2, 2, device=device, dtype=dtype),  # type: ignore
        ]

        num_heads = self.layers[0].self_attn.num_heads  # type: ignore
        head_dim = self.layers[0].self_attn.head_dim  # type: ignore
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for _ in self.layers:
            kv_caches.append(
                (
                    torch.zeros(1, num_heads, 0, head_dim, device=device, dtype=dtype),  # type: ignore
                    torch.zeros(1, num_heads, 0, head_dim, device=device, dtype=dtype),  # type: ignore
                )
            )

        return AuTStreamingState(
            stft_cache=None,
            running_max=float("-inf"),
            conv_caches=conv_caches,
            kv_caches=kv_caches,
            num_frames_produced=0,
        )

    def forward(
        self,
        input_features: torch.Tensor,
        feature_lens: torch.Tensor,
        causal: bool = False,
        streaming_state: AuTStreamingState | None = None,
    ) -> AuTEncoderOutput:
        """Run the full AuT encoder forward pass.

        Dispatches to the appropriate downsampling strategy, then runs the
        transformer encoder layers and output projection.

        When ``causal=False`` (default), uses the original chunked windowing
        downsampling with bidirectional attention (preserves HuggingFace
        equivalence).

        When ``causal=True``, uses non-chunked causal convolution downsampling
        and causal (autoregressive) attention.

        When ``streaming_state`` is provided, uses the streaming path (implies
        ``causal=True``, batch_size=1).

        Args:
            input_features: Mel spectrogram inputs.
                Non-streaming shape: [batch_size, num_mel_bins, max_frames].
                Streaming shape: [num_mel_bins, total_frames]. Dtype: float.
            feature_lens: Per-sample mel frame counts.
                Shape: [batch_size]. Dtype: long.
            causal: If True, use causal convolution and causal attention.
            streaming_state: If provided, run in streaming mode with cached
                state from the previous call.

        Returns:
            ``AuTEncoderOutput`` with ``last_hidden_state`` containing the
            projected encoder output, ``output_lens`` with per-sample output
            frame counts, and ``streaming_state`` if streaming mode is active.
        """
        if streaming_state is not None:
            # Streaming path: incremental causal encoding with KV cache.
            hidden_states, output_lens, streaming_state = self._downsample_causal_streaming(
                input_features, feature_lens, streaming_state
            )

            new_kv_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
            for encoder_layer, past_kv in zip(self.layers, streaming_state.kv_caches, strict=True):
                hidden_states, new_kv = encoder_layer.forward_streaming(hidden_states, past_kv)  # type: ignore
                new_kv_caches.append(new_kv)

            streaming_state = AuTStreamingState(
                stft_cache=streaming_state.stft_cache,
                running_max=streaming_state.running_max,
                conv_caches=streaming_state.conv_caches,
                kv_caches=new_kv_caches,
                num_frames_produced=streaming_state.num_frames_produced,
            )

            hidden_states = self.ln_post(hidden_states)
            hidden_states = self.proj1(hidden_states)
            hidden_states = self.act(hidden_states)
            hidden_states = self.proj2(hidden_states)
            return AuTEncoderOutput(
                last_hidden_state=hidden_states,
                output_lens=output_lens,
                streaming_state=streaming_state,
            )

        # Non-streaming path.
        if causal:
            hidden_states, valid_mask, output_lens = self._downsample_causal(input_features, feature_lens)
        else:
            hidden_states, valid_mask, output_lens = self._downsample_chunked(input_features, feature_lens)
            max_output_len = max(output_lens)
            chunk_length = self._get_noncausal_chunk_length()
            assert max_output_len <= chunk_length, (
                f"Non-causal AuT requires one attention chunk per sample. "
                f"Got max output length {max_output_len}, chunk length {chunk_length}."
            )

        attention_mask = self._prepare_batched_attention_mask(valid_mask=valid_mask)

        for encoder_layer in self.layers:
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                causal=causal,
                padding_mask=valid_mask,
            )
            hidden_states = layer_outputs[0]

        hidden_states = self.ln_post(hidden_states)
        hidden_states = self.proj1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.proj2(hidden_states)
        hidden_states = hidden_states * valid_mask.unsqueeze(-1).to(hidden_states.dtype)
        return AuTEncoderOutput(last_hidden_state=hidden_states, output_lens=output_lens)


def _load_audio_tower_state_dict(
    model_path: str,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Load audio tower weights from an ASR checkpoint.

    Reads the safetensors index (or a single safetensors file) to find which
    shards contain audio tower weights, then loads only those tensors with
    the ``thinker.audio_tower.`` prefix stripped.

    Args:
        model_path: HuggingFace model ID or local directory path.
        dtype: Target dtype for the loaded tensors.

    Returns:
        State dict with keys matching ``AuTEncoder`` (prefix stripped).
    """
    from raon.utils.misc import load_safetensors_by_prefix

    result = load_safetensors_by_prefix(
        model_path,
        prefixes={"audio_tower": _AUDIO_TOWER_PREFIX},
        dtype=dtype,
    )
    return result["audio_tower"]


def _load_audio_encoder_config(model_path: str) -> Qwen3OmniMoeAudioEncoderConfig:
    """Build an ``AuTEncoder``-compatible config from an ASR model config.json.

    Reads the nested ``thinker_config.audio_config`` and maps its fields into
    a ``Qwen3OmniMoeAudioEncoderConfig`` (which has an identical field set).

    Args:
        model_path: HuggingFace model ID or local directory path.

    Returns:
        Audio encoder config suitable for constructing ``AuTEncoder``.
    """
    is_local = os.path.isdir(model_path)
    if is_local:
        config_path = os.path.join(model_path, "config.json")
    else:
        config_path = hf_hub_download(repo_id=model_path, filename="config.json")

    with open(config_path) as f:
        full_config = json.load(f)

    audio_cfg = full_config["thinker_config"]["audio_config"]
    return Qwen3OmniMoeAudioEncoderConfig(
        d_model=audio_cfg["d_model"],
        encoder_layers=audio_cfg["encoder_layers"],
        encoder_attention_heads=audio_cfg["encoder_attention_heads"],
        encoder_ffn_dim=audio_cfg["encoder_ffn_dim"],
        num_mel_bins=audio_cfg["num_mel_bins"],
        max_source_positions=audio_cfg["max_source_positions"],
        n_window=audio_cfg["n_window"],
        n_window_infer=audio_cfg["n_window_infer"],
        output_dim=audio_cfg["output_dim"],
        conv_chunksize=audio_cfg["conv_chunksize"],
        downsample_hidden_size=audio_cfg["downsample_hidden_size"],
        scale_embedding=audio_cfg["scale_embedding"],
        activation_function=audio_cfg["activation_function"],
        dropout=audio_cfg.get("dropout", 0),
        attention_dropout=audio_cfg.get("attention_dropout", 0),
        activation_dropout=audio_cfg.get("activation_dropout", 0),
    )


class AuTWrapper(nn.Module):
    """Wrapper that runs the AuT encoder on raw audio waveforms.

    Handles resampling, feature extraction, and output length alignment.
    """

    config: Qwen3OmniMoeAudioEncoderConfig

    def __init__(
        self,
        config: Qwen3OmniMoeAudioEncoderConfig,
        feature_extractor: WhisperFeatureExtractor,
        encoder: AuTEncoder,
    ) -> None:
        super().__init__()
        self.config = config
        self.feature_extractor = feature_extractor
        self.encoder = encoder

        self.input_sample_rate = 24000
        self.encoder_sample_rate: int = feature_extractor.sampling_rate
        self.frame_rate = 12.5
        self.hidden_size = config.output_dim

        # Whisper STFT requires at least n_fft samples; pad shorter audio to avoid errors.
        self._min_encoder_samples: int = feature_extractor.n_fft

        self.config.sampling_rate = self.input_sample_rate

    @classmethod
    def from_config(
        cls,
        config: Qwen3OmniMoeAudioEncoderConfig,
        dtype: torch.dtype = torch.bfloat16,
    ) -> AuTWrapper:
        """Create an AuTWrapper with randomly-initialized weights from a config.

        Useful for building test checkpoints or when pretrained weights are not
        needed (e.g. the weights will be loaded from a saved checkpoint later).

        Args:
            config: Audio encoder config specifying architecture dimensions.
            dtype: Parameter dtype for the encoder.

        Returns:
            Initialized ``AuTWrapper`` with random weights.
        """
        feature_extractor = WhisperFeatureExtractor(
            feature_size=config.num_mel_bins,
            sampling_rate=16000,
        )
        aut_encoder = AuTEncoder(config)
        aut_encoder.to(dtype)  # type: ignore
        return cls(config=config, feature_extractor=feature_extractor, encoder=aut_encoder)

    @classmethod
    def from_asr_checkpoint(
        cls,
        pretrained_model_name_or_path: str,
        config: Qwen3OmniMoeAudioEncoderConfig | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> AuTWrapper:
        """Load an AuTWrapper from an ASR checkpoint.

        Extracts the audio tower weights from the full ASR model
        (``thinker.audio_tower.*``) and loads them into the local ``AuTEncoder``.

        Args:
            pretrained_model_name_or_path: HuggingFace model ID or local
                directory path.
            config: Optional override config. If None, derived from the
                checkpoint's ``config.json``.
            dtype: Parameter dtype for the encoder.

        Returns:
            Initialized ``AuTWrapper`` with pretrained audio encoder weights.
        """
        feature_extractor = WhisperFeatureExtractor.from_pretrained(
            pretrained_model_name_or_path,
        )

        if config is None:
            config = _load_audio_encoder_config(pretrained_model_name_or_path)

        state_dict = _load_audio_tower_state_dict(pretrained_model_name_or_path, dtype)

        aut_encoder = AuTEncoder(config)
        aut_encoder.load_state_dict(state_dict)
        aut_encoder.to(dtype)  # type: ignore

        return cls(config=config, feature_extractor=feature_extractor, encoder=aut_encoder)

    @classmethod
    def from_omni_checkpoint(
        cls,
        pretrained_model_name_or_path: str,
        config: Qwen3OmniMoeAudioEncoderConfig | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> AuTWrapper:
        """Load an AuTWrapper from a standalone audio encoder checkpoint.

        Loads the HuggingFace audio encoder weights and transfers them into
        the local ``AuTEncoder`` (which has an identical state dict layout).

        Args:
            pretrained_model_name_or_path: HuggingFace model ID or local path
                pointing to a standalone audio encoder checkpoint.
            config: Optional override config. If None, uses the checkpoint config.
            dtype: Parameter dtype for the encoder.

        Returns:
            Initialized ``AuTWrapper`` with pretrained weights.
        """
        feature_extractor = WhisperFeatureExtractor.from_pretrained(
            pretrained_model_name_or_path,
        )

        hf_encoder: Qwen3OmniMoeAudioEncoder = Qwen3OmniMoeAudioEncoder.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=dtype,
        )
        if config is None:
            config = hf_encoder.config

        aut_encoder = AuTEncoder(config)
        aut_encoder.load_state_dict(hf_encoder.state_dict())
        aut_encoder.to(dtype)  # type: ignore

        return cls(config=config, feature_extractor=feature_extractor, encoder=aut_encoder)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        config: Qwen3OmniMoeAudioEncoderConfig | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> AuTWrapper:
        """Load an AuTWrapper from a pretrained checkpoint.

        Automatically detects whether the checkpoint is an ASR model (with
        nested ``thinker_config.audio_config``) or a standalone audio encoder,
        and dispatches to the appropriate loader.

        Args:
            pretrained_model_name_or_path: HuggingFace model ID or local path.
            config: Optional override config. If None, derived from checkpoint.
            dtype: Parameter dtype for the encoder.

        Returns:
            Initialized ``AuTWrapper`` with pretrained weights.
        """
        is_local = os.path.isdir(pretrained_model_name_or_path)

        # Detect checkpoint type by checking for thinker_config in config.json.
        is_asr_checkpoint = False
        try:
            if is_local:
                config_path = os.path.join(pretrained_model_name_or_path, "config.json")
            else:
                config_path = hf_hub_download(
                    repo_id=pretrained_model_name_or_path,
                    filename="config.json",
                )
            with open(config_path) as f:
                raw_config = json.load(f)
            is_asr_checkpoint = "thinker_config" in raw_config
        except Exception:
            pass

        if is_asr_checkpoint:
            return cls.from_asr_checkpoint(pretrained_model_name_or_path, config=config, dtype=dtype)
        return cls.from_omni_checkpoint(pretrained_model_name_or_path, config=config, dtype=dtype)

    @property
    def device(self) -> torch.device:
        """Return the device of the encoder parameters."""
        return next(self.encoder.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Return the dtype of the encoder parameters."""
        return next(self.encoder.parameters()).dtype

    def compute_expected_output_length(self, num_samples: int) -> int:
        """Compute the expected number of output frames for a given number of audio samples.

        Args:
            num_samples: Number of input audio samples at ``input_sample_rate``.

        Returns:
            Expected number of output frames.
        """
        samples_per_frame = int(self.input_sample_rate / self.frame_rate)
        return math.ceil(num_samples / samples_per_frame)

    def _get_stft_params(self, device: torch.device) -> tuple[int, int, torch.Tensor, torch.Tensor]:
        """Return STFT parameters for mel feature extraction.

        Returns:
            A tuple of (n_fft, hop_length, window, mel_filters).
        """
        n_fft: int = self.feature_extractor.n_fft
        hop_length: int = self.feature_extractor.hop_length
        window = torch.hann_window(n_fft, device=device)
        mel_filters = torch.from_numpy(np.array(self.feature_extractor.mel_filters)).to(device=device, dtype=torch.float32)
        return n_fft, hop_length, window, mel_filters

    def _preprocess_audio(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Downmix stereo to mono, squeeze channel dim, and resample if needed.

        Args:
            audio: Raw audio waveform.
                Shape: [batch_size, num_channels, num_samples]. Dtype: float.
            audio_lengths: Valid sample lengths in ``audio``.
                Shape: [batch_size]. Dtype: long. May be ``None``.

        Returns:
            A tuple of (audio, audio_lengths) with the channel dim removed and
            sample rate converted to ``encoder_sample_rate``.
        """
        if audio.shape[1] == 2:
            audio = audio.mean(dim=1, keepdim=True)

        audio = audio.squeeze(1)
        if audio_lengths is not None:
            audio_lengths = audio_lengths.to(device=audio.device, dtype=torch.long)
            audio_lengths = audio_lengths.clamp(min=0, max=audio.shape[-1])

        if self.input_sample_rate != self.encoder_sample_rate:
            audio = torchaudio.functional.resample(
                waveform=audio,
                orig_freq=self.input_sample_rate,
                new_freq=self.encoder_sample_rate,
            )
            if audio_lengths is not None:
                audio_lengths = torch.floor(audio_lengths.float() * self.encoder_sample_rate / self.input_sample_rate).to(
                    torch.long
                )
                audio_lengths = audio_lengths.clamp(min=0, max=audio.shape[-1])

        return audio, audio_lengths

    def _extract_features(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract Whisper mel features from raw audio waveforms.

        Uses the standard centered STFT from the ``WhisperFeatureExtractor``.
        Downmixes stereo to mono, resamples to the encoder sample rate, runs
        the Whisper feature extractor, and returns padded batched mel features
        with per-sample frame lengths.

        Args:
            audio: Raw audio waveform.
                Shape: [batch_size, num_channels, num_samples]. Dtype: float.
            audio_lengths: Valid sample lengths in ``audio``.
                Shape: [batch_size]. Dtype: long.

        Returns:
            A tuple of:
                - batched_features: Padded mel spectrogram.
                    Shape: [batch_size, num_mel_bins, max_frames]. Dtype: float.
                - audio_feature_lengths: Per-sample mel frame counts.
                    Shape: [batch_size]. Dtype: long.
        """
        audio, audio_lengths = self._preprocess_audio(audio, audio_lengths)

        n_fft, hop_length, window, mel_filters = self._get_stft_params(device=self.device)

        if audio_lengths is None:
            audio_lengths = torch.full(
                (audio.shape[0],),
                audio.shape[-1],
                dtype=torch.long,
                device=audio.device,
            )

        # Match WhisperFeatureExtractor behavior for short clips:
        # each sample is padded to at least n_fft before batch padding.
        effective_lengths = torch.maximum(audio_lengths, torch.full_like(audio_lengths, n_fft))
        target_len = int(effective_lengths.max().item())

        if audio.shape[-1] < target_len:
            audio = F.pad(audio, (0, target_len - audio.shape[-1]))
        elif audio.shape[-1] > target_len:
            audio = audio[:, :target_len]

        # Remove tail samples beyond each sequence's valid length so padded
        # regions do not affect STFT/mel features.
        sample_indices = torch.arange(target_len, device=audio.device).unsqueeze(0)
        sample_mask = sample_indices < effective_lengths.unsqueeze(1)
        waveform = audio.to(device=self.device, dtype=torch.float32) * sample_mask.to(dtype=torch.float32)

        # Mirror WhisperFeatureExtractor._torch_extract_fbank_features.
        log_spec = compute_log_mel_spectrogram(waveform, window, mel_filters, n_fft, hop_length)

        # Match WhisperFeatureExtractor.__call__ mask rescaling exactly.
        feature_attention_mask = sample_mask[:, ::hop_length]
        if target_len % hop_length != 0:
            feature_attention_mask = feature_attention_mask[:, :-1]

        audio_feature_lengths = feature_attention_mask.sum(dim=1).to(dtype=torch.long, device=self.device)
        batched_features = log_spec.to(device=self.device, dtype=self.dtype)
        return batched_features, audio_feature_lengths

    def _extract_features_causal(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract mel features using a left-padded causal STFT.

        Prepends ``n_fft // 2`` zeros before running ``center=False`` STFT so
        that each frame is aligned to the same window center as the noncausal
        centered STFT.  The last STFT frame is dropped to match the noncausal
        frame count exactly.  Causal running-max normalization is applied: for
        each frame the floor is ``running_max - 8.0`` where ``running_max`` is
        the cumulative maximum of per-frame maxima up to that point.  This
        produces identical output whether the full utterance is processed at
        once (parallel) or chunk-by-chunk (streaming).

        Args:
            audio: Raw audio waveform.
                Shape: [batch_size, num_channels, num_samples]. Dtype: float.
            audio_lengths: Valid sample lengths in ``audio``.
                Shape: [batch_size]. Dtype: long.

        Returns:
            A tuple of:
                - batched_features: Padded mel spectrogram.
                    Shape: [batch_size, num_mel_bins, max_frames]. Dtype: float.
                - audio_feature_lengths: Per-sample mel frame counts.
                    Shape: [batch_size]. Dtype: long.
        """
        audio, audio_lengths = self._preprocess_audio(audio, audio_lengths)

        if audio_lengths is None:
            if audio.shape[-1] < self._min_encoder_samples:
                audio = F.pad(audio, (0, self._min_encoder_samples - audio.shape[-1]))
            audio_lengths = torch.full(
                (audio.shape[0],),
                audio.shape[-1],
                dtype=torch.long,
                device=audio.device,
            )

        n_fft, hop_length, window, mel_filters = self._get_stft_params(device=self.device)

        all_log_specs: list[torch.Tensor] = []
        frame_counts: list[int] = []

        for waveform, length in zip(audio, audio_lengths, strict=True):
            waveform_f32 = waveform[: int(length.item())].to(device=self.device, dtype=torch.float32)

            if waveform_f32.numel() < n_fft:
                waveform_f32 = F.pad(waveform_f32, (0, n_fft - waveform_f32.numel()))

            # Left-pad by n_fft // 2 to align frame centers with centered STFT.
            padded = F.pad(waveform_f32, (n_fft // 2, 0))
            stft = torch.stft(
                padded,
                n_fft,
                hop_length,
                window=window,
                center=False,
                return_complex=True,
            )
            # Drop the last frame to match centered STFT frame count.
            magnitudes = stft[..., :-1].abs() ** 2

            mel_spec = mel_filters.T @ magnitudes
            log_spec = torch.clamp(mel_spec, min=1e-10).log10()

            # Causal running-max normalization: the floor at each frame is
            # determined by the cumulative maximum of per-frame maxima seen so
            # far, matching what the streaming path computes incrementally.
            per_frame_max = log_spec.max(dim=0, keepdim=True)[0]  # [1, num_frames]
            running_max = torch.cummax(per_frame_max, dim=1)[0]  # [1, num_frames]
            log_spec = torch.maximum(log_spec, running_max.expand_as(log_spec) - 8.0)
            log_spec = (log_spec + 4.0) / 4.0

            num_frames = log_spec.shape[1]
            all_log_specs.append(log_spec)
            frame_counts.append(num_frames)

        batched_features = nn.utils.rnn.pad_sequence(
            [log_spec.T for log_spec in all_log_specs],
            batch_first=True,
        ).permute(0, 2, 1)
        batched_features = batched_features.to(device=self.device, dtype=self.dtype)
        audio_feature_lengths = torch.tensor(frame_counts, dtype=torch.long, device=self.device)

        return batched_features, audio_feature_lengths

    def _extract_features_causal_streaming(
        self,
        audio: torch.Tensor,
        stft_cache: torch.Tensor | None,
        running_max: float = float("-inf"),
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Extract causal mel features for a single streaming chunk.

        Uses left-padded causal STFT (``n_fft // 2`` zeros prepended on the
        first chunk) and causal running-max normalization to match the parallel
        ``_extract_features_causal`` output exactly.

        The parallel path drops the last STFT frame (``stft[..., :-1]``).  To
        match this in streaming, the waveform cache always retains enough
        samples so that the last STFT frame of the full sequence is never
        emitted until the next chunk proves it is not the final frame.  This is
        achieved by keeping ``hop_length`` extra tail samples in the cache
        beyond what is strictly unconsumed.

        Only supports ``batch_size=1``.

        Args:
            audio: Raw audio waveform chunk (mono, already at encoder sample rate).
                Shape: [1, num_samples]. Dtype: float.
            stft_cache: Unconsumed waveform samples carried over from the
                previous chunk.  Shape: [num_leftover_samples]. Dtype: float.
                ``None`` on the very first call.
            running_max: Running maximum of per-frame log-mel maxima from
                previous chunks.

        Returns:
            A tuple of:
                - packed_features: Mel spectrogram for the new frames only.
                    Shape: [num_mel_bins, new_frames]. Dtype: float.
                - audio_feature_lengths: Frame count for this chunk.
                    Shape: [1]. Dtype: long.
                - new_stft_cache: Unconsumed tail samples for the next call.
                    Shape: [num_leftover_samples]. Dtype: float.
                - new_running_max: Updated running maximum.
        """
        n_fft, hop_length, window, mel_filters = self._get_stft_params(device=self.device)

        waveform = audio[0].to(device=self.device, dtype=torch.float32)

        is_first_chunk = stft_cache is None
        if is_first_chunk:
            waveform = F.pad(waveform, (n_fft // 2, 0))
        else:
            waveform = torch.cat([stft_cache, waveform], dim=0)

        total_samples = waveform.shape[0]

        if total_samples < n_fft:
            packed_features = torch.zeros(
                mel_filters.shape[0],
                0,
                device=self.device,
                dtype=self.dtype,
            )
            audio_feature_lengths = torch.tensor([0], dtype=torch.long, device=self.device)
            return packed_features, audio_feature_lengths, waveform, running_max

        num_frames = (total_samples - n_fft) // hop_length + 1

        # Hold back the last frame to match the parallel path's [:-1] drop.
        # The held-back frame's samples stay in the cache and will be
        # re-computed (and emitted) when the next chunk arrives.
        if num_frames <= 1:
            packed_features = torch.zeros(
                mel_filters.shape[0],
                0,
                device=self.device,
                dtype=self.dtype,
            )
            audio_feature_lengths = torch.tensor([0], dtype=torch.long, device=self.device)
            return packed_features, audio_feature_lengths, waveform, running_max

        emit_frames = num_frames - 1
        consumed = (emit_frames - 1) * hop_length + n_fft

        stft = torch.stft(
            waveform[:consumed],
            n_fft,
            hop_length,
            window=window,
            center=False,
            return_complex=True,
        )
        magnitudes = stft.abs() ** 2
        mel_spec = mel_filters.T @ magnitudes
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()

        # Causal running-max normalization matching the parallel path.
        per_frame_max = log_spec.max(dim=0)[0]  # [emit_frames]
        new_running_max = running_max
        normalized_frames: list[torch.Tensor] = []
        for t in range(log_spec.shape[1]):
            frame_max = per_frame_max[t].item()
            new_running_max = max(new_running_max, frame_max)
            frame = torch.maximum(log_spec[:, t], log_spec.new_tensor(new_running_max - 8.0))
            normalized_frames.append(frame)
        if normalized_frames:
            log_spec = torch.stack(normalized_frames, dim=1)

        log_spec = (log_spec + 4.0) / 4.0

        # Cache starts at the beginning of the held-back frame.
        leftover_start = emit_frames * hop_length
        new_stft_cache = waveform[leftover_start:].clone()

        packed_features = log_spec.to(device=self.device, dtype=self.dtype)
        audio_feature_lengths = torch.tensor([emit_frames], dtype=torch.long, device=self.device)

        return packed_features, audio_feature_lengths, new_stft_cache, new_running_max

    def init_streaming_state(self) -> AuTStreamingState:
        """Create an initial streaming state for frame-by-frame encoding.

        Returns:
            A fresh ``AuTStreamingState`` ready for the first ``forward`` call
            with ``streaming_state=...``.
        """
        return self.encoder.init_streaming_state(device=self.device, dtype=self.dtype)

    def forward(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor | None = None,
        causal: bool = False,
        streaming_state: AuTStreamingState | None = None,
    ) -> CausalAudioEncoderOutput:
        """Encode raw audio waveforms into hidden state embeddings.

        Args:
            audio: Raw audio waveform.
                Shape: [batch_size, num_channels, num_samples]. Dtype: float.
            audio_lengths: Valid sample lengths in ``audio``.
                Shape: [batch_size]. Dtype: long.
            causal: If True, use causal feature extraction, causal convolution,
                and causal attention throughout the pipeline.
            streaming_state: If provided, run in incremental streaming mode
                (implies ``causal=True``, ``batch_size=1``). Pass the returned
                ``streaming_state`` from the output into the next call.

        Returns:
            ``CausalAudioEncoderOutput`` with the encoder embeddings.
            When streaming, ``embeds`` contains only the **new** output frames
            and ``streaming_state`` holds the updated state.
        """
        assert 1 <= audio.shape[1] <= 2, f"Number of audio channels must be 1 or 2, but got {audio.shape[1]}."

        if streaming_state is not None:
            # Streaming path: single-sample, causal only.
            audio, _ = self._preprocess_audio(audio, None)

            packed_features, audio_feature_lengths, new_stft_cache, new_running_max = (
                self._extract_features_causal_streaming(audio, streaming_state.stft_cache, streaming_state.running_max)
            )

            streaming_state = AuTStreamingState(
                stft_cache=new_stft_cache,
                running_max=new_running_max,
                conv_caches=streaming_state.conv_caches,
                kv_caches=streaming_state.kv_caches,
                num_frames_produced=streaming_state.num_frames_produced,
            )

            if audio_feature_lengths[0].item() == 0:
                # Not enough audio to produce any mel frames yet.
                return CausalAudioEncoderOutput(
                    embeds=torch.zeros(1, 0, self.hidden_size, device=self.device, dtype=self.dtype),
                )

            encoder_output = self.encoder(
                input_features=packed_features,
                feature_lens=audio_feature_lengths,
                streaming_state=streaming_state,
            )

            embeds = encoder_output.last_hidden_state.unsqueeze(0)  # [1, new_frames, output_dim]
            return CausalAudioEncoderOutput(
                embeds=embeds,
            )

        # Non-streaming path.
        num_samples = audio.shape[2]
        expected_output_length = self.compute_expected_output_length(num_samples)

        if causal:
            batched_features, audio_feature_lengths = self._extract_features_causal(audio, audio_lengths=audio_lengths)
        else:
            batched_features, audio_feature_lengths = self._extract_features(audio, audio_lengths=audio_lengths)

        encoder_output = self.encoder(
            input_features=batched_features,
            feature_lens=audio_feature_lengths,
            causal=causal,
        )
        last_hidden_state = encoder_output.last_hidden_state
        embeds = last_hidden_state

        actual_output_length = embeds.shape[1]
        if actual_output_length > expected_output_length:
            embeds = embeds[:, :expected_output_length]
        elif actual_output_length < expected_output_length:
            # Pad with zeros to match the expected length. The extra frames will be
            # masked out by audio_embeds_mask in get_audio_input_embeds, so they do
            # not affect the model. This can happen in causal mode where cumulative
            # rounding in the stride-2 conv stack produces fewer frames than the
            # ceil(num_samples / samples_per_frame) formula predicts.
            embeds = F.pad(embeds, (0, 0, 0, expected_output_length - actual_output_length))

        return CausalAudioEncoderOutput(embeds=embeds)
