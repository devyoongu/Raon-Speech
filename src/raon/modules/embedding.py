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

"""EmbeddingAdaptor: projects audio encoder embeddings into LM embedding space."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import (
    PretrainedConfig,
    Qwen3Config,
    Qwen3Model,
)
from transformers.utils.generic import ModelOutput


class EmbeddingAdaptorConfig(PretrainedConfig):
    """Configuration for EmbeddingAdaptor.

    Controls the projection from audio encoder embeddings to LM embedding space,
    including the time-scale ratio, MLP depth, optional transformer decoder, and
    optional post-projection RMSNorm.

    Args:
        input_size: Feature dimension of the encoder output (e.g. 512 for Mimi).
        output_size: Feature dimension expected by the LM (e.g. 4096 for Qwen3-7B).
        output_time_scale: Ratio of output frames to input frames. Values >= 1
            upsample (expand time); values < 1 downsample (compress time).
            Must be a reciprocal integer in either direction.
        num_layers: Number of MLP layers (1 or 2). Ignored in transformer mode.
        hidden_size: Hidden dimension for the 2-layer MLP. Defaults to output_size.
        decoder_config: If provided, uses a lightweight Qwen3 transformer instead
            of an MLP for the adaptor projection.
        use_post_norm: If True, apply RMSNorm to the output embeddings.
        norm_eps: Epsilon for RMSNorm.
        post_norm_init_scale: If set, initialize RMSNorm weight to this value
            (useful for residual scaling at initialisation).
    """

    model_type = "embedding_adaptor"

    def __init__(
        self,
        input_size: int = 512,
        output_size: int = 4096,
        output_time_scale: float = 1.0,
        num_layers: int = 1,
        hidden_size: int | None = None,
        decoder_config: dict[str, Any] | Qwen3Config | None = None,
        use_post_norm: bool = False,
        norm_eps: float = 1e-6,
        post_norm_init_scale: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.input_size = input_size
        self.output_size = output_size
        self.output_time_scale = output_time_scale
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.use_post_norm = use_post_norm
        self.norm_eps = norm_eps
        self.post_norm_init_scale = post_norm_init_scale

        # Parse decoder_config for transformer adaptor mode
        if isinstance(decoder_config, dict):
            decoder_config = Qwen3Config(**decoder_config)
        self.decoder_config = decoder_config


@dataclass
class EmbeddingAdaptorOutput(ModelOutput):
    """Output of EmbeddingAdaptor.forward().

    Attributes:
        outputs_embeds: Projected embeddings in the LM space.
            Shape: [batch, out_seq_len, output_size].
        mask: Boolean validity mask for the output sequence, or None if no mask
            was provided. Shape: [batch, out_seq_len].
    """

    outputs_embeds: torch.Tensor
    mask: torch.Tensor | None = None


class EmbeddingAdaptor(nn.Module):
    """Projects audio encoder embeddings into LM embedding space with optional time rescaling.

    Supports three backends controlled by constructor arguments:
    - 1-layer linear MLP (default)
    - 2-layer MLP with GELU activation
    - Lightweight Qwen3 transformer decoder (when ``decoder_config`` is provided)

    Time rescaling via ``output_time_scale`` allows matching different encoder/LM
    frame rates: values >= 1 upsample (reshape + linear), values < 1 downsample
    (stack adjacent frames then project).
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        output_time_scale: float = 1.0,
        num_layers: int = 1,
        hidden_size: int | None = None,
        decoder_config: Qwen3Config | None = None,
        use_post_norm: bool = False,
        norm_eps: float = 1e-6,
        post_norm_init_scale: float | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.output_time_scale = output_time_scale
        self.decoder_config = decoder_config

        if output_time_scale >= 1:
            scale = int(output_time_scale)
            assert scale == output_time_scale, (
                f"`output_time_scale` must be an integer when >= 1, got `{output_time_scale}`."
            )
            proj_input_size = input_size
            final_output_size = output_size * scale
        else:
            scale = int(1 / output_time_scale)
            assert scale == 1 / output_time_scale, (
                f"`1/output_time_scale` must be an integer when < 1, got `{output_time_scale}`."
            )
            proj_input_size = input_size * scale
            final_output_size = output_size

        # Check if we should use transformer mode
        if decoder_config is not None:
            # Transformer adaptor mode
            self.is_linear = False
            decoder_hidden_size = decoder_config.hidden_size
            self.input_proj = nn.Linear(
                proj_input_size,
                int(decoder_hidden_size * output_time_scale),
                bias=False,
                dtype=dtype,
            )
            self.decoder = Qwen3Model._from_config(decoder_config, dtype=dtype)
            # Remove unused embedding layer to save memory
            del self.decoder.embed_tokens
            self.decoder.embed_tokens = None  # type: ignore
            self.output_proj = nn.Linear(decoder_hidden_size, output_size, bias=False, dtype=dtype)
        elif num_layers == 1:
            # MLP mode (1 layer)
            self.is_linear = True
            self.proj = nn.Linear(proj_input_size, final_output_size, bias=False, dtype=dtype)
        elif num_layers == 2:
            # MLP mode (2 layers)
            self.is_linear = True
            hidden = hidden_size or final_output_size
            self.proj = nn.Sequential(
                nn.Linear(proj_input_size, hidden, bias=False, dtype=dtype),
                nn.GELU(),
                nn.Linear(hidden, final_output_size, bias=False, dtype=dtype),
            )
        else:
            raise ValueError(f"num_layers must be 1 or 2, got {num_layers}")

        self.post_norm = nn.RMSNorm(output_size, eps=norm_eps, dtype=dtype) if use_post_norm else None
        if self.post_norm is not None and post_norm_init_scale is not None:
            self.post_norm.weight.data.fill_(post_norm_init_scale)

    @classmethod
    def from_config(cls, config: Any, *, dtype: torch.dtype | None = None) -> EmbeddingAdaptor:
        """Create an EmbeddingAdaptor from a config object."""
        return cls(
            input_size=config.input_size,
            output_size=config.output_size,
            output_time_scale=config.output_time_scale,
            num_layers=getattr(config, "num_layers", 1),
            hidden_size=getattr(config, "hidden_size", None),
            decoder_config=getattr(config, "decoder_config", None),
            use_post_norm=getattr(config, "use_post_norm", False),
            norm_eps=getattr(config, "norm_eps", 1e-6),
            post_norm_init_scale=getattr(config, "post_norm_init_scale", None),
            dtype=dtype,
        )

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor | None = None) -> EmbeddingAdaptorOutput:
        """Project encoder embeddings to LM space, applying time rescaling.

        Args:
            inputs: Encoder output embeddings. Shape: [batch, seq_len, input_size].
            mask: Optional boolean validity mask. Shape: [batch, seq_len].
                True indicates a valid (non-padded) frame.

        Returns:
            EmbeddingAdaptorOutput with projected embeddings of shape
            [batch, out_seq_len, output_size] and a corresponding mask.
            When output_time_scale >= 1, out_seq_len = seq_len * scale (upsample).
            When output_time_scale < 1, out_seq_len = ceil(seq_len / scale) (downsample).
        """
        batch_size, seq_length, _ = inputs.shape

        # output_time_scale >= 1: upsample -- each input frame expands to `scale` output frames.
        #   The projection output dimension is output_size * scale; a view() then splits
        #   it into (seq_len * scale, output_size).
        # output_time_scale < 1: downsample -- every `scale` consecutive input frames are
        #   concatenated along the feature axis before projection, reducing sequence length
        #   by a factor of `scale`. The sequence is right-padded with the last frame if
        #   its length is not divisible by `scale`.
        if self.output_time_scale >= 1:
            scale = int(self.output_time_scale)

            if self.is_linear:
                # MLP mode
                outputs_embeds = self.proj(inputs)
            else:
                # Transformer mode
                inputs_embeds = self.input_proj(inputs)
                # Convert mask to attention mask format if provided
                attention_mask = mask.to(inputs_embeds.dtype) if mask is not None else None
                decoder_outputs = self.decoder(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                )
                outputs_embeds = self.output_proj(decoder_outputs.last_hidden_state)

            outputs_embeds = outputs_embeds.view(batch_size, seq_length * scale, self.output_size)

            if mask is not None:
                output_mask = mask.repeat_interleave(scale, dim=1)
            else:
                output_mask = None
        else:
            scale = int(1 / self.output_time_scale)
            remainder = seq_length % scale
            if remainder != 0:
                padding_length = scale - remainder
                last_embed = inputs[:, -1:].expand(-1, padding_length, -1)
                inputs = torch.cat([inputs, last_embed], dim=1)
                if mask is not None:
                    mask = F.pad(mask, (0, padding_length), value=False)

            new_seq_length = inputs.shape[1] // scale
            inputs = inputs.view(batch_size, new_seq_length, scale * self.input_size)

            if self.is_linear:
                # MLP mode
                outputs_embeds = self.proj(inputs)
            else:
                # Transformer mode
                inputs_embeds = self.input_proj(inputs)
                attention_mask = mask.to(inputs_embeds.dtype) if mask is not None else None
                decoder_outputs = self.decoder(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                )
                outputs_embeds = self.output_proj(decoder_outputs.last_hidden_state)

            if mask is not None:
                output_mask = mask.view(batch_size, new_seq_length, scale).any(dim=-1)
            else:
                output_mask = None

        if self.post_norm is not None:
            outputs_embeds = self.post_norm(outputs_embeds)

        return EmbeddingAdaptorOutput(outputs_embeds=outputs_embeds, mask=output_mask)
