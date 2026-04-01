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
# This module wraps Qwen3OmniMoeTalkerCodePredictorModel from HuggingFace Transformers.

"""Autoregressive audio code predictor model."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from einops import einsum
from torch import nn
from transformers import (
    GenerationMixin,
    Qwen3OmniMoePreTrainedModel,
    Qwen3OmniMoeTalkerCodePredictorModel,
    StaticCache,
)
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeTalkerCodePredictorConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeTalkerCodePredictorOutputWithPast

from ..utils.misc import cast_to_module_dtype


class RaonCodePredictorModel(Qwen3OmniMoePreTrainedModel, GenerationMixin):  # type: ignore
    """Code predictor for autoregressive audio code generation with fused codec embedding."""

    config_class: type[Qwen3OmniMoeTalkerCodePredictorConfig] = Qwen3OmniMoeTalkerCodePredictorConfig  # type: ignore[assignment]

    def __init__(self, config: Qwen3OmniMoeTalkerCodePredictorConfig):
        super().__init__(config)
        self.num_code_groups = config.num_code_groups
        _dtype = getattr(config, "torch_dtype", None) or torch.float32
        if isinstance(_dtype, str):
            _dtype = getattr(torch, _dtype, torch.float32)
        self.model = Qwen3OmniMoeTalkerCodePredictorModel._from_config(config, dtype=_dtype)
        input_embeddings = self.model.get_input_embeddings()
        assert isinstance(input_embeddings, nn.ModuleList), "Expected input embeddings to be a ModuleList."
        weights: list[torch.Tensor] = []
        for i in range(self.num_code_groups):
            embed = input_embeddings[i - 1]
            assert isinstance(embed, nn.Embedding)
            weights.append(embed.weight)

        fused_code_embed_weight = torch.cat(weights)
        self.codec_embedding = nn.Embedding(
            fused_code_embed_weight.shape[0],
            fused_code_embed_weight.shape[1],
            dtype=fused_code_embed_weight.dtype,
        )
        with torch.no_grad():
            self.codec_embedding.weight.copy_(fused_code_embed_weight)

        del self.model.codec_embedding
        self.vocab_size = config.vocab_size
        self.fused_lm_head = nn.Parameter(
            torch.randn(
                self.num_code_groups - 1,
                self.vocab_size,
                self.config.hidden_size,
                dtype=_dtype,
            )
            * (self.config.hidden_size**-0.5)
        )
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.Tensor | None = None,
        generation_steps: int | None = None,
        **kwargs: Any,
    ) -> Qwen3OmniMoeTalkerCodePredictorOutputWithPast:
        """Run one autoregressive step of code-group prediction.

        During prefill (inputs_embeds provided with seq_len > 1), generation_steps
        is derived from the sequence length. During decoding (input_ids provided),
        generation_steps must be supplied to select the correct per-step LM head
        and to offset the fused codec embedding lookup.

        Args:
            input_ids: Token ids for single-step decoding. Shape: [batch, 1]. Dtype: long.
            attention_mask: Attention mask for the full sequence. Shape: [batch, seq_len].
            position_ids: Position ids. Shape: [batch, seq_len].
            past_key_values: KV cache from previous steps.
            inputs_embeds: Precomputed embeddings (used during prefill).
                Shape: [batch, seq_len, hidden_size].
            use_cache: Whether to return updated KV cache.
            cache_position: Absolute positions of the current tokens in the cache.
            generation_steps: Which code group is being predicted (0-indexed).
                Inferred from inputs_embeds during prefill; required during decoding.
            **kwargs: Additional arguments forwarded to the inner model.

        Returns:
            Qwen3OmniMoeTalkerCodePredictorOutputWithPast with logits of shape
            [batch, seq_len, vocab_size], updated past_key_values, and
            generation_steps incremented by 1.
        """
        inputs_embeds = cast_to_module_dtype(inputs_embeds, self)
        if inputs_embeds is not None and inputs_embeds.shape[1] > 1:
            generation_steps = inputs_embeds.shape[1] - 2
        else:
            assert input_ids is not None and generation_steps is not None, f"{input_ids=}, {generation_steps=}"
            inputs_embeds = self.get_input_embeddings()(input_ids + generation_steps * self.vocab_size)

        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        logits = F.linear(outputs.last_hidden_state, self.fused_lm_head[generation_steps])
        return Qwen3OmniMoeTalkerCodePredictorOutputWithPast(
            logits=logits,  # type: ignore
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            generation_steps=generation_steps + 1,
        )

    def parallel_forward(self, hidden_embeds: torch.Tensor, audio_codes: torch.Tensor) -> torch.Tensor:
        """Predict all code groups in parallel given hidden states and teacher-forced codes.

        Args:
            hidden_embeds: Hidden states from the LM. Shape: [batch_size, hidden_size].
                Dtype: float.
            audio_codes: Teacher-forced audio codes (all but last group).
                Shape: [batch_size, num_code_groups]. Dtype: long.

        Returns:
            Logits for the next code group at each position. Shape: [batch_size,
            num_code_groups - 1, vocab_size]. Dtype: float.
        """
        hidden_embeds = cast_to_module_dtype(hidden_embeds, self)
        generation_step = torch.arange(self.config.num_code_groups - 1, device=audio_codes.device)
        audio_code_embeds = self.codec_embedding(audio_codes[:, :-1] + generation_step * self.vocab_size)
        inputs_embeds = torch.cat((hidden_embeds[:, None], audio_code_embeds), dim=1).contiguous()
        last_hidden_state = self.model(inputs_embeds=inputs_embeds).last_hidden_state
        logits: torch.Tensor = einsum(last_hidden_state[:, 1:], self.fused_lm_head, "b s h, s c h -> b s c")
        return logits

    def generate_greedy(self, inputs_embeds: torch.Tensor, past_key_values: StaticCache) -> torch.Tensor:
        """Generate audio codes greedily given initial embeddings and KV cache.

        Args:
            inputs_embeds: Initial input embeddings. Shape: [batch_size, seq_length,
                hidden_size]. Dtype: float.
            past_key_values: StaticCache holding past KV for incremental decoding.

        Returns:
            Greedily sampled code sequence. Shape: [batch_size, num_code_groups - 1].
            Dtype: long.
        """
        cache_position = torch.arange(2, device=inputs_embeds.device)
        optional_input_ids: torch.Tensor | None = None
        optional_inputs_embeds: torch.Tensor | None = inputs_embeds
        sequences = torch.empty(
            (inputs_embeds.shape[0], self.num_code_groups - 1),
            dtype=torch.int64,
            device=inputs_embeds.device,
        )
        for i in range(self.num_code_groups - 1):
            logits: torch.Tensor = self(
                input_ids=optional_input_ids,
                inputs_embeds=optional_inputs_embeds,
                past_key_values=past_key_values,
                cache_position=cache_position,
                generation_steps=i,
            ).logits
            optional_inputs_embeds = None
            optional_input_ids = logits[:, -1:].argmax(dim=-1)
            cache_position = cache_position[-1:] + 1
            sequences[:, i] = optional_input_ids[:, -1]

        return sequences

    def _update_model_kwargs_for_generation(  # type: ignore
        self,
        outputs: Qwen3OmniMoeTalkerCodePredictorOutputWithPast,
        model_kwargs: dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> dict[str, Any]:
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs=outputs,
            model_kwargs=model_kwargs,
            is_encoder_decoder=is_encoder_decoder,
            num_new_tokens=num_new_tokens,
        )
        model_kwargs["generation_steps"] = outputs.generation_steps
        return model_kwargs

    def predict_codes(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """Predict full audio code sequence from input embeddings via greedy generation.

        Args:
            inputs_embeds: Input embeddings. Shape: [batch_size, seq_length, hidden_size].
                Dtype: float.

        Returns:
            Predicted audio codes. Shape: [batch_size, num_code_groups - 1]. Dtype: long.
        """
        inputs_embeds = cast_to_module_dtype(inputs_embeds, self)
        past_key_values = StaticCache(self.config, max_cache_len=self.num_code_groups, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        return self.generate_greedy(inputs_embeds=inputs_embeds, past_key_values=past_key_values)

    def get_input_embeddings(self) -> nn.Embedding:
        """Return the fused codec embedding layer."""
        return self.codec_embedding
