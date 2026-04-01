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

"""Loss computation mixin for RaonModel: audio loss and text loss weighting."""

from __future__ import annotations

import logging
from typing import cast

import torch
import torch.nn.functional as F

from .delay import delay_audio_codes
from .special_tokens import (
    AUDIO_END,
    AUDIO_OUTPUT_BC,
    AUDIO_OUTPUT_END_PAD,
    AUDIO_OUTPUT_PAD,
    AUDIO_OUTPUT_PLACEHOLDER,
    DUPLEX_SIL,
    LOSS_IGNORE_INDEX,
)

logger = logging.getLogger(__name__)


class RaonLossMixin:
    """Mixin providing loss computation methods for RaonModel.

    Expects the following attributes on the concrete class (all set by RaonModel.__init__):
        audio_lm_head, proj_code, code_predictor, output_adaptor, speaker_encoder,
        audio_loss_weight, text_loss_weight, audio_output_pad_text_loss_weight,
        epad_loss_weight, audio_end_text_loss_weight, code_predictor_grad_scale,
        num_code_groups, audio_lm_head_vocab_size, codebook_size, max_delay, delays,
        supports_audio_output, use_duplex_end_pad, is_pretrained_speaker_encoder,
        speaker_token_id.

    Expects the following method on the concrete class:
        shift_labels(labels, pad_length=1) -> torch.Tensor
    """

    def unreduced_causal_lm_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute per-position cross-entropy loss for causal LM (no reduction).

        Args:
            logits: Language model logits. Shape: [batch_size, seq_length, vocab_size]. Dtype: float.
            labels: Ground-truth token IDs. Shape: [batch_size, seq_length]. Dtype: long.

        Returns:
            Per-position loss. Shape: [batch_size, seq_length]. Dtype: float.
        """
        _, seq_length, vocab_size = logits.shape
        logits = logits.reshape(-1, vocab_size).float()
        labels = self.shift_labels(labels, pad_length=seq_length - labels.shape[1] + 1).reshape(-1).to(logits.device)  # type: ignore[attr-defined]
        loss = F.cross_entropy(logits, labels, reduction="none", ignore_index=LOSS_IGNORE_INDEX)
        return loss.reshape(-1, seq_length)

    def _compute_audio_loss(
        self,
        hidden_embeds: torch.Tensor,
        audio_output_codes: torch.Tensor | None,
        audio_output_codes_mask: torch.Tensor | None,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Compute cross-entropy loss over audio codes for positions marked AUDIO_OUTPUT_PLACEHOLDER.

        Args:
            hidden_embeds: Talker hidden states. Shape: [batch_size, seq_length, hidden_size]. Dtype: float.
            audio_output_codes: Ground-truth audio codes. Shape: [batch_size, num_frames, num_code_groups].
                Dtype: long.
            audio_output_codes_mask: Valid frame mask. Shape: [batch_size, num_frames]. Dtype: bool.
            input_ids: Input token IDs used to identify audio-end prediction positions.
                Shape: [batch_size, seq_length]. Dtype: long.
            labels: Ground-truth token IDs used to identify audio frame positions.
                Shape: [batch_size, seq_length]. Dtype: long.

        Returns:
            Tuple of (audio_logits, audio_loss, audio_output_mask, audio_end_mask, audio_end_loss)
            when audio positions exist, else None.
            audio_logits Shape: [num_audio_positions, num_code_groups, codebook_size + 1]. Dtype: float.
            audio_loss Shape: [num_audio_positions, num_code_groups]. Dtype: float.
            audio_output_mask Shape: [batch_size, seq_length]. Dtype: bool.
            audio_end_mask Shape: [batch_size, seq_length]. Dtype: bool.
            audio_end_loss Shape: [num_audio_end_positions]. Dtype: float.
        """
        if audio_output_codes is None:
            return None
        assert self.audio_lm_head is not None, (
            "audio_lm_head is unavailable when supports_audio_output is False."
        )  # type: ignore[attr-defined]
        assert self.proj_code is not None, "proj_code is unavailable when supports_audio_output is False."  # type: ignore[attr-defined]
        assert self.code_predictor is not None, "code_predictor is unavailable when supports_audio_output is False."  # type: ignore[attr-defined]

        assert audio_output_codes_mask is not None, (
            "`audio_output_codes_mask` is required when `audio_output_codes` is provided."
        )
        shifted_labels = self.shift_labels(labels)  # type: ignore[attr-defined]
        shifted_input_ids = self.shift_labels(input_ids)  # type: ignore[attr-defined]
        audio_output_mask = shifted_labels == AUDIO_OUTPUT_PLACEHOLDER.id
        # Train AUDIO_END only for output-audio segments, not user input-audio tags.
        # Duplex: AUDIO_END appears in labels (first silence frame's [A]).
        # Pretrain/TTS: AUDIO_END appears in shifted_input_ids (token after last [A]).
        audio_end_mask = (labels == AUDIO_END.id) & (input_ids == AUDIO_OUTPUT_PLACEHOLDER.id)
        audio_end_mask |= (shifted_input_ids == AUDIO_END.id) & (input_ids == AUDIO_OUTPUT_PLACEHOLDER.id)
        if not audio_output_mask.any() and not audio_end_mask.any():
            return None

        audio_output_hidden_embeds = hidden_embeds[audio_output_mask.to(hidden_embeds.device)]
        audio_end_hidden_embeds = hidden_embeds[audio_end_mask.to(hidden_embeds.device)]
        audio_output_codes = audio_output_codes[audio_output_codes_mask.to(torch.bool)]

        # Apply delays for training if configured.
        # Moshi-style: use delayed codes as code predictor input and labels.
        if self.max_delay > 0:  # type: ignore[attr-defined]
            B = audio_output_codes.shape[0]
            assert B == 1, (
                f"Acoustic delay requires batch_size=1, got {B}. Flattened codes would leak across batch boundaries."
            )
            delayed_audio_codes = delay_audio_codes(self.delays, audio_output_codes, padding_value=0)  # type: ignore[attr-defined]
            delayed_audio_codes_labels = delay_audio_codes(
                self.delays,  # type: ignore[attr-defined]
                audio_output_codes,
                padding_value=LOSS_IGNORE_INDEX,
            )
        else:
            delayed_audio_codes = audio_output_codes
            delayed_audio_codes_labels = audio_output_codes

        audio_logits = torch.empty(
            0,
            self.num_code_groups,  # type: ignore[attr-defined]
            self.audio_lm_head_vocab_size,  # type: ignore[attr-defined]
            device=hidden_embeds.device,
            dtype=hidden_embeds.dtype,
        )
        audio_loss = torch.empty(0, self.num_code_groups, device=hidden_embeds.device, dtype=hidden_embeds.dtype)  # type: ignore[attr-defined]
        if audio_output_hidden_embeds.shape[0] > 0:
            # Truncate to shorter length when audio encoder and tokenizer frame counts
            # differ by a small amount (off-by-one from conv stride rounding).
            min_len = min(audio_output_hidden_embeds.shape[0], audio_output_codes.shape[0])
            if audio_output_hidden_embeds.shape[0] != audio_output_codes.shape[0]:
                audio_output_hidden_embeds = audio_output_hidden_embeds[:min_len]
                audio_output_codes = audio_output_codes[:min_len]
                delayed_audio_codes = delayed_audio_codes[:min_len]
                delayed_audio_codes_labels = delayed_audio_codes_labels[:min_len]
                # Keep the audio-position mask aligned with truncated audio codes.
                flat_audio_output_mask = audio_output_mask.reshape(-1)
                audio_output_indices = flat_audio_output_mask.nonzero(as_tuple=False).squeeze(-1)
                trimmed_flat_audio_output_mask = torch.zeros_like(flat_audio_output_mask, dtype=torch.bool)
                if min_len > 0:
                    trimmed_flat_audio_output_mask[audio_output_indices[:min_len]] = True
                audio_output_mask = trimmed_flat_audio_output_mask.view_as(audio_output_mask)
            audio_lm_head_logits = self.audio_lm_head(audio_output_hidden_embeds)  # type: ignore[attr-defined]

            grad_scaled_hidden_embeds = (
                self.code_predictor_grad_scale * audio_output_hidden_embeds  # type: ignore[attr-defined]
                + (1 - self.code_predictor_grad_scale) * audio_output_hidden_embeds.detach()  # type: ignore[attr-defined]
            )
            code_predictor_input_hidden_embeds = self.proj_code(grad_scaled_hidden_embeds)  # type: ignore[attr-defined]

            code_predictor_logits = self.code_predictor.parallel_forward(  # type: ignore[attr-defined]
                hidden_embeds=code_predictor_input_hidden_embeds,
                audio_codes=delayed_audio_codes,
            )
            code_predictor_logits = F.pad(
                code_predictor_logits,
                (0, 1),
                value=torch.finfo(code_predictor_logits.dtype).min,
            )

            audio_logits = torch.cat((audio_lm_head_logits[:, None], code_predictor_logits), dim=1)
            audio_loss = F.cross_entropy(
                audio_logits.reshape(-1, self.audio_lm_head_vocab_size),  # type: ignore[attr-defined]
                delayed_audio_codes_labels.reshape(-1),
                reduction="none",
                ignore_index=LOSS_IGNORE_INDEX,
            ).reshape(-1, self.num_code_groups)  # type: ignore[attr-defined]

        audio_end_loss = torch.empty(0, device=hidden_embeds.device, dtype=hidden_embeds.dtype)
        if audio_end_hidden_embeds.shape[0] > 0:
            audio_end_logits = self.audio_lm_head(audio_end_hidden_embeds)  # type: ignore[attr-defined]
            audio_end_targets = torch.full(
                (audio_end_logits.shape[0],),
                fill_value=self.codebook_size,  # type: ignore[attr-defined]
                dtype=torch.long,
                device=audio_end_logits.device,
            )
            audio_end_loss = F.cross_entropy(
                audio_end_logits,
                audio_end_targets,
                reduction="none",
            )

        return audio_logits, audio_loss, audio_output_mask, audio_end_mask, audio_end_loss

    def _dummy_audio_loss(self, hidden_embeds: torch.Tensor) -> torch.Tensor:
        """Return a zero scalar loss when no audio output positions exist, to keep gradients flowing."""
        assert self.audio_lm_head is not None, (
            "audio_lm_head is unavailable when supports_audio_output is False."
        )  # type: ignore[attr-defined]
        assert self.proj_code is not None, "proj_code is unavailable when supports_audio_output is False."  # type: ignore[attr-defined]
        assert self.code_predictor is not None, "code_predictor is unavailable when supports_audio_output is False."  # type: ignore[attr-defined]
        hidden_embeds = hidden_embeds[0, :1]
        audio_lm_head_logits = self.audio_lm_head(hidden_embeds)  # type: ignore[attr-defined]
        code_predictor_input_hidden_embeds = self.proj_code(hidden_embeds)  # type: ignore[attr-defined]
        dummy_audio_codes = torch.zeros(
            (1, self.num_code_groups),  # type: ignore[attr-defined]
            dtype=torch.long,
            device=code_predictor_input_hidden_embeds.device,
        )
        code_predictor_logits = self.code_predictor.parallel_forward(  # type: ignore[attr-defined]
            hidden_embeds=code_predictor_input_hidden_embeds,
            audio_codes=dummy_audio_codes,
        )
        code_predictor_logits = F.pad(code_predictor_logits, (0, 1), value=torch.finfo(code_predictor_logits.dtype).min)
        audio_logits = torch.cat((audio_lm_head_logits[:, None], code_predictor_logits), dim=1)
        result = 0 * audio_logits.sum(dim=-1)
        return result

    def _dummy_output_adaptor_loss(self) -> torch.Tensor:
        """Return a zero scalar tied to output_adaptor parameters for DDP safety."""
        if self.output_adaptor is None:  # type: ignore[attr-defined]
            return torch.tensor(0.0)
        first_param = next(self.output_adaptor.parameters(), None)  # type: ignore[attr-defined]
        if first_param is None:
            return torch.tensor(0.0)
        total = torch.zeros((), device=first_param.device, dtype=first_param.dtype)
        for parameter in self.output_adaptor.parameters():  # type: ignore[attr-defined]
            total = total + 0 * parameter.sum()
        return total

    def _dummy_speaker_loss(self) -> torch.Tensor:
        """Return a zero scalar when speaker_encoder exists but batch has no speaker positions; keeps DDP from hanging."""
        if self.speaker_encoder is None:  # type: ignore[attr-defined]
            return torch.tensor(0.0)

        first_param = next(self.speaker_encoder.parameters())  # type: ignore[attr-defined]
        if self.is_pretrained_speaker_encoder:  # type: ignore[attr-defined]
            # Minimum 24kHz duration to exceed ECAPA minimum after resample.
            dummy = torch.zeros(1, 4000, device=first_param.device, dtype=first_param.dtype)
            dummy_lengths = torch.tensor([dummy.shape[1]], device=dummy.device, dtype=torch.long)
            output = self.speaker_encoder(dummy, dummy_lengths)  # type: ignore[attr-defined]
        else:
            speaker_encoder_input_size = cast(int, self.speaker_encoder.input_size)  # type: ignore[attr-defined]
            dummy = torch.zeros(1, 1, speaker_encoder_input_size, device=first_param.device, dtype=first_param.dtype)
            dummy_mask = torch.ones(1, 1, dtype=torch.bool, device=dummy.device)
            output = self.speaker_encoder(dummy, mask=dummy_mask)  # type: ignore[attr-defined]

        return 0 * output.sum()

    def _apply_text_loss_weights(self, loss: torch.Tensor, text_labels: torch.Tensor) -> torch.Tensor:
        """Apply per-token-type loss weights to unreduced text loss.

        Args:
            loss: Per-position unreduced loss. Shape: [batch_size, seq_length]. Dtype: float.
            text_labels: Labels with AUDIO_OUTPUT_PLACEHOLDER already replaced. Shape: [batch_size, seq_length].
                Dtype: long.

        Returns:
            Weighted loss. Shape: [batch_size, seq_length]. Dtype: float.
        """
        shifted_text_labels = self.shift_labels(text_labels)  # type: ignore[attr-defined]
        assert (shifted_text_labels != AUDIO_OUTPUT_PLACEHOLDER.id).all(), (
            "text_labels must not contain AUDIO_OUTPUT_PLACEHOLDER tokens. "
            "Use get_text_labels(labels) to convert labels before passing to this method."
        )
        weighted_loss = loss.clone()

        pad_mask = shifted_text_labels == AUDIO_OUTPUT_PAD.id
        epad_mask = shifted_text_labels == AUDIO_OUTPUT_END_PAD.id
        audio_end_mask = shifted_text_labels == AUDIO_END.id
        text_mask = (
            (shifted_text_labels != AUDIO_OUTPUT_PAD.id)
            & (shifted_text_labels != AUDIO_OUTPUT_END_PAD.id)
            & (shifted_text_labels != AUDIO_END.id)
            & (shifted_text_labels != AUDIO_OUTPUT_BC.id)
            & (shifted_text_labels != LOSS_IGNORE_INDEX)
        )

        weighted_loss[text_mask] = weighted_loss[text_mask] * self.text_loss_weight  # type: ignore[attr-defined]
        weighted_loss[pad_mask] = weighted_loss[pad_mask] * self.audio_output_pad_text_loss_weight  # type: ignore[attr-defined]
        weighted_loss[audio_end_mask] = weighted_loss[audio_end_mask] * self.audio_end_text_loss_weight  # type: ignore[attr-defined]

        if self.use_duplex_end_pad:  # type: ignore[attr-defined]
            weighted_loss[epad_mask] = weighted_loss[epad_mask] * self.epad_loss_weight  # type: ignore[attr-defined]

        if self.use_sil_token:  # type: ignore[attr-defined]
            sil_mask = shifted_text_labels == DUPLEX_SIL.id
            weighted_loss[sil_mask] = weighted_loss[sil_mask] * self.sil_loss_weight  # type: ignore[attr-defined]

        if getattr(self, "use_backchannel_token", False):
            bc_mask = shifted_text_labels == AUDIO_OUTPUT_BC.id
            weighted_loss[bc_mask] = weighted_loss[bc_mask] * self.bc_loss_weight  # type: ignore[attr-defined]

        return weighted_loss

    def _combine_losses(
        self,
        text_loss: torch.Tensor | None,
        audio_loss: torch.Tensor | None,
        audio_output_mask: torch.Tensor | None,
        audio_end_loss: torch.Tensor | None,
        audio_end_mask: torch.Tensor | None,
        text_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Combine weighted text loss and audio loss into a single unreduced loss tensor.

        Args:
            text_loss: Per-position text loss. Shape: [batch_size, seq_length]. Dtype: float.
            audio_loss: Per-position-per-code-group audio loss. Shape: [num_audio_positions, num_code_groups].
                Dtype: float.
            audio_output_mask: Boolean mask for positions with audio. Shape: [batch_size, seq_length]. Dtype: bool.
            audio_end_loss: Per-position audio end loss. Shape: [num_audio_end_positions]. Dtype: float.
            audio_end_mask: Boolean mask for positions predicting AUDIO_END. Shape: [batch_size, seq_length].
                Dtype: bool.
            text_labels: Labels with placeholders replaced. Shape: [batch_size, seq_length]. Dtype: long.

        Returns:
            Combined per-position loss. Shape: [batch_size, seq_length]. Dtype: float.
        """
        shifted_text_labels = self.shift_labels(text_labels)  # type: ignore[attr-defined]
        if text_loss is None:
            assert audio_loss is not None, "audio_loss is required when text_loss is None."
            loss = torch.zeros(shifted_text_labels.shape, dtype=audio_loss.dtype, device=audio_loss.device)
        else:
            loss = self._apply_text_loss_weights(text_loss, text_labels)

        if audio_loss is not None:
            if audio_output_mask is None:
                loss += audio_loss.sum()
                logger.warning("WARNING from `RaonModel._combine_losses`: `audio_loss` given but not `audio_output_mask`.")
            else:
                num_audio_positions = int(audio_output_mask.sum().item())
                num_audio_loss_positions = int(audio_loss.shape[0])
                if num_audio_positions != num_audio_loss_positions:
                    min_len = min(num_audio_positions, num_audio_loss_positions)
                    logger.warning(
                        "WARNING from `RaonModel._combine_losses`: audio_output_mask/audio_loss length mismatch "
                        "(mask=%d, loss=%d). Clipping to %d.",
                        num_audio_positions,
                        num_audio_loss_positions,
                        min_len,
                    )
                    flat_audio_output_mask = audio_output_mask.reshape(-1)
                    audio_output_indices = flat_audio_output_mask.nonzero(as_tuple=False).squeeze(-1)
                    trimmed_flat_audio_output_mask = torch.zeros_like(flat_audio_output_mask, dtype=torch.bool)
                    if min_len > 0:
                        trimmed_flat_audio_output_mask[audio_output_indices[:min_len]] = True
                    audio_output_mask = trimmed_flat_audio_output_mask.view_as(audio_output_mask)
                    audio_loss = audio_loss[:min_len]
                self.audio_loss_weight = self.audio_loss_weight.to(loss.device)  # type: ignore[attr-defined]
                loss[audio_output_mask] += (self.audio_loss_weight * audio_loss).sum(dim=1)  # type: ignore[attr-defined]
        if audio_end_loss is not None and audio_end_loss.numel() > 0:
            if audio_end_mask is None:
                loss += audio_end_loss.sum()
                logger.warning("WARNING from `RaonModel._combine_losses`: `audio_end_loss` given but not `audio_end_mask`.")
            else:
                self.audio_loss_weight = self.audio_loss_weight.to(loss.device)  # type: ignore[attr-defined]
                loss[audio_end_mask] += self.audio_loss_weight[0] * audio_end_loss  # type: ignore[attr-defined]

        return loss

    def ddp_safe_loss(
        self,
        text_loss: torch.Tensor,
        text_labels: torch.Tensor,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        hidden_embeds: torch.Tensor,
        audio_output_codes: torch.Tensor | None,
        audio_output_codes_mask: torch.Tensor | None,
        speaker_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Compute audio loss, combine with text loss, and add DDP-safe dummy losses.

        Args:
            text_loss: Per-position unreduced text loss.
                Shape: [batch_size, seq_length]. Dtype: float.
            text_labels: Text-only labels (AUDIO_OUTPUT_PLACEHOLDER replaced).
                Shape: [batch_size, seq_length]. Dtype: long.
            input_ids: Input token IDs used to identify audio-end prediction positions.
                Shape: [batch_size, seq_length]. Dtype: long.
            labels: Ground-truth token IDs used to identify audio frame positions.
                Shape: [batch_size, seq_length]. Dtype: long.
            hidden_embeds: Talker hidden states.
                Shape: [batch_size, seq_length, hidden_size]. Dtype: float.
            audio_output_codes: Ground-truth audio codes.
                Shape: [batch_size, num_frames, num_code_groups]. Dtype: long.
            audio_output_codes_mask: Valid frame mask.
                Shape: [batch_size, num_frames]. Dtype: bool.
            speaker_embeds: Optional speaker conditioning.
                Shape: [batch_size, num_frames, feature_dim]. Dtype: float.

        Returns:
            Tuple of (loss, audio_loss, audio_logits).
            loss: Combined per-position loss with DDP-safe speaker term.
                Shape: [batch_size, seq_length]. Dtype: float.
            audio_loss: Per-position-per-code-group audio loss, or None when audio output is disabled.
                Shape: [num_audio_positions, num_code_groups]. Dtype: float.
            audio_logits: Audio logits if audio positions exist, else None.
                Shape: [num_audio_positions, num_code_groups, codebook_size]. Dtype: float.
        """
        audio_logits: torch.Tensor | None = None
        audio_loss: torch.Tensor | None = None
        audio_output_mask: torch.Tensor | None = None
        audio_end_loss: torch.Tensor | None = None
        audio_end_mask: torch.Tensor | None = None
        dummy_audio_loss: torch.Tensor | None = None
        dummy_output_adaptor_loss: torch.Tensor | None = None
        audio_outputs = self._compute_audio_loss(
            hidden_embeds=hidden_embeds,
            audio_output_codes=audio_output_codes,
            audio_output_codes_mask=audio_output_codes_mask,
            input_ids=input_ids,
            labels=labels,
        )
        if audio_outputs is not None:
            audio_logits, audio_loss, audio_output_mask, audio_end_mask, audio_end_loss = audio_outputs
        if self.supports_audio_output and (audio_loss is None or audio_loss.numel() == 0):  # type: ignore[attr-defined]
            dummy_audio_loss = self._dummy_audio_loss(hidden_embeds=hidden_embeds).sum()
            if audio_outputs is None:
                logger.warning("WARNING from `ddp_safe_loss`: using `_dummy_audio_loss`.")
        if self.supports_audio_output and (audio_output_mask is None or not audio_output_mask.any()):  # type: ignore[attr-defined]
            dummy_output_adaptor_loss = self._dummy_output_adaptor_loss()

        loss = self._combine_losses(
            text_loss=text_loss,
            audio_loss=audio_loss,
            audio_output_mask=audio_output_mask,
            audio_end_loss=audio_end_loss,
            audio_end_mask=audio_end_mask,
            text_labels=text_labels,
        )
        if dummy_audio_loss is not None:
            loss = loss + dummy_audio_loss
        if dummy_output_adaptor_loss is not None:
            loss = loss + dummy_output_adaptor_loss

        # Ensure speaker_encoder parameters always participate in backward pass
        # to prevent DDP hangs when batches have no speaker_token_id positions.
        if speaker_embeds is not None:
            loss = loss + 0 * speaker_embeds.sum()
        elif self.speaker_encoder is not None:  # type: ignore[attr-defined]
            loss = loss + self._dummy_speaker_loss()

        return loss, audio_loss, audio_logits
