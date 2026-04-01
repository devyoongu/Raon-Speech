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

import enum
from dataclasses import dataclass
from typing import Literal

import torch

from .special_tokens import (
    AUDIO_END,
    AUDIO_INPUT_PLACEHOLDER,
    AUDIO_OUTPUT_BC,
    AUDIO_OUTPUT_END_PAD,
    AUDIO_OUTPUT_PAD,
    AUDIO_OUTPUT_PLACEHOLDER,
    AUDIO_START,
    DUPLEX_SIL,
    IM_END,
    IM_START,
    SPEAKER_EMBEDDING_PLACEHOLDER,
)


class DuplexPhase(enum.Enum):
    """Two-phase duplex state: silence or active speech."""

    SIL = "SIL"
    SPEECH = "SPEECH"


@dataclass
class DuplexMachineState:
    """Current Mealy machine state with last emitted frame tokens."""

    phase: DuplexPhase
    last_frame_tokens: list[int]

    @property
    def num_input_tokens(self) -> int:
        return len(self.last_frame_tokens)

    @property
    def emitted_audio(self) -> bool:
        return AUDIO_OUTPUT_PLACEHOLDER.id in self.last_frame_tokens or AUDIO_START.id in self.last_frame_tokens


@dataclass(frozen=True)
class DuplexStateConfig:
    """Immutable configuration for DuplexStateManager."""

    use_duplex_end_pad: bool = False
    use_sil_token: bool = False
    no_audio_in_sil: bool = False
    sequence_mode: Literal["tua", "uta"] | None = None
    duplex_pad_token_id: int = AUDIO_OUTPUT_PAD.id
    duplex_end_pad_token_id: int = AUDIO_OUTPUT_END_PAD.id
    duplex_sil_token_id: int = DUPLEX_SIL.id
    use_backchannel_token: bool = False
    duplex_bc_token_id: int = AUDIO_OUTPUT_BC.id

    @property
    def effective_sequence_mode(self) -> Literal["tua", "uta"]:
        return self.sequence_mode or "tua"


# Structural tokens that must never be sampled as text predictions.
_BLOCKED_STRUCTURAL: frozenset[int] = frozenset(
    {
        AUDIO_INPUT_PLACEHOLDER.id,
        AUDIO_OUTPUT_PLACEHOLDER.id,
        AUDIO_START.id,
        AUDIO_END.id,
        IM_START.id,
        IM_END.id,
        SPEAKER_EMBEDDING_PLACEHOLDER.id,
    }
)


class DuplexStateManager:
    """Mealy state machine for duplex inference: transitions + logit masking."""

    def __init__(self, config: DuplexStateConfig) -> None:
        self._config = config

    def initial_state(self, speak_first: bool = False) -> DuplexMachineState:
        """Return the initial machine state (SIL phase)."""
        return DuplexMachineState(
            phase=DuplexPhase.SIL,
            last_frame_tokens=[AUDIO_INPUT_PLACEHOLDER.id, AUDIO_OUTPUT_PLACEHOLDER.id],
        )

    def initial_forced_prediction_id(self, speak_first: bool) -> int | None:
        """Return the token ID to force for the first [U] prediction.

        Speak/listen mode is controlled via runtime token forcing.
        The first text-side prediction must be forced:
        - speak-first: force EPAD to enter onset/speech mode
        - listen-first: force SIL to remain in silence mode

        Returns:
            Token ID to force, or None when no explicit override is configured.
        """
        cfg = self._config
        if speak_first:
            if cfg.use_duplex_end_pad:
                return cfg.duplex_end_pad_token_id
            return None
        if cfg.use_sil_token:
            return cfg.duplex_sil_token_id
        return None

    def transition(
        self,
        state: DuplexMachineState,
        predicted_id: int,
        device: torch.device,
    ) -> tuple[DuplexMachineState, list[int], bool]:
        """Compute the next state and emitted frame tokens from a text prediction.

        Returns:
            Tuple of (new_state, frame_tokens, emitted_audio).
        """
        cfg = self._config
        aip = AUDIO_INPUT_PLACEHOLDER.id
        aop = AUDIO_OUTPUT_PLACEHOLDER.id
        sil_id = cfg.duplex_sil_token_id
        epad_id = cfg.duplex_end_pad_token_id
        pad_id = cfg.duplex_pad_token_id
        bc_id = cfg.duplex_bc_token_id
        is_uta = cfg.effective_sequence_mode == "uta"

        is_sil_prediction = predicted_id == sil_id

        if state.phase == DuplexPhase.SIL:
            if is_sil_prediction:
                tokens = [aip, aop]
                return DuplexMachineState(DuplexPhase.SIL, tokens), tokens, True

            if cfg.use_duplex_end_pad and predicted_id == epad_id:
                tokens = [aip, epad_id, aop] if is_uta else [epad_id, aip, aop]
                return DuplexMachineState(DuplexPhase.SPEECH, tokens), tokens, True

            # SIL -> SPEECH via backchannel onset.
            if cfg.use_backchannel_token and predicted_id == bc_id:
                tokens = [aip, bc_id, aop] if is_uta else [bc_id, aip, aop]
                return DuplexMachineState(DuplexPhase.SPEECH, tokens), tokens, True

            # SIL -> SPEECH via direct text.
            if is_uta:
                tokens = [aip, predicted_id, aop]
            else:
                tokens = [predicted_id, aip, aop]
            return DuplexMachineState(DuplexPhase.SPEECH, tokens), tokens, True

        # SPEECH phase
        if is_sil_prediction:
            tokens = [aip, aop]
            return DuplexMachineState(DuplexPhase.SIL, tokens), tokens, True

        if predicted_id == pad_id:
            tokens = [aip, aop]
            return DuplexMachineState(DuplexPhase.SPEECH, tokens), tokens, True

        if predicted_id == epad_id:
            tokens = [aip, epad_id, aop] if is_uta else [epad_id, aip, aop]
            return DuplexMachineState(DuplexPhase.SPEECH, tokens), tokens, True

        # SPEECH -> SPEECH (text token).
        if is_uta:
            tokens = [aip, predicted_id, aop]
        else:
            tokens = [predicted_id, aip, aop]
        return DuplexMachineState(DuplexPhase.SPEECH, tokens), tokens, True

    def apply_logit_mask(
        self,
        user_logits: torch.Tensor,
        state: DuplexMachineState,
        vocab_size: int,
    ) -> torch.Tensor:
        """Mask logits to enforce valid state-machine transitions.

        Returns:
            Masked logits with invalid tokens set to -inf.
        """
        cfg = self._config
        sil_id = cfg.duplex_sil_token_id
        epad_id = cfg.duplex_end_pad_token_id
        pad_id = cfg.duplex_pad_token_id
        bc_id = cfg.duplex_bc_token_id

        # Build onset token set (EPAD and optionally BC).
        onset_ids = {epad_id}
        if cfg.use_backchannel_token:
            onset_ids.add(bc_id)

        user_logits = user_logits.clone()
        mask = torch.full_like(user_logits, float("-inf"))
        max_token_id = mask.shape[-1]

        def _allow_token(token_id: int) -> None:
            if token_id < max_token_id:
                mask[:, :, token_id] = 0.0

        if state.phase == DuplexPhase.SIL:
            # SILENCE: only SIL, EPAD, or BC allowed.
            _allow_token(sil_id)
            if cfg.use_duplex_end_pad:
                _allow_token(epad_id)
            if cfg.use_backchannel_token:
                _allow_token(bc_id)
        elif state.phase == DuplexPhase.SPEECH:
            context_token = self._extract_context_token(state)

            if context_token is not None and context_token in onset_ids:
                # After EPAD/BC onset: only text tokens allowed.
                mask[:, :, :vocab_size] = 0.0
                for block_id in _BLOCKED_STRUCTURAL | {sil_id, pad_id} | onset_ids:
                    if block_id < mask.shape[-1]:
                        mask[:, :, block_id] = float("-inf")
            elif context_token is not None and context_token not in (_BLOCKED_STRUCTURAL | onset_ids | {pad_id, sil_id}):
                # After text: text + PAD + EPAD + SIL allowed (BC only from SIL phase).
                mask[:, :, :vocab_size] = 0.0
                _allow_token(pad_id)
                _allow_token(epad_id)
                _allow_token(sil_id)
                for block_id in _BLOCKED_STRUCTURAL:
                    if block_id < mask.shape[-1]:
                        mask[:, :, block_id] = float("-inf")
            else:
                # PAD frame (no context token): PAD + EPAD + SIL allowed.
                _allow_token(pad_id)
                _allow_token(epad_id)
                _allow_token(sil_id)

        return user_logits + mask

    def _extract_context_token(self, state: DuplexMachineState) -> int | None:
        """Extract the text or EPAD context token from the last frame."""
        tokens = state.last_frame_tokens
        if len(tokens) != 3:
            return None

        is_uta = self._config.effective_sequence_mode == "uta"
        if is_uta:
            return tokens[1]
        else:
            return tokens[0]
