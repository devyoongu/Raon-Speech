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

"""Projection from thinker hidden states to talker input space."""

from __future__ import annotations

import torch
from torch import nn


class ThinkerToTalkerProjection(nn.Module):
    """Projection from thinker hidden states to talker input space.

    Supports two modes:
    - ``"linear"``: RMSNorm (optional) followed by a single linear layer (no bias).
    - ``"mlp"``: Optional RMSNorm followed by a two-layer MLP with SiLU activation
      and bias, matching the original TalkerResizeMLP design.

    Args:
        thinker_hidden_size: Dimension of thinker hidden states.
        talker_hidden_size: Dimension of talker input.
        intermediate_size: Hidden dimension for the MLP mode. Required when
            ``mode="mlp"``, ignored for ``"linear"``.
        mode: Projection type — ``"linear"`` or ``"mlp"``.
        use_norm: If True, apply RMSNorm before projection (both modes).
        rms_norm_eps: Epsilon for RMSNorm.
    """

    def __init__(
        self,
        thinker_hidden_size: int,
        talker_hidden_size: int,
        intermediate_size: int | None = None,
        mode: str = "linear",
        use_norm: bool = True,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.norm: nn.RMSNorm | None = nn.RMSNorm(thinker_hidden_size, eps=rms_norm_eps) if use_norm else None
        if mode == "mlp":
            assert intermediate_size is not None, "intermediate_size is required for mlp mode."
            self.linear_fc1 = nn.Linear(thinker_hidden_size, intermediate_size, bias=True)
            self.linear_fc2 = nn.Linear(intermediate_size, talker_hidden_size, bias=True)
            self.act_fn = nn.SiLU()
            self.linear = None
        else:
            self.linear = nn.Linear(thinker_hidden_size, talker_hidden_size, bias=False)
            self.linear_fc1 = None
            self.linear_fc2 = None
            self.act_fn = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project thinker hidden states to talker input space.

        Args:
            hidden_states: Thinker hidden states.
                Shape: [batch_size, seq_len, thinker_hidden_size]. Dtype: float.

        Returns:
            Projected hidden states. Shape: [batch_size, seq_len, talker_hidden_size]. Dtype: float.
        """
        if self.norm is not None:
            hidden_states = self.norm(hidden_states)
        if self.mode == "mlp":
            return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_states)))
        return self.linear(hidden_states)
