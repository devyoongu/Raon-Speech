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

"""Shared HuggingFace Trainer callbacks for raon training scripts."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from transformers import TrainerCallback

if TYPE_CHECKING:
    from raon.utils.processor import RaonProcessor


class StepLoggingCallback(TrainerCallback):
    """Replace 'epoch' with 'step' in training logs."""

    def on_log(self, args, state, _control, logs=None, **kwargs):
        if logs is not None:
            logs.pop("epoch", None)
            logs["step"] = state.global_step


class SaveTokenizerCallback(TrainerCallback):
    """Save tokenizer alongside every checkpoint so checkpoints are self-contained."""

    def __init__(self, processor: "RaonProcessor") -> None:
        self.processor = processor

    def on_save(self, args, state, _control, **kwargs):
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        self.processor.tokenizer.save_pretrained(checkpoint_dir)
