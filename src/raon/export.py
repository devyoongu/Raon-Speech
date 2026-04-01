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

"""Export a RAON HuggingFace checkpoint to SGLang bundle format.

Usage:
    python -m raon.export \\
        --input_path /path/to/iter_0006000_hf \\
        --output_path /path/to/iter_0006000_sglang

    # With dtype override:
    python -m raon.export \\
        --input_path /path/to/hf_checkpoint \\
        --output_path /path/to/sglang_bundle \\
        --dtype float16
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch


def export_hf_to_sglang(
    input_path: str | Path,
    output_path: str | Path,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Export a RAON HuggingFace checkpoint to SGLang bundle.

    Splits the full model into:
    1. text_model/ — the text backbone, saved standalone for SGLang
    2. raon_runtime/ — the rest (audio encoder, talker, code predictor, etc.)

    Args:
        input_path: Path to the HuggingFace checkpoint directory.
        output_path: Path to the output SGLang bundle directory.
        dtype: Model dtype for loading/saving.
    """
    from copy import deepcopy

    from transformers import AutoConfig, AutoTokenizer, PreTrainedModel
    from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING

    from raon.models.raon import RaonModel
    from raon.utils.special_tokens import update_tokenizer

    input_path = Path(input_path)
    output_path = Path(output_path)

    assert input_path.exists(), f"Input path does not exist: {input_path}"

    if output_path.exists():
        print(f"Removing existing output at {output_path}")
        shutil.rmtree(output_path, ignore_errors=True)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── Load full model on CPU ──────────────────────────────────────────
    print(f"Loading model from {input_path} ...")
    config = AutoConfig.from_pretrained(input_path)
    model = RaonModel.from_pretrained(input_path, torch_dtype=dtype, device_map="cpu")
    print(f"Model loaded: {type(model).__name__}")

    # ── Step 1: Save text_model standalone ──────────────────────────────
    text_model_path = output_path / "text_model"
    text_model_path.mkdir(parents=True, exist_ok=True)

    print(f"Saving text_model to {text_model_path} ...")
    # Build a standalone ForCausalLM wrapper (text_model + lm_head) so SGLang
    # can load it as a standard causal LM. Without this, we'd save Qwen3Model
    # (no lm_head) instead of Qwen3ForCausalLM.
    text_model_config = deepcopy(config.text_model_config)
    causal_lm_class = MODEL_FOR_CAUSAL_LM_MAPPING[type(text_model_config)]
    causal_lm = object.__new__(causal_lm_class)
    PreTrainedModel.__init__(causal_lm, text_model_config)
    causal_lm.config = text_model_config
    causal_lm.model = model.text_model
    causal_lm.lm_head = model.lm_head
    causal_lm.save_pretrained(text_model_path)

    # Save tokenizer with special tokens
    tokenizer = AutoTokenizer.from_pretrained(input_path)
    update_tokenizer(tokenizer)

    vocab_size = int(config.text_model_config.vocab_size)
    assert vocab_size >= len(tokenizer), (
        f"Model vocab_size ({vocab_size}) < tokenizer vocab ({len(tokenizer)}). "
        "Tokenizer has tokens not covered by the model embedding."
    )
    tokenizer.save_pretrained(text_model_path)
    print(f"  text_model saved ({sum(1 for _ in causal_lm.parameters())} params)")

    # ── Step 2: Save raon_runtime (model without text_model) ───────────
    runtime_path = output_path / "raon_runtime"
    runtime_path.mkdir(parents=True, exist_ok=True)

    print(f"Saving raon_runtime to {runtime_path} ...")

    # Detach text_model to avoid saving it twice
    original_text_model = model.text_model
    model.text_model = None  # type: ignore[assignment]

    model.save_pretrained(runtime_path)

    # Restore text_model (in case caller wants to keep using the model)
    model.text_model = original_text_model

    # Also save tokenizer in runtime dir
    tokenizer.save_pretrained(runtime_path)
    print(f"  raon_runtime saved")

    # ── Summary ─────────────────────────────────────────────────────────
    print()
    print("Export complete!")
    print(f"  Input:        {input_path}")
    print(f"  text_model:   {text_model_path}")
    print(f"  raon_runtime: {runtime_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export RAON HuggingFace checkpoint to SGLang bundle.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the HuggingFace checkpoint directory (e.g., iter_0006000_hf).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to the output SGLang bundle directory (e.g., iter_0006000_sglang).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype (default: bfloat16).",
    )
    args = parser.parse_args()

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    export_hf_to_sglang(args.input_path, args.output_path, dtype=dtype_map[args.dtype])


if __name__ == "__main__":
    main()
