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

"""Fine-tuning script for the RAON model.

Usage:
    python -m raon.train \
        --model_path /path/to/model \
        --data_dir /path/to/data_dir \
        --output_dir /path/to/output \
        --max_steps 1000 \
        --batch_size 1 \
        --learning_rate 1e-5 \
        --dtype bfloat16
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import torch
from transformers import Trainer, TrainerCallback, TrainingArguments

from raon.models.raon import RaonModel
from raon.utils.data import DatasetConfig, make_raon_data_module
from raon.utils.processor import RaonProcessor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom Trainer
# ---------------------------------------------------------------------------


from raon.utils.training_callbacks import StepLoggingCallback, SaveTokenizerCallback


class RaonTrainer(Trainer):
    def __init__(self, *args, use_speaker_embedding: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._use_speaker_embedding = use_speaker_embedding

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs["use_speaker_embedding"] = self._use_speaker_embedding
        return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune the Raon model.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the pretrained model directory.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help=(
            "Data path(s) for training. Accepts: "
            "a single jsonl, comma-separated jsonl files, "
            "a single directory, or comma-separated directories. "
            "Directories are expanded to all *.jsonl files inside."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where checkpoints and the final model will be saved.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=1000,
        help="Maximum number of training steps (default: 1000).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Per-device training batch size (default: 1).",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate (default: 1e-5).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype (default: bfloat16).",
    )
    parser.add_argument(
        "--max_audio_seq_length",
        type=int,
        default=192000,
        help="Maximum audio sequence length in samples for chunking (default: 192000).",
    )
    parser.add_argument(
        "--use_speaker_embedding",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable speaker conditioning for TTS training (default: disabled).",
    )
    parser.add_argument(
        "--use_packing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable sequence packing for training (default: enabled). Use --no-use_packing to disable.",
    )
    parser.add_argument(
        "--max_packed_seq_length",
        type=int,
        default=8192,
        help="Maximum packed sequence length in tokens (default: 8192).",
    )
    parser.add_argument(
        "--log_first_n_batches",
        type=int,
        default=0,
        help="Log shapes of the first N batches for debugging (default: 0).",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        choices=["fa", "sdpa", "eager"],
        help="Attention implementation (default: sdpa). Use `fa` for FlashAttention.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps (default: 500).",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="cosine",
        help="Learning rate scheduler type (default: cosine).",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=10,
        help="Number of warmup steps for the learning rate scheduler (default: 10).",
    )
    # Loss weights
    parser.add_argument("--text_loss_weight", type=float, default=None, help="Text token loss weight.")
    parser.add_argument(
        "--audio_output_pad_text_loss_weight", type=float, default=None, help="Audio output pad token loss weight."
    )
    parser.add_argument("--audio_end_text_loss_weight", type=float, default=None, help="Audio end token loss weight.")
    parser.add_argument("--semantic_loss_weight", type=float, default=None, help="Semantic codebook loss weight.")
    parser.add_argument(
        "--acoustic_loss_weight", type=float, default=None, help="Acoustic codebook loss weight (applied to all groups)."
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    log_level = logging.INFO if local_rank == 0 else logging.WARNING
    logging.basicConfig(level=log_level, format="%(name)s - %(levelname)s - %(message)s")
    args = parse_args()
    if args.attn_implementation == "fa":
        args.attn_implementation = "flash_attention_2"

    # Resolve paths.
    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Map dtype string to torch dtype.
    from raon.utils.misc import resolve_dtype

    torch_dtype = resolve_dtype(args.dtype)

    logger.info("Loading model from %s ...", model_path)
    model = RaonModel.from_pretrained(str(model_path), torch_dtype=torch_dtype)
    model._set_attention_implementation(args.attn_implementation)
    logger.info("Attention implementation: %s", args.attn_implementation)
    logger.info("Speaker embedding enabled: %s", args.use_speaker_embedding)

    # Freeze modules: audio_encoder + aligner modules frozen, LLM backbone trainable.
    _frozen_modules = [
        "audio_encoder", "input_adaptor", "output_adaptor",
        "audio_lm_head", "proj_code", "code_predictor", "speaker_encoder",
    ]
    for module_name in _frozen_modules:
        module = getattr(model, module_name, None)
        if module is not None:
            for param in module.parameters():
                param.requires_grad = False
            logger.info("Froze %s", module_name)

    # Override loss weights from CLI (None = keep model default from config)
    if args.text_loss_weight is not None:
        model.text_loss_weight = args.text_loss_weight
    if args.audio_output_pad_text_loss_weight is not None:
        model.audio_output_pad_text_loss_weight = args.audio_output_pad_text_loss_weight
    if args.audio_end_text_loss_weight is not None:
        model.audio_end_text_loss_weight = args.audio_end_text_loss_weight
    if args.semantic_loss_weight is not None:
        model.semantic_loss_weight = args.semantic_loss_weight
    if args.acoustic_loss_weight is not None:
        model.acoustic_loss_weights = [args.acoustic_loss_weight] * model.num_code_groups
    # Rebuild audio_loss_weight tensor to reflect any CLI overrides above.
    if any(v is not None for v in [args.semantic_loss_weight, args.acoustic_loss_weight]):
        model.audio_loss_weight = torch.tensor(
            [model.semantic_loss_weight] + model.acoustic_loss_weights,
            dtype=model.audio_loss_weight.dtype,
        )

    logger.info("Loading processor from %s ...", model_path)
    processor = RaonProcessor.from_pretrained(str(model_path))

    # Build data module
    from raon.utils.data import resolve_data_dir

    jsonl_paths = resolve_data_dir(args.data_dir)
    configs = [DatasetConfig(jsonl_path=p) for p in jsonl_paths]
    data_module = make_raon_data_module(
        processor=processor,
        dataset_configs=configs,
        max_audio_seq_length=args.max_audio_seq_length,
        use_packing=args.use_packing,
        max_packed_seq_length=args.max_packed_seq_length,
        log_first_n_batches=args.log_first_n_batches,
        use_speaker_embedding=args.use_speaker_embedding,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        # Critical: the model forward() receives custom TypedDict fields that
        # are not standard HF fields; disabling column removal preserves them.
        remove_unused_columns=False,
        bf16=(args.dtype == "bfloat16"),
        fp16=(args.dtype == "float16"),
        # Log every step so overfitting progress is visible.
        logging_strategy="steps",
        logging_steps=1,
        # Pin memory can cause issues with custom tensor types.
        dataloader_pin_memory=False,
        save_strategy="steps",
        save_steps=args.save_steps,
        report_to="tensorboard",
        ddp_find_unused_parameters=True if int(os.environ.get("WORLD_SIZE", 1)) > 1 else None,
    )

    trainer = RaonTrainer(
        model=model,
        args=training_args,
        train_dataset=data_module["train_dataset"],
        data_collator=data_module["data_collator"],
        use_speaker_embedding=args.use_speaker_embedding,
        callbacks=[StepLoggingCallback(), SaveTokenizerCallback(processor)],
    )

    logger.info("Starting training ...")
    trainer.train()

    logger.info("Saving model to %s ...", output_dir)
    trainer.save_model(str(output_dir))
    if trainer.is_world_process_zero():
        processor.tokenizer.save_pretrained(str(output_dir))

    logger.info("Done.")


if __name__ == "__main__":
    main()
