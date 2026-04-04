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

"""Inference script for RAON on JSONL conversation data.

Usage:
    python scripts/inference.py \
        --model_path /path/to/model \
        --data_dir /path/to/data \
        --output_dir /path/to/output \
        --device cuda \
        --dtype bfloat16
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import soundfile as sf
import torch
import yaml
from torch.nn.utils.rnn import pad_sequence

from raon.models.raon import RaonModel
from raon.utils.processor import (
    MultiModalMessage,
    RaonProcessor,
    augment_stt_messages,
    augment_tts_messages,
    convert_to_multimodal,
    resolve_audio_paths,
)
from raon.utils.processor import (
    detect_task_type as _detect_task_type,
)

logger = logging.getLogger(__name__)


_DEFAULT_INFERENCE_CONFIG = Path(__file__).parent.parent.parent / "config" / "infer.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAON inference on JSONL conversation data.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained model directory.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the JSONL data files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs.")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum number of new tokens to generate.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on (e.g. 'cuda', 'cpu').")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Torch dtype for model weights.",
    )
    parser.add_argument(
        "--speaker_audio",
        type=str,
        default=None,
        help="Path to speaker reference audio for TTS voice conditioning.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of samples to process in parallel per batch (default: 4).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(_DEFAULT_INFERENCE_CONFIG),
        help="Path to YAML file with task-specific inference parameters.",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        choices=["fa", "sdpa", "eager"],
        help="Attention implementation (default: sdpa). Use `fa` for FlashAttention.",
    )
    return parser.parse_args()


from raon.utils.misc import resolve_dtype


def load_task_params(config_path: Path) -> dict[str, dict]:
    """Load task-specific inference parameters from a YAML file."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


detect_task_type = _detect_task_type


def build_messages(sample: dict, task_type: str, use_speaker_embedding: bool = False) -> list[MultiModalMessage]:
    """Build inference messages from a JSONL record.

    Uses the top-level ``audios`` field for audio paths (matching duplex-model),
    not the per-turn ``source_path``. Each ``<audio>`` tag in conversation
    content consumes one path from the ``audios`` list in order.

    Args:
        sample: A single parsed JSONL record.
        task_type: Task identifier (e.g. 'tts', 'stt', 'speechqa').
        use_speaker_embedding: Whether to prepend speaker token for TTS.

    Returns:
        List of MultiModalMessage dicts with audio paths resolved.
    """
    audio_paths = resolve_audio_paths(sample.get("audios", []))
    audio_iter = iter(audio_paths)
    messages: list[MultiModalMessage] = []
    for turn in sample.get("conversations", []):
        role_raw: str = turn.get("from", "human")
        role = {"human": "user", "gpt": "assistant"}.get(role_raw, role_raw)
        value: str = turn.get("value", "")

        # Skip assistant turns — model will generate these
        if role == "assistant":
            continue

        msg: dict = {"role": role, "content": value}
        msg = convert_to_multimodal(msg, audio_iter)
        messages.append(msg)

    # STT: merge transcription instruction into the last user (audio) turn.
    if task_type == "stt" and messages:
        augment_stt_messages(messages)

    # TTS: prepend instruction (+ speaker token if enabled) to first user message.
    if task_type == "tts" and messages:
        augment_tts_messages(messages, use_speaker_embedding=use_speaker_embedding)

    return messages


from raon.utils.audio_io import save_audio


def load_speaker_audio(
    speaker_audio_path: str | None,
    sample: dict,
    sampling_rate: int,
    device: str,
) -> torch.Tensor | None:
    """Load speaker reference audio for TTS voice conditioning.

    Priority: CLI --speaker_audio > sample's speaker_ref_audios > sample's GT audio.
    """
    from raon.utils.audio_io import load_audio as _load_audio_shared

    path = speaker_audio_path
    if path is None:
        # Use speaker reference audio from sample, fallback to GT audio.
        speaker_ref_audios = sample.get("speaker_ref_audios", [])
        if speaker_ref_audios:
            path = speaker_ref_audios[0]
        else:
            audios = sample.get("audios", [])
            if audios:
                path = audios[0]

    if path is None:
        return None

    audio, _ = _load_audio_shared(path, sampling_rate, mono=True, device=device)
    return audio


def _save_batch_results(
    output: dict,
    batch_items: list[tuple[int, dict]],
    input_lengths: list[int],
    is_audio_output: bool,
    processor: RaonProcessor,
    output_dir: Path,
    stem: str,
) -> None:
    """Save results from a batched generate() call, one file per sample."""
    batch_size = len(batch_items)

    if is_audio_output:
        audio = output.get("audio")
        audio_lengths = output.get("audio_lengths")
        if audio is None or audio_lengths is None:
            logger.warning("    Batch produced no audio output, skipping.")
            return
        for i in range(batch_size):
            idx, _ = batch_items[i]
            valid_length = int(audio_lengths[i].item())
            output_path = output_dir / f"{stem}_{idx:05d}.wav"
            save_audio(audio[i], processor.sampling_rate, output_path, length=valid_length)
            logger.info(
                "    [%d] Saved audio: %d samples (%.2fs) -> %s",
                idx,
                valid_length,
                valid_length / processor.sampling_rate,
                output_path,
            )
    else:
        sequences = output.get("sequences")
        if sequences is None:
            logger.warning("    Batch produced no sequences, skipping.")
            return
        for i in range(batch_size):
            idx, _ = batch_items[i]
            generated_ids = sequences[i, input_lengths[i] :]
            text = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
            output_path = output_dir / f"{stem}_{idx:05d}.txt"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(text, encoding="utf-8")
            logger.info("    [%d] Generated: %s%s", idx, text[:100], "..." if len(text) > 100 else "")


def run_inference(
    jsonl_path: Path,
    model: RaonModel,
    processor: RaonProcessor,
    output_dir: Path,
    task_params: dict[str, dict],
    max_new_tokens: int,
    device: str,
    dtype: torch.dtype,
    speaker_audio_path: str | None = None,
    batch_size: int = 4,
) -> None:
    """Run batched inference on every sample in a single JSONL file.

    Samples are grouped by task type so each batch shares the same generation
    parameters (temperature, force_audio_output, etc.).
    """
    with open(jsonl_path, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    logger.info("Processing %s: %d samples (batch_size=%d)", jsonl_path.name, len(lines), batch_size)

    # Group samples by task type
    task_groups: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for idx, line in enumerate(lines):
        sample: dict = json.loads(line)
        channel: str = sample.get("channel", "")
        task_type = detect_task_type(channel)
        if task_type == "unknown":
            logger.info("  [%d/%d] Skipping unknown channel '%s'", idx + 1, len(lines), channel)
            continue
        task_groups[task_type].append((idx, sample))

    has_speaker_encoder = hasattr(model, "speaker_encoder") and model.speaker_encoder is not None
    stem = jsonl_path.stem

    for task_type, items in task_groups.items():
        params = task_params[task_type]
        is_audio_output = params["force_audio_output"]
        logger.info("  Task '%s': %d samples", task_type, len(items))

        # Process in batches
        for batch_start in range(0, len(items), batch_size):
            batch_items = items[batch_start : batch_start + batch_size]
            cur_batch_size = len(batch_items)
            batch_indices = [idx for idx, _ in batch_items]
            logger.info("    Batch [%s] (%d samples)", ",".join(str(i) for i in batch_indices), cur_batch_size)

            # Build messages for each sample in the batch
            batch_messages: list[list[MultiModalMessage]] = []
            for _, sample in batch_items:
                msgs = build_messages(sample, task_type, use_speaker_embedding=has_speaker_encoder)
                batch_messages.append(msgs)

            # Skip batch if any sample produced empty messages
            if any(not msgs for msgs in batch_messages):
                logger.warning("    Some samples have no messages, falling back to per-sample processing.")
                for (idx, _sample), msgs in zip(batch_items, batch_messages, strict=False):
                    if not msgs:
                        logger.warning("    [%d] No messages, skipping.", idx)
                continue

            # Process batched messages into model inputs
            inputs = processor(
                batch_messages,
                add_generation_prompt=True,
                force_audio_output=is_audio_output,
                device=device,
                dtype=dtype,
                max_audio_chunk_length=params.get("max_audio_chunk_length"),
            )

            # Track per-sample input lengths (for decoding output later).
            # After left-padding, each sample's real tokens start at a different offset.
            # input_ids shape: [batch, max_seq_len], attention_mask: [batch, max_seq_len]
            input_lengths = [int(inputs["attention_mask"][i].sum().item()) for i in range(cur_batch_size)]

            # Load speaker audio for TTS batch
            speaker_audio = None
            if is_audio_output:
                speaker_audios = [
                    load_speaker_audio(speaker_audio_path, sample, processor.sampling_rate, device)
                    for _, sample in batch_items
                ]
                valid_audios = [a for a in speaker_audios if a is not None]
                if valid_audios:
                    speaker_audio = pad_sequence(
                        [a.squeeze(0) for a in valid_audios],
                        batch_first=True,
                        padding_value=0,
                    ).to(device)

            # Run batched generation
            output = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                audio_input=inputs.get("audio_input"),
                audio_input_lengths=inputs.get("audio_input_lengths"),
                max_new_tokens=params["max_new_tokens"],
                temperature=params["temperature"],
                do_sample=True,
                force_audio_output=is_audio_output,
                force_text_output=params["force_text_output"],
                ras_enabled=params.get("ras_enabled", False),
                ras_window_size=params.get("ras_window_size", 50),
                ras_repetition_threshold=params.get("ras_repetition_threshold", 0.5),
                speaker_audio=speaker_audio,
                disable_tqdm=False,
            )

            _save_batch_results(
                output=output,
                batch_items=batch_items,
                input_lengths=input_lengths,
                is_audio_output=is_audio_output,
                processor=processor,
                output_dir=output_dir,
                stem=stem,
            )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    if args.attn_implementation == "fa":
        args.attn_implementation = "flash_attention_2"

    dtype = resolve_dtype(args.dtype)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    task_params = load_task_params(Path(args.config))
    logger.info("Loaded inference config from %s (%d task types)", args.config, len(task_params))

    logger.info("Loading model from %s ...", args.model_path)
    model: RaonModel = RaonModel.from_pretrained(args.model_path, torch_dtype=dtype).to(args.device).eval()
    model._set_attention_implementation(args.attn_implementation)
    logger.info("Attention implementation: %s", args.attn_implementation)
    logger.info("Model loaded.")

    logger.info("Loading processor from %s ...", args.model_path)
    processor = RaonProcessor.from_pretrained(args.model_path)
    logger.info("Processor loaded.")

    # Collect all .jsonl files in data_dir
    jsonl_files = sorted(data_dir.glob("*.jsonl"))
    if not jsonl_files:
        logger.info("No .jsonl files found in %s", data_dir)
        return

    logger.info("Found %d JSONL file(s) in %s", len(jsonl_files), data_dir)

    for jsonl_path in jsonl_files:
        run_inference(
            jsonl_path=jsonl_path,
            model=model,
            processor=processor,
            output_dir=output_dir,
            task_params=task_params,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
            dtype=dtype,
            speaker_audio_path=args.speaker_audio,
            batch_size=args.batch_size,
        )

    logger.info("Done. Output saved to %s", output_dir)


if __name__ == "__main__":
    main()
