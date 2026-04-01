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

"""Data loading, multi-dataset sampling, and sequence packing for raon fine-tuning."""

from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import ConcatDataset, Dataset

from ..types import RaonInputs
from .processor import (
    RaonProcessor,
    augment_stt_messages,
    augment_tts_messages,
    convert_to_multimodal,
    detect_task_type,
    resolve_audio_paths,
)
from .special_tokens import AUDIO_INPUT_PLACEHOLDER, AUDIO_OUTPUT_PLACEHOLDER, SPEAKER_EMBEDDING_PLACEHOLDER

logger = logging.getLogger(__name__)


def resolve_data_dir(data_dir: str) -> list[str]:
    """Resolve ``--data_dir`` value into a list of JSONL file paths.

    Accepts:
      - Single file: ``data/train/tts.jsonl``
      - Multiple files: ``data/train/tts.jsonl,data/train/stt.jsonl``
      - Single directory: ``data/train`` → all ``*.jsonl`` inside.
      - Multiple directories: ``data/train,data/eval`` → all ``*.jsonl`` in each.

    Returns:
        List of resolved JSONL file paths.
    """
    entries = [e.strip() for e in data_dir.split(",")]
    jsonl_paths: list[str] = []
    for entry in entries:
        p = Path(entry)
        if p.is_file() and p.suffix == ".jsonl":
            jsonl_paths.append(str(p))
        elif p.is_dir():
            found = sorted(p.glob("*.jsonl"))
            if not found:
                raise FileNotFoundError(f"No .jsonl files found in {p}")
            jsonl_paths.extend(str(f) for f in found)
        else:
            raise FileNotFoundError(f"Path does not exist or is not a directory/jsonl file: {entry}")
    if not jsonl_paths:
        raise ValueError(f"No datasets resolved from --data_dir: {data_dir}")
    logger.info("Resolved %d dataset(s) from --data_dir:", len(jsonl_paths))
    for jp in jsonl_paths:
        logger.info("  %s", jp)
    return jsonl_paths


@dataclass
class DatasetConfig:
    """Configuration for a single training dataset.

    Attributes:
        jsonl_path: Path to the JSONL file containing conversation samples.
        sampling_rate: Fraction of samples to use (1.0 = full dataset, 0.5 = 50% subsample).
        name: Optional human-readable label for logging and identification.
    """

    jsonl_path: str
    sampling_rate: float = 1.0  # 1.0=all, 0.5=50% subsample
    name: str | None = None


class RaonLazyDataset(Dataset):
    """PyTorch Dataset that reads a JSONL file and lazily processes each sample
    into RaonInputs via the processor, with retry logic on errors."""

    def __init__(
        self,
        config: DatasetConfig,
        processor: RaonProcessor,
        max_audio_chunk_length: int | None = None,
        use_speaker_embedding: bool = False,
    ) -> None:
        self.processor = processor
        self.max_audio_chunk_length = max_audio_chunk_length
        self.use_speaker_embedding = use_speaker_embedding
        self.source = config.jsonl_path
        self.samples: list[dict[str, Any]] = []

        jsonl_path = Path(config.jsonl_path)
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

        if config.sampling_rate < 1.0:
            k = int(len(self.samples) * config.sampling_rate)
            self.samples = random.sample(self.samples, k)

        logger.info(f"Loaded {len(self.samples)} samples from {config.jsonl_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> RaonInputs:
        # Audio files can be transiently unavailable (NFS timeouts, corrupted files,
        # mid-write states). Rather than crashing a long training run, we retry the
        # same sample 3 times, then fall back to the next sample 3 times (giving a
        # different audio file a chance to succeed), then make one final attempt on
        # the original index before raising. The 1-second sleep between attempts
        # allows transient I/O errors to resolve.
        # Retry pattern: 3 attempts on current idx, 3 on next idx, 1 final on original
        for attempt in range(7):
            if attempt < 3:
                try_idx = idx
            elif attempt < 6:
                try_idx = (idx + 1) % len(self.samples)
            else:
                try_idx = idx
            try:
                return self._process_sample(try_idx)
            except Exception:
                logger.warning(
                    f"[RaonLazyDataset] Error processing sample {try_idx} (attempt {attempt + 1}/7)",
                    exc_info=True,
                )
                if attempt < 6:
                    time.sleep(1)
        # Should not reach here, but raise if all retries fail
        raise RuntimeError(f"Failed to process sample {idx} after 7 attempts")

    def _process_sample(self, idx: int) -> RaonInputs:
        sample = self.samples[idx]
        conversations = sample.get("conversations", [])

        audio_paths = resolve_audio_paths(sample.get("audios", []))
        audio_iter = iter(audio_paths)

        messages: list[dict[str, Any]] = []
        for turn in conversations:
            from_role = turn.get("from", "human")
            value = turn.get("value", "")
            role = "user" if from_role == "human" else "assistant"

            msg: dict[str, Any] = {"role": role, "content": value}
            msg = convert_to_multimodal(msg, audio_iter)
            messages.append(msg)

        # Task-specific prompt augmentation.
        channel = sample.get("channel", "")
        task_type = detect_task_type(channel)
        if task_type == "stt" and messages:
            augment_stt_messages(messages)
        elif task_type == "tts" and messages:
            augment_tts_messages(messages, use_speaker_embedding=self.use_speaker_embedding)

        system_prompt = sample.get("system", "")
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        inputs = self.processor.process_single(
            messages,
            add_generation_prompt=False,
            audio_preprocessor=None,
            max_audio_chunk_length=self.max_audio_chunk_length,
        )
        return inputs


class RaonMultiDataset:
    """Combines multiple RaonLazyDatasets via simple concatenation."""

    def __init__(self, datasets: list[RaonLazyDataset]):
        self.dataset = ConcatDataset(datasets)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> RaonInputs:
        return self.dataset[idx]


class RaonStandardCollator:
    """Wraps processor._collate() for non-packing mode."""

    def __init__(self, processor: RaonProcessor, log_first_n: int = 0) -> None:
        self.processor = processor
        self.log_first_n = log_first_n
        self._batch_count = 0

    def __call__(self, batch: list[RaonInputs]) -> RaonInputs:
        result = self.processor._collate(batch)
        if self._batch_count < self.log_first_n:
            self._log_batch(result, self._batch_count)
            self._batch_count += 1
        return result

    def _log_batch(self, batch: RaonInputs, batch_idx: int) -> None:
        logger.info(f"[Batch {batch_idx}] input_ids shape: {batch['input_ids'].shape}")
        logger.info(f"[Batch {batch_idx}] attention_mask shape: {batch['attention_mask'].shape}")
        if batch.get("audio_input") is not None:
            logger.info(f"[Batch {batch_idx}] audio_input shape: {batch['audio_input'].shape}")
        if batch.get("audio_output") is not None:
            logger.info(f"[Batch {batch_idx}] audio_output shape: {batch['audio_output'].shape}")


class RaonPackingCollator:
    """Sequence packing collator. Concatenates multiple samples into one sequence."""

    def __init__(
        self,
        processor: RaonProcessor,
        max_packed_seq_length: int = 8192,
        max_audio_seq_length: int = 192000,
        log_first_n: int = 0,
    ) -> None:
        self.processor = processor
        self.max_packed_seq_length = max_packed_seq_length
        self.max_audio_seq_length = max_audio_seq_length
        self.log_first_n = log_first_n
        self._batch_count = 0
        self._sampling_rate = int(processor.sampling_rate)
        self._frame_rate = float(processor.frame_rate)

    def _update_audio_lengths_single(
        self,
        input_ids: torch.Tensor,
        audio_lengths: torch.Tensor | None,
        audio_token_id: int,
    ) -> torch.Tensor | None:
        if audio_lengths is None:
            return None

        assert input_ids.shape[0] == 1, (
            f"RaonPackingCollator._update_audio_lengths_single: Expected batch size 1 "
            f"but got `{input_ids.shape[0]=}`."
        )

        num_audio_tokens_from_input_ids = (input_ids[:, : self.max_packed_seq_length] == audio_token_id).long().sum(dim=1)
        num_audio_tokens_from_audio_lengths = (audio_lengths * self._frame_rate / self._sampling_rate).ceil().long()
        assert num_audio_tokens_from_audio_lengths.sum() >= num_audio_tokens_from_input_ids.sum(), (
            "RaonPackingCollator._update_audio_lengths_single: Not enough audio for audio tokens in input_ids. "
            f"Available: `{num_audio_tokens_from_audio_lengths.sum().item()}`, "
            f"required: `{num_audio_tokens_from_input_ids.sum().item()}`."
        )
        total_num_audio_tokens = num_audio_tokens_from_audio_lengths.cumsum(dim=0).minimum(num_audio_tokens_from_input_ids)
        num_audio_tokens_from_audio_lengths = torch.cat(
            (total_num_audio_tokens[:1], total_num_audio_tokens[1:] - total_num_audio_tokens[:-1])
        )
        max_audio_length = (num_audio_tokens_from_audio_lengths.double() * self._sampling_rate / self._frame_rate).long()
        return audio_lengths.minimum(max_audio_length)

    def _collect_audio_rows_and_lengths(
        self,
        audios: list[torch.Tensor | None],
        lengths: list[torch.Tensor | None],
    ) -> tuple[list[torch.Tensor], torch.Tensor] | tuple[None, None]:
        rows: list[torch.Tensor] = []
        valid_lengths: list[int] = []
        for audio, length in zip(audios, lengths, strict=True):
            if audio is None or length is None:
                continue
            lens = length.to(dtype=torch.long).view(-1)
            if audio.shape[0] != lens.shape[0]:
                raise ValueError(
                    "RaonPackingCollator: audio rows and lengths mismatch "
                    f"({audio.shape[0]} vs {lens.shape[0]})."
                )
            for row, row_len in zip(audio, lens, strict=True):
                row_len_int = int(row_len.item())
                if row_len_int <= 0:
                    continue
                rows.append(row[:row_len_int])
                valid_lengths.append(row_len_int)
        if len(rows) == 0:
            return None, None
        lengths_tensor = torch.tensor(valid_lengths, dtype=torch.long, device=rows[0].device)
        return rows, lengths_tensor

    def _batch_unchunked_audio(
        self,
        audios: list[torch.Tensor | None],
        lengths: list[torch.Tensor | None],
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        rows, lengths_tensor = self._collect_audio_rows_and_lengths(audios, lengths)
        if rows is None or lengths_tensor is None:
            return None, None
        padded = pad_sequence(rows, batch_first=True, padding_value=0.0)
        return padded, lengths_tensor

    def _sync_speaker_audio(
        self,
        input_ids: torch.Tensor,
        speaker_encoder_audio: torch.Tensor | None,
        speaker_encoder_audio_lengths: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        num_speaker_tokens = int((input_ids[:, : self.max_packed_seq_length] == SPEAKER_EMBEDDING_PLACEHOLDER.id).sum().item())
        has_speaker_audio = speaker_encoder_audio is not None and speaker_encoder_audio_lengths is not None
        if not has_speaker_audio:
            return speaker_encoder_audio, speaker_encoder_audio_lengths

        assert speaker_encoder_audio is not None and speaker_encoder_audio_lengths is not None
        assert speaker_encoder_audio.shape[0] == speaker_encoder_audio_lengths.shape[0], (
            "RaonPackingCollator._sync_speaker_audio: speaker audio and lengths must match batch size. "
            f"Got `{speaker_encoder_audio.shape[0]=}` and `{speaker_encoder_audio_lengths.shape[0]=}`."
        )

        speaker_batch_size = int(speaker_encoder_audio.shape[0])
        if speaker_batch_size < num_speaker_tokens:
            pad_count = num_speaker_tokens - speaker_batch_size
            pad_audio = torch.zeros(
                (pad_count, *speaker_encoder_audio.shape[1:]),
                dtype=speaker_encoder_audio.dtype,
                device=speaker_encoder_audio.device,
            )
            pad_lengths = torch.zeros(
                pad_count,
                dtype=speaker_encoder_audio_lengths.dtype,
                device=speaker_encoder_audio_lengths.device,
            )
            speaker_encoder_audio = torch.cat([speaker_encoder_audio, pad_audio], dim=0)
            speaker_encoder_audio_lengths = torch.cat([speaker_encoder_audio_lengths, pad_lengths], dim=0)
        elif speaker_batch_size > num_speaker_tokens:
            speaker_encoder_audio = speaker_encoder_audio[:num_speaker_tokens]
            speaker_encoder_audio_lengths = speaker_encoder_audio_lengths[:num_speaker_tokens]

        return speaker_encoder_audio, speaker_encoder_audio_lengths

    def __call__(self, batch: list[RaonInputs]) -> dict[str, Any]:
        # 1. Extract valid tokens (where attention_mask == 1) from each sample
        all_ids: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        all_pos: list[torch.Tensor] = []
        audio_inputs: list[torch.Tensor | None] = []
        audio_input_lengths: list[torch.Tensor | None] = []
        audio_outputs: list[torch.Tensor | None] = []
        audio_output_lengths: list[torch.Tensor | None] = []
        speaker_audios: list[torch.Tensor | None] = []
        speaker_audio_lengths: list[torch.Tensor | None] = []

        for sample in batch:
            ids = sample["input_ids"]  # [1, seq_len]
            mask = sample["attention_mask"]  # [1, seq_len]
            labs = sample["labels"]  # [1, seq_len]

            # Filter to valid positions
            valid = mask[0] == 1
            valid_ids = ids[0][valid]
            valid_labels = labs[0][valid]
            # Position IDs: per-sample cumsum reset (0, 1, 2, ...)
            valid_pos = torch.arange(valid.sum(), dtype=torch.long)

            all_ids.append(valid_ids)
            all_labels.append(valid_labels)
            all_pos.append(valid_pos)

            # Collect audio as per-sample entries
            audio_inputs.append(sample.get("audio_input"))
            audio_input_lengths.append(sample.get("audio_input_lengths"))
            audio_outputs.append(sample.get("audio_output"))
            audio_output_lengths.append(sample.get("audio_output_lengths"))
            speaker_audios.append(sample.get("speaker_encoder_audio"))
            speaker_audio_lengths.append(sample.get("speaker_encoder_audio_lengths"))

        # 2. Concatenate and truncate to max_packed_seq_length
        packed_ids = torch.cat(all_ids)[: self.max_packed_seq_length]
        packed_labels = torch.cat(all_labels)[: self.max_packed_seq_length]
        packed_pos = torch.cat(all_pos)[: self.max_packed_seq_length]

        input_rows, input_lengths = self._collect_audio_rows_and_lengths(audio_inputs, audio_input_lengths)
        output_rows, output_lengths = self._collect_audio_rows_and_lengths(audio_outputs, audio_output_lengths)

        # 3. Chunk audio inputs
        chunked_audio_input, chunked_audio_input_lengths = self._chunk_rows(input_rows, input_lengths)
        chunked_audio_output, chunked_audio_output_lengths = self._chunk_rows(output_rows, output_lengths)
        packed_ids_batch = packed_ids.unsqueeze(0)
        chunked_audio_input_lengths = self._update_audio_lengths_single(
            input_ids=packed_ids_batch,
            audio_lengths=chunked_audio_input_lengths,
            audio_token_id=AUDIO_INPUT_PLACEHOLDER.id,
        )
        chunked_audio_output_lengths = self._update_audio_lengths_single(
            input_ids=packed_ids_batch,
            audio_lengths=chunked_audio_output_lengths,
            audio_token_id=AUDIO_OUTPUT_PLACEHOLDER.id,
        )

        # Speaker audio: batch unchunked waveforms with padding.
        speaker_audio_cat, speaker_audio_len_cat = self._batch_unchunked_audio(speaker_audios, speaker_audio_lengths)
        speaker_audio_cat, speaker_audio_len_cat = self._sync_speaker_audio(
            input_ids=packed_ids_batch,
            speaker_encoder_audio=speaker_audio_cat,
            speaker_encoder_audio_lengths=speaker_audio_len_cat,
        )

        result: dict[str, Any] = {
            "input_ids": packed_ids_batch,  # [1, total_len]
            "attention_mask": None,
            "position_ids": packed_pos.unsqueeze(0),  # [1, total_len]
            "labels": packed_labels.unsqueeze(0),  # [1, total_len]
            "audio_input": chunked_audio_input,
            "audio_input_lengths": chunked_audio_input_lengths,
            "audio_output": chunked_audio_output,
            "audio_output_lengths": chunked_audio_output_lengths,
            "speaker_encoder_audio": speaker_audio_cat,
            "speaker_encoder_audio_lengths": speaker_audio_len_cat,
        }

        if self._batch_count < self.log_first_n:
            self._log_batch(result, self._batch_count)
            self._batch_count += 1

        return result

    def _chunk_rows(
        self,
        rows: list[torch.Tensor] | None,
        lengths: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if rows is None or lengths is None:
            return None, None
        padded = pad_sequence(rows, batch_first=True, padding_value=0.0)
        return RaonProcessor._chunk_audio(padded, lengths, self.max_audio_seq_length)

    def _log_batch(self, batch: dict[str, Any], batch_idx: int) -> None:
        logger.info(
            f"[PackedBatch {batch_idx}] input_ids: {batch['input_ids'].shape}, "
            f"position_ids: {batch.get('position_ids', 'N/A')}"
        )
        if batch.get("audio_input") is not None:
            logger.info(f"[PackedBatch {batch_idx}] audio_input: {batch['audio_input'].shape}")


def make_raon_data_module(
    processor: RaonProcessor,
    dataset_configs: list[DatasetConfig],
    max_audio_seq_length: int = 192000,
    use_packing: bool = False,
    max_packed_seq_length: int = 8192,
    log_first_n_batches: int = 0,
    use_speaker_embedding: bool = False,
) -> dict[str, Any]:
    """Create dataset + collator for HF Trainer.

    Returns dict with 'train_dataset' and 'data_collator'.
    """
    max_chunk = max_audio_seq_length if max_audio_seq_length > 0 else None

    datasets = [RaonLazyDataset(cfg, processor, max_chunk, use_speaker_embedding) for cfg in dataset_configs]
    if len(datasets) == 1:
        train_dataset: Dataset = datasets[0]
    else:
        multi = RaonMultiDataset(datasets)
        train_dataset = multi.dataset

    if use_packing:
        collator: RaonStandardCollator | RaonPackingCollator = RaonPackingCollator(
            processor=processor,
            max_packed_seq_length=max_packed_seq_length,
            max_audio_seq_length=max_audio_seq_length,
            log_first_n=log_first_n_batches,
        )
    else:
        collator = RaonStandardCollator(processor=processor, log_first_n=log_first_n_batches)

    return {
        "train_dataset": train_dataset,
        "data_collator": collator,
    }
