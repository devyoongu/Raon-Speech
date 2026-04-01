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
#
# Uses the SpeechBrain ECAPA-TDNN speaker encoder (speechbrain/spkrec-ecapa-voxceleb).

"""Speaker encoder: PretrainedSpeakerEncoder (ECAPA-TDNN) with trainable projection."""

from __future__ import annotations

import os
from typing import Any

import torch
from torch import nn
from transformers import PretrainedConfig


class SpeakerEncoderConfig(PretrainedConfig):
    """Configuration for SpeakerEncoder: input/output sizes, attention heads, and frame window."""

    model_type = "speaker_encoder"

    def __init__(
        self,
        input_size: int = 512,
        output_size: int = 4096,
        num_heads: int = 8,
        min_seconds: float = 2.0,
        max_seconds: float = 10.0,
        frame_rate: float = 12.5,
        encoder_type: str = "from_scratch",
        pretrained_model_id: str | None = None,
        pretrained_dim: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.input_size = input_size
        self.output_size = output_size
        self.num_heads = num_heads
        self.min_seconds = min_seconds
        self.max_seconds = max_seconds
        self.frame_rate = frame_rate
        self.encoder_type = encoder_type
        self.pretrained_model_id = pretrained_model_id
        self.pretrained_dim = pretrained_dim


class PretrainedSpeakerEncoder(nn.Module):
    """Frozen pretrained speaker encoder with a trainable projection.

    This encoder consumes raw 24kHz audio, resamples internally to 16kHz, runs
    a frozen SpeechBrain ECAPA model, and projects the pretrained embedding to
    the raon hidden size.
    """

    def __init__(self, config: SpeakerEncoderConfig, dtype: torch.dtype | None = None) -> None:
        """Initialize ECAPA speaker encoder wrapper.

        Args:
            config: Speaker encoder configuration.
            dtype: Dtype for the trainable projection layer.
        """
        super().__init__()
        assert config.encoder_type == "ecapa_tdnn", f"Only ecapa_tdnn is supported, got: {config.encoder_type}"
        assert config.pretrained_dim is not None, (
            f"The `pretrained_dim` attribute must be set for encoder_type={config.encoder_type}."
        )

        self.encoder_type = config.encoder_type
        self.pretrained_model_id = config.pretrained_model_id
        self.pretrained_dim = config.pretrained_dim
        self.output_size = config.output_size
        self.source_sample_rate = 24000
        self.target_sample_rate = 16000
        self.min_seconds = config.min_seconds
        self.max_seconds = config.max_seconds

        self.projection = nn.Linear(config.pretrained_dim, config.output_size, bias=False, dtype=dtype)

        # Backend is loaded lazily on first forward to avoid redundant loads.
        self._backend: Any = None
        self._backend_device = torch.device("cpu")

    def _load_backend(self) -> None:
        """Load the frozen ECAPA backend for a single local process."""
        import torchaudio

        if not hasattr(torchaudio, "list_audio_backends"):
            torchaudio.list_audio_backends = lambda: []

        # Patch huggingface_hub for speechbrain compat: speechbrain 1.0.x
        # passes the removed `use_auth_token` kwarg to hf_hub_download.
        import huggingface_hub as _hfhub

        _orig_hf_download = _hfhub.hf_hub_download

        def _patched_hf_download(*args, **kwargs):
            kwargs.pop("use_auth_token", None)
            return _orig_hf_download(*args, **kwargs)

        _hfhub.hf_hub_download = _patched_hf_download

        from speechbrain.inference.speaker import EncoderClassifier  # type: ignore

        model_id = self.pretrained_model_id or "speechbrain/spkrec-ecapa-voxceleb"

        if os.path.isdir(model_id):
            # Local path: use directly without downloading.
            local_dir = model_id
        else:
            # HuggingFace repo ID: download to cache.
            cache_root = os.environ.get(
                "SPEECHBRAIN_ECAPA_SAVEDIR",
                os.path.expanduser("~/.cache/raon/speechbrain"),
            )
            os.makedirs(cache_root, exist_ok=True)
            local_dir = os.path.join(cache_root, model_id.replace("/", "_"))
            from huggingface_hub import snapshot_download  # type: ignore

            snapshot_download(model_id, local_dir=local_dir)

        backend = EncoderClassifier.from_hparams(
            source=local_dir,
            savedir=local_dir,
            run_opts={"device": "cpu"},
        )

        object.__setattr__(self, "_backend", backend)
        if hasattr(self._backend, "parameters"):
            for param in self._backend.parameters():
                param.requires_grad = False

        self._backend_device = torch.device("cpu")

    def _ensure_backend_device(self) -> None:
        """Move backend to the projection device if needed."""
        target_device = self.projection.weight.device
        if self._backend is None:
            self._load_backend()
        if self._backend_device == target_device:
            return

        # SpeechBrain Pretrained keeps modules in backend.mods.
        self._backend.device = str(target_device)
        for mod in self._backend.mods.values():
            mod.to(target_device)
        self._backend_device = target_device

    def _extract_embedding(self, audio_16k: torch.Tensor, lengths_16k: torch.Tensor) -> torch.Tensor:
        """Extract frozen ECAPA embeddings from 16kHz audio.

        Args:
            audio_16k: Resampled mono audio. Shape: [batch_size, num_samples_16k]. Dtype: float.
            lengths_16k: Valid sample lengths. Shape: [batch_size]. Dtype: long.

        Returns:
            Pretrained embedding. Shape: [batch_size, pretrained_dim]. Dtype: float.
        """
        min_samples_16k = 1600
        batch_size = audio_16k.shape[0]
        if audio_16k.shape[1] < min_samples_16k:
            return torch.zeros(
                batch_size,
                self.pretrained_dim,
                device=audio_16k.device,
                dtype=audio_16k.dtype,
            )

        lengths_16k = lengths_16k.clamp(min=min_samples_16k)
        reference_length = max(1, int(audio_16k.shape[1]))
        wav_lens = lengths_16k.float() / float(reference_length)
        wav_lens = wav_lens.clamp(max=1.0)
        autocast_dtype: torch.dtype | None = None
        if audio_16k.device.type == "cuda" and self.projection.weight.dtype in {torch.float16, torch.bfloat16}:
            autocast_dtype = self.projection.weight.dtype

        with torch.no_grad():
            with torch.autocast(
                device_type=audio_16k.device.type,
                dtype=autocast_dtype,
                enabled=autocast_dtype is not None,
            ):
                embeddings = self._backend.encode_batch(audio_16k, wav_lens)

        return embeddings.squeeze(1)

    def forward(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Compute speaker embedding from raw 24kHz audio.

        Args:
            audio: Raw mono waveform at 24kHz. Shape: [batch_size, num_samples]. Dtype: float.
            audio_lengths: Valid sample lengths. Shape: [batch_size]. Dtype: long.

        Returns:
            Speaker embedding. Shape: [batch_size, 1, output_size]. Dtype: float.
        """
        self._ensure_backend_device()
        import torchaudio.functional

        audio_input = audio
        if (
            audio_input.device.type == "cuda"
            and self.projection.weight.dtype in {torch.float16, torch.bfloat16}
            and audio_input.dtype != self.projection.weight.dtype
        ):
            audio_input = audio_input.to(dtype=self.projection.weight.dtype)
        elif audio_input.dtype not in {torch.float32, torch.float64, torch.float16, torch.bfloat16}:
            audio_input = audio_input.float()

        audio_16k = torchaudio.functional.resample(
            audio_input,
            orig_freq=self.source_sample_rate,
            new_freq=self.target_sample_rate,
        )
        lengths_16k = (audio_lengths.float() * self.target_sample_rate / self.source_sample_rate).long()

        # Random crop (training) or front-truncate (inference).
        min_samples = int(self.min_seconds * self.target_sample_rate)
        max_samples = int(self.max_seconds * self.target_sample_rate)
        min_valid = int(lengths_16k.min().item())

        if self.training and min_valid > min_samples:
            target_len = int(torch.randint(min_samples, min(max_samples, min_valid) + 1, (1,)).item())
            if min_valid > target_len:
                start = int(torch.randint(0, min_valid - target_len + 1, (1,)).item())
                audio_16k = audio_16k[:, start : start + target_len]
                lengths_16k = (lengths_16k - start).clamp(min=0, max=target_len)
        else:
            if audio_16k.shape[1] > max_samples:
                audio_16k = audio_16k[:, :max_samples]
                lengths_16k = lengths_16k.clamp(max=max_samples)

        raw_embedding = self._extract_embedding(audio_16k, lengths_16k)
        raw_embedding = raw_embedding.to(dtype=self.projection.weight.dtype)
        projected = self.projection(raw_embedding)
        return projected.unsqueeze(1)
