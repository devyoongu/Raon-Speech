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

"""SGLang-compatible inference backend with custom KV-cache management."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F
from safetensors.torch import load_file


def _ensure_runtime_env() -> None:
    """Set up PATH and CUDA_HOME if not already configured.

    This function is NOT called at import time to avoid side effects.
    Call it explicitly before initializing SGLangRaonModel.
    """
    repo_root = Path(__file__).resolve().parents[3]
    venv_bin = repo_root / ".venv" / "bin"
    if venv_bin.is_dir():
        path_entries = os.environ.get("PATH", "").split(":")
        if str(venv_bin) not in path_entries:
            os.environ["PATH"] = f"{venv_bin}:{os.environ.get('PATH', '')}"

    if os.environ.get("CUDA_HOME"):
        return

    # Auto-detect CUDA toolkit by searching common install paths.
    import glob

    cuda_candidates = sorted(glob.glob("/usr/local/cuda*"), reverse=True)
    for toolkit_root_str in cuda_candidates:
        toolkit_root = Path(toolkit_root_str)
        if (toolkit_root / "bin" / "nvcc").is_file():
            os.environ["CUDA_HOME"] = str(toolkit_root)
            return

    # Fallback: check pip-installed nvidia cuda_runtime package.
    candidate = repo_root / ".venv" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}"
    runtime_root = candidate / "site-packages" / "nvidia" / "cuda_runtime"
    if (runtime_root / "include" / "cuda.h").is_file():
        os.environ["CUDA_HOME"] = str(runtime_root)

from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod  # type: ignore
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead  # type: ignore
from sglang.srt.managers.schedule_batch import ModelWorkerBatch  # type: ignore
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch  # type: ignore
from sglang.srt.model_executor.model_runner import (  # type: ignore
    ForwardMode,
    LogitsProcessorOutput,
    ModelConfig,
    ModelRunner,
    ModelRunnerOutput,
    ReqToTokenPool,
    ServerArgs,
)
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo  # type: ignore
from torch import nn
from torch._logging._internal import set_logs
from transformers import DynamicCache, PreTrainedModel

from raon.models.raon import TEXT_MODELS, RaonConfig, RaonModel
from raon.models.wrapper import RaonInferenceModel
from raon.modules import (
    EmbeddingAdaptor,
    PretrainedSpeakerEncoder,
    RaonCodePredictorModel,
    ThinkerToTalkerProjection,
    VoxtralWrapper,
)
from raon.modules.audio_tokenizer import StreamingMimiModel
from raon.types import AudioDecoderOutput, AudioEncoderOutput, AudioTokenizerOutput
from raon.utils.special_tokens import (
    AUDIO_INPUT_PLACEHOLDER,
    AUDIO_OUTPUT_BC,
    AUDIO_OUTPUT_END_PAD,
    AUDIO_OUTPUT_PAD,
    AUDIO_OUTPUT_PLACEHOLDER,
    AUDIO_START,
    DUPLEX_SIL,
    IM_START,
    SPEAKER_EMBEDDING_PLACEHOLDER,
)


def load_raon_model_without_text_model(path: str | Path, device: str, dtype: torch.dtype) -> RaonModel:
    """Load runtime checkpoint shards while intentionally skipping thinker text model weights."""
    checkpoint_path = Path(path)
    config_path = checkpoint_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing runtime config: {config_path}")
    config = RaonConfig(**json.loads(config_path.read_text(encoding="utf-8")))

    state_dict: dict[str, torch.Tensor] = {}
    for safetensor_file in checkpoint_path.glob("*.safetensors"):
        state_dict.update(load_file(safetensor_file))

    new_model = object.__new__(RaonModel)
    PreTrainedModel.__init__(new_model, config)
    new_model.config = config
    new_model.hidden_size = int(config.text_model_config.hidden_size)
    new_model.vocab_size = int(config.text_model_config.vocab_size)
    new_model.codebook_size = int(config.audio_tokenizer_config.codebook_size)
    new_model.audio_lm_head_vocab_size = new_model.codebook_size + 1
    new_model.supports_audio_input = bool(getattr(config, "supports_audio_input", True))
    new_model.supports_audio_output = bool(getattr(config, "supports_audio_output", True))
    new_model.use_duplex_end_pad = bool(getattr(config, "use_duplex_end_pad", False))
    new_model.use_sil_token = bool(getattr(config, "use_sil_token", False))
    new_model.use_inline_text_prediction = bool(getattr(config, "use_inline_text_prediction", False))
    new_model.no_audio_in_sil = bool(getattr(config, "no_audio_in_sil", False))
    new_model.sequence_mode = getattr(config, "sequence_mode", None)
    new_model.duplex_sil_token_id = int(getattr(config, "duplex_sil_token_id", DUPLEX_SIL.id))
    new_model.num_code_groups = int(config.code_predictor_config.num_code_groups)
    new_model.input_num_code_groups = getattr(config, "input_num_code_groups", None) or new_model.num_code_groups
    new_model.sampling_rate = int(config.audio_tokenizer_config.sampling_rate)
    frame_rate = getattr(config.audio_tokenizer_config, "_frame_rate", None)
    if frame_rate is None:
        raise ValueError("audio_tokenizer_config._frame_rate is required for realtime decoding.")
    new_model.frame_rate = float(frame_rate)
    acoustic_delay = getattr(config, "acoustic_delay", None)
    delays = getattr(config, "delays", None)
    if isinstance(delays, list):
        new_model.delays = list(delays)
    elif isinstance(acoustic_delay, list):
        new_model.delays = list(acoustic_delay)
    elif isinstance(acoustic_delay, int):
        new_model.delays = [0] + [acoustic_delay] * max(0, new_model.num_code_groups - 1)
    else:
        new_model.delays = [0] * new_model.num_code_groups
    if len(new_model.delays) != new_model.num_code_groups:
        new_model.delays = [0] * new_model.num_code_groups
    new_model.max_delay = max(new_model.delays) if new_model.delays else 0

    if new_model.supports_audio_output and config.speaker_encoder_config is not None:
        if getattr(config.speaker_encoder_config, "encoder_type", "") == "ecapa_tdnn":
            new_model.speaker_encoder = PretrainedSpeakerEncoder(config.speaker_encoder_config, dtype=dtype)
            new_model.is_pretrained_speaker_encoder = True
            new_model.speaker_token_id = SPEAKER_EMBEDDING_PLACEHOLDER.id
        else:
            new_model.speaker_encoder = None
            new_model.is_pretrained_speaker_encoder = False
            new_model.speaker_token_id = None
    else:
        new_model.speaker_encoder = None
        new_model.is_pretrained_speaker_encoder = False
        new_model.speaker_token_id = None

    new_model.text_model = None  # type: ignore[assignment]
    new_model.audio_encoder = (
        VoxtralWrapper.from_config(config=config.audio_encoder_config, dtype=dtype)
        if new_model.supports_audio_input
        else None
    )
    new_model.aut_is_causal = bool(getattr(config, "aut_is_causal", False))
    new_model.audio_tokenizer = (
        StreamingMimiModel._from_config(config.audio_tokenizer_config, dtype=dtype)
        if new_model.supports_audio_output
        else None
    )
    new_model.input_adaptor = (
        EmbeddingAdaptor.from_config(config.input_adaptor_config, dtype=dtype)
        if new_model.supports_audio_input
        else None
    )
    new_model.output_adaptor = (
        EmbeddingAdaptor.from_config(config.output_adaptor_config, dtype=dtype)
        if new_model.supports_audio_output
        else None
    )
    new_model.lm_head = None  # type: ignore[assignment]

    num_talker_layers = int(config.num_talker_layers)
    num_thinker_layers = int(config.text_model_config.num_hidden_layers)
    new_model.num_talker_layers = num_talker_layers
    accept_hidden_layer = int(getattr(config, "accept_hidden_layer", -1))
    if accept_hidden_layer < 0:
        accept_hidden_layer = num_thinker_layers + accept_hidden_layer
    new_model.accept_hidden_layer = accept_hidden_layer
    new_model.thinker_capture_layer_index = accept_hidden_layer

    talker_hidden_size = (
        int(config.talker_config.hidden_size)
        if config.talker_config is not None
        else int(config.text_model_config.hidden_size)
    )
    if new_model.supports_audio_output:
        new_model.proj_code = torch.nn.Linear(
            talker_hidden_size,
            config.code_predictor_config.hidden_size,
            bias=bool(getattr(config, "proj_code_bias", False)),
            dtype=dtype,
        )
        new_model.code_predictor = RaonCodePredictorModel._from_config(config.code_predictor_config, dtype=dtype)  # type: ignore[assignment]
        new_model.audio_lm_head = torch.nn.Linear(
            talker_hidden_size,
            new_model.audio_lm_head_vocab_size,
            bias=False,
            dtype=dtype,
        )
    else:
        new_model.proj_code = None
        new_model.code_predictor = None
        new_model.audio_lm_head = None

    if new_model.supports_audio_output and config.talker_config is not None:
        projection_mode = getattr(config, "thinker_to_talker_projection_mode", "linear")
        projection_intermediate_size = getattr(config, "thinker_to_talker_intermediate_size", None)
        if projection_mode == "mlp" and projection_intermediate_size is None:
            projection_intermediate_size = int(config.talker_config.intermediate_size)
        new_model.thinker_to_talker_proj = ThinkerToTalkerProjection(
            thinker_hidden_size=int(config.text_model_config.hidden_size),
            talker_hidden_size=talker_hidden_size,
            intermediate_size=projection_intermediate_size,
            mode=projection_mode,
            use_norm=bool(getattr(config, "thinker_to_talker_pre_norm", False)),
            rms_norm_eps=float(getattr(config.text_model_config, "rms_norm_eps", 1e-6)),
        ).to(dtype=dtype)
        talker = TEXT_MODELS[config.talker_config.model_type]._from_config(config.talker_config, dtype=dtype)
        talker.embed_tokens = None
        new_model.talker = talker
    else:
        new_model.thinker_to_talker_proj = None
        new_model.talker = None

    prefix_maps = {
        "audio_encoder.": new_model.audio_encoder,
        "audio_tokenizer.": new_model.audio_tokenizer,
        "input_adaptor.": new_model.input_adaptor,
        "output_adaptor.": new_model.output_adaptor,
        "proj_code.": new_model.proj_code,
        "code_predictor.": new_model.code_predictor,
        "audio_lm_head.": new_model.audio_lm_head,
        "speaker_encoder.": new_model.speaker_encoder,
        "thinker_to_talker_proj.": new_model.thinker_to_talker_proj,
        "talker.": new_model.talker,
    }
    for prefix, module in prefix_maps.items():
        if module is None:
            continue
        module_state = {k.replace(prefix, ""): v for k, v in state_dict.items() if k.startswith(prefix)}
        if module_state:
            strict = False if prefix == "talker." else True
            module.load_state_dict(module_state, strict=strict)

    adaptor_state = {k.replace("adaptor.", ""): v for k, v in state_dict.items() if k.startswith("adaptor.")}
    if adaptor_state:
        if new_model.input_adaptor is not None:
            new_model.input_adaptor.load_state_dict(adaptor_state)
        if new_model.output_adaptor is not None:
            new_model.output_adaptor.load_state_dict(adaptor_state)

    return new_model.to(device)  # type: ignore[return-value]


@dataclass
class SGLangDecodingMetadata:
    """Per-request decoding state for SGLang KV-cache: pool indices, sequence lengths, and cache offsets."""

    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool_allocator: Any
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    max_sequence_length: int
    batch_size: int
    sampling_info: SamplingBatchInfo
    device: str
    talker_past_key_values: DynamicCache | None = field(default=None)
    talker_attention_mask: torch.Tensor | None = field(default=None)

    def _get_last_loc(self, prefix_lens: torch.Tensor) -> torch.Tensor:
        """Return the last allocated physical KV slot per request, or ``-1`` if empty."""
        last_loc = torch.full((self.batch_size,), -1, dtype=torch.int64, device=self.device)
        for row_index in range(self.batch_size):
            prefix_len = int(prefix_lens[row_index].item())
            if prefix_len == 0:
                continue

            req_pool_idx = int(self.req_pool_indices[row_index].item())
            last_loc[row_index] = self.req_to_token_pool.req_to_token[req_pool_idx, prefix_len - 1].to(torch.int64)
        return last_loc

    def free_allocated_kv_slots(self) -> None:
        """Free any thinker KV slots currently owned by this metadata object."""
        active_indices: list[torch.Tensor] = []
        for row_index in range(self.batch_size):
            seq_len = int(self.seq_lens[row_index].item())
            if seq_len == 0:
                continue

            req_pool_idx = int(self.req_pool_indices[row_index].item())
            active_indices.append(self.req_to_token_pool.req_to_token[req_pool_idx, :seq_len].to(torch.int64))
            self.req_to_token_pool.req_to_token[req_pool_idx, :seq_len] = 0

        if active_indices:
            self.token_to_kv_pool_allocator.free(torch.cat(active_indices))

    def update(
        self,
        attention_mask: torch.Tensor,
        is_prefill: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Update sequence lengths and cache locations after a forward pass.

        Args:
            attention_mask: Mask indicating valid tokens.
                Shape: [batch_size, seq_length]. Dtype: long or bool.
            is_prefill: Whether this forward is an extend/prefill step rather than decode.

        Returns:
            Tuple of (old_seq_lengths, new_seq_lengths, out_cache_loc).
            old_seq_lengths: Previous sequence lengths. Shape: [batch_size]. Dtype: long.
            new_seq_lengths: Updated sequence lengths. Shape: [batch_size]. Dtype: long.
            out_cache_loc: Flat cache indices for written tokens.
                Shape: [num_valid_tokens]. Dtype: long.
        """
        old_seq_lengths = self.seq_lens.clone()
        extend_lens = attention_mask.to(torch.long).sum(dim=1)
        new_seq_lengths = old_seq_lengths + extend_lens

        total_new_tokens = int(extend_lens.sum().item())
        if total_new_tokens == 0:
            self.seq_lens = new_seq_lengths
            return old_seq_lengths, new_seq_lengths, torch.empty(0, dtype=torch.int64, device=self.device)

        allocator = self.token_to_kv_pool_allocator
        page_size = int(getattr(allocator, "page_size", 1))
        out_cache_loc: torch.Tensor | None
        if page_size == 1:
            out_cache_loc = allocator.alloc(total_new_tokens)
        else:
            last_loc = self._get_last_loc(old_seq_lengths)
            if is_prefill:
                out_cache_loc = allocator.alloc_extend(
                    prefix_lens=old_seq_lengths,
                    prefix_lens_cpu=old_seq_lengths.cpu(),
                    seq_lens=new_seq_lengths,
                    seq_lens_cpu=new_seq_lengths.cpu(),
                    last_loc=last_loc,
                    extend_num_tokens=total_new_tokens,
                )
            else:
                out_cache_loc = allocator.alloc_decode(
                    seq_lens=new_seq_lengths,
                    seq_lens_cpu=new_seq_lengths.cpu(),
                    last_loc=last_loc,
                )

        if out_cache_loc is None:
            available_size = allocator.available_size() if hasattr(allocator, "available_size") else "unknown"
            mode = "prefill" if is_prefill else "decode"
            raise RuntimeError(
                "Failed to allocate thinker KV slots for "
                f"{mode}: requested={total_new_tokens}, page_size={page_size}, available_size={available_size}."
            )
        out_cache_loc = out_cache_loc.to(torch.int64)

        start_idx = 0
        req_to_token_dtype = self.req_to_token_pool.req_to_token.dtype
        for req_pool_idx, old_seq_length, extend_len in zip(
            self.req_pool_indices,
            old_seq_lengths,
            extend_lens,
            strict=True,
        ):
            extend_len_int = int(extend_len.item())
            if extend_len_int == 0:
                continue

            end_idx = start_idx + extend_len_int
            req_pool_idx_int = int(req_pool_idx.item())
            old_seq_length_int = int(old_seq_length.item())
            new_seq_length_int = old_seq_length_int + extend_len_int
            self.req_to_token_pool.req_to_token[req_pool_idx_int, old_seq_length_int:new_seq_length_int] = out_cache_loc[
                start_idx:end_idx
            ].to(req_to_token_dtype)
            start_idx = end_idx

        assert start_idx == total_new_tokens, "Cache slot writeback consumed an unexpected number of tokens."
        self.seq_lens = new_seq_lengths
        return old_seq_lengths, new_seq_lengths, out_cache_loc

    def reset(self) -> None:
        """Reset sequence lengths to zero for reuse of the metadata object."""
        self.free_allocated_kv_slots()
        self.seq_lens.zero_()


class SGLangRaonModel(RaonInferenceModel):
    """SGLang-compatible inference backend for duplex TTS and full-duplex streaming with custom KV-cache management."""

    vocab_size: int
    codebook_size: int
    num_code_groups: int
    sampling_rate: int
    frame_rate: float

    def __getattr__(self, name: str) -> Any:
        raon_duplex_model = self.__dict__.get("raon_duplex_model")
        if raon_duplex_model is not None and hasattr(raon_duplex_model, name):
            return getattr(raon_duplex_model, name)
        raise AttributeError(f"{type(self).__name__!s} object has no attribute {name!r}")

    def __init__(
        self,
        path: str,
        dtype: str = "auto",
        mem_fraction_static: float = 0.75,
        disable_cuda_graph: bool | None = None,
        cuda_graph_max_bs: int | None = None,
        # Reduce SGLang continuous batching / pool sizing (optional).
        # These correspond to common SGLang server args; we set them on ServerArgs if present.
        max_running_requests: int | None = None,
        prefill_max_requests: int | None = None,
        max_total_tokens: int | None = None,
        max_prefill_tokens: int | None = None,
        chunked_prefill_size: int | None = None,
        gpu_id: int = 0,
        tp_rank: int = 0,
        tp_size: int = 1,
        moe_ep_rank: int = 0,
        moe_ep_size: int = 1,
        pp_rank: int = 0,
        pp_size: int = 1,
        nccl_port: int = 0,
        max_allocated_req_pool_indices: int = 1024,
    ) -> None:
        """Initialize the SGLang duplex model by loading the duplex submodel and text model runner from a bundle path."""
        _ensure_runtime_env()
        from pathlib import Path

        bundle_path = Path(path)
        runtime_model_path = bundle_path / "raon_runtime"
        if not runtime_model_path.is_dir():
            runtime_model_path = bundle_path / "duplex_model"
        if not runtime_model_path.is_dir():
            raise FileNotFoundError(
                f"Expected a runtime model directory at {bundle_path / 'raon_runtime'} or {bundle_path / 'duplex_model'}."
            )
        self.device = f"cuda:{gpu_id}"
        self.dtype = getattr(torch, dtype, torch.bfloat16)

        self.raon_duplex_model = load_raon_model_without_text_model(
            path=str(runtime_model_path),
            device=self.device,
            dtype=self.dtype,
        )

        self.text_model_runner = self._load_text_model_runner(
            path=str(bundle_path / "text_model"),
            dtype=dtype,
            mem_fraction_static=mem_fraction_static,
            disable_cuda_graph=disable_cuda_graph,
            cuda_graph_max_bs=cuda_graph_max_bs,
            max_running_requests=max_running_requests,
            prefill_max_requests=prefill_max_requests,
            max_total_tokens=max_total_tokens,
            max_prefill_tokens=max_prefill_tokens,
            chunked_prefill_size=chunked_prefill_size,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            tp_size=tp_size,
            moe_ep_rank=moe_ep_rank,
            moe_ep_size=moe_ep_size,
            pp_rank=pp_rank,
            pp_size=pp_size,
            nccl_port=nccl_port,
        )

        assert isinstance(lm_head := self.text_model_runner.model.lm_head, ParallelLMHead), (
            "LM head must be a ParallelLMHead."
        )
        assert isinstance(lm_head.quant_method, UnquantizedEmbeddingMethod), (
            "LM head quant_method must be UnquantizedEmbeddingMethod."
        )
        assert isinstance(lm_head_weight := lm_head.weight, torch.Tensor), "LM head weight must be a tensor."
        self.lm_head_weight = lm_head_weight

        self.thinker_capture_layer_index = self.raon_duplex_model.thinker_capture_layer_index
        self.accepted_thinker_hidden_states: torch.Tensor | None = None
        self.register_thinker_capture_hook()

        self.supports_audio_output = bool(getattr(self.raon_duplex_model, "supports_audio_output", True))
        if self.supports_audio_output:
            assert self.raon_duplex_model.talker is not None, "Separated talker is required."
            assert self.raon_duplex_model.thinker_to_talker_proj is not None, "thinker_to_talker_proj is required."
        # Talker KV cache is stored per-session in SGLangDecodingMetadata,
        # not on the model instance, to allow concurrent sessions.

        self.vocab_size = self.raon_duplex_model.vocab_size
        self.codebook_size = self.raon_duplex_model.codebook_size
        self.use_duplex_end_pad = self.raon_duplex_model.use_duplex_end_pad
        self.num_code_groups = self.raon_duplex_model.num_code_groups
        self.sampling_rate = self.raon_duplex_model.sampling_rate
        self.frame_rate = self.raon_duplex_model.frame_rate
        self.delays = self.raon_duplex_model.delays
        self.max_delay = self.raon_duplex_model.max_delay
        self.sequence_mode = self.raon_duplex_model.sequence_mode
        self.use_sil_token = self.raon_duplex_model.use_sil_token
        self.use_inline_text_prediction = bool(
            getattr(self.raon_duplex_model, "use_inline_text_prediction", False)
        )
        self.no_audio_in_sil = self.raon_duplex_model.no_audio_in_sil
        self.use_backchannel_token = getattr(self.raon_duplex_model, "use_backchannel_token", False)
        self.duplex_bc_token_id = getattr(self.raon_duplex_model, "duplex_bc_token_id", AUDIO_OUTPUT_BC.id)
        self.duplex_sil_token_id = self.raon_duplex_model.duplex_sil_token_id
        self.duplex_pad_token_id = getattr(self.raon_duplex_model, "duplex_pad_token_id", AUDIO_OUTPUT_PAD.id)
        self.duplex_end_pad_token_id = getattr(
            self.raon_duplex_model, "duplex_end_pad_token_id", AUDIO_OUTPUT_END_PAD.id
        )
        self.audio_input_token_id = getattr(self.raon_duplex_model, "audio_input_token_id", AUDIO_INPUT_PLACEHOLDER.id)
        self.audio_output_token_id = getattr(
            self.raon_duplex_model, "audio_output_token_id", AUDIO_OUTPUT_PLACEHOLDER.id
        )
        self.audio_start_token_id = getattr(self.raon_duplex_model, "audio_start_token_id", AUDIO_START.id)
        self.im_start_token_id = getattr(self.raon_duplex_model, "im_start_token_id", IM_START.id)
        self.speaker_token_id = getattr(self.raon_duplex_model, "speaker_token_id", SPEAKER_EMBEDDING_PLACEHOLDER.id)
        self.text_vocab_size = getattr(self.raon_duplex_model, "text_vocab_size", self.vocab_size - self.codebook_size)
        self.tokenizer = getattr(self.raon_duplex_model, "tokenizer", None)
        if self.tokenizer is None:
            try:
                from transformers import AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(str(runtime_model_path), local_files_only=True)
            except Exception:
                self.tokenizer = None
        self.eos_token_id = getattr(
            getattr(self.text_model_runner, "model", None), "eos_token_id", getattr(self.tokenizer, "eos_token_id", None)
        )

        self.max_allocated_req_pool_indices = max_allocated_req_pool_indices
        self.allocated_req_pool_indices: set[int] = set()

        RaonInferenceModel.__init__(self)

    def register_thinker_capture_hook(self) -> None:
        """Register a forward hook on the thinker capture layer to capture pre-norm hidden states.

        In SGLang/vLLM, decoder layers use fused residual-add + RMSNorm and return
        ``(hidden_states, residual)`` where ``hidden_states`` is the MLP output delta
        and ``residual`` is the accumulated residual stream.  The actual pre-norm hidden
        state (before the model's final RMSNorm) is ``hidden_states + residual``.

        In HuggingFace Transformers, each layer returns the full hidden state directly,
        so ``output[0]`` is already the pre-norm hidden state.
        """
        model = self.text_model_runner.model
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
        elif hasattr(model, "layers"):
            layers = model.layers
        else:
            raise AttributeError("register_thinker_capture_hook: Cannot find layers in text_model_runner.model.")

        def hook(module: nn.Module, input: Any, output: Any) -> None:
            if isinstance(output, tuple) and len(output) == 2:
                # SGLang/vLLM returns (hidden_states, residual); pre-norm = sum.
                self.accepted_thinker_hidden_states = output[0] + output[1]
            elif isinstance(output, tuple):
                self.accepted_thinker_hidden_states = output[0]
            else:
                self.accepted_thinker_hidden_states = output

        cast(nn.Module, layers[self.thinker_capture_layer_index]).register_forward_hook(hook)

    def inference_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor,
        audio_input: torch.Tensor | None = None,
        audio_output: torch.Tensor | None = None,
        audio_input_lengths: torch.Tensor | None = None,
        audio_output_lengths: torch.Tensor | None = None,
        audio_output_codes: torch.Tensor | None = None,
        audio_output_codes_mask: torch.Tensor | None = None,
        audio_input_embeds: torch.Tensor | None = None,
        audio_input_embeds_mask: torch.Tensor | None = None,
        speaker_embeds: torch.Tensor | None = None,
        use_cache: bool | None = False,
        past_key_values: SGLangDecodingMetadata | None = None,
        cache_position: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a single inference step; returns talker hidden states and text logits.

        Args:
            input_ids: Token IDs. Shape: [batch_size, seq_length]. Dtype: long.
            attention_mask: Mask for valid tokens. Shape: [batch_size, seq_length]. Dtype: long or bool.
            position_ids: Token positions. Shape: [batch_size, seq_length]. Dtype: long.
            audio_input: Raw waveform for input-side conditioning.
                Shape: [batch_size, num_samples]. Dtype: float.
            audio_output: Raw waveform for output conditioning.
                Shape: [batch_size, num_samples]. Dtype: float.
            audio_input_lengths: Valid sample count per batch.
                Shape: [batch_size]. Dtype: long.
            audio_output_lengths: Valid sample count per batch.
                Shape: [batch_size]. Dtype: long.
            audio_output_codes: Pre-tokenized audio codes.
                Shape: [batch_size, num_frames, num_code_groups]. Dtype: long.
            audio_output_codes_mask: Mask for valid audio codes.
                Shape: [batch_size, num_frames]. Dtype: bool.
            audio_input_embeds: Pre-computed audio input embeddings.
                Shape: [batch_size, num_frames, feature_dim]. Dtype: float.
            audio_input_embeds_mask: Mask for valid audio input positions.
                Shape: [batch_size, num_frames]. Dtype: bool.
            speaker_embeds: Speaker embeddings.
                Shape: [batch_size, 1, speaker_dim]. Dtype: float.
            use_cache: Unused; SGLang uses metadata-based cache.
            past_key_values: Decoding metadata (KV-cache state); required.

        Returns:
            Tuple of (padded_talker_last_hidden_states, logits).
            padded_talker_last_hidden_states: Talker hidden states, padded to input shape.
                Shape: [batch_size, seq_length, hidden_size]. Dtype: float.
            logits: Text logits over vocab.
                Shape: [batch_size, seq_length, vocab_size]. Dtype: float.
        """
        assert (metadata := past_key_values) is not None, "past_key_values (decoding metadata) is required."

        # Separate the current-frame mask (for SGLang thinker) from the cumulative mask (for HF talker).
        # The thinker uses SGLang's metadata-based KV cache and only needs the current frame's mask.
        # The talker uses HF DynamicCache and needs a full-sequence mask covering all cached + current positions.
        if attention_mask is None:
            current_frame_mask = torch.ones_like(input_ids)
            attention_mask = current_frame_mask
        else:
            current_frame_mask = attention_mask[:, -input_ids.shape[1] :]
            attention_mask = current_frame_mask

        if audio_output_codes is None and self.supports_audio_output:
            assert self.raon_duplex_model.config.code_predictor_config.num_code_groups is not None, (
                "Config code_predictor num_code_groups must be set."
            )
            audio_output_inputs = self.raon_duplex_model.tokenize_audio(
                audio=audio_output,
                audio_lengths=audio_output_lengths,
                num_code_groups=self.raon_duplex_model.config.code_predictor_config.num_code_groups,
            )
            audio_output_codes = audio_output_inputs.audio_codes
            audio_output_codes_mask = audio_output_inputs.audio_codes_mask

        if audio_input_embeds is None and audio_input is not None:
            audio_input_outputs = self.raon_duplex_model.get_audio_input_embeds(
                audio=audio_input,
                audio_lengths=audio_input_lengths,
            )
            audio_input_embeds = audio_input_outputs.audio_embeds
            audio_input_embeds_mask = audio_input_outputs.audio_embeds_mask

        inputs_embeds = self.get_input_embeddings()(input_ids)
        inputs_embeds = self.raon_duplex_model.update_inputs_embeds(
            inputs_embeds=inputs_embeds,
            input_ids=input_ids,
            audio_output_codes=audio_output_codes,
            audio_output_codes_mask=audio_output_codes_mask,
            audio_input_embeds=audio_input_embeds,
            audio_input_embeds_mask=audio_input_embeds_mask,
            speaker_embeds=speaker_embeds,
        )

        inputs = self.get_inputs_and_update_metadata(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            metadata=metadata,
        )
        runner_output = self.text_model_runner.forward(inputs)
        if isinstance(runner_output, tuple):
            outputs = runner_output[0]
        elif isinstance(runner_output, ModelRunnerOutput):
            outputs = runner_output.logits_output
        else:
            outputs = runner_output

        # Runner output is the thinker's post-norm hidden states (text_model.norm applied inside runner).
        thinker_post_norm: torch.Tensor = outputs.hidden_states

        # Capture pre-norm hidden states from hook (set by register_thinker_capture_hook).
        assert self.accepted_thinker_hidden_states is not None, (
            "accepted_thinker_hidden_states must be set by thinker capture hook."
        )
        captured_pre_norm = self.accepted_thinker_hidden_states
        self.accepted_thinker_hidden_states = None

        # Pad post-norm for text logits.
        padded_thinker = torch.zeros(
            (*input_ids.shape, thinker_post_norm.shape[-1]),
            dtype=thinker_post_norm.dtype,
            device=thinker_post_norm.device,
        )
        padded_thinker[attention_mask == 1] = thinker_post_norm
        logits = F.linear(padded_thinker, self.lm_head_weight)[..., : self.vocab_size]

        if not self.supports_audio_output:
            # Audio output disabled: skip talker computation entirely.
            # Return dummy talker hidden states (never used when force_text_output=True).
            return logits.new_zeros((*input_ids.shape, 1)), logits

        # Pad pre-norm for talker input.
        padded_captured = torch.zeros(
            (*input_ids.shape, captured_pre_norm.shape[-1]),
            dtype=captured_pre_norm.dtype,
            device=captured_pre_norm.device,
        )
        padded_captured[attention_mask == 1] = captured_pre_norm

        # Project thinker hidden states → talker input, then run talker with cache.
        # Build cumulative attention mask for the HF talker: past (all cached) + current frame.
        # Talker KV cache is stored per-session in metadata (not on self) for concurrent safety.
        if metadata.talker_attention_mask is None:
            talker_mask = current_frame_mask
        else:
            talker_mask = torch.cat([metadata.talker_attention_mask, current_frame_mask], dim=1)
        metadata.talker_attention_mask = talker_mask

        talker_input = self.raon_duplex_model.thinker_to_talker_proj(padded_captured)
        talker_outputs = self.raon_duplex_model.talker(
            inputs_embeds=talker_input,
            attention_mask=talker_mask,
            past_key_values=metadata.talker_past_key_values,
            use_cache=True,
        )
        metadata.talker_past_key_values = talker_outputs.past_key_values

        return talker_outputs.last_hidden_state, logits

    def get_inputs_and_update_metadata(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        metadata: SGLangDecodingMetadata,
    ) -> ForwardBatch:
        """Flatten inputs for SGLang, update metadata, and build a ForwardBatch for the text model runner.

        Args:
            inputs_embeds: Combined token embeddings.
                Shape: [batch_size, seq_length, hidden_size]. Dtype: float.
            attention_mask: Mask for valid positions.
                Shape: [batch_size, seq_length]. Dtype: long or bool.
            position_ids: Token positions. Shape: [batch_size, seq_length]. Dtype: long.
            metadata: Decoding metadata to update with new sequence lengths and cache locations.

        Returns:
            ForwardBatch ready for text_model_runner.forward().
        """
        assert inputs_embeds.shape[0] == metadata.batch_size, "inputs_embeds batch size must match metadata batch_size."
        is_prefill = inputs_embeds.shape[1] > 1

        old_seq_lengths, new_seq_lengths, out_cache_loc = metadata.update(
            attention_mask=attention_mask,
            is_prefill=is_prefill,
        )
        new_seq_lengths_sum = int(new_seq_lengths.sum().item())

        flat_inputs_embeds = inputs_embeds[attention_mask == 1]
        dummy_input_ids = torch.zeros(flat_inputs_embeds.shape[0], dtype=torch.long, device=inputs_embeds.device)

        batch = ModelWorkerBatch(
            forward_mode=ForwardMode.EXTEND if is_prefill else ForwardMode.DECODE,
            input_ids=dummy_input_ids,
            req_pool_indices=metadata.req_pool_indices,
            seq_lens=new_seq_lengths,
            out_cache_loc=out_cache_loc,
            seq_lens_cpu=new_seq_lengths.cpu(),
            seq_lens_sum=new_seq_lengths_sum,
            return_logprob=False,
            top_logprobs_nums=None,
            token_ids_logprobs=None,
            global_num_tokens=None,
            global_num_tokens_for_logprob=None,
            is_extend_in_batch=is_prefill,
            can_run_dp_cuda_graph=False,
            tbo_split_seq_index=None,
            global_forward_mode=None,
            extend_num_tokens=flat_inputs_embeds.shape[0] if is_prefill else None,
            extend_seq_lens=(new_seq_lengths - old_seq_lengths).cpu().tolist() if is_prefill else None,
            extend_prefix_lens=old_seq_lengths.cpu().tolist() if is_prefill else None,
            extend_logprob_start_lens=None,
            extend_input_logprob_token_ids=None,
            multimodal_inputs=None,
            encoder_cached=None,
            encoder_lens=None,
            encoder_lens_cpu=None,
            encoder_out_cache_loc=None,
            lora_ids=None,
            sampling_info=metadata.sampling_info,
            input_embeds=flat_inputs_embeds,
            capture_hidden_mode=CaptureHiddenMode.FULL,
        )
        return ForwardBatch.init_new(batch, model_runner=self.text_model_runner)

    def tokenize_audio(self, *args: Any, **kwargs: Any) -> AudioTokenizerOutput:
        """Tokenize audio into codes; delegates to the underlying duplex model.

        Typical args: audio, audio_lengths, num_code_groups. Returns audio_codes and audio_codes_mask.

        Returns:
            AudioTokenizerOutput with audio_codes (Shape: [batch_size, num_frames, num_code_groups]. Dtype: long.),
            audio_codes_mask (Shape: [batch_size, num_frames]. Dtype: bool.), and optional mimi_features.
        """
        return self.raon_duplex_model.tokenize_audio(*args, **kwargs)

    def get_audio_input_embeds(self, *args: Any, **kwargs: Any) -> AudioEncoderOutput:
        """Compute audio input embeddings from raw waveform; delegates to the underlying duplex model.

        Typical args: audio, audio_lengths. Returns audio_embeds and audio_embeds_mask.

        Returns:
            AudioEncoderOutput with audio_embeds (Shape: [batch_size, num_frames, feature_dim]. Dtype: float.),
            audio_embeds_mask (Shape: [batch_size, num_frames]. Dtype: bool.).
        """
        return self.raon_duplex_model.get_audio_input_embeds(*args, **kwargs)

    def decode_audio(self, *args: Any, **kwargs: Any) -> AudioDecoderOutput:
        """Decode audio codes to waveform; delegates to the underlying duplex model.

        Typical first arg: audio_codes. Shape: [batch_size, num_frames, num_code_groups]. Dtype: long.

        Returns:
            AudioDecoderOutput with audio waveform.
                Shape: [batch_size, num_channels, num_samples]. Dtype: float.
        """
        return self.raon_duplex_model.decode_audio(*args, **kwargs)

    def generate_audio_codes(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Generate next-frame audio codes from talker hidden state; delegates to the underlying duplex model.

        Typical first arg: talker_last_hidden_state. Shape: [batch_size, seq_length, hidden_size]. Dtype: float.

        Returns:
            Sampled audio codes for the next frame. Shape: [batch_size, num_code_groups]. Dtype: long.
        """
        return self.raon_duplex_model.generate_audio_codes(*args, **kwargs)

    def get_input_embeddings(self) -> nn.Embedding:
        """Return the text model's token embedding layer."""
        model = self.text_model_runner.model
        if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            return model.model.embed_tokens  # type: ignore
        elif hasattr(model, "embed_tokens"):
            return model.embed_tokens  # type: ignore
        else:
            raise AttributeError("Cannot find embed_tokens in text_model_runner.model.")

    def get_proj_code(self) -> nn.Linear:
        """Return the audio code projection layer."""
        return self.raon_duplex_model.get_proj_code()

    def get_model(self) -> RaonModel:
        """Return the underlying duplex model (without the SGLang text model runner)."""
        return self.raon_duplex_model

    def init_past_key_values(
        self,
        batch_size: int,
        max_sequence_length: int,
        prev_cache: SGLangDecodingMetadata | None = None,
    ) -> SGLangDecodingMetadata:
        """Initialize or reuse decoding metadata for KV-cache tracking.

        Args:
            batch_size: Number of concurrent requests.
            max_sequence_length: Maximum sequence length for the cache.
            prev_cache: Optional existing metadata to reset and reuse.

        Returns:
            SGLangDecodingMetadata with req_pool_indices, seq_lens, and sampling info.
        """
        assert isinstance(req_to_token_pool := self.text_model_runner.req_to_token_pool, ReqToTokenPool), (
            "req_to_token_pool must be a ReqToTokenPool instance."
        )
        req_pool_capacity = int(getattr(req_to_token_pool, "size", req_to_token_pool.req_to_token.shape[0]))
        assert self.max_allocated_req_pool_indices <= req_pool_capacity, (
            "max_allocated_req_pool_indices must not exceed req_to_token_pool size: "
            f"{self.max_allocated_req_pool_indices} > {req_pool_capacity}."
        )
        assert hasattr(self.text_model_runner, "token_to_kv_pool_allocator"), (
            "text_model_runner must expose token_to_kv_pool_allocator."
        )
        token_to_kv_pool_allocator = self.text_model_runner.token_to_kv_pool_allocator

        if prev_cache is not None:
            assert prev_cache.req_pool_indices.numel() == batch_size, (
                "prev_cache batch size must match the requested batch size for in-place reuse."
            )
            # `prev_cache` is consumed in place by the new state; callers must not free the old owner afterwards.
            prev_cache.reset()
            prev_cache.req_to_token_pool = req_to_token_pool
            prev_cache.token_to_kv_pool_allocator = token_to_kv_pool_allocator
            prev_cache.max_sequence_length = max_sequence_length
            prev_cache.batch_size = batch_size
            prev_cache.talker_past_key_values = DynamicCache()
            prev_cache.talker_attention_mask = None
            return prev_cache

        sampling_info = SamplingBatchInfo(
            temperatures=torch.tensor([[1.0]], device=self.device),
            top_ps=torch.tensor([1.0], device=self.device),
            top_ks=torch.tensor([0], dtype=torch.int32, device=self.device),
            min_ps=torch.tensor([0.0], device=self.device),
            is_all_greedy=True,
            need_top_p_sampling=False,
            need_top_k_sampling=False,
            need_min_p_sampling=False,
            vocab_size=self.vocab_size,
        )

        return SGLangDecodingMetadata(
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            req_pool_indices=self.allocate_req_pool_indices(batch_size),
            seq_lens=torch.zeros(batch_size, dtype=torch.long, device=self.device),
            max_sequence_length=max_sequence_length,
            batch_size=batch_size,
            sampling_info=sampling_info,
            device=self.device,
            talker_past_key_values=DynamicCache(),
            talker_attention_mask=None,
        )

    def allocate_req_pool_indices(self, batch_size: int) -> torch.Tensor:
        """Allocate request pool indices from the available pool for a batch of requests.

        Args:
            batch_size: Number of indices to allocate.

        Returns:
            Tensor of req_pool indices. Shape: [batch_size]. Dtype: long.
        """
        req_to_token_pool = self.text_model_runner.req_to_token_pool
        req_pool_capacity = int(getattr(req_to_token_pool, "size", req_to_token_pool.req_to_token.shape[0]))
        alloc_limit = min(self.max_allocated_req_pool_indices, req_pool_capacity)

        indices: list[int] = []
        for i in range(alloc_limit):
            if len(indices) >= batch_size:
                break

            if i not in self.allocated_req_pool_indices:
                self.allocated_req_pool_indices.add(i)
                indices.append(i)

        assert len(indices) == batch_size, f"Failed to allocate {batch_size} req pool indices."
        return torch.tensor(indices, dtype=torch.long, device=self.device)

    def free_req_pool_indices(self, metadata: SGLangDecodingMetadata) -> None:
        """Return req_pool indices from metadata to the pool for reuse."""
        for i in metadata.req_pool_indices.tolist():
            self.allocated_req_pool_indices.discard(i)

    def free_past_key_values(self, past_key_values: SGLangDecodingMetadata) -> None:
        """Release resources associated with past_key_values (req_pool indices and talker cache)."""
        past_key_values.free_allocated_kv_slots()
        past_key_values.seq_lens = torch.zeros_like(past_key_values.seq_lens)
        self.free_req_pool_indices(past_key_values)
        past_key_values.talker_past_key_values = None
        past_key_values.talker_attention_mask = None

    @staticmethod
    def _load_text_model_runner(
        path: str,
        dtype: str,
        mem_fraction_static: float,
        disable_cuda_graph: bool | None,
        cuda_graph_max_bs: int | None,
        max_running_requests: int | None,
        prefill_max_requests: int | None,
        max_total_tokens: int | None,
        max_prefill_tokens: int | None,
        chunked_prefill_size: int | None,
        gpu_id: int,
        tp_rank: int,
        tp_size: int,
        moe_ep_rank: int,
        moe_ep_size: int,
        pp_rank: int,
        pp_size: int,
        nccl_port: int,
    ) -> ModelRunner:
        set_logs(recompiles=False)
        model_config = ModelConfig(model_path=path, dtype=dtype)
        server_args = ServerArgs(model_path=path, dtype=dtype)

        # Prefer Triton attention to avoid FlashInfer JIT/nvcc requirements
        # in the repo-local verification/demo environment.
        if hasattr(server_args, "attention_backend"):
            server_args.attention_backend = "triton"
        if hasattr(server_args, "sampling_backend"):
            server_args.sampling_backend = "pytorch"

        # Ensure SGLang uses our requested memory fraction (Ray/Hydra override).
        # Some sglang versions consult `server_args.mem_fraction_static` during pool init.
        if hasattr(server_args, "mem_fraction_static"):
            server_args.mem_fraction_static = mem_fraction_static

        # Reduce/disable CUDA graph capture (avoids "Capturing batches ..." init-time memory spikes).
        if disable_cuda_graph is not None:
            for attr in ("disable_cuda_graph", "disable_cuda_graph_capture"):
                if hasattr(server_args, attr):
                    setattr(server_args, attr, bool(disable_cuda_graph))

        if cuda_graph_max_bs is not None:
            for attr in ("cuda_graph_max_bs", "cuda_graph_max_batch_size"):
                if hasattr(server_args, attr):
                    setattr(server_args, attr, int(cuda_graph_max_bs))

        # Reduce continuous batching / pool sizing (helps multi-worker-per-GPU).
        if max_running_requests is not None and hasattr(server_args, "max_running_requests"):
            server_args.max_running_requests = int(max_running_requests)

        if prefill_max_requests is not None and hasattr(server_args, "prefill_max_requests"):
            server_args.prefill_max_requests = int(prefill_max_requests)

        if max_total_tokens is not None and hasattr(server_args, "max_total_tokens"):
            server_args.max_total_tokens = int(max_total_tokens)

        if max_prefill_tokens is not None and hasattr(server_args, "max_prefill_tokens"):
            server_args.max_prefill_tokens = int(max_prefill_tokens)

        if chunked_prefill_size is not None and hasattr(server_args, "chunked_prefill_size"):
            server_args.chunked_prefill_size = int(chunked_prefill_size)

        model_runner = ModelRunner(
            model_config=model_config,
            mem_fraction_static=mem_fraction_static,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            tp_size=tp_size,
            moe_ep_rank=moe_ep_rank,
            moe_ep_size=moe_ep_size,
            pp_rank=pp_rank,
            pp_size=pp_size,
            nccl_port=nccl_port,
            server_args=server_args,
        )

        def patched_forward_extend(
            self: ModelRunner,
            forward_batch: ForwardBatch,
            skip_attn_backend_init: bool = False,
            pp_proxy_tensors: Any = None,
        ) -> tuple[LogitsProcessorOutput, bool]:
            kwargs = {}
            if self.support_pp:
                kwargs["pp_proxy_tensors"] = pp_proxy_tensors

            if forward_batch.input_embeds is not None:
                kwargs["input_embeds"] = forward_batch.input_embeds

            if not self.is_generation:
                kwargs["get_embedding"] = True

            can_run_graph = (
                self.piecewise_cuda_graph_runner is not None
                and self.piecewise_cuda_graph_runner.can_run(forward_batch)
            )
            if can_run_graph:
                return self.piecewise_cuda_graph_runner.replay(forward_batch, **kwargs), can_run_graph  # type: ignore[return-value]

            if not skip_attn_backend_init:
                self.attn_backend.init_forward_metadata(forward_batch)

            return (
                self.model.forward(
                    forward_batch.input_ids,
                    forward_batch.positions,
                    forward_batch,
                    **kwargs,
                ),
                can_run_graph,
            )

        def patched_forward_decode(
            self: ModelRunner,
            forward_batch: ForwardBatch,
            skip_attn_backend_init: bool = False,
            pp_proxy_tensors: Any = None,
        ) -> LogitsProcessorOutput:
            if not skip_attn_backend_init:
                self.attn_backend.init_forward_metadata(forward_batch)

            kwargs = {}
            if self.support_pp:
                kwargs["pp_proxy_tensors"] = pp_proxy_tensors

            if forward_batch.input_embeds is not None:
                kwargs["input_embeds"] = forward_batch.input_embeds

            return self.model.forward(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
                **kwargs,
            )

        model_runner.forward_extend = types.MethodType(patched_forward_extend, model_runner)
        model_runner.forward_decode = types.MethodType(patched_forward_decode, model_runner)

        set_logs(recompiles=True)
        return model_runner


__all__ = [
    "SGLangRaonModel",
]
