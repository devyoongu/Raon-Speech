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

"""Utility functions: loss parameter readers and dtype casting helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, overload

import torch
from torch import nn


def load_safetensors_by_prefix(
    model_path: str | Path,
    prefixes: dict[str, str],
    dtype: torch.dtype | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    """Load safetensors shards and split state_dict by key prefixes.

    Handles both sharded (``model.safetensors.index.json``) and single-file
    checkpoints, and works with both local directories and HuggingFace Hub IDs.
    Only shards containing at least one key matching a requested prefix are
    opened.

    Args:
        model_path: Path to directory containing safetensors files, or a
            HuggingFace Hub model ID.
        prefixes: Mapping of ``{name: prefix_string}`` to extract.  Each
            prefix should include a trailing ``"."`` if applicable.
        dtype: Optional dtype to cast tensors to.  When ``None`` tensors are
            returned in their stored dtype.

    Returns:
        Dict mapping each name from ``prefixes`` to a prefix-stripped
        state_dict containing the matching tensors.
    """
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    model_path_str = str(model_path)
    is_local = os.path.isdir(model_path_str)

    def _resolve(filename: str) -> str:
        if is_local:
            return os.path.join(model_path_str, filename)
        return hf_hub_download(repo_id=model_path_str, filename=filename)

    # Build shard -> list-of-keys mapping from the index, or fall back to a
    # single safetensors file.
    weight_map: dict[str, str] | None = None
    try:
        index_path = _resolve("model.safetensors.index.json")
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        shard_files: list[str] = sorted(set(weight_map.values()))
    except Exception:
        shard_files = ["model.safetensors"]

    # Initialise result containers.
    result: dict[str, dict[str, torch.Tensor]] = {name: {} for name in prefixes}

    for shard_name in shard_files:
        # When we have a weight map, skip shards that contain no relevant keys.
        if weight_map is not None:
            shard_keys = [k for k, v in weight_map.items() if v == shard_name]
            if not any(
                k.startswith(prefix)
                for k in shard_keys
                for prefix in prefixes.values()
            ):
                continue

        shard_path = _resolve(shard_name)
        with safe_open(shard_path, framework="pt") as f:
            for key in f.keys():
                for name, prefix in prefixes.items():
                    if key.startswith(prefix):
                        stripped_key = key.removeprefix(prefix)
                        tensor = f.get_tensor(key)
                        if dtype is not None:
                            tensor = tensor.to(dtype)
                        result[name][stripped_key] = tensor
                        break  # a key can only match one prefix

    return result


def _read_loss_param(env_key: str, config: Any, attr_name: str, default: float) -> float:
    """Read a loss parameter with precedence: environment variable > config attribute > default.

    Args:
        env_key: Environment variable name (e.g. ``RAON_TEXT_LOSS_WEIGHT``).
        config: Model config object to fall back to.
        attr_name: Attribute name on the config (e.g. ``text_loss_weight``).
        default: Default value if neither env var nor config attribute is set.

    Returns:
        The resolved parameter value as a float.
    """
    env_val = os.environ.get(env_key)
    if env_val is not None:
        return float(env_val)
    return float(getattr(config, attr_name, default))


def _read_acoustic_loss_weights(config: Any, num_code_groups: int) -> list[float]:
    """Read acoustic loss weights with precedence: env var > config > default.

    The environment variable ``RAON_ACOUSTIC_LOSS_WEIGHTS`` is a comma-separated
    string of floats (one per acoustic codebook, i.e. ``num_code_groups - 1`` values).

    Args:
        config: Model config object to fall back to.
        num_code_groups: Total number of code groups (semantic + acoustic).

    Returns:
        List of acoustic loss weights with length ``num_code_groups - 1``.
    """
    env_val = os.environ.get("RAON_ACOUSTIC_LOSS_WEIGHTS")
    if env_val is not None:
        weights = [float(w.strip()) for w in env_val.split(",")]
        assert len(weights) == num_code_groups - 1, (
            f"RAON_ACOUSTIC_LOSS_WEIGHTS has {len(weights)} values, expected {num_code_groups - 1}"
        )
        return weights
    config_weights = getattr(config, "acoustic_loss_weights", None)
    if config_weights is not None:
        return [float(w) for w in config_weights]
    return [0.1] * (num_code_groups - 1)


def _get_module_dtype(module: nn.Module) -> torch.dtype:
    try:
        return next(module.parameters()).dtype
    except StopIteration:
        return torch.float32


@overload
def cast_float_inputs(tensor: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor: ...
@overload
def cast_float_inputs(tensor: None, target_dtype: torch.dtype) -> None: ...


def cast_float_inputs(tensor: torch.Tensor | None, target_dtype: torch.dtype) -> torch.Tensor | None:
    if tensor is None:
        return None
    if tensor.is_floating_point() and tensor.dtype != target_dtype:
        return tensor.to(target_dtype)
    return tensor


@overload
def cast_to_module_dtype(tensor: torch.Tensor, module: nn.Module) -> torch.Tensor: ...
@overload
def cast_to_module_dtype(tensor: None, module: nn.Module) -> None: ...


def cast_to_module_dtype(tensor: torch.Tensor | None, module: nn.Module) -> torch.Tensor | None:
    return cast_float_inputs(tensor, _get_module_dtype(module))


DTYPE_MAP: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def resolve_dtype(dtype_str: str) -> torch.dtype:
    """Convert a dtype string to a torch.dtype.

    Args:
        dtype_str: One of ``"bfloat16"``, ``"float16"``, ``"float32"``.

    Returns:
        Corresponding ``torch.dtype``.
    """
    return DTYPE_MAP[dtype_str]
