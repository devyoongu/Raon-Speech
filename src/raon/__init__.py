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

"""Public API for the raon package."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "DatasetConfig",
    "DuplexInputs",
    "RaonConfig",
    "RaonDuplexConfig",
    "RaonDuplexModel",
    "RaonDuplexProcessor",
    "RaonModel",
    "RaonPipeline",
    "RaonProcessor",
    "make_raon_data_module",
]

_EXPORT_TO_MODULE = {
    "DatasetConfig": "raon.utils.data",
    "DuplexInputs": "raon.types",
    "make_raon_data_module": "raon.utils.data",
    "RaonConfig": "raon.models.raon",
    "RaonDuplexConfig": "raon.models.raon",
    "RaonDuplexModel": "raon.models.raon",
    "RaonDuplexProcessor": "raon.utils.processor",
    "RaonModel": "raon.models.raon",
    "RaonPipeline": "raon.pipeline",
    "RaonProcessor": "raon.utils.processor",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module 'raon' has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


# Eagerly register with HuggingFace AutoConfig/AutoModel so that
# ``import raon`` alone is sufficient for AutoModel.from_pretrained() to work.
import_module("raon.models.raon")
