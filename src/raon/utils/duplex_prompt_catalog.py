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

"""Duplex system prompt builder and persona catalog loader."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

_DEFAULT_CATALOG_PATH = Path(__file__).resolve().parents[3] / "data" / "duplex" / "personas.json"

_cached_catalog: dict[str, Any] | None = None

# Embedded fallback catalog used when the JSON file is not available (e.g. Hub loading).
_EMBEDDED_CATALOG: dict[str, Any] = {
    "system_prompt_base": "You are engaging in real-time conversation.",
    "name": "Raon",
    "personas": {
        "general": "a friendly and helpful assistant",
        "game": "a game host who facilitates interactive speech games with the user",
        "scenario_movie": "a movie and entertainment guide",
        "scenario_banking": "a personal banking advisor",
        "scenario_fitness": "a fitness coach",
        "scenario_shopping": "an online shopping assistant",
        "scenario_pet": "a pet care advisor",
        "scenario_healthcare": "a healthcare scheduling assistant",
        "scenario_realestate": "a real estate advisor",
        "scenario_techsupport": "a tech support specialist",
        "scenario_carrental": "a car rental and navigation assistant",
        "scenario_event": "an event planning assistant",
        "scenario_restaurant": "a restaurant and food delivery assistant",
        "scenario_language": "a language tutor",
        "scenario_travel": "a travel planning assistant",
        "scenario_interview": "a job interview coach",
        "scenario_game_npc": "a game NPC assistant",
    },
}


def load_persona_catalog(catalog_path: str | Path | None = None) -> dict[str, Any]:
    """Load persona catalog from JSON file.

    Args:
        catalog_path: Path to catalog JSON. Defaults to raon/data/duplex/personas.json.

    Returns:
        Parsed catalog dictionary.
    """
    global _cached_catalog
    path = Path(catalog_path) if catalog_path is not None else _DEFAULT_CATALOG_PATH
    if _cached_catalog is None or catalog_path is not None:
        if path.exists():
            with open(path) as f:
                catalog = json.load(f)
        else:
            catalog = _EMBEDDED_CATALOG
        if catalog_path is None:
            _cached_catalog = catalog
        return catalog
    return _cached_catalog


DEFAULT_ASSISTANT_PERSONA = "a friendly and helpful assistant"


def build_system_prompt(
    persona: str | None = None,
    context: str | None = None,
    name: str | None = None,
    record: dict | None = None,
    catalog_path: str | Path | None = None,
    deterministic: bool = False,
) -> str:
    """Build a system prompt from persona, context, name, or a data record.

    Supports two usage patterns:

    **Direct args** (inference CLI, notebook)::

        build_system_prompt(
            persona="scenario_restaurant",
            context="Discuss restaurant menu choices and delivery preferences with a customer.",
        )

    **Record-based** (training data)::

        build_system_prompt(record={"name": "Raon", "persona": "a restaurant assistant", "context": "..."})

    When ``record`` is provided, the mode is auto-detected:

    - **context** mode (``context`` present): 50% ``"{base} {context}"``, 50% ``"{base} You are {name}, {persona}."``
    - **persona** mode (``persona`` present): ``"{base} You are {name}, {persona}."``
    - **assistant** mode (only ``name`` present): ``"{base} You are {name}, a friendly and helpful assistant."``
    - **none** mode (no fields): ``"{base}"``

    Direct args (``persona``, ``context``, ``name``) override record values.

    Args:
        persona: Persona key (resolved via catalog) or raw persona description string.
        context: Additional context sentence.
        name: Assistant name override.
        record: Data record dict with optional persona/context/name fields.
        catalog_path: Optional override for catalog file path.

    Returns:
        Formatted system prompt string.
    """
    catalog = load_persona_catalog(catalog_path)
    base = catalog.get("system_prompt_base", "You are engaging in real-time conversation.")

    # Merge record fields with direct args (direct args take precedence).
    if record is not None:
        if persona is None:
            persona = record.get("persona")
        if context is None:
            context = record.get("context")
        if name is None:
            name = record.get("name")

    if name is None:
        name = catalog.get("name", "Raon")

    # Try resolving persona key from catalog; if not found, use raw string.
    if persona is not None:
        resolved = catalog.get("personas", {}).get(persona)
        if resolved is not None:
            persona = resolved

    # Auto-detect mode based on available fields.
    if context is not None:
        # Context mode: deterministic always uses context; training randomly
        # alternates between context and persona for data diversity.
        if deterministic or random.random() < 0.5:
            return f"{base} {context}"
        else:
            p = persona if persona else DEFAULT_ASSISTANT_PERSONA
            return f"{base} You are {name}, {p}."
    elif persona is not None:
        # Persona mode.
        return f"{base} You are {name}, {persona}."
    elif record is not None and name != catalog.get("name", "Raon"):
        # Assistant mode: name was explicitly set in record.
        return f"{base} You are {name}, {DEFAULT_ASSISTANT_PERSONA}."
    elif record is not None and record.get("name") is not None:
        # Assistant mode: name field present in record.
        return f"{base} You are {name}, {DEFAULT_ASSISTANT_PERSONA}."
    elif record is not None:
        # None mode: record provided but no persona/context/name.
        return base
    else:
        # No record, no persona — base only.
        return base
