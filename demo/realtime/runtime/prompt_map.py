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

"""Prompt-key resolver for realtime duplex sessions."""

from __future__ import annotations

from raon.utils.duplex_data import CHANNEL_DUPLEX_TO_SYSTEM_MESSAGE, get_duplex_system_message_key
from raon.utils.duplex_prompt_catalog import build_system_prompt


def resolve_prompt(
    prompt_text: str,
    prompt_role: str,
    persona: str | None = None,
    persona_context: str | None = None,
) -> str:
    """Resolve canonical or shorthand system-prompt keys to prompt text.

    If persona is provided, builds a persona-based system prompt instead.
    """
    if prompt_role != "system":
        return prompt_text

    # Persona-based prompt takes priority when persona is specified.
    if persona is not None:
        return build_system_prompt(persona=persona, context=persona_context, deterministic=True)

    if prompt_text in CHANNEL_DUPLEX_TO_SYSTEM_MESSAGE:
        return CHANNEL_DUPLEX_TO_SYSTEM_MESSAGE[prompt_text]

    parts = [part.strip() for part in prompt_text.split(":") if part.strip()]
    if len(parts) == 2:
        language = "eng"
        channel, speak_mode = parts
    elif len(parts) == 3:
        language, channel, speak_mode = parts
    else:
        return prompt_text

    if language != "eng":
        return prompt_text
    if channel not in {"full_duplex", "duplex_instruct"}:
        return prompt_text
    if speak_mode not in {"speak-first", "listen-first"}:
        return prompt_text

    key = get_duplex_system_message_key(
        language=language,  # type: ignore[arg-type]
        channel=channel,  # type: ignore[arg-type]
        speak_first=(speak_mode == "speak-first"),
    )
    return CHANNEL_DUPLEX_TO_SYSTEM_MESSAGE.get(key, prompt_text)
