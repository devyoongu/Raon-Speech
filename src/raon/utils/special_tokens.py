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

"""Special token definitions and tokenizer patching utilities for the raon model."""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpecialToken:
    """Frozen container for a special token's id and surface text."""

    id: int
    text: str

    def __int__(self) -> int:
        return self.id

    def __str__(self) -> str:
        return self.text


PAD = SpecialToken(id=151679, text="<|endoftext|>")
IM_START = SpecialToken(id=151644, text="<|im_start|>")
IM_END = SpecialToken(id=151645, text="<|im_end|>")
AUDIO_START = SpecialToken(id=151669, text="<|audio_start|>")
AUDIO_END = SpecialToken(id=151670, text="<|audio_end|>")
SPEAKER_EMBEDDING_PLACEHOLDER = SpecialToken(id=151671, text="<|speaker_embedding_placeholder|>")
AUDIO_OUTPUT_PLACEHOLDER = SpecialToken(id=151675, text="<|audio_output_placeholder|>")
AUDIO_INPUT_PLACEHOLDER = SpecialToken(id=151676, text="<|audio_input_placeholder|>")
AUDIO_OUTPUT_PAD = SpecialToken(id=151677, text="<|audio_output_pad|>")
AUDIO_OUTPUT_END_PAD = SpecialToken(id=151678, text="<|audio_output_end_pad|>")

# Duplex SIL token (dedicated token, not repurposed FIM)
DUPLEX_SIL = SpecialToken(id=151672, text="<|audio_output_sil|>")

# Backchannel onset token (marks "uh-huh", "mm-hmm" turns instead of EPAD)
AUDIO_OUTPUT_BC = SpecialToken(id=151673, text="<|audio_output_backchannel|>")

PRETRAINING_AUDIO_TAG = "<audio>"
AUDIO_PLACEHOLDER = "<|audio_placeholder|>"
LOSS_IGNORE_INDEX = -100

ALL_SPECIAL_TOKENS: list[SpecialToken] = [
    PAD,
    IM_START,
    IM_END,
    AUDIO_START,
    AUDIO_END,
    SPEAKER_EMBEDDING_PLACEHOLDER,
    AUDIO_OUTPUT_PLACEHOLDER,
    AUDIO_INPUT_PLACEHOLDER,
    AUDIO_OUTPUT_PAD,
    AUDIO_OUTPUT_END_PAD,
    DUPLEX_SIL,
    AUDIO_OUTPUT_BC,
]


def _mk_added_token_payload(token_id: int, content: str) -> dict[str, Any]:
    return {
        "id": token_id,
        "content": content,
        "single_word": False,
        "lstrip": False,
        "rstrip": False,
        "normalized": False,
        "special": True,
    }


def _tokenizer_is_aligned(tokenizer: Any) -> bool:
    for token in ALL_SPECIAL_TOKENS:
        encoded = tokenizer.encode(token.text, add_special_tokens=False)
        if encoded != [token.id]:
            return False
    return True


def patch_tokenizer_files(tokenizer_dir: Path) -> None:
    """Patch tokenizer files on disk to align special token ids and surface text.

    Modifies vocab.json, tokenizer.json, added_tokens.json, tokenizer_config.json,
    and special_tokens_map.json in place, and ensures ALL_SPECIAL_TOKENS are
    correctly registered.

    Args:
        tokenizer_dir: Directory containing the tokenizer files to patch.
    """
    expected_by_id = {token.id: token.text for token in ALL_SPECIAL_TOKENS}
    vocab_path = tokenizer_dir / "vocab.json"
    if vocab_path.exists():
        vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
        for token_id, token_text in expected_by_id.items():
            vocab[token_text] = token_id
        vocab_path.write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")

    tokenizer_json_path = tokenizer_dir / "tokenizer.json"
    tokenizer_json = json.loads(tokenizer_json_path.read_text(encoding="utf-8"))

    model_vocab = tokenizer_json.get("model", {}).get("vocab")
    if isinstance(model_vocab, dict):
        for token_id, token_text in expected_by_id.items():
            model_vocab[token_text] = token_id

    added_tokens: list[dict[str, Any]] = tokenizer_json.get("added_tokens", [])
    by_id: dict[int, dict[str, Any]] = {}
    for entry in added_tokens:
        token_id = int(entry["id"])
        by_id[token_id] = entry

    for token_id, token_text in expected_by_id.items():
        entry = by_id.get(token_id)
        if entry is None:
            by_id[token_id] = _mk_added_token_payload(token_id, token_text)
            continue
        entry["content"] = token_text
        entry["single_word"] = False
        entry["lstrip"] = False
        entry["rstrip"] = False
        entry["normalized"] = False
        entry["special"] = True

    tokenizer_json["added_tokens"] = [by_id[token_id] for token_id in sorted(by_id.keys())]
    tokenizer_json_path.write_text(json.dumps(tokenizer_json, ensure_ascii=False, indent=2), encoding="utf-8")

    added_tokens_path = tokenizer_dir / "added_tokens.json"
    if added_tokens_path.exists():
        added_tokens_map = json.loads(added_tokens_path.read_text(encoding="utf-8"))
        for token in ALL_SPECIAL_TOKENS:
            added_tokens_map[token.text] = token.id
        added_tokens_path.write_text(json.dumps(added_tokens_map, ensure_ascii=False, indent=2), encoding="utf-8")

    tokenizer_config_path = tokenizer_dir / "tokenizer_config.json"
    if tokenizer_config_path.exists():
        tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
        tokenizer_config["additional_special_tokens"] = [AUDIO_INPUT_PLACEHOLDER.text]
        tokenizer_config_path.write_text(json.dumps(tokenizer_config, ensure_ascii=False, indent=2), encoding="utf-8")

    special_tokens_map_path = tokenizer_dir / "special_tokens_map.json"
    if special_tokens_map_path.exists():
        special_tokens_map = json.loads(special_tokens_map_path.read_text(encoding="utf-8"))
        special_tokens_map["audio_bos_token"] = AUDIO_START.text
        special_tokens_map["audio_eos_token"] = AUDIO_END.text
        special_tokens_map["audio_token"] = AUDIO_OUTPUT_PLACEHOLDER.text
        special_tokens_map["additional_special_tokens"] = [AUDIO_INPUT_PLACEHOLDER.text]
        special_tokens_map_path.write_text(json.dumps(special_tokens_map, ensure_ascii=False, indent=2), encoding="utf-8")


def update_tokenizer(tokenizer: Any) -> Any:
    """Ensure tokenizer special tokens match expected mapping; patch in-place if needed.

    Saves tokenizer to a temp directory, patches files via patch_tokenizer_files,
    loads the patched tokenizer, and updates the original tokenizer's attributes.
    Raises RuntimeError if alignment cannot be achieved.

    Args:
        tokenizer: HuggingFace tokenizer instance to update.

    Returns:
        The same tokenizer instance with updated special token mappings.
    """
    if _tokenizer_is_aligned(tokenizer):
        return tokenizer

    logger.warning("Tokenizer special token mapping is outdated. Applying overrides.")
    for token in ALL_SPECIAL_TOKENS:
        current_text = tokenizer.convert_ids_to_tokens(token.id)
        if current_text != token.text:
            logger.warning(f"id {token.id} token mismatch: current={current_text!r}, expected={token.text!r}. Overriding.")

    tokenizer_cls = tokenizer.__class__
    with tempfile.TemporaryDirectory(prefix="raon_tokenizer_patch_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        tokenizer.save_pretrained(tmp_path)
        patch_tokenizer_files(tmp_path)
        patched = tokenizer_cls.from_pretrained(tmp_path)

    tokenizer.__dict__.update(patched.__dict__)

    if not _tokenizer_is_aligned(tokenizer):
        raise RuntimeError("Failed to align tokenizer special tokens with required mapping.")

    return tokenizer
