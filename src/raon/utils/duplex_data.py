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

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .special_tokens import (
    AUDIO_OUTPUT_END_PAD,
    AUDIO_OUTPUT_PAD,
    LOSS_IGNORE_INDEX,
)
from ..types import DuplexInputs

# ---------------------------------------------------------------------------
# Data model dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SpeakerItem:
    word: str
    start: float  # seconds
    end: float  # seconds


@dataclass
class UtteranceBound:
    start: float  # seconds
    end: float  # seconds


@dataclass
class DuplexMetadata:
    script: list[list[SpeakerItem]]  # per-channel word lists
    speak_first: list[int]
    include_in_training: list[int]
    language: str
    channel: str
    timeline: list[list[UtteranceBound]] | None = None
    system_prompt: str | None = None


# ---------------------------------------------------------------------------
# System-message lookup
# ---------------------------------------------------------------------------

CHANNEL_DUPLEX_TO_SYSTEM_MESSAGE: dict[str, str] = {
    "eng:full_duplex:speak-first": "You are engaging in real-time conversation.",
    "eng:full_duplex:listen-first": "You are engaging in real-time conversation.",
    "eng:duplex_instruct:speak-first": "You are engaging in real-time conversation.",
    "eng:duplex_instruct:listen-first": "You are engaging in real-time conversation.",
    "kor:full_duplex:speak-first": "당신은 실시간 대화에 참여하고 있습니다. 먼저 말씀하세요.",
    "kor:full_duplex:listen-first": "당신은 실시간 대화에 참여하고 있습니다. 먼저 말하지 마세요.",
    "kor:duplex_instruct:speak-first": "당신은 실시간 대화에 참여하고 있습니다. 먼저 말씀하세요.",
    "kor:duplex_instruct:listen-first": "당신은 사용자와 실시간 대화에 참여하는 어시스턴트입니다. 먼저 말하지 마세요.",
}


def get_duplex_system_message_key(
    language: Literal["eng", "kor"],
    channel: Literal["full_duplex", "duplex_instruct"],
    speak_first: bool,
) -> str:
    speak_mode = "speak-first" if speak_first else "listen-first"
    return f"{language}:{channel}:{speak_mode}"


# ---------------------------------------------------------------------------
# Metadata parsing
# ---------------------------------------------------------------------------


def timeline_turns_to_metadata(record: dict) -> DuplexMetadata:
    """Convert a timeline record into DuplexMetadata.

    Supports two formats:
    1. ``turns`` format (ultrachat, speech-chat): each turn has ``start_sample``,
       ``end_sample``, ``channel``, and ``ipus[].words[]`` with per-word
       ``start_sample``/``end_sample``.
    2. ``scripts`` + ``timeline``/``rough_timeline`` format (fisher, ami):
       ``scripts`` is ``[ch0_words, ch1_words]`` with per-word ``start``/``end``
       in seconds. ``timeline`` or ``rough_timeline`` is a flat list of segment
       dicts with ``channel``, ``start_sec``, ``end_sec``.
    """
    num_channels = 2
    script: list[list[SpeakerItem]] = [[] for _ in range(num_channels)]
    timeline: list[list[UtteranceBound]] = [[] for _ in range(num_channels)]

    if "turns" in record:
        # Format 1: turns-based (ultrachat, speech-chat)
        sample_rate = record.get("sample_rate", 24000)
        for turn in record["turns"]:
            ch = turn["channel"]
            turn_start_sec = turn["start_sample"] / sample_rate
            turn_end_sec = turn["end_sample"] / sample_rate
            timeline[ch].append(UtteranceBound(start=turn_start_sec, end=turn_end_sec))

            for ipu in turn["ipus"]:
                for w in ipu["words"]:
                    script[ch].append(
                        SpeakerItem(
                            word=w["word"],
                            start=w["start_sample"] / sample_rate,
                            end=w["end_sample"] / sample_rate,
                        )
                    )
    elif "scripts" in record:
        # Format 2: scripts + timeline/rough_timeline (fisher, ami)
        raw_timeline = record.get("timeline") or record.get("rough_timeline") or []
        raw_scripts = record["scripts"]  # [ch0_words, ch1_words]

        # Build timeline from segment dicts.
        # If segments have "channel", use it directly. Otherwise (e.g. fisher
        # rough_timeline = mixed-channel speech regions), derive per-channel
        # utterance bounds from scripts word timestamps instead.
        if raw_timeline and "channel" in raw_timeline[0]:
            for seg in raw_timeline:
                ch = seg["channel"]
                timeline[ch].append(UtteranceBound(start=seg["start_sec"], end=seg["end_sec"]))
        else:
            # Derive per-channel timeline from scripts (word timestamps).
            rt_meta = record.get("rough_timeline_meta") or {}
            gap_threshold = rt_meta.get("gap_seconds", 1.5)
            for ch in range(min(num_channels, len(raw_scripts))):
                if not raw_scripts[ch]:
                    continue
                seg_start = raw_scripts[ch][0]["start"]
                seg_end = raw_scripts[ch][0]["end"]
                for w in raw_scripts[ch][1:]:
                    if w["start"] - seg_end > gap_threshold:
                        timeline[ch].append(UtteranceBound(start=seg_start, end=seg_end))
                        seg_start = w["start"]
                    seg_end = max(seg_end, w["end"])
                timeline[ch].append(UtteranceBound(start=seg_start, end=seg_end))

        # Build script from per-channel word lists (times already in seconds)
        for ch in range(min(num_channels, len(raw_scripts))):
            for w in raw_scripts[ch]:
                script[ch].append(SpeakerItem(word=w["word"], start=w["start"], end=w["end"]))
    else:
        raise KeyError(
            f"timeline_turns_to_metadata: record must have 'turns' or 'scripts' key. Got keys: {sorted(record.keys())}"
        )

    for ch in range(num_channels):
        script[ch].sort(key=lambda item: item.start)
        timeline[ch].sort(key=lambda item: item.start)

    # Resolve system_prompt: explicit field takes priority, then build from persona/context/name.
    from .duplex_prompt_catalog import build_system_prompt

    system_prompt = record.get("system_prompt")
    if system_prompt is None:
        system_prompt = build_system_prompt(record=record)

    return DuplexMetadata(
        script=script,
        speak_first=[bool(x) for x in record["speak_first"]],
        include_in_training=[bool(x) for x in record["include_in_training"]],
        channel=record["channel"],
        language=record["language"],
        timeline=timeline,
        system_prompt=system_prompt,
    )


# ---------------------------------------------------------------------------
# Audio fixup
# ---------------------------------------------------------------------------


def fix_duplex_input_sequences(
    audio: torch.Tensor,
    text_data: list[list[SpeakerItem]],
    sampling_rate: float,
) -> None:
    assert audio.ndim == 2, f"fix_duplex_input_sequences: Expected 2D audio but got `{audio.ndim=}`."
    assert audio.shape[0] == len(text_data), (
        f"fix_duplex_input_sequences: Audio batch size does not match text data length. "
        f"Got `{audio.shape[0]=}` and `{len(text_data)=}`."
    )
    for speaker_text_data in text_data:
        for i in range(len(speaker_text_data) - 1):
            if speaker_text_data[i + 1].start < speaker_text_data[i].end:
                speaker_text_data[i].end = speaker_text_data[i + 1].start
                print(
                    "WARNING fix_duplex_input_sequences: Overlapping segments detected. "
                    f"Got `end={speaker_text_data[i].end}` and `next_start={speaker_text_data[i + 1].start}`."
                )

        for item in speaker_text_data:
            if item.end * sampling_rate > audio.shape[1]:
                item.end = audio.shape[1] / sampling_rate
                if item.start > item.end:
                    item.start = item.end
                print("WARNING fix_duplex_input_sequences: Speaker timestamp exceeds audio length.")

        for item in speaker_text_data:
            assert item.start <= item.end + 1e-6, (
                f"fix_duplex_input_sequences: Invalid timing with `{item.start=}` > `{item.end=}`."
            )


# ---------------------------------------------------------------------------
# Speaker reference audio sampling
# ---------------------------------------------------------------------------


def sample_speaker_reference_audio(
    channel_audio: torch.Tensor,
    utterance_bounds: list[UtteranceBound] | None,
    sampling_rate: float,
) -> tuple[torch.Tensor, int]:
    """Pick a speaker-reference waveform from non-silence timeline spans.

    Preference:
    1) longest available non-silence span
    2) full channel audio (fallback)
    """
    if channel_audio.ndim != 1:
        raise ValueError(f"sample_speaker_reference_audio: expected 1D audio, got {channel_audio.ndim=}")

    num_samples = int(channel_audio.shape[0])
    if num_samples <= 0:
        return channel_audio, 0

    if utterance_bounds:
        spans: list[tuple[int, int]] = []
        for bound in utterance_bounds:
            start = max(0, int(math.floor(bound.start * sampling_rate)))
            end = min(num_samples, int(math.ceil(bound.end * sampling_rate)))
            if end > start:
                spans.append((start, end))

        if len(spans) > 0:
            start, end = max(spans, key=lambda x: x[1] - x[0])
            ref = channel_audio[start:end]
            if ref.numel() > 0:
                return ref, int(ref.numel())

    return channel_audio, num_samples


# ---------------------------------------------------------------------------
# SIL no-audio sequence builder
# ---------------------------------------------------------------------------


def _compute_text_segments(
    word_frames: list[tuple[str, int]],
    num_audio_frames: int,
    processor: Any,
    use_duplex_end_pad: bool,
) -> tuple[list[list[int]], list[int], list[bool], list[int]]:
    """Compute text tokenization and segment layout from word-frame pairs.

    Returns:
        text_ids: Per-word token ID lists.
        audio_frame_start_indices: Frame index where each word starts.
        is_consecutive: Whether each word immediately follows the previous.
        segments: Number of frames allocated to each word segment.
    """
    text_ids: list[list[int]] = [processor.tokenizer.encode(f" {word.strip()}") for word, _ in word_frames]
    audio_frame_start_indices = [start_frame for _, start_frame in word_frames]

    is_consecutive: list[bool] = [audio_frame_start_indices[0] == 0] if word_frames else []
    segments: list[int] = []
    prev_end = 0

    for i in range(len(audio_frame_start_indices) - 1):
        start = max(prev_end, audio_frame_start_indices[i])
        tw_end = start + len(text_ids[i])
        is_consecutive.append(audio_frame_start_indices[i + 1] == tw_end)
        needs_epad = use_duplex_end_pad and not is_consecutive[-1]
        min_frames = len(text_ids[i]) + (1 if needs_epad else 0)
        end = max(start + min_frames, audio_frame_start_indices[i + 1])
        segments.append(end - start)
        prev_end = end

    if word_frames:
        start = max(prev_end, audio_frame_start_indices[-1])
        last_seg = num_audio_frames - start
        if last_seg > 0:
            segments.append(last_seg)
            if len(text_ids[-1]) > last_seg:
                text_ids[-1] = text_ids[-1][:last_seg]
        else:
            text_ids.pop()
            is_consecutive.pop()

    return text_ids, audio_frame_start_indices, is_consecutive, segments


def _build_no_audio_in_sil_sequence(
    word_frames: list[tuple[str, int]],
    num_audio_frames: int,
    processor: Any,
    utt_frames: list[tuple[int, int]],
    loss_ignore_index: int = -100,
) -> tuple[list[int], list[int], int]:
    """Build duplex sequence with SIL no-audio.

    State machine:
        SIL  → SIL, EPAD
        EPAD → TEXT (always)
        TEXT → TEXT, EPAD, PAD, SIL
        PAD  → PAD, SIL, EPAD

    Frame structures:
        SIL:   [U]              → no audio
        Onset: [U] EPAD AUD_S   → EPAD predicts TEXT (always)
        TEXT:  [U] T [A]        → audio predicted
        EPAD:  [U] EPAD [A]     → EPAD predicts TEXT, audio predicted
        PAD:   [U] [A]          → audio predicted
        When SIL predicted from TEXT/PAD: [A] label = ignore (-100)

    Only supports UTA + EPAD mode.
    """
    use_duplex_end_pad = getattr(processor, "use_duplex_end_pad", False)
    duplex_end_pad_token_id = getattr(processor, "duplex_end_pad_token_id", processor.eos_token_id)
    duplex_pad_token_id = getattr(processor, "duplex_pad_token_id", processor.eos_token_id)
    duplex_sil_token_id = getattr(processor, "duplex_sil_token_id", processor.eos_token_id)
    speaker_token_id = getattr(processor, "speaker_token_id", None)

    # text_lookahead is handled by the caller shifting frame indices before calling this function.

    assert num_audio_frames > 0

    # Use shared helper for segment computation
    text_ids, audio_frame_start_indices, is_consecutive, segments = _compute_text_segments(
        word_frames=word_frames,
        num_audio_frames=num_audio_frames,
        processor=processor,
        use_duplex_end_pad=use_duplex_end_pad,
    )
    # Sync word_frames with text_ids after possible truncation in _compute_text_segments
    word_frames = word_frames[: len(text_ids)]

    # Build a per-frame map: frame_idx → (text_token_ids, is_epad_before)
    frame_text: dict[int, list[int]] = {}  # frame_idx → text token ids
    frame_epad_before: set[int] = set()  # frames that need EPAD before them

    if word_frames:
        # Map text tokens to absolute frame indices and compute EPAD frame positions.
        # EPAD replaces the last PAD before non-consecutive text (no extra frames added).
        frame_is_epad: set[int] = set()  # frames that are EPAD (replace PAD)
        frame_offset = audio_frame_start_indices[0] if word_frames else 0
        for wi, (wids, seg_len) in enumerate(zip(text_ids, segments)):
            has_next = wi + 1 < len(text_ids)
            needs_epad = use_duplex_end_pad and has_next and not is_consecutive[wi + 1]
            num_pad_frames = seg_len - len(wids) - (1 if needs_epad else 0)

            for ti, tok in enumerate(wids):
                abs_frame = frame_offset + ti
                if abs_frame not in frame_text:
                    frame_text[abs_frame] = []
                frame_text[abs_frame].append(tok)

            # Mark EPAD for first word if non-consecutive
            if wi == 0 and use_duplex_end_pad and not is_consecutive[0]:
                frame_epad_before.add(audio_frame_start_indices[0])

            # Mark EPAD frame position and EPAD-before marker for next word
            if needs_epad:
                epad_frame = frame_offset + len(wids) + num_pad_frames
                frame_is_epad.add(epad_frame)
                next_start = audio_frame_start_indices[wi + 1]
                frame_epad_before.add(next_start)

            frame_offset += seg_len
    else:
        frame_is_epad: set[int] = set()

    # Build speech region mask: a frame is "speech" only if it's within an utterance
    # AND there are text frames in its contiguous utterance region.
    # Frames within utterance bounds but before any text → SIL (onset fires at text).
    # This ensures EPAD → TEXT (always).
    #
    # Algorithm: for each utterance region, find first and last text frame.
    # Only frames between first_text and last_text+padding are speech. Rest are SIL.
    sil_frame_mask: list[bool] = [True] * num_audio_frames  # default: all SIL

    if frame_text:
        # For each utterance bound, find text frames within it
        for ub_start_frame, ub_end_frame in utt_frames:
            ub_end_frame = min(ub_end_frame, num_audio_frames)

            # Find first text frame in this utterance
            first_text_frame = None
            last_text_frame = None
            for f in range(ub_start_frame, ub_end_frame):
                if f in frame_text:
                    if first_text_frame is None:
                        first_text_frame = f
                    last_text_frame = f

            if first_text_frame is not None:
                # Speech region: from onset frame (first_text - 1) to end of utterance.
                # Onset frame is 1 frame before first text, so EPAD is predicted at -2.
                # Clamp onset to ub_start_frame so it never extends into the inter-utterance
                # gap (which is not covered by audio_output_segments), preventing a 1-frame
                # available < required mismatch when adjacent utterances are 2+ frames apart.
                onset_frame = max(ub_start_frame, max(0, first_text_frame - 1))
                for f in range(onset_frame, ub_end_frame):
                    if f < num_audio_frames:
                        sil_frame_mask[f] = False

    sil_frame_count = sum(sil_frame_mask)

    input_ids: list[int] = []
    shifted_labels: list[int] = []
    in_speech = False  # Currently in audio-producing region

    # Prefill: [SPK, IM_S]
    input_ids.extend([speaker_token_id, processor.im_start_token_id])
    # IM_S label: predicts first frame's text label
    if sil_frame_mask[0]:
        shifted_labels.extend([loss_ignore_index, duplex_sil_token_id])
    else:
        # First frame is speech — need EPAD onset
        shifted_labels.extend([loss_ignore_index, duplex_end_pad_token_id])

    # Compute per-frame text labels for UTA mode
    def get_frame_text_label(frame_idx: int) -> int:
        """Get the text prediction label for a frame (what [U] predicts in UTA mode)."""
        if frame_idx in frame_is_epad:
            return duplex_end_pad_token_id
        if frame_idx in frame_text:
            return frame_text[frame_idx][0]
        if frame_idx in frame_epad_before:
            return duplex_end_pad_token_id
        if sil_frame_mask[frame_idx] if frame_idx < num_audio_frames else True:
            return duplex_sil_token_id
        return duplex_pad_token_id

    def get_next_text_token_for_word_frame(frame_idx: int, token_offset: int) -> int:
        """Get the next text token prediction for a text frame."""
        tokens = frame_text.get(frame_idx, [])
        if token_offset + 1 < len(tokens):
            return tokens[token_offset + 1]
        # Look for next word's first token
        for nf in range(frame_idx + 1, num_audio_frames):
            if nf in frame_text:
                return frame_text[nf][0]
            if nf in frame_epad_before:
                return duplex_end_pad_token_id
            if not sil_frame_mask[nf]:
                return duplex_pad_token_id
            else:
                return duplex_sil_token_id
        return duplex_pad_token_id

    frame_idx = 0
    while frame_idx < num_audio_frames:
        is_sil = sil_frame_mask[frame_idx]

        if is_sil and not in_speech:
            # === SIL frame (not in speech) ===
            # Determine label: SIL, or EPAD if next non-SIL is coming
            # Look ahead for transition
            next_non_sil = None
            for nf in range(frame_idx + 1, num_audio_frames):
                if not sil_frame_mask[nf]:
                    next_non_sil = nf
                    break

            if next_non_sil is not None and next_non_sil == frame_idx + 1:
                # Next frame starts speech → label = EPAD
                label = duplex_end_pad_token_id
            else:
                label = duplex_sil_token_id

            input_ids.append(processor.audio_input_token_id)
            shifted_labels.append(label)
            frame_idx += 1

        elif not is_sil and not in_speech:
            # === Onset frame (separate frame, 1 frame before first speech text) ===
            in_speech = True
            # Find the first text token in this speech region (at frame_idx or later).
            first_label = None
            for ff in range(frame_idx, num_audio_frames):
                if sil_frame_mask[ff]:
                    break
                if ff in frame_text:
                    first_label = frame_text[ff][0]
                    break
            assert first_label is not None, (
                f"Onset at frame {frame_idx} but no text found in speech region. "
                f"sil_frame_mask should ensure onset only fires before text."
            )

            # [U] EPAD AUD_S → labels: -, first_label, [A]
            # Onset is its own frame — the next frame handles speech tokens.
            input_ids.extend(
                [
                    processor.audio_input_token_id,
                    duplex_end_pad_token_id,
                    processor.audio_start_token_id,
                ]
            )
            shifted_labels.extend(
                [
                    loss_ignore_index,
                    first_label,
                    processor.audio_output_token_id,
                ]
            )
            # Onset already provides EPAD for the first text word in this speech region.
            # Remove it from frame_epad_before to prevent double EPAD.
            for ff in range(frame_idx, num_audio_frames):
                if sil_frame_mask[ff]:
                    break
                if ff in frame_epad_before:
                    frame_epad_before.discard(ff)
                    break
            frame_idx += 1

        elif not is_sil and in_speech:
            # === Speech frame (continuing) ===
            if frame_idx in frame_is_epad:
                # EPAD frame: replaces PAD, [U] EPAD [A]. EPAD predicts TEXT.
                first_text = None
                for ff in range(frame_idx + 1, num_audio_frames):
                    if sil_frame_mask[ff]:
                        break
                    if ff in frame_text:
                        first_text = frame_text[ff][0]
                        break
                if first_text is None:
                    first_text = duplex_pad_token_id
                input_ids.extend([processor.audio_input_token_id, duplex_end_pad_token_id, processor.audio_output_token_id])
                shifted_labels.extend([loss_ignore_index, first_text, processor.audio_output_token_id])
                frame_idx += 1
            elif frame_idx in frame_text:
                tokens = frame_text[frame_idx]
                for ti, tok in enumerate(tokens):
                    next_tok = get_next_text_token_for_word_frame(frame_idx, ti)
                    input_ids.extend([processor.audio_input_token_id, tok, processor.audio_output_token_id])
                    shifted_labels.extend([loss_ignore_index, next_tok, processor.audio_output_token_id])
                frame_idx += 1
            else:
                # PAD frame: [U] predicts PAD/SIL/EPAD.
                if frame_idx + 1 < num_audio_frames and sil_frame_mask[frame_idx + 1]:
                    next_label = duplex_sil_token_id
                else:
                    next_label = (
                        get_frame_text_label(frame_idx + 1) if frame_idx + 1 < num_audio_frames else duplex_sil_token_id
                    )

                input_ids.extend([processor.audio_input_token_id, processor.audio_output_token_id])
                shifted_labels.extend([next_label, processor.audio_output_token_id])
                frame_idx += 1

        elif is_sil and in_speech:
            # === Speech → SIL offset ===
            in_speech = False
            # When SIL is predicted, the [A] in the same frame is ignored (no audio loss).
            if shifted_labels and shifted_labels[-1] == processor.audio_output_token_id:
                shifted_labels[-1] = loss_ignore_index
            # Don't increment frame_idx — this SIL frame will be processed in next iteration

    # If we ended in speech mode, last [A] is ignored
    if in_speech:
        if shifted_labels and shifted_labels[-1] == processor.audio_output_token_id:
            shifted_labels[-1] = loss_ignore_index

    # Build unshifted labels
    assert len(input_ids) == len(shifted_labels), (
        f"_build_no_audio_in_sil_sequence: length mismatch: {len(input_ids)=} != {len(shifted_labels)=}"
    )
    unshifted_labels = [loss_ignore_index] + shifted_labels[:-1]

    return input_ids, unshifted_labels, sil_frame_count


# ---------------------------------------------------------------------------
# Core sequence builder
# ---------------------------------------------------------------------------


def build_duplex_sequence_input_ids_and_labels(
    text_data: list[SpeakerItem],
    audio_length: int,
    processor: Any,
    utterance_bounds: list[UtteranceBound] | None = None,
    loss_ignore_index: int = -100,
    *,
    word_frames: list[tuple[str, int]] | None = None,
    num_audio_frames: int | None = None,
    utt_frames: list[tuple[int, int]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Build input_ids and labels for duplex sequence.

    Supports two sequence modes (controlled by processor.sequence_mode):
    - "tua" (default): T [U] [A] — text predicted at [U] position
    - "uta": [U] T [A] — text predicted at T position

    Labels (shifted_labels) are identical in both modes; only input_ids order differs.

    With EPAD (use_duplex_end_pad=True), inserts <epad> before non-consecutive speech onsets.
    When utterance_bounds are provided and use_sil_token=True:
    - Labels only: pure-silence [U] labels outside utterance spans are SIL instead of PAD.
    - SIL no-audio: [A] tokens are completely removed from SIL frames, so no audio
      prediction is made for silence outside utterances.

    Returns:
        (input_ids, labels, sil_frame_count): sil_frame_count is the number of frames
        where [A] was removed (0 when use_sil_token=False).
    """
    # Get configuration from processor
    use_duplex_end_pad = getattr(processor, "use_duplex_end_pad", False)
    use_sil_token = getattr(processor, "use_sil_token", False)
    duplex_end_pad_token_id = getattr(processor, "duplex_end_pad_token_id", processor.eos_token_id)
    duplex_pad_token_id = getattr(processor, "duplex_pad_token_id", processor.eos_token_id)
    duplex_sil_token_id = getattr(processor, "duplex_sil_token_id", processor.eos_token_id)
    sequence_mode = getattr(processor, "sequence_mode", "tua")
    is_uta = sequence_mode == "uta"
    if sequence_mode == "uta-s":
        raise NotImplementedError("uta-s sequence mode is not yet implemented")
    if is_uta:
        assert use_duplex_end_pad, (
            "build_duplex_sequence_input_ids_and_labels: UTA sequence mode requires use_duplex_end_pad=True"
        )
    speaker_token_id = getattr(processor, "speaker_token_id", None)
    assert speaker_token_id is not None, (
        "build_duplex_sequence_input_ids_and_labels: processor.speaker_token_id is required "
        "for speaker prefill token insertion."
    )

    if num_audio_frames is None:
        num_audio_frames = math.ceil(audio_length * processor.frame_rate / processor.sampling_rate)
    assert num_audio_frames > 0, (
        f"build_duplex_sequence_input_ids_and_labels: Audio must have at least 1 frame. "
        f"Got `{num_audio_frames=}` from `{audio_length=}`, `{processor.frame_rate=}`, `{processor.sampling_rate=}`."
    )

    # SIL no-audio: delegate to dedicated builder with AUD_S/AUD_E boundaries.
    # When processor.no_audio_in_sil=False ("SIL audio" mode), skip early-return so
    # the "labels only" post-processing path handles SIL labeling
    # while keeping [A] tokens for continuous audio prediction.
    no_audio_in_sil = (
        use_sil_token
        and getattr(processor, "no_audio_in_sil", False)
        and (utt_frames is not None or utterance_bounds is not None)
    )
    if no_audio_in_sil:
        if word_frames is None:
            word_frames = [(item.word, math.floor(item.start * processor.frame_rate)) for item in text_data]
        if utt_frames is None:
            utt_frames = [
                (math.floor(ub.start * processor.frame_rate), math.ceil(ub.end * processor.frame_rate))
                for ub in utterance_bounds  # type: ignore[union-attr]
            ]
        ids, labels, sil_count = _build_no_audio_in_sil_sequence(
            word_frames=word_frames,
            num_audio_frames=num_audio_frames,
            processor=processor,
            utt_frames=utt_frames,
            loss_ignore_index=loss_ignore_index,
        )
        return torch.tensor(ids), torch.tensor(labels), sil_count

    # Pre-compute SIL frame mask (not used when no_audio_in_sil delegates above).
    sil_frame_mask: list[bool] = [False] * num_audio_frames
    sil_frame_count = 0

    if len(text_data) == 0:
        input_ids = [speaker_token_id, processor.im_start_token_id, processor.audio_start_token_id]
        shifted_labels: list[int] = [loss_ignore_index]
        for f in range(num_audio_frames):
            if no_audio_in_sil and sil_frame_mask[f]:
                # SIL frame: [U] only, no [A]
                input_ids.append(processor.audio_input_token_id)
                shifted_labels.append(duplex_sil_token_id)
            else:
                input_ids.extend([processor.audio_input_token_id, processor.audio_output_token_id])
                shifted_labels.extend([duplex_pad_token_id, processor.audio_output_token_id])
        # Trailing ignore labels to match input_ids length.
        while len(shifted_labels) < len(input_ids):
            shifted_labels.append(loss_ignore_index)
        unshifted_labels = [loss_ignore_index] + shifted_labels[:-1]
        return torch.tensor(input_ids), torch.tensor(unshifted_labels), sil_frame_count

    # Use shared helper for segment computation
    if word_frames is None:
        word_frames = [(item.word, math.floor(item.start * processor.frame_rate)) for item in text_data]
    text_ids, audio_frame_start_indices, is_consecutive, audio_segment_lengths = _compute_text_segments(
        word_frames=word_frames,
        num_audio_frames=num_audio_frames,
        processor=processor,
        use_duplex_end_pad=use_duplex_end_pad,
    )
    # Sync text_data with text_ids after possible truncation
    text_data = text_data[: len(text_ids)]

    assert sum(audio_segment_lengths) + audio_frame_start_indices[0] == num_audio_frames, (
        f"build_duplex_sequence_input_ids_and_labels: Audio segments don't sum correctly. "
        f"Got `{sum(audio_segment_lengths)=}`, `{audio_frame_start_indices[0]=}`, `{num_audio_frames=}`."
    )

    for word_ids, audio_segment_length, item in zip(text_ids, audio_segment_lengths, text_data, strict=True):
        assert len(word_ids) <= audio_segment_length, (
            f"build_duplex_sequence_input_ids_and_labels: Word '{item.word}' has {len(word_ids)} tokens "
            f"but only {audio_segment_length} audio frames available."
        )

    num_pre_text_frames = audio_frame_start_indices[0]

    # Initialize input_ids with speaker prefill, im_start, and audio_start tokens.
    if is_uta and num_pre_text_frames == 0:
        # UTA speak-first: insert EPAD in prefill to predict first text token.
        # [SPK, IM_S, EPAD, AUD_S] → labels: [-, -, T(1), audio]
        input_ids = [
            speaker_token_id,
            processor.im_start_token_id,
            duplex_end_pad_token_id,
            processor.audio_start_token_id,
        ]
        shifted_labels = [
            loss_ignore_index,
            loss_ignore_index,
            text_ids[0][0],
            processor.audio_output_token_id,
        ]
    else:
        # TUA mode, or listen-first (both modes).
        # [SPK, IM_S, AUD_S] → labels: [-, first_u_label, audio]
        input_ids = [speaker_token_id, processor.im_start_token_id, processor.audio_start_token_id]
        if num_pre_text_frames > 0:
            if use_duplex_end_pad and num_pre_text_frames == 1:
                first_u_label = duplex_end_pad_token_id
            else:
                first_u_label = duplex_pad_token_id
        else:
            first_u_label = text_ids[0][0]
        shifted_labels = [loss_ignore_index, first_u_label, processor.audio_output_token_id]

    # Text is predicted at [U] positions:
    # text [U] [A] -> labels: -, next_text, audio.
    #
    # Handle pre-text silence frames.
    if num_pre_text_frames > 0:
        if use_duplex_end_pad and not is_consecutive[0]:
            # With EPAD: last silence frame is replaced by EPAD frame.
            # Total frames = num_pre_text_frames (not num_pre_text_frames + 1).
            for frame_idx in range(num_pre_text_frames - 1):
                input_ids.extend([processor.audio_input_token_id, processor.audio_output_token_id])
                is_second_last = frame_idx == num_pre_text_frames - 2
                if is_second_last:
                    shifted_labels.extend([duplex_end_pad_token_id, processor.audio_output_token_id])
                else:
                    shifted_labels.extend([duplex_pad_token_id, processor.audio_output_token_id])

            # Last frame is EPAD frame (replaces last silence).
            # TUA: EPAD [U] [A] | UTA: [U] EPAD [A] → labels: -, first_text, audio.
            if is_uta:
                input_ids.extend([processor.audio_input_token_id, duplex_end_pad_token_id, processor.audio_output_token_id])
            else:
                input_ids.extend([duplex_end_pad_token_id, processor.audio_input_token_id, processor.audio_output_token_id])
            shifted_labels.extend([loss_ignore_index, text_ids[0][0], processor.audio_output_token_id])
        else:
            # Without EPAD: silence frames predict pad, last one predicts first text.
            for frame_idx in range(num_pre_text_frames - 1):
                input_ids.extend([processor.audio_input_token_id, processor.audio_output_token_id])
                shifted_labels.extend([duplex_pad_token_id, processor.audio_output_token_id])
            input_ids.extend([processor.audio_input_token_id, processor.audio_output_token_id])
            shifted_labels.extend([text_ids[0][0], processor.audio_output_token_id])

    for word_idx, (word_ids, audio_segment_length) in enumerate(zip(text_ids, audio_segment_lengths, strict=True)):
        has_next_word = word_idx + 1 < len(text_ids)
        next_word_first_token = text_ids[word_idx + 1][0] if has_next_word else duplex_pad_token_id
        next_needs_epad = use_duplex_end_pad and has_next_word and not is_consecutive[word_idx + 1]
        num_pad_frames = audio_segment_length - len(word_ids) - (1 if next_needs_epad else 0)

        for i in range(len(word_ids)):
            text_token = word_ids[i]

            if i + 1 < len(word_ids):
                next_text_token = word_ids[i + 1]
            elif next_needs_epad and num_pad_frames == 0:
                next_text_token = duplex_end_pad_token_id
            elif num_pad_frames > 0:
                next_text_token = duplex_pad_token_id
            else:
                next_text_token = next_word_first_token

            # TUA: T [U] [A] | UTA: [U] T [A] → labels: -, next_text, audio.
            if is_uta:
                input_ids.extend([processor.audio_input_token_id, text_token, processor.audio_output_token_id])
            else:
                input_ids.extend([text_token, processor.audio_input_token_id, processor.audio_output_token_id])
            shifted_labels.extend([loss_ignore_index, next_text_token, processor.audio_output_token_id])

        # Handle silence frames between words.
        if next_needs_epad:
            # EPAD as separate frame after silence:
            # [pad frames x num_pad_frames] + [EPAD frame].
            for frame_idx in range(num_pad_frames):
                input_ids.extend([processor.audio_input_token_id, processor.audio_output_token_id])
                is_last = frame_idx == num_pad_frames - 1
                if is_last:
                    shifted_labels.extend([duplex_end_pad_token_id, processor.audio_output_token_id])
                else:
                    shifted_labels.extend([duplex_pad_token_id, processor.audio_output_token_id])

            # TUA: EPAD [U] [A] | UTA: [U] EPAD [A] → labels: -, next_first_text, audio.
            if is_uta:
                input_ids.extend([processor.audio_input_token_id, duplex_end_pad_token_id, processor.audio_output_token_id])
            else:
                input_ids.extend([duplex_end_pad_token_id, processor.audio_input_token_id, processor.audio_output_token_id])
            shifted_labels.extend([loss_ignore_index, next_word_first_token, processor.audio_output_token_id])
        elif num_pad_frames > 0:
            for frame_idx in range(num_pad_frames):
                input_ids.extend([processor.audio_input_token_id, processor.audio_output_token_id])
                is_last_silence = frame_idx == num_pad_frames - 1
                if is_last_silence:
                    user_label = next_word_first_token
                else:
                    user_label = duplex_pad_token_id
                shifted_labels.extend([user_label, processor.audio_output_token_id])

    unshifted_labels = [loss_ignore_index] + shifted_labels[:-1]

    # Post-processing: replace PAD with SIL for silence frames outside utterance bounds.
    # Only applies to pure silence frames ([U] [A]), not text/EPAD frames.
    # Resolve utt_frames for SIL post-processing (use pre-computed if available).
    _sil_utt_frames = utt_frames
    if _sil_utt_frames is None and utterance_bounds is not None:
        _sil_utt_frames = [
            (math.floor(ub.start * processor.frame_rate), math.ceil(ub.end * processor.frame_rate))
            for ub in utterance_bounds
        ]
    if _sil_utt_frames is not None and use_sil_token:
        # Also handle IM_S prefill label (shifted_labels[1]) — it predicts the first [U] label
        # but SIL post-processing below only iterates [U] positions, missing IM_S.
        if shifted_labels[1] == duplex_pad_token_id:
            if not any(s <= 0 < e for s, e in _sil_utt_frames):
                shifted_labels[1] = duplex_sil_token_id

        frame_idx = 0
        for pos in range(len(shifted_labels)):
            token_id = input_ids[pos]
            if token_id == processor.audio_input_token_id:
                # Detect pure silence frames ([U] [A]) vs text/EPAD frames.
                if is_uta:
                    # UTA: silence = [U] immediately followed by [A].
                    # Text frames are [U] T [A], so next token is NOT [A].
                    next_token = input_ids[pos + 1] if pos + 1 < len(input_ids) else -1
                    is_silence_frame = next_token == processor.audio_output_token_id
                else:
                    # TUA: silence = [U] immediately after [A] or <audio_start>.
                    # Text frames are T [U] [A], so prev token of [U] is T, not [A].
                    prev_token = input_ids[pos - 1] if pos > 0 else -1
                    is_silence_frame = prev_token in (processor.audio_output_token_id, processor.audio_start_token_id)
                if is_silence_frame:
                    in_utterance = any(s <= frame_idx < e for s, e in _sil_utt_frames)
                    if not in_utterance and shifted_labels[pos] == duplex_pad_token_id:
                        shifted_labels[pos] = duplex_sil_token_id
            if token_id == processor.audio_output_token_id:
                frame_idx += 1
        unshifted_labels = [loss_ignore_index] + shifted_labels[:-1]

    # SIL no-audio: remove [A] tokens from SIL frames so no audio prediction is made.
    # Uses shifted_labels (set by SIL post-processing above) to identify actual SIL silence frames.
    # This correctly excludes EPAD and text frames that happen to fall outside utterance bounds.
    if no_audio_in_sil:
        new_input_ids: list[int] = []
        new_unshifted: list[int] = []
        actual_sil_removals = 0
        for i in range(len(input_ids)):
            token = input_ids[i]
            if token == processor.audio_output_token_id:
                # Check if the [U] right before this [A] has a SIL label in shifted_labels.
                u_pos = i - 1
                is_sil_labeled = (
                    u_pos >= 0
                    and input_ids[u_pos] == processor.audio_input_token_id
                    and shifted_labels[u_pos] == duplex_sil_token_id
                )
                if is_sil_labeled:
                    # Skip [A] token; fix [U]'s label from A_out (audio) to SIL (text).
                    if new_unshifted:
                        new_unshifted[-1] = duplex_sil_token_id
                    actual_sil_removals += 1
                    continue
            new_input_ids.append(input_ids[i])
            new_unshifted.append(unshifted_labels[i])
        input_ids = new_input_ids
        unshifted_labels = new_unshifted
        sil_frame_count = actual_sil_removals

    return torch.tensor(input_ids), torch.tensor(unshifted_labels), sil_frame_count


# ---------------------------------------------------------------------------
# High-level sequence builder (stereo audio separation + system prompt)
# ---------------------------------------------------------------------------


def build_duplex_input_sequences(
    audio: torch.Tensor,
    metadata: DuplexMetadata,
    processor: Any,
    loss_ignore_index: int = -100,
) -> list[DuplexInputs]:
    """Build per-channel DuplexInputs from stereo audio and metadata."""
    text_data = metadata.script
    timeline_data = metadata.timeline
    fix_duplex_input_sequences(audio=audio, text_data=text_data, sampling_rate=processor.sampling_rate)
    other_audio = audio.sum(dim=0, keepdim=True) - audio
    active_mask = (audio != 0).to(audio.dtype)
    audio_input = other_audio / (active_mask.sum(dim=0, keepdim=True) - active_mask).clamp_min(1)

    # Convert timestamps to frame domain once. All subsequent arithmetic is integer-based,
    # eliminating floating-point rounding errors (especially with text_lookahead shifts).
    frame_rate = processor.frame_rate
    all_word_frames: list[list[tuple[str, int]]] = [
        [
            (item.word, math.floor(item.start * frame_rate))
            for item in speaker_items
            if item.start < item.end  # exclude zero-duration words clipped by fix_duplex_input_sequences
        ]
        for speaker_items in text_data
    ]
    all_utt_frames: list[list[tuple[int, int]]] | None = None
    if timeline_data is not None:
        all_utt_frames = [
            [(math.floor(ub.start * frame_rate), math.ceil(ub.end * frame_rate)) for ub in channel_bounds]
            for channel_bounds in timeline_data
        ]
    # Clamp word/utterance frame indices to valid range [0, base_num_audio_frames].
    max_frames = math.ceil(audio.shape[1] * frame_rate / processor.sampling_rate)
    for ch_words in all_word_frames:
        ch_words[:] = [(w, min(s, max_frames - 1)) for w, s in ch_words]
    if all_utt_frames is not None:
        for ch_utts in all_utt_frames:
            ch_utts[:] = [(min(s, max_frames), min(e, max_frames)) for s, e in ch_utts]
        # Merge adjacent utterance bounds whose gap is <= text_lookahead frames.
        # Data may contain utterances that should have been merged (gap <= 1.5s) but
        # weren't due to preprocessing issues; small frame-domain gaps cause onset
        # frames to fall outside any audio segment, leading to [A]/audio mismatches.
        merge_gap = getattr(processor, "text_lookahead", 0) + 1  # frames
        merged_all_utt_frames: list[list[tuple[int, int]]] = []
        for channel_utt_frames in all_utt_frames:
            merged: list[tuple[int, int]] = []
            for s, e in channel_utt_frames:
                if merged and s - merged[-1][1] <= merge_gap:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], e))
                else:
                    merged.append((s, e))
            merged_all_utt_frames.append(merged)
        all_utt_frames = merged_all_utt_frames
    base_num_audio_frames = math.ceil(audio.shape[1] * frame_rate / processor.sampling_rate)

    system_prefix_ids_list: list[torch.Tensor] = []
    for i, speak_first in enumerate(metadata.speak_first):
        # Use per-sample system_prompt if available.
        if metadata.system_prompt is not None:
            system_message = metadata.system_prompt
        else:
            key = get_duplex_system_message_key(metadata.language, metadata.channel, speak_first)
            system_message = CHANNEL_DUPLEX_TO_SYSTEM_MESSAGE.get(key)
        if system_message is None:
            system_message = "null"
            assert not metadata.include_in_training[i], (
                f"build_duplex_input_sequences: "
                f"Sample must not be included in training when no system message is found. "
                f"Got `{key=}` and `{metadata.include_in_training[i]=}`."
            )

        messages = [{"role": "system", "content": system_message}]
        system_text = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        assert isinstance(system_text, str), f"build_duplex_input_sequences: Expected `str` but got `{type(system_text)=}`."
        system_ids = processor.tokenizer.encode(system_text)
        system_prefix_ids_list.append(torch.tensor(system_ids))

    input_ids_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    valid_indices: list[int] = []
    slot_utt_frames_per_valid: list[list[tuple[int, int]] | None] = []

    text_lookahead = getattr(processor, "text_lookahead", 0)

    for i in range(len(text_data)):
        slot_utt_frames = list(all_utt_frames[i]) if all_utt_frames is not None else None
        slot_word_frames = list(all_word_frames[i])
        slot_num_audio_frames = base_num_audio_frames

        # Unified text_lookahead: pad if first word is too close to start, then shift.
        # No speak_first branching — the natural leading silence is preserved.
        if text_lookahead > 0 and slot_word_frames is not None and len(slot_word_frames) > 0:
            first_word_start = min(s for _, s in slot_word_frames)
            pad_frames = max(0, text_lookahead + 1 - first_word_start)

            if pad_frames > 0:
                slot_word_frames = [(w, s + pad_frames) for w, s in slot_word_frames]
                if slot_utt_frames is not None:
                    slot_utt_frames = [(s + pad_frames, e + pad_frames) for s, e in slot_utt_frames]
                slot_num_audio_frames += pad_frames

            # Apply text_lookahead: shift word/utt starts earlier (text leads audio).
            slot_word_frames = [(w, max(0, s - text_lookahead)) for w, s in slot_word_frames]
            if slot_utt_frames is not None:
                slot_utt_frames = [(max(0, s - text_lookahead), e) for s, e in slot_utt_frames]

        try:
            seq_input_ids, seq_labels, _ = build_duplex_sequence_input_ids_and_labels(
                text_data=text_data[i],
                audio_length=audio.shape[1],
                processor=processor,
                loss_ignore_index=loss_ignore_index,
                word_frames=slot_word_frames,
                num_audio_frames=slot_num_audio_frames,
                utt_frames=slot_utt_frames,
            )
        except AssertionError as e:
            print(f"build_duplex_input_sequences: Skipping sequence {i}: {e}")
            continue

        system_prefix_ids = system_prefix_ids_list[i]
        system_prefix_labels = torch.full_like(system_prefix_ids, fill_value=loss_ignore_index)

        input_ids_list.append(torch.cat([system_prefix_ids, seq_input_ids]))
        labels_list.append(torch.cat([system_prefix_labels, seq_labels]))
        valid_indices.append(i)
        slot_utt_frames_per_valid.append(slot_utt_frames)

    attention_masks = [torch.ones_like(ids) for ids in input_ids_list]

    use_sil_token = getattr(processor, "use_sil_token", False)

    outputs: list[DuplexInputs] = []
    for j, i in enumerate(valid_indices):
        if not metadata.include_in_training[i] or len(metadata.script[i]) == 0:
            continue

        utterance_bounds = timeline_data[i] if timeline_data is not None else None

        # Per-slot audio: pad at front for speak_first with text_lookahead.
        slot_audio_output = audio[i]  # [audio_len]
        slot_audio_input = audio_input[i]  # [audio_len]
        if text_lookahead > 0 and metadata.speak_first[i]:
            spf = int(processor.sampling_rate / processor.frame_rate)
            pad_samples = text_lookahead * spf
            slot_audio_output = torch.cat(
                [torch.zeros(pad_samples, dtype=audio.dtype, device=audio.device), slot_audio_output]
            )
            slot_audio_input = torch.cat(
                [torch.zeros(pad_samples, dtype=audio_input.dtype, device=audio_input.device), slot_audio_input]
            )

        speaker_ref_audio_1d, speaker_ref_len = sample_speaker_reference_audio(
            channel_audio=slot_audio_output,
            utterance_bounds=utterance_bounds,
            sampling_rate=processor.sampling_rate,
        )
        speaker_ref_lengths = torch.tensor([speaker_ref_len], dtype=torch.long, device=audio.device)

        # Compute audio_output_segments using text_lookahead-adjusted frame-domain boundaries.
        # Use slot_utt_frames (already shifted for text_lookahead) so that the segment frame
        # count matches the [A] token count in input_ids. Also clip against slot_audio_output
        # (which is padded for speak_first slots) instead of the original audio tensor.
        audio_output_segments: list[tuple[int, int]] | None = None
        orig_utt_frames = slot_utt_frames_per_valid[j]
        if orig_utt_frames is not None and use_sil_token and getattr(processor, "no_audio_in_sil", False):
            spf = int(processor.sampling_rate / processor.frame_rate)  # samples per frame
            slot_num_frames = math.ceil(slot_audio_output.shape[0] / spf)
            audio_output_segments = []
            for seg_start_frame, seg_end_frame in orig_utt_frames:
                seg_end_frame = min(seg_end_frame, slot_num_frames)
                if seg_end_frame > seg_start_frame:
                    seg_start_sample = seg_start_frame * spf
                    seg_end_sample = min(seg_end_frame * spf, slot_audio_output.shape[0])
                    audio_output_segments.append((seg_start_sample, seg_end_sample))

        # Pass full audio_output to the model. When audio_output_segments is available
        # (no_audio_in_sil mode), the model uses tokenize_audio_segments to encode
        # each speech segment independently without cross-segment conv context.
        sample_audio_output = slot_audio_output[None]
        sample_audio_output_lengths = torch.tensor(
            [slot_audio_output.shape[0]],
            dtype=torch.long,
            device=audio.device,
        )

        outputs.append(
            {
                "input_ids": input_ids_list[j][None],
                "attention_mask": attention_masks[j][None],
                "audio_input": slot_audio_input[None],
                "audio_output": sample_audio_output,
                "audio_input_lengths": torch.tensor([slot_audio_input.shape[0]], dtype=torch.long, device=audio.device),
                "audio_output_lengths": sample_audio_output_lengths,
                "audio_output_segments": audio_output_segments,
                "speaker_encoder_audio": speaker_ref_audio_1d[None],
                "speaker_encoder_audio_lengths": speaker_ref_lengths,
                "labels": labels_list[j][None],
                "sample_slot": i,
            }
        )

    # Clamp audio_output_lengths to match the actual number of [A] placeholder tokens
    # in input_ids. This ensures consistency when no_audio_in_sil removes silence frames
    # (fewer [A] tokens than total audio frames).
    for output in outputs:
        num_placeholders = (output["input_ids"] == processor.audio_output_token_id).sum().item()
        max_audio_length = int(num_placeholders * processor.sampling_rate / processor.frame_rate)
        output["audio_output_lengths"] = output["audio_output_lengths"].clamp(max=max_audio_length)

    for output in outputs:
        input_ids_shape = output["input_ids"].shape
        attention_mask_shape = output["attention_mask"].shape
        labels_shape = output["labels"].shape
        assert input_ids_shape == attention_mask_shape == labels_shape, (
            f"build_duplex_input_sequences: Shape mismatch for input, attention mask, or labels. "
            f"Got `{input_ids_shape=}`, `{attention_mask_shape=}`, `{labels_shape=}`."
        )
        num_audio_input = (output["input_ids"] == processor.audio_output_token_id).sum()
        num_audio_output = (output["labels"] == processor.audio_output_token_id).sum()
        if not getattr(processor, "no_audio_in_sil", False):
            assert num_audio_input == num_audio_output, (
                f"build_duplex_input_sequences: Audio token count mismatch between input and output. "
                f"Got `{num_audio_input=}` and `{num_audio_output=}`."
            )
        assert output["audio_output_lengths"] is not None, "build_duplex_input_sequences: `audio_output_lengths` is None."
        total_audio_samples = output["audio_output_lengths"].sum().item()
        expected_num_audio_frames = math.ceil(total_audio_samples * processor.frame_rate / processor.sampling_rate)
        if metadata.timeline is not None or all_utt_frames is not None:
            assert num_audio_input <= expected_num_audio_frames, (
                f"build_duplex_input_sequences: Timeline audio token count exceeds frame count. "
                f"Got `{expected_num_audio_frames=}` and `{num_audio_input=}`."
            )
        else:
            assert expected_num_audio_frames == num_audio_input, (
                f"build_duplex_input_sequences: Audio length does not match token count. "
                f"Expected `{expected_num_audio_frames}` but got `{num_audio_input=}`."
            )

    return outputs


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------


class RaonDuplexDataset(Dataset):
    """Simple JSONL reader dataset for duplex-timeline data.

    Each line must be a JSON object parseable by ``timeline_turns_to_metadata``.
    Audio is read from the ``audio_path`` field in the record (stereo, 2-channel).
    """

    def __init__(
        self,
        jsonl_path: str,
        processor: Any,
        loss_ignore_index: int = LOSS_IGNORE_INDEX,
        max_seq_length: int | None = None,
    ) -> None:
        self.jsonl_path = jsonl_path
        self.processor = processor
        self.loss_ignore_index = loss_ignore_index
        self.max_seq_length = max_seq_length
        self._records: list[dict] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self._records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> list[DuplexInputs]:
        record = self._records[idx]
        from .audio_io import load_audio as _load_audio_shared

        audio_path = record["audio_path"]
        audio, _ = _load_audio_shared(audio_path, self.processor.sampling_rate, mono=False)
        # pad audio to frame boundary so tokenize_audio doesn't fail
        samples_per_frame = int(self.processor.sampling_rate / self.processor.frame_rate)
        remainder = audio.shape[-1] % samples_per_frame
        if remainder != 0:
            pad_len = samples_per_frame - remainder
            audio = torch.nn.functional.pad(audio, (0, pad_len))
        # audio shape: [num_channels, num_samples]
        metadata = timeline_turns_to_metadata(record)
        channels = build_duplex_input_sequences(
            audio=audio,
            metadata=metadata,
            processor=self.processor,
            loss_ignore_index=self.loss_ignore_index,
        )
        # Truncate sequences that exceed max_seq_length at dataset level.
        if self.max_seq_length is not None:
            msl = self.max_seq_length
            for ch in channels:
                if ch["input_ids"].shape[-1] > msl:
                    ch["input_ids"] = ch["input_ids"][:, :msl]
                    ch["labels"] = ch["labels"][:, :msl]
                    ch["attention_mask"] = ch["attention_mask"][:, :msl]
        return channels


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class _DuplexDataModule(TypedDict):
    train_dataset: Any


def make_raon_duplex_data_module(
    processor: Any,
    jsonl_paths: list[str],
    loss_ignore_index: int = LOSS_IGNORE_INDEX,
    **kwargs: Any,
) -> _DuplexDataModule:
    """Build a data module dict from a list of JSONL file paths.

    Args:
        processor: The processor for tokenization.
        jsonl_paths: List of JSONL file paths.
        loss_ignore_index: Label index to ignore in loss computation.

    Returns:
        Dict with ``train_dataset`` set to the resulting dataset.
    """
    datasets: list[RaonDuplexDataset] = []
    for path in jsonl_paths:
        ds = RaonDuplexDataset(
            jsonl_path=path,
            processor=processor,
            loss_ignore_index=loss_ignore_index,
        )
        datasets.append(ds)

    if len(datasets) == 1:
        train_dataset = datasets[0]
    else:
        from torch.utils.data import ConcatDataset

        train_dataset = ConcatDataset(datasets)

    return {"train_dataset": train_dataset}


# ---------------------------------------------------------------------------
# Duplex collator
# ---------------------------------------------------------------------------


def duplex_collate_fn(batch: list[list[dict[str, Any]]]) -> dict[str, Any]:
    """Collate a batch of duplex samples.

    Each item in ``batch`` is a list[DuplexInputs] (one per channel).
    We flatten all channels and pad into a single batch.
    """
    flat: list[dict[str, Any]] = []
    for channels in batch:
        for item in channels:
            flat.append(item)

    if not flat:
        raise ValueError("Empty batch after flattening.")

    # Pad text tensors (input_ids, labels, attention_mask) to same length
    input_ids_list = [item["input_ids"].squeeze(0) for item in flat]
    labels_list = [item["labels"].squeeze(0) for item in flat]
    attention_mask_list = [item["attention_mask"].squeeze(0) for item in flat]

    input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=0)
    labels = pad_sequence(labels_list, batch_first=True, padding_value=LOSS_IGNORE_INDEX)
    attention_mask = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)

    # Flatten audio rows so multi-segment audio_output ([N_seg, T]) is collated safely.
    audio_input_rows = [row for item in flat for row in item["audio_input"]]
    audio_input_lengths = torch.cat([item["audio_input_lengths"].to(dtype=torch.long) for item in flat], dim=0)
    if len(audio_input_rows) != audio_input_lengths.shape[0]:
        raise ValueError(
            "duplex_collate_fn: audio_input rows and lengths mismatch "
            f"({len(audio_input_rows)} vs {audio_input_lengths.shape[0]})."
        )
    audio_input = pad_sequence(audio_input_rows, batch_first=True, padding_value=0.0)

    audio_output_rows = [row for item in flat for row in item["audio_output"]]
    audio_output_lengths = torch.cat([item["audio_output_lengths"].to(dtype=torch.long) for item in flat], dim=0)
    if len(audio_output_rows) != audio_output_lengths.shape[0]:
        raise ValueError(
            "duplex_collate_fn: audio_output rows and lengths mismatch "
            f"({len(audio_output_rows)} vs {audio_output_lengths.shape[0]})."
        )
    audio_output = pad_sequence(audio_output_rows, batch_first=True, padding_value=0.0)

    result: dict[str, Any] = {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "audio_input": audio_input,
        "audio_input_lengths": audio_input_lengths,
        "audio_output": audio_output,
        "audio_output_lengths": audio_output_lengths,
    }

    # Audio output segments (for no_audio_in_sil segmented encoding).
    # Only batch_size=1 is supported for duplex training, so take from first item.
    if flat[0].get("audio_output_segments") is not None:
        result["audio_output_segments"] = flat[0]["audio_output_segments"]

    # Speaker audio
    if flat[0].get("speaker_encoder_audio") is not None:
        speaker_rows = [row for item in flat for row in item["speaker_encoder_audio"]]
        speaker_lengths = torch.cat([item["speaker_encoder_audio_lengths"].to(dtype=torch.long) for item in flat], dim=0)
        if len(speaker_rows) != speaker_lengths.shape[0]:
            raise ValueError(
                f"duplex_collate_fn: speaker rows and lengths mismatch ({len(speaker_rows)} vs {speaker_lengths.shape[0]})."
            )
        result["speaker_encoder_audio"] = pad_sequence(speaker_rows, batch_first=True, padding_value=0.0)
        result["speaker_encoder_audio_lengths"] = speaker_lengths
        result["use_speaker_embedding"] = True

    return result
