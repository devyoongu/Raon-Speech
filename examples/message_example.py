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

"""Example: RAON inference with direct message input.

Usage:
    python examples/inference.py --model_path /path/to/model
    python examples/inference.py --model_path /path/to/model --task tts --text "Hello world"
    python examples/inference.py --model_path /path/to/model --task stt --audio /path/to/audio.wav
"""

from __future__ import annotations

import argparse
from pathlib import Path

from raon import RaonPipeline

_EXAMPLE_AUDIO_DIR = Path("data/speechllm/eval/audio")


def default_asset(name: str, fallback: str | None = None) -> str | None:
    path = _EXAMPLE_AUDIO_DIR / name
    if path.exists():
        return str(path)
    return fallback


# ---------------------------------------------------------------------------
# Task runners
# ---------------------------------------------------------------------------


def run_stt(pipe: RaonPipeline, args: argparse.Namespace, output_dir: Path) -> None:
    """STT: audio → text."""
    audio_path = args.audio or default_asset("stt.wav")
    if audio_path is None:
        raise FileNotFoundError("No default STT audio found under data/speechllm/eval/audio")
    print(f"\n[STT] Transcribing: {audio_path}")

    text = pipe.stt(audio_path)
    print(f"[STT] Transcript: {text}")

    out_path = output_dir / "stt_output.txt"
    out_path.write_text(text, encoding="utf-8")
    print(f"[STT] Saved to {out_path}")


def run_tts(pipe: RaonPipeline, args: argparse.Namespace, output_dir: Path) -> None:
    """TTS: text → audio."""
    text = args.text or "Hello, how are you today?"
    print(f"\n[TTS] Synthesising: {text!r}")

    speaker_audio = args.speaker_audio or default_asset("spk_ref.wav")
    audio, sr = pipe.tts(text, speaker_audio=speaker_audio)

    out_path = output_dir / "tts_output.wav"
    pipe.save_audio((audio, sr), str(out_path))
    print(f"[TTS] Saved audio to {out_path} ({audio.shape[0] / sr:.2f}s)")


def run_tts_continuation(pipe: RaonPipeline, args: argparse.Namespace, output_dir: Path) -> None:
    """TTS Continuation: text + reference audio → audio."""
    text = args.text or "This is the continuation sentence."
    default_ref = default_asset("spk_ref.wav")
    ref_audio_path = args.ref_audio or args.speaker_audio or default_ref
    if ref_audio_path is None:
        print("\n[TTS-Cont] Skipped: requires --ref_audio or --speaker_audio")
        return
    ref_text = args.ref_text
    print(f"\n[TTS-Cont] Target text: {text!r}")
    print(f"[TTS-Cont] Ref audio: {ref_audio_path}")
    if ref_text:
        print(f"[TTS-Cont] Ref text: {ref_text!r}")
    else:
        print("[TTS-Cont] Ref text: (will auto-transcribe)")

    audio, sr = pipe.tts_continuation(
        target_text=text,
        ref_audio=ref_audio_path,
        ref_text=ref_text,
        speaker_audio=args.speaker_audio,
    )

    out_path = output_dir / "tts_continuation_output.wav"
    pipe.save_audio((audio, sr), str(out_path))
    print(f"[TTS-Cont] Saved audio to {out_path} ({audio.shape[0] / sr:.2f}s)")


def run_textqa(pipe: RaonPipeline, args: argparse.Namespace, output_dir: Path) -> None:
    """TextQA: text + optional audio → text."""
    audio_path = args.audio or default_asset("textqa.wav")
    if audio_path is None:
        raise FileNotFoundError("No default TextQA audio found under data/speechllm/eval/audio")
    question = args.text or "What is the gender of the speaker?"
    print(f"\n[TextQA] Audio: {audio_path}")
    print(f"[TextQA] Question: {question!r}")

    answer = pipe.textqa(question, audio=audio_path)
    print(f"[TextQA] Answer: {answer}")

    out_path = output_dir / "textqa_output.txt"
    out_path.write_text(answer, encoding="utf-8")
    print(f"[TextQA] Saved to {out_path}")


def run_speech_chat(pipe: RaonPipeline, args: argparse.Namespace, output_dir: Path) -> None:
    """SpeechChat: audio → text."""
    audio_path = args.audio or default_asset("speech-chat.wav")
    if audio_path is None:
        raise FileNotFoundError("No default SpeechChat audio found under data/speechllm/eval/audio")
    print(f"\n[SpeechChat] Spoken question audio: {audio_path}")

    answer = pipe.speech_chat(audio_path)
    print(f"[SpeechChat] Answer: {answer}")

    out_path = output_dir / "speech-chat_output.txt"
    out_path.write_text(answer, encoding="utf-8")
    print(f"[SpeechChat] Saved to {out_path}")


def run_chat(pipe: RaonPipeline, args: argparse.Namespace, output_dir: Path) -> None:
    """Chat: text + audio (free-form) → text / audio."""
    audio_path = args.audio or default_asset("stt.wav")
    if audio_path is None:
        raise FileNotFoundError("No default chat audio found under data/speechllm/eval/audio")
    print(f"\n[Chat] Free-form multimodal conversation (audio: {audio_path})")
    prompt = "Please transcribe this audio and summarise the content."

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    response = pipe.chat(messages)
    print(f"[Chat] Response: {response}")

    out_path = output_dir / "chat_output.txt"
    out_path.write_text(response, encoding="utf-8")
    print(f"[Chat] Saved to {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="RAON inference examples")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained RAON model directory.")
    parser.add_argument(
        "--task",
        type=str,
        default="all",
        choices=["all", "stt", "tts", "tts_continuation", "textqa", "speech_chat", "chat"],
        help="Which task to run (default: all).",
    )
    parser.add_argument("--audio", type=str, default=None, help="Input audio path (overrides dummy default).")
    parser.add_argument("--text", type=str, default=None, help="Input text (overrides built-in default).")
    parser.add_argument(
        "--speaker_audio",
        type=str,
        default=None,
        help="Speaker reference audio for TTS voice conditioning.",
    )
    parser.add_argument("--ref_audio", type=str, default=None, help="Reference audio path for TTS continuation.")
    parser.add_argument("--ref_text", type=str, default=None, help="Transcription of ref audio (auto-transcribed if omitted).")
    parser.add_argument("--output_dir", type=str, default="output/pipeline", help="Directory to save outputs.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on (e.g. 'cuda', 'cpu').")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Torch dtype for model weights.",
    )
    args = parser.parse_args()

    print(f"Loading RaonPipeline from {args.model_path} ...")
    pipe = RaonPipeline(args.model_path, device=args.device, dtype=args.dtype)
    print("Pipeline ready.\n")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.task in ("all", "stt"):
        run_stt(pipe, args, output_dir)
    if args.task in ("all", "tts"):
        run_tts(pipe, args, output_dir)
    if args.task in ("all", "tts_continuation"):
        run_tts_continuation(pipe, args, output_dir)
    if args.task in ("all", "textqa"):
        run_textqa(pipe, args, output_dir)
    if args.task in ("all", "speech_chat"):
        run_speech_chat(pipe, args, output_dir)
    if args.task in ("all", "chat"):
        run_chat(pipe, args, output_dir)

    print(f"\nAll outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
