"""Raon-Speech inference test: STT, TTS, SpeechChat."""

from __future__ import annotations

import argparse
import time
from pathlib import Path


def run_test(name: str, fn, *args, **kwargs):  # type: ignore[no-untyped-def]
    print(f"\n{'='*60}")
    print(f"[{name}] Starting...")
    print(f"{'='*60}")
    t0 = time.time()
    try:
        result = fn(*args, **kwargs)
        elapsed = time.time() - t0
        print(f"[{name}] SUCCESS ({elapsed:.1f}s)")
        return result
    except Exception as e:
        elapsed = time.time() - t0
        print(f"[{name}] FAILED ({elapsed:.1f}s): {e}")
        import traceback
        traceback.print_exc()
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Raon-Speech inference test")
    parser.add_argument("--model_path", type=str, default="KRAFTON/Raon-Speech-9B")
    parser.add_argument("--audio", type=str, default="wav/femail_achernar.wav")
    parser.add_argument("--output_dir", type=str, default="output/test")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_path = args.audio
    print(f"Model: {args.model_path}")
    print(f"Audio: {audio_path}")
    print(f"Device: {args.device} / dtype: {args.dtype}")

    # Print GPU info
    if args.device == "cuda":
        import torch

        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        if gpu_mem < 25:
            print("WARNING: 9B model needs ~18GB+ VRAM. OOM may occur during inference.")
            print("  If OOM, try: --dtype float16 or reduce max_new_tokens")

    # Load pipeline
    from raon import RaonPipeline

    print("\nLoading RaonPipeline...")
    t0 = time.time()
    pipe = RaonPipeline(args.model_path, device=args.device, dtype=args.dtype)
    print(f"Pipeline loaded in {time.time() - t0:.1f}s\n")

    # --- 1. STT ---
    def test_stt():  # type: ignore[no-untyped-def]
        text = pipe.stt(audio_path)
        print(f"  Transcript: {text}")
        out = output_dir / "stt_output.txt"
        out.write_text(text, encoding="utf-8")
        print(f"  Saved to {out}")
        return text

    transcript = run_test("STT", test_stt)

    # --- 2. TTS ---
    def test_tts():  # type: ignore[no-untyped-def]
        text = "안녕하세요, 라온 스피치 음성 합성 테스트입니다. 오늘 날씨가 정말 좋네요."
        print(f"  Input text: {text!r}")
        audio, sr = pipe.tts(text, speaker_audio=audio_path)
        out = output_dir / "tts_output.wav"
        pipe.save_audio((audio, sr), str(out))
        print(f"  Generated audio: {audio.shape[0] / sr:.2f}s @ {sr}Hz")
        print(f"  Saved to {out}")
        return audio, sr

    run_test("TTS", test_tts)

    # --- 3. SpeechChat (text response) ---
    def test_speech_chat():  # type: ignore[no-untyped-def]
        answer = pipe.speech_chat(audio_path)
        print(f"  Answer: {answer}")
        out = output_dir / "speech_chat_output.txt"
        out.write_text(answer, encoding="utf-8")
        print(f"  Saved to {out}")
        return answer

    run_test("SpeechChat (text)", test_speech_chat)

    # --- 4. Audio-to-Audio: audio input → audio output ---
    def test_audio_to_audio():  # type: ignore[no-untyped-def]
        messages = [
            {
                "role": "user",
                "content": [{"type": "audio", "audio": audio_path}],
            }
        ]
        result = pipe.chat(
            messages,
            force_audio_output=True,
            force_text_output=False,
            speaker_audio=audio_path,
        )
        audio, sr = result
        out = output_dir / "audio_to_audio_output.wav"
        pipe.save_audio((audio, sr), str(out))
        print(f"  Generated audio: {audio.shape[0] / sr:.2f}s @ {sr}Hz")
        print(f"  Saved to {out}")
        return audio, sr

    run_test("Audio-to-Audio", test_audio_to_audio)

    # --- Summary ---
    print(f"\n{'='*60}")
    print("TEST COMPLETE")
    print(f"Outputs saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
