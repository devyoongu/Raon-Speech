# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Raon-Speech is a KRAFTON multilingual SpeechLM (9B, English/Korean) built on top of HuggingFace Transformers (Qwen3 backbone). It supports two main tracks:

- **SpeechLM (offline):** multi-task TTS, STT, SpeechChat, TextQA via JSONL conversations
- **Full-Duplex (realtime):** streaming conversational AI with turn-taking, backchanneling, and interruption handling

## Setup

```bash
# Option A: pip
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

# Option B: uv
uv sync

# With Gradio demos and realtime server
pip install -e ".[demo]"  # or: uv sync --extra demo

# Optional: FlashAttention for faster inference
pip install flash-attn
```

Hardware requirement: NVIDIA GPU with CUDA 12.4+. The model requires ~48GB VRAM (tested on RTX 6000 Pro, L40S). Minimum viable with FlashAttention may be 24GB.

## Common Commands

```bash
# Inference (SpeechLM)
bash scripts/infer.sh
python -m raon.generate

# Inference (Full-Duplex)
bash scripts/duplex_infer.sh
python -m raon.duplex_generate

# Training (SpeechLM)
bash scripts/train.sh
python -m raon.train

# Training (Full-Duplex)
bash scripts/duplex_train.sh
python -m raon.duplex_train

# Export HF checkpoint to SGLang bundle (for realtime server)
bash scripts/export.sh
python -m raon.export

# Gradio demos
bash demo/run_gradio_demo.sh           # SpeechLM
bash demo/run_gradio_duplex_demo.sh    # Full-Duplex (port 7861)

# Lint and format
ruff check src/raon
ruff format src/raon

# Type checking
mypy src/raon --strict

# Tests (no tests currently in repo, but framework configured)
pytest
pytest -m "not slow"  # skip slow tests (model downloads, CPU inference)
```

## Architecture

### High-Level Structure

```
RaonModel
├── LM backbone: Qwen3 9B (text transformer)
├── Audio Encoder: VoxtralEncoder + CausalAudioEncoder (streaming)
├── Audio Tokenizer: Mimi codec (discrete audio codes)
├── Speaker Encoder: PretrainedSpeakerEncoder (speaker-conditioned TTS)
├── Embedding Adaptor: text-audio alignment
└── Audio LM Head: code_predictor for next-token audio prediction

RaonDuplexModel (extends RaonModel)
└── Streaming state machine for real-time dual-channel interaction
```

### Key Source Files

| File | Purpose |
|------|---------|
| `src/raon/pipeline.py` | `RaonPipeline` — high-level Python API for all tasks |
| `src/raon/models/raon.py` | `RaonModel`, `RaonDuplexModel`, `RaonConfig` — core model + forward/loss |
| `src/raon/models/wrapper.py` | `RaonInferenceModel` — inference-time wrapper |
| `src/raon/modules/audio_encoder.py` | `VoxtralEncoder` + `AuTWrapper` / `CausalAudioEncoder` |
| `src/raon/modules/audio_tokenizer.py` | Mimi codec + `StreamingMimiModel` |
| `src/raon/modules/speaker_encoder.py` | Speaker embeddings for TTS conditioning |
| `src/raon/utils/processor.py` | `RaonProcessor`, `RaonDuplexProcessor` — tokenization + collation |
| `src/raon/utils/data.py` | `RaonLazyDataset` — SpeechLLM dataset loading |
| `src/raon/utils/duplex_data.py` | Full-duplex dataset loading + preprocessing |
| `src/raon/utils/loss.py` | `RaonLossMixin` — causal LM + audio code losses |
| `src/raon/utils/special_tokens.py` | `AUDIO_START`, `AUDIO_OUTPUT_PLACEHOLDER`, etc. |
| `src/raon/utils/state_machine.py` | Full-duplex turn-taking state logic |
| `src/raon/generate.py` | SpeechLLM CLI entry point (reads JSONL, writes output) |
| `src/raon/duplex_generate.py` | Full-duplex CLI entry point |
| `src/raon/train.py` | HuggingFace Trainer-based training |
| `src/raon/types.py` | `RaonInputs`, `DuplexInputs`, `AudioEncoderOutput`, etc. |

### Data Formats

**SpeechLLM JSONL** (`channel` ∈ `stt`, `tts`, `speech-chat`, `textqa`):
```json
{
  "conversations": [
    {"from": "human", "value": "Transcribe <audio>"},
    {"from": "gpt", "value": "Hello world"}
  ],
  "audios": ["/path/to/audio.wav"],
  "speaker_ref_audios": ["/path/to/ref.wav"],
  "channel": "stt"
}
```

**Full-Duplex JSONL** (`channel` = `full_duplex`):
```json
{
  "audio_path": "/path/to/stereo.wav",
  "language": "eng",
  "channel": "full_duplex",
  "speak_first": [true, false],
  "include_in_training": [true, true],
  "turns": [{"channel": 0, "start_sample": 0, "end_sample": 48000, "ipus": [...]}]
}
```

Sample data lives in `data/speechllm/` and `data/duplex/`.

### Inference Config

`config/infer.yaml` — task-specific defaults (max_new_tokens, temperature, RAS). `config/duplex_infer.yaml` — full-duplex streaming parameters.

### Pipeline API Quick Reference

```python
from raon import RaonPipeline

pipe = RaonPipeline("/path/to/model", device="cuda", dtype="bfloat16",
                    attn_implementation="sdpa")  # or "fa" (FlashAttention)

text = pipe.stt("/path/to/audio.wav")
audio, sr = pipe.tts("Hello, world!", speaker_audio="/ref.wav")
response = pipe.speech_chat("/path/to/audio.wav")
answer = pipe.textqa("What did they say?", audio="/path/to/audio.wav")
pipe.save_audio((audio, sr), "/path/to/out.wav")
```

### Realtime Server (Full-Duplex)

The realtime demo uses SGLang for speculative decoding. Workflow: export HF checkpoint → `python -m raon.export` → start Docker container → access Gradio at port 7861. See `docs/realtime-server-setup.md` (Korean) for full deployment details.

## Code Style

- Python 3.11+, line length 125, ruff rules: E, F, I, B, UP, Q
- mypy strict mode with `disallow_untyped_defs=True`
- `torch.bfloat16` preferred dtype; `attn_implementation` is user-configurable (`eager`, `sdpa`, `fa`)