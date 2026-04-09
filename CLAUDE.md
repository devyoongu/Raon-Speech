# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Raon-Speech is a bilingual (English/Korean) SpeechLM with 9B parameters by KRAFTON AI. It supports STT, TTS, SpeechChat, and TextQA tasks. There are two model variants:
- **Raon-Speech**: Offline SpeechLLM for speech understanding and generation
- **Raon-SpeechChat**: Full-duplex model for real-time conversational interaction

## Commands

### Setup
```bash
pip install -r requirements.txt && pip install -e .
# Or with uv:
uv sync
# Demo extras: pip install -e ".[demo]"
# Dev extras:  pip install -e ".[dev]"
```

### Lint & Type Check
```bash
ruff check src/                # Lint
ruff format src/               # Format
mypy src/raon/                 # Type check
```

### Tests
```bash
pytest tests/                  # All tests
pytest tests/ -m "not slow"    # Skip slow tests (model downloads, CPU inference)
pytest tests/test_foo.py -k "test_name"  # Single test
```

### Training
```bash
bash scripts/train.sh          # SpeechLLM training (wraps torchrun -m raon.train)
bash scripts/duplex_train.sh   # Full-duplex training (wraps torchrun -m raon.duplex_train)
```

### Inference
```bash
bash scripts/infer.sh          # SpeechLLM inference (wraps python -m raon.generate)
bash scripts/duplex_infer.sh   # Full-duplex inference (wraps python -m raon.duplex_generate)
```

### Demo
```bash
bash demo/run_gradio_demo.sh           # Gradio UI for offline tasks
bash scripts/export.sh                 # Export to SGLang format (required before duplex demo)
bash demo/run_gradio_duplex_demo.sh    # Realtime full-duplex demo
```

## Architecture

### Source Layout (`src/raon/`)

- **models/raon.py**: `RaonModel` and `RaonDuplexModel` (PreTrainedModel subclasses). Core model combining Qwen3 LM backbone, audio encoder, Mimi codec, speaker encoder, and code predictor.
- **models/wrapper.py**: `RaonInferenceModel` — inference wrapper with KV caching, streaming, and logits processing.
- **pipeline.py**: `RaonPipeline` — high-level API (`stt()`, `tts()`, `speech_chat()`, `textqa()`, `duplex()`).
- **modules/**: Neural network components — `audio_encoder.py`, `audio_tokenizer.py` (Mimi codec), `voxtral_encoder.py`, `code_predictor.py`, `adaptor.py`, `speaker_encoder.py`.
- **utils/processor.py**: `RaonProcessor` / `RaonDuplexProcessor` — converts JSONL conversations to tokenized model inputs.
- **utils/data.py**: `RaonLazyDataset` — lazy-loading JSONL dataset with audio retry logic.
- **utils/duplex_data.py**: Duplex-specific data loading with stereo audio and turn annotations.
- **utils/loss.py**: `RaonLossMixin` — combined text + audio loss computation with per-codebook and per-channel masking.
- **utils/special_tokens.py**: 12 special token IDs (AUDIO_START, AUDIO_END, DUPLEX_SIL, etc.) and tokenizer patching functions.
- **utils/state_machine.py**: `DuplexStateManager` — interaction state tracking (LISTEN, SPEAK, OVERLAP, BACKCHANNEL).
- **types.py**: TypedDicts for `RaonInputs`, `DuplexInputs`, `AudioEncoderOutput`.
- **train.py** / **duplex_train.py**: Training entry points.
- **generate.py** / **duplex_generate.py**: Batch inference entry points.
- **export.py**: Export HF checkpoint to SGLang bundle.

### Data Format

Training data uses JSONL with conversation format. Each line has `conversations` (human/gpt turns), `audios` (paths), and `channel` (task type: stt/tts/speech-chat/textqa). Duplex data uses stereo WAV files with per-channel turn/word annotations.

### Key Design Patterns

- Models integrate with HuggingFace `PreTrainedModel` / `from_pretrained()` ecosystem
- Audio is encoded to discrete codes via Mimi codec (8 codebook groups)
- Special tokens control audio boundaries and task routing
- Loss computation masks padding/placeholder tokens and supports per-channel weighting for duplex
- Inference uses streaming with Conv1d padding cache for the codec decoder

## Code Style

- Python 3.11+, ruff for linting/formatting (line length 125)
- Double quotes, space indentation
- Ruff rules: E, F, I (isort), B (bugbear), UP (pyupgrade), Q (quotes)
- Type annotations enforced by mypy (`disallow_untyped_defs = true`)