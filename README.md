# Raon-Speech

<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/Raon-Speech-Gradient-White.png">
    <img src="assets/Raon-Speech-Gradient-Black.png" alt="Raon-Speech Logo" width="360">
  </picture>
</div>
<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/Raon-Speechchat-Gradient-White.png">
    <img src="assets/Raon-Speechchat-Gradient-Black.png" alt="Raon-SpeechChat Logo" width="360">
  </picture>
</div>

<p align="center">
  <a href="https://www.krafton.ai/ko/"><img src="https://img.shields.io/badge/Homepage-KRAFTON%20AI-blue?style=flat&logo=google-chrome&logoColor=white" alt="Homepage"></a>
  <a href="https://github.com/krafton-ai/Raon-Speech"><img src="https://img.shields.io/badge/GitHub-Raon%20Speech-white?style=flat&logo=github&logoColor=black" alt="GitHub"></a>
  <a href="https://huggingface.co/KRAFTON"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-KRAFTON-yellow?style=flat" alt="Hugging Face"></a>
  <a href="https://x.com/Krafton_AI"><img src="https://img.shields.io/badge/X-KRAFTON%20AI-white?style=flat&logo=x&logoColor=black" alt="X"></a>
</p>

**Links**
- GitHub: https://github.com/krafton-ai/Raon-Speech
- Official Demo: https://raon.krafton.ai
- Hugging Face Org: https://huggingface.co/KRAFTON
- Speech model card: https://huggingface.co/KRAFTON/Raon-Speech-9B
- SpeechChat (Full-Duplex) model card: https://huggingface.co/KRAFTON/Raon-SpeechChat-9B

Raon is a speech model built on HuggingFace Ecosystem.  
This repo contains two tracks:

- Raon-Speech (offline): `TTS`, `STT`, `SpeechChat`, `TextQA`
- Raon-SpeechChat (Full-Duplex) (realtime/offline duplex decoding)

Both tracks share the same core model family and processor stack under `src/raon/`.

## Requirements

- Python `>=3.11`
- CUDA GPU recommended (`bfloat16` / `float16`)
- PyTorch + Torchaudio matching your CUDA environment

## Model Loading (Local or Hugging Face Hub)

All model entry points accept either:

- local checkpoint directory
- Hugging Face `repo_id` (downloaded automatically by `from_pretrained`)

Examples:

```bash
# SpeechLLM
bash scripts/infer.sh KRAFTON/Raon-Speech-9B /path/to/data_dir /path/to/output_dir

# Full-Duplex
bash scripts/duplex_infer.sh KRAFTON/Raon-SpeechChat-9B /path/to/input.wav /path/to/output_dir
```

```python
from raon import RaonPipeline

pipe = RaonPipeline("KRAFTON/Raon-Speech-9B", device="cuda", dtype="bfloat16")
# or
pipe = RaonPipeline("KRAFTON/Raon-SpeechChat-9B", device="cuda", dtype="bfloat16")
```

If you want to pre-download first:

```bash
huggingface-cli download KRAFTON/Raon-Speech-9B --local-dir /path/to/model_dir
# or
huggingface-cli download KRAFTON/Raon-SpeechChat-9B --local-dir /path/to/model_dir
```

## Execution Modes

### Mode 1: `raon` installed (recommended)

After `pip install -e .` (or `uv sync`), all entry points are supported:

- `scripts/infer.sh`, `scripts/duplex_infer.sh`
- `scripts/train.sh`, `scripts/duplex_train.sh`
- `demo/run_gradio_demo.sh`, `demo/run_gradio_duplex_demo.sh`
- Python API: `from raon import RaonPipeline`

### Mode 2: without `raon` install

Supported:

- Raon-Speech Gradio demo (`demo/gradio_demo.py`) has a fallback that loads Hub remote code when `raon` import fails.
  Run directly from HF repo:

```bash
# from repo root
bash demo/run_gradio_demo.sh --model KRAFTON/Raon-Speech-9B --port 7860
```

- Pure Transformers flow using Hub remote code (advanced):
  `AutoModel.from_pretrained(..., trust_remote_code=True)`.
  See examples: `examples/message_example.ipynb` and `examples/duplex_example.ipynb`.
  Minimal pipeline load without `raon` install:

```python
import torch
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

MODEL_ID = "KRAFTON/Raon-Speech-9B"

_cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
RaonPipeline = get_class_from_dynamic_module(
    "modeling_raon.RaonPipeline",
    MODEL_ID,
    revision=getattr(_cfg, "_commit_hash", None),
)
del _cfg

pipe = RaonPipeline(MODEL_ID, device="cuda", dtype="bfloat16")
```

Not supported as-is:

- `python -m raon.*` module commands (used by `scripts/*.sh`)
- Raon-SpeechChat (Full-duplex) realtime demo runtime (`demo/gradio_duplex_demo.py`) because runtime code imports `raon.*` modules.

If you do not want package installation but run from source checkout, set `PYTHONPATH` to `src`:

```bash
export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"
```

## Environment Setup

### Option A: `venv` + `pip`

```bash
git clone https://github.com/krafton-ai/Raon-Speech
cd Raon-Speech
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Option B: `uv`

```bash
git clone https://github.com/krafton-ai/Raon-Speech
cd Raon-Speech
uv sync
```

If you want to pin the virtualenv directory name:

```bash
UV_PROJECT_ENVIRONMENT=venv_uv uv sync
```

### Demo dependencies (optional)

The realtime duplex demo needs extra packages (`sglang`, `gradio`, `fastapi`, `uvicorn`):

```bash
pip install -e ".[demo]"
# or
uv sync --extra demo
```

### FlashAttention (optional)

If you want to use FlashAttention during training, install it separately:

```bash
pip install flash-attn
```

## Project Layout

```text
Raon-Speech/
├── src/raon/                 # package code
│   ├── models/               # RaonModel / RaonDuplexModel
│   ├── modules/              # audio encoder, tokenizer, speaker encoder, etc.
│   ├── utils/                # processor, datasets, losses, prompts, special tokens
│   ├── train.py              # SpeechLLM training entry
│   ├── duplex_train.py       # Full-duplex training entry
│   ├── generate.py           # SpeechLLM JSONL inference entry
│   ├── duplex_generate.py    # Full-duplex inference entry
│   └── pipeline.py           # high-level API (RaonPipeline)
├── scripts/                  # shell wrappers
├── demo/                     # Gradio demos + realtime app
├── config/                   # inference configs
├── data/                     # sample datasets
└── examples/                 # notebooks and scripts
```

## Model Architecture

- One shared backbone: `RaonModel` (LM backbone + audio encoder + Mimi codec path).
- Two model types:
  `raon` (Raon-Speech) and `raon_duplex` (full-duplex; `RaonDuplexModel` alias with duplex defaults).
- Main trainable blocks include text/audio alignment and audio code prediction
  (`input_adaptor`, `output_adaptor`, `audio_lm_head`, `proj_code`, `code_predictor`).

## SpeechLLM

### Data Format (JSONL)

Each line is one sample.

| Field | Type | Description |
|---|---|---|
| `conversations` | `list[dict]` | turns with `from` (`human`/`gpt`) and `value` |
| `audios` | `list[str]` | audio paths consumed by `<audio>` tags in order |
| `speaker_ref_audios` | `list[str]` | optional speaker reference audio for TTS |
| `channel` | `str` | `tts`, `stt`, `speech-chat`, `textqa` |
| `system` | `str` | optional system prompt |

Sample eval data is under `data/speechllm/eval`.

### Inference (JSONL batch)

```bash
bash scripts/infer.sh /path/to/model /path/to/data_dir /path/to/output_dir
```

Key points:

- `scripts/infer.sh` wraps `python -m raon.generate`
- task defaults come from `config/infer.yaml`
- `BATCH_SIZE` env controls batch size (default `1` in script)
- `ATTN_IMPLEMENTATION` env controls attention backend (default `sdpa`; use `fa` for FlashAttention)
- `max_audio_chunk_length` defaults to `192000` for audio-input text tasks

### Pipeline API

```python
from raon import RaonPipeline

pipe = RaonPipeline("/path/to/model", device="cuda", dtype="bfloat16")

text = pipe.stt("audio.wav")
audio, sr = pipe.tts("Hello!", speaker_audio="spk_ref.wav")
ans1 = pipe.speech_chat("question.wav")
ans2 = pipe.textqa("What did the speaker say?", audio="audio.wav")

pipe.save_audio((audio, sr), "tts.wav")
```

Continuation TTS is also supported:

```python
audio, sr = pipe.tts_continuation(
    target_text="Continue this sentence.",
    ref_audio="ref.wav",
    ref_text="Optional transcription of ref audio.",
)
```

See:

- `examples/message_example.py`
- `examples/message_example.ipynb`

### Training

```bash
bash scripts/train.sh /path/to/model /path/to/data_dir /path/to/output_dir
```

Common env knobs:

- `NPROC_PER_NODE` (multi-GPU torchrun)
- `MAX_STEPS` (default `1000`)
- `SAVE_STEPS` (default `500`)
- `BATCH_SIZE` (default `1`)
- `USE_SPEAKER_EMBEDDING` (default `true` in `scripts/train.sh`)

Notes:

- In `raon.train`, packing is enabled by default (`--use_packing`)
- Script default enables speaker-conditioning inputs; underlying CLI supports `--no-use_speaker_embedding`
- Current training code freezes these modules: `audio_encoder`, `input_adaptor`, `output_adaptor`, `audio_lm_head`, `proj_code`, `code_predictor`, `speaker_encoder`
- Default training attention implementation is `sdpa`
- To use FlashAttention for training, pass `--attn_implementation fa`

## Full-Duplex

### Data Format (JSONL)

Training (`duplex_train.py`) expects one JSON object per line with stereo audio metadata.

Required top-level fields:

| Field | Type | Description |
|---|---|---|
| `audio_path` | `str` | Path to stereo wav (`2` channels). |
| `language` | `str` | Language code (for prompt selection), e.g. `eng`, `kor`. |
| `channel` | `str` | Duplex channel type, e.g. `full_duplex` or `duplex_instruct`. |
| `speak_first` | `list[int|bool]` | Per-channel flag (length 2). |
| `include_in_training` | `list[int|bool]` | Per-channel train inclusion flag (length 2). |
| `turns` or `scripts` | `list` | Conversation timing/transcript annotations (one of the two formats below). |

Supported annotation format A (`turns`):

- Each turn contains `channel`, `start_sample`, `end_sample`, and `ipus`.
- Each IPU contains `words`, where each word has `word`, `start_sample`, `end_sample`.

Supported annotation format B (`scripts`):

- `scripts` is `[ch0_words, ch1_words]`, and each word item has `word`, `start`, `end` (seconds).
- Optional `timeline` or `rough_timeline` can be provided for utterance bounds.

Optional fields:

- `sample_rate` (used for `turns` timestamps; defaults to `24000` if omitted)
- `system_prompt` (if omitted, prompt is built from persona/context/name metadata when available)

Inference metadata sidecar (`duplex_generate.py`, same stem `.jsonl` near input wav) uses a lightweight format:

| Field | Type | Description |
|---|---|---|
| `audio_path` | `str` | Input audio path metadata (optional for CLI run, used in dataset-style records). |
| `speak_first` | `bool` | Override initial speaking mode. |
| `language` | `str` | Optional language hint. |
| `persona` | `str` | Persona key/text for prompt builder. |
| `context` | `str` | Extra context appended to system prompt. |
| `system_prompt` | `str` | Explicit prompt override. |
| `speaker_audio` | `str` | Speaker reference wav path (preferred key). |
| `speaker_ref_audios` | `list[str]` | Fallback speaker reference list (first element used). |

Sample data:

- train: `data/duplex/train`
- eval: `data/duplex/eval`
- persona catalog: `data/duplex/personas.json`

Training data uses timeline JSONL + stereo audio assets.  
Eval examples are provided as mono user-input wavs (`data/duplex/eval/audio/duplex_00.wav`, etc.).

### Inference (simple wrapper)

```bash
bash scripts/duplex_infer.sh /path/to/model /path/to/input.wav /path/to/output_dir
```

`scripts/duplex_infer.sh` forwards fixed core args and optional `SPEAKER_AUDIO` env:

```bash
SPEAKER_AUDIO=data/duplex/eval/audio/spk_ref.wav \
bash scripts/duplex_infer.sh /path/to/model /path/to/input.wav /path/to/output_dir
```

Attention backend can be set with env var (default `eager`; use `fa` for FlashAttention):

```bash
ATTN_IMPLEMENTATION=fa \
bash scripts/duplex_infer.sh /path/to/model /path/to/input.wav /path/to/output_dir
```

### Inference (advanced CLI options)

For persona/context/sampling options, run the python module directly:

```bash
python -m raon.duplex_generate \
  --model_path /path/to/model \
  --audio_input data/duplex/eval/audio/duplex_00.wav \
  --output_dir /path/to/output_dir \
  --speaker_audio data/duplex/eval/audio/spk_ref.wav \
  --persona scenario_restaurant \
  --context "Discuss restaurant menu choices with a customer." \
  --temperature 0.9 --top_k 66 --top_p 0.95
```

Metadata auto-load is supported: if `duplex_00.jsonl` exists next to `duplex_00.wav` (or one level up), prompt/speaker settings are read from metadata unless overridden by CLI.

### Pipeline API

```python
from raon import RaonPipeline

pipe = RaonPipeline("/path/to/duplex-model", device="cuda", dtype="bfloat16")
audio = pipe.load_audio("data/duplex/eval/audio/duplex_00.wav")

summary = pipe.duplex(
    audio_input=audio,
    output_dir="/path/to/output_dir",
    speak_first=False,
    system_prompt="You are engaging in real-time conversation.",
    speaker_audio="data/duplex/eval/audio/spk_ref.wav",
)
print(summary)
```

See:

- `examples/duplex_example.ipynb`

### Training

```bash
bash scripts/duplex_train.sh /path/to/model /path/to/data_dir /path/to/output_dir
```

Current behavior:

- `batch_size` is enforced to `1` in code
- packing is enabled by default (`--use_packing`, disable with `--no-use_packing`)
- defaults: `MAX_STEPS=100`, `SAVE_STEPS=50`
- frozen modules include `speaker_encoder` (same freeze list pattern as SpeechLLM training)
- Default training attention implementation is `sdpa`
- To use FlashAttention for training, pass `--attn_implementation fa`

## Gradio Demos

### Raon-Speech demo

```bash
bash demo/run_gradio_demo.sh --model /path/to/model --port 7860
```

### Raon-SpeechChat realtime demo

1. Export HF checkpoint to SGLang bundle:

```bash
bash scripts/export.sh /path/to/hf_duplex_checkpoint /path/to/sglang_bundle
```

2. Run demo:

```bash
bash demo/run_gradio_duplex_demo.sh --model-path /path/to/sglang_bundle --port 7861
```

Notes:

- Full-Duplex realtime demo expects bundle layout containing `text_model/` and `raon_runtime/` (or `duplex_model/`)
- Launcher keeps compile warmup enabled (`FD_ENABLE_COMPILE_AUDIO_MODULES=1`)

## License

Apache License 2.0. See `LICENSE`.
