# Raon-SpeechChat Realtime 서버 구축 가이드

Raon-SpeechChat-9B 모델을 사용한 Full-Duplex Realtime 서버 구축 절차입니다.

## 요구사항

| 항목 | 사양 |
|------|------|
| GPU | README 기준 테스트 GPU: **NVIDIA RTX 6000 Pro / L40S** (48GB VRAM) |
| Python | 3.11 이상 (Docker 사용 시 자동 해결) |
| CUDA Driver | 575+ (CUDA 12.8 이상) |
| 디스크 | ~50GB (모델 19GB + Docker 이미지 ~15GB + SGLang 번들 ~17GB) |
| Docker | nvidia-container-toolkit 설치 필요 (`--gpus all` 지원) |

> **참고:** RTX 3090 (24GB)에서 테스트한 결과, SGLang text model + audio encoder/decoder + concurrent audio decoder + KV cache 로딩 시 CUDA OOM이 발생했습니다. 24GB GPU에서는 실행이 확인되지 않았습니다.

## 전체 절차

### Step 1: 모델 다운로드

```bash
docker run --rm \
    -v /home/posicube/.cache/huggingface:/root/.cache/huggingface \
    -e HF_HOME=/root/.cache/huggingface \
    python:3.11-slim \
    bash -c "pip install -q huggingface_hub && python -c \"from huggingface_hub import snapshot_download; snapshot_download('KRAFTON/Raon-SpeechChat-9B')\""
```

다운로드 확인:
```bash
ls /home/posicube/.cache/huggingface/hub/models--KRAFTON--Raon-SpeechChat-9B/snapshots/
# 출력 예: d844aee2c3da129b92fce3e0193c07eb98b88443
```

이후 명령어에서 사용할 snapshot 경로를 변수로 지정:
```bash
SNAPSHOT_HASH=$(ls /home/posicube/.cache/huggingface/hub/models--KRAFTON--Raon-SpeechChat-9B/snapshots/)
HF_MODEL_PATH="/root/.cache/huggingface/hub/models--KRAFTON--Raon-SpeechChat-9B/snapshots/${SNAPSHOT_HASH}"
```

### Step 2: .dockerignore 생성

빌드 컨텍스트에서 대용량 파일을 제외합니다:
```bash
cat > .dockerignore << 'EOF'
output/
.git/
.idea/
__pycache__/
*.pyc
.venv/
EOF
```

### Step 3: Docker 이미지 빌드

```bash
docker build -t raon-realtime -f Dockerfile.realtime .
```

`Dockerfile.realtime`은 다음을 포함합니다:
- Python 3.11-slim 베이스
- PyTorch CUDA 12.8
- 프로젝트 의존성 + demo extras (sglang, gradio, fastapi, uvicorn)

### Step 4: SGLang 번들 Export

Realtime 서버는 HuggingFace 체크포인트를 직접 사용하지 않고, SGLang 번들 형식이 필요합니다.

```bash
docker run --rm --gpus all \
    -v /home/posicube/.cache/huggingface:/root/.cache/huggingface \
    -e HF_HOME=/root/.cache/huggingface \
    -v $(pwd)/output:/app/output \
    raon-realtime \
    python -m raon.export \
        --input_path "${HF_MODEL_PATH}" \
        --output_path output/sglang-bundle \
        --dtype bfloat16
```

성공 시 출력:
```
Export complete!
  text_model:   output/sglang-bundle/text_model
  raon_runtime: output/sglang-bundle/raon_runtime
```

### Step 5: Gradio Realtime 서버 실행

```bash
docker run --rm --gpus all \
    -p 7861:7861 \
    -v /home/posicube/.cache/huggingface:/root/.cache/huggingface \
    -e HF_HOME=/root/.cache/huggingface \
    -v $(pwd)/output:/app/output \
    -v $(pwd)/wav:/app/wav \
    -v $(pwd)/data:/app/data \
    raon-realtime \
    python demo/gradio_duplex_demo.py \
        --host 0.0.0.0 \
        --port 7861 \
        --model-path output/sglang-bundle \
        --result-root output/fd_gradio_demo \
        --speaker-audio wav/femail_achernar.wav \
        --compile-audio-modules false
```

서버가 정상 기동되면 아래 메시지가 표시됩니다:
```
INFO:     Uvicorn running on http://0.0.0.0:7861
```

### Step 6: 브라우저 접속

원격 서버의 마이크를 브라우저에서 사용하려면 HTTPS 또는 localhost가 필요합니다.

**방법 A: SSH 터널 (권장)**
```bash
# 로컬 PC에서 실행
ssh -L 7861:localhost:7861 posicube@<서버IP>
```
그 후 브라우저에서 `http://localhost:7861` 접속

**방법 B: Gradio Share Link**
```bash
# Step 5 명령어에 --share 추가
python demo/gradio_duplex_demo.py ... --share
```
Gradio가 생성하는 공유 URL로 접속

### 브라우저 UI 사용법

1. **Start** 클릭 → 마이크 권한 허용
2. 마이크로 말하면 AI가 실시간 음성으로 응답 (전화 통화처럼)
3. **Finish** 클릭 → 세션 종료
4. Downloads 섹션에서 녹음 파일 다운로드 가능

## 서버 실행 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--host` | 0.0.0.0 | 바인드 주소 |
| `--port` | 7861 | 포트 |
| `--model-path` | KRAFTON/Raon-SpeechChat-9B | SGLang 번들 경로 |
| `--speaker-audio` | data/duplex/eval/audio/spk_ref.wav | 기본 화자 참조 오디오 |
| `--compile-audio-modules` | true | 오디오 모듈 torch.compile (false로 설정 시 시작 빠름) |
| `--disable-cuda-graph` | false | CUDA graph 비활성화 (VRAM 절약, 성능 저하) |
| `--mem-fraction-static` | 0.88 | GPU 메모리 할당 비율 |
| `--share` | false | Gradio 공유 링크 생성 |

## 아키텍처

```
브라우저 (마이크/스피커)
    │
    │  WebSocket (/realtime/ws)
    │  ↕ 실시간 PCM float32 오디오 스트리밍
    │
FastAPI + Gradio 서버 (port 7861)
    │
    ├── POST /realtime/session/start  → 세션 생성
    ├── WS   /realtime/ws             → 양방향 오디오 스트리밍
    └── POST /realtime/session/finish → 세션 종료, 녹음 파일 반환
    │
    ├── RealtimeRuntimeManager (세션 1개 관리)
    ├── LocalRealtimeSession (세션 오케스트레이션)
    │   ├── feed_audio()  ← 사용자 음성 입력
    │   ├── step()        → duplex 추론 1스텝
    │   └── handle_audio_frame() → 응답 오디오 + 텍스트
    │
    └── SGLangRaonModel (GPU 추론 백엔드)
        ├── text_model (Qwen3, SGLang ModelRunner)
        └── raon_runtime (audio encoder/decoder/codec)
```

## 트러블슈팅

| 에러 | 원인 | 해결 |
|------|------|------|
| `CUDA error: out of memory` | VRAM 부족 | 48GB+ GPU 필요 |
| `Could not find CUDA installation` | CUDA toolkit 없음 | NVIDIA CUDA devel 이미지 사용 또는 `--disable-cuda-graph` |
| `No space left on device` (Docker build) | 디스크 부족 | `docker system prune -a`로 미사용 이미지 정리 |
| 빌드 컨텍스트 20GB+ | .dockerignore 없음 | Step 2 참조 |
| 브라우저 마이크 안 됨 | HTTP에서 마이크 차단 | SSH 터널 또는 `--share` 사용 |
