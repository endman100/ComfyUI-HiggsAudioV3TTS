# ComfyUI-HiggsAudioV3TTS

ComfyUI custom node for `bosonai/higgs-audio-v3-tts-4b`.

The node talks to the official SGLang-Omni `/v1/audio/speech` API and returns native ComfyUI `AUDIO`, so it can be connected directly to the built-in `PreviewAudio`, `Save Audio (FLAC)`, `Save Audio (MP3)`, and audio/video nodes.

## What It Supports

- Zero-shot text-to-speech
- Voice cloning through `references`
- SSE streaming WAV chunks
- Raw PCM streaming
- Direct local pipeline mode with a ComfyUI-style model loader node
- Inline Higgs control tokens such as `<|emotion:amusement|>`, `<|style:shouting|>`, `<|prosody:pause|>`, and `<|sfx:laughter|>Haha`

Higgs Audio v3 is released by Boson AI for research and non-commercial use. Review the model license before using generated audio.

## Install

Place this folder in:

```text
ComfyUI/custom_nodes/ComfyUI-HiggsAudioV3TTS
```

Install the small client-side requirements in the same Python environment used by ComfyUI:

```bash
cd /path/to/ComfyUI
source /path/to/comfyui-venv/bin/activate
pip install -r custom_nodes/ComfyUI-HiggsAudioV3TTS/requirements.txt
```

## Direct Local Pipeline Mode

You do not have to manually run `sgl-omni serve` if you use:

```text
Higgs Audio V3 Model Loader -> Higgs Audio V3 Local TTS -> SaveAudio
```

This mode starts a local Higgs pipeline managed by the ComfyUI node and calls it directly, without exposing an HTTP server. The loader UI intentionally stays small: `model_path` and `device`.

The `model_path` widget follows the usual ComfyUI model-directory pattern. Put a local copy under either of these roots and select the relative model name in the loader:

```text
ComfyUI/models/higgs_audio/bosonai/higgs-audio-v3-tts-4b
ComfyUI/models/LLM/bosonai/higgs-audio-v3-tts-4b
```

If you use `extra_model_paths.yaml`, the loader also scans `higgs_audio`, `llm`, and `LLM` entries. A workflow value of `bosonai/higgs-audio-v3-tts-4b` will resolve to the local directory first when one exists, and only falls back to Hugging Face when no local model folder is found.

By default the loader starts a node-managed Python worker using the current Python executable. If SGLang-Omni is installed in a different Python environment, configure it with environment variables before starting ComfyUI:

```text
HIGGS_AUDIO_V3_PYTHON_EXECUTABLE=/path/to/python
HIGGS_AUDIO_V3_RUNTIME_MODE=python_worker
HIGGS_AUDIO_V3_SGLANG_OMNI_PYTHON_PATH=/optional/editable/checkout
HIGGS_AUDIO_V3_ATTENTION_BACKEND=triton
HIGGS_AUDIO_V3_DISABLE_CUDA_GRAPH=true
HIGGS_AUDIO_V3_STARTUP_TIMEOUT_SECONDS=600
```

Use `HIGGS_AUDIO_V3_RUNTIME_MODE=in_process` only when SGLang-Omni is installed in the ComfyUI Python environment and its dependency versions are compatible.

## Start Higgs Server

The original `Higgs Audio V3 TTS` node can still talk to a separately served SGLang-Omni HTTP API. Use this mode when you want to share one Higgs service across multiple ComfyUI sessions or machines.

```bash
docker pull lmsysorg/sglang-omni:dev
docker run -it --gpus all --shm-size 32g --ipc host --network host --privileged \
  lmsysorg/sglang-omni:dev /bin/zsh
```

Inside the container:

```bash
git clone https://github.com/sgl-project/sglang-omni.git
cd sglang-omni
uv venv .venv -p 3.12
source .venv/bin/activate
uv pip install -v -e .
hf download bosonai/higgs-audio-v3-tts-4b
sgl-omni serve --model-path bosonai/higgs-audio-v3-tts-4b --port 8000
```

### WSL2 / No CUDA Toolkit Workaround

On systems with a working PyTorch CUDA runtime but no local CUDA toolkit (`nvcc`), FlashInfer JIT may fail with:

```text
RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist
```

For this setup, serve Higgs with SGLang's Triton attention backend and CUDA graph disabled for the Higgs `tts_engine` stage. One way is to generate a config:

```bash
cd /path/to/sglang-omni
source .venv/bin/activate
python - <<'PY'
from pathlib import Path
import yaml
from sglang_omni.config.manager import ConfigManager

cfg = ConfigManager.from_model_path("bosonai/higgs-audio-v3-tts-4b").merge_config({})
engine = cfg.stages[2]
assert engine.name == "tts_engine"
factory_args = dict(engine.factory_args or {})
overrides = dict(factory_args.get("server_args_overrides") or {})
overrides.update({"disable_cuda_graph": True, "attention_backend": "triton"})
factory_args["server_args_overrides"] = overrides
engine.factory_args = factory_args
Path("higgs_triton_no_nvcc.yaml").write_text(
    yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False),
    encoding="utf-8",
)
PY

sgl-omni serve --config higgs_triton_no_nvcc.yaml --host 0.0.0.0 --port 8000
```

If you use reference audio from ComfyUI with a Docker-hosted server, make sure the container can read the reference path emitted by ComfyUI. The simplest setup is running SGLang-Omni directly in the same environment as ComfyUI, or mounting your ComfyUI directory into the container at a stable shared path.

## Node Inputs

- `server_url`: SGLang-Omni server URL, usually `http://127.0.0.1:8000`
- `text`: speech text; inline Higgs control tokens are allowed
- `response_mode`: `standard_wav`, `stream_sse_wav`, or `stream_pcm`
- `reference_audio`: optional ComfyUI `AUDIO` for voice cloning
- `reference_audio_path`: optional server-visible local path or URL for voice cloning
- `reference_text`: transcript for the reference audio
- `temperature`, `top_k`, `max_new_tokens`: sampling controls sent to the API
- `Higgs Audio V3 Model Loader`: local loader with `model_path` and `device`

## Examples

Zero-shot:

```text
Hello, how are you?
```

Inline control:

```text
<|emotion:amusement|><|prosody:expressive_high|>Wait, wait, that was hilarious. <|sfx:laughter|>Hehe, I was not ready for that.
```

Voice cloning:

Connect an `AUDIO` node to `reference_audio`, enter the transcript in `reference_text`, then run the node.

## Test

Unit tests use a local mock `/v1/audio/speech` server and do not download the model:

```bash
cd /path/to/ComfyUI/custom_nodes/ComfyUI-HiggsAudioV3TTS
source /path/to/comfyui-venv/bin/activate
python -m pytest -q
```

If `pytest` is not installed in the ComfyUI environment:

```bash
python scripts/run_mock_tests.py
```

## Real Model Smoke Tests

After the SGLang-Omni server is running on port 8000, test the same API modes used by the ComfyUI node:

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello, how are you?"}' \
  --output higgs_zero_shot.wav

curl -X POST http://127.0.0.1:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "<|emotion:amusement|><|prosody:expressive_high|>Wait, wait, that was hilarious. <|sfx:laughter|>Hehe.", "temperature": 0.8, "top_k": 50, "max_new_tokens": 1024}' \
  --output higgs_control_tokens.wav
```

Then run the ComfyUI node with:

- `response_mode=standard_wav`
- `response_mode=stream_sse_wav`
- `response_mode=stream_pcm`
- `reference_audio` connected, plus `reference_text`

The repo also includes executable smoke tests:

```bash
# Tests the official API modes against a running SGLang-Omni server.
python scripts/run_live_higgs_tests.py \
  --base-url http://127.0.0.1:8000 \
  --reference-audio docs/_static/audio/male-voice.wav \
  --out-dir /tmp/higgs_audio_tests

# Tests the ComfyUI node class for the streaming modes.
python scripts/run_comfy_node_live_tests.py \
  --base-url http://127.0.0.1:8000 \
  --out-dir /tmp/higgs_comfy_node_tests
```
