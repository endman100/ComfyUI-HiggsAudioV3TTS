# ComfyUI-HiggsAudioV3TTS

ComfyUI custom nodes for `bosonai/higgs-audio-v3-tts-4b`.

The primary workflow is a local ComfyUI-style model loader:

```text
Higgs Audio V3 Model Loader -> Higgs Audio V3 Local TTS -> SaveAudio
```

This starts and reuses a Higgs Audio v3 pipeline from inside ComfyUI. You do not need to manually run a separate `sgl-omni serve` process for the included local workflows.

## Features

- Zero-shot text-to-speech.
- Voice cloning from a connected ComfyUI `AUDIO` input.
- Multilingual text input supported by Higgs Audio v3.
- Inline Higgs control tokens such as `<|emotion:amusement|>`, `<|style:whispering|>`, `<|prosody:pause|>`, and `<|sfx:laughter|>`.
- ComfyUI model-folder lookup through `models/higgs_audio`, `models/LLM`, and `extra_model_paths.yaml`.

Higgs Audio v3 is released by Boson AI for research and non-commercial use. Review the model license before using generated audio.

## Nodes

### Higgs Audio V3 Model Loader

Loads the model once and returns a reusable `HIGGS_AUDIO_V3_MODEL` object.

Inputs:

- `model_path`: local model choice or Hugging Face model id. Local ComfyUI model folders are preferred.
- `device`: `cuda` or `cpu`.

Hidden runtime settings are configured through environment variables, not exposed as normal ComfyUI widgets.

### Higgs Audio V3 Local TTS

Generates audio through a model loaded by `Higgs Audio V3 Model Loader`.

Inputs:

- `model`: output from `Higgs Audio V3 Model Loader`.
- `text`: text to synthesize. Higgs control tokens are supported.
- `temperature`, `top_k`, `max_new_tokens`: sampling controls.
- `timeout_seconds`: per-request timeout.
- `reference_audio`: optional connected ComfyUI `AUDIO` for voice cloning.
- `reference_audio_path`: optional local path visible to the Higgs runtime.
- `reference_text`: transcript for the reference audio.

Outputs:

- `audio`: native ComfyUI `AUDIO`.
- `request_json`: JSON payload used for generation.

## Install

Place this repository in:

```text
ComfyUI/custom_nodes/ComfyUI-HiggsAudioV3TTS
```

Install the node requirements in the same Python environment used by ComfyUI:

```bash
cd /path/to/ComfyUI
source /path/to/comfyui-venv/bin/activate
pip install -r custom_nodes/ComfyUI-HiggsAudioV3TTS/requirements.txt
```

Restart ComfyUI after installing.

## Model Placement

The loader resolves local model folders before falling back to Hugging Face. This prevents unnecessary downloads when a complete local snapshot exists.

Recommended layout:

```text
ComfyUI/models/higgs_audio/bosonai/higgs-audio-v3-tts-4b
```

Also supported:

```text
ComfyUI/models/LLM/bosonai/higgs-audio-v3-tts-4b
ComfyUI/models/LLM/higgs-audio-v3-tts-4b
```

The model folder should contain a complete Hugging Face snapshot, including files such as `config.json` and the required model weights.

If `extra_model_paths.yaml` defines any of these model categories, they are scanned too:

```yaml
my_models:
  base_path: /path/to/models
  higgs_audio: higgs_audio
  LLM: llm
  llm: llm
```

A workflow value of:

```text
bosonai/higgs-audio-v3-tts-4b
```

will first try local model folders such as:

```text
ComfyUI/models/higgs_audio/bosonai/higgs-audio-v3-tts-4b
ComfyUI/models/LLM/bosonai/higgs-audio-v3-tts-4b
```

If no local folder is found, the Higgs runtime receives the original Hugging Face model id.

## Runtime Configuration

By default, the loader starts a node-managed Python worker using the current Python executable.

If SGLang-Omni is installed in a different environment, set these environment variables before starting ComfyUI:

```text
HIGGS_AUDIO_V3_RUNTIME_MODE=python_worker
HIGGS_AUDIO_V3_PYTHON_EXECUTABLE=/path/to/python
HIGGS_AUDIO_V3_SGLANG_OMNI_PYTHON_PATH=/optional/editable/sglang-omni
HIGGS_AUDIO_V3_ATTENTION_BACKEND=triton
HIGGS_AUDIO_V3_DISABLE_CUDA_GRAPH=true
HIGGS_AUDIO_V3_STARTUP_TIMEOUT_SECONDS=600
```

Use `HIGGS_AUDIO_V3_RUNTIME_MODE=in_process` only when SGLang-Omni is installed in the ComfyUI Python environment and dependency versions are compatible.

### WSL2 Notes

For WSL2 setups where PyTorch CUDA works but no local CUDA toolkit is installed, the recommended defaults are:

```text
HIGGS_AUDIO_V3_ATTENTION_BACKEND=triton
HIGGS_AUDIO_V3_DISABLE_CUDA_GRAPH=true
```

When ComfyUI runs inside WSL2, Windows model paths are seen through WSL mounts, for example:

```text
/mnt/<drive>/ComfyUI/models/higgs_audio/bosonai/higgs-audio-v3-tts-4b
```

## Workflows

Example workflows are in:

```text
examples/
```

Included files:

- `higgs_audio_v3_tts_local_model_workflow.json`: minimal local loader -> local TTS -> SaveAudio workflow.
- `higgs_audio_v3_tts_local_features_workflow.json`: local workflow covering generated reference audio, multilingual text, and inline control tokens.
- `examples/README.md`: quick notes for loading the workflow files.

These are ComfyUI UI workflow files. Drag them into the ComfyUI canvas.

## Prompt Examples

Basic:

```text
Hello, this is Higgs Audio v3 running directly inside ComfyUI.
```

Inline controls:

```text
<|emotion:amusement|>That was unexpectedly fun. <|prosody:pause|><|style:whispering|>Now this part is quieter.
```

Voice cloning:

1. Connect a ComfyUI `AUDIO` output to `reference_audio`.
2. Add the transcript to `reference_text`.
3. Generate with `Higgs Audio V3 Local TTS`.

## Test

Unit tests use a fake local model and do not download the model:

```bash
cd /path/to/ComfyUI/custom_nodes/ComfyUI-HiggsAudioV3TTS
python -m pytest -q
```

## Troubleshooting

### The loader still downloads from Hugging Face

Check that the model folder exists under a scanned root:

```text
ComfyUI/models/higgs_audio/bosonai/higgs-audio-v3-tts-4b
ComfyUI/models/LLM/bosonai/higgs-audio-v3-tts-4b
```

The folder must be a complete model snapshot. Restart ComfyUI to refresh the model dropdown.

### The model loader dropdown only shows `bosonai/higgs-audio-v3-tts-4b`

No local Higgs model folder was found. The workflow can still run through Hugging Face cache/download behavior, but local model-folder resolution requires a complete local model directory.

### CUDA toolkit / `nvcc` errors

Set:

```text
HIGGS_AUDIO_V3_ATTENTION_BACKEND=triton
HIGGS_AUDIO_V3_DISABLE_CUDA_GRAPH=true
```

### Reference audio path problems

Prefer connecting ComfyUI `AUDIO` directly to `reference_audio`. Use `reference_audio_path` only when the Higgs runtime can read that path from the same filesystem.

## Repository Layout

```text
ComfyUI-HiggsAudioV3TTS/
  __init__.py
  nodes.py
  local_worker.py
  js/
  examples/
  tests/
  requirements.txt
```

`js/workflow_dragdrop_compat.js` keeps the included UI workflow JSON files drag-and-drop friendly in current ComfyUI frontends.
