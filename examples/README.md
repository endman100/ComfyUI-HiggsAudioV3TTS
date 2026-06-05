# Higgs Audio V3 TTS Example Workflows

These examples assume:

- ComfyUI is running at `http://127.0.0.1:8188`.
- `Higgs Audio V3 Model Loader` can start a local Higgs runtime. If SGLang-Omni is installed outside ComfyUI's Python environment, set `HIGGS_AUDIO_V3_PYTHON_EXECUTABLE` before starting ComfyUI.
- Every JSON file in this folder is a drag-and-drop ComfyUI UI workflow. API `/prompt` JSON is intentionally not stored here because it does not load as nodes when dragged onto the canvas.

## Files

- `higgs_audio_v3_tts_local_model_workflow.json`
  - Minimal no-server workflow:
    `Higgs Audio V3 Model Loader -> Higgs Audio V3 Local TTS -> SaveAudio`.

- `higgs_audio_v3_tts_local_features_workflow.json`
  - Drag this into the ComfyUI UI to test local model loading, basic TTS, generated `AUDIO` reference input, multilingual text, and inline Higgs control tokens.

## Expected Output

The workflows write audio files under:

`<ComfyUI output>/higgs_audio_v3_tts/`
