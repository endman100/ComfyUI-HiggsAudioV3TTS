# Higgs Audio V3 TTS Example Workflows

These workflows assume:

- ComfyUI is running at `http://127.0.0.1:8188`.
- The Higgs/SGLang-Omni server is running at `http://127.0.0.1:8000`.
- For voice cloning by path, the reference file is visible to the Higgs server at:
  `/path/to/server-visible/reference.wav`

## Files

- `higgs_audio_v3_tts_all_features_workflow.json`
  - Drag this into the ComfyUI UI.
  - Tests zero-shot TTS, multilingual text, inline control tokens, voice cloning by server-visible path, SSE streaming, and PCM streaming.

- `higgs_audio_v3_tts_all_features_api_prompt.json`
  - Same core branches as the UI workflow, but in ComfyUI API `/prompt` format.

- `higgs_audio_v3_tts_reference_audio_input_workflow.json`
  - Tests the custom node's `reference_audio` input by connecting `LoadAudio`.
  - This requires the ComfyUI process and Higgs server to share the temp/reference audio path. If ComfyUI is on Windows and Higgs runs inside WSL as a separate server, use the server-visible path workflow above instead.

- `higgs_audio_v3_tts_reference_audio_input_api_prompt.json`
  - API `/prompt` version of the `LoadAudio` reference input workflow.

## Expected Output

The all-features workflow writes FLAC files under:

`<ComfyUI output>/higgs_audio_v3_tts/`
