from __future__ import annotations

import io
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import nodes  # noqa: E402
from nodes import HiggsAudioV3LocalTTS, HiggsAudioV3ModelLoader  # noqa: E402


def _wav_bytes() -> bytes:
    t = np.linspace(0, 0.1, 2400, endpoint=False, dtype=np.float32)
    audio = 0.2 * np.sin(2 * np.pi * 440 * t)
    buf = io.BytesIO()
    sf.write(buf, audio, 24000, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _assert_audio(audio):
    assert set(audio.keys()) == {"waveform", "sample_rate"}
    assert audio["sample_rate"] == 24000
    assert isinstance(audio["waveform"], torch.Tensor)
    assert audio["waveform"].shape[0] == 1
    assert audio["waveform"].shape[1] == 1
    assert audio["waveform"].shape[2] > 0


class FakeLocalModel:
    def __init__(self):
        self.payloads = []

    def speech(self, payload, timeout_seconds):
        self.payloads.append((payload, timeout_seconds))
        return _wav_bytes()


def test_local_tts_uses_loaded_model_without_http():
    model = FakeLocalModel()
    audio, request_json = HiggsAudioV3LocalTTS().generate(
        model,
        "Local hello",
        0.8,
        50,
        1024,
        10,
    )

    _assert_audio(audio)
    assert json.loads(request_json)["input"] == "Local hello"
    assert model.payloads[0][0]["input"] == "Local hello"


def test_local_tts_reference_audio_payload_contains_reference():
    model = FakeLocalModel()
    reference_audio = {"waveform": torch.zeros(1, 1, 2400), "sample_rate": 24000}

    _audio, request_json = HiggsAudioV3LocalTTS().generate(
        model,
        "Clone this",
        0.8,
        50,
        1024,
        10,
        reference_audio=reference_audio,
        reference_text="Reference transcript",
    )

    request = json.loads(request_json)
    payload = model.payloads[-1][0]
    assert request["references"][0]["text"] == "Reference transcript"
    assert payload["references"][0]["audio_path"].endswith(".wav")


def test_model_loader_ui_hides_runtime_internals():
    inputs = HiggsAudioV3ModelLoader.INPUT_TYPES()["required"]

    assert list(inputs) == ["model_path", "device"]
    assert "runtime_mode" not in inputs
    assert "python_executable" not in inputs
    assert "sglang_omni_python_path" not in inputs
    assert "attention_backend" not in inputs
    assert "disable_cuda_graph" not in inputs
    assert "startup_timeout_seconds" not in inputs


def test_model_path_prefers_local_comfy_model_folder(tmp_path, monkeypatch):
    model_dir = tmp_path / "LLM" / "bosonai" / "higgs-audio-v3-tts-4b"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(nodes, "_higgs_model_roots", lambda: [str(tmp_path / "LLM")])

    assert nodes._resolve_higgs_model_path("bosonai/higgs-audio-v3-tts-4b") == str(model_dir)
