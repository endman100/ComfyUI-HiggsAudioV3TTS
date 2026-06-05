from __future__ import annotations

import base64
import io
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import soundfile as sf
import torch

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from nodes import HiggsAudioV3TTS  # noqa: E402


def _wav_bytes() -> bytes:
    t = np.linspace(0, 0.1, 2400, endpoint=False, dtype=np.float32)
    audio = 0.2 * np.sin(2 * np.pi * 440 * t)
    buf = io.BytesIO()
    sf.write(buf, audio, 24000, format="WAV", subtype="PCM_16")
    return buf.getvalue()


class MockHiggsHandler(BaseHTTPRequestHandler):
    payloads = []
    wav_data = _wav_bytes()

    def do_POST(self):
        length = int(self.headers.get("content-length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        self.__class__.payloads.append(payload)

        if payload.get("response_format") == "pcm":
            self.send_response(200)
            self.send_header("Content-Type", "audio/pcm; rate=24000")
            self.send_header("X-Sample-Rate", "24000")
            self.end_headers()
            pcm = (np.ones(2400, dtype=np.float32) * 0.1 * 32767).astype("<i2").tobytes()
            self.wfile.write(pcm)
            return

        if payload.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            encoded = base64.b64encode(self.wav_data).decode("ascii")
            self.wfile.write(f'data: {{"audio": {{"data": "{encoded}"}}, "finish_reason": null}}\n\n'.encode())
            self.wfile.write(f'data: {{"audio": {{"data": "{encoded}"}}, "finish_reason": null}}\n\n'.encode())
            self.wfile.write(b'data: {"audio": null, "finish_reason": "stop"}\n\n')
            self.wfile.write(b"data: [DONE]\n\n")
            return

        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.end_headers()
        self.wfile.write(self.wav_data)

    def log_message(self, *args):
        return


def _server():
    MockHiggsHandler.payloads = []
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), MockHiggsHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def _assert_audio(audio):
    assert set(audio.keys()) == {"waveform", "sample_rate"}
    assert audio["sample_rate"] == 24000
    assert isinstance(audio["waveform"], torch.Tensor)
    assert audio["waveform"].shape[0] == 1
    assert audio["waveform"].shape[1] == 1
    assert audio["waveform"].shape[2] > 0


def test_standard_wav_generation():
    httpd, url = _server()
    try:
        audio, request_json = HiggsAudioV3TTS().generate(url, "Hello", "standard_wav", 0.8, 50, 1024, 10)
        _assert_audio(audio)
        assert json.loads(request_json)["input"] == "Hello"
    finally:
        httpd.shutdown()


def test_stream_sse_wav_generation():
    httpd, url = _server()
    try:
        audio, _ = HiggsAudioV3TTS().generate(url, "Hello", "stream_sse_wav", 0.8, 50, 1024, 10)
        _assert_audio(audio)
        assert audio["waveform"].shape[2] == 4800
    finally:
        httpd.shutdown()


def test_stream_pcm_generation():
    httpd, url = _server()
    try:
        audio, _ = HiggsAudioV3TTS().generate(url, "Hello", "stream_pcm", 0.8, 50, 1024, 10)
        _assert_audio(audio)
    finally:
        httpd.shutdown()


def test_reference_audio_payload_contains_reference():
    httpd, url = _server()
    reference_audio = {"waveform": torch.zeros(1, 1, 2400), "sample_rate": 24000}
    try:
        HiggsAudioV3TTS().generate(
            url,
            "Clone this",
            "standard_wav",
            0.8,
            50,
            1024,
            10,
            reference_audio=reference_audio,
            reference_text="Reference transcript",
        )
        payload = MockHiggsHandler.payloads[-1]
        assert payload["references"][0]["text"] == "Reference transcript"
        assert payload["references"][0]["audio_path"].endswith(".wav")
    finally:
        httpd.shutdown()
