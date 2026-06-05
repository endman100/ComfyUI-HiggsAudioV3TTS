from __future__ import annotations

import base64
import io
import json
import os
import tempfile
import uuid
import wave
from typing import Any

import numpy as np
import requests
import soundfile as sf
import torch

try:
    import folder_paths
except Exception:  # pragma: no cover - lets unit tests import outside ComfyUI
    folder_paths = None

try:
    from comfy.utils import ProgressBar
except Exception:  # pragma: no cover
    ProgressBar = None


DEFAULT_SERVER_URL = "http://127.0.0.1:8000"
DEFAULT_SAMPLE_RATE = 24000
RESPONSE_MODES = ["standard_wav", "stream_sse_wav", "stream_pcm"]


def _clean_server_url(server_url: str) -> str:
    server_url = (server_url or DEFAULT_SERVER_URL).strip().rstrip("/")
    if not server_url:
        return DEFAULT_SERVER_URL
    return server_url


def _audio_to_wav_bytes(audio: dict[str, Any]) -> bytes:
    if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
        raise ValueError("Reference audio must be a ComfyUI AUDIO object.")

    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.as_tensor(waveform)
    if waveform.ndim != 3:
        raise ValueError(f"Expected reference waveform shape [B, C, T], got {tuple(waveform.shape)}.")

    wav = waveform[0].detach().cpu().float().numpy().T
    if wav.ndim == 2 and wav.shape[1] == 1:
        wav = wav[:, 0]

    buf = io.BytesIO()
    sf.write(buf, wav, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _write_reference_audio(audio: dict[str, Any]) -> str:
    if folder_paths is not None:
        temp_dir = folder_paths.get_temp_directory()
    else:
        temp_dir = tempfile.gettempdir()
    os.makedirs(temp_dir, exist_ok=True)
    path = os.path.join(temp_dir, f"higgs_audio_v3_ref_{uuid.uuid4().hex}.wav")
    with open(path, "wb") as f:
        f.write(_audio_to_wav_bytes(audio))
    return path


def _read_audio_bytes(data: bytes) -> tuple[np.ndarray, int]:
    if not data:
        raise ValueError("Higgs server returned an empty audio response.")

    try:
        wav, sample_rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception as exc:
        prefix = data[:80]
        raise ValueError(f"Could not decode Higgs audio response as WAV/FLAC/OGG bytes. Prefix={prefix!r}") from exc

    return wav, int(sample_rate)


def _array_to_comfy_audio(wav: np.ndarray, sample_rate: int) -> dict[str, Any]:
    if wav.size == 0:
        raise ValueError("Decoded audio response contains zero samples.")

    waveform = torch.from_numpy(wav.T.copy()).float().unsqueeze(0)
    return {"waveform": waveform, "sample_rate": int(sample_rate)}


def _wav_bytes_to_comfy_audio(data: bytes) -> dict[str, Any]:
    wav, sample_rate = _read_audio_bytes(data)
    return _array_to_comfy_audio(wav, sample_rate)


def _wav_chunks_to_comfy_audio(chunks: list[bytes]) -> dict[str, Any]:
    if not chunks:
        raise ValueError("Higgs server returned an empty SSE audio stream.")

    arrays: list[np.ndarray] = []
    sample_rate: int | None = None
    channels: int | None = None
    for chunk in chunks:
        wav, chunk_sample_rate = _read_audio_bytes(chunk)
        chunk_channels = wav.shape[1]
        if sample_rate is None:
            sample_rate = chunk_sample_rate
            channels = chunk_channels
        elif chunk_sample_rate != sample_rate:
            raise ValueError(f"SSE WAV chunks use mixed sample rates: {sample_rate} and {chunk_sample_rate}.")
        elif chunk_channels != channels:
            raise ValueError(f"SSE WAV chunks use mixed channel counts: {channels} and {chunk_channels}.")
        arrays.append(wav)

    return _array_to_comfy_audio(np.concatenate(arrays, axis=0), int(sample_rate or DEFAULT_SAMPLE_RATE))


def _pcm16_bytes_to_comfy_audio(data: bytes, sample_rate: int) -> dict[str, Any]:
    if not data:
        raise ValueError("Higgs server returned an empty PCM stream.")
    if len(data) % 2:
        data = data[:-1]
    samples = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    waveform = torch.from_numpy(samples.copy()).float().view(1, 1, -1)
    return {"waveform": waveform, "sample_rate": int(sample_rate or DEFAULT_SAMPLE_RATE)}


def _sample_rate_from_headers(headers: requests.structures.CaseInsensitiveDict[str]) -> int:
    for key in (
        "x-sample-rate",
        "x-audio-sample-rate",
        "sample-rate",
        "x-sglang-sample-rate",
    ):
        value = headers.get(key)
        if value:
            try:
                return int(value)
            except ValueError:
                pass
    content_type = headers.get("content-type", "")
    for part in content_type.split(";"):
        part = part.strip().lower()
        if part.startswith("rate="):
            try:
                return int(part.split("=", 1)[1])
            except ValueError:
                pass
    return DEFAULT_SAMPLE_RATE


def _raise_for_bad_response(resp: requests.Response) -> None:
    if resp.ok:
        return
    text = resp.text[:1200] if resp.text else ""
    raise RuntimeError(f"Higgs server request failed: HTTP {resp.status_code} {resp.reason}. {text}")


def _collect_sse_wav_chunks(resp: requests.Response) -> list[bytes]:
    chunks: list[bytes] = []
    for line in resp.iter_lines():
        if not line or line == b"data: [DONE]":
            continue
        if not line.startswith(b"data: "):
            continue
        event = json.loads(line[len(b"data: "):])
        if event.get("finish_reason") == "stop":
            break
        audio = event.get("audio") or {}
        encoded = audio.get("data")
        if encoded:
            chunks.append(base64.b64decode(encoded))
    return chunks


class HiggsAudioV3TTS:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "server_url": ("STRING", {"default": DEFAULT_SERVER_URL}),
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Hello, how are you?",
                        "tooltip": "Text sent to /v1/audio/speech. Inline Higgs control tokens are supported.",
                    },
                ),
                "response_mode": (RESPONSE_MODES, {"default": "standard_wav"}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_k": ("INT", {"default": 50, "min": 0, "max": 500, "step": 1}),
                "max_new_tokens": ("INT", {"default": 1024, "min": 1, "max": 8192, "step": 64}),
                "timeout_seconds": ("INT", {"default": 300, "min": 5, "max": 3600, "step": 5}),
            },
            "optional": {
                "reference_audio": ("AUDIO", {"tooltip": "Optional ComfyUI audio used for zero-shot voice cloning."}),
                "reference_audio_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Optional path or URL visible to the Higgs/SGLang server. Ignored when reference_audio is connected.",
                    },
                ),
                "reference_text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Transcript for the reference audio. Hugging Face recommends providing it for better cloning.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "request_json")
    FUNCTION = "generate"
    CATEGORY = "audio/Higgs Audio V3"
    DESCRIPTION = "Generate 24 kHz speech with bosonai/higgs-audio-v3-tts-4b through an SGLang-Omni /v1/audio/speech server."

    def generate(
        self,
        server_url: str,
        text: str,
        response_mode: str,
        temperature: float,
        top_k: int,
        max_new_tokens: int,
        timeout_seconds: int,
        reference_audio: dict[str, Any] | None = None,
        reference_audio_path: str = "",
        reference_text: str = "",
    ):
        if not text or not text.strip():
            raise ValueError("Text cannot be empty.")
        if response_mode not in RESPONSE_MODES:
            raise ValueError(f"Unsupported response_mode: {response_mode}")

        pbar = ProgressBar(3) if ProgressBar is not None else None
        if pbar:
            pbar.update_absolute(1, 3)

        payload: dict[str, Any] = {
            "input": text,
            "temperature": float(temperature),
            "top_k": int(top_k),
            "max_new_tokens": int(max_new_tokens),
        }

        temp_ref_path = None
        ref_path = (reference_audio_path or "").strip()
        if reference_audio is not None:
            temp_ref_path = _write_reference_audio(reference_audio)
            ref_path = temp_ref_path
        if ref_path:
            reference: dict[str, Any] = {"audio_path": ref_path}
            if reference_text and reference_text.strip():
                reference["text"] = reference_text.strip()
            payload["references"] = [reference]

        if response_mode == "stream_sse_wav":
            payload["stream"] = True
        elif response_mode == "stream_pcm":
            payload.update(
                {
                    "stream": True,
                    "stream_format": "audio",
                    "response_format": "pcm",
                    "initial_codec_chunk_frames": 1,
                }
            )

        url = f"{_clean_server_url(server_url)}/v1/audio/speech"
        request_json = json.dumps(payload, ensure_ascii=False, indent=2)

        try:
            if response_mode == "stream_sse_wav":
                with requests.post(url, json=payload, timeout=int(timeout_seconds), stream=True) as resp:
                    _raise_for_bad_response(resp)
                    chunks = _collect_sse_wav_chunks(resp)
                audio = _wav_chunks_to_comfy_audio(chunks)
            elif response_mode == "stream_pcm":
                with requests.post(url, json=payload, timeout=int(timeout_seconds), stream=True) as resp:
                    _raise_for_bad_response(resp)
                    sample_rate = _sample_rate_from_headers(resp.headers)
                    data = b"".join(resp.iter_content(chunk_size=1024 * 64))
                audio = _pcm16_bytes_to_comfy_audio(data, sample_rate)
            else:
                resp = requests.post(url, json=payload, timeout=int(timeout_seconds))
                _raise_for_bad_response(resp)
                audio = _wav_bytes_to_comfy_audio(resp.content)
        finally:
            if temp_ref_path:
                try:
                    os.remove(temp_ref_path)
                except OSError:
                    pass

        if pbar:
            pbar.update_absolute(3, 3)

        return (audio, request_json)
